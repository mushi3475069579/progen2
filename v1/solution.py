"""E10候选：FP32推理加输入无关RoPE表缓存，待正式DCU验证。

沿用已验收v0的权重加载、确定性生成和请求内动态KV；初始化时安装
rope_cache.py的等价注意力实现。无额外依赖，不读取参考输出。
"""

from __future__ import annotations

from pathlib import Path
import sys


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class Solution:
    def __init__(self, model_dir: str, tokenizer_path: str, device: str = "cuda:0"):
        import torch
        from tokenizers import Tokenizer

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("请在比赛分配的 DCU / PyTorch HIP 环境运行。")

        model_path = Path(model_dir).expanduser().resolve()
        tokenizer_file = Path(tokenizer_path).expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(model_path / "config.json")
        if not (model_path / "pytorch_model.bin").is_file():
            raise FileNotFoundError(model_path / "pytorch_model.bin")
        if not tokenizer_file.is_file():
            raise FileNotFoundError(tokenizer_file)

        # 公共模型源码与 tokenizer.json 同在资源根目录；使用传入路径定位。
        # append 保留作品目录中后续自有 models 包的正常导入优先级。
        assets = str(tokenizer_file.parent)
        if assets not in sys.path:
            sys.path.append(assets)
        from models.progen.modeling_progen import ProGenForCausalLM

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        amino_ids = [self.tokenizer.token_to_id(letter) for letter in AMINO_ACIDS]
        if any(value is None for value in amino_ids) or len(set(amino_ids)) != 20:
            raise ValueError("公共 Tokenizer 的 20 种氨基酸映射异常。")
        self._amino_ids = torch.tensor(amino_ids, dtype=torch.long, device=self.device)
        self._letter_by_id = dict(zip(amino_ids, AMINO_ACIDS))

        self.model = ProGenForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model.eval().to(self.device)
        self.max_positions = int(self.model.config.max_position_embeddings)
        from rope_cache import install_rope_cache
        self._rope_cache_info = install_rope_cache(self.model)
        print("E10_ROPE_CACHE_READY", self._rope_cache_info, flush=True)

    def reset(self) -> None:
        # 有效 KV 与生成结果均为 generate 内部的局部变量，没有跨请求状态。
        # 模型与固定的氨基酸映射可在请求之间保留。
        pass

    def generate(self, prefixes: list[str], max_new_tokens: int) -> list[str]:
        torch = self._torch
        if not isinstance(prefixes, list) or not prefixes:
            raise ValueError("prefixes 必须为非空字符串列表。")
        if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
            raise ValueError("每条前缀必须是非空字符串。")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须为正整数。")

        # 按输入原样编码，不删除、替换或额外添加开头的控制标记。
        encoded = [self.tokenizer.encode(prefix).ids for prefix in prefixes]
        lengths = [len(row) for row in encoded]
        if not lengths[0] or len(set(lengths)) != 1:
            raise ValueError("同一请求的前缀 Token 长度应相同且非空。")
        if lengths[0] + max_new_tokens > self.max_positions:
            raise ValueError("输入与生成长度超过模型上下文容量。")

        next_input = torch.tensor(encoded, dtype=torch.long, device=self.device)
        past_key_values = None
        generated_steps = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                result = self.model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = result.logits[:, -1, :]
                # 候选顺序固定；argmax 并列取首个，符合赛题字母顺序。
                scores = logits.index_select(-1, self._amino_ids)
                winner = scores.argmax(dim=-1)
                next_input = self._amino_ids.index_select(0, winner).unsqueeze(-1)
                generated_steps.append(next_input)
                past_key_values = result.past_key_values

        # 只返回本次新增字母，不包含输入前缀；固定 G 步，不按 EOS 停止。
        generated = torch.cat(generated_steps, dim=1).detach().cpu().tolist()
        return ["".join(self._letter_by_id[token_id] for token_id in row) for row in generated]
