#!/usr/bin/env bash
# E12 experiment preparation. Run in the contest root, with .venv active.
set -euo pipefail
E12_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$E12_ROOT"
[[ -f run_public_test.sh && -f public_test.py ]] || { echo "Upload this script to the contest root."; exit 2; }
mkdir -p results
E12_RUN_DIR="results/E12_rope_expand_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$E12_RUN_DIR"
export E12_RUN_DIR

python - <<'E12_PREPARE' 2>&1 | tee "$E12_RUN_DIR/preparation.log"
from pathlib import Path
from datetime import datetime
import hashlib, json, os, shutil, sys
root = Path.cwd()
run = root / os.environ["E12_RUN_DIR"]
if sys.prefix == sys.base_prefix:
    raise SystemExit("Activate the existing project .venv first.")
expected = {'solution.py': '8c7a6dd485471c82d9f9f5914af593e7de4c7d0cb6ac9eea61045816639faa09', 'rope_cache.py': '4c4dc7b0a3d5245dbb67acae56f2091d3a19d11c3122b38c927064a198446e68'}
for name, digest in expected.items():
    path = root / "submission" / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit("Current source is not verified E10/v1: " + name + ". No source was changed.")
previous = None
for candidate in sorted(root.glob("results/E10_rope_cache_*/evidence/submission_public.json"), reverse=True):
    directory = candidate.parent.parent
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if len(data["loads"]) != 4 or not data["all_public_references_exact"]:
            continue
        if not all(hashlib.sha256((directory / "submission" / name).read_bytes()).hexdigest() == digest for name, digest in expected.items()):
            continue
        if not all((directory / "evidence" / name).is_file() for name in ["baseline_public.json", "baseline.log", "submission.log", "environment_check.log"]):
            continue
        previous = directory
        break
    except (OSError, ValueError, KeyError):
        continue
if previous is None:
    raise SystemExit("Verified full E10 result directory not found. Mount the original file storage or restore the E10 ZIP before preparing E12.")
import torch
if not torch.cuda.is_available() or torch.version.hip is None:
    raise SystemExit("DCU/HIP is not available in this container. No source was changed.")
actual = {"torch": str(torch.__version__), "hip": torch.version.hip,
          "device": torch.cuda.get_device_name(torch.device(os.environ.get("DEVICE", "cuda:0")))}
official = json.loads((previous / "evidence/baseline_public.json").read_text(encoding="utf-8"))
for reference in [official, data]:
    for key in ["torch", "hip", "device"]:
        if reference.get(key) != actual[key]:
            raise SystemExit("Environment differs from the saved baseline/E10: " + key + ". Re-establish matching baseline evidence before comparing versions. No source was changed.")
(run / "preparation_environment.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
backup = root / "versions" / ("E10_before_E12_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
backup.mkdir(parents=True, exist_ok=False)
shutil.copytree(root / "submission", backup / "submission", ignore=shutil.ignore_patterns("evidence", ".cache", "__pycache__", "*.pyc"))
shutil.copytree(previous / "evidence", backup / "evidence")
evidence = run / "evidence"
shutil.copytree(previous / "evidence", evidence)
(evidence / "submission_public.json").rename(evidence / "previous_public.json")
(evidence / "submission.log").rename(evidence / "previous.log")
(run / "previous_version.json").write_text(json.dumps({"previous_run": str(previous), "source_sha256": expected}, indent=2), encoding="utf-8")
print("V1_BACKUP:", backup, flush=True)
print("E12_OUTPUT:", run, flush=True)
E12_PREPARE

cat > submission/solution.py <<'E12_SOLUTION'
"""E12候选：FP32推理加RoPE系数预扩展与共享视图，待正式DCU验证。

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
        print("E12_ROPE_EXPANDED_READY", self._rope_cache_info, flush=True)

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
E12_SOLUTION
cat > submission/rope_cache.py <<'E12_HELPER'
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
# Derived from the contest's models/progen/modeling_progen.py attention forward.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E12: share input-independent FP32 RoPE tables across the attention layers.

Coefficient expansion moves to initialization; Q/K share one view per layer. Weights,
attention arithmetic, dynamic per-request KV and generation rules are retained.
The public model files on disk and their module globals are left untouched.
"""

from __future__ import annotations

import torch
from models.progen.modeling_progen import ProGenAttention, fixed_pos_embedding, rotate_every_two



def apply_rotary_expanded(x, sincos):
    """The original elementwise operation order, with ready-to-broadcast views."""
    sin, cos = sincos
    return (x * cos) + (rotate_every_two(x) * sin)


class CachedRoPEAttention(ProGenAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Same calculation and ordering as the supplied ProGenAttention.forward.
        qkv = self.qkv_proj(hidden_states)
        mp_num = 8
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0
        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if seq_len > self._e12_rope_capacity:
            raise ValueError("RoPE position exceeds the initialized model context.")

        # Share these input-independent coefficient views between Q and K.
        current_length = key.shape[1]
        sincos = (
            self._e12_rope_sin.narrow(1, offset, current_length),
            self._e12_rope_cos.narrow(1, offset, current_length),
        )
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]
            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_expanded(k_rot, sincos)
            q_rot = apply_rotary_expanded(q_rot, sincos)
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_expanded(key, sincos)
            query = apply_rotary_expanded(query, sincos)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


def install_rope_cache(model) -> dict:
    """Initialize tables before generation and retain every existing Parameter.

    Registers shared nonpersistent buffers. No input, generated suffix or
    effective KV is stored here. Installation is performed once on a fresh model.
    """
    attentions = [module for module in model.modules() if isinstance(module, ProGenAttention)]
    if not attentions or any(type(module) is not ProGenAttention for module in attentions):
        raise RuntimeError("E12 requires the supplied, unmodified ProGenAttention implementation.")
    capacity = int(model.config.max_position_embeddings)
    if capacity <= 0:
        raise ValueError("Invalid context capacity.")

    tables = {}
    before_parameters = tuple(id(parameter) for parameter in model.parameters())
    with torch.inference_mode():
        for attention in attentions:
            device = attention.qkv_proj.weight.device
            if attention.qkv_proj.weight.dtype != torch.float32:
                raise RuntimeError("E12 is an FP32-only comparison; do not combine precision changes.")
            dim = attention.head_dim if attention.rotary_dim is None else min(
                int(attention.rotary_dim), attention.head_dim
            )
            if dim <= 0 or dim % 2:
                raise ValueError("Unsupported rotary dimension.")
            key = (device.type, device.index, dim)
            if key not in tables:
                probe = torch.empty((1, 1, 1, dim), device=device, dtype=torch.float32)
                # Use the original formula, dtype, CPU frequency construction
                # and device sine/cosine, retaining its numerical behavior.
                sin, cos = fixed_pos_embedding(probe, 1, seq_len=capacity)
                if sin.dtype != torch.float32 or cos.dtype != torch.float32:
                    raise RuntimeError("Original RoPE tables must be FP32.")
                # A small input-independent numerical check runs at init only.
                lengths = sorted({1, max(1, capacity // 4), max(1, capacity // 2), capacity - 1, capacity})
                for length in lengths:
                    if length <= 0:
                        continue
                    ref_sin, ref_cos = fixed_pos_embedding(probe, 1, seq_len=length)
                    if not torch.equal(sin[:length], ref_sin) or not torch.equal(cos[:length], ref_cos):
                        raise RuntimeError("Cached RoPE differs from the original positional table.")
                # Expand each coefficient pair once, before timed generation.
                expanded_sin = sin.repeat_interleave(2, dim=-1)[None, :, None, :]
                expanded_cos = cos.repeat_interleave(2, dim=-1)[None, :, None, :]
                for offset, length in [(0, 1), (capacity // 2, 1), (max(0, capacity - 2), min(2, capacity))]:
                    expected_sin = sin[None, offset:offset + length, None, :].repeat_interleave(2, 3)
                    expected_cos = cos[None, offset:offset + length, None, :].repeat_interleave(2, 3)
                    if not torch.equal(expanded_sin.narrow(1, offset, length), expected_sin) or not torch.equal(expanded_cos.narrow(1, offset, length), expected_cos):
                        raise RuntimeError("Expanded RoPE coefficients differ from the original layout.")
                tables[key] = (expanded_sin, expanded_cos)

            sin, cos = tables[key]
            attention.register_buffer("_e12_rope_sin", sin, persistent=False)
            attention.register_buffer("_e12_rope_cos", cos, persistent=False)
            attention._e12_rope_capacity = capacity
            # Change only the Python implementation of forward on this model's
            # existing objects; parameters and module structure are preserved.
            attention.__class__ = CachedRoPEAttention

    if before_parameters != tuple(id(parameter) for parameter in model.parameters()):
        raise RuntimeError("E12 must not replace or remove any model parameters.")
    return {
        "attention_layers": len(attentions),
        "shared_tables": len(tables),
        "capacity": capacity,
        "dtype": "fp32",
        "expanded_coefficients": True,
        "shared_qk_views": True,
        "table_bytes": sum(
            sin.numel() * sin.element_size() + cos.numel() * cos.element_size()
            for sin, cos in tables.values()
        ),
    }
E12_HELPER

python - <<'E12_SNAPSHOT'
from pathlib import Path
import ast, hashlib, json, os, shutil
root = Path.cwd()
run = root / os.environ["E12_RUN_DIR"]
for name in ["solution.py", "rope_cache.py"]:
    ast.parse((root / "submission" / name).read_text(encoding="utf-8"), feature_version=(3, 10))
shutil.copytree(root / "submission", run / "submission", ignore=shutil.ignore_patterns("evidence", ".cache", "__pycache__", "*.pyc"))
hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run / "submission").rglob("*")) if p.is_file() and p.suffix in {".py", ".sh"}}
(run / "source_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
(run / "change_notes.txt").write_text("E12 candidate: pre-expand input-independent FP32 RoPE sin/cos; share one coefficient slice between Q/K per layer. Dynamic KV, model weights and deterministic generation retained. No extra dependency. Actual speed/consistency depend on this run.\n", encoding="utf-8")
E12_SNAPSHOT

if RESULT_FILE="$E12_ROOT/$E12_RUN_DIR/evidence/submission_public.json" PYTHONUNBUFFERED=1 bash run_public_test.sh submission 2>&1 | tee "$E12_RUN_DIR/evidence/submission.log"; then
  echo "Official four-load test finished; inspecting raw consistency."
else
  E12_STATUS=$?
  echo "E12 failed (exit=$E12_STATUS). Sources and current log retained: $E12_RUN_DIR"
  exit "$E12_STATUS"
fi
python - <<'E12_COMPARE' | tee "$E12_RUN_DIR/comparison.txt"
from pathlib import Path
import json, os
run = Path(os.environ["E12_RUN_DIR"])
def read(name):
    return json.loads((run / "evidence" / name).read_text(encoding="utf-8"))
official, previous, current = read("baseline_public.json"), read("previous_public.json"), read("submission_public.json")
for key in ["device", "hip", "torch"]:
    if not official.get(key) == previous.get(key) == current.get(key):
        raise SystemExit("Environment mismatch: " + key + ". Raw results are retained; no speedup ratio is reported.")
if not all(len(d["loads"]) == 4 for d in [official, previous, current]):
    raise SystemExit("Expected four complete loads.")
print("load_id                    official_s E10_s      E12_s      official/E12 E10/E12 consistency")
for a,b,c in zip(official["loads"], previous["loads"], current["loads"]):
    for key in ["load_id", "batch_size", "prefix_tokens", "new_tokens"]:
        if not a[key] == b[key] == c[key]:
            raise SystemExit("Task mismatch: " + key)
    print(f'{c["load_id"]:<26} {a["time_s"]:9.4f} {b["time_s"]:10.4f} {c["time_s"]:10.4f} {a["time_s"]/c["time_s"]:12.4f} {b["time_s"]/c["time_s"]:7.4f} {100*c["consistency_rate"]:10.2f}%')
print("OUTPUT:", run.resolve())
if not current["all_public_references_exact"]:
    raise SystemExit("Sequence differences: keep evidence and investigate before retaining this equivalent FP32 candidate.")
print("E12_ALL_FOUR_EXACT: true. Compare measured times; one run does not prove stable speedup.")
E12_COMPARE
echo "E12_DONE: $E12_ROOT/$E12_RUN_DIR"
