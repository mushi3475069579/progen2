#!/usr/bin/env bash
# Run only on the contest DCU platform, after activating its existing .venv.
set -euo pipefail
E10_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$E10_SCRIPT_DIR"
if [[ ! -f check_environment.py || ! -f run_public_test.sh || ! -d submission ]]; then
  echo "Upload this script to the contest root beside run_public_test.sh."
  exit 2
fi

# Preserve the known working version before touching either candidate source.
python - <<'E10_BACKUP_PY'
from pathlib import Path
from datetime import datetime
import hashlib, json, shutil, sys
root = Path.cwd()
if sys.prefix == sys.base_prefix:
    raise SystemExit("Activate the existing project .venv before running this script.")
source = root / "submission/solution.py"
expected = "8f430db92eeadf6ebe3f611d0c9f28b0b9c86d140988374ed08c3ee31f7cbf1f"
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise SystemExit("Current solution.py is not the verified v0. Stop; do not overwrite unknown changes. For an already installed E10, use the retest commands in the guide.")
names = ["baseline_public.json", "baseline.log", "submission_public.json", "submission.log", "environment_check.log"]
missing = [name for name in names if not (root / "results" / name).is_file()]
if missing:
    raise SystemExit("Required v0 archive files are missing: " + ", ".join(missing))
backup = root / "versions" / ("E09_fp32_before_E10_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
backup.mkdir(parents=True, exist_ok=False)
shutil.copytree(root / "submission", backup / "submission", ignore=shutil.ignore_patterns("evidence", ".cache", "__pycache__", "*.pyc"))
(backup / "evidence").mkdir()
for name in names:
    shutil.copy2(root / "results" / name, backup / "evidence" / name)
(backup / "source_sha256.txt").write_text(expected + "  submission/solution.py\n", encoding="utf-8")
print("V0_BACKUP:", backup, flush=True)
E10_BACKUP_PY

cat > submission/rope_cache.py <<'E10_HELPER_SOURCE'
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
"""E10: share input-independent FP32 RoPE tables across the attention layers.

Only the positional table lookup changes in attention.forward. Weights,
attention arithmetic, dynamic per-request KV and generation rules are retained.
The public model files on disk and their module globals are left untouched.
"""

from __future__ import annotations

import torch
from models.progen.modeling_progen import ProGenAttention, apply_rotary_pos_emb, fixed_pos_embedding


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

        if seq_len > self._e10_rope_capacity:
            raise ValueError("RoPE position exceeds the initialized model context.")

        # This table depends on position and rotary dimension, never on input.
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]
            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = (self._e10_rope_sin, self._e10_rope_cos)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = (self._e10_rope_sin, self._e10_rope_cos)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

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
        raise RuntimeError("E10 requires the supplied, unmodified ProGenAttention implementation.")
    capacity = int(model.config.max_position_embeddings)
    if capacity <= 0:
        raise ValueError("Invalid context capacity.")

    tables = {}
    before_parameters = tuple(id(parameter) for parameter in model.parameters())
    with torch.inference_mode():
        for attention in attentions:
            device = attention.qkv_proj.weight.device
            if attention.qkv_proj.weight.dtype != torch.float32:
                raise RuntimeError("E10 is an FP32-only comparison; do not combine precision changes.")
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
                tables[key] = (sin, cos)

            sin, cos = tables[key]
            attention.register_buffer("_e10_rope_sin", sin, persistent=False)
            attention.register_buffer("_e10_rope_cos", cos, persistent=False)
            attention._e10_rope_capacity = capacity
            # Change only the Python implementation of forward on this model's
            # existing objects; parameters and module structure are preserved.
            attention.__class__ = CachedRoPEAttention

    if before_parameters != tuple(id(parameter) for parameter in model.parameters()):
        raise RuntimeError("E10 must not replace or remove any model parameters.")
    return {
        "attention_layers": len(attentions),
        "shared_tables": len(tables),
        "capacity": capacity,
        "dtype": "fp32",
        "table_bytes": sum(
            sin.numel() * sin.element_size() + cos.numel() * cos.element_size()
            for sin, cos in tables.values()
        ),
    }
E10_HELPER_SOURCE

python - <<'E10_PATCH_PY'
from pathlib import Path
import ast, hashlib
target = Path("submission/solution.py")
text = target.read_text(encoding="utf-8")
changes = [('"""正式赛基础 FP32 入口，供平台验收后作为优化起点。\n\n参照题目包 baseline/solution.py 的加载方式和生成规则实现；不依赖或\n修改 baseline 目录，不加载参考输出，不包含性能优化及其收益声明。\n"""', '"""E10候选：FP32推理加输入无关RoPE表缓存，待正式DCU验证。\n\n沿用已验收v0的权重加载、确定性生成和请求内动态KV；初始化时安装\nrope_cache.py的等价注意力实现。无额外依赖，不读取参考输出。\n"""'), ('        self.max_positions = int(self.model.config.max_position_embeddings)', '        self.max_positions = int(self.model.config.max_position_embeddings)\n        from rope_cache import install_rope_cache\n        self._rope_cache_info = install_rope_cache(self.model)\n        print("E10_ROPE_CACHE_READY", self._rope_cache_info, flush=True)')]
for old, new in changes:
    if text.count(old) != 1:
        raise SystemExit("Unexpected v0 source; original backup retained; stop patching.")
    text = text.replace(old, new, 1)
ast.parse(text, feature_version=(3, 10))
ast.parse(Path("submission/rope_cache.py").read_text(encoding="utf-8"), feature_version=(3, 10))
target.write_bytes(text.encode("utf-8"))
print("E10_SOURCE_READY", hashlib.sha256(target.read_bytes()).hexdigest())
E10_PATCH_PY

E10_RUN_DIR="results/E10_rope_cache_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$E10_RUN_DIR"
export E10_RUN_DIR
echo "E10_OUTPUT: $E10_SCRIPT_DIR/$E10_RUN_DIR"

# Archive these sources before the test, and keep all v0 result files intact.
python - <<'E10_SNAPSHOT_PY'
from pathlib import Path
import hashlib, json, os, shutil
root = Path.cwd()
run = root / os.environ["E10_RUN_DIR"]
shutil.copytree(root / "submission", run / "submission", ignore=shutil.ignore_patterns("evidence", ".cache", "__pycache__", "*.pyc"))
evidence = run / "evidence"
evidence.mkdir()
for name in ["baseline_public.json", "baseline.log", "environment_check.log"]:
    shutil.copy2(root / "results" / name, evidence / name)
for original, archived in [("submission_public.json", "fp32_starting_public.json"), ("submission.log", "fp32_starting.log")]:
    shutil.copy2(root / "results" / original, evidence / archived)
fingerprints = {
    str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((run / "submission").rglob("*"))
    if path.is_file() and path.suffix in {".py", ".sh"}
}
(run / "source_sha256.json").write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")
(run / "change_notes.txt").write_text(
    "E10 candidate: FP32, precompute shared input-independent RoPE sine/cosine tables at init. "
    "Keep existing parameters, deterministic decoding, request-local dynamic KV and official four-load test. "
    "No new dependencies or build step. Actual speed and sequence consistency depend on this run.\n",
    encoding="utf-8",
)
E10_SNAPSHOT_PY

if RESULT_FILE="$E10_SCRIPT_DIR/$E10_RUN_DIR/evidence/submission_public.json" \
   PYTHONUNBUFFERED=1 bash run_public_test.sh submission 2>&1 | tee "$E10_RUN_DIR/evidence/submission.log"; then
  echo "Official four-load run finished; checking exact sequence consistency."
else
  E10_FAILURE=$?
  echo "E10 failed (exit=$E10_FAILURE). Keep this run's sources and full log: $E10_RUN_DIR"
  echo "Do not substitute an old JSON for a failed run."
  exit "$E10_FAILURE"
fi

python - <<'E10_COMPARE_PY' | tee "$E10_RUN_DIR/comparison.txt"
from pathlib import Path
import json, os
run = Path(os.environ["E10_RUN_DIR"])
evidence = run / "evidence"
official = json.loads((evidence / "baseline_public.json").read_text(encoding="utf-8"))
starting = json.loads((evidence / "fp32_starting_public.json").read_text(encoding="utf-8"))
candidate = json.loads((evidence / "submission_public.json").read_text(encoding="utf-8"))
print("load_id                    official_s  v0_s       E10_s      official/E10  v0/E10   consistency")
if not all(len(data["loads"]) == 4 for data in [official, starting, candidate]):
    raise SystemExit("Expected four loads in every result.")
for a, b, c in zip(official["loads"], starting["loads"], candidate["loads"]):
    for key in ["load_id", "batch_size", "prefix_tokens", "new_tokens"]:
        if not (a[key] == b[key] == c[key]):
            raise SystemExit("Mismatched task specifications: " + key)
    print(f'{c["load_id"]:<26} {a["time_s"]:10.4f} {b["time_s"]:10.4f} {c["time_s"]:10.4f} '
          f'{a["time_s"]/c["time_s"]:12.4f} {b["time_s"]/c["time_s"]:8.4f} '
          f'{100*c["consistency_rate"]:10.2f}%')
print("OUTPUT:", run.resolve())
print("Single official timed run per load; no stable speedup claim from one run.")
if not candidate["all_public_references_exact"]:
    raise SystemExit("E10 has sequence differences. Keep evidence and investigate before retaining this FP32-equivalent change.")
print("E10_ALL_FOUR_EXACT: true. Compare timings to decide whether this candidate helps.")
E10_COMPARE_PY

echo "E10_DONE: $E10_SCRIPT_DIR/$E10_RUN_DIR"
echo "Download this complete E10 run directory to preserve sources and matching evidence."
