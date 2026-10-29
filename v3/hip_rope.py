"""Build once outside timing; generation uses only the loaded HIP extension."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

HOST_FLAGS = ['-O2', '-fno-fast-math', '-ffp-contract=off']
DEVICE_FLAGS = ['-O2', '-fno-fast-math', '-ffp-contract=off', '-fdenormal-fp-math=ieee']
VERSION = 'E14-hip-rope-v1'


def extension_spec(device):
    import torch
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError('E14 requires the allocated DCU and a HIP PyTorch build.')
    architecture = torch.cuda.get_device_properties(device).gcnArchName.split(':', 1)[0]
    if not re.fullmatch(r'gfx[0-9a-z]+', architecture):
        raise RuntimeError('Unrecognized HIP architecture: ' + architecture)
    root = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ['hip_rope.py', 'ops/rope_qk.cu']}
    identity = {'version': VERSION, 'source_sha256': hashes, 'torch': str(torch.__version__),
                'torch_distribution': importlib.metadata.version('torch'),
                'torch_location': str(Path(torch.__file__).resolve()), 'hip': torch.version.hip,
                'python_abi': sys.implementation.cache_tag,
                'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                'architecture': architecture, 'host_flags': HOST_FLAGS,
                'device_flags': DEVICE_FLAGS + ['--offload-arch=' + architecture]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    name = 'e14_hip_rope_' + fingerprint
    cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', root / '.cache/torch_extensions')).expanduser().resolve()
    return name, cache / name, identity, root


def load_extension(device, *, build=False):
    """Runtime never invokes the compiler. build=True is for build.sh only."""
    name, folder, identity, root = extension_spec(device)
    if name in sys.modules and not build:
        return sys.modules[name]
    if build:
        from torch.utils.cpp_extension import IS_HIP_EXTENSION, load
        if not IS_HIP_EXTENSION:
            raise RuntimeError('PyTorch does not recognize the HIP extension environment.')
        os.environ['MAX_JOBS'] = '1'
        folder.mkdir(parents=True, exist_ok=True)
        # HIPify only cache copies, keeping submitted source files immutable.
        source = folder / 'rope_qk.cu'
        content = (root / 'ops/rope_qk.cu').read_bytes()
        if not source.exists() or source.read_bytes() != content:
            source.write_bytes(content)
        module = load(name=name, sources=[str(source)], build_directory=str(folder),
                      extra_cflags=HOST_FLAGS, extra_cuda_cflags=identity['device_flags'],
                      with_cuda=True, verbose=True)
        metadata = dict(identity, module_file=Path(module.__file__).name)
        (folder / 'build_identity.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return module
    metadata_file = folder / 'build_identity.json'
    if not metadata_file.is_file():
        raise RuntimeError('E14 extension is not built for this source/environment. Run bash build.sh before loading the model.')
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise RuntimeError('E14 build identity mismatch; rebuild outside the timed region.')
    binary_name = metadata.get('module_file', '')
    if not binary_name or Path(binary_name).name != binary_name:
        raise RuntimeError('Invalid E14 extension binary name.')
    binary = folder / binary_name
    if not binary.is_file():
        raise RuntimeError('E14 extension binary is missing; run bash build.sh.')
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def self_test(module, device):
    """Synthetic device tests, not model inference or contest reference fitting."""
    import torch
    device = torch.device(device)
    cases = [(1, 1, 0, 48, False), (1, 32, 0, 48, False),
             (8, 32, 0, 48, False), (4, 512, 0, 48, False),
             (4, 1, 511, 48, False), (8, 1, 1023, 48, False),
             (2, 64, 512, 48, True), (2, 9, 100, 96, True)]
    records = []
    generator = torch.Generator(device=device).manual_seed(1401)
    with torch.inference_mode():
        for batch, length, offset, rotary, permuted in cases:
            # Mimic strided Q/K views of packed QKV; also test permuted strides.
            shape = (batch, 16, length, 288) if permuted else (batch, length, 16, 288)
            storage = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            if permuted:
                storage = storage.permute(0, 2, 1, 3)
            q, k = storage[..., :96], storage[..., 192:]
            inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary, 2) / rotary))
            phase = torch.einsum('i,j->ij', torch.arange(1024), inv_freq).to(device).float()
            sin = torch.sin(phase).repeat_interleave(2, -1)[None, :, None, :]
            cos = torch.cos(phase).repeat_interleave(2, -1)[None, :, None, :]
            s, c = sin[:, offset:offset + length], cos[:, offset:offset + length]
            def reference(x):
                xr = x[..., :rotary]
                rotate = torch.stack((-xr[..., 1::2], xr[..., ::2]), dim=-1).flatten(-2)
                result = (xr * c) + (rotate * s)
                return torch.cat((result, x[..., rotary:]), -1)
            reference_q, reference_k = reference(q), reference(k)
            actual_q, actual_k = module.rope_qk(q, k, sin, cos, offset)
            exact = all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in [(actual_q, reference_q), (actual_k, reference_k)])
            records.append({'batch': batch, 'length': length, 'offset': offset,
                            'rotary': rotary, 'permuted': permuted, 'bitwise_equal': exact})
            if not exact:
                raise RuntimeError('E14 synthetic bitwise FP32 test failed: ' + str(records[-1]))
        # Check signed zero / very small finite values on a non-default stream.
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            values = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0**-126, -2.0**-126,
                                   2.0**-140, -2.0**-140], dtype=torch.float32, device=device)
            q = values.repeat(12).reshape(1, 1, 1, 96)
            k = q.flip(-1)
            s = torch.full((1, 1, 1, 48), 0.5, device=device, dtype=torch.float32)
            c = torch.full_like(s, 0.75)
            def special_reference(x):
                xr = x[..., :48]
                rotate = torch.stack((-xr[..., 1::2], xr[..., ::2]), dim=-1).flatten(-2)
                return torch.cat(((xr*c)+(rotate*s), x[..., 48:]), -1)
            actual_q, actual_k = module.rope_qk(q, k, s, c, 0)
            exact = all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in [(actual_q, special_reference(q)), (actual_k, special_reference(k))])
        stream.synchronize()
        if not exact:
            raise RuntimeError('E14 signed-zero/subnormal/current-stream test failed.')
        records.append({'special_values_and_current_stream': True, 'bitwise_equal': exact})
        try:
            module.rope_qk(q, k, s, c, 1)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('E14 failed to reject out-of-bounds positions.')
    return {'synthetic_tests': records, 'all_bitwise_equal': True,
            'model_loaded': False, 'formal_generation_test': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'cuda:0'))
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('hip_rope_validation.json'))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    module = load_extension(args.device, build=args.build)
    result = {'script_version': VERSION, 'extension_spec': extension_spec(args.device)[2],
              'module_file': str(module.__file__), 'compilation_requested': args.build}
    if args.self_test:
        result.update(self_test(module, args.device))
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('E14_HIP_BUILD_AND_SELF_TEST_OK' if args.self_test else 'E14_HIP_EXTENSION_READY', flush=True)
    print('VALIDATION:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
