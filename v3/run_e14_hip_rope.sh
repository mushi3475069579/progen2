#!/usr/bin/env bash
# E14 platform experiment; upload this single file beside run_public_test.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -u - <<'E14_PLATFORM_RUNNER'
"""E14 platform runner template. Candidate sources are embedded by the builder."""
from __future__ import annotations

from datetime import datetime
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zlib

PAYLOAD = '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'
EXPECTED_E12 = {
    'solution.py': '452a6de690586326e24235f3a850fe484789228150e0139cffbcf2accb9e5c59',
    'rope_cache.py': 'b4153a9cdaabff2e849725227e5a3d539c16816e3a9488fcbd18690e5f11db97',
}
EXPECTED_LOADS = [
    ('load1_b1_l32_g256', 1, 32, 256), ('load2_b8_l32_g256', 8, 32, 256),
    ('load3_b4_l32_g512', 4, 32, 512), ('load4_b4_l512_g64', 4, 512, 64),
]
SOURCE_SUFFIXES = {'.py', '.sh', '.cu', '.hip', '.cpp', '.h'}
IGNORED = shutil.ignore_patterns('evidence', '.cache', '__pycache__', '*.pyc', 'report.pdf', 'report.docx')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.suffix in SOURCE_SUFFIXES and
            not any(part in {'__pycache__', '.cache', 'evidence'} for part in p.relative_to(folder).parts)}


def validate_result(data, label):
    rows = data.get('loads', [])
    if len(rows) != 4:
        raise RuntimeError(label + ': expected four complete loads')
    for row, expected in zip(rows, EXPECTED_LOADS):
        actual = tuple(row.get(key) for key in ['load_id', 'batch_size', 'prefix_tokens', 'new_tokens'])
        if actual != expected:
            raise RuntimeError(label + ': task identity mismatch')
        time_s = row.get('time_s')
        if not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s <= 0:
            raise RuntimeError(label + ': invalid measured time')
    return rows


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    root = Path.cwd().resolve()
    if not all((root / name).is_file() for name in ['run_public_test.sh', 'public_test.py', 'check_environment.py']):
        raise RuntimeError('Upload run_e14_hip_rope.sh to the contest root and run it there.')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Activate the existing project .venv first.')
    current = root / 'submission'
    if sources(current) != EXPECTED_E12:
        raise RuntimeError('Current submission is not the expected E12/v2 source set. No source changed.')
    previous = None
    for path in sorted(root.glob('results/E12_rope_expand_*/evidence/submission_public.json'), reverse=True):
        directory = path.parent.parent
        try:
            data = read(path)
            validate_result(data, 'E12')
            if sources(directory / 'submission') != EXPECTED_E12:
                continue
            if not data.get('all_public_references_exact') or not all(
                row.get('exact_match') is True and row.get('consistency_rate') == 1.0 for row in data['loads']
            ):
                continue
            if not all((directory / 'evidence' / name).is_file() for name in
                       ['baseline_public.json', 'baseline.log', 'submission.log', 'environment_check.log']):
                continue
            previous = directory
            break
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
    if previous is None:
        raise RuntimeError('Complete matching E12 directory not found. Keep/restore the original E12 run folder first. No source changed.')

    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MAX_JOBS']:
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    device = torch.device(os.environ.get('DEVICE', 'cuda:0'))
    if device.type != 'cuda' or not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError('Allocated DCU/HIP unavailable. No source changed.')
    actual = {'torch': str(torch.__version__), 'hip': torch.version.hip,
              'device': torch.cuda.get_device_name(device)}
    official, prior = read(previous / 'evidence/baseline_public.json'), read(previous / 'evidence/submission_public.json')
    validate_result(official, 'Official baseline')
    for data in [official, prior]:
        for key in ['torch', 'hip', 'device']:
            if data.get(key) != actual[key]:
                raise RuntimeError('Environment differs from saved baseline/E12: ' + key + '. No source changed.')

    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run = root / 'results' / ('E14_hip_rope_' + tag)
    run.mkdir(parents=True, exist_ok=False)
    print('E14_OUTPUT:', run, flush=True)
    try:
        candidate = run / 'submission'
        candidate.mkdir()
        payload = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
        for name, value in payload.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Invalid embedded source path')
            destination = candidate / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value.encode('utf-8'))
        snapshot = sources(candidate)
        write(run / 'source_sha256.json', {'submission/' + name: value for name, value in snapshot.items()})
        write(run / 'preparation_environment.json', actual)
        write(run / 'previous_version.json', {'previous_run': str(previous), 'source_sha256': EXPECTED_E12})
        (run / 'change_notes.txt').write_text(
            'E14 candidate on E12: one HIP Q/K RoPE kernel including passthrough channels; '
            'FP32, preexpanded shared coefficients, model weights, original attention/matrices, '
            'dynamic request-local KV and deterministic generation preserved. Build/self-test outside generation. '
            'No package installation. Actual acceptance/speed depend on this run.\n', encoding='utf-8')
        # Preserve E12 before any candidate testing or activation.
        backup = root / 'versions' / ('E12_before_E14_' + tag)
        shutil.copytree(current, backup / 'submission', ignore=IGNORED)
        shutil.copytree(previous / 'evidence', backup / 'evidence')
        print('E12_BACKUP:', backup, flush=True)
        evidence = run / 'evidence'
        evidence.mkdir()
        for name in ['baseline_public.json', 'baseline.log', 'environment_check.log']:
            shutil.copy2(previous / 'evidence' / name, evidence / name)
        shutil.copy2(previous / 'evidence/submission_public.json', evidence / 'previous_public.json')
        shutil.copy2(previous / 'evidence/submission.log', evidence / 'previous.log')
        write(run / 'preparation.json', {'e12_backup': str(backup), 'candidate': str(candidate),
                                        'source_sha256': snapshot, 'model_loaded_by_runner': False})

        env = os.environ.copy()
        env['RESULT_FILE'] = str(evidence / 'submission_public.json')
        env['PYTHONUNBUFFERED'] = '1'
        # The unchanged official script performs environment check, then build.sh
        # and synthetic tests outside timing, then all four complete model loads.
        with (evidence / 'submission.log').open('wb') as log:
            process = subprocess.Popen(['bash', 'run_public_test.sh', str(candidate)], cwd=root,
                                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for chunk in iter(process.stdout.readline, b''):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            status = process.wait()
        if sources(candidate) != snapshot:
            raise RuntimeError('Candidate source changed during testing; no activation or comparison allowed.')
        if status:
            raise RuntimeError('Official test/build failed, exit=' + str(status) + '. E12 remains active; retain complete log.')
        current_data = read(evidence / 'submission_public.json')
        measured = validate_result(current_data, 'E14')
        for key in ['torch', 'hip', 'device']:
            if current_data.get(key) != actual[key]:
                raise RuntimeError('Measured environment mismatch: ' + key)
        exact = current_data.get('all_public_references_exact') is True and all(
            row.get('exact_match') is True and row.get('consistency_rate') == 1.0 for row in measured
        )
        all_faster = all(new['time_s'] < old['time_s'] for new, old in zip(measured, prior['loads']))
        lines = ['load_id                    official_s E12_s      E14_s      official/E14 E12/E14 consistency']
        for a, b, c in zip(official['loads'], prior['loads'], measured):
            lines.append(f'{c["load_id"]:<26} {a["time_s"]:9.4f} {b["time_s"]:10.4f} {c["time_s"]:10.4f} '
                         f'{a["time_s"]/c["time_s"]:12.4f} {b["time_s"]/c["time_s"]:7.4f} {100*c["consistency_rate"]:10.2f}%')
        lines += ['E14_ALL_FOUR_EXACT: ' + str(exact).lower(),
                  'One official timed run per load; no stable speedup claim.']
        text = '\n'.join(lines) + '\n'
        (run / 'comparison.txt').write_text(text, encoding='utf-8')
        print(text, flush=True)
        if not exact:
            write(run / 'activation.json', {'activated': False, 'reason': 'not_all_exact', 'e12_backup': str(backup)})
            raise RuntimeError('E14 did not retain 100% sequence match. Candidate retained, E12 remains active.')
        if sources(current) != EXPECTED_E12:
            raise RuntimeError('Current submission changed concurrently; no activation.')
        if all_faster:
            added, replaced = [], []
            try:
                for name in payload:
                    destination = current / name
                    (replaced if destination.exists() else added).append(name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + '.e14_pending')
                    temporary.write_bytes((candidate / name).read_bytes())
                    temporary.replace(destination)
                if sources(current) != snapshot:
                    raise RuntimeError('Activated source fingerprint mismatch')
            except Exception:
                for name in replaced:
                    shutil.copy2(backup / 'submission' / name, current / name)
                for name in added:
                    (current / name).unlink(missing_ok=True)
                raise
            write(run / 'activation.json', {'activated': True, 'basis': 'all_exact_and_all_four_faster_this_run',
                                            'e12_backup': str(backup), 'source_sha256': snapshot})
            print('E14_ACTIVATED: current submission updated. E12 backup retained.', flush=True)
        else:
            write(run / 'activation.json', {'activated': False, 'reason': 'not_all_four_faster_this_run',
                                            'e12_backup': str(backup)})
            print('E14_NOT_ACTIVATED: candidate completed, but not all four were faster; current E12 unchanged.', flush=True)
        (run / 'COMPLETED').write_text('Official four-load candidate test and comparison completed.\n', encoding='utf-8')
        print('E14_DONE:', run, flush=True)
        print('Archive this complete E14 directory, including its submission and evidence.', flush=True)
    except BaseException:
        (run / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('E14_INCOMPLETE: retain directory and error/log:', run, flush=True)
        raise


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)

E14_PLATFORM_RUNNER
