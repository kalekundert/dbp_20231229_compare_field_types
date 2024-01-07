#!/usr/bin/env python3

import re
import sys

from pathlib import Path
from subprocess import run
from tqdm import tqdm
from tabulate import tabulate
from dataclasses import dataclass

@dataclass
class JobInfo:
    job_id: int
    hparam_id: int
    hparam_name: str
    status: str
    wall_time: str
    max_rss_gb: float
    gpu: str

jobs = []

root = Path(__file__).parent

hparam_ids = {
        'regular_max_freqs_1_fourier_gelu':            0,
        'regular_max_freqs_2_fourier_gelu':            1,
        'regular_max_freqs_1_2_2_fourier_gelu':        2,
        'quotient_max_freqs_1_fourier_gelu':           3,
        'quotient_max_freqs_2_fourier_gelu':           4,
        'quotient_max_freqs_3_fourier_gelu':           5,
        'quotient_max_freqs_4_fourier_gelu':           6,
        'quotient_max_freqs_5_fourier_gelu':           7,
        'quotient_max_freqs_4_5_5_fourier_gelu':       8,
        'regular_max_freqs_1_gated_gelu_unpack':       9,
        'regular_max_freqs_1_gated_gelu':             10,
        'regular_max_freqs_2_gated_gelu_unpack':      11,
        'regular_max_freqs_2_gated_gelu':             12,
        'regular_max_freqs_1_2_2_gated_gelu_unpack':  13,
        'regular_max_freqs_1_2_2_gated_gelu':         14,
        'quotient_max_freqs_1_gated_gelu_unpack':     15,
        'quotient_max_freqs_1_gated_gelu':            16,
        'quotient_max_freqs_2_gated_gelu_unpack':     17,
        'quotient_max_freqs_2_gated_gelu':            18,
        'quotient_max_freqs_3_gated_gelu_unpack':     19,
        'quotient_max_freqs_3_gated_gelu':            20,
        'quotient_max_freqs_4_gated_gelu_unpack':     21,
        'quotient_max_freqs_4_gated_gelu':            22,
        'quotient_max_freqs_5_gated_gelu_unpack':     23,
        'quotient_max_freqs_5_gated_gelu':            24,
        'quotient_max_freqs_4_5_5_gated_gelu_unpack': 25,
        'quotient_max_freqs_4_5_5_gated_gelu':        26,
        'polynomial_terms_1_gated_gelu':              27,
        'polynomial_terms_2_gated_gelu':              28,
        'polynomial_terms_3_gated_gelu':              29,
        'polynomial_terms_4_gated_gelu':              30,
        'polynomial_terms_3_4_4_gated_gelu':          31,
        'single_irrep_1_gated_gelu':                  32,
        'single_irrep_2_gated_gelu':                  33,
        'single_irrep_3_gated_gelu':                  34,
}

for log_path in tqdm(list(root.glob('*.err'))):
    log = log_path.read_text()

    m = re.fullmatch(r'compare_field_types.sbatch_(\d+).err', log_path.name)
    job_id = m.group(1)

    m = re.search(r'INFO: using hyperparameters: (.*)', log)
    hparam_name = m.group(1)

    if m := re.search('slurmstepd: error: Detected (\d+) oom_kill event', log):
        status = f'out-of-memory ({m.group(1)})'

    elif 'torch.cuda.OutOfMemoryError: CUDA out of memory.' in log:
        status = f'out-of-memory (cuda)'

    elif 'Representation "irrep_0 X irrep_1" does not support "gated" non-linearity' in log:
        status = 'error (gated polynomial)'

    elif '`Trainer.fit` stopped: `max_epochs=50` reached.' in log:
        status = 'complete'

    else:
        status = 'unknown'

    wall_time = None
    max_rss_gb = None

    if '--fast' not in sys.argv:
        p = run(['seff', job_id], capture_output=True, text=True)

        if m := re.search(r'Job Wall-clock time: (\d+:\d+:\d+)', p.stdout):
            wall_time = m.group(1)

        if m := re.search(r'Memory Utilized: (\d+\.\d+) GB', p.stdout):
            max_rss_gb = float(m.group(1))

    m = re.search(r'GPU 0: (.*) \(UUID:', log)
    gpu = m.group(1)

    jobs.append(JobInfo(
        job_id = job_id,
        hparam_id = hparam_ids[hparam_name],
        hparam_name = hparam_name,
        status = status,
        wall_time = wall_time,
        max_rss_gb = max_rss_gb,
        gpu = gpu,
    ))

    if '--move' in sys.argv and status != 'complete':
        log_path.rename(root / 'incomplete' / log_path.name)

        out_path = log_path.with_suffix('.out')
        out_path.rename(root / 'incomplete' / out_path.name)

        traj_path = root / hparam_name
        if traj_path.exists():
            traj_path.rename(root / 'incomplete' / traj_path.name)

jobs.sort(key=lambda x: x.hparam_id)

print(tabulate(jobs, headers='keys'))
