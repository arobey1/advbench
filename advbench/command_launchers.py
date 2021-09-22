

import subprocess
import time
import torch

def local_launcher(commands):
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):

    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None for _ in range(n_gpus)]

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]

            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}',
                    shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break

        time.sleep(1)
    
    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher
}