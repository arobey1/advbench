import json
import hashlib
import os
import copy
import shlex
import numpy as np
import tqdm
import shutil
import argparse

from advbench.lib import misc
from advbench import algorithms
from advbench import datasets
from advbench import command_launchers

def make_args_list(cl_args):

    def _make_args(trial_seed, dataset, algorithm, hparams_seed):
        return {
            'dataset': dataset,
            'algorithm': algorithm,
            'hparams_seed': hparams_seed,
            'data_dir': cl_args.data_dir,
            'trial_seed': trial_seed,
            'seed': misc.seed_hash(dataset, algorithm, hparams_seed, trial_seed),
            'evaluators': cl_args.evaluators
        }

    args_list = []
    for trial_seed in range(cl_args.n_trials):
        for dataset in cl_args.datasets:
            for algorithm in cl_args.algorithms:
                for hparams_seed in range(cl_args.n_hparams):
                    args = _make_args(trial_seed, dataset, algorithm, hparams_seed)
                    args_list.append(args)

    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == 'y':
        print('Nevermind!')
        exit(0)

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'advbench.scripts.train']

        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(os.path.join(self.output_dir)):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (
            self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['hparams_seed']
        )
        return f'{self.state}: {self.output_dir} {job_info}'

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=datasets.DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--command_launcher', type=str, default='multi_gpu')
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--evaluators', type=str, nargs='+', default=['Clean'])
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    args_list = make_args_list(args)
    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:

        done_jobs = len([j for j in jobs if j.state == Job.DONE])
        incomp_jobs = len([j for j in jobs if j.state == Job.INCOMPLETE])
        unlaunched_jobs = len([j for j in jobs if j.state == Job.NOT_LAUNCHED])
        print(job)
        print(f'{len(jobs)} jobs: {done_jobs} done, {incomp_jobs} incomplete, {unlaunched_jobs} not launched.')

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)