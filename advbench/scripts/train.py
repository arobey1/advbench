import argparse
import torch
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import time
import collections
from humanfriendly import format_timespan

from advbench import datasets
from advbench import algorithms
from advbench import evalulation_methods
from advbench import hparams_registry
from advbench.lib import misc, meters, reporting

def main(args, hparams, test_hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    # paths for saving output
    json_path = os.path.join(args.output_dir, 'results.json')
    ckpt_path = misc.stage_path(args.output_dir, 'ckpts')
    train_df_path = os.path.join(args.output_dir, 'train.pd')
    selection_df_path = os.path.join(args.output_dir, 'selection.pd')

    dataset = vars(datasets)[args.dataset](args.data_dir, device)

    train_loader = DataLoader(
        dataset=dataset.splits['train'],
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        pin_memory=False,
        shuffle=True)
    validation_loader = DataLoader(
        dataset=dataset.splits['validation'],
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        pin_memory=False,
        shuffle=False)
    test_loader = DataLoader(
        dataset=dataset.splits['test'],
        batch_size=100,
        num_workers=dataset.N_WORKERS,
        pin_memory=False,
        shuffle=False)

    algorithm = vars(algorithms)[args.algorithm](
        dataset.INPUT_SHAPE, 
        dataset.NUM_CLASSES,
        hparams,
        device).to(device)

    def save_checkpoint(epoch):
        torch.save(
            obj={'state_dict': algorithm.state_dict()}, 
            f=os.path.join(ckpt_path, f'model_ckpt_{epoch}.pkl')
        )

    evaluators = [
        vars(evalulation_methods)[e](
            algorithm=algorithm,
            device=device,
            test_hparams=test_hparams)
        for e in args.evaluators]

    adjust_lr = None if dataset.HAS_LR_SCHEDULE is False else dataset.adjust_lr

    total_time = 0
    for epoch in range(0, dataset.N_EPOCHS):

        if adjust_lr is not None:
            adjust_lr(algorithm.optimizer, epoch, hparams)

        timer = meters.TimeMeter()
        epoch_start = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_loader):

            timer.batch_start()
            if not dataset.ON_DEVICE:
                imgs, labels = imgs.to(device), labels.to(device)
            algorithm.step(imgs, labels)

            if batch_idx % dataset.LOG_INTERVAL == 0:
                print(f'Epoch {epoch+1}/{dataset.N_EPOCHS} ', end='')
                print(f'[{batch_idx * imgs.size(0)}/{len(train_loader.dataset)}', end=' ')
                print(f'({100. * batch_idx / len(train_loader):.0f}%)]\t', end='')
                for name, meter in algorithm.meters.items():
                    print(f'{name}: {meter.val:.3f} (avg. {meter.avg:.3f})\t', end='')
                print(f'Time: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')

            timer.batch_end()

        results = {'Epoch': epoch, 'Train': {}, 'Validation': {}, 'Test': {}}

        for name, meter in algorithm.meters.items():
            results['Train'].update({name: meter.avg})

        for evaluator in evaluators:
            for k, v in evaluator.calculate(validation_loader).items():
                results['Validation'].update({k: v})

        for evaluator in evaluators:
            for k, v in evaluator.calculate(test_loader).items():
                results['Test'].update({k: v})

        epoch_time = time.time() - epoch_start
        total_time += epoch_time

        results.update({
            'Epoch-Time': epoch_time,
            'Total-Time': total_time})

        # print results
        print(f'Epoch: {epoch+1}/{dataset.N_EPOCHS}\t', end='')
        print(f'Epoch time: {format_timespan(epoch_time)}\t', end='')
        print(f'Total time: {format_timespan(total_time)}')

        results.update({'hparams': hparams, 'args': vars(args)})

        with open(json_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + '\n')

        if args.save_model_every_epoch is True:
            save_checkpoint(epoch)

        algorithm.reset_meters()        

    save_checkpoint('final')

    records = reporting.load_record(json_path)

    train_dict = collections.defaultdict(lambda: [])
    validation_dict = collections.defaultdict(lambda: [])
    test_dict = collections.defaultdict(lambda: [])

    for record in records:
        for k in records[0]['Train'].keys():
            train_dict[k].append(record['Train'][k])

        for k in records[0]['Validation'].keys():
            validation_dict[k].append(record['Validation'][k])
            test_dict[k].append(record['Test'][k])

    def dict_to_dataframe(split, d):
        df = pd.DataFrame.from_dict(d)
        df['Split'] = split
        df = df.join(pd.DataFrame({
            'Algorithm': args.algorithm,
            'trial_seed': args.trial_seed,
            'seed': args.seed,
            'path': args.output_dir
        }, index=df.index))
        df['Epoch'] = range(dataset.N_EPOCHS)
        return df

    train_df = dict_to_dataframe('Train', train_dict)
    validation_df = dict_to_dataframe('Validation', validation_dict)
    test_df = dict_to_dataframe('Test', test_dict)
    selection_df = pd.concat([validation_df, test_df], ignore_index=True)

    train_df.to_pickle(train_df_path)
    selection_df.to_pickle(selection_df_path)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness')
    parser.add_argument('--data_dir', type=str, default='./advbench/data')
    parser.add_argument('--output_dir', type=str, default='train_output')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--evaluators', type=str, nargs='+', default=['Clean'])
    parser.add_argument('--save_model_every_epoch', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        seed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, seed)

    print ('Hparams:')
    for k, v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=2)

    test_hparams = hparams_registry.test_hparams(args.algorithm, args.dataset)

    print('Test hparams:')
    for k, v in sorted(test_hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'test_hparams.json'), 'w') as f:
        json.dump(test_hparams, f, indent=2)

    main(args, hparams, test_hparams)