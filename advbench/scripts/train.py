import argparse
import torch
import os
from collections import defaultdict
import json

from advbench import datasets
from advbench import algorithms
from advbench import attacks
from advbench import hparams_registry
from advbench.lib import misc, meters

def main(args, hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    dataset = vars(datasets)[args.dataset](args.data_dir)
    train_ldr, val_ldr, test_ldr = datasets.to_loaders(dataset, hparams)

    algorithm = vars(algorithms)[args.algorithm](
        dataset.INPUT_SHAPE, 
        dataset.NUM_CLASSES,
        hparams).to(device)

    test_attacks = {
        a: vars(attacks)[a](algorithm.classifier, hparams) for a in args.test_attacks}

    timer = meters.TimeMeter()
    
    metrics = defaultdict(lambda: defaultdict(dict))
    step_dict = {}

    for epoch in range(0, dataset.N_EPOCHS):

        loss_meter = meters.AverageMeter()
        for batch_idx, (imgs, labels) in enumerate(train_ldr):

            global_step = epoch * len(train_ldr) + batch_idx

            timer.batch_start()
            imgs, labels = imgs.to(device), labels.to(device)
            step_vals = algorithm.step(imgs, labels)
            step_dict[global_step] = step_vals
            loss_meter.update(step_vals['loss'], n=imgs.size(0))
            timer.batch_end()

        metrics[epoch]['val']['clean'] = misc.accuracy(algorithm, val_ldr, device)
        metrics[epoch]['test']['clean'] = misc.accuracy(algorithm, test_ldr, device)

        for attack_name, attack in test_attacks.items():
            metrics[epoch]['val'][attack_name] = misc.adv_accuracy(algorithm, val_ldr, device, attack)
            metrics[epoch]['test'][attack_name] = misc.adv_accuracy(algorithm, test_ldr, device, attack)
            
        print(f'Epoch: {epoch+1}/{dataset.N_EPOCHS}')
        print(f'Avg. train loss: {loss_meter.avg}\t', end='')
        print(f'Clean val. accuracy: {metrics[epoch]["val"]["clean"]:.3f}\t', end='')
        for attack_name in test_attacks.keys():
            print(f'{attack_name} val. accuracy: {metrics[epoch]["test"][attack_name]:.3f}\t', end='')
        print('\n')

        with open(os.path.join(args.output_dir, 'results.json'), 'a') as f:
            f.write(json.dumps(metrics[epoch], sort_keys=True) + "\n")

    torch.save(
        {'model': algorithm.state_dict()}, 
        os.path.join(args.output_dir, f'ckpt.pkl'))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--data_dir', type=str, default='./advbench/data')
    parser.add_argument('--output_dir', type=str, default='train_output')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    parser.add_argument('--test_attacks', type=str, nargs='+', default=['PGD_Linf'])
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    args = parser.parse_args()

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

    main(args, hparams)