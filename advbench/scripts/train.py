import argparse
import torch

from advbench import datasets
from advbench import algorithms
from advbench import hparams_registry
from advbench.lib import misc, meters

def main(args, hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = vars(datasets)[args.dataset](args.data_dir)
    train_ldr, val_ldr, test_ldr = datasets.to_loaders(dataset, hparams)


    algorithm = vars(algorithms)[args.algorithm](
        dataset.input_shape, 
        dataset.num_classes,
        hparams).to(device)

    timer = meters.TimeMeter()
    train_acc_meter = meters.AverageMeter()

    for epoch in range(0, dataset.n_epochs):

        for batch_idx, (imgs, labels) in enumerate(train_ldr):

            timer.batch_start()
            imgs, labels = imgs.to(device), labels.to(device)
            step_vals = algorithm.step(imgs, labels)
        
            # print(step_vals['loss'])

            timer.batch_end()

        val_acc = misc.accuracy(algorithm, val_ldr, device)
        test_acc = misc.accuracy(algorithm, test_ldr, device)

        print(test_acc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--data_dir', type=str, default='./advbench/data')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        seed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, seed)

    print ('Hparams:')
    for k, v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    # TODO(AR): need some way of saving hparams and args

    main(args, hparams)