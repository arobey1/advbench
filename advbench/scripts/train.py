import argparse
import torch
import os
import json
import pandas as pd
import time
from humanfriendly import format_timespan


from advbench import datasets
from advbench import algorithms
from advbench import attacks
from advbench import hparams_registry
from advbench.lib import misc, meters

# TODO(AR): Keep track of timing information

def main(args, hparams, test_hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = vars(datasets)[args.dataset](args.data_dir)
    train_ldr, val_ldr, test_ldr = datasets.to_loaders(dataset, hparams)

    algorithm = vars(algorithms)[args.algorithm](
        dataset.INPUT_SHAPE, 
        dataset.NUM_CLASSES,
        hparams,
        device).to(device)

    adjust_lr = None if dataset.HAS_LR_SCHEDULE is False else dataset.adjust_lr

    test_attacks = {
        a: vars(attacks)[a](algorithm.classifier, test_hparams, device) for a in args.test_attacks}
    
    columns = ['Epoch', 'Accuracy', 'Eval-Method', 'Split', 'Train-Alg', 'Dataset', 'Trial-Seed', 'Output-Dir']
    results_df = pd.DataFrame(columns=columns)
    def add_results_row(data):
        defaults = [args.algorithm, args.dataset, args.trial_seed, args.output_dir]
        results_df.loc[len(results_df)] = data + defaults

    for epoch in range(0, dataset.N_EPOCHS):

        if adjust_lr is not None:
            adjust_lr(algorithm.optimizer, epoch, hparams)

        loss_meter = meters.AverageMeter()
        timer = meters.TimeMeter()
        epoch_start = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_ldr):

            timer.batch_start()
            imgs, labels = imgs.to(device), labels.to(device)
            step_vals = algorithm.step(imgs, labels)
            loss_meter.update(step_vals['loss'], n=imgs.size(0))

            if batch_idx % dataset.LOG_INTERVAL == 0:
                print(f'Train epoch {epoch}/{dataset.N_EPOCHS} ', end='')
                print(f'[{batch_idx * imgs.size(0)}/{len(train_ldr.dataset)} ({100. * batch_idx / len(train_ldr):.0f}%)]\t', end='')
                print(f'Loss: {step_vals["loss"]:.3f} (avg. {loss_meter.avg:.3f})\t', end='')
                print(f'Time: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')

            timer.batch_end()

        # save clean accuracies on validation/test sets
        val_clean_acc = misc.accuracy(algorithm, val_ldr, device)
        add_results_row([epoch, val_clean_acc, 'ERM', 'Validation'])

        test_clean_acc = misc.accuracy(algorithm, test_ldr, device)
        add_results_row([epoch, test_clean_acc, 'ERM', 'Test'])

        # save adversarial accuracies on validation/test sets
        val_adv_accs = []
        for attack_name, attack in test_attacks.items():
            val_adv_acc = misc.adv_accuracy(algorithm, val_ldr, device, attack)
            add_results_row([epoch, val_adv_acc, attack_name, 'Validation'])
            val_adv_accs.append(val_adv_acc)

            test_adv_acc = misc.adv_accuracy(algorithm, test_ldr, device, attack)
            add_results_row([epoch, test_adv_acc, attack_name, 'Test'])

        epoch_end = time.time()

        # print results
        print(f'Epoch: {epoch+1}/{dataset.N_EPOCHS}\t', end='')
        print(f'Epoch time: {format_timespan(epoch_end - epoch_start)}\t', end='')
        print(f'Training alg: {args.algorithm}\t', end='')
        print(f'Dataset: {args.dataset}\t', end='')
        print(f'Path: {args.output_dir}')
        print(f'Avg. train loss: {loss_meter.avg:.3f}\t', end='')
        print(f'Clean val. accuracy: {val_clean_acc:.3f}\t', end='')
        for attack_name, acc in zip(test_attacks.keys(), val_adv_accs):
            print(f'{attack_name} val. accuracy: {acc:.3f}\t', end='')
        print('\n')

        # save results dataframe to file
        results_df.to_pickle(os.path.join(args.output_dir, 'results.pkl'))

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