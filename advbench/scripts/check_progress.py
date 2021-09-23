import os
import argparse

def main(args):
    
    n_dirs, n_done = 0, 0
    for d in os.listdir(args.input_dir):
        pth = os.path.join(args.input_dir, d)
        if os.path.isdir(pth) is True:
            done_pth = os.path.join(pth, 'done')
            if os.path.exists(done_pth):
                n_done += 1
            n_dirs += 1

    print(f'Completed tasks: {n_done}/{n_dirs} ({100 * float(n_done)/n_dirs:.2f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check progress of sweep')
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
