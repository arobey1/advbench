import numpy as np
from advbench.lib import misc

def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}

def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}

def _hparams(algorithm: str, dataset: str, random_seed: int):
    """Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""

        assert(name not in hparams)
        random_state = np.random.RandomState(misc.seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('batch_size', 64, lambda r: int(2 ** r.uniform(3, 8)))

    # optimization
    _hparam('learning_rate', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    _hparam('sgd_momentum', 0.9, lambda r: r.uniform(0.8, 0.95))
    _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-6, -3))

    # Algorithm specific

    


    return hparams