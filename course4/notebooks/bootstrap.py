import numpy as _np


def get_bootstrap_samples(data, n_samples):
    ints = _np.random.randint(0, len(data), size=(n_samples, len(data)))
    return _np.asarray(data)[ints]


def tolerance_int(stat, alpha):
    return _np.percentile(stat, [100*alpha/2, 100*(1-alpha/2)])
