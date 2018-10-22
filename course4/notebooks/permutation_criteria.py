import itertools
import numpy as np


def _permutation_t_stat_1sample(sample, mean):
    t_stat = sum(map(lambda x: x - mean, sample))
    return t_stat


def _permutation_zero_distr_1sample(sample, mean, max_permutations=None):
    centered_sample = list(map(lambda x: x - mean, sample))
    if max_permutations:
        signs_array = set(
            [tuple(x) for x in 2 * np.random.randint(
                2, size=(max_permutations, len(sample))) - 1])
    else:
        signs_array = itertools.product([-1, 1], repeat=len(sample))
    distr = [sum(centered_sample * np.array(signs)) for signs in signs_array]
    return distr


def permutation_test_1sample(sample, mean, max_permutations=None,
                             alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = _permutation_t_stat_1sample(sample, mean)

    zero_distr = _permutation_zero_distr_1sample(sample, mean,
                                                 max_permutations)

    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)


def _permutation_t_stat_ind(sample1, sample2):
    import numpy as np
    return np.mean(sample1) - np.mean(sample2)


def _get_random_combinations(n1, n2, max_combinations):
    import numpy as np
    index = np.arange(n1 + n2)
    indices = set([tuple(index)])
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    return [(index[:n1], index[n1:]) for index in indices]


def _permutation_zero_dist_ind(sample1, sample2, max_combinations=None):
    import numpy as np
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_combinations:
        indices = _get_random_combinations(n1, len(sample2), max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n)))
                   for index in itertools.combinations(range(n), n1)]

    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean()
             for i in indices]
    return distr


def permutation_test_2sample_ind(sample1, sample2, max_permutations=1000,
                                 alternative='two-sided'):

    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = _permutation_t_stat_ind(sample1, sample2)

    zero_distr = _permutation_zero_dist_ind(sample1, sample2, max_permutations)

    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)
