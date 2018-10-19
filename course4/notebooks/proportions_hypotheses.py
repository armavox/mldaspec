import numpy as np
import scipy


def proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    """Confidence interval for two independent proportions.

    Parameters
    ----------
    sample1 : array_like
        array of zeros and ones, which in mean returns proportion
        of success.
    sample2 : array_like

    Returns
    -------
    boundaries : tuple

    """
    z = scipy.stats.norm.ppf(1 - alpha / 2.)

    p1 = float(sum(sample1)) / len(sample1)
    p2 = float(sum(sample2)) / len(sample2)

    left_boundary = (
        (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))
        )
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))

    return (left_boundary, right_boundary)


def proportions_diff_confint_rel(sample1, sample2, alpha=0.05):
    """Confidence interval for two related proportions.

    Parameters
    ----------
    sample1 : array_like
        array of zeros and ones, which in mean returns proportion of success.
    sample2 : array_like

    Returns
    -------
    boundaries : tuple

    """
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    left_boundary = float(f - g) / n - z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    right_boundary = float(f - g) / n + z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    return (left_boundary, right_boundary)


def proportions_diff_z_stat_ind(sample1, sample2):
    """Proportion test fot two proportions based on normal distribution.

    Parameters
    ----------
    sample1 : array_like
    sample2 : array_like

    Returns
    -------
    z_stat : float

    """
    n1 = len(sample1)
    n2 = len(sample2)

    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2
    P = float(p1*n1 + p2*n2) / (n1 + n2)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_stat_rel(sample1, sample2):
    """Returns z-statistic for two samples of proportions.

    Parameters
    ----------
    sample1 : array_like
        array of zeros and ones, which in mean returns proportion
        of success.
    sample2 : array_like

    Returns
    -------
    z_stat : float

    """
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    return float(f - g) / np.sqrt(f + g - float((f - g)**2) / n)


def proportions_diff_z_test(z_stat, alternative='two-sided'):
    """Return p-value based on normal distribution test.

    """
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)
