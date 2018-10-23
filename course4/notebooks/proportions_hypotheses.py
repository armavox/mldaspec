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


def proportions_diff_z_stat_ind(sample1, sample2, alternative='two-sided'):
    """Proportion test fot two proportions based on normal distribution.

    Parameters
    ----------
    sample1 : array_like
    sample2 : array_like

    Returns
    -------
    z_stat : float
    p_value : float

    """
    n1 = len(sample1)
    n2 = len(sample2)

    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2
    P = float(p1*n1 + p2*n2) / (n1 + n2)

    z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))
    p_value = _proportions_diff_z_test(z_stat, alternative)

    return z_stat, p_value


def proportions_diff_z_stat_rel(sample1, sample2, alternative='two-sided'):
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
    p_value : float

    """
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    z_stat = float(f - g) / np.sqrt(f + g - float((f - g)**2) / n)
    p_value = _proportions_diff_z_test(z_stat, alternative)

    return z_stat, p_value


def _proportions_diff_z_test(z_stat, alternative='two-sided'):
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


def cramers_corrected_stat(confusion_matrix):
    """Calculates Cramers V statistic for
        categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328

    Parameters
    ----------
    confusion_matrix : pandas confusion matrix
        >>> import pandas as pd
        >>> confusion_matrix = pd.crosstab(dataframe_1, dataframe_2)

    Returns
    -------
    Cramers'V : float
        Cramers'V correlation coefficient
    p_value : float
        p_value of hypothesis testing with null-hypothesis of
        Cramers'V is equal zero.

    """
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2[0] / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    return cramers_v, chi2[1]


def check_ct(ct):
    """Checks following conditions to use chi2_contingency and Cramers'V.

    Conditions are as follows:
        - n_obs >= 40
        - (ni+ * n+j / n_obs) < 5 in less than 20% cells
        Where ni+ is sum of the i-th row in contingency table and n+j is
        the sum of the i-th columns in the contingency table.

    Parameters
    ----------
    ct : pd.crosstab(df.feature_1, df.feature_2) or ndarray
        pandas crosstab or any confusion matrix

    Returns
    -------
    num_obs : int
        number of observations
    observed_prop : float

    Notes
    -----
    num_obs must be equal or greater than 40
    observed_prop must be less than 0.2

    """
    ct_check = ct.copy()
    for i in ct.index:
        for j in ct.columns:
            ct_check.loc[i, j] = (ct.loc[i].sum() * ct[j].sum()) / ct.sum().sum()
    n_obs = ct.sum().sum()
    observed_prop = (ct_check < 5).values.mean()
    return n_obs, observed_prop
