class SeriesModel:
    """Handmade class for fit SARIMAX models from statsmodels library.

    Base method – fit_sarimax

    Parameters
    ---------
    series : ndarray or array-like
        Time-series array.

    Attributes
    ----------
    best_model : statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        Statsmodels wrapper. To return result, use ``summary()``

    results_table : pandas DataFrame
        Sorted by increase, table of parameters with AIC (Akaike
        Information Criterion) for each combination of parameters.
        use ``results_table.head()`` for top 5 models.

    """
    def __init__(self, series):
        self.series = series
        self.best_model = None
        self.results_table = None

    def fit_sarimax(self, p, d, q, P, D, Q, S):
        """Method to find best params for SARIMAX model

        Parameters
        ----------
        d : int
            Number of times of non-seasonal differentioation of the
            series.
        D : int
            Number of times of seasonal differentioation of the
            series.
        S : int
            Number of lags in series which represent one season.
        q, Q, p, P : int
            Parameters for SARIMAX(p, d, q)x(P, D, Q, S) model.

        """
        import warnings
        from itertools import product as prod
        import pandas as pd
        import statsmodels.api as sm

        ps = range(p+1)
        qs = range(q+1)
        Ps = range(P+1)
        Qs = range(Q+1)

        parameters_list = list(prod(ps, qs, Ps, Qs))
        results = []
        best_aic = float("inf")
        warnings.filterwarnings('ignore')

        for param in parameters_list:
            # The 'try except' block is nessesary because of
            # some combinations of params are incompatible
            # with the model.
            try:
                model = sm.tsa.statespace.SARIMAX(
                    self.series,
                    order=(param[0], d, param[1]),
                    seasonal_order=(param[2], D, param[3], S)).fit(disp=-1)
            # Show parameters which is not compatible with the model
            # and continue.
            except ValueError:
                print('Wrong parameters:', param)
                continue

            # Saving the best model, AIC, and params.
            aic = model.aic
            if aic < best_aic:
                self.best_model = model
                best_aic = aic
            results.append([param, model.aic])

        warnings.filterwarnings('default')

        results_table = pd.DataFrame(results)
        results_table.columns = ['p, q, P, Q', 'AIC']
        self.results_table = results_table.sort_values(
            by='AIC', ascending=True)
