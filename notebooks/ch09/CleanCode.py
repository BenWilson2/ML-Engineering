import warnings as warn
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stat
from scipy.stats import shapiro, normaltest, anderson
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot


class DistributionAnalysis(object):

    def __init__(self, series, histogram_bins, **kwargs):
        self.series = series
        self.histogram_bins = histogram_bins
        self.series_name = kwargs.get('series_name', 'data')
        self.plot_bins = kwargs.get('plot_bins', 200)
        self.best_plot_size = kwargs.get('best_plot_size', (20, 16))
        self.all_plot_size = kwargs.get('all_plot_size', (24, 30))
        self.MIN_BOUNDARY = 0.001
        self.MAX_BOUNDARY = 0.999
        self.ALPHA = kwargs.get('alpha', 0.05)

    def _get_series_bins(self):
        return int(np.ceil(self.series.index.values.max()))

    @staticmethod
    def _get_distributions():
        scipy_ver = scipy.__version__
        if (int(scipy_ver[2]) >= 5) and (int(scipy_ver[4:]) > 3):
            names, gen_names = stat.get_distribution_names(stat.pairs, stat.rv_continuous)
        else:
            names = stat._continuous_distns._distn_names
        return names

    @staticmethod
    def _extract_params(params):
        return {'arguments': params[:-2], 'location': params[-2], 'scale': params[-1]}

    @staticmethod
    def _generate_boundaries(distribution, parameters, x):
        args = parameters['arguments']
        loc = parameters['location']
        scale = parameters['scale']
        return distribution.ppf(x, *args, loc=loc, scale=scale) if args else distribution.ppf(x, loc=loc, scale=scale)

    @staticmethod
    def _build_pdf(x, distribution, parameters):
        if parameters['arguments']:
            pdf = distribution.pdf(x, loc=parameters['location'], scale=parameters['scale'], *parameters['arguments'])
        else:
            pdf = distribution.pdf(x, loc=parameters['location'], scale=parameters['scale'])
        return pdf

    def plot_normalcy(self):
        qqplot(self.series, line='s')

    def check_normalcy(self):

        def significance_test(value, threshold):
            return "Data set {} normally distributed from".format('is' if value > threshold else 'is not')

        shapiro_stat, shapiro_p_value = shapiro(self.series)
        dagostino_stat, dagostino_p_value = normaltest(self.series)
        anderson_stat, anderson_crit_vals, anderson_significance_levels = anderson(self.series)
        anderson_report = list(zip(list(anderson_crit_vals), list(anderson_significance_levels)))
        shapiro_statement = """Shapiro-Wilk stat: {:.4f}
        Shapiro-Wilk test p-Value: {:.4f}
        {} Shapiro-Wilk Test""".format(
            shapiro_stat, shapiro_p_value, significance_test(shapiro_p_value, self.ALPHA))

        dagostino_statement = """\nD'Agostino stat: {:.4f}
        D'Agostino test p-Value: {:.4f}
        {}  D'Agostino Test""".format(
            dagostino_stat, dagostino_p_value, significance_test(dagostino_p_value, self.ALPHA))

        anderson_statement = '\nAnderson statistic: {:.4f}'.format(anderson_stat)
        for i in anderson_report:
            anderson_statement = anderson_statement + """
            For signifance level {} of Anderson-Darling test: {} the evaluation. Critical value: {}""".format(
                i[1], significance_test(i[0], anderson_stat), i[0])

        return "{}{}{}".format(shapiro_statement, dagostino_statement, anderson_statement)

    def _calculate_fit_loss(self, x, y, dist):
        with warn.catch_warnings():
            warn.filterwarnings('ignore')
            estimated_distribution = dist.fit(x)
            params = self._extract_params(estimated_distribution)
            pdf = self._build_pdf(x, dist, params)
        return np.sum(np.power(y - pdf, 2.0)), estimated_distribution

    def _generate_probability_distribution(self, distribution, parameters, bins):
        starting_point = self._generate_boundaries(distribution, parameters, self.MIN_BOUNDARY)
        ending_point = self._generate_boundaries(distribution, parameters, self.MAX_BOUNDARY)
        x = np.linspace(starting_point, ending_point, bins)
        y = self._build_pdf(x, distribution, parameters)
        return pd.Series(y, x)

    def find_distribution_fit(self):

        y_hist, x_hist_raw = np.histogram(self.series, self.histogram_bins, density=True)
        x_hist = (x_hist_raw + np.roll(x_hist_raw, -1))[:-1] / 2.
        full_distribution_results = {}

        best_loss = np.inf
        best_fit = stat.norm
        best_params = (0., 1.)
        for dist in self._get_distributions():
            histogram = getattr(stat, dist)
            results, parameters = self._calculate_fit_loss(x_hist, y_hist, histogram)
            full_distribution_results[dist] = {'hist': histogram,
                                               'loss': results,
                                               'params': {
                                                   'arguments': parameters[:-2],
                                                   'location': parameters[-2],
                                                   'scale': parameters[-1]
                                               }
                                               }
            if best_loss > results > 0:
                best_loss = results
                best_fit = histogram
                best_params = parameters
        return {'best_distribution': best_fit,
                'best_loss': best_loss,
                'best_params': {
                    'arguments': best_params[:-2],
                    'location': best_params[-2],
                    'scale': best_params[-1]
                },
                'all_results': full_distribution_results
                }

    def plot_best_fit(self):

        fits = self.find_distribution_fit()
        best_fit_distribution = fits['best_distribution']
        best_fit_parameters = fits['best_params']
        distribution_series = self._generate_probability_distribution(best_fit_distribution,
                                                                      best_fit_parameters,
                                                                      self._get_series_bins())
        with plt.style.context(style='seaborn'):
            fig = plt.figure(figsize=self.best_plot_size)
            ax = self.series.plot(kind='hist', bins=self.plot_bins, normed=True,
                                  alpha=0.5, label=self.series_name, legend=True)
            distribution_series.plot(lw=3, label=best_fit_distribution.__class__.__name__, legend=True, ax=ax)
            ax.legend(loc='best')
        return fig

    def plot_all_fits(self):

        fits = self.find_distribution_fit()
        series_bins = self._get_series_bins()

        with warn.catch_warnings():
            warn.filterwarnings('ignore')
            with plt.style.context(style='seaborn'):
                fig = plt.figure(figsize=self.all_plot_size)
                ax = self.series.plot(kind='hist', bins=self.plot_bins, normed=True, alpha=0.5,
                                      label=self.series_name, legend=True)
                y_max = ax.get_ylim()
                x_max = ax.get_xlim()
                for dist in fits['all_results']:
                    hist = fits['all_results'][dist]
                    distribution_data = self._generate_probability_distribution(hist['hist'],
                                                                                hist['params'],
                                                                                series_bins)
                    distribution_data.plot(lw=2, label=dist, alpha=0.6, ax=ax)
                ax.legend(loc='best')
                ax.set_ylim(y_max)
                ax.set_xlim(x_max)
        return fig
