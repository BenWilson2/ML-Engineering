import warnings as warn
import pandas as pd
import numpy as np
import scipy.stats as stat
from scipy.stats import shapiro, normaltest, anderson
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

### NOTE: This is for demonstration purposes ONLY. DO NOT run this
### if you want to use your computer for the next few hours.

data = pd.read_csv('/sf-airbnb-clean.csv')
series = data['price']
shapiro, pval = shapiro(series)
print('Shapiro score: ' + str(shapiro) + ' with pvalue: ' + str(pval))
dagastino, pval = normaltest(series)
print("D'Agostino score: " + str(dagastino) + " with pvalue: " + str(pval))
anderson_stat, crit, sig = anderson(series)
print("Anderson statistic: " + str(anderson_stat))
anderson_rep = list(zip(list(crit), list(sig)))
for i in anderson_rep:
    print('Significance: ' + str(i[0]) + ' Crit level: ' + str(i[1]))
bins = int(np.ceil(series.index.values.max()))
y, x = np.histogram(series, 200, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.
bl = np.inf
bf = stat.norm
bp = (0., 1.)
with warn.catch_warnings():
    warn.filterwarnings('ignore')
    fam = stat._continuous_distns._distn_names
    for d in fam:
        h = getattr(stat, d)
        f = h.fit(series)
        pdf = h.pdf(x, loc=f[-2], scale=f[-1], *f[:-2])
        loss = np.sum(np.power(y - pdf, 2.))
        if bl > loss > 0:
            bl = loss
            bf = h
            bp = f
start = bf.ppf(0.001, *bp[:-2], loc=bp[-2], scale=bp[-1])
end = bf.ppf(0.999, *bp[:-2], loc=bp[-2], scale=bp[-1])
xd = np.linspace(start, end, bins)
yd = bf.pdf(xd, loc=bp[-2], scale=bp[-1], *bp[:-2])
hdist = pd.Series(yd, xd)
with warn.catch_warnings():
    warn.filterwarnings('ignore')
    with plt.style.context(style='seaborn'):
        fig = plt.figure(figsize=(16,12))
        ax = series.plot(kind='hist', bins=100, normed=True, alpha=0.5, label='Airbnb SF Price', legend=True)
        ymax = ax.get_ylim()
        xmax = ax.get_xlim()
        hdist.plot(lw=3, label='best dist ' + bf.__class__.__name__, legend=True, ax=ax)
        ax.legend(loc='best')
        ax.set_xlim(xmax)
        ax.set_ylim(ymax)
qqplot(series, line='s')
