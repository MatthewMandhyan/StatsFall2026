import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import librosa
from scipy.special import softmax
from scipy import stats


# ----- Question 2 Part A -----

y = pd.read_csv('lynx.csv').iloc[:,-1]
n = len(y)
I = np.abs(np.fft.fft(y - y.mean()))**2 / n
f = np.arange(n) / n

keep = slice(1,n//2 + 1)

plt.plot(f[keep], I[keep]); plt.xlabel("frequency (cycles/yr)"); plt.ylabel("power")
fpeak = f[keep][np.argmax(I[keep])]



# ----- Question 2 Part B -----

X = np.zeros((n,3))
t = np.arange(n)


def grid_posterier(freqs):
    log_post = np.zeros((len(freqs)))
    for i, fg in enumerate(freqs):
        X[:,0] = 1
        X[:,1] = np.cos(2*np.pi*fg*t)
        X[:,2] = np.sin(2*np.pi*fg*t)
        model = sm.OLS(y,X).fit()
        rss = np.sum(model.resid**2)
        sign, det = np.linalg.slogdet(X.T @ X)
        log_post[i] = -0.5*det - ((n-3)/2)*np.log(rss)
    return log_post

freqs = np.linspace(fpeak - 1/n, fpeak + 1/n, 500)
log_post = grid_posterier(freqs)
best_f = freqs[np.argmax(log_post)]

probs = softmax(log_post)
cdf = np.cumsum(probs)

lower_idx = np.searchsorted(cdf, 0.025)
upper_idx = np.searchsorted(cdf, 0.975)

f_lower, f_upper = freqs[lower_idx], freqs[upper_idx]

print(f"Point estimate of f: {best_f} cycles/year")
print(f"95% interval for f: [{f_lower}, {f_upper}]")

print(f"Point estimate of period: {1/best_f} years")
print(f"95% interval for period: [{1/f_upper}, {1/f_lower}] years")




# ----- Question 2 Part C -----

X[:,0] = 1
X[:,1] = np.cos(2*np.pi*best_f*t)
X[:,2] = np.sin(2*np.pi*best_f*t)

model2 = sm.OLS(y,X).fit()
b0hat, b1hat, b2hat = model2.params
rss2 = np.sum(model2.resid**2)
sighat = np.sqrt(rss2/(n-3))

se0,  se1, se2 = model2.bse

test_crit = stats.t.ppf(.975,n-3)

print(f"Uncertainity Interval for B0: [{b0hat - test_crit*se0},{b0hat + test_crit*se0}]")
print(f"Uncertainity Interval for B1: [{b1hat - test_crit*se1},{b1hat + test_crit*se1}]")
print(f"Uncertainity Interval for B2: [{b2hat - test_crit*se2},{b2hat + test_crit*se2}]")

q1 = stats.chi2.ppf(.025,n-3)
q2 = stats.chi2.ppf(.975,n-3)

print(f"Uncertainity Interval for Sigma: [{np.sqrt(rss2/q2)},{np.sqrt(rss2/q1)}]")

resid = model2.resid

plt.figure(figsize=(10, 5))
plt.plot(t, y, label='Observed')
plt.plot(t, model2.fittedvalues, label='Fitted sinusoid')
plt.xlabel('Years since 1821')
plt.ylabel('Lynx trappings')
plt.title('Data vs. fitted model')
plt.legend()


plt.show()