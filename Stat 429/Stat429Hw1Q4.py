import pandas as pd
import numpy as np
from scipy.stats import t
import statsmodels.api as sm
from scipy.stats import chi2
import matplotlib.pyplot as plt
import sympy as sp

df = pd.read_csv('time_series_US_20031231-1800_20260914-2040.csv')

df['y'] = df['yahoo']

df['Time'] = pd.to_datetime(df['Time'])
df['t'] = np.arange(len(df))

#------ Question 4 Part A ------

N = len(df)
k = 5

M = np.zeros((N,k+1))

for i in range(k+1):
    M[:,i] = df['t']**i

fits = {}
for k in range(1, 6):
    X_k = M[:, :k+1]           # take columns t^0 through t^k
    model_k = sm.OLS(df['y'], X_k).fit()
    fits[k] = model_k

fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)

for k, ax in zip(range(1, 6), axes):
    model_k = fits[k]
    ax.plot(df['Time'], df['y'], color='black', alpha=0.5, label='Observed')
    ax.plot(df['Time'], model_k.fittedvalues, color='red', linewidth=2, label=f'k={k} fit')
    ax.set_title(f'Degree k = {k}')
    ax.legend()

plt.xlabel('Time')
plt.tight_layout()


#------ Question 4 Part B ------

k_chosen = 4
model = fits[k_chosen]

p = k_chosen + 1
dof = N - p

beta_hat = model.params.values
cov_beta = model.cov_params().values
s2 = model.mse_resid
t_grid = df['t'].values
plt.figure(figsize=(12, 7))
plt.plot(df['Time'], df['y'], color='black', alpha=0.6, label='Observed')

draws = 300

for i in range(draws):
    chi2_draw = np.random.chisquare(df=dof)
    sigma2_draw = dof*s2 / chi2_draw
    beta_draw = np.random.multivariate_normal(beta_hat, cov= cov_beta* (sigma2_draw / s2))

    X_grid = M[:, :k_chosen + 1]
    y_draw = X_grid @ beta_draw
    plt.plot(df['Time'], y_draw, color='blue', alpha=0.03)

plt.plot(df['Time'], model.fittedvalues, color='red', linewidth=2, label=f'LS fit (k={k_chosen})')
plt.legend()
plt.title(f'Posterior draws of polynomial trend (k={k_chosen})')

plt.show()