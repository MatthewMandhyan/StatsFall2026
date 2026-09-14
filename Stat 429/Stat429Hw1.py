import pandas as pd
import numpy as np
from scipy.stats import t
import statsmodels.api as sm
from scipy.stats import chi2
import matplotlib.pyplot as plt


#------- Question 1 Part G -------

df = pd.read_csv('CPIAUCSL.csv')

y = np.log(df['CPIAUCSL']).diff().dropna()

#We want to calculate uncertainty interval
#[ybar - (s/sqrt(n)*t_{n-1, alpha/2}), ybar + (s/sqrt(n)*t_{n-1, alpha/2})]
#we will also assume a 95% confidence level

n = len(y)

ybar = np.sum(y)/n

#remember s^2 = 1/(n-1) * sum(yi - ybar)**2 
ydif = (y - ybar)
sqy = [i**2 for i in ydif]
s = np.sqrt(
    1/(n - 1) * (np.sum(sqy))
)

t_val = t.ppf(.975, df=n-1)

value = s/np.sqrt(n)*t_val

print(f"Interval: [{ybar - value}, {ybar + value}]")

#------- Question 1 Part I -------
#Now we want to compare with this model:
#logCt = B0 + B1*t + epsilon_t
C = np.log(df['CPIAUCSL']).dropna()
X = sm.add_constant(np.arange(len(C)))
model = sm.OLS(C, X).fit()
print(model.conf_int())


#------- Question 2 Part A -------
df1 = pd.read_csv('CAPOP.csv')
held_out = df1['CAPOP'].iloc[-1]
df1 = df1.iloc[:-1]
Y = df1['CAPOP']
X2 = sm.add_constant(np.arange(len(Y)))
model2 = sm.OLS(Y, X2).fit()
print(model2.conf_int())
print(model2.params)

#------- Question 2 Part B -------
N = len(df1)
p = 2
dof = N - p
sigma2_hat = model2.mse_resid
samples = chi2.rvs(df=dof, size=100)
sigma2_sample = dof*sigma2_hat / samples
XtX_inv = model2.normalized_cov_params

beta_samples = np.array([
    np.random.multivariate_normal(mean=model2.params, cov= sigma2_sample[i] * XtX_inv) for i in range(100)])

t_range = np.linspace(0, N-1, 100)   

plt.scatter(np.arange(len(Y)), Y, color='black', s=10, label='Observed data', zorder=3)

for k in range(100):
    b0, b1 = beta_samples[k]
    plt.plot(t_range, b0 + b1*t_range, color='steelblue', alpha=0.1)

plt.xlabel('Time index (t)')
plt.ylabel('CA Population (thousands)')
plt.title('100 posterior draws of the regression line')


#------- Question 2 Part C -------
t_new = N

y_new_samples = np.array([
    beta_samples[i][0] + beta_samples[i][1]*t_new + np.random.normal(0, np.sqrt(sigma2_sample[i]))
    for i in range(100)
])
point_estimate = np.mean(y_new_samples)
lower = np.percentile(y_new_samples, 2.5)
upper = np.percentile(y_new_samples, 97.5)
print(f"Point Estimate: {point_estimate}, Uncertainty Interval: [{lower}, {upper}]")


#------- Question 2 Part D -------

#logyt = B0 + b1t + eplsilon 

logyt = np.log(df1['CAPOP']).dropna()
x3 = sm.add_constant(np.arange(len(logyt)))
model3 = sm.OLS(logyt, x3).fit()
print(model3.conf_int())
print(model3.params)


#------- Question 2 Part E -------

#we know follows t dist
#we solved for t and got T < (B1hat - .03)/SE(B1hat)
B1hat = model3.params['x1']
se_beta1 = model3.bse['x1']
dof2 = N - 2
t_stat = (B1hat - .03)/se_beta1

prob = t.cdf(t_stat, df=dof2)
print(f"P(beta_1 > 0.03 | data) = {prob}")

#------- Question 2 Part F -------

#need to find sigma for epsilon dist
sigma2_hat_4 = model3.mse_resid
dof_4 = len(logyt) - 2
W_4 = chi2.rvs(df=dof_4, size=100)
sigma2_sample_4 = dof_4 * sigma2_hat_4 / W_4

xtx_inv_4 = model3.normalized_cov_params

beta_samples_4 = np.array([
    np.random.multivariate_normal(mean = model3.params, cov = sigma2_sample_4[i] * xtx_inv_4) for i in range(100) 
])

logy_new_samples = np.array([
    beta_samples_4[i][0] + beta_samples_4[i][1]*t_new + np.random.normal(0, np.sqrt(sigma2_sample_4[i])) for i in range(100)
])

y_new_samples_4 = np.exp(logy_new_samples)

point_estimate_4 = np.mean(y_new_samples_4)
lower_4 = np.percentile(y_new_samples_4, 2.5)
upper_4 = np.percentile(y_new_samples_4, 97.5)

print(f"Point Estimate: {point_estimate_4}, Uncertainty Interval: [{lower_4}, {upper_4}]")

#------- Question 2 Part G -------

X_5 = sm.add_constant(np.column_stack((
    np.arange(len(df1)),
    np.arange(len(df1))**2
)))

model5 = sm.OLS(logyt, X_5).fit()
sigma2_hat5 = model5.mse_resid
samples = chi2.rvs(df=dof, size=100)
sigma2_sample5 = dof*sigma2_hat5 / samples
XtX_inv5 = model5.normalized_cov_params


t_new = N
t_new_squared = N**2

beta_samples_5 = np.array([
    np.random.multivariate_normal(mean= model5.params, cov= XtX_inv5 * sigma2_sample5[i]) for i in range(100) 
])

logy_new_samples5 = np.array([
    beta_samples_5[i][0] + beta_samples_5[i][1]*t_new + beta_samples_5[i][2]*t_new_squared + np.random.normal(0, np.sqrt(sigma2_sample5[i]))
    for i in range(100)
])
y_new_samples5 = np.exp(logy_new_samples5)
point_estimate5 = np.mean(y_new_samples5)
lower5 = np.percentile(y_new_samples5, 2.5)
upper5 = np.percentile(y_new_samples5, 97.5)

print(f"Point Estimate: {point_estimate5}, Uncertainty Interval: [{lower5}.{upper5}")

#------- Question 2 Part H -------
diff_3 = abs(point_estimate - held_out)
diff_4 = abs(point_estimate_4 - held_out)
diff_5 = abs(point_estimate5 - held_out)

print(f"Actual held-out value: {held_out}")
print(f"Model 3 point estimate: {point_estimate}, difference: {diff_3}")
print(f"Model 4 point estimate: {point_estimate_4}, difference: {diff_4}")
print(f"Model 5 point estimate: {point_estimate5}, difference: {diff_5}")


plt.show()