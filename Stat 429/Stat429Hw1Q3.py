import pandas as pd
import numpy as np
from scipy.stats import t
import statsmodels.api as sm
from scipy.stats import chi2
import matplotlib.pyplot as plt

df = pd.read_csv("MRTSSM4453USN.csv", parse_dates=["observation_date"])
df = df.rename(columns={"observation_date": "date", "MRTSSM4453USN": "y"})

#------- Question 3 Part B -------


df["log_y"] = np.log(df["y"])
df["t"] = np.arange(len(df))         
t0 = df.index[df["date"] == "2020-04-01"][0]

df["I"]  = (df["t"] >= t0).astype(int)          # I{t >= t0}
df["tI"] = df["t"] * df["I"]                    # t * I{t >= t0}

# --- Model (6): log y_t = b0 + b1 t + e_t ---
X6 = sm.add_constant(df["t"])
model6 = sm.OLS(df["log_y"], X6).fit()
df["fit6"] = model6.fittedvalues

# --- Model (7): log y_t = b0 + b1 t + b2 I + b3 tI + e_t ---
X7 = sm.add_constant(df[["t", "I", "tI"]])
model7 = sm.OLS(df["log_y"], X7).fit()
df["fit7"] = model7.fittedvalues

# 4. Plot fitted values of both models against the data
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["log_y"], label="Observed log y_t", color="black", alpha=0.6)
plt.plot(df["date"], df["fit6"], label="Model (6) fit", color="red", linewidth=2)
plt.plot(df["date"], df["fit7"], label="Model (7) fit", color="blue", linewidth=2)
plt.axvline(df["date"].iloc[t0], color="gray", linestyle="--", label="t0 = Apr 2020")
plt.xlabel("Date")
plt.ylabel("log(Sales)")
plt.legend()
plt.title("Fitted values: Model (6) vs Model (7)")


#------- Question 3 Part C -------

from scipy import stats

n = df.shape[0]
p = X7.shape[1]         
dof = n - p

beta_hat = model7.params
se = model7.bse
cov = model7.cov_params()

# --- P(beta3 < 0) ---
t3 = beta_hat["tI"] / se["tI"]
p_beta3_neg = stats.t.cdf(-t3, df=dof)

# --- P(beta2 + beta3*t0 > 0) ---
c = np.zeros(p)
c[X7.columns.get_loc("I")]  = 1
c[X7.columns.get_loc("tI")] = t0
L_hat  = c @ beta_hat.values
var_L  = c @ cov.values @ c
se_L   = np.sqrt(var_L)
p_L_pos = stats.t.cdf(L_hat / se_L, df=dof)

print(f"P(beta3 < 0 | data)          = {p_beta3_neg:.4f}")
print(f"P(beta2 + beta3*t0 > 0 | data) = {p_L_pos:.4f}")


#------- Question 3 Part D -------
k = 4

X = df[["t", "I", "tI"]].copy()

for i in range(1, k+1):
    X[f"cos{i}"] = np.cos(2*np.pi*i*df["t"]/12)
    X[f"sin{i}"] = np.sin(2*np.pi*i*df["t"]/12)

X = sm.add_constant(X)
print(X.columns.tolist())

model8 = sm.OLS(df['y'],X).fit()
print(model8.summary())

df["fit_model8"] = model8.fittedvalues

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["y"], label="Observed sales", color="black", alpha=0.6)
plt.plot(df["date"], df["fit_model8"], label="Model (8), k=4 fit", color="blue", linewidth=1.5)
plt.axvline(df["date"].iloc[t0], color="gray", linestyle="--", label="t0 = Apr 2020")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Model (8) with k=4: Fitted vs Observed")
plt.legend()


#------- Question 3 Part E -------

n_future = 36 #36 months into the future
t_future = np.arange(df['t'].iloc[-1] + 1, df['t'].iloc[-1] + 1 + n_future )

X_future = pd.DataFrame({'t': t_future})
X_future["I"] = 1
X_future['tI'] = t_future

X_future = sm.add_constant(X_future, has_constant='add')
for i in range(1, 5):
    X_future[f"cos{i}"] = np.cos(2*np.pi*i*t_future/12)
    X_future[f"sin{i}"] = np.sin(2*np.pi*i*t_future/12)

forecast = model8.predict(X_future)

future_dates = pd.date_range(start=df['date'].iloc[-1] + pd.DateOffset(months=1), periods=36, freq='MS')

plt.plot(future_dates, forecast, label="Forecast (36 months)", color="red")

plt.show()


