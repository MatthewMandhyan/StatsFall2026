import pandas as pd
import numpy as np
from scipy.stats import t
import statsmodels.api as sm
from scipy.stats import chi2
import matplotlib.pyplot as plt
import sympy as sp

df = pd.read_csv('time_series_US_20031231-1800_20260914-2040.csv')

x = np.ones(len(df)).T

b_0, b_1, b_2, b_3, b_4, b_5, t= sp.symbols('b_0 b_1 b_2 b_3 b_4 b_5 t')

bs = [b_0, b_1, b_2, b_3, b_4, b_5]
matthew  = sp.zeros(len(bs), len(bs))
print(matthew)
for l in range(len(bs)):
    print(matthew[l])
    matthew[l, l:] = sp.ones(1, len(bs) - l) * bs[l]

print(matthew.T)