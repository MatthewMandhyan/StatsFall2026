from Stat433HW3_Excercise1_13 import PageRank, A1, states
import numpy as np
import pandas as pd
from scipy import linalg

delta = .15
damping_factor = 1 - delta

outdeg = np.sum(A1, axis = 1)

pr = PageRank(A1, outdeg, damping_factor)

for i in range(1,21):
    start = np.random.choice([states], p=1/7)
    next_pos = np.random.choice
    pass