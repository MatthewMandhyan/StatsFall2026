from Stat433HW3_Excercise1_13 import A1,states
import numpy as np
import pandas as pd
from scipy import linalg

delta = .15
damping_factor = 1 - delta

outdeg = np.sum(A1, axis = 1)


index = np.arange(len(A1))

pos = np.random.choice(index)
order=[states[pos]]
for i in range(1,21):
    pos = np.random.choice(index, p =A1[pos,:]/outdeg[pos])
    order.append(states[pos])
print(order)