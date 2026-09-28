import numpy as np

p = 0.2
q = 0.3
y0 = 0
y = [y0]
for i in range(30):
    U = np.random.rand()
    if y[i] == 0:
        if U <= p:
            y.append(1)
        else:
            y.append(0)
    if y[i] == 1:
        if  U <= q:
            y.append(0)
        else:
            y.append(1)
print(y)