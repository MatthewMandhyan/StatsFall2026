import numpy as np
y0 = 0
y = [y0]
P = np.array((
    [.8,.1,.1],
    [0,.1,.9],
    [.4,.4,.2]
))

for i in range(30):
    next_state = np.random.choice([0, 1, 2], p=P[y[i], :])
    y.append(next_state)

print(y)