import numpy as np
import pandas as pd
from Stat433HW3_Excercise1_13 import PageRank


df = pd.read_csv("Wiki-Vote.txt", sep='\t', comment='#', names=['FromNodeId', 'ToNodeId'])

damping_factor = .85



nodes = np.unique(df[['FromNodeId', 'ToNodeId']].values)
idx = {node: i for i, node in enumerate(nodes)}

A = np.zeros((len(nodes),len(nodes)))

for i, j in zip(df['FromNodeId'], df['ToNodeId']):
    A[idx[i], idx[j]] = 1

outward_degree = pd.Series(np.sum(A, axis =1))
inward_degree = pd.Series(np.sum(A, axis =0))


print(f"Top 10 based on In Degrees: {inward_degree.head(10)}")
print(f"Top 10 based on Out Degrees: {outward_degree.head(10)}")

pr = PageRank(A,outward_degree, damping_factor)
pr = pd.Series(pr)
print(pr.sort_values(ascending=False).head(10))

