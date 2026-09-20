import numpy as np
import pandas as pd
from Stat433HW3_Excercise1_13 import PageRank


df = pd.read_csv("Wiki-Vote.txt", sep='\t', comment='#', names=['FromNodeId', 'ToNodeId'])

# N = len(df)

# outward_degree = df['FromNodeId'].value_counts(ascending=True)
# inward_degree = df['ToNodeId'].value_counts(ascending=True)

# print(f"Top 10 based on In Degrees: {inward_degree}")
# print(f"Top 10 based on Out Degrees: {outward_degree}")

# df = .85

# A = np.zeros((N,N))

print("MIN" ,df['FromNodeId'].min())