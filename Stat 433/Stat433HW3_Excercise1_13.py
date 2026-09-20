import numpy as np
import pandas as pd
from scipy import linalg

def get_G(MC,outdeg,df):
    P = np.zeros((len(MC),len(MC)))
    for i in range(len(MC)):
        P[i,:] = MC[i,:]/outdeg[i]
    G = df*P + (1-df)*(1/len(MC))*np.ones((len(MC),len(MC)))
    return G


def PageRank(MC,outdeg,df):
    G1 = get_G(MC,outdeg,df)
    M = (G1 - np.eye(len(G1))).T
    null_space = linalg.null_space(M)
    total = np.sum(null_space)
    pi = (null_space.flatten())/total
    return pi


 if __name__ == "__main__"

    A1 = np.array([
        [0, 0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
    ])

    states = ["a", "b", "c", "d", "e", "f", "g"]

    indeg1  = A1.sum(axis=0)   # column sums
    outdeg1 = A1.sum(axis=1) 

    print("Indegree of each node on Graph 1:", indeg1)
    print("Outdegree of each node on Graph 1:", outdeg1)


    damping_factor = 0.85

    PR1 = PageRank(A1,outdeg1,damping_factor)

    print(f"Page Rank for Graph 1: {PR1}")

    A2 = np.array([
        [0, 0, 1, 0, 0, 0, 0],  
        [1, 0, 0, 0, 0, 0, 0],  
        [0, 1, 0, 1, 0, 0, 0],   
        [1, 0, 0, 0, 0, 0, 0],   
        [0, 0, 0, 0, 0, 1, 0],   
        [0, 0, 0, 0, 1, 0, 1],   
        [0, 0, 0, 0, 1, 0, 0],   
    ])

    indeg2  = A2.sum(axis=0)   # column sums
    outdeg2 = A2.sum(axis=1) 

    print("Indegree of each node on Graph 2:", indeg2)
    print("Outdegree of each node on Graph 2:", outdeg2)

    PR2 = PageRank(A2,outdeg2,damping_factor)

    print(f"Page Rank for Graph 2:{PR2}")

    #----- Part ii -----

    A1_ii = np.maximum(A1,A1.T)
    A2_ii = np.maximum(A2, A2.T)

    degrees1 = np.sum(A1_ii, axis=0)
    degrees2 = np.sum(A2_ii, axis=0)

    PR1_ii = PageRank(A1_ii, degrees1, damping_factor)
    PR2_ii = PageRank(A2_ii, degrees2, damping_factor)

    print(f"Page Rank for Graph 1 part 2: {PR1_ii}")
    print(f"Page Rank for Graph 2 part 2: {PR2_ii}")


    #----------------------

    def show_orderings(measures, title):
        df = pd.DataFrame(measures, index=states)
        print(f"\n===== {title} =====")
        print(df.round(4))
        for col in df.columns:
            order = df[col].sort_values(ascending=False, kind="stable").index
            print(f"Order by {col}: {' > '.join(order)}")

    # Part (i): directed
    show_orderings({"indegree": indeg1, "outdegree": outdeg1, "pagerank": PR1},
                "Network 1 (directed)")
    show_orderings({"indegree": indeg2, "outdegree": outdeg2, "pagerank": PR2},
                "Network 2 (directed)")

    # Part (ii): undirected
    show_orderings({"degree": degrees1, "pagerank": PR1_ii},
                "Network 1 (undirected)")
    show_orderings({"degree": degrees2, "pagerank": PR2_ii},
                "Network 2 (undirected)")
