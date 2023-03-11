import ctypes
import numpy as np
import pydclust.dclust
from enum import IntEnum
from typing import Optional, Tuple


# Define array types
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i1_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
ndmat_i2_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i3_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i4_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i5_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i6_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i7_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
# Define input/output types
pydclust.dclust.restype = ndmat_i1_type  # Nx2 (i, j) coordinates or None
pydclust.dclust.argtypes = [
    ndmat_i2_type,   # edges
    ndmat_f_type,   # weights
    ndmat_i3_type,   # lengths
    ndmat_i4_type,   # locations
    ndmat_i5_type,   # neighbors
    ndmat_i6_type,   # actives
    ctypes.c_int,   # edge_num
    ctypes.c_int,   # node_num
    ctypes.c_int,   # cluster_num
    ctypes.c_int,   # actives_num
]

class Heuristic(IntEnum):
    """The supported heuristics."""

    DEFAULT = 0
    ORTHOGONAL_X = 1
    ORTHOGONAL_Y = 2

def dclust_assign(
        weights: np.ndarray,
        edges: np.ndarray, k:int, n:int) -> Optional[np.ndarray]:
    """
    Run astar algorithm on 2d weights.

    param np.ndarray weights: A grid of weights e.g. np.ones((10, 10), dtype=np.float32)
    param Tuple[int, int] start: (i, j)
    param Tuple[int, int] goal: (i, j)
    param bool allow_diagonal: Whether to allow diagonal moves
    param Heuristic heuristic_override: Override heuristic, see Heuristic(IntEnum)

    """
    
    # Ensure start is within bounds.
    #if ((starts[:,0] < 0).any() or (starts[:,0] >= weights.shape[0]).any() or
    #        (starts[:,1] < 0).any() or (starts[:,1] >= weights.shape[1]).any()):
    #    raise ValueError(f"Start of {start} lies outside grid.")
    # Ensure goal is within bounds.
    #if (goal[0] < 0 or goal[0] >= weights.shape[0] or
    #        goal[1] < 0 or goal[1] >= weights.shape[1]):
    #    raise ValueError(f"Goal of {goal} lies outside grid.")

    #start_idx = np.ravel_multi_index(starts.T, (height, width)).astype(np.intc)
    #goal_idx = np.ravel_multi_index(goals.T, (height, width)).astype(np.intc)
    m = edges.shape[0]
    #n = edges.max() + 1
    
    E0 = edges[:,0]
    E0, EC = np.unique(E0, return_counts=True)
    lengths = np.zeros((n,), dtype=np.intc)
    lengths[E0] = EC

    locations = np.zeros((n), dtype=np.intc)
    locations[1:] = np.cumsum(lengths)[:-1]

    #neighbors = np.zeros((lengths.sum()), dtype=np.intc)
    edges = edges[edges[:, 0].argsort()]
    neighbors = np.split(edges[:,1], np.unique(edges[:, 0], return_index=True)[1][1:])

    new_neighbors = np.concatenate(neighbors,axis=0)
    
    actives = E0
    
    #print(start_idx, goal_idx)
    #print(actives[0], actives.shape[0], m,n, k)
    
    #"""
    path = pydclust.dclust.dclust(edges.flatten().astype(np.intc),
                                   weights.flatten().astype(np.float32),
                                   lengths.flatten().astype(np.intc),
                                   locations.flatten().astype(np.intc),
                                   new_neighbors.flatten().astype(np.intc),
                                   actives.flatten().astype(np.intc),
                                   m, n, k, actives.shape[0],
    )
    #"""
    return path
