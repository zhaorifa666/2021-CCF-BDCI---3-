import numpy as np
from . import tools_ctrgcn

num_node = 25
self_link = [(i, i) for i in range(num_node)]

inward_ori_index = [(2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
                    (1, 8), (9, 8), (10, 9), (11, 10), (24, 11), (22, 11), (23, 22),
                    (12, 8), (13, 12), (14, 13), (21, 14), (19, 14), (20, 19),
                    (0, 1), (17, 15), (15, 0), (16, 0), (18, 16)]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

num_node_1 = 11
indices_1 = [8, 0, 6, 7, 3, 4, 13, 19, 10, 22, 1]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(1, 11), (2, 11), (3, 11), (4, 3), (5, 11), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 5
indices_2 = [3, 5, 6, 8, 10]
self_link_2 = [(i ,i) for i in range(num_node_2)]
inward_ori_index_2 = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools_ctrgcn.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools_ctrgcn.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools_ctrgcn.edge2mat(neighbor, num_node)
        self.A_norm = tools_ctrgcn.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools_ctrgcn.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = tools_ctrgcn.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools_ctrgcn.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
