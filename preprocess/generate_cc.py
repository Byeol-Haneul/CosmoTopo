'''
Author: Jun-Young Lee

Summary:
This module processes cosmological simulation data to construct combinatorial complexes, 
and capture multi-scale geometric & topological structures using the TopoNetX framework. 

# ---- FEATURES ---- #
# NODES:        4 (Mstar, Rstar, Metal, Vmax). Node Features are NOT USED in the study.
# EDGES:        3 (distance, angle1, angle2)
# TETRA:        5 (volume, 4 areas)
# CLUSTERS:     7 (num_galaxies, e1, e2, e3, gyradius, angle1, angle2)
# HYPEREDGES: 3 (distance, angle1, angle2)
# ------------------ #

Notes:
- Only kdtree_edges are used. Addition of tetra_edges show degraded performance. 
- # of cells are governed by the configurations in preprocess/config_preprocess.py

References
----------
.. [TopoModelX] https://github.com/pyt-team/TopoModelX/blob/main/topomodelx/nn/combinatorial/
.. [CosmoGraphNet] https://github.com/PabloVD/CosmoGraphNet
'''

from mpi4py import MPI
import numpy as np
import h5py
import random
from itertools import product, combinations
import sys
import os 
import torch

from scipy.spatial import Delaunay, distance, KDTree
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler

import networkx as nx
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.readwrite.serialization import to_pickle
from toponetx.readwrite.serialization import load_from_pickle

from invariants import Invariants, cell_invariants_torch, cross_cell_invariants
from neighbors import get_neighbors
from config_preprocess import *
from config.machine import *

def load_catalog(directory, filename):
    '''
    Modified from CosmoGraphNet
    arXiv:2204.13713
    https://github.com/PabloVD/CosmoGraphNet/
    '''
    if TYPE == "Quijote":
        pos = np.loadtxt(directory + filename)/BOXSIZE
    else:
        f = h5py.File(directory + filename, 'r')
        pos   = f['/Subhalo/SubhaloPos'][:]/BOXSIZE
        Mstar = f['/Subhalo/SubhaloMassType'][:,4] * MASS_UNIT
        Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]
        Nstar = f['/Subhalo/SubhaloLenType'][:,4] 
        Metal = f["Subhalo/SubhaloStarMetallicity"][:]
        Vmax = f["Subhalo/SubhaloVmax"][:]
        f.close()
    
    # Some simulations are slightly outside the box, correct it
    pos[np.where(pos<0.0)]+=1.0
    pos[np.where(pos>1.0)]-=1.0

    if TYPE == "Quijote":
        feat = np.zeros(pos.shape)
    else:
        indexes = np.where(Nstar>Nstar_th)[0]
        #indexes = np.where(Mstar>MASS_CUT)[0]
        pos     = pos[indexes]
        Mstar   = Mstar[indexes]
        Rstar   = Rstar[indexes]
        Metal   = Metal[indexes]
        Vmax    = Vmax[indexes]

        #Normalization
        Mstar = np.log10(1.+ Mstar)
        Rstar = np.log10(1.+ Rstar)
        Metal = np.log10(1.+ Metal)
        Vmax  = np.log10(1.+ Vmax)

        feat = np.vstack((Mstar, Rstar, Metal, Vmax)).T

    return pos, feat

class AbstractCells:
    def __init__(self, nodes, pos):
        self.nodes = set(nodes)  
        self.node_position = self.get_corrected_pos(np.array([pos[node] for node in self.nodes]))
        self.centroid = self.calculate_centroid()

    def get_corrected_pos(self, pos):
        pos -= global_centroid
        if pos.ndim == 1:
            pos = pos.reshape(1, 3)
        
        for i, pos_i in enumerate(pos):
            for j, coord in enumerate(pos_i):
                if coord > 0.5:
                    pos[i, j] -= 1.  
                elif -coord > 0.5:
                    pos[i, j] += 1.
        
        return pos

    def calculate_centroid(self):
        if len(list(self.nodes)) == 1:
            return self.node_position[0]
        else:
            return np.mean(self.node_position, axis=0)


class Node(AbstractCells):  
    def __init__(self, node, pos, feat):
        super().__init__([node], pos)  
        self.node = node 
        self.features = feat[node]

    def __repr__(self):
        return f"Node(node={self.node}, pos={self.node_position}, features={self.features})"

    def __hash__(self):
        return hash(self.node)

    def __eq__(self, other):
        return self.node == other.node


class Edge(AbstractCells):  
    def __init__(self, nodes, pos=None):
        super().__init__(nodes, pos)  
        self.nodes = tuple(sorted(nodes))  
        self.distance = None
        self.angles = [None, None]
        self.features = []
        self.add_feature()

    def add_feature(self):
        self.calculate_distance()
        self.calculate_angles()
        self.features.insert(0, self.distance)
        self.features.extend(self.angles)

    def calculate_distance(self):
        diff = self.node_position[0] - self.node_position[1]
        for i, coord in enumerate(diff):
            if coord > r_link:
                diff[i] -= 1.0
            elif coord < -r_link:
                diff[i] += 1.0
        self.distance = np.linalg.norm(diff)
        self.distance = normalize(self.distance, "ISDISTANCE")

    def calculate_angles(self):
        row, col = self.node_position[0], self.node_position[1]
        diff = row - col

        # Normalizing
        unitrow = row / np.linalg.norm(row)
        unitcol = col / np.linalg.norm(col)
        unitdiff = diff / np.linalg.norm(diff)

        # Dot products between unit vectors
        cos1 = np.dot(unitrow.T, unitcol)
        #cos1 = 1 - cos1 if cos1 > 0 else cos1 + 1  # Values close to 1.
        cos2 = np.dot(unitrow.T, unitdiff)
        self.angles = [cos1, cos2]

    def __repr__(self):
        return f"Edge(nodes={self.nodes}, pos={self.node_position}, features={self.features})"

    def __hash__(self):
        return hash(self.nodes)

    def __eq__(self, other):
        return self.nodes == other.nodes


class Tetrahedron(AbstractCells):  
    def __init__(self, index, nodes, pos):
        super().__init__(nodes, pos)  
        self.index = index
        self.volume = None
        self.areas = None
        self.features = []
        self.add_features()

    def calculate_volume(self):
        mat = np.zeros([4, 4])
        mat[:, :3] = self.node_position
        mat[:, 3] = 1.0
        self.volume = normalize(np.abs(np.linalg.det(mat)) / 6, "ISVOLUME")

    def add_features(self):
        self.calculate_volume()
        self.calculate_areas()
        self.features.append(self.volume)
        self.features.extend(self.areas)

    def triangle_area(self, a, b, c):
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        return normalize(area, "ISAREA")

    def calculate_areas(self):
        a, b, c, d = self.node_position
        self.areas = [
            self.triangle_area(a, b, c),
            self.triangle_area(a, b, d),
            self.triangle_area(a, c, d),
            self.triangle_area(b, c, d)
        ]

    def __repr__(self):
        return f"Tetrahedron(index={self.index}, nodes={self.nodes}, features={self.features})"


class Cluster(AbstractCells):  
    def __init__(self, label, tetrahedra, pos):
        self.label = label
        self.tetrahedra = tetrahedra

        nodes = self.get_nodes()
        super().__init__(nodes, pos)  

        self.eigenvalues = None
        self.gyradius = None
        self.covariance_matrix = None
        self.volumes = []
        self.angles = []
        self.features = []

        self.add_features()

    def get_nodes(self):
        nodes = set()
        for tetra in self.tetrahedra:
            nodes.update(tetra.nodes)
        return nodes

    def calculate_covariance_matrix(self):
        if self.node_position is not None and self.centroid is not None:
            centered_positions = self.node_position - self.centroid
            self.covariance_matrix = np.cov(centered_positions, rowvar=False)

    def calculate_eigenvalues(self):
        if self.covariance_matrix is not None:
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
            self.eigenvalues = normalize(eigenvalues, "ISAREA")
            self.eigenvectors = normalize(eigenvectors, "ISDISTANCE")

    def calculate_gyradius(self):
        if self.node_position is not None and self.centroid is not None:
            centered_positions = self.node_position - self.centroid
            squared_distances = np.sum(centered_positions ** 2, axis=1)
            self.gyradius = normalize(np.sqrt(np.mean(squared_distances)), "ISDISTANCE")

    def calculate_angles(self):
        angles = []
        if self.node_position is not None and self.eigenvectors is not None:
            vector = self.centroid
            unit_vector = vector / np.linalg.norm(vector)
            for eigenvector in self.eigenvectors.T:
                unit_eigenvector = eigenvector / np.linalg.norm(eigenvector)
                cos_angle = np.dot(unit_vector, unit_eigenvector)
                angles.append(cos_angle)

        self.angles = angles[:2]  # only use two axes.

    def add_features(self):
        self.calculate_covariance_matrix()
        self.calculate_eigenvalues()
        self.calculate_gyradius()
        self.calculate_angles()

        # Add features
        self.features.append(np.log10(len(list(self.nodes)) + 1))
        self.features.extend(self.eigenvalues)
        self.features.append(self.gyradius)
        self.features.extend(self.angles)

    def __repr__(self):
        return f"Cluster(label={self.label}, features={self.features})"

class Hyperedge(AbstractCells):  
    def __init__(self, cluster1, cluster2, dist, pos):
        # Combine nodes and positions from both clusters
        nodes = cluster1.nodes | cluster2.nodes
        super().__init__(nodes, pos)  

        self.cluster1 = cluster1
        self.cluster2 = cluster2
        self.cluster1_label = cluster1.label
        self.cluster2_label = cluster2.label

        self.distance = normalize(dist, "ISDISTANCE")
        self.num_nodes_cluster1 = np.log10(len(cluster1.nodes) + 1)
        self.num_nodes_cluster2 = np.log10(len(cluster2.nodes) + 1)

        self.angles = self.calculate_angles()
        self.features = [self.distance] + self.angles

    def calculate_angles(self):
        row, col = self.cluster1.centroid, self.cluster2.centroid
        diff = row - col

        # Normalizing
        unitrow = row / np.linalg.norm(row)
        unitcol = col / np.linalg.norm(col)
        unitdiff = diff / np.linalg.norm(diff)

        # Dot products between unit vectors
        cos1 = np.dot(unitrow.T, unitcol)
        cos2 = np.dot(unitrow.T, unitdiff)
        angles = [cos1, cos2]
        return angles

    def __repr__(self):
        return f"Hyperedge(cluster1={self.cluster1_label}, cluster2={self.cluster2_label}, features={self.features})"


def create_mst(clusters):
    centroids = np.array([cluster.centroid for cluster in clusters.values()])
    labels = list(clusters.keys())
    dist_matrix = distance.cdist(centroids, centroids, 'euclidean')

    G = nx.Graph()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            G.add_edge(labels[i], labels[j], weight=dist_matrix[i, j])

    mst = nx.minimum_spanning_tree(G)
    return mst


def get_tetrahedra(pos, feat):
    # Generate Delaunay triangulation
    tri = Delaunay(pos)

    tetrahedra = []
    for i, simplex in enumerate(tri.simplices):
        tetra = Tetrahedron(i, simplex, pos)
        tetrahedra.append(tetra)

    tetrahedra.sort(key=lambda x: x.volume)  # Sorting by volume
    print("[LOG] GENERATED TETRA", file=sys.stderr)
    return tetrahedra

def get_single_tetra_edges(pos, tetra):
    edge_objects = []
    for (i, j) in combinations(range(4), 2):
        edge = tuple(sorted((tetra.nodes[i], tetra.nodes[j])))
        edge_obj = Edge(nodes=edge, pos=pos)
        edge_objects.append(edge_obj)
    return edge_objects

def get_all_tetra_edges(pos, tetrahedra):
    tetra_edge_set = set() 
    for tetra in tetrahedra:
        tetra_edge_set.update(get_single_tetra_edges(pos, tetra))

    return tetra_edge_set

def get_kdtree_edges(pos, r_link=0.015):
    '''
    Modified from CosmoGraphNet
    arXiv:2204.13713
    https://github.com/PabloVD/CosmoGraphNet/
    '''
    kd_tree = KDTree(pos, leafsize=16, boxsize=1.0001)
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray")

    kdtree_edge_set = set()
    for src, dst in edge_index:
        edge_obj = Edge([src, dst],pos)
        kdtree_edge_set.add(edge_obj)

    #return kdtree_edge_set
    return kdtree_edge_set


def create_cc(in_dir, in_filename):    
    ##########################
    global NUMPOINTS, NUMEDGES, NUMTETRA
    
    # Read in data
    pos, feat = load_catalog(in_dir, in_filename)
    pos[np.where(pos<0.0)]+=1.0
    pos[np.where(pos>1.0)]-=1.0

    global global_centroid
    global_centroid = np.mean(pos, axis=0)

    nodes = [Node(node, pos, feat) for node in range(len(pos))]

    # 1. Get Tetrahedra
    tetrahedra = get_tetrahedra(pos, feat)
    tetrahedra = sorted(tetrahedra, key=lambda tetra: tetra.volume)[:NUMTETRA]
    
    # 2. Get edges
    tetra_edge_set = set()     #get_all_tetra_edges(pos, tetrahedra)
    kdtree_edge_set = get_kdtree_edges(pos, r_link)
    edge_set = kdtree_edge_set #| tetra_edge_set
    edges = list(edge_set)
    edges = sorted(edges, key=lambda edge: edge.distance)[:NUMEDGES]
    
    # 3. Clustering on Tetrahedra
    clusters = clustering(tetrahedra, pos)

    # 4. Get Hyperedges using MST
    hyperedges = {}
    mst = create_mst(clusters)
    for edge in mst.edges(data=True):
        cluster1_label = edge[0]
        cluster2_label = edge[1]
        dist = edge[2]['weight']

        cluster1 = clusters[cluster1_label]
        cluster2 = clusters[cluster2_label]

        hyperedge_label = f"hyper_{cluster1_label}_{cluster2_label}"
        hyperedge = Hyperedge(cluster1, cluster2, dist, pos)
        hyperedges[hyperedge_label] = hyperedge

    
    NUMPOINTS = pos.shape[0] if (NUMPOINTS == -1 or NUMPOINTS>pos.shape[0]) else NUMPOINTS
    NUMEDGES = len(edge_set) if (NUMEDGES == -1 or NUMEDGES>len(edge_set)) else NUMEDGES
    NUMTETRA = len(tetrahedra) if (NUMTETRA == -1 or NUMTETRA>len(tetrahedra)) else NUMTETRA
    
    # 5. Generate Combinatorial Complex
    print(f"""[LOG] We will select {NUMEDGES} edges and {NUMTETRA} tetra""", file=sys.stderr)
    print(f"""[LOG] Edges from tetra {len(tetra_edge_set)} and KDTree {len(kdtree_edge_set)} with {len(tetra_edge_set & kdtree_edge_set)} edges in common.""", file=sys.stderr)

    cc = CombinatorialComplex()

    ## 5-1 ADD NODES ##
    for node in nodes:
        cc.add_cell([node.node], rank=0)
    
    node_data = {node.node: node.features for node in nodes}

    ## 5-2 ADD EDGES ##
    for edge in edges:
        cc.add_cell(edge.nodes, rank=1)
    
    edge_data = {edge.nodes: edge.features for edge in edges}

    ## 5-3 ADD TETRA ##
    for tetra in tetrahedra:
        cc.add_cell(list(tetra.nodes), rank=2)

    tetra_data = {tuple(tetra.nodes): tetra.features for tetra in tetrahedra} 

    ## 5-4 ADD Clusters ##
    for cluster in clusters.values():
        cc.add_cell(list(cluster.nodes), rank=3)

    cluster_data = {tuple(cluster.nodes): cluster.features for cluster in clusters.values()}

    # 5-5 ADD HyperCluster ##
    hyperedges = remove_subset_clusters(hyperedges)
    for hyperedge in hyperedges.values():
        cc.add_cell(list(hyperedge.nodes), rank=4)

    hyperedge_data = {tuple(hyperedge.nodes): hyperedge.features for hyperedge in hyperedges.values()}

    # ADD CELL ATTRIBUTES
    cc.set_cell_attributes(node_data, name="node_feat")
    cc.set_cell_attributes(edge_data, name="edge_feat")
    cc.set_cell_attributes(tetra_data, name="tetra_feat")
    cc.set_cell_attributes(cluster_data, name="cluster_feat")
    cc.set_cell_attributes(hyperedge_data, name="hyperedge_feat")
    return cc, nodes, edges, tetrahedra, clusters, hyperedges

def remove_subset_clusters(clusters):
    to_remove = set()
    cluster_labels = list(clusters.keys())

    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels)):
            if i == j:
                continue

            A = clusters[cluster_labels[i]].nodes
            B = clusters[cluster_labels[j]].nodes

            if A & B == A or A & B == B:
                if len(A) <= len(B):
                    to_remove.add(cluster_labels[i])
                else:
                    to_remove.add(cluster_labels[j])

    for label in to_remove:
        del clusters[label]

    print(f"[LOG] Removed {len(to_remove)} subset clusters.", file=sys.stderr)
    return clusters

def clustering(tetrahedra, pos):
    global r_link
    embeddings = np.array([tetra.centroid for i, tetra in enumerate(tetrahedra)])
    db = HDBSCAN(min_samples=MINCLUSTER).fit(embeddings)
    labels = db.labels_

    clusters = {}
    for label in set(labels):
        if label == -1:
            continue
        indices = np.where(labels == label)[0]
        merged_tetra = [tetrahedra[idx] for idx in indices]

        if len(merged_tetra) == 1:
            continue
        else:
            cluster = Cluster(label, merged_tetra, pos)
            clusters[label] = cluster

    # Remove subset clusters
    clusters = remove_subset_clusters(clusters)

    print(f"""
    [LOG] We Currently have {len(embeddings)} Tetrahedra.
    [LOG] Generated {len(clusters)} Clusters of Tetrahedra. 
    [LOG] Mean number of nodes per cluster is {np.mean([len(cluster.nodes) for cluster in clusters.values()])}
    [LOG] Max number of nodes per cluster is {np.max([len(cluster.nodes) for cluster in clusters.values()])} and the number is {np.argmax([len(cluster.nodes) for cluster in clusters.values()])}""", file=sys.stderr)

    return clusters

def main(array):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    total_elements = len(array)
    jobs_per_process = total_elements // size
    extra_jobs = total_elements % size

    if rank < extra_jobs:
        start_idx = rank * (jobs_per_process + 1)
        end_idx = start_idx + jobs_per_process + 1
    else:
        start_idx = rank * jobs_per_process + extra_jobs
        end_idx = start_idx + jobs_per_process

    slice_array = array[start_idx:end_idx]
    
    for num in slice_array:
        if TYPE == "Quijote":
            in_filename = f"catalog_{num}.txt"
            out_filename = f"data_{num}.pickle"
        else:
            in_filename = f"data_{num}.hdf5"
            out_filename = f"data_{num}.pickle"

        cc, nodes, edges, tetrahedra, clusters, hyperedges = create_cc(in_dir, in_filename)

        print(f"[LOG] Process {rank}: Created combinatorial complex for file {in_filename}", file=sys.stderr)
        to_pickle(cc, cc_dir + out_filename)

        print(f"[LOG] Process {rank}: Calculating Neighbors", file=sys.stderr)
        neighbors = get_neighbors(num, cc)

        print(f"[LOG] Process {rank}: Calculating Cross-Cell-Invariants", file=sys.stderr)
        invariants = cross_cell_invariants(num, nodes, edges, tetrahedra, clusters, hyperedges, neighbors)

    comm.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    num_array = list(range(CATALOG_SIZE))
    main(num_array)

