import numpy as np

# Kinect v1 20-joint configuration used in UTD-MHAD
# 0: Hip Center (Root)
# 1: Spine
# 2: Shoulder Center / Neck
# 3: Head
# 4: Shoulder Left
# 5: Elbow Left
# 6: Wrist Left
# 7: Hand Left
# 8: Shoulder Right
# 9: Elbow Right
# 10: Wrist Right
# 11: Hand Right
# 12: Hip Left
# 13: Knee Left
# 14: Ankle Left
# 15: Foot Left
# 16: Hip Right
# 17: Knee Right
# 18: Ankle Right
# 19: Foot Right

UTD_JOINT_NAMES = [
    "Hip_Center",       # 0 (Root)
    "Spine",            # 1
    "Shoulder_Center",  # 2
    "Head",             # 3
    "Shoulder_Left",    # 4
    "Elbow_Left",       # 5
    "Wrist_Left",       # 6
    "Hand_Left",        # 7
    "Shoulder_Right",   # 8
    "Elbow_Right",      # 9
    "Wrist_Right",      # 10
    "Hand_Right",       # 11
    "Hip_Left",         # 12
    "Knee_Left",        # 13
    "Ankle_Left",       # 14
    "Foot_Left",        # 15
    "Hip_Right",        # 16
    "Knee_Right",       # 17
    "Ankle_Right",      # 18
    "Foot_Right"        # 19
]

# Directed kinematic edges pointing away from root (centrifugal)
UTD_EDGES = [
    (0, 1),   # Hip Center -> Spine
    (1, 2),   # Spine -> Shoulder Center
    (2, 3),   # Shoulder Center -> Head
    (2, 4),   # Shoulder Center -> Shoulder Left
    (4, 5),   # Shoulder Left -> Elbow Left
    (5, 6),   # Elbow Left -> Wrist Left
    (6, 7),   # Wrist Left -> Hand Left
    (2, 8),   # Shoulder Center -> Shoulder Right
    (8, 9),   # Shoulder Right -> Elbow Right
    (9, 10),  # Elbow Right -> Wrist Right
    (10, 11), # Wrist Right -> Hand Right
    (0, 12),  # Hip Center -> Hip Left
    (12, 13), # Hip Left -> Knee Left
    (13, 14), # Knee Left -> Ankle Left
    (14, 15), # Ankle Left -> Foot Left
    (0, 16),  # Hip Center -> Hip Right
    (16, 17), # Hip Right -> Knee Right
    (17, 18), # Knee Right -> Ankle Right
    (18, 19)  # Ankle Right -> Foot Right
]

class SkeletonGraph:
    """
    Spatial Graph representation for ST-GCN (Yan et al., AAAI 2018).
    Constructs normalized adjacency matrices according to spatial configuration partition:
      A_0: Self-connection (identity)
      A_1: Inward / Centripetal (closer to gravity center / root)
      A_2: Outward / Centrifugal (further from gravity center / root)
    """
    def __init__(self, num_nodes=20, edges=UTD_EDGES, center_joint=0, max_hop=1):
        self.num_nodes = num_nodes
        self.edges = edges
        self.center_joint = center_joint
        self.max_hop = max_hop
        self.hop_dis = self._get_hop_distance()
        self.A = self._get_spatial_graph()

    def _get_hop_distance(self):
        # Compute shortest path distances between all pairs of joints
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i, j in self.edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        
        # Floyd-Warshall shortest path
        dist = np.full((self.num_nodes, self.num_nodes), np.inf, dtype=np.float32)
        np.fill_diagonal(dist, 0.0)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adj[i, j] > 0:
                    dist[i, j] = 1.0
        
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    def _get_spatial_graph(self):
        # Distance to gravity center / root
        root_dist = self.hop_dis[self.center_joint]
        
        # 3 partitions: self, centripetal (inward), centrifugal (outward)
        A = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                pass
        
        A_self = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        A_in = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        A_out = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.hop_dis[i, j] <= self.max_hop:
                    if self.hop_dis[i, j] == 0:
                        A_self[i, j] = 1.0
                    elif root_dist[j] < root_dist[i]:
                        # j is closer to root than i -> centripetal
                        A_in[i, j] = 1.0
                    elif root_dist[j] > root_dist[i]:
                        # j is further from root than i -> centrifugal
                        A_out[i, j] = 1.0
                    else:
                        # same distance to root
                        A_self[i, j] = 1.0
        
        # Normalize each partition: D^{-1} A
        def normalize_digraph(A_part):
            Dl = np.sum(A_part, axis=0)
            num_nodes = A_part.shape[0]
            Dn = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for i in range(num_nodes):
                if Dl[i] > 0:
                    Dn[i, i] = Dl[i] ** (-1)
            AD = np.dot(A_part, Dn)
            return AD

        A_partitions = np.stack([
            normalize_digraph(A_self),
            normalize_digraph(A_in),
            normalize_digraph(A_out)
        ])
        return A_partitions # Shape: (3, num_nodes, num_nodes)
