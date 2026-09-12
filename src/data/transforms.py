import torch
import numpy as np
import math

class RandomRotation3D:
    """
    Applies small random rotation around the vertical (Y) axis:
    Y is the vertical/height axis in Kinect coordinates.
    """
    def __init__(self, max_angle_deg=15.0, prob=0.5):
        self.max_angle = math.radians(max_angle_deg)
        self.prob = prob

    def __call__(self, x):
        # x: (C, T, V) where C=3: (x, y, z)
        if np.random.rand() > self.prob:
            return x
        
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix around Y axis:
        # [ cos  0  sin]
        # [  0   1   0 ]
        # [-sin  0  cos]
        R = np.array([
            [cos_a,  0.0, sin_a],
            [0.0,    1.0, 0.0  ],
            [-sin_a, 0.0, cos_a]
        ], dtype=np.float32)
        
        C, T, V = x.shape
        x_flat = x.transpose(1, 2, 0).reshape(-1, 3) # (T*V, 3)
        x_rot = np.dot(x_flat, R.T)
        x_rot = x_rot.reshape(T, V, 3).transpose(2, 0, 1) # (3, T, V)
        return x_rot.astype(np.float32)

class RandomJitter:
    """Adds small Gaussian noise to joint coordinates."""
    def __init__(self, sigma=0.005, prob=0.5):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, x):
        if np.random.rand() > self.prob:
            return x
        noise = np.random.normal(0, self.sigma, size=x.shape).astype(np.float32)
        return x + noise

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
