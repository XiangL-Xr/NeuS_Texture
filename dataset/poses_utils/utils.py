# !/usr/bin/python3
# coding: utf-8

import os, sys
import numpy as np

from typing import Optional

# Rather ugly pose generation code, derived from NeRF
def _trans_t(t):
    return np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, t],
                        [0, 0, 0, 1],
                    ],
                    dtype = np.float32)

def _rot_phi(phi):
    return np.array([
                        [1, 0, 0, 0],
                        [0, np.cos(phi), -np.sin(phi), 0],
                        [0, np.sin(phi), np.cos(phi), 0],
                        [0, 0, 0, 1],
                    ],
                    dtype = np.float32)

def _rot_theta(th):
    return np.array([
                        [np.cos(th), 0, -np.sin(th), 0],
                        [0, 1, 0, 0],
                        [np.sin(th), 0, np.cos(th), 0],
                        [0, 0, 0, 1],
                    ],
                    dtype = np.float32)

def pose_spherical(theta: float, phi: float, radius: float, offset: Optional[np.ndarray] = None,
                    vec_up: Optional[np.ndarray] = None):
    """
    Generate spherical rendering poses, from NeRF. Forgive the code horror
    :return: r (3,), t (3,)
    """
    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32) @c2w)

    if vec_up is not None:
        vec_up = vec_up / np.linalg.norm(vec_up)
        vec_1 = np.array([vec_up[0], -vec_up[2], vec_up[1]])
        vec_2 = np.cross(vec_up, vec_1)

        trans = np.eye(4, 4, dtype = np.float32)
        trans[:3, 0] = vec_1
        trans[:3, 1] = vec_2
        trans[:3, 2] = vec_up
        c2w = trans @ c2w
    
    c2w = c2w @ np.diag(np.array([1, -1, -1, 1], dtype = np.float32))
    if offset is not None:
        c2w[:3, 3] += offset
    
    return c2w
