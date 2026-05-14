"""
utils/math_utils.py
===================
Geometry and rotation conversion utilities.
"""
import cv2
import numpy as np

def rotation_vector_to_euler(rvec):
    """
    Converts a rotation vector to Euler angles (Yaw, Pitch, Roll) in degrees.
    """
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles from rotation matrix
    # Note: ArUco coordinate system is Z-forward, X-right, Y-down
    # We want standard Euler angles
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

def estimate_blur(image):
    """
    Estimates image blur using Laplacian variance.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()
