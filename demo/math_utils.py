import numpy as np

def rotate_vector_3d(vector, theta, axis="x"):
    """Rotate a 3D vector by theta degrees counterclockwise around an axis."""

    theta_rad = np.radians(theta)
    cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)

    if axis.lower() == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ])
    elif axis.lower() == 'y':
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
    elif axis.lower() == 'z':
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return np.dot(rotation_matrix, vector)