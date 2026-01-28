import numpy as np
import logging

logger = logging.getLogger(__name__)

def is_solution_upside_down(transformation: np.ndarray, idx_gravity_axis: int) -> bool:
    """Check if the given transformation results in an upside-down alignment.

    This function examines the rotation component of the provided transformation
    matrix to determine if the direction corresponding to the specified gravity axis
    is inverted (i.e., points in the opposite direction). This can be useful for
    validating registration results against expected orientations.

    Args:
        transformation: A 4x4 transformation matrix to evaluate.
        idx_gravity_axis: The index of the gravity axis (0 for x, 1 for y, 2 for z).
    Returns:
        True if the solution is upside down, False otherwise.
    Raises:
        ValueError: If idx_gravity_axis is not 0, 1, or 2.
    """
    if idx_gravity_axis < 0 or idx_gravity_axis > 2:
        raise ValueError("idx_gravity_axis must be 0 (x), 1 (y), or 2 (z)")

    gravity = np.eye(3)[:, idx_gravity_axis]
    direction = transformation[:3, :3] @ gravity
    logger.info(f"Transformed direction: {direction}")
    return np.dot(direction, gravity) < 0