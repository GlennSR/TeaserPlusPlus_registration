"""Transformation and rotation utilities."""

from typing import Tuple
import numpy as np
import numpy.typing as npt

def gravity_transformation(
    gravity_direction: np.ndarray, gravity_axis: int = 1
) -> np.ndarray:
    """Compute a transformation matrix to align a given gravity direction.

    This function computes a rotation matrix that aligns the specified gravity
    direction with the desired gravity axis (default is y-axis). This can be useful if the
    gravity vector is given by an IMU sensor in a reference system similar (same up direction)
    to the one of the point cloud.
    It returns a 4x4 transformation matrix that can be applied to point clouds so that the
    point cloud is aligned with the gravity direction.

    Args:
        gravity_direction: A 3D vector representing the measured gravity direction in the point cloud's reference frame.
        gravity_axis: The axis index (0 for x, 1 for y, 2 for z) to align the gravity direction to.

    Returns:
        A 4x4 transformation matrix with null translation that aligns the gravity direction with the specified axis.
    """
    if gravity_axis < 0 or gravity_axis > 2:
        raise ValueError("gravity_axis must be 0 (x), 1 (y), or 2 (z)")

    # @TODO this is to add some noise to the gravity direction
    dst_gravity_direction = perturb_direction(
        np.eye(3)[:, gravity_axis], sigma=np.deg2rad(1)
    )
    dst_gravity_direction = np.eye(3)[:, gravity_axis]

    gravity_aligned_rotation = rotation_aligning_two_directions(
        gravity_direction, dst_gravity_direction
    )
    gravity_transform = rototranslation_from_rotation_translation(
        gravity_aligned_rotation, np.zeros(3)
    )
    return gravity_transform

def rototranslation_from_rotation_translation(
    rot: np.ndarray, trans: np.ndarray
) -> np.ndarray:
    """Combine rotation matrix and translation vector into a 4x4 transformation matrix.

    Args:
        rot: A 3x3 rotation matrix.
        trans: A 3D translation vector. (shape (3,))

    Returns:
        A 4x4 transformation matrix combining rotation and translation.

    Raises:
        ValueError: If the rotation matrix is not 3x3 or the translation vector is not 3D.
    """
    if rot.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if trans.shape != (3,):
        raise ValueError("Translation vector must be a 3D vector.")

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def axis_angle_from_rotation(rot_mat: np.ndarray) -> Tuple[np.ndarray, float]:
    r"""Convert a rotation matrix to axis-angle representation.

    Extracts the rotation axis and angle from a 3x3 rotation matrix using
    the Rodrigues formula. Handles special cases including identity rotation
    (angle ≈ 0) and 180° rotation.

    Args:
        rot_mat: A 3x3 rotation matrix (proper orthogonal matrix with det(R) = +1).

    Returns:
        A tuple containing:
            - axis: A (3,) unit vector representing the rotation axis.
                   Undefined/arbitrary if rotation is approximately identity.
            - angle: Rotation angle in radians, in the range [0, π].

    Note:
        The rotation angle is extracted using:

        .. math::

            \theta = \arccos\left(\frac{\text{trace}(R) - 1}{2}\right)

        For small rotations (:math:`\theta \approx 0`), the axis is set to [1, 0, 0]
        by convention.

        For 180° rotations (:math:`\theta = \pi`), the axis is extracted from:

        .. math::

            k_i = \sqrt{\frac{R_{ii} + 1}{2}}

        where :math:`i \in \{x, y, z\}` and :math:`R_{ii}` are the diagonal elements.
    """
    eps = 1e-12
    angle = np.arccos(np.clip((np.trace(rot_mat) - 1) / 2.0, -1.0, 1.0))

    if np.isclose(angle, 0.0, atol=1e-8):
        # No rotation → arbitrary axis
        return np.array([1.0, 0.0, 0.0]), 0.0

    if np.isclose(angle, np.pi, atol=1e-6):
        # 180° rotation → extract from diagonal elements
        axis = np.sqrt(np.maximum(np.diagonal(rot_mat) + 1.0, 0.0)) / np.sqrt(2.0)
        axis = axis / np.linalg.norm(axis + eps)
        return axis, angle

    axis = np.array(
        [
            rot_mat[2, 1] - rot_mat[1, 2],
            rot_mat[0, 2] - rot_mat[2, 0],
            rot_mat[1, 0] - rot_mat[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    axis = axis / np.linalg.norm(axis + eps)
    return axis, angle


def cross_matrix(v: np.ndarray) -> np.ndarray:
    r"""Create a cross-product matrix from a 3D vector.

    Args:
        v: A 3D vector (shape (3,)).

    Returns:
        A 3x3 skew-symmetric matrix :math:`[v]_\times` such that
        :math:`[v]_\times w = v \times w` for any vector :math:`w`.

    Raises:
        ValueError: If the input vector is not 3D.

    Note:
        The cross-product matrix is defined as:

        .. math::

            [v]_\times = \begin{bmatrix}
                0 & -v_z & v_y \\
                v_z & 0 & -v_x \\
                -v_y & v_x & 0
            \end{bmatrix}

        This matrix satisfies :math:`[v]_\times^T = -[v]_\times` (skew-symmetry).
    """
    if v.shape != (3,):
        raise ValueError("Input vector must be a 3D vector.")

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    r"""Generate a rotation matrix from an axis-angle representation.

    Uses Rodrigues' rotation formula to compute the rotation matrix.
    The input axis is automatically normalized to a unit vector.

    Args:
        axis: A 3D vector representing the rotation axis (will be normalized).
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix corresponding to the given axis and angle.

    Raises:
        ValueError: If the input axis is not a 3D vector.

    Note:
        The rotation matrix is computed using Rodrigues' formula:

        .. math::

            R = \cos(\theta) I + \sin(\theta) [k]_\times + (1 - \cos(\theta)) k k^T

        where :math:`\theta` is the angle, :math:`k` is the unit rotation axis,
        :math:`I` is the 3x3 identity matrix, :math:`[k]_\times` is the
        skew-symmetric cross-product matrix, and :math:`k k^T` is the outer
        product of the axis with itself.

        The skew-symmetric matrix :math:`[k]_\times` is defined as:

        .. math::

            [k]_\times = \begin{bmatrix}
                0 & -k_z & k_y \\
                k_z & 0 & -k_x \\
                -k_y & k_x & 0
            \end{bmatrix}
    """
    if axis.shape != (3,):
        raise ValueError("Input vector must be a 3D vector.")

    axis = axis / np.linalg.norm(axis)
    rot = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross_matrix(axis)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )
    return rot


def rotation_error_angle(rot_est: np.ndarray, rot_gt: np.ndarray) -> float:
    r"""Calculate the angular error between two rotation matrices.

    Computes the angle (in radians) of the relative rotation between an
    estimated rotation matrix and a ground truth rotation matrix. This is
    equivalent to finding the rotation angle needed to transform R_est to R_gt.

    Args:
        rot_est: Estimated 3x3 rotation matrix.
        rot_gt: Ground truth 3x3 rotation matrix.

    Returns:
        The rotation error angle in radians, in the range [0, π].

    Note:
        The error is computed using the formula:

        .. math::

            \theta = \arccos\left(\frac{\text{trace}(R_{\text{est}} R_{\text{gt}}^T) - 1}{2}\right)

        which gives the geodesic distance on SO(3).
    """
    rot_err = rot_est @ rot_gt.T
    trace = np.clip((np.trace(rot_err) - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)  # radians
    return angle


def translation_error(
    rot_est: np.ndarray, t_est: np.ndarray, rot_gt: np.ndarray, t_gt: np.ndarray
) -> Tuple[float, npt.NDArray[np.floating]]:
    r"""Calculate the translation error between two transformations.

    Computes the translation error accounting for the rotation difference.

    Args:
        rot_est: Estimated 3x3 rotation matrix.
        t_est: Estimated 3D translation vector.
        rot_gt: Ground truth 3x3 rotation matrix.
        t_gt: Ground truth 3D translation vector.

    Returns:
        A tuple containing:
            - norm: The Euclidean norm (magnitude) of the translation error.
            - vector: The 3D translation error vector.

    Note:
        The translation error is computed as:

        .. math::

            t_{\text{err}} = t_{\text{est}} - R_{\text{err}} \, t_{\text{gt}}

        where :math:`R_{\text{err}} = R_{\text{est}} R_{\text{gt}}^T` is the relative rotation.
        This ensures the error is measured in the same reference frame.
    """
    rot_err = rot_est @ rot_gt.T
    t_err = t_est - rot_err @ t_gt
    norm = float(np.linalg.norm(t_err))
    return norm, t_err  # (norm, vector)


def transformation_error(t_est: np.ndarray, t_gt: np.ndarray) -> Tuple[float, float]:
    r"""Calculate both rotation and translation errors between two transformations.

    Decomposes two 4x4 transformation matrices into rotation and translation
    components, then computes the angular error between rotations and the
    translation error magnitude.

    Args:
        t_est: Estimated 4x4 transformation matrix (homogeneous coordinates).
        t_gt: Ground truth 4x4 transformation matrix (homogeneous coordinates).

    Returns:
        A tuple containing:
            - rot_err: Rotation error angle in radians, in the range [0, π].
            - trans_err: Translation error magnitude (Euclidean norm).

    Raises:
        ValueError: If either T_est or T_gt is not a 4x4 matrix.

    Note:
        The transformation matrices should be in the standard form:

        .. math::

            T = \begin{bmatrix}
                R & t \\
                0 & 1
            \end{bmatrix}

        where :math:`R` is a 3×3 rotation matrix and :math:`t` is a 3D translation vector.
    """
    # check the matrices are 4x4
    if t_est.shape != (4, 4) or t_gt.shape != (4, 4):
        raise ValueError("Both T_est and T_gt must be 4x4 matrices.")

    rot_est = t_est[:3, :3]
    tra_est = t_est[:3, 3]
    rot_gt = t_gt[:3, :3]
    tra_gt = t_gt[:3, 3]
    rot_err = rotation_error_angle(rot_est, rot_gt)
    trans_err, trans_vec = translation_error(rot_est, tra_est, rot_gt, tra_gt)
    return rot_err, trans_err


def generate_random_rotation_matrix() -> np.ndarray:
    """Generate a random 3x3 rotation matrix.

    Uses QR method to generate a random rotation matrix uniformly sampled from SO(3).

    Returns:
        A random 3x3 rotation matrix.
    """
    random_state = np.random.default_rng()
    A = random_state.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)

    # Ensure a proper rotation (det(Q) = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1

    return Q


def is_rotation_matrix(mat: np.ndarray) -> bool:
    """Check if a matrix is a valid rotation matrix.

    A valid rotation matrix is orthogonal (R.T @ R = I) and has a determinant of +1.

    Args:
        mat: A square NxN matrix to check.

    Returns:
        True if the matrix is a valid rotation matrix, False otherwise.
    """

    # Check orthogonality
    if not np.allclose(mat.T @ mat, np.eye(3)):
        return False
    # Check determinant
    if not np.isclose(np.linalg.det(mat), 1):
        return False
    return True


def rotation_aligning_two_directions(
    src_dir: np.ndarray, tgt_dir: np.ndarray
) -> np.ndarray:
    """Find the rotation matrix that aligns two directions.

    Args:
        src_dir: Source direction vector (3,).
        tgt_dir: Target direction vector (3,).

    Returns:
        A 3x3 rotation matrix that aligns src_dir to tgt_dir, i.e., R @ src_dir = tgt_dir.
    """
    # Validate input dimensions
    if src_dir.shape != (3,) or tgt_dir.shape != (3,):
        raise ValueError("Both source and target directions must be 3D vectors.")

    # Normalize once at the beginning
    src_normalized = src_dir / np.linalg.norm(src_dir)
    tgt_normalized = tgt_dir / np.linalg.norm(tgt_dir)

    # Check if vectors are already aligned (dot product ≈ 1)
    dot_product = np.dot(src_normalized, tgt_normalized)

    if np.isclose(dot_product, 1.0, atol=1e-10):
        return np.eye(3)

    # Check if vectors are opposite (dot product ≈ -1)
    if np.isclose(dot_product, -1.0, atol=1e-10):
        # Find an arbitrary orthogonal axis for 180° rotation
        if not np.isclose(src_normalized[0], 0) or not np.isclose(src_normalized[1], 0):
            ortho_axis = np.array([-src_normalized[1], src_normalized[0], 0.0])
        else:
            ortho_axis = np.array([0.0, -src_normalized[2], src_normalized[1]])
        ortho_axis /= np.linalg.norm(ortho_axis)
        return rotation_matrix_from_axis_angle(ortho_axis, np.pi)

    # General case: compute rotation axis and angle
    v = np.cross(src_normalized, tgt_normalized)
    v = v / np.linalg.norm(v)
    # Clip to avoid numerical errors in arccos
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return rotation_matrix_from_axis_angle(v, angle)


def perturb_direction(direction: np.ndarray, sigma: float) -> np.ndarray:
    """Perturb a direction vector by rotating by a small random rotation

    Args:
        direction: A 3D unit vector representing the original direction.
        sigma: Standard deviation of the Gaussian noise to be added.

    Returns:
        A new 3D unit vector representing the perturbed direction.

    Raises:
        ValueError: If the input direction is not a 3D vector.
    """
    if direction.shape != (3,):
        raise ValueError("Input direction must be a 3D vector.")

    # Generate a random axis orthogonal to the original direction
    random_axis = np.random.normal(size=3)
    random_axis -= random_axis.dot(direction) * direction
    random_axis /= np.linalg.norm(random_axis)

    # Generate a small random angle
    angle = np.random.normal(0, sigma)

    new_axis = rotation_matrix_from_axis_angle(random_axis, angle) @ direction
    return new_axis / np.linalg.norm(new_axis)


def random_small_rotation(sigma):
    """Generate a small random rotation matrix.

    Uses the exponential map to generate a small random rotation matrix
    by sampling a random rotation axis and an angle from a Gaussian distribution
    with standard deviation sigma.

    Args:
        sigma: Standard deviation of the Gaussian noise to be added (in radians).

    Returns:
        A 3x3 rotation matrix representing a small random rotation.
    """

    # Random small rotation vector
    axis = np.random.normal(0.0, sigma, 3)
    theta = np.linalg.norm(axis)

    if theta < 1e-12:
        return np.eye(3)  # negligible rotation

    k = axis / theta  # unit axis
    k_mat = cross_matrix(k)

    # Rodrigues' formula: exp([axis]x)
    rot = np.eye(3) + np.sin(theta) * k_mat + (1 - np.cos(theta)) * (k_mat @ k_mat)
    return rot


def perturb_rotation_matrix(rot_mat: np.ndarray, sigma: float) -> np.ndarray:
    """Perturb a rotation matrix by applying a small random rotation.

    Args:
        rot_mat: A 3x3 rotation matrix to be perturbed.
        sigma: Standard deviation of the Gaussian noise to be added (in radians).

    Returns:
        A new 3x3 rotation matrix representing the perturbed rotation.

    Raises:
        ValueError: If the input matrix is not a valid rotation matrix.
    """
    if not is_rotation_matrix(rot_mat):
        raise ValueError("Input matrix must be a valid rotation matrix.")

    dR = random_small_rotation(sigma)
    return dR @ rot_mat


def rot_mat_x(angle: float) -> np.ndarray:
    """Generate a rotation matrix for a rotation around the x-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix representing the rotation around the x-axis.
    """
    return rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), angle)


def rot_mat_y(angle: float) -> np.ndarray:
    """Generate a rotation matrix for a rotation around the y-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix representing the rotation around the y-axis.
    """
    return rotation_matrix_from_axis_angle(np.array([0.0, 1.0, 0.0]), angle)


def rot_mat_z(angle: float) -> np.ndarray:
    """Generate a rotation matrix for a rotation around the z-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix representing the rotation around the z-axis.
    """
    return rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)


def get_flip_transform(axis: str) -> np.ndarray:
    """Generate a 4x4 transformation matrix that flips a point cloud along a specified axis.

    This function creates a transformation that rotates the point cloud by ±90° around
    the specified axis. The "flip" is achieved through a 90° rotation, which effectively
    reorients the point cloud along that axis direction.

    Args:
        axis: The axis specification for the flip transformation. Valid values are:
            - "x" or "nx": Flip along the x-axis (rotate by +90° or -90°)
            - "y" or "ny": Flip along the y-axis (rotate by +90° or -90°)
            - "z" or "nz": Flip along the z-axis (rotate by +90° or -90°)
            The "n" prefix indicates a negative rotation direction.

    Returns:
        A 4x4 homogeneous transformation matrix with the rotation component set
        to the specified flip rotation and zero translation.

    Raises:
        ValueError: If the axis parameter is not one of the valid values.

    Note:
        The transformation matrix has the form:

        .. math::

            T = \\begin{bmatrix}
                R & 0 \\\\
                0 & 1
            \\end{bmatrix}

        where :math:`R` is a 3x3 rotation matrix for ±90° around the specified axis.
    """
    transform = np.eye(4)
    if axis == "x" or axis == "nx":
        sign = -1 if axis == "nx" else 1
        transform[:3, :3] = rot_mat_x(sign * np.pi / 2)
    elif axis == "y" or axis == "ny":
        sign = -1 if axis == "ny" else 1
        transform[:3, :3] = rot_mat_y(sign * np.pi / 2)
    elif axis == "z" or axis == "nz":
        sign = -1 if axis == "nz" else 1
        transform[:3, :3] = rot_mat_z(sign * np.pi / 2)
    else:
        raise ValueError(f"Invalid flip axis: {axis}")

    return transform
