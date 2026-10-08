from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


Array3D = np.ndarray


@dataclass(frozen=True)
class MaskAlignmentResult:
    mask_hwd: Array3D
    original_shape_hwd: Tuple[int, int, int]
    aligned_shape_hwd: Tuple[int, int, int]
    was_resampled: bool
    reason: str


def _shape_hwd(image: nib.spatialimages.SpatialImage) -> Tuple[int, int, int]:
    shape = tuple(int(v) for v in image.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D image, got shape {shape}")
    return shape


def _resize_binary_mask_to_shape(mask_hwd: Array3D, target_shape_hwd: Tuple[int, int, int]) -> Array3D:
    zoom_factors = tuple(float(target) / float(source) for source, target in zip(mask_hwd.shape, target_shape_hwd))
    resized = ndimage.zoom(mask_hwd.astype(np.float32, copy=False), zoom=zoom_factors, order=0, prefilter=False)

    if resized.shape != target_shape_hwd:
        adjusted = np.zeros(target_shape_hwd, dtype=np.float32)
        common_shape = tuple(min(int(src), int(dst)) for src, dst in zip(resized.shape, target_shape_hwd))
        source_slices = tuple(slice(0, length) for length in common_shape)
        target_slices = tuple(slice(0, length) for length in common_shape)
        adjusted[target_slices] = resized[source_slices]
        resized = adjusted

    return (np.asarray(resized, dtype=np.float32) > 0.5).astype(np.float32, copy=False)


def load_binary_mask_aligned_to_image(
    image_nifti: nib.spatialimages.SpatialImage,
    mask_path: str,
    affine_atol: float = 1e-3,
) -> MaskAlignmentResult:
    mask_nifti = nib.load(mask_path)
    image_shape = _shape_hwd(image_nifti)
    mask_shape = _shape_hwd(mask_nifti)

    mask_hwd = np.asarray(mask_nifti.get_fdata(dtype=np.float32))
    if tuple(int(v) for v in mask_hwd.shape) != mask_shape:
        raise ValueError(f"Unexpected mask array shape {mask_hwd.shape}, expected {mask_shape}")

    same_shape = mask_shape == image_shape
    same_affine = np.allclose(mask_nifti.affine, image_nifti.affine, atol=affine_atol)
    binary_mask_hwd = (mask_hwd > 0.5).astype(np.float32, copy=False)
    original_nonzero = int(np.count_nonzero(binary_mask_hwd))

    if same_shape and same_affine:
        return MaskAlignmentResult(
            mask_hwd=binary_mask_hwd,
            original_shape_hwd=mask_shape,
            aligned_shape_hwd=image_shape,
            was_resampled=False,
            reason="",
        )

    reason_parts = []
    if not same_shape:
        reason_parts.append(f"shape {mask_shape} -> {image_shape}")
    if not same_affine:
        reason_parts.append("affine mismatch")

    # Resample masks with nearest-neighbor interpolation so labels stay binary.
    binary_mask = nib.Nifti1Image(binary_mask_hwd, mask_nifti.affine, mask_nifti.header)
    resampled = resample_from_to(
        binary_mask,
        (image_shape, image_nifti.affine),
        order=0,
        mode="constant",
        cval=0.0,
    )
    aligned_mask_hwd = (np.asarray(resampled.dataobj, dtype=np.float32) > 0.5).astype(np.float32, copy=False)

    if int(np.count_nonzero(aligned_mask_hwd)) == 0 and original_nonzero > 0:
        aligned_mask_hwd = _resize_binary_mask_to_shape(binary_mask_hwd, image_shape)
        reason_parts.append("affine_resample_empty")
        reason_parts.append("fallback_shape_only_resize")

    return MaskAlignmentResult(
        mask_hwd=aligned_mask_hwd,
        original_shape_hwd=mask_shape,
        aligned_shape_hwd=image_shape,
        was_resampled=True,
        reason="; ".join(reason_parts),
    )
