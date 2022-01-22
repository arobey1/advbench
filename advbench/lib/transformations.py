import torch
from einops import rearrange, reduce, repeat
from kornia.geometry import warp_affine

def deg2rad(angle):
  return angle*torch.pi/180.0

def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    if not 2 <= len(tensor.shape) <= 4:
        raise AssertionError(f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}.")
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
    return center


def angle_to_rotation_matrix(angle, imgs):
    """Create a rotation matrix out of angles in degrees.

    Args:
        angle: tensor of angles in degrees,  shape Bx1.

    Returns:
        tensor of rotation matrices with shape (B, 2, 3).
    """
    B = angle.shape[0]
    ang_rad = deg2rad(angle)
    cos_a = torch.cos(ang_rad)
    sin_a = torch.sin(ang_rad)
    center = _compute_tensor_center(imgs)
    rotat_m = repeat(torch.eye(3), 'd1 d2 -> b d1 d2',b=B).clone()
    a_mat = torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1)
    rotat_m[:, :2, :2] = rearrange(a_mat, 'b (d1 d2) -> b d1 d2', d1=2, d2=2).clone()
    center = repeat(_compute_tensor_center(imgs), 'd -> b d',b=B).clone()
    shift_m = txs_to_translation_matrix(center)  
    shift_m_inv = txs_to_translation_matrix(-center)
    return  shift_m @ rotat_m @ shift_m_inv

def se_matrix(delta, imgs):
  """
  delta: Bx3 (third dimension is rotation , w translation, h traslation)
  returns se: Bx2x3
  """
  angle, txs  = delta[:, 0], delta[:,1:]
  affine = torch.zeros((angle.shape[0],2,3))
  rotat_m = angle_to_rotation_matrix(angle, imgs)
  trans_m = txs_to_translation_matrix(txs)
  return (rotat_m@trans_m)[:,:2,:]

def txs_to_translation_matrix(txs):
    """Create a translation matrix out of angles in pixels.

    Args:
        txs: tensor of  translations in pixels,  shape Bx2
    Returns:
        tensor of translation matrices with shape (B, 3, 3).
    """
    shift_m = repeat(torch.eye(3), 'd1 d2 -> b d1 d2', b=txs.shape[0]).clone()
    shift_m[:, :2, 2] = txs
    return shift_m

def se_transform(imgs, delta):
    return warp_affine(imgs, se_matrix(delta, imgs).to(imgs.device), imgs.shape[2:])