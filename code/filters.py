import numpy as np
from im2col import im2col

def conv2d_im2col(img, weight, bias=0.0, stride=1, padding=0, dilation=1):
    """TODO: Convolve a grayscale image with filters using im2col.
    Args:
        img: (H, W) in [0,1]
        weight: (C_out, 1, kH, kW)
        bias: float or (C_out,)
    Returns:
        out: (C_out, H_out, W_out)
    Steps to implement:
      1) Use im2col to get columns of shape (kH*kW, outHW).
      2) Reshape weight to (C_out, kH*kW) and matrix-multiply.
      3) Add bias and reshape back to (C_out, H_out, W_out).
    """
    # TODO: your code here
    raise NotImplementedError("Implement conv2d_im2col.")

def kernels():
    gaussian_3x3 = (1/16.0) * np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32)
    sharpen_3x3  = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    def pack(*mats):
        return np.stack([m[np.newaxis, :, :] for m in mats], axis=0)
    return {
        "gaussian_3x3": pack(gaussian_3x3),
        "sharpen_3x3":  pack(sharpen_3x3),
        "sobel_x":      pack(sobel_x),
        "sobel_y":      pack(sobel_y),
    }
