import numpy as np
from im2col import im2col, _pair

def conv2d_im2col(img, weight, bias=0.0, stride=1, padding=0, dilation=1):
    H, W = img.shape
    C_out, C_in, kH, kW = weight.shape
    if C_in != 1:
        raise ValueError(f"Expected weight with C_in=1 for grayscale, got {C_in}")

    sH, sW = _pair(stride)
    pH, pW = _pair(padding)
    dH, dW = _pair(dilation)

    kH_eff = (kH - 1) * dH + 1
    kW_eff = (kW - 1) * dW + 1

    outH = (H + 2 * pH - kH_eff) // sH + 1
    outW = (W + 2 * pW - kW_eff) // sW + 1
    if outH <= 0 or outW <= 0:
        raise ValueError("Invalid output size; check kernel/stride/padding/dilation.")

    xC = img[np.newaxis, :, :]
    cols = im2col(xC, kH, kW, stride=stride, padding=padding, dilation=dilation)

    W_mat = weight.reshape(C_out, -1)
    out_cols = W_mat @ cols

    if np.isscalar(bias):
        out_cols += float(bias)
    else:
        bias = np.asarray(bias, dtype=out_cols.dtype)
        if bias.shape != (C_out,):
            raise ValueError(f"bias must be scalar or shape ({C_out},), got {bias.shape}")
        out_cols += bias[:, None]

    out = out_cols.reshape(C_out, outH, outW)
    return out

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
