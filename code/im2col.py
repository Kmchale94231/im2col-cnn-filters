import numpy as np

def _pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    return int(x), int(x)

def im2col(x, kH, kW, stride=1, padding=0, dilation=1):
    """TODO: Convert 2D convolution into matrix multiplication via columnization.
    Args:
        x: (C, H, W) input
        kH, kW: kernel height/width
        stride: int or (sH, sW)
        padding: int or (pH, pW)
        dilation: int or (dH, dW)
    Returns:
        cols: (C*kH*kW, outH*outW)
    Steps to implement:
      1) Parse stride/padding/dilation with _pair (provided).
      2) Compute effective kernel size with dilation.
      3) Compute out spatial dims (H_out, W_out) and pad the input.
      4) Extract each sliding window into a column and stack.
    """
    C, H, W = x.shape

    sH, sW = _pair(stride)
    pH, pW = _pair(padding)
    dH, dW = _pair(dilation)

    kH_eff = (kH - 1) * dH + 1
    kW_eff = (kW - 1) * dW + 1

    outH = (H + 2*pH - kH_eff) // sH + 1
    outW = (W + 2*pW - kW_eff) // sW + 1
    if outH <= 0 or outW <= 0:
        raise ValueError("Invalid output size — adjust stride/padding/dilation.")

    if pH > 0 or pW > 0:
        x_pad = np.pad(x, ((0,0), (pH,pH), (pW,pW)), mode="constant")
    else:
        x_pad = x

    cols = np.empty((C * kH * kW, outH * outW), dtype=x.dtype)

    col_idx = 0
    for oh in range(outH):
        for ow in range(outW):
            h_start = oh * sH
            w_start = ow * sW

            patch = x_pad[:, 
                          h_start : h_start + kH_eff : dH,
                          w_start : w_start + kW_eff : dW]   # (C, kH, kW)

            cols[:, col_idx] = patch.reshape(-1)
            col_idx += 1

    return cols

