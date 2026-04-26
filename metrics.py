from skimage import metrics
import torch
import torch.hub
from lpips.lpips import LPIPS
import os
import numpy as np

photometric = {
    "mse": None,
    "ssim": None,
    "psnr": None,
    "lpips": None
}
lpips_unavailable = False

def compute_img_metric(im1t: torch.Tensor, im2t: torch.Tensor,
                       metric="mse", margin=0, mask=None):
    """
    im1t, im2t: torch.tensors with batched imaged shape, range from (0, 1)
    """
    if metric not in photometric.keys():
        raise RuntimeError(f"img_utils:: metric {metric} not recognized")
    if photometric[metric] is None:
        if metric == "mse":
            photometric[metric] = metrics.mean_squared_error
        elif metric == "ssim":
            photometric[metric] = metrics.structural_similarity
        elif metric == "psnr":
            photometric[metric] = metrics.peak_signal_noise_ratio
        elif metric == "lpips":
            global lpips_unavailable
            if lpips_unavailable:
                return float("nan")
            try:
                photometric[metric] = LPIPS().cpu()
            except Exception as exc:
                lpips_unavailable = True
                print(f"[WARN] LPIPS unavailable, skip metric: {exc}")
                return float("nan")

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, 3, -1, -1)
        mask = mask.permute(0, 2, 3, 1).numpy()
        batchsz, hei, wid, _ = mask.shape
        if margin > 0:
            marginh = int(hei * margin) + 1
            marginw = int(wid * margin) + 1
            mask = mask[:, marginh:hei - marginh, marginw:wid - marginw]

    if im1t.dim() == 3:
        im1t = im1t.unsqueeze(0)
        im2t = im2t.unsqueeze(0)
    im1t = im1t.detach().cpu()
    im2t = im2t.detach().cpu()

    if im1t.shape[-1] == 3:
        im1t = im1t.permute(0, 3, 1, 2)
        im2t = im2t.permute(0, 3, 1, 2)

    im1 = im1t.permute(0, 2, 3, 1).numpy()
    im2 = im2t.permute(0, 2, 3, 1).numpy()
    batchsz, hei, wid, _ = im1.shape
    if margin > 0:
        marginh = int(hei * margin) + 1
        marginw = int(wid * margin) + 1
        im1 = im1[:, marginh:hei - marginh, marginw:wid - marginw]
        im2 = im2[:, marginh:hei - marginh, marginw:wid - marginw]
    im1 = np.clip(im1, 0.0, 1.0)
    im2 = np.clip(im2, 0.0, 1.0)
    lpips_im1 = (im1t * 2 - 1).clamp(-1, 1)
    lpips_im2 = (im2t * 2 - 1).clamp(-1, 1)
    values = []

    for i in range(batchsz):
        if metric in ["mse", "psnr"]:
            im1_i = im1[i]
            im2_i = im2[i]
            if mask is not None:
                im1_i = im1_i * mask[i]
                im2_i = im2_i * mask[i]
            value = photometric[metric](
                im1_i, im2_i
            )
            if mask is not None:
                hei, wid, _ = im1_i.shape
                pixelnum = mask[i, ..., 0].sum()
                value = value - 10 * np.log10(hei * wid / pixelnum)
        elif metric in ["ssim"]:
            win_size = min(7, hei, wid)
            if win_size % 2 == 0:
                win_size -= 1
            if win_size < 3:
                raise ValueError("SSIM requires image height/width >= 3 after cropping.")
            value, ssimmap = photometric["ssim"](
                im1[i], im2[i], channel_axis=-1, data_range=1.0, full=True, win_size=win_size
            )
            if mask is not None:
                value = (ssimmap * mask[i]).sum() / mask[i].sum()
        elif metric in ["lpips"]:
            value = photometric[metric](
                lpips_im1[i:i + 1], lpips_im2[i:i + 1]
            )
        else:
            raise NotImplementedError
        values.append(value)

    return sum(values) / len(values)
