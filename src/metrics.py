import numpy as np
from skimage.metrics import structural_similarity

def quality_metrics(reference, candidate):
    a=reference.astype(np.float32); b=candidate.astype(np.float32)
    mse=float(np.mean((a-b)**2))
    psnr=float("inf") if mse == 0 else float(10*np.log10((255.0**2)/mse))
    ssim=float(structural_similarity(reference, candidate, data_range=255))
    return {"MSE":mse,"PSNR (dB)":psnr,"SSIM":ssim}
