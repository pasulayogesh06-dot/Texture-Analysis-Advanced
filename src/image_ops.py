import cv2
import numpy as np

def to_gray(image):
    if image.ndim == 2: return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def enhance(gray, method="CLAHE", clip_limit=2.0, tile_size=8):
    if method == "Histogram Equalization":
        return cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
    return clahe.apply(gray)

def filter_image(gray, method="Gaussian", kernel=5):
    k = max(3, int(kernel) | 1)
    if method == "Gaussian": return cv2.GaussianBlur(gray, (k,k), 0)
    if method == "Median": return cv2.medianBlur(gray, k)
    if method == "Bilateral": return cv2.bilateralFilter(gray, k, 75, 75)
    if method == "Box": return cv2.blur(gray, (k,k))
    if method == "Sharpen":
        blur = cv2.GaussianBlur(gray, (k,k), 0)
        return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return gray

def detect_edges(gray, method="Canny", low=80, high=160):
    if method == "Canny": return cv2.Canny(gray, int(low), int(high))
    if method == "Sobel":
        x=cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); y=cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag=cv2.magnitude(x,y); return cv2.convertScaleAbs(mag)
    if method == "Laplacian": return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F))
    return cv2.Canny(gray, int(low), int(high))

def segment(gray, method="Otsu", threshold=127, k=3):
    if method == "Otsu":
        _, out=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU); return out
    if method == "Adaptive":
        block=max(3, int(k)*2+1)
        return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,2)
    _, out=cv2.threshold(gray,int(threshold),255,cv2.THRESH_BINARY)
    return out

def fourier_visualization(gray):
    f=np.fft.fft2(gray.astype(np.float32)); shifted=np.fft.fftshift(f)
    magnitude=20*np.log(np.abs(shifted)+1)
    magnitude=cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude

def gabor_response(gray, ksize=21, sigma=4.0, theta=0.0, lambd=10.0, gamma=0.5):
    kernel=cv2.getGaborKernel((int(ksize)|1,int(ksize)|1),float(sigma),float(theta),float(lambd),float(gamma),0,ktype=cv2.CV_32F)
    response=cv2.filter2D(gray,cv2.CV_32F,kernel)
    return cv2.normalize(np.abs(response),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
