import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

def glcm_features(gray, distances=(1,), levels=8):
    q=np.floor(gray.astype(np.float32)*levels/256).astype(np.uint8)
    q=np.clip(q,0,levels-1)
    matrix=graycomatrix(q, distances=list(distances), angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                        levels=levels, symmetric=True, normed=True)
    names=("contrast","correlation","energy","homogeneity")
    result={n:float(np.mean(graycoprops(matrix,n))) for n in names}
    return result, matrix

def lbp_features(gray, points=8, radius=1):
    lbp=local_binary_pattern(gray, int(points), float(radius), method="uniform")
    bins=int(points)+3
    hist,_=np.histogram(lbp.ravel(), bins=np.arange(-0.5,bins+0.5,1), density=False)
    hist=hist.astype(float); hist/=max(hist.sum(),1)
    return lbp,hist

def gabor_bank(gray):
    import cv2
    responses=[]
    for angle in (0,np.pi/4,np.pi/2,3*np.pi/4):
        kernel=cv2.getGaborKernel((21,21),4.0,angle,10.0,0.5,0,ktype=cv2.CV_32F)
        r=cv2.filter2D(gray,cv2.CV_32F,kernel)
        responses.append(cv2.normalize(np.abs(r),None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
    return responses

def hog_features(gray):
    from skimage.feature import hog
    feature, visualization=hog(gray, orientations=9, pixels_per_cell=(8,8),
        cells_per_block=(2,2), block_norm="L2-Hys", visualize=True, feature_vector=True)
    return feature, visualization
