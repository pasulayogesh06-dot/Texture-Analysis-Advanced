import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.image_ops import enhance, filter_image, detect_edges, segment, fourier_visualization, gabor_response
from src.texture import glcm_features, lbp_features, hog_features
from src.metrics import quality_metrics
from src.report import create_pdf_report


# --------------------------- PAGE & THEME ---------------------------
st.set_page_config("Texture Analysis Studio", "◈", layout="wide")

st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 15% 0%,#182b40 0%,#0b1220 48%,#080d17 100%);color:#e7edf7}
[data-testid="stHeader"]{background:rgba(8,13,23,.75)}
.block-container{padding-top:1.5rem;max-width:1500px}
h1,h2,h3,h4{color:#f4f7fc!important}
p,label,.stMarkdown{color:#bdc9da}
.hero{padding:26px 30px;border:1px solid #263a54;border-radius:18px;background:linear-gradient(120deg,#1a3651,#111b2d);margin-bottom:20px}
.hero h1{color:white!important;margin:0}.hero p{color:#9db3cc}
div[data-testid="stMetric"]{background:#121e30;border:1px solid #263b55;padding:13px;border-radius:12px}
div[data-testid="stMetricValue"]{color:#72e0c0}
.stButton>button,.stDownloadButton>button{background:#123b3a;color:#e8fff8;border:1px solid #2b806e;border-radius:9px}
</style>
<div class="hero"><h1>◈ Texture Analysis Studio</h1>
<p>Advanced Digital Image Processing · GLCM · LBP · Enhancement · Segmentation · Frequency Analysis</p></div>
""", unsafe_allow_html=True)


# ------------------------------ HELPERS ------------------------------
def show_pair(left, right, left_title="Before", right_title="After"):
    a, b = st.columns(2)
    a.image(left, caption=left_title, use_container_width=True)
    b.image(right, caption=right_title, use_container_width=True)


def download_image(label, image, filename):
    ok, data = cv2.imencode(".png", image)
    if ok:
        st.download_button(label, data.tobytes(), filename, "image/png")


def show_histogram(values, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(values)), values)
    ax.set(title=title, xlabel=xlabel, ylabel="Normalized frequency")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------- SIDEBAR -------------------------------
with st.sidebar:
    st.header("WORKSPACE")
    uploaded = st.file_uploader("Input image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    st.divider()
    st.subheader("Analysis settings")
    levels = st.select_slider("GLCM gray levels", [4, 8, 16, 32], value=8)
    points = st.select_slider("LBP sampling points", [8, 16, 24], value=8)
    radius = st.slider("LBP radius", 1, 4, 1)

if uploaded is None:
    st.subheader("Explore the processing workspace")
    st.write("Upload an image from the sidebar to use all processing modules.")
    a, b, c = st.columns(3)
    a.markdown("#### Texture descriptors")
    a.write("GLCM, LBP, Gabor and HOG.")
    b.markdown("#### Spatial processing")
    b.write("Enhancement, filtering, edges and segmentation.")
    c.markdown("#### Frequency & evaluation")
    c.write("Fourier, PSNR, SSIM, MSE and PDF reports.")
    st.info("For PSNR / SSIM, upload a reference image with matching dimensions.")
    st.stop()


# --------------------------- IMAGE PROCESSING ---------------------------
try:
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Unable to read this image. Please upload another file.")
        st.stop()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) > 2400:
        scale = 2400 / max(h, w)
        size = (int(w * scale), int(h * scale))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        h, w = gray.shape
        st.warning("Large image resized to 2400 px maximum dimension.")

    glcm, glcm_matrix = glcm_features(gray, distances=(1,), levels=int(levels))
    lbp, lbp_hist = lbp_features(gray, points=int(points), radius=int(radius))

    st.subheader("Image overview")
    m = st.columns(4)
    for col, label, value in zip(m, ["Width", "Height", "Pixels", "Channels"],
                                 [f"{w:,} px", f"{h:,} px", f"{w*h:,}", "RGB · 8-bit"]):
        col.metric(label, value)

    names = ["Overview", "Enhancement", "Filtering", "Edges", "Segmentation",
             "Fourier", "Texture Lab", "Gabor & HOG", "Quality & Report"]
    tabs = st.tabs(names)

    # ------------------------------- OVERVIEW -------------------------------
    with tabs[0]:
        st.subheader("Input image")
        show_pair(rgb, gray, "Original RGB", "Grayscale")
        st.markdown("#### GLCM features")
        cols = st.columns(4)
        for col, (name, value) in zip(cols, glcm.items()):
            col.metric(name.title(), f"{value:.4f}")
        glcm_df = pd.DataFrame({"Feature": [x.title() for x in glcm], "Value": list(glcm.values())})
        st.dataframe(glcm_df, hide_index=True, use_container_width=True)
        st.download_button("Download GLCM CSV", glcm_df.to_csv(index=False), "glcm_features.csv", "text/csv")
        show_histogram(lbp_hist, "LBP normalized histogram", "Pattern code")
        lbp_df = pd.DataFrame({"Pattern": range(len(lbp_hist)), "Frequency": lbp_hist})
        st.download_button("Download LBP CSV", lbp_df.to_csv(index=False), "lbp_histogram.csv", "text/csv")

    # ----------------------------- ENHANCEMENT -----------------------------
    with tabs[1]:
        st.subheader("Image enhancement")
        method = st.selectbox("Method", ["CLAHE", "Histogram Equalization"])
        clip = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, .5)
        tile = st.slider("CLAHE tile size", 4, 16, 8, 2)
        result = enhance(gray, method, clip_limit=clip, tile_size=tile)
        show_pair(gray, result)
        download_image("Download enhanced image", result, "enhanced.png")

    # -------------------------------- FILTERING --------------------------------
    with tabs[2]:
        st.subheader("Image filtering")
        method = st.selectbox("Filter", ["Gaussian", "Median", "Bilateral", "Box", "Sharpen"])
        kernel = st.slider("Kernel size", 3, 15, 5, 2)
        result = filter_image(gray, method, kernel)
        show_pair(gray, result, "Input", f"{method} result")
        download_image("Download filtered image", result, "filtered.png")

    # ---------------------------------- EDGES ----------------------------------
    with tabs[3]:
        st.subheader("Edge detection")
        method = st.selectbox("Operator", ["Canny", "Sobel", "Laplacian"])
        low = st.slider("Lower threshold", 0, 255, 80)
        high = st.slider("Upper threshold", 0, 255, 160)
        if low > high:
            st.warning("Lower threshold must not exceed upper threshold.")
        else:
            result = detect_edges(gray, method, low, high)
            st.image(result, caption=f"{method} edge map", use_container_width=True)
            download_image("Download edge map", result, "edges.png")

    # ------------------------------ SEGMENTATION ------------------------------
    with tabs[4]:
        st.subheader("Image segmentation")
        method = st.selectbox("Method", ["Otsu", "Adaptive", "Global threshold"])
        threshold = st.slider("Global threshold", 0, 255, 127)
        block_scale = st.slider("Adaptive block scale", 1, 10, 3)
        result = segment(gray, method, threshold=threshold, k=block_scale)
        show_pair(gray, result, "Grayscale input", f"{method} result")
        download_image("Download segmentation mask", result, "segmentation.png")

    # ---------------------------------- FOURIER ----------------------------------
    with tabs[5]:
        st.subheader("2D Fourier transform")
        result = fourier_visualization(gray)
        st.image(result, caption="Centered log-magnitude spectrum", use_container_width=True)
        st.caption("Bright central energy represents low spatial frequencies.")
        download_image("Download Fourier spectrum", result, "fourier_spectrum.png")

    # ------------------------------- TEXTURE LAB -------------------------------
    with tabs[6]:
        st.subheader("Advanced GLCM and LBP")
        lo, hi = float(np.min(lbp)), float(np.max(lbp))
        lbp_view = (((lbp.astype(np.float32) - lo) / (hi - lo)) * 255).astype(np.uint8) if hi > lo else np.zeros(lbp.shape, np.uint8)
        left, right = st.columns(2)
        left.image(lbp_view, caption="LBP code map", use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.imshow(glcm_matrix[:, :, 0, 0], cmap="magma")
        ax.set(title="GLCM · distance 1 · angle 0°", xlabel="Gray level", ylabel="Gray level")
        fig.tight_layout()
        right.pyplot(fig)
        plt.close(fig)
        show_histogram(lbp_hist, "LBP histogram", "Uniform LBP code")
        st.write("GLCM measures gray-level relationships; LBP describes local intensity patterns.")

    # ------------------------------- GABOR & HOG -------------------------------
    with tabs[7]:
        st.subheader("Gabor filters and HOG")
        angle = st.select_slider("Gabor orientation", [0, 45, 90, 135], value=0)
        gabor = gabor_response(gray, theta=np.deg2rad(angle))
        left, right = st.columns(2)
        left.image(gabor, caption=f"Gabor response · {angle}°", use_container_width=True)
        if min(gray.shape) >= 16:
            descriptor, hog = hog_features(gray)
            hog_view = cv2.normalize(hog.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            right.image(hog_view, caption="HOG visualization", use_container_width=True)
            st.caption(f"HOG descriptor length: {len(descriptor):,}")
        else:
            right.warning("HOG requires an image at least 16 × 16 pixels.")
        download_image("Download Gabor response", gabor, "gabor.png")

    # ----------------------------- QUALITY & REPORT -----------------------------
    with tabs[8]:
        st.subheader("Image quality metrics")
        st.write("Compare the current image with a reference image of matching dimensions.")
        ref_file = st.file_uploader("Reference image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="reference")
        quality = None

        if ref_file:
            ref = cv2.imdecode(np.frombuffer(ref_file.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if ref is None:
                st.error("Could not read the reference image.")
            elif ref.shape != gray.shape:
                st.warning(f"Dimensions differ: reference {ref.shape[::-1]}, current {gray.shape[::-1]}.")
            elif min(gray.shape) < 7:
                st.warning("Image must be at least 7 × 7 pixels to calculate SSIM.")
            else:
                quality = quality_metrics(ref, gray)
                cols = st.columns(len(quality))
                for col, (name, value) in zip(cols, quality.items()):
                    col.metric(name, "∞" if not np.isfinite(value) else f"{value:.4f}")
                diff = cv2.absdiff(ref, gray)
                st.image(diff, caption="Absolute difference map", use_container_width=True)
        else:
            st.info("Upload a reference image to calculate PSNR, SSIM and MSE.")

        report = create_pdf_report(
            "analysis",
            {"Filename": uploaded.name, "Width": w, "Height": h, "Pixels": w*h,
             "Color space": "RGB / Grayscale"},
            glcm, quality
        )
        st.download_button("Download PDF report", report, "texture_analysis_report.pdf", "application/pdf")

except Exception as error:
    st.error(f"Processing failed: {type(error).__name__}: {error}")
    st.exception(error)


# -------------------------------- FOOTER --------------------------------
st.divider()
st.caption("Texture Analysis Studio · Digital Image Processing · OpenCV · scikit-image · Streamlit")
