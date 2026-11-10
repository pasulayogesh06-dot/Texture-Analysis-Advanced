# Texture Analysis Studio — Advanced Digital Image Processing

A modular Streamlit application upgraded from the original **Texture Analysis Using GLCM and LBP** project. The original image upload, grayscale conversion, four GLCM properties, LBP map, normalized histogram and CSV exports are retained, with additional DIP modules.

## Features

- **Original texture analysis:** GLCM Contrast, Correlation, Energy and Homogeneity; Local Binary Pattern map and normalized histogram.
- **Enhancement:** CLAHE and histogram equalization.
- **Filtering:** Gaussian, median, bilateral, box and sharpening filters.
- **Edge detection:** Canny, Sobel and Laplacian.
- **Segmentation:** Otsu, adaptive and global thresholding.
- **Frequency domain:** centered log-magnitude 2D Fourier spectrum.
- **Advanced descriptors:** configurable GLCM/ LBP, four-orientation Gabor response and HOG visualization.
- **Quality evaluation:** MSE, PSNR and SSIM for a user-supplied reference image with matching dimensions.
- **Exports:** CSV feature files, processed PNG images and a PDF analysis report.

## Project structure

```text
Texture_Analysis_Advanced/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── image_ops.py
    ├── texture.py
    ├── metrics.py
    └── report.py
```

## Windows setup (VS Code / PowerShell)

1. Extract the ZIP file.
2. Open the extracted `Texture_Analysis_Advanced` folder in VS Code.
3. Open **Terminal → New Terminal** (PowerShell).
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use this for the current terminal only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

6. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Start the application:

   ```powershell
   streamlit run app.py
   ```

Streamlit will print a local URL, usually `http://localhost:8501`.

## How to use

1. Upload an image in the sidebar.
2. Open the required module tab.
3. Adjust the available parameters and inspect the resulting image.
4. Download the processed PNG, feature CSV, or PDF report.
5. For PSNR / SSIM / MSE, upload a reference image with the same dimensions as the input image. Metrics are intentionally not fabricated when a reference pair is missing or incompatible.

## Technical notes and interpretation

- GLCM quantizes grayscale values to the selected number of levels and averages the four standard directions (0°, 45°, 90°, 135°). The default 8 levels and distance 1 preserve the original project's calculation setup.
- LBP uses the uniform pattern method. Defaults remain P=8 and R=1, matching the original project.
- PSNR, SSIM and MSE compare two images; a single image alone is insufficient to calculate a meaningful reference-based quality score.
- The application caps the longest image dimension at 2400 pixels for interactive performance. The UI indicates when resizing occurs.
- This is a classical computer-vision / DIP project; it does not claim to train a deep-learning model.

## Testing

The source files are syntax-checked and the processing modules are exercised with a generated test image during packaging. Before a college demonstration, also launch the Streamlit app locally and test it with several real images and a matching reference pair.

## Dependencies

Python 3.10–3.12 recommended. Dependencies are listed in `requirements.txt`.
