# Linear Discriminant Analysis

Linear Discriminant Analysis on UIUC Car Database

A collection of 2556 training image patches of which 2442 are patches of background, tagged as class label C<sub>2</sub> , whereas
the rest 124 are patches of containing cars and tagged as class label C<sub>1</sub>. Each of these ground truth image patches is of size 81 × 31. The 2D visualization of projection vector **w** computed is shown below. From this figure, which is obtained from least squares regression training, and there is no car like structural traits upon visualization of **w**.
Before deciding best classifier on test data, the precision-recall curves were studied. One such best classifier is obtianed is shown below. 

![Test result](images/uiucTestResults/TEST_RHO_9_0_search_000.jpg) Test Image
![Projection Vector](images/Visualizing_projection_vector.jpg) Projector vector


# Projection Vector

The projection vector thus obtained is as follows
(**X**<sup>T</sup>**X**)<sup>-1</sup>**X**<sup>T</sup>y


# Separable Discriminant Analysis 

Separable Tensor based discriminant analysis

In the previous approach where training image patches x ∈ R m×n of size m × n are vectorized into single 1 dimensional vector. Instead treating images for what they are, we use tensors. In this approach we compute the projection tensor<sup>1</sup> by applying tensor contractions to the given set of training image patches and use alternating least squares. 


A tensor also known as n-array or multidimensional matrix or n-mode matrix, is a higher order generalization of a vector (first order tensor) and a matrix (second order tensor). In this short description on second order tensors **X** to represent images.
A typical training set representing grey-value images of size m x n and training set consists of **N** image patches. Tensor discriminant analysis requires a projection tensor **W** which solves the regression problem. This approach address the problem of singular matrices. The tensor projector and the test images are shown below.


![Test Image](images/robust.jpg) (Test Image)

![Tensor Projector](images/Visualizing_tensor_projection_9.jpg) (Visualizing Tensor projector)


## Build and Run Instructions

### 1. Install dependencies

- On macOS with Homebrew:
  - `brew install opencv boost`

### 2. Build with CMake

From the repository root:

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

This produces the executable `lda_run` in the `build/` directory.

Alternatively, use an out-of-source build from the repo root:

```bash
cmake -S . -B build
cmake --build build --target lda_run
```

### 3. Run the executable

#### Image-based processing (Default)

From `build/`:

```bash
./lda_run ../images/uiucTrain/ ../images/uiucTest/ ../images/uiucTestResults/ 9 ../images/visualizeW_RHO_9.PGM
```

#### Video stream processing (NEW)

Process a video file with frame-by-frame car detection and annotation:

```bash
./lda_run ../images/uiucTrain/ input_video.mp4 output_dir/ 9 ../images/visualizeW_RHO_9.PGM
```

The program automatically detects video input by file extension (.mp4, .avi, .mov, .mkv, .flv, .wmv, .m4v, .webm) and processes it frame-by-frame with the trained LDA template. Output video with bounding box annotations is written to `output_dir/output_video.mp4`.

**Features:**
- Supports multiple video formats
- Per-frame template matching with non-maximum suppression
- Annotated output video with detection bounding boxes
- Frame-by-frame progress reporting

### 4. Run tests

From `build/`:

```bash
cmake --build . --target lda_tests
ctest --output-on-failure
```

### 5. Verify output

- Check generated result images in `images/uiucTestResults/`.

### 6. Notes

- The repository includes a root `CMakeLists.txt` for easy builds.
- If your environment differs, adjust OpenCV/Boost paths or use CMake cache variables.
- Local runtime artifacts such as `lda_run` and `file.txt` are ignored via `.gitignore`.

## Hugging Face integration (Option 2)

A Python helper is provided to run a Hugging Face segmentation model and produce a binary mask. It can be used as a modern replacement for the classical segmentation algorithm.

Files:
- `python/aneurysm_segmentation_hf.py` — runner script that loads a SegFormer model and outputs a mask
- `python/requirements.txt` — Python dependencies (PyTorch, Transformers, Pillow)

Quick usage (create a venv first):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
python python/aneurysm_segmentation_hf.py images/visualizeW_RHO_9.PGM out_mask.png
```

Notes:
- Downloading the model requires internet access and may be slow on first run.
- You can replace the default model with any HF-compatible semantic segmentation model by passing its name as the third argument.




