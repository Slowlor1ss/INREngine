# Engine Features

Everything except for image parsing and some mertrics such as SSIM have been implemented from scratch **without the use of any libraries**.

## **CUDA**
* **GPU Pipeline**:
    * Batch GEMM acceleration via cuBLAS (mainly `cublas<T>gemm`'s).
    * Cuda graphs and using Page locked (pinned) memory
    * Chunked inference engine (`PredictGPU()`) to bypass VRAM bounds on arbitrary output resolutions.

## General Neural Network/INR features
* **Adam Optimizer** bias correction using state tracking per layer.
* **Spatial gradient** Implemnted first and second derivatives for all activation function, so we can use spatial gradient for (in theory) sharper edges, in practice I had mixed results using this.
* **Forward & Backward pass** as in any neural netowrk we needed a forward and backwards pass.

## **Coordinate Encoders**
* **Normalized Direct Mapping**: Bounded [-1.0, 1.0] 2D spatial coordinate inputs.
* **Positional Encoding (PE)**: Standard sine/cosine frequency bands with spatial derivatives (`d/dx, d/dy`).
* **Gaussian Positional Encoding**: Gaussian Fourier features for high frequency spatial mapping.
* **Multi Resolution Dense Grid Encoding**:
    * Resolution growth from [base] to [finest].
    * Bilinear feature interpolation per level with closed form spatial slope computation.
    * Direct atomic gradient scattering (`ScatterLevelGradient()`) into VRAM feature tables.


## **Activation Functions**
* **SIREN (Sine Implicit Representation)**
* **WIRE (Wavelet Implicit Representation)** with dual weights (frequency and scale parameters)
* **FINER**
* **Standard activation funcitons**: LeakyReLU, ReLU, Sigmoid, Tanh, and Linear.


## **Loss Functions & Metrics**
* **Cost Functions**: MSE, Charbonnier Loss, and Joint Spatial Gradient Loss.
* **Python Metrics (`metrics.py` etc.)**:
Note: I used a library to compue a more accurate SSIM here as it requres sliding windows and I just wanted to focus on the actual INR more then implementing this
    * Error metrics: MSE, RMSE, MAE, max_err
    * PSNR (dB)
    * SSIM
    * Graphing for each
* **C++ Metrics (`ImageUtils()`)**:
these have been manually implemented
    * Global Mean Squared Error (MSE)
    * Peak Signal to Noise Ratio (PSNR).
    * Global Channel Structural Similarity Index Measure (Global SSIM).

## **Scripts**
* **Python "Integration"**:
    * I took the easy route and used windows' shared memory page to use python for the visualization in the end and some more complicated metrics and graph-ing as implementing this using the windows libraries would have been to far from the point of this project.

## **Type switching**
* **EngineFloat**:
    * I've implemented a "EngineFloat" which allows us to easily swap from a 32 bit float type to any other float type, such a bfloats (brain floats) which is a type of 16 bit float with a bigger exponent then the default half float, it's commonly used in neurals networks. [Link to a C++ con talk on bfloats from one of my old teachers.](https://youtu.be/Kftvt_l8wsQ?si=Tc8eVzc4ruyobmHN)

## **Legacy**
* **CPU Pipeline**: 
    * While I dont update the cpu pipeline anymore its still fully functional and can be used when running with (`--use-gpu 0`).
    * Using a Lock free multi threaded execution pool.
* C++ metrics and visulization are techniqually also depricated.
