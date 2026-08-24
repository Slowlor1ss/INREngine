# Implicit Neural Representation Engine

A (**no libraries**) **custom made** **C++/CUDA** framework for Implicit Neural Representations (INRs). Using CUDA to run the network on the GPU.

---

**TODO**: put some images here / demonstration

## **Key Architecture Highlights**
* **Dual Execution Pipelines**: Seamless switching between:
    * **Lock free** C++ **multi threaded** CPU engine (`TrainingThreadPool.h/.cpp`) 
    * **GPU CUDA/cuBLAS** implementation.
* **Grid Encoding**: Multi resolution feature grids.
* **CUDA Graph Acceleration**: No overhead kernel launches using captured CUDA execution graphs.
* **Spatial Gradient Loss**: Backpropagates both color errors and continuous slopes (Delta x, Delta y) to preserve high frequency edges.
* and a lot more see: [**Features.**](Docs/FEATURES.md)

## **Links**
* [**Features**](Docs/FEATURES.md): List of encodings, activation functions, pipelines, and general features
* [**Results & Metrics**](Docs/RESULTS.md): PSNR/SSIM comparisons, render outputs, and benchmark logs
* [**Performance**](Docs/PERFORMANCE.md): Computational performance comparisons aginst popular public libraries
* [**Future Work**](Docs/FUTURE_WORK.md): Planned work


## **What is a Implicit Neural Representation (INR) network?**

To understand INRs, it helps to look at how computers normally handle images.

Normally, an image is saved as a grid of pixels. For a standard 1080p image, you store a list of more then 2 million rgb pixels. 

An **Implicit Neural Representation (INR)** *instead* of saving a list of pixels, it uses a neural network to learn the image as a continuous mathematical function. 

Instead of asking the computer, *"What color is pixel number 405?"* 
You ask the neural network, *"If I give you the **(X, Y) coordinates**, what color should go there?"* 
And network spits out the exact **(R, G, B) color** for that location. (*it's not the same keep reading*)

### Whats the point?
The image is no longer a fixed grid of pixels, it is a **continuous** math equation. 

* **"Infinite" Resolution:** You can ask the network for the color at a fractional coordinate like `(42.5, 54.2)`. It doesn't matter that there is no "half pixel" on a normal screen. Unlike normal; the network will just calculate the exact color that *should* exist exactly between those points. This makes it usefull for upscaling withouth having the risk of random things beign hallucinated as opposed to using a generative model. And a more "mathmatical" approach as opposed to simply interpolating the pixels.
* **Compression:** Instead of storing millions of pixels, you only need to store the weights of a very small neural network. This becomes especially useful for gigapixel images [E.g.](https://shacira.github.io/)

In short: **The neural network doesn't look at the image; the neural network *becomes* the image.** *(-Bruce Lee probably)*

## **Quick Start**

### **Running the Engine**
Run the `BUILD.bat` \
OR\
You can also just run the engine by compiling the code using visual studio (I chose not to use CMake **for this project**) and running the generated 3b1b_backprop.exe, doing so will run the network with the default settings which can be changed in (`config.h`)

Alternativly you can change the hyperparameters using command line arguments.

Example useage:
```bash
3b1b_backprop.exe --i 0064_x4.bmp --o 0064_x4_siren_2x256.bmp --layers 256 256 3 --act Siren Siren None --batch 65536 --lr 1 --set-live 0
```

Command line arguments: \
(you can print these using `--h`)
```bash
Usage:
  --i            | Set input filename
  --o            | Set output path (can specify file as well e.g. weights_biases.csv)
  --HDin         | Set path to HD reference image
  --gpu          | Enable CUDA
  --benchmark    | Enable Benchmarking
  --layers       | Set the layers e.g. --layers 128 128 3
  --act          | Set the activation function(s) e.g. --act Wire Siren None
                 | Valid options are: Finer, LeakyReLU, None, ReLU, Sigmoid, Siren, Tanh, Wire, WireHybrid
  --set-pe 0/ 1  | Enable positional encoding
  --freq         | Set positional encoding Frequencies
  --set-gaussian-pe 0/1 | Set Gaussian Positional Encoding (does NOT use --freq)
  --batch        | Set batch Size
  --lr           | Set the learning Rate
  --scale        | Set the resize scale
  --set-live 0/1 | Disable live viewer on start; Note: this can be re-enabled during runtime using 'v'
  --denoise 0/1  | Weather or not we use denoise jitter
```

## Thanks

Thanks to [@Jan300100] (https://www.github.com/Jan300100) for getting me in to this project and who was a huge help with the start of this project.