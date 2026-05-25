# Assignment Overview (till 26.05)

Image convolution is a fundamental operation in image processing where each
output pixel is calculated as a weighted sum of neighboring input pixels. This
operation forms the basis for various image filters like blurring, sharpening,
edge detection, and embossing. Implementing convolution efficiently on GPUs
presents an excellent opportunity to learn parallel programming concepts and
optimization techniques.

## Part 1: Basic Implementation

- CPU Implementation: Implement a sequential CPU version of 2D convolution
- Support arbitrary kernel sizes (3×3, 5×5, 7×7, etc.)
- Handle image boundaries appropriately (zero padding or clamping)
- Naive CUDA Implementation: Create a basic CUDA kernel where each thread computes one output pixel
- Implement appropriate memory transfers between the host and the device
- Ensure correctness by comparing with the CPU implementation
- Measure and compare performance between CPU and GPU implementations

## Part 2: Optimization Techniques

- Shared Memory Optimization: Implement a tiled version using shared memory to reduce global memory accesses
- Analyze the impact of different tile sizes on performance
- Constant Memory for Filter Kernels: Store the convolution filter in constant memory
- Analyze the performance impact of different filter sizes
- Separable Filters Implementation: For separable filters (like Gaussian blur), implement a two-pass approach
- Compare the performance with the direct 2D implementation
- Memory Access Optimization: Ensure coalesced memory access patterns
- Explore padding strategies to avoid bank conflicts
- Compare aligned vs. unaligned memory access performance

## Part 3: Advanced Features and Analysis

- Performance Analysis: Measure and compare execution times for all implementations
- Analyze performance across different image sizes and filter dimensions
- Identify performance bottlenecks using Nsight Systems
- Report and Documentation: Document implementation approaches and optimization techniques
- Analyze results with charts comparing different implementations
- Discuss observed behavior and performance characteristics
- Suggest potential further optimizations

Preferably share the code using GitHub and use the following template:
https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda/blob/main/templates/convolution_filter.cu

A guide with a theoretical background about convolution filters "Image Convolution with CUDA.pdf" can be found on Blackboard
