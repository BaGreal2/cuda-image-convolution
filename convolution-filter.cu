#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_WIDTH 16
#define MAX_FILTER_RADIUS 4
#define SHARED_WIDTH (TILE_WIDTH + 2 * MAX_FILTER_RADIUS)

float boxBlur3x3[9] = {1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

float gaussianBlur5x5[25] = {1 / 273.0f,  4 / 273.0f, 7 / 273.0f, 4 / 273.0f,  1 / 273.0f,  4 / 273.0f,  16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f, 7 / 273.0f, 26 / 273.0f, 41 / 273.0f,
                             26 / 273.0f, 7 / 273.0f, 4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,  1 / 273.0f,  4 / 273.0f,  7 / 273.0f, 4 / 273.0f, 1 / 273.0f};

float trueGaussian1D[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

float trueGaussian2D[25] = {1 / 256.0f,  4 / 256.0f, 6 / 256.0f, 4 / 256.0f,  1 / 256.0f,  4 / 256.0f,  16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f, 6 / 256.0f, 24 / 256.0f, 36 / 256.0f,
                            24 / 256.0f, 6 / 256.0f, 4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,  1 / 256.0f,  4 / 256.0f,  6 / 256.0f, 4 / 256.0f, 1 / 256.0f};

float sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

float sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

float sharpen[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

__constant__ float d_filter[81];
__shared__ float s_tile[SHARED_WIDTH][SHARED_WIDTH + 1];

#define CHECK_CUDA_ERROR(call)                                                                                                                                                                         \
  {                                                                                                                                                                                                    \
    cudaError_t err = call;                                                                                                                                                                            \
    if (err != cudaSuccess) {                                                                                                                                                                          \
      fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__);                                                                                          \
      exit(EXIT_FAILURE);                                                                                                                                                                              \
    }                                                                                                                                                                                                  \
  }

typedef struct {
  unsigned char *data;
  int width;
  int height;
  int channels;
} Image;

bool validateResults(unsigned char *cpuData, unsigned char *gpuData, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (abs((int)cpuData[i] - (int)gpuData[i]) > 2) {
      printf("[ERROR]: Mismatch found.\n CPU: %d | GPU: %d\n", cpuData[i], gpuData[i]);
      return false;
    }
  }
  return true;
}

void convolutionCPU(const Image *input, Image *output, const float *filter, int filterWidth) {

  int halfWidth = filterWidth / 2;

  for (int y = 0; y < input->height; ++y) {
    for (int x = 0; x < input->width; ++x) {
      for (int c = 0; c < input->channels; ++c) {

        float sum = 0.0f;

        for (int ky = -halfWidth; ky <= halfWidth; ++ky) {
          for (int kx = -halfWidth; kx <= halfWidth; ++kx) {

            int neighborY = y + ky;
            int neighborX = x + kx;

            if (neighborY < 0)
              neighborY = 0;
            else if (neighborY >= input->height)
              neighborY = input->height - 1;

            if (neighborX < 0)
              neighborX = 0;
            else if (neighborX >= input->width)
              neighborX = input->width - 1;

            int pixelIndex = (neighborY * input->width + neighborX) * input->channels + c;
            int filterIndex = (ky + halfWidth) * filterWidth + (kx + halfWidth);
            sum += input->data[pixelIndex] * filter[filterIndex];
          }
        }

        if (sum < 0.0f)
          sum = 0.0f;
        if (sum > 255.0f)
          sum = 255.0f;

        int outIndex = (y * input->width + x) * input->channels + c;
        output->data[outIndex] = (unsigned char)sum;
      }
    }
  }
}

__global__ void convolutionKernelUncoalesced(unsigned char *input, unsigned char *output, int filterWidth, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.y;
  int y = blockIdx.y * blockDim.y + threadIdx.x;

  if (x >= width || y >= height)
    return;

  int halfWidth = filterWidth / 2;
  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (int ky = -halfWidth; ky <= halfWidth; ++ky) {
      for (int kx = -halfWidth; kx <= halfWidth; ++kx) {
        int neighborY = y + ky;
        int neighborX = x + kx;
        neighborY = (neighborY < 0) ? 0 : ((neighborY >= height) ? height - 1 : neighborY);
        neighborX = (neighborX < 0) ? 0 : ((neighborX >= width) ? width - 1 : neighborX);

        int pixelIndex = (neighborY * width + neighborX) * channels + c;
        int filterIndex = (ky + halfWidth) * filterWidth + (kx + halfWidth);
        sum += input[pixelIndex] * d_filter[filterIndex];
      }
    }
    if (sum < 0.0f)
      sum = 0.0f;
    if (sum > 255.0f)
      sum = 255.0f;
    output[(y * width + x) * channels + c] = (unsigned char)sum;
  }
}

__global__ void convolutionKernelNaive(unsigned char *input, unsigned char *output, int filterWidth, int width, int height, int channels) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int halfWidth = filterWidth / 2;

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;

    for (int ky = -halfWidth; ky <= halfWidth; ++ky) {
      for (int kx = -halfWidth; kx <= halfWidth; ++kx) {

        int neighborY = y + ky;
        int neighborX = x + kx;

        neighborY = (neighborY < 0) ? 0 : ((neighborY >= height) ? height - 1 : neighborY);
        neighborX = (neighborX < 0) ? 0 : ((neighborX >= width) ? width - 1 : neighborX);

        int pixelIndex = (neighborY * width + neighborX) * channels + c;
        int filterIndex = (ky + halfWidth) * filterWidth + (kx + halfWidth);

        sum += input[pixelIndex] * d_filter[filterIndex];
      }
    }

    if (sum < 0.0f)
      sum = 0.0f;
    if (sum > 255.0f)
      sum = 255.0f;

    int outIndex = (y * width + x) * channels + c;
    output[outIndex] = (unsigned char)sum;
  }
}

__global__ void convolutionKernelShared(unsigned char *input, unsigned char *output, int filterWidth, int width, int height, int channels) {
  __shared__ float s_tile[SHARED_WIDTH][SHARED_WIDTH];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int r = filterWidth / 2;
  int tile_w = blockDim.x + 2 * r;

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int total_threads = blockDim.x * blockDim.y;
  int total_elements = tile_w * tile_w;

  int start_x = blockIdx.x * blockDim.x - r;
  int start_y = blockIdx.y * blockDim.y - r;

  for (int c = 0; c < channels; ++c) {
    for (int i = tid; i < total_elements; i += total_threads) {
      int local_y = i / tile_w;
      int local_x = i % tile_w;

      int global_x = start_x + local_x;
      int global_y = start_y + local_y;

      global_x = (global_x < 0) ? 0 : ((global_x >= width) ? width - 1 : global_x);
      global_y = (global_y < 0) ? 0 : ((global_y >= height) ? height - 1 : global_y);

      int pixelIndex = (global_y * width + global_x) * channels + c;
      s_tile[local_y][local_x] = input[pixelIndex];
    }

    __syncthreads();

    if (x < width && y < height) {
      float sum = 0.0f;

      for (int ky = -r; ky <= r; ++ky) {
        for (int kx = -r; kx <= r; ++kx) {
          float pixel = s_tile[threadIdx.y + r + ky][threadIdx.x + r + kx];
          int filterIndex = (ky + r) * filterWidth + (kx + r);
          sum += pixel * d_filter[filterIndex];
        }
      }

      if (sum < 0.0f)
        sum = 0.0f;
      if (sum > 255.0f)
        sum = 255.0f;

      int outIndex = (y * width + x) * channels + c;
      output[outIndex] = (unsigned char)sum;
    }

    __syncthreads();
  }
}

__constant__ float d_filter_1D[15];

__global__ void convolutionRowKernel(unsigned char *input, unsigned char *output, int filterWidth, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int r = filterWidth / 2;
  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (int kx = -r; kx <= r; ++kx) {
      int neighborX = x + kx;
      neighborX = (neighborX < 0) ? 0 : ((neighborX >= width) ? width - 1 : neighborX);
      int pixelIndex = (y * width + neighborX) * channels + c;
      sum += input[pixelIndex] * d_filter_1D[kx + r];
    }

    output[(y * width + x) * channels + c] = (unsigned char)sum;
  }
}

__global__ void convolutionColKernel(unsigned char *input, unsigned char *output, int filterWidth, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int r = filterWidth / 2;
  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (int ky = -r; ky <= r; ++ky) {
      int neighborY = y + ky;
      neighborY = (neighborY < 0) ? 0 : ((neighborY >= height) ? height - 1 : neighborY);
      int pixelIndex = (neighborY * width + x) * channels + c;
      sum += input[pixelIndex] * d_filter_1D[ky + r];
    }

    if (sum < 0.0f)
      sum = 0.0f;
    if (sum > 255.0f)
      sum = 255.0f;
    output[(y * width + x) * channels + c] = (unsigned char)sum;
  }
}

int main(int argc, char **argv) {
  const char *inputPath = "michael.png";
  const char *outputCpuPath = "michael_output_cpu.png";
  const char *outputGpuNaivePath = "michael_output_naive.png";
  const char *outputGpuSharedPath = "michael_output_shared.png";
  const char *outputGpuSeparablePath = "michael_output_separable.png";
  const char *outputGpuUncoalescedPath = "michael_output_uncoalesced.png";

  Image inputImage;

  inputImage.data = stbi_load(inputPath, &inputImage.width, &inputImage.height, &inputImage.channels, 0);

  if (inputImage.data == NULL) {
    fprintf(stderr, "[ERROR]: Failed to load image %s\n", inputPath);
    return EXIT_FAILURE;
  }

  printf("[LOG]: Loaded %s (%dx%d, %d channels)\n", inputPath, inputImage.width, inputImage.height, inputImage.channels);

  Image outputImage;
  outputImage.width = inputImage.width;
  outputImage.height = inputImage.height;
  outputImage.channels = inputImage.channels;

  size_t imageSize = inputImage.width * inputImage.height * inputImage.channels * sizeof(unsigned char);
  outputImage.data = (unsigned char *)malloc(imageSize);

  unsigned char *cpuValidationData = (unsigned char *)malloc(imageSize);

  float *currentFilter = trueGaussian2D;
  int currentFilterWidth = 5;

  // cpu
  auto startCPU = std::chrono::high_resolution_clock::now();

  convolutionCPU(&inputImage, &outputImage, currentFilter, currentFilterWidth);

  auto endCPU = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;
  printf("[LOG]: CPU Convolution complete in: %.2f ms\n", durationCPU.count());

  stbi_write_png(outputCpuPath, outputImage.width, outputImage.height, outputImage.channels, outputImage.data, outputImage.width * outputImage.channels);
  printf("[LOG]: CPU output written to: %s\n", outputCpuPath);

  memcpy(cpuValidationData, outputImage.data, imageSize);

  // gpu (naive)
  unsigned char *d_input, *d_output;
  size_t filterSize = currentFilterWidth * currentFilterWidth * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, imageSize));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, imageSize));
  CHECK_CUDA_ERROR(cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_filter, currentFilter, filterSize));

  dim3 blockSize(16, 16);
  dim3 gridSize((inputImage.width + blockSize.x - 1) / blockSize.x, (inputImage.height + blockSize.y - 1) / blockSize.y);

  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);

  cudaEventRecord(startGPU);

  convolutionKernelNaive<<<gridSize, blockSize>>>(d_input, d_output, currentFilterWidth, inputImage.width, inputImage.height, inputImage.channels);

  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);

  float millisecondsGPU = 0;
  cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
  printf("[LOG]: GPU (Naive) Convolution complete in: %.2f ms\n", millisecondsGPU);

  CHECK_CUDA_ERROR(cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost));

  stbi_write_png(outputGpuNaivePath, outputImage.width, outputImage.height, outputImage.channels, outputImage.data, outputImage.width * outputImage.channels);
  printf("[LOG]: GPU (Naive) output written to: %s\n", outputGpuNaivePath);

  if (validateResults(cpuValidationData, outputImage.data, imageSize)) {
    printf("[LOG]: Validation succeeded\n");
  } else {
    printf("[ERROR]: Validation failed\n");
  }

  // gpu (unaligned)
  cudaEventRecord(startGPU);

  convolutionKernelUncoalesced<<<gridSize, blockSize>>>(d_input, d_output, currentFilterWidth, inputImage.width, inputImage.height, inputImage.channels);

  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);

  float millisecondsUncoalesced = 0;
  cudaEventElapsedTime(&millisecondsUncoalesced, startGPU, stopGPU);
  printf("[LOG]: GPU (Uncoalesced) Convolution complete in: %.2f ms\n", millisecondsUncoalesced);

  CHECK_CUDA_ERROR(cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost));
  stbi_write_png(outputGpuUncoalescedPath, outputImage.width, outputImage.height, outputImage.channels, outputImage.data, outputImage.width * outputImage.channels);
  printf("[LOG]: GPU (Uncoalesced) output written to: %s\n", outputGpuUncoalescedPath);

  if (validateResults(cpuValidationData, outputImage.data, imageSize)) {
    printf("[LOG]: Validation succeeded\n");
  } else {
    printf("[ERROR]: Validation failed\n");
  }

  // gpu (shared memory)
  cudaEventRecord(startGPU);

  convolutionKernelShared<<<gridSize, blockSize>>>(d_input, d_output, currentFilterWidth, inputImage.width, inputImage.height, inputImage.channels);

  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);

  float millisecondsShared = 0;
  cudaEventElapsedTime(&millisecondsShared, startGPU, stopGPU);
  printf("[LOG]: GPU (Shared memory) Convolution complete in: %.2f ms\n", millisecondsShared);

  CHECK_CUDA_ERROR(cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost));
  stbi_write_png(outputGpuSharedPath, outputImage.width, outputImage.height, outputImage.channels, outputImage.data, outputImage.width * outputImage.channels);
  printf("[LOG]: GPU (Shared memory) output written to: %s\n", outputGpuSharedPath);

  if (validateResults(cpuValidationData, outputImage.data, imageSize)) {
    printf("[LOG]: Validation succeeded\n");
  } else {
    printf("[ERROR]: Validation failed\n");
  }

  // gpu (separable two-pass)
  unsigned char *d_intermediate;
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_intermediate, imageSize));
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_filter_1D, trueGaussian1D, currentFilterWidth * sizeof(float)));

  cudaEventRecord(startGPU);

  convolutionRowKernel<<<gridSize, blockSize>>>(d_input, d_intermediate, currentFilterWidth, inputImage.width, inputImage.height, inputImage.channels);
  convolutionColKernel<<<gridSize, blockSize>>>(d_intermediate, d_output, currentFilterWidth, inputImage.width, inputImage.height, inputImage.channels);

  cudaEventRecord(stopGPU);
  cudaEventSynchronize(stopGPU);

  float millisecondsSeparable = 0;
  cudaEventElapsedTime(&millisecondsSeparable, startGPU, stopGPU);
  printf("[LOG]: GPU (Separble Two-Pass) Convolution complete in: %.2f ms\n", millisecondsSeparable);

  CHECK_CUDA_ERROR(cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost));
  stbi_write_png(outputGpuSeparablePath, outputImage.width, outputImage.height, outputImage.channels, outputImage.data, outputImage.width * outputImage.channels);
  printf("[LOG]: GPU (Separble Two-Pass) output written to: %s\n", outputGpuSeparablePath);

  if (validateResults(cpuValidationData, outputImage.data, imageSize)) {
    printf("[LOG]: Validation succeeded\n");
  } else {
    printf("[ERROR]: Validation failed\n");
  }

  // performance
  printf("\n");
  printf("CPU Baseline:\t1.00x (%.2f ms)\n", durationCPU.count());
  printf("GPU Naive:\t%.2fx faster (%.2f ms)\n", durationCPU.count() / millisecondsGPU, millisecondsGPU);
  printf("GPU Uncoal.:\t%.2fx faster (%.2f ms)\n", durationCPU.count() / millisecondsUncoalesced, millisecondsUncoalesced);
  printf("GPU Shared:\t%.2fx faster (%.2f ms)\n", durationCPU.count() / millisecondsShared, millisecondsShared);
  printf("GPU Separable:\t%.2fx faster (%.2f ms)\n", durationCPU.count() / millisecondsSeparable, millisecondsSeparable);

  // cleanup
  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));
  CHECK_CUDA_ERROR(cudaFree(d_intermediate));
  cudaEventDestroy(startGPU);
  cudaEventDestroy(stopGPU);
  stbi_image_free(inputImage.data);
  free(outputImage.data);
  free(cpuValidationData);

  return 0;
}
