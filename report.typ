#set page(paper: "a4", margin: 1in)
#set text(size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[Image Convolution Optimization with NVIDIA CUDA] \
  #v(1em)
  #text(size: 12pt)[Performance Analysis and Optimization Report]
]

#v(2em)

= Implementation Approaches

Image convolution is inherently parallelizable. Four distinct implementations were developed to explore the performance characteristics of CPU versus GPU execution:

- *CPU Baseline:* A sequential implementation utilizing standard nested loops to iterate over image height, width, and color channels. Edges were handled via clamping.
- *GPU Naive:* A direct port of the CPU logic. Each thread in a 2D grid computes exactly one output pixel, reading filter weights from constant memory and pixel data directly from global memory.
- *GPU Shared Memory:* A tiled approach utilizing a 16x16 thread block to collaboratively load an 18x18 "halo" tile into `__shared__` memory. This heavily reduces redundant global memory reads by caching neighboring pixels.
- *GPU Separable (Two-Pass):* Exploiting the mathematical separability of a Gaussian blur, the $O(N^2)$ 2D convolution was split into two $O(N)$ 1D passes (horizontal, then vertical), significantly reducing the arithmetic workload.

= Performance Analysis and Charting

The following benchmarks were recorded using a 4000x2407 pixel image (3 channels) with a 5x5 filter. The CPU baseline completed in *6266.30 ms*. Due to the extreme difference in execution time, the chart below focuses exclusively on GPU kernel performance.

#figure(
  align(center)[
    #box(
      width: 80%,
      stroke: 0.5pt + luma(200),
      inset: 10pt,
      radius: 4pt,
      [
        #text(weight: "bold")[GPU Execution Times (5x5 Filter, 4000x2407 Image)]
        #v(5pt)
        #grid(
          columns: (100pt, 40pt, 1fr),
          align: (right, center, left),
          row-gutter: 12pt,
          
          [*Uncoalesced*], [6.97 ms], [#rect(width: 6.97 * 25pt, height: 12pt, fill: rgb("E63946"))],
          [*Shared*], [2.20 ms], [#rect(width: 2.20 * 25pt, height: 12pt, fill: rgb("457B9D"))],
          [*Naive*], [2.14 ms], [#rect(width: 2.14 * 25pt, height: 12pt, fill: rgb("1D3557"))],
          [*Separable*], [1.17 ms], [#rect(width: 1.17 * 25pt, height: 12pt, fill: rgb("2A9D8F"))],
        )
      ],
    )
  ],
  caption: [Performance comparison of CUDA implementations. Shorter bars indicate faster execution.],
)

= Memory Access Optimizations

== Coalesced and Unaligned Memory
Proper memory alignment is critical for GPU performance. In the standard kernels, the `threadIdx.x` dimension was mapped to the image width. This allowed warps (groups of 32 threads) to request contiguous memory blocks, resulting in perfectly coalesced reads. To test the impact of this, a deliberately uncoalesced kernel was written where the axes were flipped (mapping `threadIdx.x` to the Y-axis).
As shown in the chart above, the uncoalesced kernel execution time spiked to *6.97 ms*, performing more than 3x slower than the correctly coalesced naive kernel (2.14 ms), despite executing the exact same arithmetic.

== Bank Conflicts and Padding
Shared memory is organized into 32 banks. To prevent bank conflicts when threads access data vertically within the shared memory tile, the array was padded by adding an extra column `[SHARED_WIDTH + 1]`. This shifts the alignment of each subsequent row, ensuring that vertical neighbors reside in different memory banks.

= Filter Dimension Impact

The performance characteristics of the kernels shift dramatically depending on the size of the convolution filter:
- *5x5 Filter:* The Naive kernel (2.14 ms) slightly outperformed the Shared Memory kernel (2.20 ms). Modern GPU architectures feature large L1/L2 caches that automatically optimize overlapping global memory reads for small filters. The overhead of collaborative loading and integer arithmetic (`%` and `/`) in the shared memory kernel outweighed the caching benefits.
- *9x9 Filter:* When the workload was increased to a 9x9 box blur (81 reads per pixel), the Shared Memory kernel (5.51 ms) overtook the Naive kernel (5.76 ms). As filter size grows, the $N^2$ memory reuse heavily favors the explicitly managed `__shared__` cache.

= Profiling with Nsight Systems

A profile generated using `nsys` identified several key system bottlenecks:
1. *Memory Bandwidth:* The uncoalesced kernel consumed 56.1% of the total recorded GPU compute time, proving that poor memory alignment causes a severe bandwidth bottleneck as the hardware fails to merge transactions.
2. *Compute and Transfer:* The `cuda_api_sum` report revealed that memory allocation (`cudaMalloc`) and host-to-device transfers (`cudaMemcpy`) heavily dominated the application's overall lifecycle. The actual separated kernel executions took ~0.57 ms each, while `cudaMemcpy` operations consumed millions of nanoseconds. Once the compute kernels are fully optimized, PCIe transfer speeds become the ultimate system bottleneck.

= Suggested Further Optimizations

While the Separable Filter provides excellent performance (over 5000x faster than the CPU), further optimizations could include:
1. *Loop Unrolling:* Applying `#pragma unroll` to the inner filter loops to eliminate loop-control overhead at the compiler level.
2. *Warp Shuffle Instructions:* Replacing the `__shared__` memory implementation with `__shfl_sync()` intrinsics. This allows threads within the same warp to share data directly via registers, eliminating shared memory latency and the risk of bank conflicts entirely.
3. *Increased Thread Granularity:* Modifying the kernel so that each thread computes 2 or 4 pixels instead of 1. This increases instruction-level parallelism and better hides memory fetch latency.
