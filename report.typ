#set page(paper: "a4", margin: 1in)
#set text(size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[Image Convolution Optimization in CUDA] \
  #v(1em)
  #text(size: 12pt)[Performance Analysis and Lab Report]
]

#v(2em)

= Implementation Approaches

Since image convolution applies the exact same math to every pixel, it's a perfect fit for the GPU. For this assignment, I built four different versions to see how much performance I could squeeze out:

- *CPU Baseline:* A standard sequential C++ version using nested loops to go through the image's rows, columns, and color channels. I handled the edges by clamping the coordinates.
- *GPU Naive:* A straight port of the CPU code to CUDA. Instead of looping over X and Y, I spawned a 2D grid of threads where each thread calculates exactly one pixel. The filter weights are stored in fast `__constant__` memory, but the pixel data is read directly from global memory.
- *GPU Shared Memory:* To cut down on global memory reads, I wrote a tiled version. A 16x16 thread block collaboratively loads an 18x18 "halo" chunk of the image into `__shared__` memory, syncs up, and then calculates the convolution.
- *GPU Separable (Two-Pass):* Using a math trick for Gaussian blurs, I broke the 2D filter into two 1D passes. It reads the rows horizontally, saves to an intermediate array, and then reads the columns vertically. This drops the workload from $O(N^2)$ to $O(2N)$.

= Performance Analysis and Charting

I tested all my implementations on a large 4K image (4000x2407, 3 channels) using a 5x5 Gaussian filter.

The CPU baseline took a massive *6266.30 ms* to finish. Because the CPU was so slow, I left it off the chart below so the GPU times would actually be readable on a normal scale.

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
  caption: [Comparison of my custom CUDA kernels. Shorter bars equal faster execution.],
)

= Memory Access Optimizations

== Coalesced vs. Unaligned Memory
To get good performance on a GPU, memory reads need to be coalesced. By mapping `threadIdx.x` to the image's X-axis, my threads grab contiguous blocks of memory at the same time. To prove how important this is, I intentionally wrote a "bad" kernel where I flipped the axes, forcing the threads to read vertically and breaking the memory alignment.
Looking at the chart, the uncoalesced kernel took *6.97 ms*—over 3 times slower than the standard Naive kernel (2.14 ms), even though it does the exact same amount of math.

== Avoiding Bank Conflicts
Shared memory is broken up into 32 banks. If multiple threads try to read vertically, they can hit the same bank and cause a bottleneck. To prevent this, I padded the shared memory array by adding an extra column `[SHARED_WIDTH + 1]`. This shifts the memory layout so that vertical neighbors end up in different banks.

= Filter Dimension Impact

One of the most interesting things I noticed was how changing the filter size completely changes which kernel is the fastest:
- *With a 5x5 Filter:* I was surprised to see the Naive kernel (2.14 ms) actually beat the Shared Memory kernel (2.20 ms). Modern GPUs have excellent L1/L2 caches, so for a small filter, the hardware handles the overlapping reads automatically. The time it took my Shared kernel to do the integer division math (`/` and `%`) for the tile coordinates ended up outweighing the memory savings.
- *With a 9x9 Filter:* When I cranked the workload up to a 9x9 filter (81 memory reads per pixel), the Shared Memory kernel finally overtook the Naive one (5.51 ms vs 5.76 ms). Because the memory reuse is so much higher, the `__shared__` cache advantage finally beats the math overhead.

= Profiling with Nsight Systems

I ran `nsys profile` on the remote server to see what was actually bottlenecking the program under the hood:
1. *Memory Bandwidth:* The profile showed the uncoalesced kernel eating up 56.1% of the GPU's compute time. Because the memory reads weren't aligned, the hardware couldn't merge the transactions, creating a massive memory bandwidth bottleneck.
2. *PCIe Transfer Overheads:* Looking at the `cuda_api_sum` report, it turns out the actual math isn't the bottleneck anymore. My kernels take barely over a millisecond, but the `cudaMemcpy` calls take significantly longer. Now that the kernels are optimized, just copying the 4K image back and forth over the PCIe bus is taking up the majority of the application's time.

= Future Optimizations

Even though my Separable filter implementation runs over 5,300x faster than my CPU baseline, there are a few more things I could do if I had more time:
1. *Loop Unrolling:* I could add `#pragma unroll` to the inner filter loops to get rid of the loop-control overhead at compile time.
2. *Warp Shuffles:* Instead of using `__shared__` memory at all, I could use `__shfl_sync()` to pass pixel data directly between threads in the same warp using registers.
3. *Process More Pixels per Thread:* Right now, one thread handles one pixel. If I changed it so each thread handles 2 or 4 pixels, it would increase instruction-level parallelism and hide the memory fetch latency a lot better.
