# Cache-Friendly Partitioning of CNN Computation
This is a quick project idea note.  
There are various partitioning schemes in CNN execution for multi-core systems. They can be roughly divided into two categories: inter-layer & intra-layer.  
In this idea, we are going to focus on intra-layer partitioning which, when done properly, can effectively reduce latency. Intra-layer partitioning utilizes data-level parallelism and distributes data to multiple cores. There are basically 2 ways of doing it.  

1. Filter partitioning/ output channel partitioning: each core takes the whole input, but takes only part of the filter
2. Input partitiong/ Tiling: each core takes the whole filter, but takes only part of the input

The idea is to apply different partitioning schemes for each layer. Using input partitioning when input size is large and filter partitioning when filter size is larger. This can be done online and can be easily applied to various models and various multi-core systems. Benefit of this method is that it can minimize memory usage of each core, resulting in more cache-friendly operation.  
Some exceptions are depthwise convolution, where one can take only part of the input using filter partitioning, and pooling, where there is no filter. However, this method is effective for standard convolution filters, pointwise filters, etc.  

Example: Moblienet V1 (UINT8 Quantization), results are floored down.  
|           | Input |     |      |           | Filter |   |      |      |           |
|-----------|-------|-----|------|-----------|--------|---|------|------|-----------|
|           | H     | W   | D    | size(KiB) | H      | W | D    | C    | size(KiB) |
| Conv2     | 224   | 224 | 3    | 147       | 3      | 3 | 3    | 32   | 0         |
| DW        | 112   | 112 | 32   | 392       | 3      | 3 | 1    | 32   | 0         |
| *Point     | 112   | 112 | 32   | 392       | 1      | 1 | 32   | 64   | 2         |
| DW2       | 112   | 112 | 64   | 784       | 3      | 3 | 1    | 64   | 0         |
| Point     | 56    | 56  | 64   | 196       | 1      | 1 | 64   | 128  | 8         |
| DW        | 56    | 56  | 128  | 392       | 3      | 3 | 1    | 128  | 1         |
| Point     | 56    | 56  | 128  | 392       | 1      | 1 | 128  | 128  | 16        |
| DW2       | 56    | 56  | 128  | 392       | 3      | 3 | 1    | 128  | 1         |
| Point     | 28    | 28  | 128  | 98        | 1      | 1 | 128  | 256  | 32        |
| DW        | 28    | 28  | 256  | 196       | 3      | 3 | 1    | 256  | 2         |
| Point     | 28    | 28  | 256  | 196       | 1      | 1 | 256  | 256  | 64        |
| DW2       | 28    | 28  | 256  | 196       | 3      | 3 | 1    | 256  | 2         |
| Point     | 14    | 14  | 256  | 49        | 1      | 1 | 256  | 512  | 128       |
| DW        | 14    | 14  | 512  | 98        | 3      | 3 | 1    | 512  | 4         |
| Point     | 14    | 14  | 512  | 98        | 1      | 1 | 512  | 512  | 256       |
| DW        | 14    | 14  | 512  | 98        | 3      | 3 | 1    | 512  | 4         |
| Point     | 14    | 14  | 512  | 98        | 1      | 1 | 512  | 512  | 256       |
| DW        | 14    | 14  | 512  | 98        | 3      | 3 | 1    | 512  | 4         |
| Point     | 14    | 14  | 512  | 98        | 1      | 1 | 512  | 512  | 256       |
| DW        | 14    | 14  | 512  | 98        | 3      | 3 | 1    | 512  | 4         |
| Point     | 14    | 14  | 512  | 98        | 1      | 1 | 512  | 512  | 256       |
| DW        | 14    | 14  | 512  | 98        | 3      | 3 | 1    | 512  | 4         |
| Point     | 14    | 14  | 512  | 98        | 1      | 1 | 512  | 512  | 256       |
| DW2       | 14    | 14  | 512  | 98        | 3      | 3 | 1    | 512  | 4         |
| **Point     | 7     | 7   | 512  | 24        | 1      | 1 | 512  | 1024 | 512       |
| DW        | 7     | 7   | 1024 | 49        | 3      | 3 | 1    | 1024 | 9         |
| Point     | 7     | 7   | 1024 | 49        | 1      | 1 | 1024 | 1024 | 1024      |
| Pool      | 7     | 7   | 1024 | 49        |        |   |      |      | 0         |
| FC        | 1     | 1   | 1024 | 1         | 1      | 1 | 1024 | 1000 | 1000      |

Example: cache organization of Snapdragon 855 (Source: WikiChip)
```
Quad-core cluster [Cortex-A75]
L1$ 512 KiB
	L1I$ 256 KiB 4x64 KiB 4-way set associative
	L1D$ 256 KiB 4x64 KiB 4-way set associative
L2$ 1 MiB 4x256 KiB 8-way set associative

Quad-core cluster [Cortex-A55]
L1$ 512 KiB
	L1I$ 256 KiB 4x64 KiB 2-way set associative
	L1D$ 256 KiB 4x64 KiB 4-way set associative
L2$ 512 KiB 4x128 KiB 8-way set associative

2 MiB L3
```

Suppose using this scheme to the first pointwise layer(marked with *) with 4 cores. Input is 401kB and filter is 2kB. Then, using input partitioning needs ```392 / 4 + 2 = 100 KiB``` and this can fit into L2 caches of Snapdragon 845 cores.  
Now, suppose using this scheme to the second to the last pointwise layer(marked with **) with 4 cores. Input is 25kB and filter is 524 kB. Then, using filter partitioning needs ```24 + 512 / 4 = 152 kB``` and it can fit into big cores' L2 cache.  
Still when parameters for a layer is too large, this scheme can guarantee minimum usage of memory compared to using only one partitioning scheme.  
