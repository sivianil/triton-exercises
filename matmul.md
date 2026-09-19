## Matrix Multiplication [Triton]
- High Performance FP16 matmul kernel achieves performance on-par with cuBLA or rocBLAS.
- $$A_{m \times k} \times B_{k \times n} = C_{m \times n}$$
1. What does one triton program compute? -> $C_{block_m \times block_n}$
```
for every output block C[m, n]:
  acc = 0
  # K dimension accumulated over multiple iterations
  for k:
    A_block = A[m : m + block_m, k : k + block_k]
    B_block = B[k : k + block_k, n : n + block_n]
    acc += A_block @ B_block
  C[m : m + block_m, n : n + block_n] = acc
```
- Let's consider BLOCK_M = 128, BLOCK_N = 256, BLOCK_K = 64. A single triton program computes C tile of shape (128, 256) by iterating over the k dimension
  of A tile (128, 64) and B tile (64, 256).
  
2. Multi-dimesional pointer Arithmetic
```
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

### Dissect ugly pid -> pid_m, pid_n L2 grouping formula
```
       col_0   col_1  col_2  col_3     Output C matrix
row_0    0       1      2      3    ->   C(0,0)  C(0,1)  C(0,2)  C(0,3)  
row_1    4       5      6      7    ->   C(1,0)  C(1,1)  C(1,2)  C(1,3)
row_2    8       9      10     11   ->   C(2,0)  C(2,1)  C(2,2)  C(2,3)
row_3    12     13      14     15   ->   C(3,0)  C(3,1)  C(3,2)  C(3,3)
```

- Take linear program id P_id, convert it into (pid_m, pid_n) coordinates, to visit output tiles in a cache-friendly manner.
- Let's start with a tiny 16 x 16 matrix, C = A x B (all are of shape 16 x 16).
- BLOCK_SIZE_M, BLOCK_SIZE_N = 4,4; GROUP_SIZE_M = 2
- Output C gets divided into 4 x 4 tiles (16 / BLOCK_SIZE_M, 16 / BLOCK_SIZE_N)
- Here each program computes 4 x 4 tile of C. total no of programs = 16
#### Find number of program ids along M and N axes
```
num_pid_m = ceil(M, BLOCK_SIZE_M) = 4
num_pid_n = ceil(N, BLOCK_SIZE_N) = 4
```
#### What would normal row-major ordering do?
```
pid = tl.program_id(axis=0); grid_n = 4
pid_m = pid // grid_n; pid_n = pid % grid_n

pid     pid_m    pid_n     C tile 
 0        0        0         C(0,0)
 1        0        1         C(0,1)
 2        0        2         C(0,2)
 3        0        3         C(0,3)
 4        1        0         C(1,0) 
 5        1        1         C(1,1)
 6        1        2         C(1,2)
 7        1        3         C(1,3)
 .        .        .           .
 .        .        .           .
 .        .        .           .
 14       3        2         C(3,2)
 15       3        3         C(3,3)
```
- Execution goes in order C(0, 0) -> C(0, 1) -> C(0,2) -> C(0,3) -> C(1,0) -> C(1,1)............
  
#### Find the program id we're currently in 
- pid = tl.program_id(axis=0)
- Triton only knows that I'm program #15. It doesn't yet know that I should calculate C tile (3, 3)
#### Compute number of M tiles
- num_pid_m = num_pid_n = 4

#### Compute number of programs in a group
```
num_pid_in_group = GROUP_SIZE_M * num_pid_n. # a group contains 8 programs (2 * 4)
```

#### Find the group id of the program we're currently working on
```
group_id = pid // num_pid_in_group
pid     pid // 8  group_id  
 0        0          0    
 1        0          0  
 2        0          0  
 3        0          0 
 4        0          0     
 5        0          0    
 6        0          0     
 7        0          0 
 .        .          . 
 .        .          . 
 .        .          .    
 14       1          1  
 15       1          1  

Total 16 programs
pid      : 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
group_id : 0 0 0 0 0 0 0 0 | 1 1 1  1  1   1  1  1
```

#### Compute the row id of the first program in the group
```
first_pid_m = group_id * GROUP_SIZE_M
Group 0 starts at M tile 0 (pid = 0)
Group 1 starts at M tile 2 (pid = 8)
```

#### Compute group_size_m if num_pid_m is not divisible by GROUP_SIZE_M
```
group_size_m = min(num_pid_m - first_pid_m , GROUP_SIZE_M)
In our case num_pid_m = 4, first_pid_m = 2
group_size_m = min(4 - 2, 2) = 2
Imgine num_pid_m = 5 
Group 0 -> M tiles 0, 1
Group 1 -> M tiles 2, 3
Group 2 -> M tile 4
```

#### Determine pid_m
```
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid = 15, first_pid_m = 2
pid_m = 2 + ((15 % 8) % 2) = 3
pid_n = (pid % num_pid_in_group) // group_size_m
pid_n = (15 % 8) // 2 = 3
pid -> (pid_m, pid_n)
15 ->  (3,       3)
pid     : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
pid_m   : 0 1 0 1 0 1 0 1 2 3 2  3  2  3  2  3
pid_n   : 0 0 1 1 2 2 3 3 0 0 1  1  2  2  3  3
Col-major ordering
                    pid_n
         col_0   col_1  col_2  col_3     Output C matrix
  row_0    0       2      4      6    ->   C(0,0)  C(0,1)  C(0,2)  C(0,3)  
  row_1    1       3      5      7    ->   C(1,0)  C(1,1)  C(1,2)  C(1,3)
  row_2    8       10     12     14   ->   C(2,0)  C(2,1)  C(2,2)  C(2,3)
  row_3    9       11     13     15   ->   C(3,0)  C(3,1)  C(3,2)  C(3,3)
```

3. a_ptrs contiguous A[M, K]
- stride_am = k, stride_ak = 1
- offs_am[:, None] -> (BLOCK_SIZE_M, 1)
- offs_ak[None, :] -> (1, BLOCK_SIZE_K)
- Two operations broadcasted together makes the shape (BLOCK_SIZE_M, BLOCK_SIZE_K)

4. Why multiply by strides?
- Fundamental pointer arithmetic idea A[i, j]
- $$A_{base} + i * stride_{i} + j * stride_{j}$$
- offs_am * stride_am moves vertically through rows
- offs_ak * stride_ak moves horizontally through columns
- a_ptr + row_offset + col_offset provides address of every element

5. b_ptrs B[K, N]
- stride_bk = n, stride_bn = 1
- row_offset = offs_k[:, None] * stride_bk
- col_offset = offs_bn[None, :] * stride_bn
- b_ptrs = b_ptr + row_offset + col_offset produces (BLOCK_SIZE_K, BLOCK_SIZE_N)
- $$A_{BM \times BK} \times B_{BK \times BN} = C_{BM \times BN}$$

6. The K Loop
```
for k in range(0, K, BLOCK_SIZE_K) moves
# K = 4096, BLOCK_SIZE_K = 64 (0, 64, 128, ............, 4032)
# After each iteration
a_ptrs += BLOCK_SIZE_K * stride_ak
b_ptrs += BLOCK_SIZE_K * stride_bk
# stride_ak = 1, stride_bk = N
a_ptrs += 64, b_ptrs += 64 * N   # Moving K slice down B means jump 64*N elements
```

7. `tl.dot` enters
```
acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
acc += tl.dot(a, b)
```
8. Interesting Part: L2 Optimization

