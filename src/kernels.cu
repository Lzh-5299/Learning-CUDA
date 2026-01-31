#include <vector>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cmath>
#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
__global__ void trace_kernel(const T* d_input, size_t rows, size_t cols, T* d_trace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < min(rows, cols)) {
        int index = idx * cols + idx; // 对角线元素的索引
        atomicAdd(d_trace, d_input[index]); // 原子加法操作，避免并发冲突
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // TODO: Implement the trace function
    T* d_input;
    T* d_trace;
    T trace_sum = 0;

    size_t size = rows * cols * sizeof(T);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_trace, sizeof(T));

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_trace, 0, sizeof(T));

    int block_size = 256;  // 线程块大小
    int num_blocks = (min(rows, cols) + block_size - 1) / block_size;  // 计算块数

    trace_kernel<T><<<num_blocks, block_size>>>(d_input, rows, cols, d_trace);

    cudaMemcpy(&trace_sum, d_trace, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_trace);

    return trace_sum;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

#define BLOCK_SIZE_Q 64
#define BLOCK_SIZE_KV 64

// 类型转换
__device__ __forceinline__ float toFloat(float x) { return x; }
__device__ __forceinline__ float toFloat(half x) { return __half2float(x); }

__device__ __forceinline__ float fromFloat(float x, float*) { return x; }
__device__ __forceinline__ half  fromFloat(float x, half*)  { return __float2half(x); }

// =======================
// Flash Attention Kernel
// =======================
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    float scale
) {
    int batch_idx = blockIdx.z;
    int head_idx  = blockIdx.y;
    int q_block   = blockIdx.x;
    int tid = threadIdx.x;

    int q_start = q_block * BLOCK_SIZE_Q;

    int q_per_kv = query_heads / kv_heads;
    int kv_head_idx = head_idx / q_per_kv;

    extern __shared__ float smem[];

    float* s_q = smem;
    float* s_k = s_q + BLOCK_SIZE_Q * head_dim;
    float* s_v = s_k + BLOCK_SIZE_KV * head_dim;
    float* s_scores = s_v + BLOCK_SIZE_KV * head_dim;
    float* s_m = s_scores + BLOCK_SIZE_Q * BLOCK_SIZE_KV;
    float* s_l = s_m + BLOCK_SIZE_Q;
    float* s_o = s_l + BLOCK_SIZE_Q;

    // 初始化
    for (int i = tid; i < BLOCK_SIZE_Q; i += blockDim.x) {
        s_m[i] = -INFINITY;
        s_l[i] = 0.0f;
    }

    for (int i = tid; i < BLOCK_SIZE_Q * head_dim; i += blockDim.x) {
        s_o[i] = 0.0f;
    }
    __syncthreads();

    // Load Q
    for (int i = tid; i < BLOCK_SIZE_Q * head_dim; i += blockDim.x) {
        int q = i / head_dim;
        int d = i % head_dim;
        int gq = q_start + q;

        if (gq < tgt_seq_len) {
            int idx = batch_idx * tgt_seq_len * query_heads * head_dim +
                      gq * query_heads * head_dim +
                      head_idx * head_dim + d;
            s_q[i] = toFloat(Q[idx]);
        } else {
            s_q[i] = 0.0f;
        }
    }
    __syncthreads();

    int num_kv_blocks = (src_seq_len + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;

    for (int kb = 0; kb < num_kv_blocks; kb++) {
        int kv_start = kb * BLOCK_SIZE_KV;

        if (is_causal && kv_start > q_start + BLOCK_SIZE_Q - 1)
            continue;

        // Load K, V
        for (int i = tid; i < BLOCK_SIZE_KV * head_dim; i += blockDim.x) {
            int k = i / head_dim;
            int d = i % head_dim;
            int gk = kv_start + k;

            if (gk < src_seq_len) {
                int idx = batch_idx * src_seq_len * kv_heads * head_dim +
                          gk * kv_heads * head_dim +
                          kv_head_idx * head_dim + d;
                s_k[i] = toFloat(K[idx]);
                s_v[i] = toFloat(V[idx]);
            } else {
                s_k[i] = 0.0f;
                s_v[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute scores
        for (int i = tid; i < BLOCK_SIZE_Q * BLOCK_SIZE_KV; i += blockDim.x) {
            int q = i / BLOCK_SIZE_KV;
            int k = i % BLOCK_SIZE_KV;

            int gq = q_start + q;
            int gk = kv_start + k;

            float score = -INFINITY;

            if (gq < tgt_seq_len && gk < src_seq_len) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += s_q[q * head_dim + d] *
                           s_k[k * head_dim + d];
                }
                dot *= scale;

                if (!(is_causal && gk > gq))
                    score = dot;
            }
            s_scores[i] = score;
        }
        __syncthreads();

        // Flash update
        for (int q = tid; q < BLOCK_SIZE_Q; q += blockDim.x) {
            int gq = q_start + q;
            if (gq >= tgt_seq_len) continue;

            float m_old = s_m[q];
            float l_old = s_l[q];

            float m_new = m_old;
            for (int k = 0; k < BLOCK_SIZE_KV; k++) {
                m_new = fmaxf(m_new, s_scores[q * BLOCK_SIZE_KV + k]);
            }

            float l_new = l_old * expf(m_old - m_new);
            for (int k = 0; k < BLOCK_SIZE_KV; k++) {
                l_new += expf(s_scores[q * BLOCK_SIZE_KV + k] - m_new);
            }

            for (int d = 0; d < head_dim; d++) {
                float o = s_o[q * head_dim + d] * expf(m_old - m_new);
                for (int k = 0; k < BLOCK_SIZE_KV; k++) {
                    float w = expf(s_scores[q * BLOCK_SIZE_KV + k] - m_new);
                    o += w * s_v[k * head_dim + d];
                }
                s_o[q * head_dim + d] = o;
            }

            s_m[q] = m_new;
            s_l[q] = l_new;
        }
        __syncthreads();
    }

    // Normalize + Write back
    for (int i = tid; i < BLOCK_SIZE_Q * head_dim; i += blockDim.x) {
        int q = i / head_dim;
        int d = i % head_dim;
        int gq = q_start + q;

        if (gq < tgt_seq_len) {
            float val = s_o[i] / s_l[q];

            int idx = batch_idx * tgt_seq_len * query_heads * head_dim +
                      gq * query_heads * head_dim +
                      head_idx * head_dim + d;

            O[idx] = fromFloat(val, (T*)nullptr);
        }
    }
}


template <typename T>
void flashAttention(
    const std::vector<T>& h_q, const std::vector<T>& h_k,
    const std::vector<T>& h_v, std::vector<T>& h_o,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // TODO: Implement the flash attention function
    // 计算输入数据的大小
    size_t q_size = batch_size * tgt_seq_len * query_heads * head_dim;
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = q_size;  // Output 和输入的 Q 大小一致

    // 确保输出数组大小正确
    h_o.resize(o_size);

    // 分配设备内存
    T *d_q, *d_k, *d_v, *d_o;
    float *d_l, *d_m;

    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, kv_size * sizeof(T));
    cudaMalloc(&d_v, kv_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));
    cudaMalloc(&d_l, batch_size * tgt_seq_len * query_heads * sizeof(float));
    cudaMalloc(&d_m, batch_size * tgt_seq_len * query_heads * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);

    // 计算 scale
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // 设置 grid 和 block 的大小
    int num_q_blocks = (tgt_seq_len + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;
    dim3 grid(num_q_blocks, query_heads, batch_size);  // Grid 大小
    dim3 block(128);  // Block 大小：使用 128 个线程

    // 计算共享内存的大小
    size_t smem_size = (BLOCK_SIZE_Q * head_dim +           // s_q
                        BLOCK_SIZE_KV * head_dim +          // s_k
                        BLOCK_SIZE_KV * head_dim +          // s_v
                        BLOCK_SIZE_Q * BLOCK_SIZE_KV +      // s_scores
                        BLOCK_SIZE_Q +                      // s_m
                        BLOCK_SIZE_Q +                      // s_l
                        BLOCK_SIZE_Q * head_dim) * sizeof(float);  // s_o

    // 启动 kernel
    flashAttentionKernel<T><<<grid, block, smem_size>>>(
        d_q, d_k, d_v, d_o, tgt_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal, scale
    );

    // 将结果从设备复制回主机
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_l);
    cudaFree(d_m);
}

template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, std::vector<float>&,
    int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
    const std::vector<half>&, std::vector<half>&,
    int, int, int, int, int, int, bool);
