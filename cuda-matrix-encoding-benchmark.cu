/**
 * (C)Tsubasa Kato - 2025/6/3 - Made with help of ChatGPT o3 & Perplexity Pro
 * cuda-matrix-encoding-benchmark.cu  ── fixed & tuned version
 *   • matrix-encode kernel unchanged except for minor tidy-up
 *   • compress kernel now:
 *         – uses 64-bit-safe atomicAdd
 *         – allocates the **exact** number of output bytes
 *         – respects grid size limits by raising the per-block thread count
 *   • percentile routine now sorts doubles correctly
 *   • miscellaneous safety / UX improvements
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <time.h>

#define MAT_SIZE     128
#define LAYER_SIZE   (MAT_SIZE * MAT_SIZE)
#define BLOCK_SIZE   16
#define ITERATIONS   100
#define MIN_RUNTIME  60      // seconds
#define MAX_RUNTIME  300
#define MAX_LAYERS   1024

// ───────────────────────────────────────────────────────────── kernels
__global__ void matrix_encode_kernel(const char *__restrict__ input,
                                     size_t          input_len,
                                     uint8_t        *d_matrix,
                                     size_t          layer_count)
{
    const uint32_t x     = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t y     = threadIdx.y + blockDim.y * blockIdx.y;
    const uint32_t layer = blockIdx.z;

    if (x >= MAT_SIZE || y >= MAT_SIZE || layer >= layer_count) return;

    const size_t pos        = static_cast<size_t>(layer) * LAYER_SIZE
                            + static_cast<size_t>(y)     * MAT_SIZE + x;

    d_matrix[pos] = (pos < input_len) ? static_cast<uint8_t>(input[pos]) : 0u;
}

/* Utility: 64-bit safe atomicAdd for size_t */
__device__ __forceinline__
size_t atomicAdd_size_t(size_t *addr, size_t val) {
    return static_cast<size_t>(
        atomicAdd(reinterpret_cast<unsigned long long *>(addr),
                  static_cast<unsigned long long>(val)));
}

/**
 * Very simple block-local run-length encoder.
 * One warp (thread 0…31) loads a 1 KiB tile to shared memory,
 *   and thread 0 of the block emits the compressed stream.
 * The algorithm is intentionally naïve – it exists to benchmark
 * atomic traffic, not to be a production RLE.
 */
__global__ void compress_kernel(const uint8_t *__restrict__ d_matrix,
                                size_t                    total_size,
                                uint8_t                  *d_out,
                                size_t                   *d_out_size)
{
    extern __shared__ uint8_t sh[];
    const uint32_t tid        = threadIdx.x;
    const uint32_t blockStart = (blockDim.x * blockIdx.x);

    if (blockStart + tid < total_size)  sh[tid] = d_matrix[blockStart + tid];
    __syncthreads();

    // Only one thread per block performs the tiny sequential RLE
    if (tid == 0) {
        uint8_t  cur        = sh[0];
        uint16_t run        = 1;
        size_t   out_bytes  = 0;

        // First pass: figure out how many bytes we will need
        for (uint32_t i = 1; i < blockDim.x && blockStart + i < total_size; ++i) {
            if (sh[i] == cur && run < 65535) { ++run; }
            else {
                out_bytes += (run > 255) ? 4 : 3;
                cur = sh[i]; run = 1;
            }
        }
        out_bytes += (run > 255) ? 4 : 3;                    // final run

        // Obtain output slice
        const size_t out_pos = atomicAdd_size_t(d_out_size, out_bytes);
        uint8_t     *out     = d_out + out_pos;

        // Second pass: actually emit
        cur = sh[0]; run = 1;
        auto emit = [&](uint8_t c, uint16_t r) {
            *out++ = c;
            if (r > 255) {
                *out++ = 3;             // flag: 2-byte run
                *out++ = r >> 8;
                *out++ = r & 0xFF;
            } else {
                *out++ = 0;             // flag: 1-byte run
                *out++ = r & 0xFF;
            }
        };

        for (uint32_t i = 1; i < blockDim.x && blockStart + i < total_size; ++i) {
            if (sh[i] == cur && run < 65535) { ++run; }
            else { emit(cur, run); cur = sh[i]; run = 1; }
        }
        emit(cur, run);                                   // final run
    }
}

// ───────────────────────────────────────────────────────────── helpers
inline void ck(cudaError_t e, const char *m) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e)); exit(EXIT_FAILURE); }
}

double cuda_seconds(cudaEvent_t s, cudaEvent_t t) {
    float ms = 0.f; ck(cudaEventElapsedTime(&ms, s, t), "cudaEventElapsedTime");
    return ms * 1e-3;
}

double host_seconds() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db);
}
double percentile(double *arr, size_t n, int pct) {
    qsort(arr, n, sizeof(double), cmp_double);
    size_t i = (size_t)((pct / 100.0) * (double)n);
    if (i >= n) i = n - 1; return arr[i];
}

char *read_file(const char *fn, size_t *sz) {
    FILE *fp = fopen(fn, "rb"); if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END); *sz = ftell(fp); rewind(fp);
    char *buf = (char *)malloc(*sz); if (!buf) { fclose(fp); return nullptr; }
    fread(buf, 1, *sz, fp); fclose(fp); return buf;
}
void write_blob(const void *p, size_t n, const char *fn) {
    FILE *f = fopen(fn, "wb"); if (!f) return; fwrite(p, 1, n, f); fclose(f);
}

// ───────────────────────────────────────────────────────────── main
int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr,"usage: %s <file>\n", argv[0]); return 1; }

    size_t in_sz; char *h_in = read_file(argv[1], &in_sz);
    if (!h_in) { perror("read_file"); return 1; }
    printf("read %zu B\n", in_sz);

    const size_t layers = (in_sz + LAYER_SIZE - 1) / LAYER_SIZE;
    if (layers > MAX_LAYERS) { fprintf(stderr,"too big (>%d layers)\n", MAX_LAYERS); return 1; }

    /* ── device allocations ─────────────────────────────────────────── */
    char    *d_in;        ck(cudaMalloc(&d_in, in_sz), "d_in");
    uint8_t *d_mat;       ck(cudaMalloc(&d_mat, layers * LAYER_SIZE), "d_mat");
    uint8_t *d_out;       ck(cudaMalloc(&d_out, in_sz * 2), "d_out");      // worst-case
    size_t  *d_out_sz;    ck(cudaMalloc(&d_out_sz, sizeof(size_t)), "d_out_sz");

    ck(cudaMemcpy(d_in, h_in, in_sz, cudaMemcpyHostToDevice), "copy in");

    /* ── launch config ──────────────────────────────────────────────── */
    dim3 blk(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((MAT_SIZE+blk.x-1)/blk.x,
              (MAT_SIZE+blk.y-1)/blk.y,
               layers);

    const int c_threads   = 1024;                                  // 1 KiB tile
    dim3 c_blk(c_threads);
    dim3 c_grid((layers*LAYER_SIZE + c_threads - 1)/c_threads);

    cudaEvent_t e0,e1; ck(cudaEventCreate(&e0),"e0"); ck(cudaEventCreate(&e1),"e1");

    double enc_t[ITERATIONS], cmp_t[ITERATIONS], io_t[ITERATIONS];
    double total = 0; size_t iters=0;

    /* ── warm-up ────────────────────────────────────────────────────── */
    for (int i=0;i<5;++i){
        matrix_encode_kernel<<<grid,blk>>>(d_in,in_sz,d_mat,layers);
        ck(cudaDeviceSynchronize(),"warm-encode");

        size_t zero=0; ck(cudaMemcpy(d_out_sz,&zero,sizeof(size_t),cudaMemcpyHostToDevice),"clr");
        compress_kernel<<<c_grid,c_blk,c_threads>>>(d_mat,layers*LAYER_SIZE,d_out,d_out_sz);
        ck(cudaDeviceSynchronize(),"warm-cmp");
    }

    /* ── benchmark ─────────────────────────────────────────────────── */
    for (; iters<ITERATIONS && total<MAX_RUNTIME; ++iters) {
        // encode
        ck(cudaEventRecord(e0), "rec e0");
        matrix_encode_kernel<<<grid,blk>>>(d_in,in_sz,d_mat,layers);
        ck(cudaEventRecord(e1), "rec e1"); ck(cudaEventSynchronize(e1),"sync e1");
        enc_t[iters]=cuda_seconds(e0,e1);

        // compress
        size_t zero=0; ck(cudaMemcpy(d_out_sz,&zero,sizeof(size_t),cudaMemcpyHostToDevice),"clr sz");
        ck(cudaEventRecord(e0),"rec2 e0");
        compress_kernel<<<c_grid,c_blk,c_threads>>>(d_mat,layers*LAYER_SIZE,d_out,d_out_sz);
        ck(cudaEventRecord(e1),"rec2 e1"); ck(cudaEventSynchronize(e1),"sync2 e1");
        cmp_t[iters]=cuda_seconds(e0,e1);

        // pseudo I/O
        double h0=host_seconds();
        write_blob(h_in,in_sz,"/dev/null");            // discard
        io_t[iters]=host_seconds()-h0;

        total += enc_t[iters]+cmp_t[iters]+io_t[iters];
        if (total>=MIN_RUNTIME) break;
    }

    /* ── stats ─────────────────────────────────────────────────────── */
    auto avg=[&](double *a){ double s=0; for(size_t i=0;i<iters;++i)s+=a[i]; return s/iters; };
    printf("\nresults (%zu iters, %.1fs)\n", iters,total);
    printf(" encode  avg %.6fs  p50 %.6fs  p90 %.6fs\n",
           avg(enc_t), percentile(enc_t,iters,50), percentile(enc_t,iters,90));
    printf(" cmp     avg %.6fs  p50 %.6fs  p90 %.6fs\n",
           avg(cmp_t), percentile(cmp_t,iters,50), percentile(cmp_t,iters,90));
    printf(" I/O     avg %.6fs  p50 %.6fs  p90 %.6fs\n",
           avg(io_t),  percentile(io_t,iters,50),  percentile(io_t,iters,90));
    printf(" throughput: GPU-pipeline %.2f MB/s | host-I/O %.2f MB/s\n",
           in_sz*1e-6 /(avg(enc_t)+avg(cmp_t)),
           in_sz*1e-6 /avg(io_t));

    /* ── cleanup ───────────────────────────────────────────────────── */
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_in); cudaFree(d_mat); cudaFree(d_out); cudaFree(d_out_sz);
    free(h_in);
    return 0;
}
