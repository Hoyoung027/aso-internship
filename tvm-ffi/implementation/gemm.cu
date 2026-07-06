// gemm.cu - 단순 GEMM(행렬곱) CUDA 커널 학습 예제
//
// 계산하는 것:  C = A * B
//   A: (M x K),  B: (K x N),  C: (M x N)
//   모두 row-major(행 우선) float 행렬이라고 가정한다.
//
// 두 가지 버전을 담았다:
//   1) gemm_naive        : 가장 단순한 버전 (이해용)
//   2) gemm_tiled        : shared memory 타일링 버전 (조금 더 빠름)
//
// 빌드:   nvcc -arch=sm_86 -o gemm gemm.cu
// 실행:   ./gemm

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// CUDA 에러를 체크하는 매크로 (실패하면 어디서 죽었는지 알려준다)
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// 1) 가장 단순한 GEMM 커널
//    스레드 1개가 결과 C의 원소 1개를 계산한다.
//    C[row][col] = sum_k A[row][k] * B[k][col]
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    // 이 스레드가 담당할 C의 좌표 (row, col)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N 방향
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M 방향

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // row-major 인덱싱: A[row][k] = A[row*K + k]
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// 2) shared memory 타일링 GEMM 커널
//    블록 단위로 A, B의 타일을 shared memory에 올려놓고 재사용 → 글로벌 메모리
//    접근 횟수를 줄여서 더 빠르다. (GPU 커널 최적화의 기본 패턴)
// ---------------------------------------------------------------------------
#define TILE 16

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {


    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for(int t = 0; t < (K + TILE - 1) / TILE; ++t){

        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // A 타일 로드
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        // B 타일 로드
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();


        // 타일 내부에서 부분합 계산
        for(int k = 0; k < TILE; ++k){
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// CPU 참조 구현 (정답 검증용)
// ---------------------------------------------------------------------------
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// ---------------------------------------------------------------------------
// 커널 실행 시간을 재는 헬퍼
//   - cudaEvent로 GPU 시간만 정확히 측정 (CPU 타이머는 비동기라 부정확)
//   - warmup으로 첫 실행 오버헤드(JIT, 캐시 등)를 제외
//   - iters번 평균을 내서 안정적인 값 산출
//   - 반환값: 1회 실행 평균 시간(ms)
// kind: 0 = naive, 1 = tiled
// ---------------------------------------------------------------------------
float benchmark(int kind, const float* dA, const float* dB, float* dC,
                int M, int N, int K, dim3 grid, dim3 block,
                int warmup, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup (시간 측정 안 함)
    for (int i = 0; i < warmup; ++i) {
        if (kind == 0) gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        else           gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 본 측정
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        if (kind == 0) gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
        else           gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));  // stop 이벤트까지 끝나길 대기

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));  // 두 이벤트 사이 ms

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;  // 1회 평균
}

int main() {
    // 행렬 크기 (작게 시작; 나중에 키워서 성능 비교 가능)
    const int M = 1024, N = 1024, K = 1024;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    // 호스트(CPU) 메모리 할당
    float* hA = (float*)malloc(bytesA);
    float* hB = (float*)malloc(bytesB);
    float* hC = (float*)malloc(bytesC);       // GPU 결과
    float* hRef = (float*)malloc(bytesC);     // CPU 정답

    // 입력을 적당한 값으로 초기화
    for (int i = 0; i < M * K; ++i) hA[i] = (float)((i % 13) - 6) * 0.1f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)((i % 7) - 3) * 0.2f;

    // 디바이스(GPU) 메모리 할당
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    // 입력을 CPU -> GPU 복사
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // 실행 구성: 블록 16x16, 그리드는 N, M을 덮도록
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // --- naive 커널 실행 ---
    gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 결과 GPU -> CPU 복사
    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    // CPU 정답과 비교
    gemm_cpu(hA, hB, hRef, M, N, K);
    double maxErr = 0.0;
    for (int i = 0; i < M * N; ++i)
        maxErr = fmax(maxErr, fabs(hC[i] - hRef[i]));
    printf("[gemm_naive] max abs error = %.6e  -> %s\n",
           maxErr, (maxErr < 1e-3) ? "PASS" : "FAIL");

    // --- tiled 커널 실행 ---
    gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    maxErr = 0.0;
    for (int i = 0; i < M * N; ++i)
        maxErr = fmax(maxErr, fabs(hC[i] - hRef[i]));
    printf("[gemm_tiled] max abs error = %.6e  -> %s\n",
           maxErr, (maxErr < 1e-3) ? "PASS" : "FAIL");

    // ----- 속도 비교 -----
    // GEMM의 총 연산량(FLOPs) = 곱셈 M*N*K + 덧셈 M*N*K = 2*M*N*K
    double flops = 2.0 * M * N * K;

    float ms_naive = benchmark(0, dA, dB, dC, M, N, K, grid, block, 5, 50);
    float ms_tiled = benchmark(1, dA, dB, dC, M, N, K, grid, block, 5, 50);

    // GFLOPS = 초당 10억 번 연산. (flops / 시간(초)) / 1e9
    double gflops_naive = flops / (ms_naive / 1e3) / 1e9;
    double gflops_tiled = flops / (ms_tiled / 1e3) / 1e9;

    printf("\n=== 속도 비교 (M=N=K=%d, 50회 평균) ===\n", M);
    printf("[gemm_naive] %8.4f ms   %8.1f GFLOPS\n", ms_naive, gflops_naive);
    printf("[gemm_tiled] %8.4f ms   %8.1f GFLOPS\n", ms_tiled, gflops_tiled);
    printf("-> tiled가 naive보다 %.2f배 빠름\n", ms_naive / ms_tiled);

    // 정리
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}
