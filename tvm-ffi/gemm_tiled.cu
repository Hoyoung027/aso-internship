#define TILE 16

extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C,
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