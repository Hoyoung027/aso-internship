// launcher.cpp - Python ↔ cubin 사이의 연결을 담당
//   cubin 안의 gemm_tiled 커널을 찾아 실행하고, Python에 mod.gemm 으로 노출

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>          // TensorView
#include <tvm/ffi/error.h>                      // TVM_FFI_ICHECK_*
#include <tvm/ffi/extra/c_env_api.h>            // TVMFFIEnvGetStream
#include <tvm/ffi/extra/cuda/cubin_launcher.h>  // EMBED_CUBIN 매크로
#include <tvm/ffi/function.h>                   // DLL_EXPORT_TYPED_FUNC

#define TILE 16

// cubin 모듈 선언. 이름(식별자, 따옴표X)
TVM_FFI_EMBED_CUBIN(gemm_tiled_cubin);

// Python에 노출할 함수.  C = A @ B
void Gemm(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    // (a) cubin에서 "gemm_tiled" 커널을 찾아 핸들로 캐싱.
    //     첫 인자 = 모듈(식별자), 둘째 = 커널 이름(문자열). static 이라 최초 1회만 탐색.
    static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(gemm_tiled_cubin, "gemm_tiled");

    // (b) 텐서 모양에서 M, K, N 추출.  A:(M,K)  B:(K,N)  C:(M,N)
    //     커널 인자가 int 라서 int 변수에 담아둔다(아래에서 주소를 넘겨야 하므로).
    int M = static_cast<int>(A.shape()[0]);
    int K = static_cast<int>(A.shape()[1]);
    int N = static_cast<int>(B.shape()[1]);

    // (c) GPU 메모리 주소(데이터 포인터)
    void* a_ptr = A.data_ptr();
    void* b_ptr = B.data_ptr();
    void* c_ptr = C.data_ptr();

    // (d) 커널 인자 배열. 
    //     커널 시그니처: gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) 
    void* args[] = {&a_ptr, &b_ptr, &c_ptr, &M, &N, &K};

    // (e) 실행 구성. gemm.cu 의 main 과 동일.
    tvm::ffi::dim3 block(TILE, TILE);
    tvm::ffi::dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // (f) 현재 디바이스의 CUDA 스트림을 얻는다(torch 스트림과 호환)
    DLDevice dev = A.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));


    // (g) 커널 실행
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
        kernel.Launch(args, grid, block, stream));
}

// 4) Python에 mod.gemm 으로 노출 (첫 인자 = 파이썬에서 쓸 이름)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, Gemm);
