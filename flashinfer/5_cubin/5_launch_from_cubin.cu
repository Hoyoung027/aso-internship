// 원본: include/flashinfer/trtllm/fmha/fmhaKernels.cuh
// 역할: cubin bytes를 CUDA Driver API로 로드하고 실행한다.

if (findModuleIter == mModules.end()) {
  std::string cubin_path = tllm_gen_fmha_cubin_path + "/" + kernelMeta.mFuncName + ".cubin";
  // Python callback을 통해 로컬 cache 또는 원격에서 cubin bytes를 가져온다.
  std::string cubin = getCubin(cubin_path, kernelMeta.sha256);
  if (cubin.empty()) {
    throw std::runtime_error("Failed to load cubin for " + kernelName);
  }

  // cubin bytes를 CUDA driver module로 로드하고 cache한다.
  cuErrCheck(cuModuleLoadData(&hmod, cubin.data()));
  mModules[kernelName] = hmod;
}

// 로드된 module 안에서 실제 kernel function handle을 찾는다.
cuErrCheck(cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName));

if (kernelMeta.mSharedMemBytes >= 48 * 1024) {
  cuErrCheck(cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                kernelMeta.mSharedMemBytes));
}

mFunctions[hashId] = funcInfo;

// 준비된 CUfunction을 현재 launch config와 인자로 실행한다.
cuErrCheck(cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));
