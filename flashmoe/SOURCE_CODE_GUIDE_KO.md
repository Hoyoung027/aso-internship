# FlashMoE 소스 코드 가이드: Persistent Kernel, Task, Doorbell, Interrupt

이 문서는 FlashMoE의 핵심 실행 구조를 이해하기 위한 입문용 문서다. 특히 다음 파일과 개념을 중심으로 설명한다.

- `csrc/include/flashmoe/moe.cuh`: 하나의 persistent kernel 안에서 CTA의 역할을 나누는 진입점
- `csrc/include/flashmoe/os.cuh`: subscriber와 scheduler를 실행하는 device-side OS
- `csrc/include/flashmoe/infra/task.cuh`: 실행할 연산을 표현하는 `Task`
- `csrc/include/flashmoe/infra/tq.cuh`: processor에 전달하는 `TQSignal`
- `csrc/include/flashmoe/scheduler.cuh`: 준비된 Task를 processor에 할당
- `csrc/include/flashmoe/processor.cuh`: doorbell을 기다리고 실제 Task를 실행

## 1. 전체 구조

FlashMoE의 핵심 MoE 연산은 다음 단계를 하나의 CUDA kernel 안에서 처리한다.

```text
Token dispatch
    -> Expert GEMM0 (up projection + activation/gating)
    -> Expert GEMM1 (down projection)
    -> Combine
```

다만 모든 CTA가 같은 작업을 하는 것은 아니다. `moe::forward`가 launch된 뒤 CTA별로 역할을 나눈다.

```text
moe::forward kernel
|
|-- 마지막 CTA
|     `-- os::start()
|           |-- scheduler: 준비된 Task를 processor에 배정
|           `-- subscriber: 통신/계산 완료 signal을 관찰하고 Task 생성
|
`-- 나머지 CTA = processor CTA
      |-- 일부 CTA는 먼저 dispatch() 수행
      `-- 이후 모두 processor::start()에서 Task 실행
```

따라서 `dispatch()`는 별도의 `__global__` kernel이 아니다. 이미 실행 중인 `moe::forward` kernel 내부에서 호출되는 `__device__` 함수다.

## 2. `moe.cuh`: fused persistent kernel의 진입점

핵심 kernel은 `csrc/include/flashmoe/moe.cuh`의 `moe::forward`다.

```cpp
template <typename Config, Activation a, Topology topo>
__launch_bounds__(Config::Threads::value, 1)
__global__ void forward(
    const __grid_constant__ KernelArgs kArgs,
    const __grid_constant__ Context ctx) {
  // ...
}
```

### 2.1 OS CTA

마지막 CTA는 계산 processor가 아니라 device-side OS 역할을 맡는다.

```cpp
const auto processors = gridDim.x - 1;

if (blockIdx.x == gridDim.x - 1) {
  constexpr auto subscriberCount =
      threads - scheduler::SCHEDULER_COUNT;

  os::start<topo, subscriberCount, threads, bM,
            Config::PSS::value, DataType>(
      flashWorkspace, kArgs.expertCounts, symHeap, ctx,
      kArgs.EC, kArgs.I / bN0, kArgs.H / bN1,
      dispatchBlocks, kArgs.E, kArgs.I, processors);
  return;
}
```

`processors = gridDim.x - 1`인 이유는 마지막 CTA 하나를 OS 전용으로 제외하기 때문이다.

### 2.2 Dispatch CTA

나머지 CTA 중 앞쪽 `dispatchBlocks`개는 먼저 token dispatch를 수행한다.

```cpp
if (blockIdx.x < dispatchBlocks) {
  dispatch<topo, Config::Threads::value, bM, bN0>(
      kArgs.H, kArgs.E, symHeap, kArgs.EC, roundEC,
      ctx.epRank, ctx.world, superBlockSize, dispatchBlocks,
      tokens, ctx.signals, kArgs.expertCounts,
      ctx.tokenIndices, ctx.dispatchSync, ctx.pel,
      flashWorkspace, stateNumber);
}
```

여기에는 kernel launch 문법인 `<<<...>>>`가 없다. 즉 `dispatch.cuh`의 코드는 실행되지만 독립된 kernel이 launch되는 것은 아니다.

또한 이 `if` 뒤에는 `return`이 없다. 그러므로 dispatch를 마친 CTA도 이어서 processor loop에 참여한다. 초기 통신 단계에는 많은 CTA를 dispatch에 사용하고, dispatch가 끝난 후에는 그 CTA를 계산 worker로 재사용하는 구조다.

### 2.3 Processor CTA

OS CTA를 제외한 모든 CTA는 최종적으로 `processor::start()`에 진입한다.

```cpp
const auto pA = processor::ProcessorArgs{
    ctx.statusQueue + blockIdx.x,
    ctx.tqs + blockIdx.x,
    ctx.gTqHeads,
    ctx.tQ,
    ctx.pTq,
    ctx.tileSync
};

processor::start</* template arguments */>(
    /* workspace, dimensions, weights, output, ... */, pA);
```

`ctx.statusQueue + blockIdx.x`와 `ctx.tqs + blockIdx.x`에서 알 수 있듯이 각 processor CTA에는 전용 상태 slot과 전용 doorbell slot이 있다.

## 3. Task: 실제 연산의 정의

`csrc/include/flashmoe/infra/task.cuh`의 `Task`는 processor가 수행할 작업을 나타낸다.

```cpp
enum class TaskType : uint8_t {
  GEMM0,
  GEMM1,
  combine,
};

struct __align__(16) Task {
  Ingredients ingredients{};
  const cuda::std::byte* aData = nullptr;
  cuda::std::array<cuda::std::byte*, GEMMs> cData = {};
  cuda::std::byte* rcData = nullptr;
  uint64_t* flags = nullptr;
  unsigned int syncIdx = 0U;
  unsigned int tileIdx = 0U;
};

static_assert(sizeof(Task) == 64);
```

`Task`에는 다음과 같은 정보가 들어 있다.

- 어떤 연산인지: `GEMM0`, `GEMM1`, `combine`
- 어느 local/global expert의 작업인지
- 입력과 출력 데이터 주소
- token tile 크기와 tile index
- 후속 작업 및 통신을 위한 synchronization 위치
- 결과를 돌려줄 peer 및 remote buffer 정보

실제 Task 객체는 `ctx.tQ`가 가리키는 global-memory task queue에 저장된다.

## 4. Doorbell의 정확한 정체

Doorbell은 CUDA가 제공하는 별도 하드웨어 객체의 이름이 아니다. FlashMoE가 scheduler-to-processor 알림 용도로 사용하는 GPU global-memory의 64비트 slot이다.

Context에는 processor 수만큼 `TQSignal` 배열이 있다.

```cpp
TQSignal* const tqs = nullptr;  // [processors]
```

`bootstrap.cuh`에서 이 배열을 할당하고 0으로 초기화한다.

```cpp
TQSignal* tqs = nullptr;
cudaMallocAsync(&tqs, sizeof(TQSignal) * processors, stream);
cudaMemsetAsync(tqs, 0, sizeof(TQSignal) * processors, stream);
```

각 processor CTA는 자신의 원소만 감시한다.

```text
processor 0 <-> ctx.tqs[0]
processor 1 <-> ctx.tqs[1]
processor 2 <-> ctx.tqs[2]
...
```

`TQSignal`의 실제 정의는 다음과 같다.

```cpp
struct __align__(8) TQSignal {
  uint signal;
  uint interrupt;

  __device__ __forceinline__
  void encodeSig(const uint& sig) {
    signal = sig + 1;
  }

  __device__ __forceinline__
  auto decodeSig() const {
    return signal - 1;
  }
};
```

따라서 doorbell의 64비트에는 다음 두 값이 함께 들어간다.

```text
TQSignal
|-- signal: Task queue index + 1
`-- interrupt: processor 종료 여부
```

`signal`에 1을 더해 저장하는 이유는 `0`을 초기 상태, 즉 아직 배정된 Task가 없는 상태로 남겨두기 위해서다.

중요한 점은 doorbell이 Task 자체나 Task의 포인터를 직접 저장하지 않는다는 것이다. 실제 Task는 task queue에 있고, doorbell의 `signal`은 그 queue 위치를 복원할 수 있게 한다.

```text
Global task queue                       Processor doorbell
+------------------+                  +---------------------+
| Task[0]          |                  | signal = index + 1  |
| Task[1]          | <--- decode -----| interrupt = 0 or 1  |
| Task[2]          |                  +---------------------+
+------------------+
```

## 5. `cuda::atomic_ref`와 memory ordering

Processor는 자신의 doorbell 메모리를 다음과 같이 감싼다.

```cpp
cuda::atomic_ref<uint64_t, cuda::thread_scope_device>
    doorbell{*pA.pDB};
```

`atomic_ref`는 새로운 저장공간을 만드는 객체가 아니다. 이미 존재하는 `*pA.pDB`라는 `uint64_t` 메모리를 atomic하게 접근할 수 있도록 감싼 참조다.

`thread_scope_device`는 scheduler CTA와 processor CTA처럼 동일 GPU에 있는 서로 다른 CTA/thread 사이에서 atomic성과 memory ordering을 보장한다.

Scheduler가 작업을 publish할 때는 release store를 사용한다.

```cpp
sig.encodeSig(taskQueueIndex);
pdb.store(cuda::std::bit_cast<uint64_t>(sig),
          cuda::memory_order_release);
```

Processor는 acquire load로 그 알림을 읽는다.

```cpp
auto payload = cuda::std::bit_cast<TQSignal>(
    doorbell.load(cuda::memory_order_acquire));
```

이 release/acquire 쌍은 다음 순서를 만든다.

```text
Scheduler CTA                         Processor CTA

Task queue에 Task 기록
        |
        v
doorbell.store(..., release)
        |
        | synchronizes-with
        v
doorbell.load(..., acquire)
        |
        v
Task queue에서 완성된 Task 읽기
```

즉 processor가 새 doorbell 값을 관찰했다면 scheduler가 그 전에 기록한 Task도 올바르게 관찰할 수 있다. `__syncthreads()`는 같은 CTA 내부만 동기화하므로 서로 다른 scheduler CTA와 processor CTA 사이에서는 이 atomic protocol이 필요하다.

## 6. Processor의 polling과 Task 실행

`processor.cuh`의 processor CTA는 persistent loop에서 doorbell을 polling한다.

```cpp
TQSignal tqs{0U, 0U};

while (!tqs.interrupt) {
  if (threadIdx.x == 0) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device>
        doorbell{*pA.pDB};

    auto payload = cuda::std::bit_cast<TQSignal>(
        doorbell.load(cuda::memory_order_acquire));

    while (payload.signal == tqs.signal &&
           payload.interrupt == 0) {
      payload = cuda::std::bit_cast<TQSignal>(
          doorbell.load(cuda::memory_order_acquire));
    }

    tqs = payload;
  }

  // CTA 내부에 signal/interrupt 전파 후 Task fetch 및 실행
}
```

Polling loop는 다음 두 경우에 종료된다.

1. `payload.signal != tqs.signal`: 새로운 Task가 배정되었다.
2. `payload.interrupt != 0`: 종료 요청을 받았다.

일반 Task라면 signal을 queue index로 decode한다.

```cpp
const auto* gtQ = pA.tQ + tqs.decodeSig();
```

그 위치에서 `Task`를 읽고 타입에 따라 실행한다.

```cpp
switch (currentTask.getTaskType()) {
case TaskType::GEMM0:
  // Up projection 및 activation/gating
  break;
case TaskType::GEMM1:
  // Down projection 및 필요 시 결과 전송/signal
  break;
case TaskType::combine:
  // Expert 결과를 원래 token 위치에 결합
  break;
}
```

## 7. Interrupt

여기서 `interrupt`는 CUDA 하드웨어 interrupt가 아니다. Scheduler가 persistent processor loop에 보내는 cooperative shutdown message다.

Scheduler는 모든 예정 Task가 처리된 뒤 다음 signal을 processor별 doorbell에 기록한다.

```cpp
constexpr auto sig = TQSignal{0U, 1U};

cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*db_p};
pdb.store(cuda::std::bit_cast<uint64_t>(sig),
          cuda::memory_order_release);
```

Processor가 이를 읽으면 다음 순서로 종료한다.

```text
1. polling 중 interrupt == 1 감지
2. interrupt 메시지를 Task로 해석하지 않음
3. 자신의 doorbell을 {0, 0}으로 초기화
4. CTA 전체 thread에 interrupt 상태를 공유
5. while (!tqs.interrupt) loop 종료
```

실제 초기화 코드는 다음과 같다.

```cpp
if (payload.interrupt) {
  constexpr auto TQSZero =
      cuda::std::bit_cast<uint64_t>(TQSignal{0, 0});
  doorbell.store(TQSZero, cuda::memory_order_relaxed);
}
```

Doorbell을 0으로 되돌리는 이유는 context와 mailbox가 다음 `forward` epoch에서 재사용되기 때문이다. 이전 epoch의 interrupt가 남아 있으면 다음 epoch의 processor가 시작하자마자 종료할 수 있다.

Interrupt는 실행 중인 GEMM을 강제로 중단하지 않는다. Processor가 현재 Task를 마친 뒤 다시 doorbell을 확인했을 때 종료 상태로 전환한다.

## 8. `os.cuh`: device-side OS

`moe.cuh`에서 마지막 CTA가 호출하는 `os::start()`는 한 CTA의 thread를 scheduler와 subscriber로 나눈다.

```cpp
if (threadIdx.x / WARP_SIZE == 0) {
  // 첫 번째 warp: scheduler
  scheduler::start</* ... */>(
      interruptScratch, schedulerBitSet, processors,
      ctx.processors_v, tilesN1, sO, gtQCl,
      interrupt, tQHeads, gtQHeads, taskBound,
      rQ, sQ, pDB);
} else {
  // 나머지 thread: subscribers
  subscriber::Args args{/* ... */};
  subscriber::start</* ... */>(symHeap, args, fSbSL);
}
```

### 8.1 Subscriber의 역할

Subscriber는 dispatch 또는 이전 계산 단계가 남긴 signal을 관찰한다. Dependency가 충족되면 `GEMM0` 또는 `combine` Task를 구성해 global task queue에 넣고 scheduler가 새 작업을 발견할 수 있도록 알린다.

`GEMM1`은 조금 다르다. `GEMM0`의 필요한 출력 tile이 모두 완료되면 processor가 `notifyNext()`를 통해 후속 `GEMM1` Task를 만든다.

```text
dispatch 완료 signal -- subscriber --> GEMM0 Task
GEMM0 완료          -- processor  --> GEMM1 Task
GEMM1/통신 완료     -- subscriber --> combine Task
```

### 8.2 Scheduler의 역할

Scheduler는 크게 다음 상태를 관리한다.

- 준비된 Task가 들어 있는 task queue
- 현재 일을 받을 수 있는 processor들의 ready queue (`rQ`)
- processor가 다음 Task를 받을 준비가 되었음을 표시하는 status queue (`sQ`)
- processor별 doorbell (`pDB`, 즉 `ctx.tqs`)

준비된 Task와 idle processor가 있으면 scheduler는 processor ID를 선택하고 해당 processor의 doorbell에 task queue 위치를 publish한다.

```cpp
auto processorId = rQ[/* ready queue position */];
auto* pdbAddr = reinterpret_cast<uint64_t*>(pDB + processorId);
cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};

sig.encodeSig(taskQueueIndex);
pdb.store(cuda::std::bit_cast<uint64_t>(sig),
          cuda::memory_order_release);
```

`os.cuh`는 kernel 시작 시 dispatch를 하지 않는 processor를 ready queue 앞쪽에 배치한다. 이 processor들은 즉시 계산 Task를 받을 수 있고, dispatch 담당 CTA들은 dispatch가 끝난 후 processor로 합류한다.

### 8.3 종료 처리

Scheduler는 처리해야 할 Task 수인 `taskBound`를 추적한다. 필요한 Task scheduling이 끝나면:

1. subscriber들의 interrupt flag를 설정하고
2. processor별 doorbell에 `interrupt=1`을 전달한다.

이렇게 OS CTA와 모든 processor CTA가 같은 epoch를 정상적으로 종료할 수 있다.

## 9. 전체 메시지 흐름

```text
Dispatch CTA
  | token tile 전송 + signal
  v
Subscriber
  | dependency 확인
  | Task를 task queue에 기록
  v
Scheduler
  | ready processor 선택
  | doorbell = {signal: task index + 1, interrupt: 0}
  v
Processor
  | acquire load로 doorbell 변화 감지
  | task queue에서 Task fetch
  | GEMM0 / GEMM1 / combine 실행
  | statusQueue에 다시 ready 표시
  `----------------------------------> Scheduler

모든 Task 완료
  Scheduler
    | doorbell = {signal: 0, interrupt: 1}
    v
  Processor
    | doorbell을 {0, 0}으로 초기화
    ` persistent loop 종료
```

## 10. 핵심 요약

- `Task`는 실제로 수행할 GEMM 또는 combine 연산의 64-byte 명세다.
- 실제 Task는 global-memory task queue에 저장된다.
- Doorbell은 processor마다 하나씩 있는 64-bit `TQSignal` 메모리 slot이다.
- Doorbell은 Task 자체를 담지 않고 task queue index와 interrupt 상태를 전달한다.
- Scheduler는 release store로 Task를 publish하고 processor는 acquire load로 수신한다.
- Processor는 doorbell을 polling하다 signal이 바뀌면 Task를 실행한다.
- `interrupt=1`은 하드웨어 interrupt가 아니라 persistent loop 종료 메시지다.
- `moe.cuh`의 하나의 kernel 안에서 dispatch, device-side OS, expert compute, combine이 CTA specialization과 task scheduling을 통해 겹쳐 실행된다.
