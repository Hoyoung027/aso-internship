"""CPU-only BF16 bit analysis and a declared, tile-local TCA-TBE size model."""
import numpy as np


# --- 1. BF16 지수 추출: 부호·가수를 제외한 8bit 지수를 얻기 ---
def exponents(bits):
    if bits.dtype != np.uint16:
        raise TypeError("Expected uint16 BF16 bit patterns, not converted floats")
    return (bits >> 7) & 255


# 0~255 지수별 등장 횟수를 집계한다.
def histogram(bits):
    return np.bincount(exponents(bits).ravel(), minlength=256).astype(np.int64)


# --- 2. 지수 구간 선택: 연속 7개 지수의 빈도 합이 최대인 시작값 찾기 ---
def best_start(hist):
    # Ascending tie break. Valid exponents are 0..255; no uint8 underflow.
    return int(np.argmax(np.convolve(hist, np.ones(7, dtype=np.int64), "valid")))


# 커버리지(p7·비연속 top7)와 entropy로 분포의 집중도를 요약한다.
def describe(hist):
    n = int(hist.sum())
    if not n:
        raise ValueError("Empty distribution")
    s = best_start(hist)
    prob = hist[hist > 0] / n
    return dict(elements=n, best_start=s,
                p7=float(hist[s:s+7].sum() / n),
                top7=float(np.sort(hist)[-7:].sum() / n),
                entropy=float(-(prob * np.log2(prob)).sum()))


# --- 3. 타일 저장량: payload·비트맵·메타데이터·패딩을 바이트로 계산 ---
# 실제 압축이나 GPU 메모리 할당을 수행하는 함수는 아니다.
def tile_storage(bits, start):
    """One independent 64x64 record; NOT a GPU allocation measurement.

    Header=16 B (mode, exponent start, payload counts/reserved).
    Median offsets=4*2*4 B, global start/end offsets=2*2*4 B.
    Sign bytes aligned to 16 B; full BF16 elements aligned to 8 elements.
    Raw fallback retains the same 16 B header. Small-tile offsets are omitted.
    """
    if bits.shape != (64, 64) or not 0 <= start <= 249:
        raise ValueError("Expected 64x64 tile and exponent start in [0,249]")
    e = exponents(bits)
    high = int(((e >= start) & (e <= start + 6)).sum())
    full = bits.size - high
    # 선택 구간 안의 원소는 1 B, 나머지는 2 B로 저장하고 각각 16 B 정렬한다.
    sign_bytes = ((high + 15) // 16) * 16
    full_bytes = ((full + 7) // 8) * 16
    bitmap_bytes = 64 * 3 * 8
    metadata_bytes = 16 + 32 + 16
    candidate = sign_bytes + full_bytes + bitmap_bytes + metadata_bytes
    # 압축본이 더 크면 헤더를 포함한 원본 BF16 레코드로 fallback한다.
    raw_record = bits.nbytes + 16
    return dict(high=high, p_selected=high / bits.size,
                sign_bytes=sign_bytes, full_bytes=full_bytes,
                bitmap_bytes=bitmap_bytes, metadata_bytes=metadata_bytes,
                candidate_bytes=candidate, raw_bytes=bits.nbytes,
                stored_bytes=min(candidate, raw_record),
                fallback=candidate >= raw_record)


# --- 4. 미완성 영역: BF16 원본과 비어 있지 않은 영역의 헤더 비용 계산 ---
def residual_storage(elements):
    # Unfinished token rows or unsupported dimension tail remain BF16.
    return 0 if elements == 0 else 2 * elements + 16
