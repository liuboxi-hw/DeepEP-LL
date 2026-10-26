#pragma once
#include "kernel_operator.h"
#include "ep_memory_server_dedup_buffer_layout.h"
#include "../../dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"

namespace CombineDedup {
constexpr uint32_t kClearFlagBlocks = 256U;
constexpr uint32_t kClearFlagUbBytes = kClearFlagBlocks * 32U;

// Host has retired the previous call; the completion exchange follows this
// helper. Only flag bytes are initialized, preserving every payload byte.
__aicore__ inline void ClearResultFlags(__gm__ uint8_t* local, uint64_t blocks,
                                       AscendC::LocalTensor<uint8_t> zeros,
                                       uint32_t core, uint32_t cores)
{
    const uint64_t base = blocks / cores, extra = blocks % cores;
    const uint64_t count = base + (core < extra ? 1U : 0U);
    const uint64_t begin = core * base + (core < extra ? core : extra);
    for (uint32_t bank = 0U; bank < 2U; ++bank) {
        AscendC::GlobalTensor<uint8_t> result;
        result.SetGlobalBuffer(local + uint64_t(bank) * DispatchDedup::kEpServerDedupWindowBytes +
                               COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET);
        result.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        for (uint64_t done = 0U; done < count; done += kClearFlagBlocks) {
            const uint32_t chunk = count - done < kClearFlagBlocks ? uint32_t(count - done) : kClearFlagBlocks;
            AscendC::DataCopyPad(result[(begin + done) * 512U + 480U], zeros,
                                {uint16_t(chunk), 32U, 0U, 480U, 0U});
        }
    }
}
}  // namespace CombineDedup
