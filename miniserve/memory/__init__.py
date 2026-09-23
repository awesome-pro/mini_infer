"""Memory package: fixed-size physical blocks holding KV cache.

This is a block-based / paged-style KV cache manager. Blocks are allocated,
freed and looked up through per-request block tables. Attention is not executed
over the physical tensors here, so this is deliberately not called PagedAttention.
"""

from miniserve.block_table import BlockTable
from miniserve.memory.block_manager import (
    BlockManager,
    BlockPlan,
    MemoryStats,
    PagedBlockManager,
)

__all__ = ["BlockManager", "BlockPlan", "BlockTable", "MemoryStats", "PagedBlockManager"]
