"""Block tables: the logical-to-physical mapping for one sequence's KV cache.

Lives outside both ``engine`` and ``memory`` because a
:class:`~mini_infer.engine.request.Request` owns one while the block manager
allocates into one; keeping it here avoids a circular import between them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


class BlockTable:
    """Physical block ids holding one request's KV, indexed by logical block.

    Logical block ``i`` covers token positions ``[i * block_size, (i + 1) * block_size)``.
    Physical blocks are not contiguous and are never assumed to be.
    """

    __slots__ = ("_blocks",)

    def __init__(self, blocks: Iterable[int] = ()) -> None:
        self._blocks: list[int] = list(blocks)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def physical_blocks(self) -> tuple[int, ...]:
        return tuple(self._blocks)

    def block_for_token(self, position: int, block_size: int) -> int:
        """Physical block holding ``position``, or raise if it is not allocated."""
        logical = position // block_size
        if logical >= len(self._blocks):
            raise IndexError(
                f"token position {position} is not covered by {len(self._blocks)} blocks"
            )
        return self._blocks[logical]

    def append(self, block_id: int) -> None:
        if block_id in self._blocks:
            raise ValueError(f"physical block {block_id} is already in this table")
        self._blocks.append(block_id)

    def clear(self) -> None:
        self._blocks.clear()

    def __len__(self) -> int:
        return len(self._blocks)

    def __iter__(self) -> Iterator[int]:
        return iter(self._blocks)

    def __getitem__(self, index: int) -> int:
        return self._blocks[index]

    def __contains__(self, block_id: object) -> bool:
        return block_id in self._blocks

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockTable):
            return self._blocks == other._blocks
        if isinstance(other, list):
            return self._blocks == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"BlockTable({self._blocks})"
