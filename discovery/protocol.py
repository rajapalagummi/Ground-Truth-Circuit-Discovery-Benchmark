from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class HookableModelProtocol(Protocol):
    node_names: list[str]
    node_layers: dict[str, int]

    def __call__(
        self,
        ids: Tensor,
        active_mask: dict[str, float] | None = None,
        patch_cache: dict[str, Tensor] | None = None,
        patch_mode: str = "none",
        patch_nodes: set[str] | None = None,
    ) -> Tensor: ...
