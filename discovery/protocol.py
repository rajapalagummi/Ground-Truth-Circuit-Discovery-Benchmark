from typing import Dict, List, Optional, Protocol, Set, runtime_checkable

from torch import Tensor


@runtime_checkable
class HookableModelProtocol(Protocol):
    node_names: List[str]
    node_layers: Dict[str, int]

    def __call__(
        self,
        ids: Tensor,
        active_mask: Optional[Dict[str, float]] = None,
        patch_cache: Optional[Dict[str, Tensor]] = None,
        patch_mode: str = "none",
        patch_nodes: Optional[Set[str]] = None,
    ) -> Tensor: ...
