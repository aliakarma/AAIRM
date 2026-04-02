"""Dataset adapters — convert raw data to the AAIRM unified schema."""

from aairm.data.adapters.synthetic_adapter import SyntheticAdapter
from aairm.data.adapters.m5_adapter import M5Adapter
from aairm.data.adapters.favorita_adapter import FavoritaAdapter
from aairm.data.adapters.instacart_adapter import InstacartAdapter

__all__ = ["SyntheticAdapter", "M5Adapter", "FavoritaAdapter", "InstacartAdapter"]
