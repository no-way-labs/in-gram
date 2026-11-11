"""In-gram: In-Context N-gram Memory. Reference implementation of the layer described in the paper."""
from .decode import InGramDecoder
from .hashing import cue_addresses, hash_cue, table_row
from .layer import InGram
from .schedule import CUE_LENGTHS, build_schedule

__all__ = ["InGram", "InGramDecoder", "build_schedule", "cue_addresses", "hash_cue", "table_row", "CUE_LENGTHS"]
__version__ = "0.1.0"
