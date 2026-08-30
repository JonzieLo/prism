from collections import Counter,defaultdict
from dataclasses import dataclass

from .chain import *
from .forwards import *
from .hygiene import *

@dataclass(frozen=True)
class FilterCount:
    reason: str
    pair_count: int
    fraction: float

@dataclass(frozen=True)
class ExpiryIssue:
    expiration_timestamp: int
    underlying_index:str
    reason: str

@dataclass(frozen=True)
class ForwardCurveResult:
    raw_option_count: int
    quotes: tuple[OptionQuote, ...]
    chain_issues: tuple[ChainIssue,...]
