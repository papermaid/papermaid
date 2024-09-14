from .ChatPage import ChatPage
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {"Chat": ChatPage}

__all__ = ["PAGE_MAP"]
