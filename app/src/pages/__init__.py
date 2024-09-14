from .ChatPage import ChatPage
from .ConstructGraphPage import ConstructGraphPage
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {"Chat": ChatPage, "Create Graph": ConstructGraphPage}

__all__ = ["PAGE_MAP"]
