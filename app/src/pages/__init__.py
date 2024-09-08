from .ChatPage import ChatPage
from .SettingPage import SettingPage
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Chat": ChatPage,
    "Settings": SettingPage,
}

__all__ = ["PAGE_MAP"]