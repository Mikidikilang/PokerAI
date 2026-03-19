"""
live_reader/  –  Élő póker képernyőolvasó és RTAManager híd

Komponensek:
  ScreenReader  – YOLO + OCR képernyőolvasó
  GameBridge    – Állapotgép, ScreenReader → RTAManager
  SeatConfig    – Szék pozíciók konfigurációja

Használat:
  from live_reader import ScreenReader, GameBridge
"""

from .screen_reader import ScreenReader, SeatConfig
from .game_bridge import GameBridge
from .data_types import (
    CardRead, PlayerRead, ScreenState,
    Detection, FrameDelta, GamePhase,
)
