import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from enum import Enum

PhotoImage = ImageTk.PhotoImage
# np.random.seed(1)

UNIT = 50  # 픽셀 수

class Options:
    def GetTilePossibilities(self):
        p = [0 for i in range(len(TileType))]
        p[TileType.ADDITIONTILE.value] = 0.235
        p[TileType.BLESSINGTILE.value] = 0.115
        p[TileType.RESONANCETILE.value] = 0.16
        p[TileType.REINFORCETILE.value] = 0.16
        p[TileType.DUPLICATIONTILE.value] = 0.16
        p[TileType.RELOCATIONTILE.value] = 0.17
        return p

    def GetCardPossibilities(self):
        p = [0 for i in range(len(CardType))]
        p[CardType.THUNDERSTROKE.value] = 0.15
        p[CardType.EXPLOSION.value] = 0.105
        p[CardType.THUNDERBOLT.value] = 0.09
        p[CardType.FIREWORKS.value] = 0.115
        p[CardType.WATERSPOUT.value] = 0.15
        p[CardType.PURIFICATION.value] = 0.10
        p[CardType.EARTHQUAKE.value] = 0.07
        p[CardType.SHOKEWAVE.value] = 0.095
        p[CardType.STORM.value] = 0.07
        p[CardType.TSUNAMI.value] = 0.055
        return p

class CardType(Enum):
    THUNDERSTROKE = 0  # 낙뢰
    EXPLOSION = 1  # 대폭발
    THUNDERBOLT = 2  # 벼락
    FIREWORKS = 3  # 업화
    WATERSPOUT = 4  # 용오름
    PURIFICATION = 5  # 정화
    EARTHQUAKE = 6  # 지진
    SHOKEWAVE = 7  # 충격파
    STORM = 8  # 폭풍우
    TSUNAMI = 9  # 해일
    # 특수카드
    WORLDTREE = 10  # 세계수의 공명
    ERUPTION = 11  # 분출


class TileType(Enum):
    BASICTILE = 0
    BROKENTILE = 1
    DISTORTEDTILE = 2
    ADDITIONTILE = 3    # 추가
    BLESSINGTILE = 4    # 축복
    RESONANCETILE = 5   # 신비
    REINFORCETILE = 6   # 강화
    DUPLICATIONTILE = 7 # 복제
    RELOCATIONTILE = 8  # 재배치
