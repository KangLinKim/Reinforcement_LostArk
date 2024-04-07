import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from enum import Enum


class CardType(Enum):
    THUNDERSTROKE = 0   # 낙뢰
    EXPLOSION = 1       # 대폭발
    THUNDERBOLT = 2     # 벼락
    FIREWORKS = 3       # 업화
    WATERSPOUT = 4      # 용오름
    PURIFICATION = 5    # 정화
    EARTHQUAKE = 6      # 지진
    SHOKEWAVE = 7       # 충격파
    STORM = 8           # 폭풍우
    TSUNAMI = 9         # 해일
    # 특수카드
    WORLDTREE = 10      # 세계수의 공명
    ERUPTION = 11       # 분출


class TileType(Enum):
    BASICTILE = 0       # 기본 타일
    BROKENTILE = 1      # 부숴진 타일
    DISTORTEDTILE = 2   # 왜곡된 타일
    ADDITIONTILE = 3    # 추가
    BLESSINGTILE = 4    # 축복
    RESONANCETILE = 5   # 신비
    REINFORCETILE = 6   # 강화
    DUPLICATIONTILE = 7 # 복제
    RELOCATIONTILE = 8  # 재배치


class MapType(Enum):
    HEADPIECE_1 = 0
    HEADPIECE_2 = 1
    HEADPIECE_3 = 2
    HEADPIECE_4 = 3
    HEADPIECE_5 = 4
    HEADPIECE_6 = 5
    HEADPIECE_7 = 6

    CHESTPIECE_1 = 7
    CHESTPIECE_2 = 8
    CHESTPIECE_3 = 9
    CHESTPIECE_4 = 10
    CHESTPIECE_5 = 11
    CHESTPIECE_6 = 12
    CHESTPIECE_7 = 13

    PANTS_1 = 14
    PANTS_2 = 15
    PANTS_3 = 16
    PANTS_4 = 17
    PANTS_5 = 18
    PANTS_6 = 19
    PANTS_7 = 20

    GLOVES_1 = 21
    GLOVES_2 = 22
    GLOVES_3 = 23
    GLOVES_4 = 24
    GLOVES_5 = 25
    GLOVES_6 = 26
    GLOVES_7 = 27

    SHOULDERPIECE_1 = 28
    SHOULDERPIECE_2 = 29
    SHOULDERPIECE_3 = 30
    SHOULDERPIECE_4 = 31
    SHOULDERPIECE_5 = 32
    SHOULDERPIECE_6 = 33
    SHOULDERPIECE_7 = 34

    WEAPON_1 = 35
    WEAPON_2 = 36
    WEAPON_3 = 37
    WEAPON_4 = 38
    # WEAPON_5 = 39
    # Weapon_6 = 40
    # Weapon_7 = 41