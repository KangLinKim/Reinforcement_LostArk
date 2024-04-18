import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import gym
import pylab
import random
import numpy as np
import threading
import time
import tkinter as tk
from PIL import ImageTk, Image
from enum import Enum
import matplotlib.pyplot as plt
from collections import deque

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform


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
    CHESTPIECE_1 = 1
    PANTS_1 = 2
    GLOVES_1 = 3
    SHOULDERPIECE_1 = 4
    # WEAPON_1 = 5

    HEADPIECE_2 = 5
    CHESTPIECE_2 = 6
    PANTS_2 = 7
    GLOVES_2 = 8
    SHOULDERPIECE_2 = 9
    # WEAPON_2 = 11

    HEADPIECE_3 = 10
    CHESTPIECE_3 = 11
    PANTS_3 = 12
    GLOVES_3 = 13
    SHOULDERPIECE_3 = 14
    # WEAPON_3 = 17

    HEADPIECE_4 = 15
    CHESTPIECE_4 = 16
    PANTS_4 = 17
    GLOVES_4 = 18
    SHOULDERPIECE_4 = 19
    # WEAPON_4 = 23

    HEADPIECE_5 = 20
    CHESTPIECE_5 = 21
    PANTS_5 = 22
    GLOVES_5 = 23
    SHOULDERPIECE_5 = 24

    HEADPIECE_6 = 25
    CHESTPIECE_6 = 26
    PANTS_6 = 27
    GLOVES_6 = 28
    SHOULDERPIECE_6 = 29

    HEADPIECE_7 = 30
    CHESTPIECE_7 = 31
    PANTS_7 = 32
    GLOVES_7 = 33
    SHOULDERPIECE_7 = 34

    # WEAPON_5 = 39
    # Weapon_6 = 40
    # Weapon_7 = 41