from imports import *

UNIT = 50  # 픽셀 수


class Map:
    def __init__(self, mapIdx):
        self.plate = [
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  1,  1,  0,  0,  1,  1, -1],
            [-1,  1,  0,  0,  0,  0,  1, -1],
            [-1,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0, -1],
            [-1,  1,  0,  0,  0,  0,  1, -1],
            [-1,  1,  1,  0,  0,  1,  1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ]

        self.HEIGHT = len(self.plate)  # 그리드 세로
        self.WIDTH = len(self.plate)  # 그리드 가로
        self.reRoll = 3
        self.maxPlayTime = 9

    def GetPlate(self):
        pass
