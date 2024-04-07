from Imports import *

UNIT = 50  # 픽셀 수


class Map:
    def __init__(self, mapIdx=-1):
        self.plate = self.GetPlate(mapIdx)

        self.HEIGHT = len(self.plate)  # 그리드 세로
        self.WIDTH = len(self.plate)  # 그리드 가로
        self.reRoll = 3
        self.maxPlayTime = 10

    def GetPlate(self, mapIdx):
        if mapIdx == -1:
            mapIdx = np.random.randint(0, len(MapType))
        if mapIdx == MapType.HEADPIECE_1.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.HEADPIECE_2.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 2, 0, 0, 0, 0, -1],
                [-1, 0, 0, 2, 0, 0, 0, -1],
                [-1, 0, 0, 0, 2, 0, 0, -1],
                [-1, 0, 0, 0, 0, 2, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.HEADPIECE_3.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 2, 0, 0, 2, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 2, 0, 0, 2, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.HEADPIECE_4.value:
            return [
                [1, 1, 0, 0, 0, 1, 1, -1],
                [1, 0, 2, 0, 0, 0, 1, -1],
                [0, 0, 0, 0, 0, 2, 0, -1],
                [0, 0, 0, 2, 0, 0, 0, -1],
                [0, 2, 0, 0, 0, 0, 0, -1],
                [1, 0, 0, 0, 2, 0, 1, -1],
                [1, 1, 0, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.HEADPIECE_5.value:
            return [
                [1, 1, 0, 0, 0, 1, 1, -1],
                [1, 0, 2, 0, 2, 0, 1, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 2, 0, 0, 0, 2, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [1, 0, 2, 0, 2, 0, 1, -1],
                [1, 1, 0, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.HEADPIECE_6.value:
            return [
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 2, 0, 0, 0, 0, 1],
                [0, 0, 0, 2, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 2, 0, 0, 0],
                [1, 0, 0, 0, 0, 2, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
            ]
        elif mapIdx == MapType.HEADPIECE_7.value:
            return [
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 2, 2, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 2, 0, 0, 2, 0],
                [0, 2, 0, 0, 2, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 2, 2, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
            ]

        elif mapIdx == MapType.SHOULDERPIECE_1.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.SHOULDERPIECE_2.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 2, 0, -1],
                [-1, 0, 0, 0, 2, 0, 0, -1],
                [-1, 0, 0, 2, 0, 0, 0, -1],
                [-1, 0, 2, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.SHOULDERPIECE_3.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 2, 0, 0, 2, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 2, 0, 0, 2, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.SHOULDERPIECE_4.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 2, 0, 0, 0, -1],
                [0, 2, 0, 0, 0, 2, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 2, 0, 0, 0, 2, 0, -1],
                [0, 0, 0, 2, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.SHOULDERPIECE_5.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 2, 0, 0, 2, 0, 0, -1],
                [0, 0, 0, 0, 0, 2, 0, -1],
                [0, 0, 0, 2, 0, 0, 0, -1],
                [0, 2, 0, 0, 0, 0, 0, -1],
                [0, 0, 2, 0, 0, 2, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.SHOULDERPIECE_6.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 0],
                [0, 2, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 2, 0],
                [0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        elif mapIdx == MapType.SHOULDERPIECE_7.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 2, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 2, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]

        elif mapIdx == MapType.CHESTPIECE_1.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.CHESTPIECE_2.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 0, 2, 2, 0, 0, -1],
                [-1, 0, 0, 2, 2, 0, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.CHESTPIECE_3.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, 1, 0, 2, 0, 0, 1, -1],
                [-1, 0, 0, 0, 0, 2, 0, -1],
                [-1, 0, 2, 0, 0, 0, 0, -1],
                [-1, 1, 0, 0, 2, 0, 1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.CHESTPIECE_4.value:
            return [
                [1, 1, 1, 0, 1, 1, 1, -1],
                [1, 1, 0, 0, 2, 1, 1, -1],
                [1, 2, 2, 0, 0, 0, 1, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [1, 0, 0, 0, 2, 2, 1, -1],
                [1, 1, 2, 0, 0, 1, 1, -1],
                [1, 1, 1, 0, 1, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.CHESTPIECE_5.value:
            return [
                [1, 1, 1, 0, 1, 1, 1, -1],
                [1, 1, 2, 0, 0, 1, 1, -1],
                [1, 0, 0, 0, 0, 2, 1, -1],
                [0, 0, 0, 2, 0, 0, 0, -1],
                [1, 2, 0, 0, 0, 0, 1, -1],
                [1, 1, 0, 0, 2, 1, 1, -1],
                [1, 1, 1, 0, 1, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.CHESTPIECE_6.value:
            return [
                [1, 1, 1, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 2, 2, 0, 2, 0, 1],
                [0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 0],
                [1, 0, 2, 0, 2, 2, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 1, 1, 1],
            ]
        elif mapIdx == MapType.CHESTPIECE_7.value:
            return [
                [1, 1, 1, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 2, 0, 0, 2, 0, 1],
                [0, 2, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 2, 0],
                [1, 0, 2, 0, 0, 2, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 1, 1, 1],
            ]

        elif mapIdx == MapType.PANTS_1.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.PANTS_2.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 2, 2, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 2, 2, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.PANTS_3.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 2, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 2, 0, 0, -1],
                [-1, 0, 0, 2, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 2, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.PANTS_4.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 2, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 2, 0, -1],
                [0, 0, 0, 2, 0, 0, 0, -1],
                [0, 2, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 2, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.PANTS_5.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 2, 0, 2, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 2, 0, 0, 0, 2, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 2, 0, 2, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.PANTS_6.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 2, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 2, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        elif mapIdx == MapType.PANTS_7.value:
            return [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 2, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 0, 2, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]

        elif mapIdx == MapType.GLOVES_1.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.GLOVES_2.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 0, 0, 2, 0, 0, -1],
                [-1, 0, 0, 0, 0, 2, 0, -1],
                [-1, 0, 2, 0, 0, 0, 0, -1],
                [-1, 0, 0, 2, 0, 0, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.GLOVES_3.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, 0, 2, 0, 2, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 2, 0, 2, 0, -1],
                [-1, 1, 0, 0, 0, 0, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.GLOVES_4.value:
            return [
                [1, 1, 0, 0, 0, 1, 1, -1],
                [1, 0, 0, 0, 0, 0, 1, -1],
                [0, 0, 2, 0, 2, 0, 0, -1],
                [0, 0, 0, 2, 0, 0, 0, -1],
                [0, 0, 2, 0, 2, 0, 0, -1],
                [1, 0, 0, 0, 0, 0, 1, -1],
                [1, 1, 0, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.GLOVES_5.value:
            return [
                [1, 1, 0, 0, 0, 1, 1, -1],
                [1, 0, 0, 2, 0, 0, 1, -1],
                [0, 0, 0, 0, 2, 0, 0, -1],
                [0, 2, 0, 0, 0, 2, 0, -1],
                [0, 0, 2, 0, 0, 0, 0, -1],
                [1, 0, 0, 2, 0, 0, 1, -1],
                [1, 1, 0, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.GLOVES_6.value:
            return [
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 2, 0, 0, 2, 0, 1],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0],
                [1, 0, 2, 0, 0, 2, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
            ]
        elif mapIdx == MapType.GLOVES_7.value:
            return [
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 2, 0, 0, 0, 1],
                [0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 2, 0],
                [0, 2, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0],
                [1, 0, 0, 0, 2, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
            ]

        elif mapIdx == MapType.WEAPON_1.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, 1, 0, 2, 0, 0, 1, -1],
                [-1, 0, 0, 2, 0, 0, 0, -1],
                [-1, 0, 0, 0, 2, 0, 0, -1],
                [-1, 1, 0, 0, 2, 0, 1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.WEAPON_2.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, 1, 0, 2, 0, 0, 1, -1],
                [-1, 0, 2, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 2, 0, -1],
                [-1, 1, 0, 0, 2, 0, 1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.WEAPON_3.value:
            return [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, 1, 0, 0, 2, 0, 1, -1],
                [-1, 0, 2, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 2, 0, -1],
                [-1, 1, 0, 2, 0, 0, 1, -1],
                [-1, 1, 1, 0, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]
        elif mapIdx == MapType.WEAPON_4.value:
            return [
                [1, 1, 1, 0, 1, 1, 1, -1],
                [1, 1, 0, 0, 2, 1, 1, -1],
                [1, 2, 0, 2, 0, 0, 1, -1],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [1, 0, 0, 2, 0, 2, 1, -1],
                [1, 1, 2, 0, 0, 1, 1, -1],
                [1, 1, 1, 0, 1, 1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ]