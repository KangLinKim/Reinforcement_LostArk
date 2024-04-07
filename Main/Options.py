from Imports import *

PhotoImage = ImageTk.PhotoImage


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

