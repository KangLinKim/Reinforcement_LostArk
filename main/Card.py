from imports import *
from Options import *

class Card:
    def __init__(self, card, cardLevel=1):
        super(Card, self).__init__()
        self.cardType = card
        self.level = cardLevel
        self.maxLevel = 3 if (
                self.cardType != CardType.WORLDTREE.value and self.cardType != CardType.ERUPTION.value) else 1
        self.imageList = self.LoadCardImages()
        self.image = self.imageList[card]
        self.effect = self.CardEffect()

    def LevelUp(self):
        self.level = self.level + 1
        self.effect = self.CardEffect()

    def LoadCardImages(self):
        THUNDERSTROKE = PhotoImage(
            Image.open("../PracticeImg/낙뢰.png").resize((30, 30)))
        EXPLOSION = PhotoImage(
            Image.open("../PracticeImg/대폭발.png").resize((30, 30)))
        THUNDERBOLT = PhotoImage(
            Image.open("../PracticeImg/벼락.png").resize((30, 30)))
        FIREWORKS = PhotoImage(
            Image.open("../PracticeImg/업화.png").resize((30, 30)))
        WATERSPOUT = PhotoImage(
            Image.open("../PracticeImg/용오름.png").resize((30, 30)))
        PURIFICATION = PhotoImage(
            Image.open("../PracticeImg/정화.png").resize((30, 30)))
        EARTHQUAKE = PhotoImage(
            Image.open("../PracticeImg/지진.png").resize((30, 30)))
        SHOKEWAVE = PhotoImage(
            Image.open("../PracticeImg/충격파.png").resize((30, 30)))
        STORM = PhotoImage(
            Image.open("../PracticeImg/폭풍우.png").resize((30, 30)))
        TSUNAMI = PhotoImage(
            Image.open("../PracticeImg/해일.png").resize((30, 30)))
        WORLDTREE = PhotoImage(
            Image.open("../PracticeImg/분출.png").resize((30, 30)))
        ERUPTION = PhotoImage(
            Image.open("../PracticeImg/세계수의공명.png").resize((30, 30)))

        return THUNDERSTROKE, EXPLOSION, THUNDERBOLT, FIREWORKS, WATERSPOUT, PURIFICATION, EARTHQUAKE, SHOKEWAVE, STORM, TSUNAMI, WORLDTREE, ERUPTION

    def CardEffect(self):
        lst = dict()
        if self.cardType == CardType.THUNDERSTROKE.value:
            lst['dx'] = [0, 0, 0, -1, 1]
            lst['dy'] = [-1, 1, 0, 0, 0]
            lst['percents'] = [50, 50, 100, 50, 50] if self.level == 1 else [100, 100, 100, 100, 100]
        elif self.cardType == CardType.EXPLOSION.value:
            lst['dx'] = [0,
                         1, 2, 3, 4, 5, 6, 7, 8,
                         -1, -2, -3, -4, -5, -6, -7, -8,
                         1, 2, 3, 4, 5, 6, 7, 8,
                         -1, -2, -3, -4, -5, -6, -7, -8,
                         ]
            lst['dy'] = [0,
                         1, 2, 3, 4, 5, 6, 7, 8,
                         -1, -2, -3, -4, -5, -6, -7, -8,
                         -1, -2, -3, -4, -5, -6, -7, -8,
                         1, 2, 3, 4, 5, 6, 7, 8,
                         ]

            lst['percents'] = [100,
                               85, 75, 65, 55, 45, 35, 25, 15,
                               85, 75, 65, 55, 45, 35, 25, 15,
                               85, 75, 65, 55, 45, 35, 25, 15,
                               85, 75, 65, 55, 45, 35, 25, 15,
                               ] if self.level == 1 else [
                100,
                100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100,
            ]
        elif self.cardType == CardType.THUNDERBOLT.value:
            lst['dx'] = []
            lst['dy'] = []
            lst['percents'] = []
        elif self.cardType == CardType.FIREWORKS.value:
            lst['dx'] = [0,
                         -1, 0, 1,
                         -2, -1, 0, 1, 2,
                         -1, 0, 1,
                         0]
            lst['dy'] = [-2,
                         -1, -1, -1,
                         0, 0, 0, 0, 0,
                         1, 1, 1,
                         2]
            lst['percents'] = [50,
                               50, 50, 50,
                               50, 50, 100, 50, 50,
                               50, 50, 50,
                               50, ] if self.level == 1 else [
                100,
                100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100,
                100,
            ]
        elif self.cardType == CardType.WATERSPOUT.value:
            lst['dx'] = [-1, -1, 0, 1, 1]
            lst['dy'] = [-1, 1, 0, -1, 1]
            lst['percents'] = [50, 50, 100, 50, 50] if self.level == 1 else [100, 100, 100, 100, 100]
        elif self.cardType == CardType.PURIFICATION.value:
            lst['dx'] = [-1, 0, 1]
            lst['dy'] = [0, 0, 0]
            lst['percents'] = [50, 100, 50]
        elif self.cardType == CardType.EARTHQUAKE.value:
            lst['dx'] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            lst['dy'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            lst['percents'] = [15, 25, 35, 45, 55, 65, 75, 85, 100, 85, 75, 65, 55, 45, 35, 25, 15] \
                if self.level == 1 else \
                [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        elif self.cardType == CardType.SHOKEWAVE.value:
            lst['dx'] = [-1, 0, 1,
                         -1, 0, 1,
                         -1, 0, 1, ]
            lst['dy'] = [-1, -1, -1,
                         0, 0, 0,
                         1, 1, 1, ]
            lst['percents'] = [75, 75, 75,
                               75, 100, 75,
                               75, 75, 75, ] \
                if self.level == 1 else \
                [100, 100, 100,
                 100, 100, 100,
                 100, 100, 100, ]
        elif self.cardType == CardType.STORM.value:
            lst['dx'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            lst['dy'] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            lst['percents'] = [15, 25, 35, 45, 55, 65, 75, 85, 100, 85, 75, 65, 55, 45, 35, 25, 15] \
                if self.level == 1 else \
                [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        elif self.cardType == CardType.TSUNAMI.value:
            lst['dx'] = [0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
            lst['dy'] = [0,
                         -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            lst['percents'] = [100,
                               85, 75, 65, 55, 45, 35, 25, 15, 15, 25, 35, 45, 55, 65, 75, 85,
                               85, 75, 65, 55, 45, 35, 25, 15, 15, 25, 35, 45, 55, 65, 75, 85, ] \
                if self.level == 1 else \
                [100,
                 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, ]
        elif self.cardType == CardType.WORLDTREE.value:
            lst['dx'] = [0, 0, -2, -1, 0, 1, 2, 0, 0]
            lst['dy'] = [-2, -1, 0, 0, 0, 0, 0, 1, 2]
            lst['percents'] = [100, 100, 100, 100, 100, 100, 100, 100, 100, ]
        elif self.cardType == CardType.ERUPTION.value:
            lst['dx'] = [0]
            lst['dy'] = [0]
            lst['percents'] = [100]

        return lst