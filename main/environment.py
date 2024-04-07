from Options import *

options = Options()

class Map:
    def __init__(self, mapIdx):
        self.plate = [
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 1],
        ]

        self.HEIGHT = len(self.plate)  # 그리드 세로
        self.WIDTH = len(self.plate)  # 그리드 가로
        self.reRoll = 3
        self.maxPlayTime = 9


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


class Env(tk.Tk):
    def __init__(self, render_speed=0.01):
        super(Env, self).__init__()
        self.mapIdx = 0
        self.map = Map(self.mapIdx)
        self.specialTile = []
        self.render_speed = render_speed
        self.action_space = self.map.plate[:][:]
        self.state_size = None  # 맵 info + 왼쪽 카드 Info + 오른쪽 카드 Info + 대기 카드 3장 + 리롤 횟수
        self.action_size = len(sum(self.action_space, [])) * 2 + 1  # 맵 info * 좌우 카드 선택 + 리롤

        # 카드 초기화
        self.leftHand = None
        self.rightHand = None
        self.waitLine = []
        self.ResetCardState()

        # 캔버스 설정
        self.title('Toy Project by Lin')
        self.geometry('{0}x{1}'.format(self.map.WIDTH * UNIT, (self.map.HEIGHT + 2) * UNIT))
        self.tileImages = self.LoadTileImages()
        self.canvas = self.DrawCanvas()

        self.breakableTiles, self.brokenTile, self.distortTiles = [], [], []
        self.playTime = 0
        self.reRoll = self.map.reRoll

        self.reset()

    # 기본 함수
    def DrawCanvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=(self.map.HEIGHT + 2) * UNIT,
                           width=self.map.WIDTH * UNIT)

        canvas.pack()

        return canvas

    def DrawGrid(self):
        self.canvas.delete('all')
        w = self.map.WIDTH
        h = self.map.HEIGHT + 2
        # 그리드 생성
        for c in range(0, w * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, h * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, h * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, h * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

    def DrawCardImg(self):
        h = self.map.HEIGHT + 1
        if self.rightHand.level == 2:
            self.canvas.create_rectangle(5 * UNIT, h * UNIT, 5 * UNIT + UNIT, h * UNIT + UNIT, fill='blue')
        elif self.rightHand.level == 3:
            self.canvas.create_rectangle(5 * UNIT, h * UNIT, 5 * UNIT + UNIT, h * UNIT + UNIT, fill='red')

        if self.leftHand.level == 2:
            self.canvas.create_rectangle(4 * UNIT, h * UNIT, 4 * UNIT + UNIT, h * UNIT + UNIT, fill='blue')
        elif self.leftHand.level == 3:
            self.canvas.create_rectangle(4 * UNIT, h * UNIT, 4 * UNIT + UNIT, h * UNIT + UNIT, fill='red')

        self.canvas.create_image(5 * UNIT + UNIT / 2, h * UNIT + UNIT / 2, image=self.rightHand.image)
        self.canvas.create_image(4 * UNIT + UNIT / 2, h * UNIT + UNIT / 2, image=self.leftHand.image)
        self.canvas.create_image(2 * UNIT + UNIT / 2, h * UNIT + UNIT / 2, image=self.waitLine[0].image)
        self.canvas.create_image(1 * UNIT + UNIT / 2, h * UNIT + UNIT / 2, image=self.waitLine[1].image)
        self.canvas.create_image(0 * UNIT + UNIT / 2, h * UNIT + UNIT / 2, image=self.waitLine[2].image)

    def DrawMapImg(self):
        for y in range(0, len(self.action_space)):
            for x in range(0, len(self.action_space)):
                tile = self.action_space[int(y)][int(x)]
                self.canvas.create_image(x * UNIT + UNIT / 2, y * UNIT + UNIT / 2,
                                         image=self.tileImages[tile])

    def Draw(self):
        self.DrawGrid()
        self.DrawMapImg()
        self.DrawCardImg()

    def TileBreak(self, hand, card, posX, posY):
        tile = self.action_space[posY][posX]
        relocation = False
        if tile == TileType.BASICTILE.value:
            self.action_space[posY][posX] = TileType.BROKENTILE.value

        elif tile == TileType.BROKENTILE.value:
            pass

        elif tile == TileType.DISTORTEDTILE.value:
            if card.cardType == CardType.PURIFICATION.value or card.cardType == CardType.WORLDTREE.value:
                self.action_space[posY][posX] = TileType.BROKENTILE.value
            elif card.level == 3:
                pass
            else:
                cnt = 3 if len(self.brokenTile) >= 3 else len(self.brokenTile)
                positionIdx = np.random.choice(range(0, len(self.brokenTile)), cnt, replace=False)
                for idx in positionIdx:
                    tmppos = self.breakableTiles[idx]
                    self.action_space[tmppos[0]][tmppos[1]] = TileType.BASICTILE.value

        elif tile == TileType.ADDITIONTILE.value:
            self.action_space[posY][posX] = TileType.BROKENTILE.value
            self.reRoll += 1

        elif tile == TileType.BLESSINGTILE.value:
            self.action_space[posY][posX] = TileType.BROKENTILE.value
            self.playTime -= 1

        elif tile == TileType.RESONANCETILE.value:
            self.action_space[posY][posX] = TileType.BROKENTILE.value
            idx = np.random.randint(2)
            if hand == 0:
                self.rightHand = Card(CardType.ERUPTION.value) if idx == 1 else Card(CardType.WORLDTREE.value)
            elif hand == 1:
                self.leftHand = Card(CardType.ERUPTION.value) if idx == 1 else Card(CardType.WORLDTREE.value)

        elif tile == TileType.REINFORCETILE.value:
            self.action_space[posY][posX] = TileType.BROKENTILE.value
            if hand == 0:
                self.rightHand.LevelUp()
            elif hand == 1:
                self.leftHand.LevelUp()

        elif tile == TileType.DUPLICATIONTILE.value:
            self.action_space[posY][posX] = TileType.BROKENTILE.value
            if hand == 0:
                self.rightHand = card
            elif hand == 1:
                self.leftHand = card

        elif tile == TileType.RELOCATIONTILE.value:
            self.action_space[posY][posX] = TileType.BROKENTILE.value
            relocation = True

        # if tile != TileType.BASICTILE.value and tile != TileType.BROKENTILE.value:
        #     print(f'specialTile Destroyed : {tile}')
        return relocation

    def Relocation(self):
        cnt = len(self.breakableTiles) + len(self.distortTiles)
        positions = np.random.choice(range(0, len(sum(self.action_space, []))), cnt, replace=False)

        action_space = [[1 for w in range(self.map.WIDTH)] for h in range(self.map.HEIGHT)]
        breakableTile = positions[:len(self.breakableTiles)]
        distortTiles = positions[len(self.breakableTiles):]

        for pos in breakableTile:
            x = int(pos % self.map.WIDTH)
            y = int(pos / self.map.HEIGHT)
            action_space[y][x] = TileType.BASICTILE.value

        for pos in distortTiles:
            x = int(pos % self.map.WIDTH)
            y = int(pos / self.map.HEIGHT)
            action_space[y][x] = TileType.DISTORTEDTILE.value

        self.action_space = action_space
        self.MapInfoUpdate()

    def SettupSpecialTile(self):
        p = options.GetTilePossibilities()
        if len(self.breakableTiles) != 0:
            if len(self.specialTile) != 0:
                if self.action_space[self.specialTile[0]][self.specialTile[1]] != TileType.BROKENTILE.value:
                    self.action_space[self.specialTile[0]][self.specialTile[1]] = TileType.BASICTILE.value

            pos = self.breakableTiles[np.random.randint(0, len(self.breakableTiles))]
            self.action_space[pos[0]][pos[1]] = np.random.choice(range(0, len(p)), 1, p=p)[0]
            self.specialTile = [pos[0], pos[1]]

    def LoadTileImages(self):
        tile0 = PhotoImage(
            Image.open("../PracticeImg/0.png").resize((30, 30)))
        tile1 = PhotoImage(
            Image.open("../PracticeImg/1.png").resize((30, 30)))
        tile2 = PhotoImage(
            Image.open("../PracticeImg/2.png").resize((30, 30)))
        tile3 = PhotoImage(
            Image.open("../PracticeImg/3.png").resize((30, 30)))
        tile4 = PhotoImage(
            Image.open("../PracticeImg/4.png").resize((30, 30)))
        tile5 = PhotoImage(
            Image.open("../PracticeImg/5.png").resize((30, 30)))
        tile6 = PhotoImage(
            Image.open("../PracticeImg/6.png").resize((30, 30)))
        tile7 = PhotoImage(
            Image.open("../PracticeImg/7.png").resize((30, 30)))
        tile8 = PhotoImage(
            Image.open("../PracticeImg/8.png").resize((30, 30)))

        return tile0, tile1, tile2, tile3, tile4, tile5, tile6, tile7, tile8

    # 카드 함수
    def GetNewCard(self):
        p = options.GetCardPossibilities()
        newCardIdx = np.random.choice(range(0, len(p)), 1, p=p)
        return newCardIdx[0]

    def CardLevelUP(self):
        if self.leftHand.level == self.leftHand.maxLevel:
            return
        while self.leftHand.cardType == self.rightHand.cardType:
            self.leftHand.LevelUp()
            self.rightHand = self.waitLine[0]
            self.waitLine = self.waitLine[1:]
            self.waitLine.append(Card(self.GetNewCard()))

    def ResetCardState(self):
        cards = [self.GetNewCard() for i in range(5)]

        self.leftHand = Card(cards[0])
        self.rightHand = Card(cards[1])
        self.waitLine = [Card(cards[2]), Card(cards[3]), Card(cards[4])]

        self.CardLevelUP()

    def Action(self, hand, pos):
        pos = [int(pos % self.map.WIDTH), int(pos / self.map.WIDTH)]
        card = self.leftHand if hand == 0 else self.rightHand
        relocation = False

        if hand == 0:
            self.leftHand = self.waitLine[0]
        elif hand == 1:
            self.rightHand = self.waitLine[0]

        if hand == 2:
            self.reRoll -= 1
        else:
            self.playTime += 1

            # 벼락 이펙트 구현
            if card.cardType == CardType.THUNDERBOLT.value:
                pass
                # cnt가 -1일 경우 타일 1개 재생성
                cnt = np.random.randint(0, 3) if card.level == 1 else (
                    np.random.randint(0, 5) if card.level == 2 else (
                        np.random.randint(0, 7)
                    )
                )
                relocation = self.TileBreak(hand, card, pos[0], pos[1])
                # self.action_space[pos[1]][pos[0]] = TileType.BROKENTILE.value

                # 재생성
                if cnt == 0:
                    tmppos = self.brokenTile[np.random.randint(len(self.brokenTile))]
                    self.action_space[tmppos[0]][tmppos[1]] = TileType.BASICTILE.value

                # !재생성
                else:
                    cnt = min(cnt - 1, len(self.breakableTiles))
                    positionIdx = np.random.choice(range(0, len(self.breakableTiles)), cnt, replace=False)
                    for idx in positionIdx:
                        tmppos = self.breakableTiles[idx]
                        relocation = self.TileBreak(hand, card, tmppos[1], tmppos[0])

            # 벼락 외 카드 이펙트 구현
            else:
                effect = card.effect
                for i in range(len(effect['percents'])):
                    num = np.random.randint(0, 99)
                    if num <= effect['percents'][i]:
                        toX = pos[0] + effect['dx'][i]
                        toY = pos[1] + effect['dy'][i]
                        if self.map.WIDTH > toX >= 0 and self.map.HEIGHT > toY >= 0:
                            relocation = self.TileBreak(hand, card, toX, toY)
                    else:
                        pass

        self.waitLine = self.waitLine[1:]
        self.waitLine.append(Card(self.GetNewCard()))
        self.CardLevelUP()

        if relocation:
            self.Relocation()
        if hand != 2:
            self.MapInfoUpdate()
            self.SettupSpecialTile()

        self.Draw()
        self.render()

        # if self.map.maxPlayTime - self.playTime >= 0:
        #     if len(self.breakableTiles) == 0:
        #         reward = 3
        #     else:
        #         reward = 0.1
        # else:
        #     if len(self.breakableTiles) == 0:
        #         reward = 0
        #     else:
        #         reward = -1

        if hand == 2:
            reward = 0
        elif self.map.maxPlayTime - self.playTime >= 0:
            if len(self.breakableTiles) == 0:
                reward = 1
            else:
                reward = -0.1
        else:
            if len(self.breakableTiles) == 0:
                reward = 0
            else:
                reward = -1

        return self.GetState(), reward, len(self.breakableTiles) == 0

    def MapInfoUpdate(self):
        self.breakableTiles = []
        self.brokenTile = []
        self.distortTiles = []
        for y in range(0, len(self.action_space)):
            for x in range(0, len(self.action_space)):
                tile = self.action_space[int(y)][int(x)]
                if tile == TileType.BROKENTILE.value:
                    self.brokenTile.append([y, x])
                elif tile == TileType.DISTORTEDTILE.value:
                    self.distortTiles.append([y, x])
                else:
                    if tile != TileType.BASICTILE.value:
                        self.specialTile = [y, x]
                    self.breakableTiles.append([y, x])

    def ResetActionSpace(self):
        self.map = Map(self.mapIdx)
        self.action_space = self.map.plate

    def reset(self):
        self.update()
        # self.render(0.05)

        self.playTime = 0

        # 카드 초기화
        self.ResetCardState()

        # 맵 초기화
        self.ResetActionSpace()
        self.MapInfoUpdate()
        self.reRoll = self.map.reRoll

        self.Draw()

        return self.GetState()

    def GetState(self):
        Info = dict()
        Info['mapInfo'] = self.GetStateMap()
        Info['reRoll'] = [self.reRoll]
        Info['cardInfo'] = [self.rightHand.cardType, self.rightHand.level, self.leftHand.cardType, self.leftHand.level,
                            self.waitLine[0].cardType, self.waitLine[1].cardType, self.waitLine[2].cardType]
        Info['playTime'] = [self.playTime]

        self.state_size = 0
        for key in list(Info.keys()):
            if type(Info[key][0]) != list:
                self.state_size += len(Info[key])
            else:
                self.state_size += len(Info[key]) * len(Info[key][0])

        return Info

    def GetStateMap(self):
        ret = []
        space = self.action_space
        cards = [self.leftHand, self.rightHand]

        for y in range(len(space)):
            for x in range(len(space[0])):
                tmp = []
                for card in cards:
                    effect = card.effect
                    reward = 0
                    if space[y][x] != TileType.BROKENTILE.value and space[y][x] != TileType.DISTORTEDTILE.value:
                        if card.cardType == CardType.THUNDERBOLT and len(self.breakableTiles) != 0:
                            if card.level == 1:
                                reward = (3 / len(self.breakableTiles))
                            elif card.level == 2:
                                reward = (5 / len(self.breakableTiles))
                            else:
                                reward = (7 / len(self.breakableTiles))
                        else:
                            for i in range(len(effect['percents'])):
                                toX = x + effect['dx'][i]
                                toY = y + effect['dy'][i]
                                if self.map.WIDTH > toX >= 0 and self.map.HEIGHT > toY >= 0:
                                    if space[toY][toX] == TileType.BROKENTILE.value:
                                        pass
                                    elif space[toY][toX] == TileType.DISTORTEDTILE.value:
                                        reward -= 3 * (effect['percents'][i] / 100)
                                    else:
                                        reward += effect['percents'][i] / 100

                    tmp.append(reward)
                tmp.append(space[y][x])
                ret.append(tmp)
        return ret

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
