from Options import *
from Map import *
from Card import *

options = Options()

global tmpSpecialTile

class Env(tk.Tk):
    def __init__(self, render_speed=0.01, _mapIdx=-1):
        super(Env, self).__init__()
        self.mapIdx = _mapIdx
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
        # self.canvas = self.DrawCanvas()

        self.breakableTiles, self.brokenTile, self.distortTiles = [], [], []
        self.playTime = 0
        self.reRoll = self.map.reRoll
        self.totalCnt, self.idxCnt = 0, 0

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
                if tile != -1:
                    self.canvas.create_image(x * UNIT + UNIT / 2, y * UNIT + UNIT / 2,
                                             image=self.tileImages[tile])

    def Draw(self):
        return
        # self.DrawGrid()
        # self.DrawMapImg()
        # self.DrawCardImg()

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
                    tmpPos = self.brokenTile[idx]
                    self.action_space[tmpPos[0]][tmpPos[1]] = TileType.BASICTILE.value
                    # print(f'{tmpPos[0]}, {tmpPos[1]} restored by DISTORTEDTILE')

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

        return relocation

    def Relocation(self):
        cnt = len(self.breakableTiles) + len(self.distortTiles)
        locatable = []
        for h in range(self.map.HEIGHT):
            for w in range(self.map.WIDTH):
                if self.action_space[h][w] != -1:
                    locatable.append([w, h])

        positions = np.random.choice(range(0, len(locatable)), cnt, replace=False)

        action_space = [[1 if self.action_space[h][w] != -1 else -1
                        for w in range(self.map.WIDTH)]
                        for h in range(self.map.HEIGHT)]

        breakableTile = positions[:len(self.breakableTiles)]
        distortTiles = positions[len(self.breakableTiles):]

        for pos in breakableTile:
            x = locatable[pos][0]
            y = locatable[pos][1]
            action_space[y][x] = TileType.BASICTILE.value

        for pos in distortTiles:
            x = locatable[pos][0]
            y = locatable[pos][1]
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

    def Action(self, action):
        mapSize = len(sum(self.action_space, []))
        if action >= mapSize * 2:
            # 리롤
            hand = 2
            pos = 0
        elif action >= mapSize:
            # 오른손 선택
            hand = 1
            pos = action - mapSize
        else:
            # 왼손
            hand = 0
            pos = action

        pos = [int(pos % self.map.WIDTH), int(pos / self.map.WIDTH)]
        card = self.leftHand if hand == 0 else self.rightHand
        relocation = False

        tileCntBeforeAction = len(self.breakableTiles)

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
                    if len(self.brokenTile) != 0:
                        tmpPos = self.brokenTile[np.random.randint(len(self.brokenTile))]
                        self.action_space[tmpPos[0]][tmpPos[1]] = TileType.BASICTILE.value
                        # print(f'{tmpPos[0]}, {tmpPos[1]} restored by THUNDERBOLT')

                # !재생성
                else:
                    cnt = min(cnt - 1, len(self.breakableTiles))
                    positionIdx = np.random.choice(range(0, len(self.breakableTiles)), cnt, replace=False)
                    for idx in positionIdx:
                        tmpPos = self.breakableTiles[idx]
                        relocation = self.TileBreak(hand, card, tmpPos[1], tmpPos[0])

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

        self.MapInfoUpdate()
        self.Draw()

        tileCntAfterAction = len(self.breakableTiles)

        cutLine = self.map.maxPlayTime - self.playTime
        if hand == 2:
            reward = 0
        else:
            tileCnt = tileCntBeforeAction - tileCntAfterAction
            reward = tileCnt * 0.1
            if len(self.breakableTiles) == 0:
                if cutLine >= 0:
                    reward += 3
                elif cutLine >= -1:
                    reward += 2
                elif cutLine >= -3:
                    reward += 1
            else:
                if cutLine < 0:
                    reward -= 0.5

        return self.GetState(), reward, len(self.breakableTiles) == 0

    def MapInfoUpdate(self):
        self.breakableTiles = []
        self.brokenTile = []
        self.distortTiles = []
        for y in range(0, self.map.HEIGHT):
            for x in range(0, self.map.WIDTH):
                tile = self.action_space[y][x]
                if tile == TileType.BROKENTILE.value:
                    self.brokenTile.append([y, x])
                elif tile == TileType.DISTORTEDTILE.value:
                    self.distortTiles.append([y, x])
                elif tile == -1:
                    pass
                else:
                    if tile != TileType.BASICTILE.value:
                        self.specialTile = [y, x]
                    self.breakableTiles.append([y, x])

    def ResetActionSpace(self):
        self.map = Map(self.mapIdx)
        self.action_space = self.map.plate
        self.specialTile = []

    def reset(self):
        # self.update()

        self.playTime = 0

        # 카드 초기화
        self.ResetCardState()

        # 맵 초기화
        self.ResetActionSpace()
        self.MapInfoUpdate()
        self.reRoll = self.map.reRoll

        self.totalCnt = len(self.breakableTiles)
        self.idxCnt = self.totalCnt / self.map.maxPlayTime

        self.Draw()

        return self.GetState()

    def GetState(self):
        Info = dict()
        Info['mapInfo'] = self.GetStateMap()
        Info['reRoll'] = [self.reRoll]
        Info['cardInfo'] = [self.rightHand.cardType, self.rightHand.level, self.leftHand.cardType, self.leftHand.level,
                            self.waitLine[0].cardType, self.waitLine[1].cardType, self.waitLine[2].cardType]
        Info['playTime'] = [self.map.maxPlayTime - self.playTime]

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
                tmp.append(space[y][x] if space[y][x] not in [TileType.BROKENTILE.value, -1] else -1)
                # for card in cards:
                #     effect = card.effect
                #     reward = 0
                #     if space[y][x] != TileType.BROKENTILE.value and space[y][x] != TileType.DISTORTEDTILE.value:
                #         if card.cardType == CardType.THUNDERBOLT and len(self.breakableTiles) != 0:
                #             if card.level == 1:
                #                 reward = (3 / len(self.breakableTiles))
                #             elif card.level == 2:
                #                 reward = (5 / len(self.breakableTiles))
                #             else:
                #                 reward = (7 / len(self.breakableTiles))
                #         else:
                #             for i in range(len(effect['percents'])):
                #                 toX = x + effect['dx'][i]
                #                 toY = y + effect['dy'][i]
                #                 if self.map.WIDTH > toX >= 0 and self.map.HEIGHT > toY >= 0:
                #                     if space[toY][toX] in [TileType.BROKENTILE.value, -1]:
                #                         pass
                #                     elif space[toY][toX] == TileType.DISTORTEDTILE.value:
                #                         reward -= 3 * (effect['percents'][i] / 100)
                #                     else:
                #                         reward += effect['percents'][i] / 100
                #
                #     tmp.append(reward)
                ret.append(tmp)
        return ret

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
