import glob
import os.path
import numpy as np
from heapq import nsmallest
import cv2

class Injection:
    def __init__(self, inj_path, kpt_path, workon, nbinj=1, rf=None):
        # preprocessing
        self.workon = workon
        self.nbInj = nbinj
        self.img_path = inj_path[:-4] + os.path.splitext(workon)[0] + ".jpg"
        self.inj_path = os.path.dirname(self.img_path) + "/injection/"
        os.mkdir(self.inj_path)
        self.preprocess(inj_path, kpt_path)

        self.rf = rf
        # self.right, self.front = rf
        self.big, self.small = max(self.a, self.b), min(self.a, self.b)
        self.loc, self.Ploc = None, None
        self.start = None
        self.side = None
        self.maxInjection = 3
        self.cx, self.cy, self.w, self.h = self.injworkon[0][1], self.injworkon[0][2], self.injworkon[0][3], self.injworkon[0][4]
        self.Art = {art: self.kptworkon[art] for art in range(21)}
        # [locDict] 0:paume, 1:pouce, 2~5:le reste
        self.locDict = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12],
                        4: [13, 14, 15, 16], 5: [17, 18, 19, 20]}

    # Preprocessing pour rendre les labels pertinents à utiliser
    def preprocess(self, inj_path, kpt_path):
        injury_labels_txt, injury_labels = [], []
        kpt_labels_txt, kpt_labels = [], []

        injury_labels_txt = glob.glob(inj_path + '/' + '*.txt')
        kpt_labels_txt = glob.glob(kpt_path + '/' + '*.txt')

        # preprocessing des labels de plaie
        print("\npreprocessing injury labels ...")
        for i in range(len(injury_labels_txt)):
            p = open(injury_labels_txt[i], 'r')
            lines = p.readlines()
            length = len(lines)
            for ilc, line in enumerate(lines):
                if ilc == length - 1:
                    injury_labels.append(line)
                else:
                    injury_labels.append(line[:-1])
        for idx in range(len(injury_labels)):
            temp = []
            for strlb in (injury_labels[idx].split()):  # ['0', '1']
                temp.append(float(strlb))  # [0, 1]
            injury_labels[idx] = temp

        # preprcessing des labels d'articulation
        print("preprocessing keypoints labels ...")
        for ilc in range(len(kpt_labels_txt)):
            p = open(kpt_labels_txt[ilc], 'r')
            kpt_labels.append(p.read()[:-1])
        for idx in range(len(kpt_labels)):
            temp1 = []
            for strkpt in kpt_labels[idx].split('\n'):  # 0, 1, 2
                temp2 = []
                for jdx, k in enumerate(strkpt.split(' ')):
                    if jdx < 2:
                        temp2.append(float(k))
                temp1.append(temp2)
            kpt_labels[idx] = temp1

        self.injworkon = injury_labels
        self.kptworkon = kpt_labels[0]

        # calcul des distances les plus proches entre les plaies et les articulations
        dist, dists = [], []
        if self.nbInj > 1:
            bestInj, max_y = None, 0
            # sélectionner un plaie
            for inj in self.injworkon:
                if (inj[2] + inj[4] / 2) > max_y:
                    max_y = (inj[2] + inj[4] / 2)
                    bestInj = inj
            self.injworkon[0] = bestInj
        # every distance between the injury and all kpts
        for i in range(len(self.kptworkon)):
            # injworkon[i][1:3] : le centre du plaie
            dist.append(np.linalg.norm(np.array(self.injworkon[0][1:3]) - np.array(self.kptworkon[i])))
        dists.append(dist)

        # les deux keypoints les plus proches du plaie
        self.a, self.b = dists[0].index(nsmallest(2, dists[0])[0]), dists[0].index(nsmallest(2, dists[0])[1])

        print("\n[Les résultats]")
        print(f"le nombre de plaies\t\t\t\t\t: {self.nbInj}")
        print(f"les articulations les plus proches\t: {self.a}, {self.b}")
        print(f"les coordonnées de {self.a} et {self.b}\t\t\t: {self.kptworkon[self.a]}, {self.kptworkon[self.b]}")
        for i, info in enumerate(self.injworkon):
            if len(self.injworkon) == 1:
                print(f"\n{i + 1}er plaie")
            elif len(self.injworkon) >= 2 and len(info) > 5:
                if i == 0:
                    print(f"\n{i+1}er plaie")
                else:
                    print(f"\n{i+1}em plaie")
            if len(info) > 5:
                print(f"center(x, y)\t\t\t\t\t\t: {(info[1], info[2])}")
                print(f"w, h        \t\t\t\t\t\t: {(info[3], info[4])}")
                print(f"confidence  \t\t\t\t\t\t: {info[5]}")
            else: continue

    # Trouver un point à partir du quel on peut commencer des injections.
    def FindStart(self):
        # "Left"
        # 0 : r-f and l-b // 1 : r-b and l-f

        self.right, self.front = None, None
        if self.rf == 'Left':
            self.right = True
            if self.Art[4][0] > self.Art[17][0]:
                self.front = True
            else:
                self.front = False
        elif self.rf == 'Right':
            self.right = False
            if self.Art[4][0] < self.Art[17][0]:
                self.front = True
            else:
                self.front = False

        self.handTypeCheck = 0 if (self.right and self.front) \
                                  or (not self.right and not self.front) else 1
        print(f"right : {self.right}\nfront : {self.front}")
        print(f"self.handTypeCheck : {self.handTypeCheck}")
        if self.right:
            if self.front:
                print("hand type checking : right hand and front-sided")
            else:
                print("hand type checking : right hand and back-sided")
        else:
            if self.front:
                print("hand type checking : left hand and front-sided")
            else:
                print("hand type checking : left hand and back-sided")

        # 1. Définir la partie sur laquelle un plaie existe.
        #    : Où le centre du plaie se situe sur
        self.__FindLoc(self.handTypeCheck)
        print(f"les articulations les plus proches modifiées\t: {self.a}, {self.b}")
        if self.loc == 0:
            if not self.handTypeCheck:
                if self.front and self.right:
                    print(f"la plaie\t: sur la paume")
                if not self.front and not self.right:
                    print(f"la plaie\t: sur le dos de la main")
            else:
                if not self.front and self.right:
                    print(f"la plaie\t: sur le dos de la main")
                if self.front and not self.right:
                    print(f"la plaie\t: sur la paume")
        elif self.loc == 1:
            print(f"la plaie\t: sur la pouce")
        elif self.loc == 2:
            print(f"la plaie\t: sur l'index")
        elif self.loc == 3:
            print(f"la plaie\t: sur le majeur")
        elif self.loc == 4:
            print(f"la plaie\t: sur l'annulaire")
        elif self.loc == 5:
            print(f"la plaie\t: sur l'auriculaire")

        # 2. Détailler la taille du plaie dépendant de la partie définie précédemment
        self.__DefineStart(self.loc)

        # 3. Définir le max nombre d'injections
        if self.loc in [2, 3, 4, 5]:
            if self.Ploc in [0, 1, 2]:
                self.maxInjection = 3
            elif self.Ploc == 3:
                self.maxInjection = 2
            elif self.Ploc == -1:
                pass # préciser maxInjection après
        elif self.loc == 1:
            if self.Ploc == 0:
                self.maxInjection = 3
            elif self.Ploc in [1,2]:
                self.maxInjection = 2
            elif self.Ploc == -1:
                pass # préciser maxInjection plus tard
        elif self.loc == 0:
            if self.start[1] < self.MidNine2Mid[1]:
                self.maxInjection = 2
            if self.start[1] > self.MidNine2Mid[1]:
                self.maxInjection = 2
            if self.start[1] > self.MidNine2Zero[1]:
                if self.start[1] < self.MidMid2Zero[1]:
                    self.maxInjection = 1
                elif self.start[1] >= self.MidMid2Zero[1]:
                    self.maxInjection = 0
        return self.loc, self.Ploc, self.start, self.maxInjection

    def FindInjectionPoints(self):
        # Find the starting point for injections
        self.FindStart()

        # (!) Need to update
        # calculer les meilleurs points d'injection
        assert self.maxInjection, "Injection refusée. la distance entre le plaie et l'Articulation 0 est assez courte."
        if self.maxInjection > 0:
            self.injections = [None for _ in range(self.maxInjection)]
        else: pass

        # le plaie sur l'un des doigts à part la pouce
        if self.loc in [2, 3, 4, 5]:
            if self.Ploc == 0: # maxInjection = 3
                assert len(self.injections) == 3, "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.injections[0] = self.__nrPoint(self.Art[self.locDict[self.loc][3]],
                                                    self.Art[self.locDict[self.loc][2]], n=1, r=2)
                self.injections[1] = self.__nrPoint(self.Art[self.locDict[self.loc][2]],
                                                    self.Art[self.locDict[self.loc][1]], n=1, r=2)
                self.injections[2] = self.__nrPoint(self.Art[self.locDict[self.loc][1]],
                                                    self.Art[self.locDict[self.loc][0]], n=1, r=3)
            elif self.Ploc == 1: # maxInjection = 3
                assert len(self.injections) == 3, "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.injections[0] = self.__nrPoint(self.Art[self.locDict[self.loc][2]],
                                                    self.Art[self.locDict[self.loc][1]], n=1, r=2)
                self.injections[1] = self.__nrPoint(self.Art[self.locDict[self.loc][1]],
                                                    self.Art[self.locDict[self.loc][0]], n=1, r=3)
                self.injections[2] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                                                    self.Art[0], n=1, r=2)
            elif self.Ploc == 2: # maxInjection = 3
                assert len(self.injections) == 3, "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.injections[0] = self.__nrPoint(self.Art[self.locDict[self.loc][1]],
                                                    self.Art[self.locDict[self.loc][0]], n=1, r=3)
                self.injections[1] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                                                    self.Art[0], n=1, r=3)
                self.injections[2] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                                                    self.Art[0], n=2, r=3)
            elif self.Ploc == 3:
                assert len(self.injections) == 2, "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.injections[0] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                                                    self.Art[0], n=1, r=3)
                self.injections[1] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                                                    self.Art[0], n=2, r=3)
            elif self.Ploc == -1:
                if self.start[1] <= self.MidNine2Zero[1]:
                    self.maxInjection = 2
                    self.injections = [None for _ in range(self.maxInjection)]
                    self.injections[0] = self.__nrPoint(self.start, self.Art[0], n=1, r=3)
                    self.injections[1] = self.__nrPoint(self.start, self.Art[0], n=2, r=3)
                elif self.start[1] > self.MidNine2Zero[1]:
                    self.maxInjection = 1
                    self.injections = [None for _ in range(self.maxInjection)]
                    self.injections[0] = self.__nrPoint(self.start, self.Art[0], n=1, r=2)
        # Le plaie à la pouce
        elif self.loc == 1:
            if self.Ploc == 0:
                assert len(self.injections) == 3, "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.injections[0] = self.__nrPoint(self.Art[self.locDict[self.loc][2]],
                                                    self.Art[self.locDict[self.loc][1]], n=1, r=2)
                self.injections[1] = self.__nrPoint(self.Art[self.locDict[self.loc][1]],
                                                    self.Art[self.locDict[self.loc][0]], n=1, r=2)
                self.injections[2] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                                                    self.Art[0], n=1, r=2)
            elif self.Ploc == 1:
                assert len(self.injections) == 2, "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.injections[0] = self.__nrPoint(self.Art[self.locDict[self.loc][1]],
                                                    self.Art[self.locDict[self.loc][0]], n=1, r=2)
                self.injections[1] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                                                    self.Art[0], n=1, r=2)
                # self.injections[2] = self.__nrPoint(self.Art[self.locDict[self.loc][0]],
                #                                     self.Art[0], n=2, r=3)
            elif self.Ploc == 2:
                assert len(self.injections) == 2, "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.injections[0] = self.__nrPoint(self.Art[2], self.Art[1], n=1, r=2)
                self.injections[1] = self.__nrPoint(self.Art[1], self.Art[0], n=1, r=2)
            elif self.Ploc == -1:
                # l-f & r-b
                if self.handTypeCheck:
                    assert (self.start[0] < self.Art[1][0]) and (self.start[1] < self.Art[1][1]), \
                        "Injection refusée. Essayez la détection des plaies une fois de plus."
                # r-f & l-b
                elif not self.handTypeCheck:
                    assert (self.start[0] > self.Art[1][0]) and (self.start[1] < self.Art[1][1]), \
                        "Injection refusée. Essayez la détection des plaies une fois de plus."
                self.maxInjection = 1
                self.injections = [None for _ in range(self.maxInjection)]
                self.injections[0] = self.__nrPoint(self.start, self.Art[0], n=1, r=2)
        # Le plaie sur le paume
        elif self.loc == 0:
            if len(self.injections) == 2:
                self.injections[0] = self.__nrPoint(self.start, self.Art[0], n=1, r=3)
                self.injections[1] = self.__nrPoint(self.start, self.Art[0], n=2, r=3)
            elif len(self.injections) == 1:
                self.injections[0] = self.__nrPoint(self.start, self.Art[0], n=1, r=2)
        print(f"injections : {self.injections}")

        # enregistrer en txt
        fInj = open(self.inj_path + "injections.txt", "w+")
        for idx, injp in enumerate(self.injections):
            if idx != len(self.injections)-1:
                fInj.write(str(injp) + "\n")
            else:
                fInj.write(str(injp))
        fInj.close()

        return self.injections

    def VisualizeInjectionPoint(self, save=True):
        img = self.__VisualizeStartPoint()
        # Denormalizing
        w, h = img.shape[1], img.shape[0]
        for iinj, injection in enumerate(self.injections):
            dnormX, dnormY = int(injection[0] * w), int(injection[1] * h)
            img = cv2.circle(img, (dnormX, dnormY), 5, (255, 0, 255), -1)
            img = cv2.putText(img, f"injection {iinj+1}", (dnormX + 10, dnormY + 7), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), thickness=2)

        cv2.imshow("Injection", img)
        cv2.waitKey(5000)
        if save:
            cv2.imwrite(self.inj_path + "injectionpoint.jpg", img)
            print(f"Image saved in {os.path.dirname(self.img_path) + '/injectionpoint.jpg'}")

    def __VisualizeStartPoint(self):
        # image to np.array
        img = cv2.imread(self.img_path)
        # Denormalizing : (!) 1920, 1080 automatization
        self.w, self.h = img.shape[1], img.shape[0]
        self.start = (int(self.start[0] * self.w), int(self.start[1] * self.h))

        img = cv2.circle(img, self.start, 5, (255, 0, 0), -1)
        img = cv2.putText(img, "start point", (self.start[0] + 20, self.start[1] + 7), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)

        return img

    # Trouver l'emplacement du centre de plaie
    def __FindLoc(self, handTypeCheck):
        self.findloccnt = 0
        if ((self.big - self.small == 1) and ((self.big != 5) and (self.small != 4)))\
                or ((self.big - self.small == 4) and ((self.big == 9 and self.small == 5) or (self.big == 13 and self.small == 9) or (self.big == 13 and self.small == 17))):
            # 1) la pouce
            if not handTypeCheck: # r-f and l-b
                if (self.big in [3, 4]) and (self.cx >= self.Art[2][0]):
                    self.loc = 1
                    print(f"\nloc   : {self.loc}")
                    return
                else: pass
            elif handTypeCheck: # r-b and l-f
                if (self.big in [3, 4]) and (self.cx <= self.Art[2][0]):
                    self.loc = 1
                    print(f"\nloc   : {self.loc}")
                    return
                else: pass

            # 2) Les doigts ou le paume
            if ((self.big in [6, 7, 8]) and (self.cy <= self.Art[5][1]))\
                    or ((self.a == 5 and self.b == 9) and (self.cy <= self.Art[5][1])):
                self.loc = 2
            elif ((self.big in [10, 11, 12]) and (self.cy <= self.Art[9][1]))\
                    or ((self.a == 9 and self.b == 5) and (self.cy <= self.Art[9][1]))\
                    or ((self.a == 9 and self.b == 13) and (self.cy <= self.Art[9][1])):
                self.loc = 3
            elif ((self.big in [14, 15, 16]) and (self.cy <= self.Art[13][1]))\
                    or ((self.a == 13 and self.b == 9) and (self.cy <= self.Art[13][1]))\
                    or ((self.a == 13 and self.b == 17) and (self.cy <= self.Art[13][1])):
                self.loc = 4
            elif ((self.big in [18, 19, 20]) and (self.cy <= self.Art[17][1]))\
                    or ((self.a == 17 and self.b == 13) and (self.cy <= self.Art[17][1])):
                self.loc = 5
            else:
                self.loc = 0

        else:
            if self.a in [8, 12, 16, 20]:
                if (self.big - self.small != 1):
                    self.b = self.a - 1
                    if self.a == self.big:
                        self.small = self.b
                    else:
                        self.big = self.b
                    self.big = max(self.small, self.big)
                    self.small = min(self.small, self.big)
                    self.__FindLoc(self.handTypeCheck)
                else:
                    pass
            elif self.a in [6, 7, 10, 11, 14, 15, 18, 19]:
                if (self.b != (self.a + 1)) or (self.b != (self.a - 1)):
                    if self.a in [6, 10, 14, 18]:
                        if self.cy < self.Art[self.a][1]:
                            self.b = self.a + 1
                        else:
                            self.b = self.a - 1
                    elif self.a in [7, 11, 15, 19]:
                        if self.cy < self.Art[self.a][1]:
                            self.b = self.a + 1
                        else:
                            self.b = self.a - 1
                    if self.a == self.big:
                        self.small = self.b
                    else:
                        self.big = self.b
                    self.big = max(self.small, self.big)
                    self.small = min(self.small, self.big)
                    self.__FindLoc(self.handTypeCheck)
                else:
                    pass
            else:
                self.loc = 0

    # Définir le point starting point
    def __DefineStart(self, loc):
        """
            MidNine2Zero : la coordonnée du centre entre Art[9] et Art[0]
            MidNine2Mid : la coordonnée du centre entre Art[9] et MidNine2Zero
            MidMid2Zero : la coordonnée du centre entre MidNine2Zero et Art[0]
        """
        self.MidNine2Zero, self.MidNine2Mid, self.MidMid2Zero = self.__Mid(self.Art)

        # les doigts à part la pouce
        if loc in [2, 3, 4, 5]:
            self.start = self.Art[self.locDict[loc][3]][1]
            self.Ploc = 0
            for i in range(3, -1, -1):
                if (self.cy + self.h / 2) > self.Art[self.locDict[loc][i]][1]:
                    if i == 0:
                        self.Ploc = -1
                        self.start = (self.cx, self.cy + self.h / 2)
                        break
                    self.start = self.Art[self.locDict[loc][i - 1]]
                    self.Ploc += 1
                else:
                    break
        # la pouce
        elif loc == 1:
            self.start = self.Art[self.locDict[loc][2]]
            self.Ploc = 0
            a = self.__CoefDirecteur(self.Art)
            # b = y - a * x
            if not self.handTypeCheck:
                bInj = (self.cy + self.h / 2) - a * (self.cx - self.w / 2)
            elif self.handTypeCheck:
                bInj = (self.cy + self.h / 2) - a * (self.cx + self.w / 2)

            for i in range(2, -1, -1):
                if i == 0:
                    if not self.handTypeCheck:
                        if self.Art[self.locDict[loc][i]][0] >= (self.cx - self.w / 2):
                            self.Ploc = -1
                            self.start = (self.cx - self.w / 2, self.cy + self.h / 2) # à gauche en bas
                            break
                        else:
                            break
                    else:
                        if self.Art[self.locDict[loc][i]][0] <= (self.cx + self.w / 2):
                            self.Ploc = -1
                            self.start = (self.cx + self.w / 2, self.cy + self.h / 2) # à droit en bas
                            break
                        else:
                            break

                b = self.Art[self.locDict[loc][i]][1] - a * self.Art[self.locDict[loc][i]][0]
                if not self.handTypeCheck:
                    if (bInj >= b) or ((self.cx - self.w / 2 <= self.Art[self.locDict[loc][i]][0]) and (self.cy + self.h / 2 >= self.Art[self.locDict[loc][i]][1])):
                        self.start = self.Art[self.locDict[loc][i - 1]]
                        self.Ploc += 1
                    else:
                        break
                else:
                    if (bInj >= b) or (self.cx + self.w / 2 >= self.Art[self.locDict[loc][i]][0]):
                        self.start = self.Art[self.locDict[loc][i - 1]]
                        self.Ploc += 1
                    else:
                        break
        # le paume
        elif loc == 0:
            self.start = (self.cx, self.cy + self.h / 2)
            # le côté de centre du plaie
            self.side = "droit" if self.cx >= self.Art[0][0] else "gauche"
        print(f"Ploc  : {self.Ploc}")
        print(f"start : {self.start}")
        if loc == 0:
            print(f"le centre du plaie se situe à {self.side} du paume.")

    # calcul le coefficient de directeur en vue de calcul de self.start et de Ploc
    # lorsque le plaie se situe sur la pouce
    def __CoefDirecteur(self, Art: dict):
        # (y2 - y1) / (x2 - x1)
        if not self.handTypeCheck:
            a = ((1 - Art[3][1]) - (1 - Art[2][1])) / (Art[3][0] - Art[2][0])
        else:
            a = ((1 - Art[2][1]) - (1 - Art[3][1])) / (Art[2][0] - Art[3][0])
        aOrthogonal = -1 / a
        return aOrthogonal

    # calcul des points centrals de la paume
    def __Mid(self, Art: dict):
        if Art[9][0] >= Art[0][0]:
            xMid = Art[9][0] - (Art[9][0] - Art[0][0]) / 2
            xMid1 = xMid + (Art[9][0] - xMid) / 2
            xMid2 = xMid - (xMid - Art[0][0]) / 2
        else:
            xMid = Art[9][0] + (Art[0][0] - Art[9][0]) / 2
            xMid1 = xMid - (xMid - Art[9][0]) / 2
            xMid2 = xMid + (Art[0][0] - xMid) / 2
        yMid = Art[9][1] + (Art[0][1] - Art[9][1]) / 2
        yMid1 = yMid - (yMid - Art[9][1]) / 2
        yMid2 = yMid + (Art[0][1] - yMid) / 2
        return (xMid, yMid), (xMid1, yMid1), (xMid2, yMid2)

    # calcul d'un point central des deux autres points
    def __nrPoint(self, p1:tuple, p2:tuple, n=1, r=2):
        # assertions
        assert n >= 0, "la valeur n doit être supérieure-égale à 0"
        assert n <= r, "la valeur n doit être inférieure-égale à la valeur r"

        xChecker1 = p1[0] == p2[0]
        yChecker1 = p1[1] == p2[1]
        if xChecker1:
            if yChecker1:
                # echec !!!
                print(f"p1 : {p1} et p2 : {p2} sont les même !")
                x, y = p1[0], p1[1]
            # ligne verticale
            else:
                yChecker2 = p1[1] > p2[1]
                if yChecker2:
                    up, dp = p2, p1
                else:
                    up, dp = p1, p2
                # calcul
                x = p1[0]
                y = up[1] + (dp[1] - up[1]) * (n / r)
        else:
            # ligne horizontale
            if yChecker1:
                xChecker2 = p1[0] > p2[0]
                if xChecker2:
                    rp, lp = p1, p2
                else:
                    rp, lp = p2, p1
                # calcul
                if self.side == "droit":
                    x = rp[0] - (rp[0] - lp[0]) / r * n
                    y = p1[1]
                elif self.side == "gauche":
                    x = lp[0] + (rp[0] - lp[0]) / r * n
                    y = p1[1]
            else:
                xChecker2 = p1[0] > p2[0]
                if xChecker2:
                    rp, lp = p1, p2
                else:
                    rp, lp = p2, p1
                yChecker2 = rp[1] > lp[1]
                if yChecker2:
                    # calcul
                    x = lp[0] + (rp[0] - lp[0]) * (n / r)
                    y = lp[1] + (rp[1] - lp[1]) * (n / r)
                else:
                    # calcul
                    x = rp[0] - (rp[0] - lp[0]) * (1 / r) * n
                    y = rp[1] + (lp[1] - rp[1]) * (1 / r) * n
        return x, y