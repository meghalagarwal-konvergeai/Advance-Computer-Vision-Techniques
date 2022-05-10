import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0, trackCon=0):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
 
    def findHands(self, vid, draw=True):
        vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(vidRGB)
        # print(results.multi_hand_landmarks)
    
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(vid, handLms,
                    self.mpHands.HAND_CONNECTIONS)
    
        return vid
    
    def findPosition(self, vid, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = vid.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                #print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(vid, (cx, cy), 5, (255, 0, 255), cv.FILLED)
                
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax                
        
        if draw:
            cv.rectangle(vid, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        print(xmin)
        return self.lmList , bbox
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
    
            # totalFingers = fingers.count(1)
    
        return fingers
    
    def findDistance(self, p1, p2, vid, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
        if draw:
            cv.line(vid, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(vid, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(vid, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(vid, (cx, cy), r, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
    
        return length, vid, [x1, y1, x2, y2, cx, cy]
    
if __name__ == "__main__":
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, vid = cap.read()
        vid = detector.findHands(vid)
        lmList, bbox = detector.findPosition(vid)
        if len(lmList) != 0:
            print(lmList)
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv.putText(vid, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 255), 3)
 
        cv.imshow("Video", vid)
        cv.waitKey(1)