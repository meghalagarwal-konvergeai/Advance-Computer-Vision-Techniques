import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

brushThickness = 25
eraserThickness = 100

folderPath = "/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/AI Virtual Painter/Header"
myList = os.listdir(folderPath)
myList.sort()
print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

#overlayList.sort()
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0,maxHands=1)
xp, yp = 0, 0
vidCanvas = np.zeros((480, 640, 3), np.uint8)

while True:

    # 1. Import image
    success, vid = cap.read()
    vid = cv.flip(vid, 1)
    #print(vid.shape)

    # 2. Find Hand Landmarks
    vid = detector.findHands(vid)
    lmList = detector.findPosition(vid, draw=False)

    if len(lmList) != 0:
        print(lmList)
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection Mode – Two finger are up
        if fingers[1] and fingers[2]:
            # xp, yp = 0, 0
            #cv.rectangle(vid, (x1, y1-25), (x2, y2+25), (255, 0, 255), cv.FILLED)
            print("Selection Mode")

            # Checking for the click
            if y1 < 100:
                if 50 < x1 < 150:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 150 < x1 < 200:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 240 < x1 < 390:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 490 < x1 < 640:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                cv.rectangle(vid, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)

        # 5. If Drawing Mode – Index finger is up
        if fingers[1] and fingers[2] == False:
            cv.circle(vid, (x1, y1), 15, drawColor, cv.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            #cv.line(vid, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv.line(vid, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(vidCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)            
            else:
                cv.line(vid, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(vidCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

            # # Clear Canvas when all fingers are up
            # if all (x >= 1 for x in fingers):
            # vidCanvas = np.zeros((720, 1280, 3), np.uint8)

    vidGray = cv.cvtColor(vidCanvas, cv.COLOR_BGR2GRAY)
    _, vidInv = cv.threshold(vidGray, 50, 255, cv.THRESH_BINARY_INV)
    vidInv = cv.cvtColor(vidInv,cv.COLOR_GRAY2BGR)
    vid = cv.bitwise_and(vid,vidInv)
    vid = cv.bitwise_or(vid,vidCanvas)

    # Setting the header image
    vid[0:100, 0:640] = header
    #vid = cv.addWeighted(vid,0.5,vidCanvas,0.5,0)
    cv.imshow("Video", vid)
    #cv.imshow("Video Canvas", vidCanvas)
    #print(vidCanvas.shape)
    # cv.imshow("Video Inv", vidInv)
    # print(vidInv.shape)
    cv.waitKey(1)