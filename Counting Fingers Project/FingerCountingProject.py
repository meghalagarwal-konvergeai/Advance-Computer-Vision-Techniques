import cv2 as cv
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Counting Fingers Project/Fingers"
myList = os.listdir(folderPath)
myList.sort()
print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0
detector = htm.handDetector(detectionCon=0)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, vid = cap.read()
    #vid = cv.flip(vid, 1)
    vid = detector.findHands(vid)
    lmList = detector.findPosition(vid, draw=False)

    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)

        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        vid[0:h, 0:w] = overlayList[totalFingers-1]

        cv.rectangle(vid, (20, 225), (170, 425), (0, 255, 0), cv.FILLED)

        cv.putText(vid, str(totalFingers), (45, 375), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(vid, f'FPS: {int(fps)}', (400, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Video", vid)
    cv.waitKey(1)