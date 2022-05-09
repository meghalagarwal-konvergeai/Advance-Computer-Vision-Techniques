import cv2 as cv
import mediapipe as mp
import time
import math
import PoseEstimationModule as pem

cap = cv.VideoCapture('/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Videos/1.mp4')
pTime = 0

detector = pem.poseDetector()

while True:
    success, vid = cap.read()
    vid = detector.findPose(vid)
    lmList = detector.findPosition(vid, draw=False)

    if len(lmList) != 0:
        print(lmList[14])
        cv.circle(vid, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(vid, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Video", vid)
    cv.waitKey(1)