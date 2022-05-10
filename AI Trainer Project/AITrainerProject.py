import cv2 as cv
import numpy as np
import time
import PoseEstimationModule as pm

cap = cv.VideoCapture("/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/AI Trainer Project/Exercise Video and Image/2022-05-09 20-18-02.mp4")

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, vid = cap.read()
    vid = cv.resize(vid, (1280, 720))
    # vid = cv.imread("/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Exercise Video and Image/test.jpg")

    vid = detector.findPose(vid, False)
    lmList = detector.findPosition(vid, False)
    # print(lmList)

    if len(lmList) != 0:
        # Right Arm
        angle = detector.findAngle(vid, 12, 14, 16)

        # # Left Arm
        #angle = detector.findAngle(vid, 11, 13, 15,False)
        per = np.interp(angle, (230, 270), (0, 100))
        bar = np.interp(angle, (230, 270), (650, 100))
        print(angle, per)

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        cv.rectangle(vid, (1100, 100), (1175, 650), color, 3)
        cv.rectangle(vid, (1100, int(bar)), (1175, 650), color, cv.FILLED)
        cv.putText(vid, f'{int(per)} %', (1100, 75), cv.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw Curl Count
        cv.rectangle(vid, (0, 450), (250, 720), (0, 255, 0), cv.FILLED)
        cv.putText(vid, str(int(count)), (45, 670), cv.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(vid, str(int(fps)), (50, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv.imshow("Video", vid)
    cv.waitKey(1)