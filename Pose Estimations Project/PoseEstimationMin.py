import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture("/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Videos/2.mp4")
pTime = 0

while True:
    success, vid = cap.read()
    vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
    results = pose.process(vidRGB)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(vid, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = vid.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(vid, (cx, cy), 5, (255, 0, 0), cv.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(vid, str(int(fps)), (70, 50), cv.FONT_HERSHEY_TRIPLEX, 3, (0,0,255), 3)

    cv.imshow("Video", vid)
    cv.waitKey(1)