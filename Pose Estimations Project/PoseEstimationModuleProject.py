import cv2 as cv
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=True, trackCon=True):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, vid, draw=True):
        vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(vidRGB)
 
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(vid, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return vid

    def findPosition(self, vid, draw=True):
        self.lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = vid.shape
                
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv.circle(vid, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        return self.lmList

if __name__ == "__main__":
    cap = cv.VideoCapture('/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Videos/1.mp4')
    pTime = 0

    detector = poseDetector()
    
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