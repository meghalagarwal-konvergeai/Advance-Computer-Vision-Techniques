import cv2 as cv
import mediapipe as mp
import time
 
cap = cv.VideoCapture("/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Videos/2.mp4")
pTime = 0
 
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
 
while True:
    success, vid = cap.read()
    vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
    results = faceMesh.process(vidRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(vid, faceLms, mpFaceMesh.FACE_CONNECTONS,drawSpec,drawSpec)
        for id,lm in enumerate(faceLms.landmark):
            ih, iw, ic = vid.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
            print(id,x,y)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(vid, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Video", vid)
    cv.waitKey(1)