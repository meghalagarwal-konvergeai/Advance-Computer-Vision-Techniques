import cv2 as cv
import mediapipe as mp
import time
 
cap = cv.VideoCapture("/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Videos/2.mp4")
pTime = 0
 
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.6)
 
while True:
    success, vid = cap.read()
 
    vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
    results = faceDetection.process(vidRGB)
    print(results)
 
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(vid, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box

            ih, iw, ic = vid.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            cv.rectangle(vid, bbox, (255, 0, 255), 2)
            cv.putText(vid, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(vid, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv.imshow("Video", vid)
    cv.waitKey(10)