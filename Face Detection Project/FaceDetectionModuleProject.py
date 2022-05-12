import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        
    def findFaces(self, vid, draw=True):
        vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(vidRGB)
        
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = vid.shape
                
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw:
                    vid = self.fancyDraw(vid,bbox)
                    cv.putText(vid, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return vid, bboxs
        
    def fancyDraw(self, vid, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv.rectangle(vid, bbox, (255, 0, 255), rt)
        cv.line(vid, (x, y), (x + l, y), (255, 0, 255), t)
        cv.line(vid, (x, y), (x, y+l), (255, 0, 255), t)
        
        # Top Right  x1,y
        cv.line(vid, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(vid, (x1, y), (x1, y+l), (255, 0, 255), t)
        
        # Bottom Left  x,y1
        cv.line(vid, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(vid, (x, y1), (x, y1 - l), (255, 0, 255), t)
        
        # Bottom Right  x1,y1
        cv.line(vid, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(vid, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        
        return vid         

if __name__ == "__main__":
    cap = cv.VideoCapture("/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Videos/2.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, vid = cap.read()
        vid, bboxs = detector.findFaces(vid)
        print(bboxs)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv.putText(vid, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv.imshow("Video", vid)
        cv.waitKey(10)