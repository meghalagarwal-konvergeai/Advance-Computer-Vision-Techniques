import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=True, minTrackCon=True):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpFaceMeshConnection = mp.solutions.face_mesh_connections.FACEMESH_CONTOURS
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, vid, vidRGB, draw=True):
        self.results = self.faceMesh.process(vidRGB)

        faces =[]
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                print("If is True")
                if draw:
                    self.mpDraw.draw_landmarks(vid, faceLms, self.mpFaceMeshConnection, self.drawSpec, self.drawSpec)
                    face = []

                for id,lm in enumerate(faceLms.landmark):                    
                    ih, iw, ic = vid.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    #cv.putText(vid, str(id), (x, y), cv.FONT_HERSHEY_PLAIN,
                    # 0.7, (0, 255, 0), 1)
                    face.append([x,y])
                    faces.append(face)
        else:
            print("Else is True")

        return vid, faces

if __name__ == "__main__":
    cap = cv.VideoCapture("/home/meghal/Personal/Konverge_AI/Training/Advance Computer Visions/Pose Estimations Project/Videos/2.mp4")
    pTime = 0

    detector = FaceMeshDetector(maxFaces=1)

    while True:
        success, vid = cap.read()
        vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
        vid, faces = detector.findFaceMesh(vid, vidRGB)

        if len(faces)!= 0:
            print(faces[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv.putText(vid, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow("Video", vid)
        cv.waitKey(10)