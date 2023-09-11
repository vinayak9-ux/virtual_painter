import cv2 as cv
import  mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


class Hand:

    def __init__(self,mode,max_hand,min_detection,min_tracking):
        self.mode=mode
        self.max_hand=max_hand
        self.min_detection=min_detection
        self.min_tracking=min_tracking
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hand = mp.solutions.hands
        self.hands=self.mp_hand.Hands(static_image_mode=self.mode,
                                      max_num_hands=max_hand,
                                      min_detection_confidence=self.min_detection,
                                      min_tracking_confidence=self.min_tracking
                                      )

    def find_hands(self,image,draw=True):
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(image)
        image=cv.cvtColor(image,cv.COLOR_RGB2BGR)

        if self.results.multi_hand_landmarks:

            for hand_landmarks in self.results.multi_hand_landmarks:

                if draw:
                   self.mp_drawing.draw_landmarks(image,
                                               hand_landmarks,
                                               self.mp_hand.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec((0,255,0),2,1),
                                               self.mp_drawing.DrawingSpec((0,255,0),2,1)
                                               )
        return image



    def findPost(self,image,handPos,draw=False):
        landmark_list=[]
        if self.results.multi_hand_landmarks:

            myhand= self.results.multi_hand_landmarks[handPos]

            for id,lm in enumerate(myhand.landmark):
                #print(id,lm.x*image.shape[1],lm.y*image.shape[0])
                landmark_list.append([id,int(lm.x*image.shape[1]),int(lm.y*image.shape[0])])
            if draw:
                self.mp_drawing.draw_landmarks(image,
                                               myhand,
                                               self.mp_hand.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec((0, 255, 0), 2, 1),
                                               self.mp_drawing.DrawingSpec((0, 255, 0), 2, 1)
                                               )
        return landmark_list

    def finner_up_count(self,List):
        finger_tip=[4,8,12,16,20]
        up_finger=[0,0,0,0,0]

        if len(List) != 0:

            #thumb
            if List[finger_tip[0]][1] < List[finger_tip[4]][1]:
                if List[finger_tip[0]][1] > List[finger_tip[0] - 1][1]:
                    up_finger[0] = 0
                else:
                    up_finger[0] = 1
            else:
                if List[finger_tip[0]][1] < List[finger_tip[0] - 1][1]:
                    up_finger[0] = 0
                else:
                    up_finger[0] = 1

            for i in range(1, 5):
                # 4 finger
                if List[finger_tip[i]][2] < List[finger_tip[i] - 2][2]:
                    up_finger[i] = 1
                else:
                    up_finger[i] = 0
        return up_finger









    def separate_color(self,image):

        mask=cv.inRange(image,(0,0,0),(0,255,0))
        image_output=cv.bitwise_and(image,image,mask=mask)
        image2=cv.cvtColor(image_output,cv.COLOR_BGR2GRAY)
        cv.threshold(image2,25,255,cv.THRESH_BINARY,image2)
        return image2


class FACE:

    def __init__(self,mode,max_face,refine_landmark,min_detection,min_tracking):
        self.mode=mode
        self.max_face=max_face
        self.refine_landmark=refine_landmark
        self.min_detection=min_detection
        self.min_tracking=min_tracking
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(static_image_mode=self.mode,
                                        max_num_faces=self.max_face,
                                        refine_landmarks=self.refine_landmark,
                                        min_tracking_confidence=self.min_tracking,
                                        min_detection_confidence=self.min_detection)

    def find_faces(self,image):
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        self.results=self.face.process(image)
        image=cv.cvtColor(image,cv.COLOR_RGB2BGR)

        if self.results.multi_face_landmarks:

            for  face_landmarks in self.results.multi_face_landmarks:


                self.mp_drawing.draw_landmarks(image,
                                               face_landmarks,
                                               self.mp_face.FACEMESH_CONTOURS,None,
                                               self.mp_drawing.DrawingSpec((0,255,0),2,1)
                                               )
        return image



    def separate_color(self,image):

        mask=cv.inRange(image,(0,0,0),(0,255,0))
        image_output=cv.bitwise_and(image,image,mask=mask)
        image2=cv.cvtColor(image_output,cv.COLOR_BGR2GRAY)
        cv.threshold(image2,25,255,cv.THRESH_BINARY,image2)
        return image2


class POSE:

    def __init__(self,mode,complexity,smooth_landmark,enable_segementation,smooth_segmentation,
                 min_detection_confidence,min_tracking_confidence):
        self.mode=mode
        self.array=[(16,14),(14,12),(12,11),(11,13),(13,15),(12,24),(11,23),(24,23)]
        self.complexity=complexity
        self.smooth_landmark=smooth_landmark
        self.enable_segementation=enable_segementation
        self.smooth_segmentation=smooth_segmentation
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence
        self.mp_drawing=mp.solutions.drawing_utils
        self.mp_pose=mp.solutions.pose
        self.count=0
        self.pose=self.mp_pose.Pose(static_image_mode=self.mode,
                                    model_complexity=self.complexity,
                                    smooth_landmarks=self.smooth_landmark,
                                    enable_segmentation=self.enable_segementation,
                                    smooth_segmentation=self.smooth_segmentation,
                                    min_detection_confidence=self.min_detection_confidence,
                                    min_tracking_confidence=self.min_tracking_confidence)
    def findPose(self,image):
        self.image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        self.result=self.pose.process(self.image)
        self.image=cv.cvtColor(self.image,cv.COLOR_RGB2BGR)


        if self.result.pose_landmarks:
           self.count=0
           for i in range(len(self.array)):

               for j in range(2):
                   width = int(self.result.pose_landmarks.landmark[self.array[i][j]].x * image.shape[1])
                   heignt = int(self.result.pose_landmarks.landmark[self.array[i][j]].y * image.shape[0])

                   cv.circle(image, (width, heignt), 2, (0, 255, 0), 1)
                   self.count += 1

               if self.count>1:
                   w1 = int(self.result.pose_landmarks.landmark[self.array[i][0]].x * image.shape[1])
                   h1 = int(self.result.pose_landmarks.landmark[self.array[i][0]].y * image.shape[0])

                   w2 = int(self.result.pose_landmarks.landmark[self.array[i][1]].x * image.shape[1])
                   h2 = int(self.result.pose_landmarks.landmark[self.array[i][1]].y * image.shape[0])

                   cv.line(self.image,(w1,h1),(w2,h2),(0,255,0),2)




        return self.image


    def segemntatoin(self,image):
        segmented_image = image.copy()

        # Probability threshold in [0, 1] that says how "tight" to make the segmentation. Greater value => tighter.
        tightness = .3

        # Stack the segmentation mask for 3 RGB channels, and then create a filter for which pixels to keep
        condition = np.stack((self.result.segmentation_mask,) * 3, axis=-1) > tightness

        # Creates a black background image
        bg_image = np.zeros(image.shape, dtype=np.uint8)

        # Can change the color of this background by specifying (0-255) RGB values. We choose green-screen green.
        bg_image[:] = [4, 244, 4]

        # For every pixel location, display the corresponding pixel from the original imgae if the condition in our filter is met (i.e. the probability of being part of the object is above the inclusiogn threshold), or else display corresponding pixel from the background array (i.e. green)
        segmented_image = np.where(condition, segmented_image, bg_image)

        return segmented_image


def main():

    hand_detection=Hand(False,2,0.5,0.7)
    face_detection=FACE(False,1,True,0.5,0.7)
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv.CAP_PROP_FPS, 60)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)

    while cap.isOpened():
        _,image=cap.read()
        cv.imshow("sdfsdfdsf",image)
        image=cv.flip(image,1)
        image=hand_detection.find_hands(image,False)
        landmark_list=hand_detection.findPost(image,0,True)
        if len(landmark_list)!=0:

            x1,y1=landmark_list[4][1],landmark_list[4][2]
            x2,y2=landmark_list[8][1],landmark_list[8][2]
            x=(x1+x2)//2
            y=(y1+y2)//2
            cv.circle(image,(x,y),5,(255,0,255),cv.FILLED)
            cv.line(image,(x1,y1),(x2,y2),(255,0,255),2)
            line_len=int(math.hypot(x1-x2,y1 -y2))
            #volume.SetMasterVolumeLevel(-10.0, None)
            min_volume=volume.GetVolumeRange()[0]
            max_volume = volume.GetVolumeRange()[1]

            vol=np.interp(line_len,[0,200],[min_volume,max_volume])
            print(line_len)
            volume.SetMasterVolumeLevel(vol,None)

            if line_len <40:
                cv.circle(image, (x, y), 5, (0, 255, 0), cv.FILLED)






        #image=face_detection.find_faces(image)
        #image=hand_detection.separate_color(image)
        cv.imshow("window",image)

        if(cv.waitKey(100) & 0xFF==27):
            break;
    cv.destroyWindow("window")



def main1():
    hand_detection = Hand(False, 2, 0.5, 0.5)
    face_detection = FACE(False, 1, True, 0.5, 0.5)
    pose_detection = POSE(False, 1, True, True, True, 0.9, 0.9)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv.CAP_PROP_FPS, 60)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
    i=0
    while cap.isOpened():
        _, image = cap.read()
        #cv.imshow("sdfsdfdsf", image)
        image = cv.flip(image, 1)
        image = face_detection.find_faces(image)
        print(face_detection.results)
        image = pose_detection.findPose(image)
        image = hand_detection.find_hands(image)
        #cv.imshow("df",image)
        #image=hand_detection.separate_color(image)
        #image=pose_detection.segemntatoin(image)

        cv.imshow("window", image)
        cv.imwrite("C:/Users/Acer/PycharmProjects/opencv/tobedelete/encoded" + str(i) + ".jpg",cv.flip(image, 1))

        mask = cv.inRange(image, (0, 0, 0), (0, 255, 0));
        #cv.imshow("dsf", cv.flip(image, 1))
        resul = cv.bitwise_and(image, image, mask=mask)
        image2 = cv.cvtColor(resul, cv.COLOR_BGR2GRAY)
        cv.threshold(image2, 25, 255, cv.THRESH_BINARY, image2)

        cv.imshow(' Face Mesh', cv.flip(image2, 1))
        cv.imwrite("C:/Users/Acer/PycharmProjects/opencv/tobedelete/decode"+str(i)+".jpg",cv.flip(image2, 1))
        i=i+1


        if (cv.waitKey(1) & 0xFF == 27):
            break;
    cv.destroyAllWindows()


if __name__=='__main__':
    #print(mp.solutions.hands.HAND_CONNECTIONS)
    main1()