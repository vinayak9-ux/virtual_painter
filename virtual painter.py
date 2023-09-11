import numpy as np
import mediapipe as mp
import  cv2 as cv
import face_and_hand_module as hand_module

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FPS, 60)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
hand_obj=hand_module.Hand(False,2,0.7,0.7)
image_canvas=np.zeros((720,1280,3),dtype="uint8")


def finner_up_count(List,up_finger):
    if len(List)!=0:


        if List[finger_tip[0]][1]<List[finger_tip[4]][1]:
          if List[finger_tip[0]][1] > List[finger_tip[0]-1][1]:
              up_finger[0]=0
          else:
              up_finger[0]=1
        else:
            if List[finger_tip[0]][1] < List[finger_tip[0] - 1][1]:
                up_finger[0] = 0
            else:
                up_finger[0] = 1



        for i in range(1,5):
            #4 finger
            if List[finger_tip[i]][2] < List[finger_tip[i]-2][2]:
                up_finger[i]=1
            else:
                up_finger[i]=0
    return List,up_finger

            #thumb
xp,yp=0,0
while cap.isOpened():
    _,image=cap.read()
    cv.flip(image,1,image)
    finger_tip=[4,8,12,16,20]
    up_finger=[0,0,0,0,0]

    #hand find part
    hand_obj.find_hands(image,False)
    List=hand_obj.findPost(image,0,True)
    finner_up_count(List,up_finger)



    if up_finger.count(1)>2 or up_finger.count(0)==5:
        xp,yp=0,0


    elif up_finger[1] and up_finger[2]:
        print("erasing mode")
        x = List[finger_tip[1]][1]
        y = List[finger_tip[1]][2]
        x1,y1=List[finger_tip[2]][1:]

        cv.circle(image_canvas, (x, y), 35, (0, 0, 0), cv.FILLED)
        xc=(x+x1)//2
        yc=(y+y1)//2
        cv.circle(image, (xc, yc), 35, (0, 0, 255), cv.FILLED)


        xp, yp = x, y



    elif up_finger[1]:
        x=List[finger_tip[1]][1]
        y=List[finger_tip[1]][2]


        cv.circle(image,(x,y),30,(0,255,0),cv.FILLED)

        print("drawing mode")

        if xp==0 and yp==0:
            xp,yp=x,y
        cv.line(image,(xp,yp),(x,y),(0,255,0),15)
        cv.line(image_canvas, (xp, yp), (x, y), (0, 255, 0), 15)

        xp,yp=x,y

    mask=cv.inRange(image_canvas,(0,20,0),(0,255,0))
    print(up_finger)
    image2=cv.bitwise_and(image_canvas,image_canvas,mask=mask)
    image=cv.add(image,image2)
    cv.imshow("window",image)
    #cv.imshow("window1", image_canvas)
    if (cv.waitKey(1) & 0xFF==27):
        break;
