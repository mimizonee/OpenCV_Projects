import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


#################
wcam, hcam = 640, 480
################

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

pTime = 0
cTime = 0

detector = htm.HandDetector(detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minvol = volRange[0]
maxvol = volRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lm_list = detector.findPositions(img, draw = False)

    if len(lm_list)!= 0 :
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 10, (200,100,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (200,100,255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (200,100,255), 2)
        cv2.circle(img, (cx, cy), 10, (200,100,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)

        vol = np.interp(length, [50, 200], [minvol, maxvol])
        volBar = np.interp(length, [50, 200], [400, 150])
        volPer = np.interp(length, [50, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        print(length, vol)

    cv2.rectangle(img, (50,150), (85,400), (100,255,150), 3)
    cv2.rectangle(img, (50,int(volBar)), (85,400), (100,255,150), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}', (40, 450), cv2.FONT_HERSHEY_PLAIN, 1, (100,255, 150), 2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255,0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


