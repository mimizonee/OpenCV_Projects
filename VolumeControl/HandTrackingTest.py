import cv2
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

detector = htm.HandDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lm_list = detector.findPositions(img)

    if len(lm_list) != 0:
        print(lm_list[4])

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

