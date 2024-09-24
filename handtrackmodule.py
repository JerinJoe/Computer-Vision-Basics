import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=int(self.max_num_hands),
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0,draw=True):

        lmList =[] 
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        
        return lmList

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    detector = handDetector()
    pTime = time.time()  # Initialize pTime before the loop
    
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0  # Avoid division by zero
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit loop if 'ESC' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
