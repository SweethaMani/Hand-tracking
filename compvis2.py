import cv2
import mediapipe as mp
import pyautogui
import math

def hand_mouse_control():
    cam = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    screen_width, screen_height = pyautogui.size()
    clicking = False
    pinch_threshold = 0.03
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )

                index_finger = hand_landmarks.landmark[8]
                x = int(index_finger.x * frame.shape[1])
                y = int(index_finger.y * frame.shape[0])

                screen_x = screen_width * index_finger.x
                screen_y = screen_height * index_finger.y
                pyautogui.moveTo(screen_x, screen_y)

                cv2.circle(frame, (x,y), 10, (0,255,0), -1)

                thumb_tip = hand_landmarks.landmark[4]
                dx = thumb_tip.x - index_finger.x
                dy = thumb_tip.y - index_finger.y
                distance = math.sqrt(dx**2 + dy**2)

                if distance < pinch_threshold and not clicking:
                    pyautogui.click()
                    clicking = True
                elif distance >= pinch_threshold and clicking:
                    clicking = False
        cv2.imshow("Hand Gesture Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    hand_mouse_control()