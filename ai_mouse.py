import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import math
import time

# --- CONFIGURATION ---
w_cam, h_cam = 640, 480       
frame_red = 180               
smoothening = 7               

# --- TUNING (Super Slow) ---
scroll_speed = 0.2            # CHANGED: Was 1. Now 0.2 (Requires LOTS of movement)
click_threshold = 50          
right_click_threshold = 50    
scroll_start_dist = 30        
scroll_end_dist = 50          

# --- SETUP ---
cv2.namedWindow("AI Mouse", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

w_screen, h_screen = pyautogui.size()

ploc_x, ploc_y = 0, 0 
cloc_x, cloc_y = 0, 0 

# Scroll State
prev_y_scroll = 0
is_scrolling = False  

def get_distance(p1, p2, img, draw=True, r=15, t=3):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
    return length, img, [x1, y1, x2, y2, cx, cy]

print("AI Mouse Started. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    cv2.rectangle(img, (frame_red, frame_red), (w_cam - frame_red, h_cam - frame_red),
                  (0, 255, 0), 2)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            fingers = []
            if lm_list[4][1] > lm_list[3][1]: fingers.append(1)
            else: fingers.append(0)
            for id in [8, 12, 16, 20]:
                if lm_list[id][2] < lm_list[id - 2][2]: fingers.append(1)
                else: fingers.append(0)

            # --- CALCULATE DISTANCES ---
            len_index, img, _ = get_distance(lm_list[8][1:], lm_list[4][1:], img, draw=False)
            len_middle, img, line_info_mid = get_distance(lm_list[12][1:], lm_list[4][1:], img, draw=True)
            len_ring, img, line_info_ring = get_distance(lm_list[16][1:], lm_list[4][1:], img, draw=True)

            # --- STATE MACHINE ---
            if not is_scrolling:
                if len_middle < scroll_start_dist: 
                    is_scrolling = True
                    prev_y_scroll = lm_list[12][2]
            else:
                if len_middle > scroll_end_dist: 
                    is_scrolling = False
                    prev_y_scroll = 0

            # --- ACTION PRIORITY ---
            
            # 1. SCROLLING (Middle Finger Grab)
            if is_scrolling:
                cv2.circle(img, (line_info_mid[4], line_info_mid[5]), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "SCROLL (HEAVY)", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                current_y = lm_list[12][2]
                delta_y = current_y - prev_y_scroll
                
                # We calculate the fractional scroll
                raw_scroll = -delta_y * scroll_speed
                scroll_amount = int(raw_scroll)
                
                # ONLY scroll if the movement was big enough to generate at least 1 unit
                if scroll_amount != 0:
                    pyautogui.scroll(scroll_amount) 
                    prev_y_scroll = current_y # Only update position if we actually scrolled

            # 2. RIGHT CLICK (Ring Finger Pinch)
            elif len_ring < right_click_threshold:
                cv2.circle(img, (line_info_ring[4], line_info_ring[5]), 15, (255, 0, 0), cv2.FILLED)
                cv2.putText(img, "RIGHT CLICK", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                pyautogui.rightClick()
                time.sleep(0.4)

            # 3. LEFT CLICK (Index Pinch)
            elif len_index < click_threshold:
                cv2.circle(img, (lm_list[8][1], lm_list[8][2]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                time.sleep(0.2)

            # 4. MOVE CURSOR (Index Finger Up)
            elif fingers[1] == 1:
                x1, y1 = lm_list[8][1:]
                x3 = np.interp(x1, (frame_red, w_cam - frame_red), (0, w_screen))
                y3 = np.interp(y1, (frame_red, h_cam - frame_red), (0, h_screen))
                
                if abs(x3 - ploc_x) > 3 or abs(y3 - ploc_y) > 3:
                    cloc_x = ploc_x + (x3 - ploc_x) / smoothening
                    cloc_y = ploc_y + (y3 - ploc_y) / smoothening
                    
                    try: pyautogui.moveTo(cloc_x, cloc_y)
                    except: pass
                    ploc_x, ploc_y = cloc_x, cloc_y

    cv2.imshow("AI Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()