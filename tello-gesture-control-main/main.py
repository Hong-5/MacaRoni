#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configargparse

import cv2 as cv

from gestures.tello_gesture_controller import TelloGestureController
from utils import CvFpsCalc

from djitellopy import Tello
from gestures import *

import threading
import time   #추가사항  

def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add("--is_keyboard", help='To use Keyboard control by default', type=bool)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)
    parser.add("--buffer_len",
               help='Length of gesture buffer',
               type=int)

    args = parser.parse_args()

    return args

# 키보드로 모드 선택하는 함수 필요 없을까? 제거해야 할까?
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def main():
    # init global vars
    global gesture_buffer
    global gesture_id
    global battery_status

    # Argument parsing
    args = get_args()
    
    # KEYBOARD_CONTROL = args.is_keyboard
    # 키보드로 control 못하도록 false로 변경
    KEYBOARD_CONTROL = False
    
    WRITE_CONTROL = False
    in_flight = False

    # Camera preparation
    # tello 연결 및 카메라 start
    tello = Tello()     #소켓 여는 함수
    tello.connect()
    tello.streamon()

    cap = tello.get_frame_read()   # cap 변수에 텔로로부터 받아오는 영상 이미지를 담고 프레임 화 대기

    # Init Tello Controllers
    gesture_controller = TelloGestureController(tello)
    
    # 키보드 컨트롤 제거하려 했으나 매개변수로 들어가있는 것이 많아 제거하지 못했음.
    keyboard_controller = TelloKeyboardController(tello)
    
    # 제스처 인식
    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    
    # 입력된 제스처 저장되는 변수
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    def tello_control(key, keyboard_controller, gesture_controller):
        global gesture_buffer
        
        # 키보드 control 삭제하고 if문 없앨 수 있나?
        if KEYBOARD_CONTROL:
            keyboard_controller.control(key)
        else:
            gesture_controller.gesture_control(gesture_buffer)
            #여기서 break 할 수 있는 값 받아와서 tello_control return 값 넣어서 while문 break

    def tello_battery(tello):
        global battery_status
        try:
            battery_status = tello.get_battery()[:-2]
        except:
            battery_status = -1

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1
    battery_status = -1
    up_num = 0

    #tello.move_down(20)
    # while 문 탈출 손가락으로 띄우자마자 
    while True:
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        # esc 입력시 while문 탈출 후 종료됨.
        key = cv.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        
        # space바 입력시 뜨고, 떠 있을 때는 착륙
        elif key == 32:  # Space
            if not in_flight:
                # Take-off drone
                tello.takeoff()
                in_flight = True

            elif in_flight:
                # Land tello
                tello.land()
                in_flight = False

       # elif key == ord('k'):
       #     mode = 0
       #     KEYBOARD_CONTROL = True
       #     WRITE_CONTROL = False
       #     tello.send_rc_control(0, 0, 0, 0)  # Stop moving
       
       # g 입력해도 키보드 컨트롤 불가능 하도록 false로 변경
        elif key == ord('g'):
            KEYBOARD_CONTROL = False
            
       # elif key == ord('n'):
       #     mode = 1
       #     WRITE_CONTROL = True
       #     KEYBOARD_CONTROL = True
   	
    #    if WRITE_CONTROL:
    #        number = -1
    #        if 48 <= key <= 57:  # 0 ~ 9
    #            number = key - 48


        # Camera capture
        image = cap.frame

        debug_image, gesture_id = gesture_detector.recognize(image, number, mode)
        
            
        up_exit = gesture_buffer.add_gesture(gesture_id) 
        
        if up_exit == 'exit':
            print("22222222222222222")
            time.sleep(7)
            # 10으로 바꿔서 해볼 것.
            break
        
        
        #gesture_buffer는 tello_control 함수에 들어감.

        # Start control thread
        threading.Thread(target=tello_control, args=(key, keyboard_controller, gesture_controller,)).start()
        threading.Thread(target=tello_battery, args=(tello,)).start()
        

        #GESTURE = threading.Thread(target=tello_control, args=(key, keyboard_controller, gesture_controller,))
        #GESTURE.start()
        #GESTURE.join()
        
        
        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)
        
        # Battery status and image rendering
        # 배터리 용량 나타냄
        cv.putText(debug_image, "Battery: {}".format(battery_status), (5, 720 - 5),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Tello Gesture Recognition', debug_image)
        
        
        # 만약 이것도 안되면 있는 제스쳐 다 없애버리고 다시 시도
        
        # if gesture_id == 2:
        #     up_num += 1
        #     print("gesture" + str(gesture_id))
        #     print("UP! UP! UP! UP!")
        #     print("2222222222222222222222222222222222222222")
            
        # if up_num == 50:
        #     print("EXIT EXIT EXIT EXIT EXIT")
        #     break
    
    print("While문 탈출")
    # 합칠 때는 안 필요한 부분
    tello.land()
    tello.end()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
