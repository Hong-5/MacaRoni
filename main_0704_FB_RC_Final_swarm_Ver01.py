'''
The script allows to control a Tello drone (SDK 2.0) and
streems the video feed to a device where a real-time object
detection runs on the feed.
'''
import cv2
import torch
import djitellopy as tello
import KeyPressModule as kp
import os
#import numpy as np
#from djitellopy import Tello

#import threading
import time

#import pandas as pd

currentPath = os.getcwd()

#import cv2 as cv

#from djitellopy import Tello


#model = torch.load('/home/piai/yolov5/0629_fine_tun/exp/weights/best.pt')
path = 'daebardaebar/tellosibar'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/piai/yolov5/fine_tuning2/exp5/weights/best.pt' ,force_reload=True)
#model = torch.hub.load(path,'yolov5s', 'common', path_or_model='best.pt')

import utils
from utils.downloads import attempt_download

# Intitialize drones
kp.init()
me1 = tello.Tello1()
me2 = tello.Tello2()
me3 = tello.Tello3()

me1.connect()
me2.connect()
me3.connect()

print(me1.get_battery())
print(me2.get_battery())
print(me3.get_battery())


global img

# Initiate video stream
# me.streamon()
me1.streamon()

me1.takeoff()
#time.sleep(0.1)
me1.move('up', 60)


me2.takeoff()
#time.sleep(0.1)
me1.move('up', 30)
me2.move('up', 60)

me3.takeoff()
me2.move('up', 30)
me3.move('up', 90)

print(me1.get_battery())
print(me2.get_battery())
print(me3.get_battery())

#temp_df = pd.DataFrame(columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
# Set control of drone
def getKeyboardInput():

    #global temp_df
    '''
    Function allows the control of the drone via keyboard.

        Parameters:
            none
        Returns:
            [lr, fb, ud, yv]: list of integers
            lr: left, right
            fb: forwards, backwards
            ud: up, down
            yv: yaw velocity
    '''
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    #
    #
    # if kp.getKey('LEFT'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['LEFT']}), ignore_index=True)
    #     lr = -speed
    #
    # elif kp.getKey('RIGHT'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['RIGHT']}), ignore_index=True)
    #     lr = speed
    #
    # if kp.getKey('UP'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['FORWARD']}), ignore_index=True)
    #     fb = speed
    #
    # elif kp.getKey('DOWN'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['BACKWARD']}), ignore_index=True)
    #     fb = -speed
    #
    # if kp.getKey('w'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['UP']}), ignore_index=True)
    #     ud = speed
    #
    # elif kp.getKey('s'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['DOWN']}), ignore_index=True)
    #     ud = -speed
    #
    # if kp.getKey('a'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['SPEED_UP']}), ignore_index=True)
    #     yv = speed
    #
    # elif kp.getKey('d'):
    #     #temp_df = temp_df.append(pd.DataFrame({'DIR' : ['SPEED_DOWN']}), ignore_index=True)
    #     yv = -speed
    #
    # if kp.getKey('q'):
    #     me.land();
    #     time.sleep(3)
    #
    # if kp.getKey('t'):
    #     me.takeoff()
    #
    # if kp.getKey('z'):
    #     cv2.imwrite(f'{time.time()}.jpg', img)  # Choose path to store the image
    #     time.sleep(.3)

    if kp.getKey('x'):
        #print(temp_df)
        #temp_df.to_csv('temp_df_2140.csv', index=False)
        me1.land()
        me1.end()
        time.sleep(0.3)

        me2.land()
        me2.end()
        time.sleep(0.3)

        me3.land()
        me3.end()
        time.sleep(0.3)

        exit()

    return [lr, fb, ud, yv]


# Control the drone
# Stream the video stream with object detection on screen
#6800 <-> 7800
fb_move_velo = 70
lr_move_velo = 70

while True:
    vals = getKeyboardInput()
    me1.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    img = me1.get_frame_read().frame
    img_detect = model(img, size = 640)
    df = img_detect.pandas().xyxy[0]
    print(df)
    img_show = img_detect.render()
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    image_box_center = (480,360)
    me1.send_rc_control(0, 0, 0, 0)
    me2.send_rc_control(0, 0, 0, 0)
    me3.send_rc_control(0, 0, 0, 0)

    if len(df) != 0:
        xg1 = df.loc[0]['xmin']
        xg2 = df.loc[0]['xmax']
        yg1 = df.loc[0]['ymin']
        yg2 = df.loc[0]['ymax']
        target_width = xg2-xg1
        target_height = yg2-yg1
        bbox_area = target_width * target_height

        # bounding Box center
        bbox_center = ((xg2+xg1)/2, (yg2+yg1)/2)

        #distance_between_center = np.sqrt((bbox_center[0] - image_box_center[0] ) ** 2 + (bbox_center[1] - image_box_center[1]) ** 2)
        distance_between_center = abs(bbox_center[0] - image_box_center[0])

        if (6500 <= bbox_area) & (bbox_area <= 7800):
            me1.send_rc_control(0, 0, 0, 0)
            me2.send_rc_control(0, 0, 0, 0)
            me3.send_rc_control(0, 0, 0, 0)
            if distance_between_center <= 100:
                me1.send_rc_control(0, 0, 0, 0)
                me2.send_rc_control(0, 0, 0, 0)
                me3.send_rc_control(0, 0, 0, 0)
            else:
                if image_box_center[0] < bbox_center[0]:
                    me1.send_rc_control(lr_move_velo, 0, 0, 0)
                    me2.send_rc_control(lr_move_velo, 0, 0, 0)
                    me3.send_rc_control(lr_move_velo, 0, 0, 0)

                else:
                    me1.send_rc_control(-lr_move_velo, 0, 0, 0)
                    me2.send_rc_control(-lr_move_velo, 0, 0, 0)
                    me3.send_rc_control(-lr_move_velo, 0, 0, 0)

        else:
            if 6500 > bbox_area:
                if distance_between_center <= 100:
                    me1.send_rc_control(0, fb_move_velo, 0, 0)
                    me2.send_rc_control(0, fb_move_velo, 0, 0)
                    me3.send_rc_control(0, fb_move_velo, 0, 0)

                else:
                    if image_box_center[0] < bbox_center[0]:
                        me1.send_rc_control(lr_move_velo, fb_move_velo, 0, 0)
                        me2.send_rc_control(lr_move_velo, fb_move_velo, 0, 0)
                        me3.send_rc_control(lr_move_velo, fb_move_velo, 0, 0)

                    else:
                        me1.send_rc_control(-lr_move_velo, fb_move_velo, 0, 0)
                        me2.send_rc_control(-lr_move_velo, fb_move_velo, 0, 0)
                        me3.send_rc_control(-lr_move_velo, fb_move_velo, 0, 0)

            else:
                if distance_between_center <= 100:
                    me1.send_rc_control(0, -fb_move_velo, 0, 0)
                    me2.send_rc_control(0, -fb_move_velo, 0, 0)
                    me3.send_rc_control(0, -fb_move_velo, 0, 0)

                else:
                    if image_box_center[0] < bbox_center[0]:
                        me1.send_rc_control(lr_move_velo, -fb_move_velo, 0, 0)
                        me2.send_rc_control(lr_move_velo, -fb_move_velo, 0, 0)
                        me3.send_rc_control(lr_move_velo, -fb_move_velo, 0, 0)

                    else:
                        me1.send_rc_control(-lr_move_velo, -fb_move_velo, 0, 0)
                        me2.send_rc_control(-lr_move_velo, -fb_move_velo, 0, 0)
                        me3.send_rc_control(-lr_move_velo, -fb_move_velo, 0, 0)