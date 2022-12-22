from djitellopy import Tello
import time


class TelloGestureController:
    def __init__(self, tello: Tello):
        self.tello = tello
        self._is_landing = False

        # RC control velocities
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0

    def gesture_control(self, gesture_buffer):
        gesture_id = gesture_buffer.get_gesture()
        print("GESTURE", gesture_id)
        # 여기서 코드 전부 삭제하고 2만 남기는 방향.
        
        if not self._is_landing:
            
            # 제스쳐 마다 원하는 동작 구현 가능
            # 제스쳐 별 id 확인
            
            # 손바닥 전체
            if gesture_id == 0:  # Forward
                self.forw_back_velocity = 30
            
            # 주먹 손바닥이 앞으로 오게    
            elif gesture_id == 1:  # STOP
                # 줄바꿈 \ 
                self.forw_back_velocity = self.up_down_velocity = \
                    self.left_right_velocity = self.yaw_velocity = 0
              
            elif gesture_id == 5:  # Back
                self.forw_back_velocity = -30

            elif gesture_id == 2:  # UP
                print("시작 대기중")
                
                # 기다리는 시간 추가
                # 기다리는 시간 잠시 조정 2초로
                time.sleep(2)
                self.tello.takeoff()
                #self.up_down_velocity = 25

            elif gesture_id == 4:  # DOWN
                self.up_down_velocity = -25

            elif gesture_id == 3:  # LAND
                self._is_landing = True
                self.forw_back_velocity = self.up_down_velocity = \
                    self.left_right_velocity = self.yaw_velocity = 0
                self.tello.land()

            elif gesture_id == 6: # LEFT
                self.left_right_velocity = 20
            elif gesture_id == 7: # RIGHT
                self.left_right_velocity = -20
                
            # 해석 필요 왜하는건지.
            # 예상으로는 새로운 모션 만들었을 경우 이 id를 적용하는 것이라 생각
            # 해당 id에 대한 모션은 stop 과 같음.
            elif gesture_id == -1:
                self.forw_back_velocity = self.up_down_velocity = \
                    self.left_right_velocity = self.yaw_velocity = 0
                    
            # tello 함수 - 원하는 방향 설정가능
            self.tello.send_rc_control(self.left_right_velocity, self.forw_back_velocity,
                                       self.up_down_velocity, self.yaw_velocity)
