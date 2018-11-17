import os
import time
import numpy as np
import cv2

# paramater
save_dir_name = input("Please Enter the save directory >> ")
save_image_name = "image"
save_image_size = 100,100
time_interval = 0.5

time_start = int(time.time())
cap = cv2.VideoCapture(0)


def image_convert(im):
    frame_position = im[0:250, 0:250]
    gray = cv2.cvtColor(frame_position, cv2.COLOR_RGB2GRAY)
    _,th = cv2.threshold(gray, 50,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return [frame_position, th]


def save_dir(name):
    dir_name = name
    dir_list = os.listdir("./")
    if (dir_name in dir_list):
        pass
    else:
        os.mkdir(dir_name)
    return dir_name
save_dir = save_dir(save_dir_name)

camera_num = 0
while True:
    _,frame = cap.read()
    frame = cv2.flip(frame, 1)
    camera = image_convert(frame)

    cv2.imshow("frame", camera[0])
    cv2.imshow("binary", camera[1])
    elepsed_time = int(time.time()) - time_start
    K = cv2.waitKey(1)&0xFF
    if K == ord("q"):
        break
    elif K == ord("m"):
        camera_num += 1
        save_image = "./" + save_dir + "/"+ save_image_name\
                     + str(camera_num) + ".jpg"
        cv2.imwrite(save_image, camera[0])

cv2.destroyAllWindows()
cap.release()
