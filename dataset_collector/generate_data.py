# usage: python stream_acc.py [mac1] [mac2] ... [mac(n)]
from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep
from threading import Event
import time
import csv
import platform
import sys

import cv2
import heapq

# MAC_ADDRESSES = ['C9:FA:47:09:CB:F4', 'C4:35:74:F9:AD:B7']
MAC_ADDRESSES = ['FD:17:87:5E:64:E4', 'F9:DF:64:AB:BC:35']

corresponding_video_frame = -1

data_dir = './data/'

def get_time_now():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


class State:
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.acc_callback = FnVoid_VoidP_DataP(self.acc_data_handler)
        self.csv_file = ''
        self.time = 0
        self.write_flag = False

    def set_file(self, csv_file):
        self.csv_file = csv_file
        with open(self.csv_file, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow('acc_x, acc_y, acc_z, video_frame_seq, timestamp'.split(', '))

    def clear(self):
        self.samples = 0
        self.csv_file = ''
        self.frame = []
        self.time = 0
        self.write_flag = False

    def acc_data_handler(self, ctx, data):
        if self.write_flag:
            self.samples += 1
            self.time = data.contents.epoch
            value = parse_value(data)
            # print('%s -> %s' % (self.device.address, value))
            self.write([value.x, value.y, value.z, corresponding_video_frame])

    def write(self, frame):
        with open(self.csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(frame+[self.time])
            print('imu ', str(self.device.address), ': ', self.time)

    # def set_flag(self, state):
    #     self.write_flag = state

if __name__ == '__main__':

    # connect to devices
    states = []
    for i in range(len(MAC_ADDRESSES)):
        d = MetaWear(MAC_ADDRESSES[i])
        d.connect()
        print("Connected to " + d.address)
        states.append(State(d))

    # configure devices
    for s in states:
        print("Configuring device")
        libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
        sleep(1.5)

    for i,s in enumerate(states):
        libmetawear.mbl_mw_acc_set_odr(s.device.board, 100)
        libmetawear.mbl_mw_acc_set_range(s.device.board, 4.0)
        libmetawear.mbl_mw_acc_write_acceleration_config(s.device.board)

        acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
        libmetawear.mbl_mw_datasignal_subscribe(acc, None, s.acc_callback)

        libmetawear.mbl_mw_acc_enable_acceleration_sampling(s.device.board)
        libmetawear.mbl_mw_acc_start(s.device.board)


    while input('press 1 to start or other to end: ') == '1':

        # if connected finished, creating imu data file
        timestamp = get_time_now()
        print(timestamp)
        for i in range(len(MAC_ADDRESSES)):
            csv_file = data_dir + timestamp + '_' + str(i) + '.csv'
            states[i].set_file(csv_file)

        # initialize camera
        cap = cv2.VideoCapture(2)

        if not cap.isOpened():
            print("Camera not found.")
            exit()

        # create video data file
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps: ', fps)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print('size: ', size)
        video_file = data_dir + timestamp + '.mp4'
        out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

        timestamp_file = data_dir + timestamp + '.csv'
        with open(timestamp_file, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['sequence', 'timestamp', 'class'])

        print('Start......')
        sleep(1.0)
        start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            video_time = int(time.time()*1000)
            corresponding_video_frame += 1

            # read imu
            for s in states:
                s.write_flag = True

            print('video: ', video_time)
            if ret == True:

                cv2.imshow('frame', frame)
                out.write(frame)
                with open(timestamp_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([corresponding_video_frame, video_time])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cv2.destroyAllWindows()
        cap.release()

        print("Total Samples Received")
        end = time.time()
        for s in states:
            print("%s -> %s" % (s.device.address, s.samples))
            print(s.samples/(end-start))
            s.clear()
        corresponding_video_frame -= corresponding_video_frame+1

    for s in states:

        libmetawear.mbl_mw_acc_stop(s.device.board)
        libmetawear.mbl_mw_acc_disable_acceleration_sampling(s.device.board)

        acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
        libmetawear.mbl_mw_datasignal_unsubscribe(acc)

        libmetawear.mbl_mw_debug_disconnect(s.device.board)
