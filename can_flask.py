#!flask/bin/python
from flask import Flask, json, request
from threading import Thread
from canlib import canlib, Frame
import pythoncom
import time
DEBUG = True


class CAN:
    def __init__(self):
        self.rfid = 0
        self.obstacle_detected = 0
        self.no_obstacle = 0
        self.speed = 0
        self.brake = 1
        self.toggled = 1
        self.t2 = Thread(target=self.initialize, args=())
        self.t2.daemon = True
        self.t2.start()

    def initialize(self):
        while True:
            try:
                self.channel = 0
                self.ch = canlib.openChannel(self.channel, canlib.canOPEN_ACCEPT_VIRTUAL)
                self.ch.setBusOutputControl(canlib.canDRIVER_NORMAL)
                self.ch.setBusParams(canlib.canBITRATE_500K)
                self.ch.busOn()
                t1 = Thread(target=self.work, args=())
                t1.daemon = True
                t1.start()
                while t1.is_alive():
                    time.sleep(1)
            except Exception as e:
                print('Error creating CAN Channel.... Retrying')

    def tearDownChannel(self):
        # self.ch.busOff()
        self.ch.close()

    def send_receive_sig(self):
        self.toggled = 0
        t = 0.1
        frame = Frame(id_=528, data=bytearray(b'\x00\x00'), flags=canlib.MessageFlag.STD)
        self.ch.write(frame)
        time.sleep(t)
        frame = Frame(id_=528, data=bytearray(b'\x01\x00'), flags=canlib.MessageFlag.STD)
        self.ch.write(frame)
        time.sleep(t)
        frame = Frame(id_=528, data=bytearray(b'\x00\x00'), flags=canlib.MessageFlag.STD)
        self.ch.write(frame)
        time.sleep(1)
        self.toggled = 1

    def work(self):
        while True:
            try:
                (msgId, msg, dlc, flg, _) = self.ch.read()
                data = ''.join(format(x, '02x') for x in msg)
                if msgId == 775:
                    req_data_rfid = data[7:8]
                    req_data_obstacle = data[11:12]
                    if req_data_rfid == '1' or req_data_rfid == '3' or req_data_rfid == '4' or req_data_rfid == '2':
                        self.rfid = req_data_rfid
                    if req_data_obstacle == '3':
                        self.obstacle_detected = int(True)
                        self.no_obstacle = int(False)
                        print('obstacle detected', time.time())
                    if req_data_obstacle == '0' and self.obstacle_detected:
                        self.no_obstacle = int(True)
                        self.obstacle_detected = int(False)
                if msgId == 768:
                    # print(self.brake)
                    req_data = data[5:6]
                    # print('Brake:', req_data)
                    if req_data == '1':
                        # Brakes applied
                        self.brake = req_data
                    if req_data == '0':
                        # Brakes removed
                        self.brake = req_data
                if msgId == 769:
                    req_data = data[4:6]
                    hex_int = int(req_data, 16) / 10
                    # if DEBUG:
                    #     print(str(hex_int) + " km/h")

            except canlib.canNoMsg as ex:
                pass
            except canlib.canError as ex:
                break


def start():
    # Initialize
    pythoncom.CoInitialize()
    xl = CAN()
    return xl


app = Flask(__name__)
can = start()


@app.route('/can/', methods=['GET', 'POST'])
def getCanData():
    return json.dumps({'rfid': can.rfid, 'obstacle_detected': can.obstacle_detected, 'no_obstacle': can.no_obstacle, 'speed': can.speed, 'brake': can.brake, 'toggled': can.toggled})


@app.route('/toggle/', methods=['GET', 'POST'])
def toggle():
    can.send_receive_sig()
    return 'True'


@app.route('/teardown/', methods=['GET', 'POST'])
def teardown():
    can.tearDownChannel()
    return 'True'


if __name__ == '__main__':
    app.run(port=5500, debug=True, threaded=True)

