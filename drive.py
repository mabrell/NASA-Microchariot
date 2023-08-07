import can
import struct
import time
import numpy as np
import json
import math
import os
import asyncio
import websockets
import threading
import picologging as logging
import sys
sys.path.append(os.path.join(os.getcwd(), 'submodules/NASA-IMU-Wrapper'))
from PoseEstimator import PoseEstimator
from typing import Any
from simple_pid import PID




addr = os.environ.get('ADDR', '172.20.10.6')
port = int(os.environ.get('PORT', '8000'))

vehicle_width = 98.425 / 100.0
vehicle_radius = 23.6855 / 100
gear_ratio = 30
rpm_scale = 100

enabled = False
start_auto = False
x_axis = 0
y_axis = 0

voltage_cutoff = 40

from collections import namedtuple
Point = namedtuple('Point', 'x y theta')

p = PoseEstimator(alpha=0.7, freq=20, imu_filter_type="M", use_mag=False)
p.reset(206)
pid = PID(1, 0.1, 0.05, setpoint = 0) #this is the PID tuning values and the setpoint!!! 
pid.sample_time = 0.02 #Also this PID timing value needs to be adjusted it is currently set at 

def ramsete(target: Point, actual: Point, v, omega, b=2, zeta=0.7):
    e = np.array([[math.cos(actual.theta), math.sin(actual.theta), 0], [-math.sin(actual.theta), math.cos(actual.theta), 0], [0, 0, 1]]) @ np.array([[target.x - actual.x], [target.y - actual.y], [target.theta - actual.theta]])
    k = 2 * zeta * math.sqrt(omega**2 + b * v**2)
    v_out = v * math.cos(e[2]) + k * e[0]
    omega_out = omega + k * e[2] + (b * v * math.sin(e[2]) * e[1])/e[2]

def dead_band(left, right, left_dead, right_dead):
    if abs(left) <= left_dead:
        left = 0
    if abs(right) <= right_dead:
        right = 0

    return (left, right)

def teleop_drive():
    global x_axis
    global y_axis
    left = math.copysign(min(abs(y_axis - x_axis), 1.0), y_axis - x_axis)
    right = math.copysign(min(abs(y_axis + x_axis), 1.0), y_axis + x_axis)
    left, right = dead_band(left, right, 0.1, 0.1)
    return (left, right)

def forward_kinematics(w_l, w_r, b, r, phi):
    V = r * (w_l + w_r) / 2
    w = (w_l - w_r) / b
    v = np.array([[math.cos(phi), 0], [math.sin(phi), 0], [0, r]]) @ np.array([[V], [w]])
    return (v[0], v[1], v[2])

def wheel_speeds(V, w, b, r):
    #print('wheel speeds call -----------')
    #print('linear velocity ', V, 'm/s')
    #print('angular velocity ', w, 'rad/s')
    w *= 1.6 # fudge factor for turning
    left_speed = (V - w * b / 2) / r
    right_speed = (V + w * b / 2) / r
    #print('left speed', left_speed)
    #print('right speed', right_speed)
    #print('end wheel speeds call -------')
    return left_speed, right_speed

def radps_to_rpm(rad):
    return (rad / (2 * math.pi)) * 60

def rpm_to_radps(rpm):
    return (rpm * 2 * math.pi) / 60

class RPMListener(can.Listener):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.is_stopped = False

        self.lf_rpm = 0
        self.rf_rpm = 0
        self.lr_rpm = 0
        self.rr_rpm = 0

        self.lf_v_in = 0.0
        self.rf_v_in = 0.0
        self.lr_v_in = 0.0
        self.rr_v_in = 0.0


    def on_message_received(self, msg: can.Message) -> None:
        if self.is_stopped:
            raise RuntimeError("reader has already been stopped")
        else:
            if msg.arbitration_id == 0x1b01:
                self.lf_v_in = self.get_v_in(msg)
            if msg.arbitration_id == 0x1b02:
                self.rf_v_in = self.get_v_in(msg)
            if msg.arbitration_id == 0x1b03:
                self.lr_v_in = self.get_v_in(msg)
            if msg.arbitration_id == 0x1b04:
                self.rr_v_in = self.get_v_in(msg)

            if msg.arbitration_id < 0x901 or msg.arbitration_id > 0x904:
                return

            if msg.arbitration_id == 0x901:
                self.lf_rpm = self.get_rpm(msg)
            if msg.arbitration_id == 0x902:
                self.rf_rpm = self.get_rpm(msg)
            if msg.arbitration_id == 0x903:
                self.lr_rpm = self.get_rpm(msg)
            if msg.arbitration_id == 0x904:
                self.rr_rpm = self.get_rpm(msg)


    def stop(self) -> None:
        self.is_stopped = True

    def get_v_in(self, msg: can.Message) -> float:
        return int.from_bytes(msg.data[4:6], byteorder='big') / 10

    def get_rpm(self, msg: can.Message) -> int:
        erpm = int.from_bytes(msg.data[0:4], byteorder='big', signed=True)
        rpm = -erpm / (6)
        return rpm



def format_can_set_erpm_data(erpm):
    erpm = int(round(erpm))
    return erpm.to_bytes(4, byteorder='big', signed=True)

def drive(lf_rpm, rf_rpm, lr_rpm, rr_rpm):
    lf_data = format_can_set_erpm_data(lf_rpm * -6)
    rf_data = format_can_set_erpm_data(rf_rpm * -6)
    lr_data = format_can_set_erpm_data(lr_rpm * -6)
    rr_data = format_can_set_erpm_data(rr_rpm * -6)

    msg_lf = can.Message(arbitration_id=0x301, data = lf_data)
    msg_rf = can.Message(arbitration_id=0x302, data = rf_data)
    msg_lr = can.Message(arbitration_id=0x303, data = lr_data)
    msg_rr = can.Message(arbitration_id=0x304, data = rr_data)
    return (msg_lf, msg_rf, msg_lr, msg_rr)

def diag_data():
    # volt = odrv0.vbus_voltage
    # l_vel = odrv0.axis0.encoder.vel_estimate
    # r_vel = odrv0.axis1.encoder.vel_estimate
    #_, real_orientation = p.get_pose()
    # return f'{volt}:{l_vel}:{r_vel}:{real_orientation}'
    return ''

async def phandler(websocket):
    while True:
        data = diag_data()
        await websocket.send(data)
        #resp = await websocket.recv()
        #print(resp)
        await asyncio.sleep(200/1000)

async def chandler(websocket):
    global x_axis
    global y_axis
    global start_auto
    global enabled
    while True:
        # try:
        #     async with asyncio.timeout(timeout=0.5):
        #         resp = await websocket.recv()
        #         print(resp)
        #         vals = resp.split(':')
        #         enabled = bool(resp[0])
        #         controller_connected = bool(resp[1])
        #         x_axis = float(resp[2])
        #         y_axis = float(resp[3])
        # except:
        #     enabled = False
        #     controller_connected = False
        #     x_axis = 0.0
        #     y_axis = 0.0
        resp = await websocket.recv()
        # print(resp)
        vals = resp.split(':')
        enabled = (vals[0] == 'True')
        start_auto_ws = (vals[1] == 'True')
        # print(start_auto_ws)
        if start_auto_ws:
            # print('bruhge')
            start_auto = True
        x_axis = float(vals[2])
        y_axis = float(vals[3])
        await asyncio.sleep(20/1000)

async def handler(websocket):
    await asyncio.gather(
        chandler(websocket),
        phandler(websocket),
    )


async def start_server():
    async with websockets.serve(handler, addr, port):
        await asyncio.Future()


async def idk():
    await asyncio.gather(start_server())


def async_main():
    asyncio.run(idk())

th = threading.Thread(target=async_main)
th.start()

if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()

    os.system('sudo ip link set can0 type can bitrate 500000')
    os.system('sudo ip link set can0 up')

    f = open("output.json", "r")
    path_json = json.load(f)

    start_time = time.perf_counter()


    filters = [
        {'can_id': 0x901, 'can_mask': 0x9ff, 'extended': True}, # rpm
        {'can_id': 0x902, 'can_mask': 0x9ff, 'extended': True},
        {'can_id': 0x903, 'can_mask': 0x9ff, 'extended': True},
        {'can_id': 0x904, 'can_mask': 0x9ff, 'extended': True},
        {'can_id': 0x1b01, 'can_mask': 0x1b01, 'extended': True}, # voltage
        {'can_id': 0x1b02, 'can_mask': 0x1b02, 'extended': True},
        {'can_id': 0x1b03, 'can_mask': 0x1b03, 'extended': True},
        {'can_id': 0x1b04, 'can_mask': 0x1b04, 'extended': True},
    ]
    with can.ThreadSafeBus(interface='socketcan',
                    channel='can0',
                    can_filters=filters,
                    receive_own_messages=True) as bus:

        rpm_listener = RPMListener()
        can.Notifier(bus, [rpm_listener])

        while True:
            t1 = 1000 * time.monotonic()

            if rpm_listener.lf_v_in + rpm_listener.rf_v_in + rpm_listener.lr_v_in + rpm_listener.rr_v_in < voltage_cutoff * 4:
                print('battery low')
                lf, rf, lr, rr = drive(0, 0, 0, 0)

                bus.send(lf)
                bus.send(rf)
                bus.send(lr)
                bus.send(rr)
                break

            # drive(odrv0, x_axis, y_axis)
            # drive(odrv0)
            left, right = teleop_drive()
            left_rpm = left * rpm_scale * gear_ratio
            right_rpm = right * rpm_scale * gear_ratio
            lf, rf, lr, rr = drive(left_rpm, right_rpm, left_rpm, right_rpm)
            bus.send(lf)
            bus.send(rf)
            bus.send(lr)
            bus.send(rr)
            logger.info('target ', left_rpm, ' ', right_rpm)
            logger.info('actual ', rpm_listener.lf_rpm, ' ', rpm_listener.rf_rpm, ' ')
            

            # print('start auto is ', start_auto)
            if start_auto:
                counter = 1
                phi = 0
                x = 0
                y = 0


                while counter < len(path_json) - 1:
                    offset_time = time.perf_counter()
                    # speeds = wheel_speeds(0, 0.25, vehicle_width, vehicle_radius)
                
                    _, real_orientation = p.get_pose()
                    target_orientation = (path_json[counter]['pose']['rotation']['radians'] * 180 / 2*math.pi) #this converts from radians to degrees
                    pid.setpoint = target_orientation
                    corrective_angular_velocity = pid(real_orientation)
                    speeds = wheel_speeds(path_json[counter]['velocity'], corrective_angular_velocity, vehicle_width, vehicle_radius)

                    #print(rpm_listener.rf_rpm, ' ', rpm_listener.lf_rpm, path_json['ang_vel'][counter])
                    #print(rpm_listener.v_in)
                    #print(len(path_json['time']) , " " , counter)
                    left_rpm = radps_to_rpm(speeds[0]) * gear_ratio
                    right_rpm = radps_to_rpm(speeds[1]) * gear_ratio
                    #print(right_rpm, ' ', left_rpm)
                    lf, rf, lr, rr = drive(left_rpm, right_rpm, left_rpm, right_rpm)

                    bus.send(lf)
                    bus.send(rf)
                    bus.send(lr)
                    bus.send(rr)


                    # w_l = rpm_to_radps(rpm_listener.lf_rpm)
                    # w_r = rpm_to_radps(rpm_listener.rf_rpm)
                    #w_l = rpm_to_radps(left_rpm / gear_ratio)
                    #w_r = rpm_to_radps(right_rpm / gear_ratio)
                    # print('error ', (rpm_listener.lf_rpm - left_rpm) / gear_ratio, ' ', (rpm_listener.rf_rpm - right_rpm) / gear_ratio)
                    #print('error ', (rpm_listener.lf_rpm / left_rpm) / gear_ratio, ' ', (rpm_listener.rf_rpm / right_rpm) / gear_ratio)
                    # v_x, v_y, v_phi = forward_kinematics(w_l, w_r, vehicle_width, vehicle_radius, phi)
                    # x += v_x * min(0.02, dt)
                    # y += v_y * min(0.02, dt)
                    # phi += v_phi * min(0.02, dt)
                    #print(left_rpm, ' ', right_rpm, path_json['ang_vel'][counter])
                    logger.info('target ', left_rpm, ' ', right_rpm, ' ', path_json[counter]['angularVelocity'])
                    logger.info('actual ', rpm_listener.lf_rpm, ' ', rpm_listener.rf_rpm, ' ')
                    # print('x y phi', x, ' ', y, ' ', phi, ' ', v_phi)
                    #print(path_json['ang_vel'][counter] / v_phi)
                    


                    offset_end_time = time.perf_counter()
                    extra_wait_time = path_json[counter]['time'] - path_json[counter - 1]['time'] - (offset_end_time - offset_time)
                    counter += 1
                    time.sleep(max(0, extra_wait_time))

                print('done running path')
                start_auto = False
                time.sleep(5)

            # time.sleep(0.001 * max(0, 20-(t1-t0)))
            time.sleep(0.05)

            t0 = t1

