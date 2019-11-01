import config, face_detect, gesture
from multiprocessing import Queue
from threading import Thread
from gesture import GestureRecognitionYoloGpu
import time
import cv2
import uuid
import os
import json
import random
from playsound import playsound
from datetime import datetime
import ssl
import requests
from speak import Speech
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from collections import defaultdict
import admin_check
import face_call
import urllib.request
import logging
import staff_call

if config.DEBUG:
    # Log generation
    LOG_FILENAME = 'data_logging.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # logging.FileHandler(LOG_FILENAME, mode='w')

# # Declaring global variables
greeting_dict = {}
speech_q = Queue(6)
greeting_dict_unknown = {}
greeting_unknown = False
random_greeting = ''
name_to_show = ''
msg_to_show = ''


def draw_border_face(visitor_name, img, pt1, pt2, color, thickness, radius):
    """Draws and Displays Face border and Face Name"""
    # Assigning
    # Outer Rectangle on Face Name

    # cv2.rectangle(frame, (x1+dd-6, y1 - size[0][1] - 45), (x1+dd + size[0][0]+6, y1-20), color, cv2.FILLED)

    # Inner Rectangle on Face Name
    # cv2.rectangle(frame, (x1+dd, y1 - size[0][1] - 40), (x1+dd + size[0][0], y1 - 25), (255, 255, 255), cv2.FILLED)

    # Face Name Writer
    # cv2.putText(frame, visitor_name.split()[0], (x1 + dd, y1 - 34), cv2.FONT_HERSHEY_SIMPLEX, 2,
    #             (100, 100, 100),
    #             3)

    x1, y1 = pt1
    x2, y2 = pt2

    size = cv2.getTextSize(visitor_name.split()[0], cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    # Difference
    dd = int((x2 - x1 - size[0][0])/2)
    # Difference
    d = (int((x2 - x1) / 4))

    if (y1 - size[0][1] - 70) > 125:
        point1 = Point(x1 + dd - 6, y1 - size[0][1] - 45)
        point2 = Point(x1 + dd + size[0][0] + 6, y1 - size[0][1] - 45)
        point3 = Point(x1 + dd + size[0][0] + 6, y1 - 20 - 5)
        point4 = Point(x1 + dd - 6, y1 - 20 - 5)
        if visitor_name != '$@':
            drawEllipseRectangle(frame, point1, point2, point3, point4, 8, 8, color)
            cv2.putText(frame, visitor_name.split()[0], (x1 + dd, y1 - 34), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (100, 100, 100),
                        3)
        # Top left
        cv2.line(img, (x1 + radius, y1), (x1 + radius + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y1 + radius + d), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - radius, y1), (x2 - radius - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y1 + radius + d), color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + radius, y2), (x1 + radius + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - radius), (x1, y2 - radius - d), color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - radius, y2), (x2 - radius - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - radius), (x2, y2 - radius - d), color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

        # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, (0, 255, 255))

    else:
        point1 = Point(x1 + dd - 6, y2 + size[0][1] + 45)
        point2 = Point(x1 + dd + size[0][0] + 6, y2 + size[0][1] + 45)
        point3 = Point(x1 + dd + size[0][0] + 6, y2 + 20 + 5)
        point4 = Point(x1 + dd - 6, y2 + 20 + 5)

        if visitor_name != '$@':
            drawEllipseRectangle(frame, point4, point3, point2, point1, 8, 8, color)
            cv2.putText(frame, visitor_name.split()[0], (x1 + dd, y2 + 34 + size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (100, 100, 100),
                        3)

        # Top left
        cv2.line(img, (x1 + radius, y1), (x1 + radius + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y1 + radius + d), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - radius, y1), (x2 - radius - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y1 + radius + d), color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + radius, y2), (x1 + radius + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - radius), (x1, y2 - radius - d), color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - radius, y2), (x2 - radius - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - radius), (x2, y2 - radius - d), color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_border_gesture(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    d = (int((x2 - x1) / 3))

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[2], b[2]) - x
    h = max(a[3], b[3]) - y
    return (x, y, w, h)


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0: return (0, 0, 0, 0)  # or (0,0,0,0) ?
    return (x, y, w, h)


def getIoU(b1, b2):
    _, _, w, h = union(b1, b2)
    union_area = w * h
    _, _, w, h = intersection(b1, b2)
    intersection_area = w * h
    return float(intersection_area) / float(union_area)


def list2str(bb_list):
    liststr = [str(x) for x in bb_list]
    return '_'.join(liststr)


def str2list(liststr):
    list_items = liststr.split('_')
    return [int(x) for x in list_items]


def drawFaces(frame, final_faces, final_unknown):
    for _id_ in final_faces:
        name, box, fc, _, _, _, _ = final_faces[_id_]
        [x1, y1, x2, y2] = box  # str2list(box_str)
        draw_border_face(name, frame, (x1, y1), (x2, y2), config.FACE_BOX_COLOR, 20, 25)

    for u_bb in final_unknown:
        fc, l_fc, _, _ = final_unknown[u_bb]
        if config.DEBUG:
            logging.debug('count difference SHOW' + str(frame_count - fc))
        # if fc - l_fc >= int(
        #         config_info.get('FRAMES_BUFFER_ADD_UNKNOWN_FACES', config.FRAMES_BUFFER_ADD_UNKNOWN_FACES)) * int(
        #         config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)):
        [x1, y1, x2, y2] = str2list(u_bb)  # str2list(box_str)
        draw_border_face('$@', frame, (x1, y1), (x2, y2), config.UNKNOWN_FACE_BOX_COLOR, 20, 25)


def sayGreetingMessage(final_faces, final_unknown):
    global greeting_unknown
    global name_to_show
    global msg_to_show
    global random_greeting
    # is_unknown_found_in_names = False
    for _id_ in final_faces:
        name, box, fc, _, _, _, _ = final_faces[_id_]
        # GREETING the passengers
        if name not in greeting_dict and name != '':
            speech_q.put(name)
            greeting_dict[name] = False

    # if greeting_unknown and not is_unknown_found_in_names:
    #     greeting_unknown = False

    if len(greeting_dict) > 0:
        for _ in list(greeting_dict.keys()):
            name1 = ''
            name2 = ''
            if not speech_q.empty():
                name1 = speech_q.get()
            if not speech_q.empty():
                name2 = speech_q.get()
            if not name1 == '':
                if not greeting_dict[name1]:
                    if name1 and name2:
                        greeting_dict[name1] = True
                        greeting_dict[name2] = True
                        random_greeting = random.choice(config.GREETING_SPEAK)
                        msg = random_greeting + ' ' + name1 + ' and ' + name2
                        name_to_show = msg + '.\nWelcome to PPM.'
                        tts.say(msg + '. Welcome to PPM. ')

                    elif name1:
                        greeting_dict[name1] = True
                        random_greeting = random.choice(config.GREETING_SPEAK)
                        msg = random_greeting + ' ' + name1
                        name_to_show = msg + '.\nWelcome to PPM.'
                        tts.say(msg + '. Welcome to PPM. ')

    if len(final_unknown) == 0:
        for name in greeting_dict:
            if not greeting_dict[name]:
                greeting_dict[name] = True
                random_greeting = random.choice(config.GREETING_SPEAK)
                msg = random_greeting + ' ' + name
                name_to_show = msg + '.\nWelcome to PPM.'
                tts.say(msg + '. Welcome to PPM. ')

    # Greeting for the Unknown persons
    if len(final_unknown) > 0:
        to_speak = False
        for u_bb in final_unknown:
            fc, l_fc, is_spoken, cnt = final_unknown[u_bb]
            if config.DEBUG:
                logging.debug('count difference SAY' + str(frame_count - fc))
            if not is_spoken:
                to_speak = True
                final_unknown[u_bb] = [fc, l_fc, True, cnt]
        if to_speak:
            tts.say(config_info.get('UNKNOWN_MSG', config.UNKNOWN_MSG))

    return final_faces, final_unknown


def trackGestures(num, dets, meta, gesture_tracker, frame_count, final_gestures):
    # final_gestures = {}
    gesture_class_names = ['palmup', 'thumbsup']
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                x1 = int(b.x - b.w / 2.)
                y1 = int(b.y - b.h / 2.)
                x2 = int(b.x + b.w / 2.)
                y2 = int(b.y + b.h / 2.)
                gesture = gesture_class_names[i]
                boxes_to_remove = []
                boxes_to_add = {}
                bb = [x1, y1, x2, y2]
                bb_str = list2str(bb)
                for box_str in gesture_tracker:
                    box_frame_count, box_gestures, is_counted, box_is_displayed = gesture_tracker[box_str]
                    if frame_count - box_frame_count >= int(
                            config_info.get('FRAMES_BUFFER_REMOVE_GESTURES', config.FRAMES_BUFFER_REMOVE_GESTURES)):
                        boxes_to_remove.append(box_str)
                        continue
                    box = str2list(box_str)
                    if getIoU(bb, box) > config.GESTURE_TRACKING_THRESH_IOU:
                        boxes_to_remove.append(box_str)
                        if gesture in box_gestures:
                            box_gestures[gesture] += 1
                        else:
                            box_gestures[gesture] = 1
                        boxes_to_add[bb_str] = [frame_count, box_gestures, is_counted, box_is_displayed]

                for box_str in boxes_to_remove:
                    gesture_tracker.pop(box_str, None)

                if bb_str in boxes_to_add:
                    gesture_tracker[bb_str] = boxes_to_add[bb_str]

                if bb_str not in gesture_tracker:
                    box_gestures = {gesture: 1}
                    is_counted = False
                    is_displayed = False
                    gesture_tracker[bb_str] = [frame_count, box_gestures, is_counted, is_displayed]

    for box_str in gesture_tracker:
        box_frame_count, box_gestures, is_counted, is_displayed = gesture_tracker[box_str]
        if is_counted:
            continue
        max_freq = 0
        max_freq_gesture = ''
        for gest in box_gestures:
            freq = box_gestures[gest]
            if gest == 'palmup':
                if freq >= int(config_info.get('FRAMES_BUFFER_PALMUP_GESTURES', config.FRAMES_BUFFER_PALMUP_GESTURES)):
                    if max_freq < freq:
                        max_freq_gesture = gest
                        max_freq = freq
            else:
                if freq >= int(config_info.get('FRAMES_BUFFER_GESTURES', config.FRAMES_BUFFER_GESTURES)):
                    if max_freq < freq:
                        max_freq_gesture = gest
                        max_freq = freq
        if max_freq > 0:
            is_counted = True
            final_gestures[max_freq_gesture] += 1
            gesture_tracker[box_str] = box_frame_count, box_gestures, is_counted, is_displayed
    return final_gestures, gesture_tracker


def drawGestures(frame, gesture_tracker, meta, frame_count):
    boxes_to_remove = []
    for box_str in gesture_tracker:
        box_frame_count, box_gestures, is_counted, box_is_displayed = gesture_tracker[box_str]
        if frame_count - box_frame_count > int(
                config_info.get('FRAMES_BUFFER_REMOVE_GESTURES', config.FRAMES_BUFFER_REMOVE_GESTURES)):
            boxes_to_remove.append(box_str)
            continue
        [x1, y1, x2, y2] = str2list(box_str)
        # x1 += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
        # x2 += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
        box_class = ''
        box_class_count = -1
        for gesture in box_gestures:
            if box_class_count < box_gestures[gesture]:
                box_class = gesture
                if config.VEHICLE_STATUS == 0 and box_class == 'palmup':
                    continue
                if config.VEHICLE_STATUS == 1 and box_class == 'thumbsup':
                    continue
                box_class_count = box_gestures[gesture]
            if box_class == 'palmup' and box_class_count >= int(
                    config_info.get('FRAMES_BUFFER_SHOW_PALMUP_GESTURES', config.FRAMES_BUFFER_SHOW_PALMUP_GESTURES)):
                if not box_is_displayed:
                    box_is_displayed = True
                    gesture_tracker[box_str] = [box_frame_count, box_gestures, is_counted, box_is_displayed]
                draw_border_gesture(frame, (x1, y1), (x2, y2), config.GESTURES_BOX_COLORS[box_class], 20, 25, 160)
            elif box_class == 'thumbsup' and box_class_count >= int(
                    config_info.get('FRAMES_BUFFER_SHOW_GESTURES', config.FRAMES_BUFFER_SHOW_GESTURES)):
                if not box_is_displayed:
                    box_is_displayed = True
                    gesture_tracker[box_str] = [box_frame_count, box_gestures, is_counted, box_is_displayed]
                draw_border_gesture(frame, (x1, y1), (x2, y2), config.GESTURES_BOX_COLORS[box_class], 20, 25, 160)
    for box_str in boxes_to_remove:
        gesture_tracker.pop(box_str, None)
    return gesture_tracker


def compare_size(a, b):
    w1 = a[2] - a[0]
    w2 = b[2] - b[0]
    h1 = a[3] - a[1]
    h2 = b[3] - b[1]
    if w1 > w2:
        w = float(w2 / w1) * 100
    else:
        w = float(w1 / w2) * 100
    if h1 > h2:
        h = float(h2 / h1) * 100
    else:
        h = float(h1 / h2) * 100
    if w > 80.0 and h > 80.0:
        return True
    else:
        return False


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def drawEllipseRectangle(img, point1, point2, point3, point4, cornerRadius, thickness, border_color):
    strip_color = defaultdict(lambda: config.DARK_BLUE)
    strip_color[config.WAIT_COLOR] = config.DARK_BLUE
    strip_color[config.STOP_COLOR] = config.DARK_RED
    strip_color[config.START_COLOR] = config.DARK_GREEN

    # Drawing the inner lines
    cv2.line(img, (point1.x + cornerRadius, point1.y), (point2.x - cornerRadius, point2.y), strip_color[border_color],
             thickness)
    cv2.line(img, (point2.x, point2.y + cornerRadius), (point3.x, point3.y - cornerRadius), strip_color[border_color],
             thickness)
    cv2.line(img, (point4.x + cornerRadius, point4.y), (point3.x - cornerRadius, point3.y), strip_color[border_color],
             thickness)
    cv2.line(img, (point1.x, point1.y + cornerRadius), (point4.x, point4.y - cornerRadius), strip_color[border_color],
             thickness)

    # Drawing the inner ellipse
    cv2.ellipse(img, (point1.x + cornerRadius, point1.y + cornerRadius), (cornerRadius, cornerRadius), 180, 0, 90,
                strip_color[border_color], thickness)
    cv2.ellipse(img, (point1.x + cornerRadius, point1.y + cornerRadius), (cornerRadius, cornerRadius), 180, 0, 90,
                config.WHITE_COLOR, cv2.FILLED)
    cv2.ellipse(img, (point2.x - cornerRadius, point2.y + cornerRadius), (cornerRadius, cornerRadius), 270, 0, 90,
                strip_color[border_color], thickness)
    cv2.ellipse(img, (point2.x - cornerRadius, point2.y + cornerRadius), (cornerRadius, cornerRadius), 270, 0, 90,
                config.WHITE_COLOR, cv2.FILLED)
    cv2.ellipse(img, (point3.x - cornerRadius, point3.y - cornerRadius), (cornerRadius, cornerRadius), 0, 0, 90,
                strip_color[border_color], thickness)
    cv2.ellipse(img, (point3.x - cornerRadius, point3.y - cornerRadius), (cornerRadius, cornerRadius), 0, 0, 90,
                config.WHITE_COLOR, cv2.FILLED)
    cv2.ellipse(img, (point4.x + cornerRadius, point4.y - cornerRadius), (cornerRadius, cornerRadius), 90, 0, 90,
                strip_color[border_color], thickness)
    cv2.ellipse(img, (point4.x + cornerRadius, point4.y - cornerRadius), (cornerRadius, cornerRadius), 90, 0, 90,
                config.WHITE_COLOR, cv2.FILLED)

    # Drawing the mid-rectangle
    cv2.rectangle(img, (point1.x + cornerRadius, point1.y), (point3.x - cornerRadius, point3.y), config.WHITE_COLOR,
                  cv2.FILLED)
    cv2.rectangle(img, (point2.x, point2.y + cornerRadius), (point4.x, point4.y - cornerRadius), config.WHITE_COLOR,
                  cv2.FILLED)

    # Drawing the outer lines
    cv2.line(img, (point1.x + cornerRadius, point1.y - thickness), (point2.x - cornerRadius, point2.y - thickness),
             border_color, thickness - 3)

    cv2.line(img, (point2.x + thickness, point2.y + cornerRadius), (point3.x + thickness, point3.y - cornerRadius),
             border_color, thickness - 3)
    cv2.line(img, (point4.x + cornerRadius, point4.y + thickness), (point3.x - cornerRadius, point3.y + thickness),
             border_color, thickness - 3)
    cv2.line(img, (point1.x - thickness, point1.y + cornerRadius), (point4.x - thickness, point4.y - cornerRadius),
             border_color, thickness - 3)

    # Drawing the outer ellipse
    cv2.ellipse(img, (point1.x + cornerRadius, point1.y + cornerRadius),
                (cornerRadius + thickness, cornerRadius + thickness), 180, 0, 90,
                border_color, thickness)
    cv2.ellipse(img, (point2.x - cornerRadius, point2.y + cornerRadius),
                (cornerRadius + thickness, cornerRadius + thickness), 270, 0, 90,
                border_color, thickness)
    cv2.ellipse(img, (point3.x - cornerRadius, point3.y - cornerRadius),
                (cornerRadius + thickness, cornerRadius + thickness), 0, 0, 90,
                border_color, thickness)
    cv2.ellipse(img, (point4.x + cornerRadius, point4.y - cornerRadius),
                (cornerRadius + thickness, cornerRadius + thickness), 90, 0, 90,
                border_color, thickness)


def drawUI(img, border_color, img_width, img_height, trip_started):
    global msg_to_show
    # # Top bar rectangle
    # cv2.rectangle(img, (0, 0), (img_width, int(float(img_height) * config.TOP_BORDER_WIDTH_FACTOR)), border_color, -1)
    # Bottom bar rectangle
    cv2.rectangle(img, (0, int(float(img_height) * (1.0 - config.TOP_BORDER_WIDTH_FACTOR))),
                  (img_width, img_height - 1), border_color, -1)

    thickness = 30
    cornerRadius = 20
    horizontal_margin = 300
    vertical_margin = 25

    point1 = Point(horizontal_margin, int(float(img_height) * (1.0 - 0.15)))
    point2 = Point(img_width - horizontal_margin, point1.y)
    point3 = Point(point2.x, img_height - 1 - vertical_margin)
    point4 = Point(point1.x, point3.y)

    drawEllipseRectangle(img, point1, point2, point3, point4, cornerRadius, thickness, border_color)

    # if len(greeting_dict) == 0 and len(face_unknown) == 0 and trip_started and track_time > 0:
    #     font = ImageFont.truetype('./font/Square721.TTF', 100)
    #     img_pil = Image.fromarray(img)
    #     draw = ImageDraw.Draw(img_pil)
    #     draw.text((point1.x + 200, point1.y + 8), 'Welcome to the PPM', font=font, fill=(100, 100, 100, 255))
    #     img = np.array(img_pil)

    if config.TRACK_FACES:
        font = ImageFont.truetype('./font/Square721.TTF', 50)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        if not trip_started:
            if config.RFID_FLAG == 2:
                msg_to_show = 'Thank you for riding with us. Please be sure \nto visit our booth in the' + \
                              ' South Hall'
            elif config.RFID_FLAG == 4:
                msg_to_show = 'Thank you for riding with us. Please be sure \nto visit our booth in the' + \
                              ' Central Hall'
            draw.text((point1.x + 50, point1.y + 5), msg_to_show, font=font, fill=(100, 100, 100, 255))

        if len(face_names) > 0 and len(face_unknown_final) == 0 and len(greeting_dict) > 0:
            draw.text((point1.x + 50, point1.y + 5), name_to_show, font=font, fill=(100, 100, 100, 255))

        if len(face_unknown_final) > 0:  # and track_time > 10.0:
            msg_to_show = 'We are sorry. We do not recognize your face. \nPlease see an attendant.'
            draw.text((point1.x + 50, point1.y + 5), msg_to_show, font=font, fill=(100, 100, 100, 255))

        if start_face_recognition_greeting_time != 0. and curr_time - start_face_recognition_greeting_time < 5.0:
            msg_to_show = 'We are going to start Face Authentication now.\nPlease look at the screen.'
            draw.text((point1.x + 50, point1.y + 5), msg_to_show, font=font, fill=(100, 100, 100, 255))

        img = np.array(img_pil)

    if config.TRACK_GESTURES:
        font = ImageFont.truetype('./font/Square721.TTF', 50)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((point1.x + 50, point1.y + 5), msg_to_show, font=font,
                  fill=(100, 100, 100, 255))
        img = np.array(img_pil)
    return img


def faceApiCall(img):
    global matched_name
    global matched_box
    global matched_id
    global matched_confidence
    global counter_face
    # resize_factor = int(config_info.get('FRAME_RESIZING_FACE', config.FRAME_RESIZING_FACE))
    # frame_resized = cv2.resize(img, (0, 0), fx=1. / resize_factor, fy=1. / resize_factor)
    frame_resized_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    azure.setFrame(frame_resized_rgb, True)
    if azure.getStatus():
        __matching_boxes = []
        _matched_box = []
        matched_name = []
        matched_box = []
        matched_id = []
        matched_confidence = []
        if config.DEBUG:
            print('Face Recognition API hit')
        matched_name, __matching_boxes, matched_id, matched_confidence = azure.getData()
        if __matching_boxes is not None and len(__matching_boxes) > 0:
            for _x, _y, _w, _h in __matching_boxes:
                # x_, y_, w_, h_ = _x * resize_factor, _y * resize_factor, _w * resize_factor, _h * resize_factor
                # _matched_box.append([x_, y_, x_ + w_, y_ + h_])
                _matched_box.append([_x, _y, _x + _w, _y + _h])
            matched_box = _matched_box
            # for bbx in _matched_box:
            #     bbx[1] += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
            #     bbx[3] += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
            #     matched_box.append(bbx)
    elif counter_face == 0:
        counter_face = 1
        azure.sendFrametoAzure()
    if matched_name == [] or matched_name is None and matched_box == [] or matched_box is None:
        return [], [], [], []
    else:
        return matched_name, matched_box, matched_id, matched_confidence


def get_staff_ids():
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(config.GET_STAFF_ID, context=context) as response:
            if response.getcode() == 200:
                data_ = response.read().decode("utf-8")
                data_dict = json.loads(data_)
                if data_dict is not None:
                    global STAFF_IDS
                    STAFF_IDS = []
                    for row in data_dict:
                        STAFF_IDS.append(row['PersonId'])

                    config.IS_STAFF_IDS_LOADED = True
    except Exception as e:
        print(e)
        pass


class MailUIAdd:

    def __init__(self):
        # For Mail UI Addition
        self.foregroundMail = cv2.imread("./images/cesHeadFoot.png")
        self.foregroundMail = self.foregroundMail.astype(float)
        self.matMail = cv2.imread("./images/cesHeadFootMat.png")
        self.matMail = self.matMail.astype(float) / 255
        self.foregroundMail = cv2.multiply(self.matMail, self.foregroundMail)

    def applyingUIMail(self, img):
        img = img.astype(float)
        img = cv2.multiply(1.0 - self.matMail, img)
        outImage = cv2.add(self.foregroundMail, img)
        return outImage / 255


class UIAdd:

    def __init__(self):
        # For Waiting
        self.foregroundWaiting = cv2.imread("./images/waiting-for-you.png")
        self.foregroundWaiting = self.foregroundWaiting.astype(float)
        self.matForegroundWaiting = cv2.imread("./images/waiting-for-youMat.png")
        self.matForegroundWaiting = self.matForegroundWaiting.astype(float) / 255
        self.foregroundWaiting = cv2.multiply(self.foregroundWaiting, self.matForegroundWaiting)

        # For Arrival
        self.foregroundArrival = cv2.imread("./images/arrival.png")
        self.foregroundArrival = self.foregroundArrival.astype(float)
        self.matForegroundArrival = cv2.imread("./images/arrivalMat.png")
        self.matForegroundArrival = self.matForegroundArrival.astype(float) / 255
        self.foregroundArrival = cv2.multiply(self.foregroundArrival, self.matForegroundArrival)

        # For Central Hall
        self.foregroundCentralHall = cv2.imread("./images/station.png")
        self.foregroundCentralHall = self.foregroundCentralHall.astype(float)
        self.matForegroundCentralHall = cv2.imread("./images/stationMat.png")
        self.matForegroundCentralHall = self.matForegroundCentralHall.astype(float) / 255
        self.foregroundCentralHall = cv2.multiply(self.matForegroundCentralHall, self.foregroundCentralHall)

        # For Next Central Hall
        self.foregroundNextCentralHall = cv2.imread("./images/next-central-hall-station.png")
        self.foregroundNextCentralHall = self.foregroundNextCentralHall.astype(float)
        self.matForegroundNextCentralHall = cv2.imread("./images/next-central-hall-stationMat.png")
        self.matForegroundNextCentralHall = self.matForegroundNextCentralHall.astype(float) / 255
        self.foregroundNextCentralHall = cv2.multiply(self.matForegroundNextCentralHall, self.foregroundNextCentralHall)

        # For South Hall
        self.foregroundSouthHall = cv2.imread("./images/station.png")
        self.foregroundSouthHall = self.foregroundSouthHall.astype(float)
        self.matForegroundSouthHall = cv2.imread("./images/stationMat.png")
        self.matForegroundSouthHall = self.matForegroundSouthHall.astype(float) / 255
        self.foregroundSouthHall = cv2.multiply(self.matForegroundSouthHall, self.foregroundSouthHall)

        # For Next South Hall
        self.foregroundNextSouthHall = cv2.imread("./images/next-south-hall-station.png")
        self.foregroundNextSouthHall = self.foregroundNextSouthHall.astype(float)
        self.matForegroundNextSouthHall = cv2.imread("./images/next-south-hall-stationMat.png")
        self.matForegroundNextSouthHall = self.matForegroundNextSouthHall.astype(float) / 255
        self.foregroundNextSouthHall = cv2.multiply(self.matForegroundNextSouthHall, self.foregroundNextSouthHall)

        # For Ready
        self.foregroundReady = cv2.imread("./images/ready.png")
        self.foregroundReady = self.foregroundReady.astype(float)
        self.matForegroundReady = cv2.imread("./images/readyMat.png")
        self.matForegroundReady = self.matForegroundReady.astype(float) / 255
        self.foregroundReady = cv2.multiply(self.matForegroundReady, self.foregroundReady)

        # For Stop Now
        self.foregroundStopNow = cv2.imread("./images/stop-now.png")
        self.foregroundStopNow = self.foregroundStopNow.astype(float)
        self.matForegroundStopNow = cv2.imread("./images/stop-nowMat.png")
        self.matForegroundStopNow = self.matForegroundStopNow.astype(float) / 255
        self.foregroundStopNow = cv2.multiply(self.matForegroundStopNow, self.foregroundStopNow)

        # For Thumbs Up
        self.foregroundThumbsUp = cv2.imread("./images/thumbs-up.png")
        self.foregroundThumbsUp = self.foregroundThumbsUp.astype(float)
        self.matForegroundThumbsUp = cv2.imread("./images/thumbs-upMat.png")
        self.matForegroundThumbsUp = self.matForegroundThumbsUp.astype(float) / 255
        self.foregroundThumbsUp = cv2.multiply(self.matForegroundThumbsUp, self.foregroundThumbsUp)

        # For Want to Stop
        self.foregroundWantToStop = cv2.imread("./images/want-to-stop.png")
        self.foregroundWantToStop = self.foregroundWantToStop.astype(float)
        self.matForegroundWantToStop = cv2.imread("./images/want-to-stopMat.png")
        self.matForegroundWantToStop = self.matForegroundWantToStop.astype(float) / 255
        self.foregroundWantToStop = cv2.multiply(self.matForegroundWantToStop, self.foregroundWantToStop)

        self.UIDict = {'WAIT_MSG': {'mat': self.matForegroundWaiting, 'foreground': self.foregroundWaiting},
                       'START_MSG': {'mat': self.matForegroundReady, 'foreground': self.foregroundReady},
                       'READY_MSG_TOP': {'mat': self.matForegroundThumbsUp, 'foreground': self.foregroundThumbsUp},
                       'STOP_MSG': {'mat': self.matForegroundStopNow, 'foreground': self.foregroundStopNow},
                       'NEAR_ARRIVAL_STATION_1': {'mat': self.matForegroundNextCentralHall,
                                                  'foreground': self.foregroundNextCentralHall},
                       'NEAR_ARRIVAL_STATION_2': {'mat': self.matForegroundNextSouthHall,
                                                  'foreground': self.foregroundNextSouthHall},
                       'ARRIVAL_STATION_1': {'mat': self.matForegroundCentralHall,
                                             'foreground': self.foregroundCentralHall},
                       'ARRIVAL_STATION_2': {'mat': self.matForegroundSouthHall,
                                             'foreground': self.foregroundSouthHall},
                       'STATION_2': {'mat': self.matForegroundCentralHall,
                                             'foreground': self.foregroundCentralHall},
                       'STATION_1': {'mat': self.matForegroundSouthHall,
                                             'foreground': self.foregroundSouthHall},
                       'READY_MSG': {'mat': self.matForegroundReady, 'foreground': self.foregroundReady},
                       'ARRIVING_MSG': {'mat': self.matForegroundArrival, 'foreground': self.foregroundArrival},
                       'WANT_TO_STOP_MSG': {'mat': self.matForegroundWantToStop,
                                            'foreground': self.foregroundWantToStop}
                       }

    def applyingUI(self, img, state):
        img = img.astype(float)
        mat = self.UIDict[state]['mat']
        foreground = self.UIDict[state]['foreground']
        img = cv2.multiply(1.0 - mat, img)
        outImage = cv2.add(foreground, img)
        return outImage / 255


def drawBoxesPic(frame, dets, num):
    for j in range(num):
        for i in range(len(gesture_class_names)):
            if i == 1 and dets[j].prob[i] > 0:
                b = dets[j].bbox
                x1 = int(b.x - b.w / 2.)
                y1 = int(b.y - b.h / 2.)
                x2 = int(b.x + b.w / 2.)
                y2 = int(b.y + b.h / 2.)
                draw_border_gesture(frame, (x1, y1), (x2, y2), config.GESTURES_BOX_COLORS['thumbsup'], 20, 25, 160)
    return frame


# From here the application starts execution
if __name__ == '__main__':
    # Speech object
    tts = Speech()

    # UI
    ui = UIAdd()

    # Face Recognition API class's object
    azure = face_call.AzureFaceApi()

    # Running as an admin
    if config.RUN_AS_ADMIN:
        rc = 0
        if not admin_check.isUserAdmin():
            print("You're not an admin.", os.getpid())
            rc = admin_check.runAsAdmin()
        else:
            print("You are an admin!", os.getpid())
            rc = 0

    # Loading config file
    print("Loading data from config file...")
    while not (os.path.isfile(config.PATH_TO_CONFIG_UPDATE_FILE)):
        continue
    with open(config.PATH_TO_CONFIG_UPDATE_FILE) as infile:
        config_info = json.load(infile)
    print("Config file loaded")

    # counter for face and staff
    counter_face = 0
    counter_staff = 0

    # getting all the current Staff ids
    print('Loading staff ids...')
    staff = staff_call.StaffApi()
    STAFF_IDS = []
    if staff.getStatus():
        if len(staff.getData()) == 0:
            pass
        else:
            STAFF_IDS = staff.getData()
    elif counter_staff == 0:
        counter_staff = 1
        staff.getData()
    print('Staff id loaded')

    video = cv2.VideoCapture(0)

    config.SKIP_FACE_RECOGNITION = bool(int(config_info.get('SKIP_FACE_RECOGNITION', config.SKIP_FACE_RECOGNITION)))

    # # Creating the Queue for Face Recognition frame processing
    input_q = Queue(5)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    if not config.SKIP_FACE_RECOGNITION:
        # Starting the FR thread
        for i in range(1):
            t = Thread(target=face_detect.worker, args=(input_q, output_q))
            t.daemon = True
            t.start()
        print('Loading face model...')
        while not config.IS_FACE_MODEL_LOADED:
            continue
        print("face model loaded")

    print('Loading gesture model...')
    gesture_obj = GestureRecognitionYoloGpu()
    while not config.IS_GESTURE_MODEL_LOADED:
        continue
    print("gesture model loaded")

    # Generating the trip id
    TRIP_ID = str(uuid.uuid4())
    # setting the frame width
    video.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    # setting the frame height
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    # setting the display window name
    cv2.namedWindow(config.WIN_NAME, cv2.WND_PROP_FULLSCREEN)
    # setting the named window to full screen
    cv2.setWindowProperty(config.WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    num_known_faces = 0
    num_unknown_faces = 0
    # setting the colors and other config messages
    border_color = config.WAIT_COLOR
    msg_to_show_at_top = config.WAIT_MSG
    loc_msg_to_show_at_top = config.WAIT_MSG_LOC
    # Defining the gesture classes
    gesture_class_names = ['palmup', 'thumbsup']



    # Initializing the variables
    face_boxes = []
    matching_names = []
    face_tracker = {}
    final_names = []
    track_time = 0
    track_time2 = 0
    track_time3 = 0
    net = gesture_obj.netMain
    meta = gesture_obj.metaMain
    gesture_counts = {}
    final_gestures = {}
    for i in range(meta.classes):
        final_gestures[gesture_class_names[i]] = 0
    gesture_tracker = {}
    count_current_face = 0
    count_current_thumbsup = 0
    vehicle_speed = 0

    # Greetings done flag
    greetings_done_1 = False
    greetings_done_2 = False
    greetings_done_3 = False
    greetings_done_4 = False
    greeting_photo_1 = False
    greeting_photo_2 = False
    photo_clicked = False
    trip_started = False
    trip_reset = False
    announcements_done = False
    announcements_done_1 = False
    announcements_done_2 = False
    announcements_done_3 = False

    silence_vehicle_status_0 = True
    silence_object_detected = True
    do_gesture_countdown = True
    obstacle_greeting_done1 = False
    obstacle_greeting_done2 = False

    stop_time_object = 0.0
    stop_time_end_trip = 0.0
    stop_time_photo_click = 0.0
    stop_time_photo_click2 = 0.0
    stop_time_ann = 0.0
    stop_time_ann1 = 0.0
    trip_start_time = 0.0
    start_face_recognition_greeting_time = 0.

    # API CHANGES
    frame_count = 0
    _frame_count = 0
    face_boxes = []
    face_names = {}
    face_staff = {}
    face_unknown = {}
    face_unknown_final = {}
    matched_name, matched_box, matched_id, matched_confidence = [], [], [], []

    # variable for unknown persons
    unknown = False
    last_unknown = False
    unknown_count_time = 0.0

    matching_names = []
    matching_boxes = []
    matching_ids = []
    matching_confidence = []

    last_matching_names = []
    last_matching_boxes = []
    last_matching_ids = []
    last_matching_confidence = []

    send_face_boxes = []
    _send_face_boxes = []

    total_persons_visited = {}

    # Pic UI
    mailUi = MailUIAdd()

    font = ImageFont.truetype('./font/Square721.TTF', 80)

    msg_to_show_at_top_dict = {config_info.get('WAIT_MSG', config.WAIT_MSG): {'name': 'WAIT_MSG', 'adjustment': 400},
                               config_info.get('START_MSG', config.START_MSG): {'name': 'START_MSG', 'adjustment': 700},
                               config_info.get('READY_MSG_TOP', config.READY_MSG_TOP): {'name': 'READY_MSG_TOP',
                                                                                        'adjustment': 575},
                               config_info.get('STOP_MSG', config.STOP_MSG): {'name': 'STOP_MSG', 'adjustment': 600},
                               config_info.get('NEAR_ARRIVAL_STATION_1', config.NEAR_ARRIVAL_STATION_1): {
                                   'name': 'NEAR_ARRIVAL_STATION_1',
                                   'adjustment': 275},
                               config_info.get('NEAR_ARRIVAL_STATION_2', config.NEAR_ARRIVAL_STATION_2): {
                                   'name': 'NEAR_ARRIVAL_STATION_2',
                                   'adjustment': 275},
                               config_info.get('ARRIVAL_STATION_1', config.ARRIVAL_STATION_1): {
                                   'name': 'ARRIVAL_STATION_1', 'adjustment': 500},
                               config_info.get('ARRIVAL_STATION_2', config.ARRIVAL_STATION_2): {
                                   'name': 'ARRIVAL_STATION_2', 'adjustment': 500},
                               config_info.get('STATION_1', config.STATION_1): {'name': 'STATION_1', 'adjustment': 500},
                               config_info.get('STATION_2', config.STATION_2): {'name': 'STATION_2', 'adjustment': 500},
                               config_info.get('READY_MSG', config.READY_MSG): {'name': 'READY_MSG', 'adjustment': 700},
                               config_info.get('ARRIVING_MSG', config.ARRIVING_MSG): {'name': 'ARRIVING_MSG',
                                                                                      'adjustment': 700},
                               config_info.get('WANT_TO_STOP_MSG', config.WANT_TO_STOP_MSG): {
                                   'name': 'WANT_TO_STOP_MSG', 'adjustment': 550}}

    while True:
        ret, frame = video.read(cv2.IMREAD_COLOR)
        if not ret:
            break

        # Getting CAN data
        try:
            with urllib.request.urlopen('http://127.0.0.1:5500/can/') as response:
                content = response.read().decode("utf-8")
                data_dict = json.loads(content)
                config.BRAKE_STATUS = int(data_dict['brake'])
                config.VEHICLE_SPEED = float(data_dict['speed'])
                rfid = int(data_dict['rfid'])
                obstacle = int(data_dict['obstacle_detected'])
                no_obstacle = int(data_dict['no_obstacle'])
                config.TOGGLE_STATUS = int(data_dict['toggled'])

                if config.TOGGLE_START or config.TOGGLE_STOP:
                    while not config.TOGGLE_STATUS:
                        config.TOGGLE_START = False
                        config.TOGGLE_STOP = False

                if config.DEBUG:
                    print('RFID', config.RFID_FLAG)
                    print('OBSTACLE', config.OBSTACLE)
                    print('VEHICLE SPEED', config.VEHICLE_SPEED)
                    print('BRAKE', config.BRAKE_STATUS)

                config.OBSTACLE = obstacle
                config.NO_OBSTACLE = no_obstacle

                if rfid and config.BRAKE_STATUS:
                    config.RFID_FLAG = rfid

        except Exception as e:
            pass
        # End of CAN signals

        h, w = frame.shape[:2]

        frame_copy = frame.copy()
        frame_copy_face = frame.copy()
        # frame = cv2.UMat(frame)
        # frame_copy_umat = cv2.UMat(frame_copy)
        # frame_copy_face_umat = cv2.UMat(frame_copy_face)

        if config.FLIP_VERTICAL:
            frame = cv2.flip(frame, 1)
            frame_copy_face = cv2.flip(frame_copy_face, 1)

        curr_time = time.time()

        keypress = cv2.waitKey(1)

        # # OLD UI to show
        # img_pil = Image.fromarray(frame)
        # draw = ImageDraw.Draw(img_pil)

        # draw.text((int(loc_msg_to_show_at_top * float(w) + msg_to_show_at_top_dict[msg_to_show_at_top]['adjustment']),
        #            int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
        #           fill=(255, 255, 255, 255))
        # frame = np.array(img_pil)

        # # staff async api call
        staff.getStaffIds()
        if len(staff.getData()) == 0:
            STAFF_IDS = []
        else:
            STAFF_IDS = staff.getData()

        if config.RFID_FLAG == 1 or config.RFID_FLAG == 3:
            if not trip_started:
                if config.DEBUG:
                    logging.debug('\n\n********NEW TRIP STARTED********\n\n')
                    logging.debug('\n\n********'+TRIP_ID+'********\n\n')
                config.LAST_RFID_FLAG = config.RFID_FLAG
                tts.silence()
                if bool(int(config_info.get('SKIP_FACE_RECOGNITION', config.SKIP_FACE_RECOGNITION))):
                    config.TRACK_FACES = False
                    config.TRACK_GESTURES = True
                    config.VEHICLE_STATUS = 0
                else:
                    border_color = config.WAIT_COLOR
                    if config.RFID_FLAG == 1:
                        msg_to_show_at_top = config_info.get('STATION_1', config.STATION_1)
                    elif config.RFID_FLAG == 3:
                        msg_to_show_at_top = config_info.get('STATION_2', config.STATION_2)
                    config.TRACK_FACES = True
                    config.TRACK_GESTURES = False
                    start_face_recognition_greeting_time = curr_time
                    if config.FACE_RECOGNITION_GREETING_FLAG:
                        tts.say(config.FACE_RECOGNITION_GREETING)
                trip_start_time = str(datetime.utcnow())
                trip_started = True
                trip_reset = False
                frame_count = 0

        if trip_started:
            # FACE TRACKING
            if config.DEBUG:
                print('Face Recognition conitoin', start_face_recognition_greeting_time != 0. and curr_time - start_face_recognition_greeting_time > 5.0 and \
                    config.TRACK_FACES and \
                    frame_count % int(config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)) == 0)
            if start_face_recognition_greeting_time != 0. and curr_time - start_face_recognition_greeting_time > 5.0 and \
                    config.TRACK_FACES and \
                    frame_count % int(config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)) == 0:
                    cropped_frame_face = frame_copy_face[int(float(h) * config.TOP_BORDER_WIDTH_FACTOR):int(
                        float(h) * (1.0 - 0.15)), 0:w]
                    border_color = config.WAIT_COLOR
                    msg_to_show_at_top = config_info.get('WAIT_MSG', config.WAIT_MSG)
                    if len(face_boxes) > 0:
                        last_matching_names = matching_names
                        last_matching_boxes = matching_boxes
                        last_matching_ids = matching_ids
                        last_matching_confidence = matching_confidence

                        send_face_boxes = face_boxes
                        _send_face_boxes = []
                        ind = 0
                        for bb in face_boxes:
                            _send_face_boxes.append([bb, ind])
                            ind += 1
                        start = time.time()
                        cv2.imwrite('./data/frame.jpg', frame_copy_face, [cv2.IMWRITE_JPEG_QUALITY, config.PHOTO_QUALITY])
                        compressed_frame = cv2.imread('./data/frame.jpg')
                        matching_names, matching_boxes, matching_ids, matching_confidence = faceApiCall(compressed_frame)
                        if config.DEBUG:
                            print('Face Recognition hit')
                    input_q.put(cropped_frame_face)
            if output_q.empty():
                pass  # fill up queue
            else:
                _frame_count += 1
                data = output_q.get()
                # face_boxes = data['face_boxes']
                __face_boxes = data['face_boxes']
                face_boxes = []
                for bb in __face_boxes:
                    bb[1] += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
                    bb[3] += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
                    face_boxes.append(bb)

                # # UNCOMMENT CODE IF YOU WANT TO REMOVE UNKNOWNS WHEN NO FACE IS DETECTED IN CURRENT FRAME
                # if len(face_boxes) == 0:
                #     unknown_faces_to_remove = []
                #     for u_f_bb in face_unknown:
                #         unknown_faces_to_remove.append(u_f_bb)
                #     unknown_faces_to_remove = set(unknown_faces_to_remove)
                #     for u_bb in unknown_faces_to_remove:
                #         if config.DEBUG:
                #             logging.debug('Entering every time for no face boxes')
                #         face_unknown.pop(u_bb)
                #
                #     unknown_faces_to_remove_final = []
                #     for u_f_bb in face_unknown_final:
                #         unknown_faces_to_remove_final.append(u_f_bb)
                #     for u_bb in unknown_faces_to_remove_final:
                #         if config.DEBUG:
                #             logging.debug('Entering every time for no face boxes final')
                #         face_unknown_final.pop(u_bb)

                names_to_remove = []
                names_to_add = {}
                boxes_names_iou = {}
                check_send_face_boxes = []

                # if len(face_boxes) > 0 and len(matching_ids) > 0:
                for bb in face_boxes:
                    x1, y1, x2, y2 = bb
                    ind = -1
                    max_iou = 0.2
                    min_value = 100000
                    for _bb, _ind in _send_face_boxes:
                        _x1, _y1, _x2, _y2 = _bb
                        iou = getIoU(bb, _bb)
                        i_x, i_y, i_w, i_h = intersection(bb, _bb)
                        i_a = i_w * i_h
                        u_x, u_y, u_w, u_h = union(bb, _bb)
                        u_a = u_w * u_h
                        _min_value = abs(_x1 - x1) + abs(y1 - _y1)
                        if min_value >= _min_value and iou > max_iou and i_a <= (_x2 - _x1) * (_y2 - _y1) <= u_a:
                            min_value = _min_value
                            max_iou = iou
                            ind = _ind
                    if ind != -1:
                        check_send_face_boxes.append([bb, ind])

                for bb, ind in check_send_face_boxes:
                    __ind = 0
                    for _bb, _ind in _send_face_boxes:
                        if __ind == ind:
                            _send_face_boxes[ind] = [bb, _ind]
                            break
                        __ind += 1

                change_tracker = False
                if len(matching_boxes) != len(last_matching_boxes):
                    change_tracker = True
                else:
                    for bb, name, _id in zip(matching_boxes, matching_names, matching_ids):
                        if _id in last_matching_boxes:
                            index1 = matching_ids.index(_id)
                            index2 = last_matching_ids.index(_id)
                            if matching_boxes[index1] == last_matching_boxes[index2]:
                                pass
                        else:
                            change_tracker = True
                            break
                face_unknown_current = []
                if change_tracker:
                    for bb, _bb, name, p_id, confidence in zip(matching_boxes, send_face_boxes, matching_names,
                                                               matching_ids, matching_confidence):
                        # if confidence != 0:
                        ind = 0
                        max_iou = 0.2
                        max_iou_index = 0
                        for x_bb in send_face_boxes:
                            iou = getIoU(bb, x_bb)
                            if iou > max_iou:
                                max_iou = iou
                                max_iou_index = ind
                            ind += 1
                        for _x_bb, _ind in _send_face_boxes:
                            if max_iou_index == _ind and max_iou > 0.4:
                                if p_id not in STAFF_IDS and confidence != 0.0:
                                    if p_id in face_names:
                                        _, _, _, _, _, _, cnt = face_names[p_id]
                                        face_names[p_id] = [name, _x_bb, _frame_count, max_iou_index, True, False,
                                                            cnt + 1]
                                        # break
                                    else:
                                        face_names[p_id] = [name, _x_bb, _frame_count, max_iou_index, False, True, 1]
                                        # break
                                elif p_id in STAFF_IDS and confidence != 0.0:
                                    if p_id in face_staff:
                                        _, _, _, _, _, _, cnt = face_staff[p_id]
                                        face_staff[p_id] = [name, _x_bb, _frame_count, max_iou_index, True, False,
                                                            cnt + 1]
                                        # break
                                    else:
                                        face_staff[p_id] = [name, _x_bb, _frame_count, max_iou_index, True, False, 1]
                                        # break
                                    # pass
                                else:
                                    if confidence == 0.0:
                                        face_unknown_current.append(_x_bb)
                if config.DEBUG:
                    logging.debug('Face unknown current' + str(face_unknown_current))

                unknown_faces_to_remove = []
                unknown_faces_to_add = {}
                if len(face_unknown) > 0:
                    for bb in face_unknown_current:
                        unknown_max_iou = 0.2
                        unknown_max_iou_bb = ''
                        _unknown_max_iou_bb = ''
                        x1, y1, x2, y2 = bb
                        min_value = 100000
                        for str_bb in face_unknown:
                            _x1, _y1, _x2, _y2 = str2list(str_bb)
                            _min_value = abs(_x1 - x1) + abs(y1 - _y1)
                            iou = getIoU(str2list(str_bb), bb)
                            if min_value >= _min_value and iou > unknown_max_iou:
                                min_value = _min_value
                                unknown_max_iou = iou
                                unknown_max_iou_bb = str_bb
                                _unknown_max_iou_bb = list2str(bb)
                        if config.DEBUG:
                            logging.debug('unknown overlap iou' + str(unknown_max_iou))
                        if unknown_max_iou != 1.0:
                            if len(unknown_max_iou_bb) > 0 and unknown_max_iou > 0.3:
                                fc, last_fc, is_spoken, cnt = face_unknown[unknown_max_iou_bb]
                                unknown_faces_to_remove.append(unknown_max_iou_bb)
                                unknown_faces_to_add[_unknown_max_iou_bb] = [_frame_count, last_fc, is_spoken, cnt + 1]
                            else:
                                logging.debug('New un-tracked face added')
                                unknown_faces_to_add[list2str(bb)] = [_frame_count, _frame_count, False, 1]
                else:
                    for _u_bb in face_unknown_current:
                        logging.debug('First face added')
                        face_unknown[list2str(_u_bb)] = [_frame_count, _frame_count, False, 1]

                # unknown_faces_to_remove = set(unknown_faces_to_remove)
                for u_bb in unknown_faces_to_remove:
                    if config.DEBUG:
                        logging.debug('face removed unknown ' + u_bb + str(face_unknown[u_bb]))
                    face_unknown.pop(u_bb)

                # unknown faces to add
                for u_bb in unknown_faces_to_add:
                    face_unknown[u_bb] = unknown_faces_to_add[u_bb]

                # checking if unknown face are being overlapped by any known one - NOT TO BE CHANGED
                unknown_faces_to_remove = []
                for id1 in face_names:
                    _, bb1, _, _, _, _, _ = face_names[id1]
                    for bb_str in face_unknown:
                        iou = getIoU(bb1, str2list(bb_str))
                        if iou > 0.6:
                            unknown_faces_to_remove.append(bb_str)

                unknown_faces_to_remove = set(unknown_faces_to_remove)
                for u_bb in unknown_faces_to_remove:
                    if config.DEBUG:
                        logging.debug('removed after tracking KNOWN OVERLAPPED' + str(u_bb))
                    face_unknown.pop(u_bb)

                # checking if unknown face are being overlapped with staff face - NOT TO BE CHANGED
                unknown_faces_to_remove = []
                for id2 in face_staff:
                    _, bb2, _, _, _, _, _ = face_staff[id2]
                    for bb_str in face_unknown:
                        iou = getIoU(bb2, str2list(bb_str))
                        if iou > 0.6:
                            unknown_faces_to_remove.append(bb_str)

                unknown_faces_to_remove = set(unknown_faces_to_remove)
                for u_bb in unknown_faces_to_remove:
                    if config.DEBUG:
                        logging.debug('removed after tracking STAFF OVERLAPPED' + str(u_bb))
                    face_unknown.pop(u_bb)  # CODE NOT TO BE CHANGED

                # Putting the tracked unknown in final unknown dict
                unknown_final_faces_to_remove = []
                unknown_final_faces_to_add = {}
                if len(face_unknown_final) > 0:
                    for str_bb in face_unknown:
                        unknown_max_iou = 0.2
                        unknown_max_iou_bb = ''
                        _unknown_max_iou_bb = ''
                        x1, y1, x2, y2 = str2list(str_bb)
                        min_value = 100000
                        for _str_bb in face_unknown_final:
                            _x1, _y1, _x2, _y2 = str2list(_str_bb)
                            _min_value = abs(_x1 - x1) + abs(y1 - _y1)
                            iou = getIoU(str2list(str_bb), str2list(_str_bb))
                            if min_value >= _min_value and iou > unknown_max_iou:
                                min_value = _min_value
                                unknown_max_iou = iou
                                unknown_max_iou_bb = str_bb
                                _unknown_max_iou_bb = _str_bb
                        if len(_unknown_max_iou_bb) > 0 and len(
                                unknown_max_iou_bb) > 0 and min_value < 100000 and unknown_max_iou != 1.0:
                            _fc, _last_fc, _is_spoken, _ind = face_unknown[unknown_max_iou_bb]
                            _, _, is_spoken, _ = face_unknown_final[_unknown_max_iou_bb]
                            unknown_final_faces_to_remove.append(_unknown_max_iou_bb)
                            if _fc - _last_fc >= int(config_info.get('FRAMES_BUFFER_ADD_UNKNOWN_FACES',
                                                                     config.FRAMES_BUFFER_ADD_UNKNOWN_FACES)):
                                unknown_final_faces_to_add[unknown_max_iou_bb] = [_fc, _last_fc, is_spoken, _ind]
                        else:
                            _fc, _last_fc, _is_spoken, _ind = face_unknown[str_bb]
                            if _fc - _last_fc >= int(config_info.get('FRAMES_BUFFER_ADD_UNKNOWN_FACES',
                                                                     config.FRAMES_BUFFER_ADD_UNKNOWN_FACES)):
                                unknown_final_faces_to_add[str_bb] = [_fc, _last_fc, True, _ind]

                else:
                    for str_bb in face_unknown:
                        _fc, _last_fc, _is_spoken, _ind = face_unknown[str_bb]
                        if _fc - _last_fc >= int(config_info.get('FRAMES_BUFFER_ADD_UNKNOWN_FACES',
                                                                 config.FRAMES_BUFFER_ADD_UNKNOWN_FACES)):
                            face_unknown_final[str_bb] = [_fc, _last_fc, _is_spoken, _ind]

                # Removing old ones
                for str_b in unknown_final_faces_to_remove:
                    if str_b in face_unknown_final:
                        if config.DEBUG:
                            logging.debug('Removing old ones')
                        face_unknown_final.pop(str_b)

                # Removing the tracked unknown from final unknown dict
                face_unknown_final_to_remove = []
                for str_bb in face_unknown_final:
                    fc, last_fc, _, _ = face_unknown_final[str_bb]
                    if _frame_count - fc >= int(config_info.get('FRAMES_BUFFER_REMOVE_UNKNOWN_FACES',
                                                                config.FRAMES_BUFFER_REMOVE_UNKNOWN_FACES)):
                        face_unknown_final_to_remove.append(str_bb)

                for final_str_bb in face_unknown_final_to_remove:
                    if config.DEBUG:
                        logging.debug('FINAL FACE UNKNOWN REMOVED' + str(final_str_bb))
                    face_unknown_final.pop(final_str_bb)

                # Adding tracked faces
                for str_b in unknown_final_faces_to_add:
                    face_unknown_final[str_b] = unknown_final_faces_to_add[str_b]

                # # checking if FINAL unknown face are being overlapped by any known one - NOT TO BE CHANGED
                # unknown_faces_to_remove = []
                # for id1 in face_names:
                #     _, bb1, _, _, _, _, _ = face_names[id1]
                #     for bb_str in face_unknown_final:
                #         iou = getIoU(bb1, str2list(bb_str))
                #         if config.DEBUG:
                #             print('unknown face are being overlapped by any known one: ' + str(iou))
                #         if iou > 0.5:
                #             unknown_faces_to_remove.append(bb_str)
                #
                # unknown_faces_to_remove = set(unknown_faces_to_remove)
                # for u_bb in unknown_faces_to_remove:
                #     if config.DEBUG:
                #         logging.debug('removed after tracking KNOWN OVERLAPPED' + str(u_bb))
                #     face_unknown_final.pop(u_bb)
                #
                # # checking if FINAL unknown face are being overlapped with staff face - NOT TO BE CHANGED
                # unknown_faces_to_remove = []
                # for id2 in face_staff:
                #     _, bb2, _, _, _, _, _ = face_staff[id2]
                #     for bb_str in face_unknown_final:
                #         iou = getIoU(bb2, str2list(bb_str))
                #         if config.DEBUG:
                #             print('unknown face are being overlapped with staff face : ' + str(iou))
                #         if iou > 0.5:
                #             unknown_faces_to_remove.append(bb_str)
                #
                # unknown_faces_to_remove = set(unknown_faces_to_remove)
                # for u_bb in unknown_faces_to_remove:
                #     if config.DEBUG:
                #         logging.debug('removed after tracking STAFF OVERLAPPED' + str(u_bb))
                #     face_unknown_final.pop(u_bb)  # FINAL CODE NOT TO BE CHANGED

                # Removing unknown persons who had left the vehicle
                unknown_faces_to_remove = []
                for bb_str in face_unknown:
                    fc, last_fc, is_spoken, _ = face_unknown[bb_str]
                    if _frame_count - fc >= int(config_info.get('FRAMES_BUFFER_REMOVE_UNKNOWN_FACES',
                                                                config.FRAMES_BUFFER_REMOVE_UNKNOWN_FACES)):
                        unknown_faces_to_remove.append(bb_str)

                unknown_faces_to_remove = set(unknown_faces_to_remove)
                for u_bb in unknown_faces_to_remove:
                    if config.DEBUG:
                        logging.debug('unknown boxes removed timeout' + str(face_unknown[u_bb]))
                    face_unknown.pop(u_bb)

                # End of code for Unknown

                # Removing persons who had left the vehicle
                persons_to_remove = []
                for _id in face_names:
                    _, _, fc, _, _, _, _ = face_names[_id]
                    if _frame_count - fc >= int(
                            config_info.get('FRAMES_BUFFER_REMOVE_FACES',
                                            config.FRAMES_BUFFER_REMOVE_FACES)) / int(
                        config_info.get('FRAMES_SKIP_FACE_DETECTION',
                                        config.FRAMES_SKIP_FACE_DETECTION)):  # and _id not in matching_ids and _id not in last_matching_ids:
                        persons_to_remove.append(_id)

                for _id_ in persons_to_remove:
                    if config.DEBUG:
                        logging.debug('Person removed:' + str(_id_))
                    face_names.pop(_id_)

                # Removing staff who had left the vehicle or not being detected for long
                staff_to_remove = []
                for _id in face_staff:
                    _, _, fc, _, _, _, _ = face_staff[_id]
                    if _frame_count - fc >= int(config_info.get('FRAMES_BUFFER_REMOVE_STAFF_FACES',
                                                                config.FRAMES_BUFFER_REMOVE_STAFF_FACES)) / int(
                        config_info.get('FRAMES_SKIP_FACE_DETECTION',
                                        config.FRAMES_SKIP_FACE_DETECTION)):  # and _id not in matching_ids and _id not in last_matching_ids:
                        staff_to_remove.append(_id)

                for _id_ in staff_to_remove:
                    if config.DEBUG:
                        logging.debug('Staff removed:' + str(_id_))
                    face_staff.pop(_id_)

                # Starting the face counter
                if len(face_names) >= int(
                        config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)) and track_time == 0:
                    track_time = curr_time
                    count_current_face = len(face_names)
                # Counter Reset
                if len(face_names) <= config.MAX_NUMBER_FACES and (
                        len(face_names) - count_current_face) >= 1:
                    count_current_face = len(face_names)
                    track_time = curr_time

                for bb, _, p_id in zip(matching_boxes, matching_names, matching_ids):
                    if p_id in face_names:
                        name, _bb, fc, ind, a, b, c = face_names[p_id]
                        max_iou = 0.2
                        x_bb = []
                        for _ in face_names:
                            iou = getIoU(bb, _bb)
                            if iou > max_iou:
                                max_iou = iou
                                x_bb = _bb
                        if max_iou > 0.8 and len(x_bb) > 0:  # and p_id not in STAFF_IDS:
                            face_names[p_id] = [name, x_bb, _frame_count, ind, a, b, c]

            # Drawing face boxes and announcing names
            if config.TRACK_FACES:
                # # DEBUGGING SECTION
                # if config.DEBUG:
                #     if len(matching_boxes) > 0 and len(matching_names) > 0:
                #         for name, m_bb in zip(matching_names, matching_boxes):
                #             cv2.rectangle(frame, (m_bb[0], m_bb[1]), (m_bb[2], m_bb[3]), (0, 255, 0), 3)
                #             cv2.putText(frame, name, (m_bb[0], m_bb[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
                #                         (255, 255, 255), 3)
                if config.DEBUG:
                    # Showing face count timer
                    cv2.putText(frame, 'Timer: ' + str(int(curr_time - track_time)), (w - 200, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2)

                if len(face_names) > 0 or len(matched_name) > 0:
                    if config.DEBUG:
                        logging.debug('Matching names:' + str(matching_names))
                        logging.debug('Matching boxes:' + str(matching_boxes))
                        logging.debug('Matching ids:' + str(matching_ids))
                        logging.debug('Matching confidence:' + str(matching_confidence))
                        logging.debug('Last matching names:' + str(last_matching_names))
                        logging.debug('Last matching boxes:' + str(last_matching_boxes))
                        logging.debug('Last Matching ids:' + str(last_matching_ids))
                        logging.debug('Last matching confidence:' + str(last_matching_confidence))
                        logging.debug('Face boxes:' + str(face_boxes))
                        logging.debug('Face names:' + str(face_names))
                        logging.debug('Frame count:' + str(_frame_count))
                        logging.debug('Staff:' + str(face_staff))
                        logging.debug('Staff ids:' + str(STAFF_IDS))
                        logging.debug('Face Unknown' + str(face_unknown))
                        logging.debug('Face Unknown final' + str(face_unknown_final))
                        logging.debug('Greeting dict' + str(greeting_dict))

                    # Drawing boxes for Yamaha Staff
                    for staff_id in face_staff:
                        _, _bb, fc, _ind, _, _, _ = face_staff[staff_id]
                        draw_border_face('Staff', frame, (_bb[0], _bb[1]), (_bb[2], _bb[3]), config.READY_COLOR, 20, 25)

                    # UI changes during FR
                    if len(face_unknown_final) > 0:  # if curr_time - unknown_count_time >= config.MSG_SHOW_TIME:
                        border_color = config.STOP_COLOR
                        msg_to_show_at_top = config_info.get('STOP_MSG', config.STOP_MSG)
                        track_time = curr_time
                    else:
                        if border_color == config.STOP_COLOR:
                            tts.silence()
                        border_color = config.WAIT_COLOR
                        msg_to_show_at_top = config_info.get('WAIT_MSG', config.WAIT_MSG)

                    delta_time = 0
                    face_names, face_unknown_final = sayGreetingMessage(face_names, face_unknown_final)
                    drawFaces(frame, face_names, face_unknown_final)

                # countdown is over and greetings for all known name are already announced
                if len(face_names) >= int(config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)) and (
                        curr_time - track_time) > float(config_info.get('FACE_TRACK_TIME', config.FACE_TRACK_TIME)):
                    config.TRACK_FACES = False
                    config.IS_FACE_RECOGNITION_DONE = True
                    config.TRACK_GESTURES = True
                    config.VEHICLE_STATUS = 0
                    final_names = face_names.keys()
                    track_time = curr_time

            # ********************************************************************
            # **************************GESTURE TRACKING**************************
            # ********************************************************************
            if config.TRACK_GESTURES:
                # config.TRACK_FACES = False
                delta_time = curr_time - track_time
                if config.VEHICLE_STATUS == 0:
                    # frame_copy_gesture = copy.deepcopy(frame)
                    # cropped_frame_gesture_ = frame_copy_gesture[int(float(h) * config.TOP_BORDER_WIDTH_FACTOR):int(
                    #     float(h) * (1.0 - 0.15)), 0:w]
                    if silence_vehicle_status_0:
                        tts.silence()
                        silence_vehicle_status_0 = False

                    if not final_gestures['thumbsup'] >= int(
                            config_info.get('MIN_COUNT_THUMBSUP_FOR_START', config.MIN_COUNT_THUMBSUP_FOR_START)):
                        border_color = config.START_COLOR
                        msg_to_show_at_top = config_info.get('READY_MSG_TOP', config.READY_MSG_TOP)
                    if delta_time > config.MSG_SHOW_TIME:  # and not photo_clicked:
                        #     if delta_time < 2 * config.MSG_SHOW_TIME:
                        __face_boxes = []
                        face_boxes = []
                        matching_names = []
                        face_tracker = {}

                    if not greetings_done_1:
                        greetings_done_1 = True
                        msg_to_show = 'If you are ready to go,\nPlease give a thumbs-up'
                        tts.say(config_info.get('READY_MSG_BOT1', config.READY_MSG_BOT1))

                    # if frame_count % config.FRAMES_SKIP_GESTURE_DETECTION == 0 or gesture_detection_thumb:
                    num, dets = gesture.detect(frame, gesture_obj)
                    final_gestures, gesture_tracker = trackGestures(num, dets, meta, gesture_tracker, frame_count,
                                                                    final_gestures)
                    gesture_tracker = drawGestures(frame, gesture_tracker, meta, frame_count)

                    if config.DEBUG:
                        # Showing thumbs-up count timer
                        cv2.putText(frame, 'Timer: ' + str(int(curr_time - track_time3)), (w - 200, h - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    if do_gesture_countdown:
                        if track_time3 == 0:
                            track_time3 = curr_time
                        # Counter Reset
                        if abs(final_gestures['thumbsup'] - count_current_thumbsup) >= 1:
                            count_current_thumbsup = final_gestures['thumbsup']
                            track_time3 = curr_time

                    if (final_gestures['thumbsup'] >= int(config_info.get('MIN_COUNT_THUMBSUP_FOR_START', config.MIN_COUNT_THUMBSUP_FOR_START)) and (curr_time - track_time3) >= float(config_info.get('THUMBSUP_COUNTDOWN_TIME', config.THUMBSUP_COUNTDOWN_TIME))) or final_gestures['thumbsup'] >= int(config_info.get('MIN_COUNT_THUMBSUP_FOR_START', config.MIN_COUNT_THUMBSUP_FOR_START)):
                        if track_time2 == 0:
                            track_time2 = curr_time
                            do_gesture_countdown = False
                            track_time3 = 0
                            count_current_face = 0

                        if curr_time - track_time2 > config.MSG_SHOW_TIME:
                            border_color = config.READY_COLOR
                            msg_to_show_at_top = config_info.get('READY_MSG', config.READY_MSG)
                            track_time2 = curr_time

                        # Greeting photo 1
                        if len(final_names) < int(config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)):
                            photo_clicked = True
                            greeting_photo_1 = True
                            greeting_photo_2 = True
                            config.IS_FACE_RECOGNITION_DONE = True

                        if len(final_names) >= int(config_info.get('MIN_NUMBER_FACES',
                                                                   config.MIN_NUMBER_FACES)) and not photo_clicked and not greeting_photo_1:
                            greeting_photo_1 = True
                            msg_to_show = 'We will now take a photo.\nPlease look at the screen and give a thumbs-up.'
                            tts.say(config_info.get('TAKE_PHOTO_MSG', config.TAKE_PHOTO_MSG))
                            stop_time_photo_click = time.time()

                        if not photo_clicked and time.time() - stop_time_photo_click > 9:
                            # Creating the new trip dir
                            if not os.path.exists(config.TRIP_DIR + TRIP_ID):
                                os.makedirs(config.TRIP_DIR + TRIP_ID)

                            # Clicking the photo
                            playsound(config.CAMERA_SHUTTER_CLICK)
                            # setting the picture click display window name
                            cv2.namedWindow(config.PIC_WIN_NAME, cv2.WND_PROP_FULLSCREEN)
                            # setting the picture click window to full screen
                            cv2.setWindowProperty(config.PIC_WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                                  cv2.WINDOW_FULLSCREEN)
                            # drawing boxes on image
                            num, dets = gesture.detect(frame_copy, gesture_obj)
                            frame_copy = drawBoxesPic(frame_copy, dets, num)
                            # changes for UI in the photo click
                            # Top bar rectangle
                            frame_copy_ui = mailUi.applyingUIMail(frame_copy)
                            cv2.imwrite(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.jpg', frame_copy_ui * 255)
                            # End of the changes
                            cv2.imshow(config.PIC_WIN_NAME, frame_copy_ui)
                            cv2.waitKey(100 * config.SHOW_PIC_TIME)
                            cv2.destroyWindow(config.PIC_WIN_NAME)

                            json_data = {}
                            if not final_names == []:
                                with open(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.json', 'w') as outfile:
                                    for idx, name in enumerate(final_names):
                                        if 'Unknown' not in name and not name == '':
                                            json_data['person' + str(idx)] = name
                                    json_data['TripStartTime'] = trip_start_time
                                    json.dump(json_data, outfile)
                            photo_clicked = True
                            msg_to_show = 'Thank you for your cooperation.\nWe will depart shortly. Please remain ' \
                                          'seated. '
                            tts.say(config_info.get('FEEDBACK_PHOTO_MSG', config.FEEDBACK_PHOTO_MSG))
                            greeting_photo_2 = True
                            stop_time_photo_click2 = time.time()

                        if photo_clicked and curr_time - stop_time_photo_click2 > 5.0:
                            # giving the START signal to CAN
                            # # changes
                            if config.BRAKE_STATUS == 1 and not config.OBSTACLE:  # May need some coding for Flow modification
                                try:
                                    requests.post('http://127.0.0.1:5500/toggle/', timeout=0.0000000001)
                                    config.VEHICLE_STATUS = 1
                                    config.TOGGLE_START = True
                                    gesture_counts = {}
                                    final_gestures = {}
                                    for i in range(meta.classes):
                                        final_gestures[gesture_class_names[i]] = 0
                                    gesture_tracker = {}
                                    continue
                                except requests.exceptions.ReadTimeout:
                                    pass

                if config.VEHICLE_STATUS == 1 and config.BRAKE_STATUS == 0:
                    # frame_copy_gesture = copy.deepcopy(frame)
                    # cropped_frame_gesture_ = frame_copy_gesture[int(float(h) * config.TOP_BORDER_WIDTH_FACTOR):int(
                    #     float(h) * (1.0 - 0.15)), 0:w]

                    # Greeting photo 1
                    if len(final_names) < int(config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)):
                        photo_clicked = True
                        greeting_photo_1 = True
                        greeting_photo_2 = True
                        config.IS_FACE_RECOGNITION_DONE = True

                    if len(final_names) >= int(config_info.get('MIN_NUMBER_FACES',
                                                               config.MIN_NUMBER_FACES)) and not photo_clicked and not greeting_photo_1:
                        border_color = config.READY_COLOR
                        msg_to_show_at_top = config_info.get('READY_MSG', config.READY_MSG)
                        greeting_photo_1 = True
                        msg_to_show = 'We will now take a photo.\nPlease look at the screen and give a thumbs-up.'
                        tts.say(config.TAKE_PHOTO_MSG)
                        stop_time_photo_click = time.time()

                    if not photo_clicked and time.time() - stop_time_photo_click > 10:
                        # Creating the new trip dir
                        if not os.path.exists(config.TRIP_DIR + TRIP_ID):
                            os.makedirs(config.TRIP_DIR + TRIP_ID)
                        # Clicking the photo
                        playsound(config.CAMERA_SHUTTER_CLICK)
                        # setting the picture click display window name
                        cv2.namedWindow(config.PIC_WIN_NAME, cv2.WND_PROP_FULLSCREEN)
                        # setting the picture click window to full screen
                        cv2.setWindowProperty(config.PIC_WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN)

                        # drawing boxes on image
                        num, dets = gesture.detect(frame_copy, gesture_obj)
                        frame_copy = drawBoxesPic(frame_copy, dets, num)

                        # changes for UI in the photo click
                        frame_copy_ui = mailUi.applyingUIMail(frame_copy)
                        # End of the changes

                        # while track_time - time.time() < config.SHOW_PIC_TIME:
                        cv2.imshow(config.PIC_WIN_NAME, frame_copy_ui)
                        cv2.waitKey(100 * config.SHOW_PIC_TIME)

                        cv2.imwrite(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.jpg', frame_copy_ui * 255)
                        cv2.destroyWindow(config.PIC_WIN_NAME)

                        json_data = {}
                        if not final_names == []:
                            with open(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.json', 'w') as outfile:
                                for idx, name in enumerate(final_names):
                                    if 'Unknown' not in name and not name == '':
                                        json_data['person' + str(idx)] = name
                                json_data['TripStartTime'] = trip_start_time
                                json.dump(json_data, outfile)
                        photo_clicked = True
                        greeting_photo_2 = True

                    if photo_clicked:
                        # ## LATEST CHANGES
                        border_color = config.WAIT_COLOR
                        ann1 = 'Thank you for choosing Yamaha PPM.\nA PPM stands for Public Personal Mobility.'
                        ann2 = 'If you want to stop in the middle of the \npathway, please raise your palm.'
                        if not announcements_done:
                            # msg_to_show = "Vehicle is starting.\nTo request a stop, please raise your palm"

                            if config.RFID_FLAG == 1 and not announcements_done_1:
                                msg_to_show = 'We are now approaching Station'
                                # msg_to_show_at_top = config_info.get('NEAR_ARRIVAL_STATION_1',
                                #                                      config.NEAR_ARRIVAL_STATION_1)
                                tts.say(msg_to_show)
                                announcements_done_1 = True
                                stop_time_ann = time.time()
                            elif config.RFID_FLAG == 3 and not announcements_done_1:
                                msg_to_show = 'We are now approaching South Hall Station'
                                msg_to_show_at_top = config_info.get('NEAR_ARRIVAL_STATION_2',
                                                                     config.NEAR_ARRIVAL_STATION_2)
                                tts.say(msg_to_show)
                                stop_time_ann = time.time()
                                announcements_done_1 = True

                            if curr_time - stop_time_ann > 3.5 and stop_time_ann != 0.0:
                                # announcement 1
                                if not announcements_done_2:
                                    msg_to_show = ann1
                                    tts.say(config.ANN1)
                                    stop_time_ann1 = time.time()
                                    announcements_done_2 = True

                                if curr_time - stop_time_ann1 > 5.0 and stop_time_ann1 != 0.0 and not announcements_done_3:
                                    # announcement 2
                                    msg_to_show_at_top = config_info.get('WANT_TO_STOP_MSG', config.WANT_TO_STOP_MSG)
                                    msg_to_show = ann2
                                    tts.say(config.ANN2)
                                    announcements_done_3 = True
                                    announcements_done = True

                        # if announcements_done:
                        if not announcements_done_3:
                            tts.say(config.ANN2)
                            announcements_done_3 = True

                        msg_to_show_at_top = config_info.get('WANT_TO_STOP_MSG', config.WANT_TO_STOP_MSG)
                        msg_to_show = ann2

                        # Starting the palmup recognition
                        num, dets = gesture.detect(frame, gesture_obj)
                        # gesture_detection_palm = False
                        final_gestures, gesture_tracker = trackGestures(num, dets, meta, gesture_tracker,
                                                                        frame_count,
                                                                        final_gestures)
                        gesture_tracker = drawGestures(frame, gesture_tracker, meta, frame_count)
                        if final_gestures['palmup'] >= config.MIN_COUNT_PALMUP_FOR_STOP:
                            border_color = config.STOP_COLOR
                            msg_to_show_at_top = config_info.get('STOP_MSG', config.STOP_MSG)
                            config.VEHICLE_STATUS = 2  # previously 2
                            track_time = curr_time
                            delta_time = 0
                            gesture_counts = {}
                            final_gestures = {}
                            for i in range(meta.classes):
                                final_gestures[gesture_class_names[i]] = 0
                            gesture_tracker = {}

                            if not greetings_done_4:
                                # Greetings 4
                                greetings_done_4 = True
                                msg_to_show = 'We will now make brief stop.\nPlease remain seated'
                                tts.silence()
                                tts.say(config_info.get('STOP_MSG_BOT2', config.STOP_MSG_BOT2))
                            if greetings_done_4:
                                # giving the stop signal to CAN
                                if config.BRAKE_STATUS == 0:
                                    try:
                                        requests.post('http://127.0.0.1:5500/toggle/', timeout=0.0000000001)
                                        config.TOGGLE_STOP = True
                                        continue
                                    except requests.exceptions.ReadTimeout:
                                        pass

                if config.VEHICLE_STATUS == 2:

                    if delta_time > 2 * config.STOP_TIME_BUFFER:

                        # For going back to Gesture - Thumbsup
                        config.VEHICLE_STATUS = 0
                        border_color = config.WAIT_COLOR
                        msg_to_show_at_top = config_info.get('READY_MSG_TOP', config.READY_MSG_TOP)
                        if delta_time < 2 * config.MSG_SHOW_TIME:
                            pass
                        greetings_done_1 = False
                        greetings_done_2 = False
                        greetings_done_3 = False
                        greetings_done_4 = False
                        track_time = curr_time
                        track_time2 = 0
                        track_time3 = 0
                        do_gesture_countdown = True
                        announcements_done_3 = False

        if bool(int(config_info.get('REMOTE_CONTROL', config.REMOTE_CONTROL))) and trip_started and config.TOGGLE_STATUS:
            # # Uncomment this code to perform remote control operation
            # Manual or Remote Control Operation (FOR STARTING THE VEHICLE ONLY)
            if config.BRAKE_STATUS == 0 and (not config.IS_FACE_RECOGNITION_DONE or config.VEHICLE_STATUS == 0):
                if config.DEBUG:
                    print('REMOTE CONTROL START')
                announcements_done = True
                config.TRACK_FACES = False
                config.TRACK_GESTURES = True
                config.IS_FACE_RECOGNITION_DONE = True
                config.VEHICLE_STATUS = 1
                msg_to_show = "Vehicle is starting.\nTo request a stop, please raise your palm"
                # Resetting previous gesture tracking information
                track_time = curr_time
                gesture_counts = {}
                final_gestures = {}
                for i in range(meta.classes):
                    final_gestures[gesture_class_names[i]] = 0
                gesture_tracker = {}
                # Resetting previous voice greetings
                tts.silence()
                # Sending new voice greeting
                greetings_done_1 = False
                greetings_done_2 = False
                greetings_done_3 = False
                greetings_done_4 = False
                announcements_done_3 = False

            # Manual or Remote Control Operation (FOR STOPPING THE VEHICLE ONLY)
            if config.TRACK_GESTURES and config.VEHICLE_STATUS == 1 and config.BRAKE_STATUS == 1:
                if config.DEBUG:
                    print('REMOTE CONTROL STOP')
                announcements_done = True
                config.TRACK_FACES = False
                config.IS_FACE_RECOGNITION_DONE = True
                config.TRACK_GESTURES = True
                config.VEHICLE_STATUS = 2
                border_color = config.STOP_COLOR
                msg_to_show_at_top = config_info.get('STOP_MSG', config.STOP_MSG)
                track_time = curr_time
                delta_time = 0
                gesture_counts = {}
                final_gestures = {}
                for i in range(meta.classes):
                    final_gestures[gesture_class_names[i]] = 0
                gesture_tracker = {}
                if not greetings_done_4:
                    # Greetings 4
                    greetings_done_4 = True
                    msg_to_show = 'We will now make brief stop.\nPlease remain seated'
                    tts.silence()
                    tts.say(config_info.get('STOP_MSG_BOT2', config.STOP_MSG_BOT2))
                # Sending new voice greeting
                greetings_done_1 = False
                greetings_done_2 = False
                greetings_done_3 = False

        # OBSTACLE DETECTION
        if config.VEHICLE_STATUS == 3:
            if config.NO_OBSTACLE:
                # if config.LAST_OBSTACLE == config.OBSTACLE:
                #     pass
                # config.LAST_OBSTACLE = config.OBSTACLE
                if not obstacle_greeting_done2:
                #     border_color = config.WAIT_COLOR
                #     if config.RFID_FLAG == 1:
                #         msg_to_show_at_top = config_info.get('NEAR_ARRIVAL_STATION_1', config.NEAR_ARRIVAL_STATION_1)
                #     elif config.RFID_FLAG == 3:
                #         msg_to_show_at_top = config_info.get('NEAR_ARRIVAL_STATION_2', config.NEAR_ARRIVAL_STATION_2)
                #     stop_time_object = time.time()
                    msg_to_show = 'Thank you for your patience.\nWe will now continue'
                    tts.say(config.START_MSG_OBSTACLE)
                    obstacle_greeting_done2 = True
                # if time.time() - stop_time_object > 3:
                #     stop_time_object = 0
                # config.LAST_OBSTACLE = config.OBSTACLE

                # if bool(int(config_info.get('SKIP_FACE_RECOGNITION', config.SKIP_FACE_RECOGNITION))):
                #     if config.TRACK_GESTURES:
                #         config.VEHICLE_STATUS = config.LAST_VEHICLE_STATUS
                # else:
                #     if config.TRACK_FACES:
                #         pass
                #     elif config.TRACK_GESTURES:
                config.VEHICLE_STATUS = config.LAST_VEHICLE_STATUS
                track_time = curr_time
                gesture_counts = {}
                final_gestures = {}
                for i in range(meta.classes):
                    final_gestures[gesture_class_names[i]] = 0
                gesture_tracker = {}
                silence_object_detected = True
                obstacle_greeting_done1 = False
                greetings_done_1 = False

                print('obstacle removed', config.BRAKE_STATUS)
                print('vehicle will start', config.BRAKE_STATUS)
                try:
                    requests.post('http://127.0.0.1:5500/toggle/', timeout=0.0000000001)
                    config.TOGGLE_START = True
                    continue
                except requests.exceptions.ReadTimeout:
                    pass

        # # Handling RFID missing state
        # if config.RFID_FLAG == 1 and config.LAST_RFID_FLAG == 2:
        #     config.VEHICLE_STATUS = 4
        #
        # if config.RFID_FLAG == 3 and config.LAST_RFID_FLAG == 1:
        #     config.VEHICLE_STATUS = 4
        # # End of RFID missing state

        if config.VEHICLE_STATUS != 0 and config.OBSTACLE == 1 and trip_started and config.RFID_FLAG != 4 and config.RFID_FLAG != 2:
            if silence_object_detected:
                tts.silence()
                silence_object_detected = False
            if not obstacle_greeting_done1:
                config.LAST_VEHICLE_STATUS = config.VEHICLE_STATUS
                msg_to_show = 'We have detected an obstacle.\nPlease remain seated.'
                tts.say(config_info.get('STOP_MSG_OBSTACLE', config.STOP_MSG_OBSTACLE))
                border_color = config.STOP_COLOR
                msg_to_show_at_top = config_info.get('STOP_MSG', config.STOP_MSG)
                obstacle_greeting_done1 = True
                obstacle_greeting_done2 = False
            config.VEHICLE_STATUS = 3
                # continue

        # END OF THE TRIP
        if config.VEHICLE_STATUS == 4:
            # Reset the application
            config.TRACK_FACES = False
            config.TRACK_GESTURES = False
            config.VEHICLE_STATUS = 0
            if config.RFID_FLAG == 1:
                msg_to_show_at_top = config_info.get('ARRIVAL_STATION_1', config.ARRIVAL_STATION_1)
            if config.RFID_FLAG == 2:
                msg_to_show_at_top = config_info.get('ARRIVAL_STATION_2', config.ARRIVAL_STATION_2)
            border_color = config.WAIT_COLOR
            # msg_to_show_at_top = config.WAIT_MSG
            # Reinitializing the greeting dictionary
            greeting_dict = {}
            greetings_done_1 = False
            greetings_done_2 = False
            greetings_done_3 = False
            greetings_done_4 = False
            greeting_unknown = False
            greeting_photo_1 = False
            greeting_photo_2 = False
            announcements_done = False
            announcements_done_1 = False
            announcements_done_2 = False
            announcements_done_3 = False
            obstacle_greeting_done1 = False
            obstacle_greeting_done2 = False

            count_current_face = 0
            count_current_thumbsup = 0

            trip_started = False

            config.IS_FACE_RECOGNITION_DONE = bool(int(config_info.get('SKIP_FACE_RECOGNITION', config.SKIP_FACE_RECOGNITION)))

            silence_vehicle_status_0 = True
            silence_object_detected = True
            do_gesture_countdown = True

            frame_count = 0
            _frame_count = 0

            num_known_faces = 0
            num_unknown_faces = 0

            greeting_dict_unknown = {}

            # Bottom display parameters
            name_to_show = ''
            msg_to_show = ''

            # Initializing the variables
            face_boxes = []
            face_names = {}
            face_staff = {}
            face_unknown = {}
            face_unknown_final = {}

            # variable for unknown persons
            unknown = False
            last_unknown = False
            unknown_count_time = 0.0

            matching_names = []
            matching_boxes = []
            matching_ids = []
            matching_confidence = []

            last_matching_ids = []
            last_matching_boxes = []
            last_matching_names = []
            last_matching_confidence = []

            matched_name, matched_box, matched_id, matched_confidence = [], [], [], []

            send_face_boxes = []
            _send_face_boxes = []

            total_persons_visited = {}

            face_tracker = {}
            track_time = 0
            track_time2 = 0
            track_time3 = 0
            gesture_counts = {}
            final_gestures = {}
            for i in range(meta.classes):
                final_gestures[gesture_class_names[i]] = 0
            gesture_tracker = {}
            count_current_face = 0
            vehicle_speed = 0

            stop_time_object = 0.0
            stop_time_end_trip = 0.0
            stop_time_photo_click = 0.0
            stop_time_photo_click2 = 0.0
            stop_time_ann = 0.0
            stop_time_ann1 = 0.0
            start_face_recognition_greeting_time = 0.

            # Sending photo to the server
            if photo_clicked and len(final_names) >= config.MIN_NUMBER_FACES:
                # print('Sending api call for ', TRIP_ID)
                persons = []
                person_info = {}
                try:
                    with open(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.json') as infile:
                        person_info = json.load(infile)
                    with open(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.json', 'w') as outfile:
                        person_info['TripEndTime'] = str(datetime.utcnow())
                        json.dump(person_info, outfile)
                    try:
                        headers = {"Content-Type": "application/json"}
                        requests.post('http://127.0.0.1:6500/photo_api', data=json.dumps({'TripId': TRIP_ID}),
                                      verify=False,
                                      headers=headers, timeout=0.01)
                    except requests.exceptions.ReadTimeout:
                        print('Photo sync api error')
                        pass
                except Exception as e:
                    print('error in photo sync')
                    pass

            # updating the current list of Staff user_ids
            final_names = []
            photo_clicked = False
            matching_names = []
            # Creating new TRIP_ID for new trip
            TRIP_ID = str(uuid.uuid4())
            continue

        # For near RFID station. NEED TO CONFIGURE IT BASED ON SOME UNIQUE RFID TAG.
        if keypress == ord('x'):
            border_color = config.WAIT_COLOR
            msg_to_show_at_top = config_info.get('ARRIVING_MSG', config.ARRIVING_MSG)
            msg_to_show = config_info.get('ARRIVING_STATION_MSG', config.ARRIVING_STATION_MSG)
            tts.silence()
            tts.say(msg_to_show)

        # Brake status also should be implemented with CAN
        # if config.RFID_FLAG == 2 or config.RFID_FLAG == 4 and config.BRAKE_STATUS == 1:
        #     if config.RFID_FLAG == 2:
        #         border_color = config.WAIT_COLOR
        #         msg_to_show_at_top = config.NEAR_ARRIVAL_STATION_1
        #
        #     elif config.RFID_FLAG == 4:
        #         border_color = config.WAIT_COLOR
        #         msg_to_show_at_top = config.NEAR_ARRIVAL_STATION_2

        # included keypress for testing purpose only
        if (config.RFID_FLAG == 2 or config.RFID_FLAG == 4) and trip_started:
            if not trip_reset:
                if config.LAST_RFID_FLAG == 0:
                    config.LAST_RFID_FLAG = config.RFID_FLAG
                config.LAST_RFID_FLAG = config.RFID_FLAG
                tts.silence()
                exit_msg = ''
                if int(config_info.get('SKIP_FACE_RECOGNITION', config.SKIP_FACE_RECOGNITION)):
                    exit_msg = config_info.get('EXIT_MSG', config.EXIT_MSG).replace('MAIL_PHOTO_GREETING', '')
                else:
                    exit_msg = config_info.get('EXIT_MSG', config.EXIT_MSG).replace('MAIL_PHOTO_GREETING', config_info.get('MAIL_PHOTO_GREETING', config.MAIL_PHOTO_GREETING))
                if config.RFID_FLAG == 2:
                    msg_to_show_at_top = config_info.get('ARRIVAL_STATION_1', config.ARRIVAL_STATION_1)
                    tts.say(exit_msg.replace('SIDE_GREETING', config_info.get('RIGHT', config.RIGHT)).replace('STATION', config_info.get('STATION_1', config.STATION_1)))
                elif config.RFID_FLAG == 4:
                    msg_to_show_at_top = config_info.get('ARRIVAL_STATION_2', config.ARRIVAL_STATION_2)
                    tts.say(exit_msg.replace('SIDE_GREETING', config_info.get('RIGHT', config.RIGHT)).replace('STATION',
                                                                                                            config_info.get(
                                                                                                                'STATION_2',
                                                                                                                config.STATION_2)))
                config.VEHICLE_STATUS = 4
                trip_reset = True

        if keypress == ord('Q'):
            break

        # Draw the message box
        frame = drawUI(frame, border_color, w, h, trip_started)

        # PIC UI to show
        frame = ui.applyingUI(frame, msg_to_show_at_top_dict[msg_to_show_at_top]['name'])
        # displaying the processed frame
        cv2.imshow(config.WIN_NAME, frame)
        frame_count += 1
    video.release()
    cv2.destroyAllWindows()
    # try:
    #     requests.post('http://127.0.0.1:5500/teardown/', timeout=0.0000000001)
    # except requests.exceptions.ReadTimeout:
    #     pass
