import config, face, gesture
from multiprocessing import Queue
from threading import Thread
from face import IdData
from gesture import GestureRecognitionYoloGpu
import time
import cv2
import uuid
import os
import json
import copy
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

# # Declaring global variables
greeting_dict = {}
greeting_dict_unknown = {}
greeting_unknown = False
random_greeting = ''
name_to_show = ''
msg_to_show = ''


# RFID
# 2 - Central Hall
# 4 - South Hall


def draw_border_face(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_border_gesture(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

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


def drawFaces(frame, final_faces):
    for name in final_faces:
        box, fc, _, freq = final_faces[name]
        [x1, y1, x2, y2] = box  # str2list(box_str)
        # if freq > int(config_info.get('HAAR_BUFFER', config.HAAR_BUFFER)) and frame_count - fc < int(config_info.get('MAX_BUFFERS', config.MAX_BUFFERS)) * int(config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)):
        #     if 'Unknown' not in name:
        #         cv2.putText(frame, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #                     (255, 255, 255), 3)
        #     if 'Unknown' in name:
        #         if name in greeting_dict_unknown and greeting_dict_unknown[name]:
        #             draw_border_face(frame, (x1, y1), (x2, y2), config.UNKNOWN_FACE_BOX_COLOR, 20, 25, 45)
        #         else:
        #             draw_border_face(frame, (x1, y1), (x2, y2), config.FACE_BOX_COLOR, 20, 25, 45)
        #     else:
        #         draw_border_face(frame, (x1, y1), (x2, y2), config.FACE_BOX_COLOR, 20, 25, 45)

        if freq > int(config_info.get('HAAR_BUFFER', config.HAAR_BUFFER)):
            if 'Unknown' not in name and frame_count - fc < int(
                    config_info.get('MAX_BUFFERS_KNOWN', config.MAX_BUFFERS_KNOWN)) * int(
                    config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)):
                cv2.putText(frame, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 3)
            if name in greeting_dict_unknown and greeting_dict_unknown[name] and frame_count - fc < int(
                    config_info.get('MAX_BUFFERS_UNKNOWN', config.MAX_BUFFERS_UNKNOWN)) * int(
                    config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)):
                draw_border_face(frame, (x1, y1), (x2, y2), config.UNKNOWN_FACE_BOX_COLOR, 20, 25, 45)
            else:
                draw_border_face(frame, (x1, y1), (x2, y2), config.FACE_BOX_COLOR, 20, 25, 45)


def sayGreetingMessage(final_faces):
    global greeting_unknown
    global name_to_show
    global random_greeting
    speech_q = Queue(6)
    is_unknown_found_in_names = False
    for name in final_faces:
        _, _, _, freq = final_faces[name]
        # GREETING the passengers
        count = 0
        if freq > int(config_info.get('HAAR_BUFFER', config.HAAR_BUFFER)):
            if 'Unknown' not in name:
                if name not in greeting_dict and not name == '':
                    speech_q.put(name)
                    greeting_dict[name] = False
            else:
                count += 1
                is_unknown_found_in_names = True
                greeting_unknown = True

    if greeting_unknown and not is_unknown_found_in_names:
        greeting_unknown = False

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
                        name_to_show = name1 + ' and ' + name2
                        tts.say(msg)
                        # print('Speech spoken', msg)

                    elif name1:
                        greeting_dict[name1] = True
                        name_to_show = name1
                        random_greeting = random.choice(config.GREETING_SPEAK)
                        msg = random_greeting + ' ' + name1
                        tts.say(msg)
                        # print('Speech spoken', msg)
    # Greeting for the Unknown persons
    if greeting_unknown:
        for name in final_faces:
            if 'Unknown' in name:
                if name not in greeting_dict_unknown:
                    greeting_dict_unknown[name] = True
                    name_to_show = 'We are sorry. We do not recognize your face. \nPlease see an attendant.'
                    tts.say(config_info.get('UNKNOWN_MSG', config.UNKNOWN_MSG))
                    break


def trackFaces(list_faces, list_names, face_tracker, frame_count):
    final_faces = {}
    for i in range(len(list_faces)):
        [x1, y1, x2, y2] = list_faces[i]
        name = list_names[i]
        boxes_to_remove = []
        boxes_to_add = {}
        bb = [x1, y1, x2, y2]
        bb_str = list2str(bb)
        for box_str in face_tracker:
            box_frame_count, box_names, is_display, box_freq = face_tracker[box_str]
            if (frame_count - box_frame_count) > config.FRAMES_BUFFER_REMOVE_FACES * config.FRAMES_SKIP_FACE_DETECTION:
                boxes_to_remove.append(box_str)
                continue
            box = str2list(box_str)
            if getIoU(bb, box) > 0.6:
                boxes_to_remove.append(box_str)
                if name in box_names:
                    box_names[name] += config.FRAMES_SKIP_FACE_DETECTION
                else:
                    box_names[name] = config.FRAMES_SKIP_FACE_DETECTION
                boxes_to_add[bb_str] = [frame_count, box_names, is_display,
                                        box_freq + config.FRAMES_SKIP_FACE_DETECTION]

        for box_str in boxes_to_remove:
            face_tracker.pop(box_str, None)

        if bb_str in boxes_to_add:
            face_tracker[bb_str] = boxes_to_add[bb_str]

        if bb_str not in face_tracker:
            box_names = {name: config.FRAMES_SKIP_FACE_DETECTION}
            is_display = False
            box_freq = config.FRAMES_SKIP_FACE_DETECTION
            face_tracker[bb_str] = [frame_count, box_names, is_display, box_freq]

    for box_str in face_tracker:
        box_frame_count, box_names, is_display, box_freq = face_tracker[box_str]

        if box_freq > config.FRAMES_BUFFER_SHOW_FACES * config.FRAMES_SKIP_FACE_DETECTION or is_display:
            if not is_display:
                is_display = True
                face_tracker[box_str] = [box_frame_count, box_names, is_display, box_freq]
            max_freq_name = ''
            max_freq = 0
            for name in box_names:
                freq = box_names[name]
                if max_freq < freq:
                    max_freq_name = name
                    max_freq = freq
            final_faces[max_freq_name] = [box_str, max_freq]

    return final_faces, face_tracker


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
                    if frame_count - box_frame_count > int(
                            config_info.get('FRAMES_BUFFER_REMOVE_GESTURES', config.FRAMES_BUFFER_REMOVE_GESTURES)):
                        boxes_to_remove.append(box_str)
                        continue
                    box = str2list(box_str)
                    if getIoU(bb, box) > 0.5:
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
                if freq > int(config_info.get('FRAMES_BUFFER_PALMUP_GESTURES', config.FRAMES_BUFFER_PALMUP_GESTURES)):
                    if max_freq < freq:
                        max_freq_gesture = gest
                        max_freq = freq
            else:
                if freq > int(config_info.get('FRAMES_BUFFER_GESTURES', config.FRAMES_BUFFER_GESTURES)):
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
            if box_class == 'palmup' and box_class_count > int(
                    config_info.get('FRAMES_BUFFER_SHOW_PALMUP_GESTURES', config.FRAMES_BUFFER_SHOW_PALMUP_GESTURES)):
                if not box_is_displayed:
                    box_is_displayed = True
                    gesture_tracker[box_str] = [box_frame_count, box_gestures, is_counted, box_is_displayed]
                draw_border_gesture(frame, (x1, y1), (x2, y2), config.GESTURES_BOX_COLORS[box_class], 20, 25, 160)
            elif box_class == 'thumbsup' and box_class_count > int(
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


def tracker(face_boxes_current_frame, matching_names, face_tracker, frame_count):
    boxes_to_remove = []
    for bb_str in face_tracker:
        bb = str2list(bb_str)
        num_frames, _ = face_tracker[bb_str]
        if (num_frames - frame_count) > frames_buffer * frames_to_skip:
            boxes_to_remove.append(bb_str)
            continue
        iou_values = []
        for box in face_boxes_current_frame:
            iou = getIoU(box, bb)
            iou_values.append(iou)

        max_iou = max(iou_values)
        if max_iou > thresh_iou:
            boxes_to_remove.append(bb_str)

    for bb_str in boxes_to_remove:
        face_tracker.pop(bb_str)

    print('Face boxes current frame', face_boxes_current_frame)
    if face_boxes_current_frame is not None:
        for i in range(len(face_boxes_current_frame)):
            if face_boxes_current_frame[i] is not None:
                box_str = list2str(face_boxes_current_frame[i])
                name = matching_names[i]
                face_tracker[box_str] = [frame_count, name]
    return face_tracker


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
    # Top bar rectangle
    cv2.rectangle(img, (0, 0), (img_width, int(float(img_height) * config.TOP_BORDER_WIDTH_FACTOR)), border_color, -1)
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
    if not trip_started:
        font = ImageFont.truetype('./font/Square721.TTF', 100)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((point1.x + 200, point1.y + 8), 'Welcome to the PPM', font=font, fill=(100, 100, 100, 255))
        img = np.array(img_pil)

    if config.TRACK_FACES:
        if len(greeting_dict) > 0 or len(greeting_dict_unknown) > 0:
            font = ImageFont.truetype('./font/Square721.TTF', 50)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            if not greeting_unknown:
                draw.text((point1.x + 120, point1.y + 5),
                          random_greeting + ' ' + name_to_show + '. Welcome to PPM.', font=font,
                          fill=(100, 100, 100, 255))
                img = np.array(img_pil)
            else:
                draw.text((point1.x + 120, point1.y + 5), name_to_show, font=font, fill=(100, 100, 100, 255))
                img = np.array(img_pil)

    if config.TRACK_GESTURES:
        font = ImageFont.truetype('./font/Square721.TTF', 50)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((point1.x + 120, point1.y + 5), msg_to_show, font=font,
                  fill=(100, 100, 100, 255))
        img = np.array(img_pil)
    return img


# From here the application starts execution
if __name__ == '__main__':

    # Creating the speech object
    tts = Speech()

    if config.RUN_AS_ADMIN:
        # Running as an admin
        rc = 0
        if not admin_check.isUserAdmin():
            print("You're not an admin.", os.getpid())
            rc = admin_check.runAsAdmin()
        else:
            print("You are an admin!", os.getpid())
            rc = 0

    # Loading data from pickle file
    print("Loading data from pickle file...")
    while not (os.path.isfile(config.PATH_TO_FACE_ENCODINGS_FILE) and os.path.getsize(
            config.PATH_TO_FACE_ENCODINGS_FILE) > 0):
        continue
    print("Pickle file loaded")

    # Loading config file
    print("Loading data from config file...")
    while not (os.path.isfile(config.PATH_TO_CONFIG_UPDATE_FILE)):
        continue
    with open(config.PATH_TO_CONFIG_UPDATE_FILE) as infile:
        config_info = json.load(infile)
    print("Config file loaded")

    video = cv2.VideoCapture(0)

    # # Creating the Queue for Face Recognition frame processing
    input_q = Queue(5)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    # Starting the FR thread
    for i in range(1):
        t = Thread(target=face.worker, args=(input_q, output_q))
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
    face_obj = IdData()

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
    final_faces = {}
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
    count_faces = 0
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
    announcements_done = False
    announcements_done_1 = False
    announcements_done_2 = False

    silence_vehicle_status_0 = True
    silence_object_detected = True
    do_gesture_countdown = True

    stop_time_object = 0.0
    stop_time_end_trip = 0.0
    stop_time_photo_click = 0.0
    stop_time_photo_click2 = 0.0
    stop_time_ann = 0.0
    stop_time_ann1 = 0.0
    trip_start_time = 0.0

    # CHANGES
    frame_count = 0
    face_boxes = []
    matching_names = []
    names_boxes = {}
    font = ImageFont.truetype('./font/Square721.TTF', 80)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_copy = copy.deepcopy(frame)
        if config.FLIP_VERTICAL:
            frame = cv2.flip(frame, 1)
        curr_time = time.time()

        h, w = frame.shape[:2]
        keypress = cv2.waitKey(1)

        # Draw the message box
        frame = drawUI(frame, border_color, w, h, trip_started)

        if greeting_unknown and config.TRACK_FACES:
            border_color = config.STOP_COLOR
            msg_to_show_at_top = config_info.get('STOP_MSG', config.STOP_MSG)
            track_time = curr_time
        else:
            if config.TRACK_FACES:
                border_color = config.WAIT_COLOR
                msg_to_show_at_top = config_info.get('WAIT_MSG', config.WAIT_MSG)

        # show wait message
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # show wait message
        if msg_to_show_at_top == config_info.get('WAIT_MSG', config.WAIT_MSG):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 400),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('START_MSG', config.START_MSG):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 700),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('READY_MSG_TOP', config.READY_MSG_TOP):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 575),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('STOP_MSG', config.STOP_MSG):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 600),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('NEAR_ARRIVAL_STATION_1', config.NEAR_ARRIVAL_STATION_1):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 550),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('ARRIVAL_STATION_1', config.NEAR_ARRIVAL_STATION_1):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 550),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('ARRIVAL_STATION_2', config.NEAR_ARRIVAL_STATION_1):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 550),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('NEAR_ARRIVAL_STATION_2', config.NEAR_ARRIVAL_STATION_2):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 550),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('READY_MSG', config.READY_MSG):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 700),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('ARRIVING_MSG', config.ARRIVING_MSG):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 700),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        elif msg_to_show_at_top == config_info.get('WANT_TO_STOP_MSG', config.WANT_TO_STOP_MSG):
            draw.text((int(loc_msg_to_show_at_top * float(w) + 600),
                       int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140), msg_to_show_at_top, font=font,
                      fill=(255, 255, 255, 255))
            frame = np.array(img_pil)

        if keypress == ord('a') or keypress == ord('c'):
            if not trip_started:
                config.LAST_RFID_FLAG = config.RFID_FLAG
                if keypress == ord('a'):
                    config.RFID_FLAG = 1
                elif keypress == ord('c'):
                    config.RFID_FLAG = 3
                print('rfid flag', config.RFID_FLAG)
                config.TRACK_FACES = True
                trip_start_time = str(datetime.utcnow())
                trip_started = True
                frame_count = 0
                tts.silence()

        if config.RFID_FLAG == 1 or config.RFID_FLAG == 3:
            # FACE TRACKING
            if config.TRACK_FACES and frame_count % int(
                    config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)) == 0:
                frame_copy_face = copy.deepcopy(frame)
                cropped_frame_face = frame_copy_face[int(float(h) * config.TOP_BORDER_WIDTH_FACTOR):int(
                    float(h) * (1.0 - 0.15)), 0:w]
                input_q.put(cropped_frame_face)
                # input_q.put(frame)
            if output_q.empty():
                pass  # fill up queue
            elif config.TRACK_FACES:
                data = output_q.get()
                # face_boxes = data['face_boxes']
                __face_boxes = data['face_boxes']
                face_boxes = []
                for bb in __face_boxes:
                    bb[1] += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
                    bb[3] += int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)
                    face_boxes.append(bb)
                matching_names = data['matching_names']
                names_to_remove = []
                names_to_add = {}
                boxes_names_iou = {}
                if len(face_boxes) > 0 and len(matching_names) > 0:
                    for bb, name in zip(face_boxes, matching_names):
                        max_iou = 0.2
                        box_max_iou = []
                        person_max_iou = ''
                        if name in names_boxes:
                            [_, _, haar_box, freq] = names_boxes[name]
                            names_boxes[name] = [bb, frame_count, haar_box, freq + 1]
                            continue
                        for person in names_boxes:
                            iou = getIoU(names_boxes[person][0], bb)
                            if iou > max_iou:
                                max_iou = iou
                                box_max_iou = bb
                                person_max_iou = name
                        if max_iou > 0.4:
                            box_max_iou_str = list2str(box_max_iou)
                            process_box = False
                            if box_max_iou_str in boxes_names_iou:
                                _name, iou = boxes_names_iou[box_max_iou_str]
                                if iou < max_iou:
                                    process_box = True
                            else:
                                process_box = True
                            if process_box:
                                if len(name) > 0 and 'NoHaar' not in name and len(person_max_iou) > 0:
                                    if 'Unknown' in person_max_iou and 'Unknown' not in name:
                                        boxes_names_iou[box_max_iou_str] = [name, max_iou]
                                    if 'Unknown' in name and 'Unknown' not in person_max_iou:
                                        boxes_names_iou[box_max_iou_str] = [person_max_iou, max_iou]
                                    if 'Unknown' not in name and 'Unknown' not in person_max_iou and not name == person_max_iou:
                                        print('RARE CASE ASSUMED FOUND')
                                else:
                                    boxes_names_iou[box_max_iou_str] = [person_max_iou, max_iou]
                        else:
                            if len(names_boxes) < 6 and len(name) > 0 and 'NoHaar' not in name:
                                names_to_add[name] = bb
                                if 'Unknown' not in name and len(
                                        names_boxes) == 6 and 'Unknown' in person_max_iou and len(
                                        name) > 0 and 'NoHaar' not in name:
                                    names_to_add[name] = bb
                                    names_to_remove.append(person_max_iou)

                    for box_str in boxes_names_iou:
                        box = str2list(box_str)
                        person, _ = boxes_names_iou[box_str]
                        if person in names_boxes:
                            _, _, haar_bb, freq = names_boxes[person]
                            names_boxes[person] = [box, frame_count, haar_bb, freq + 1]
                        else:
                            max_iou = 0.
                            person_max_iou = ''
                            for name in names_boxes:
                                iou = getIoU(names_boxes[name][0], box)
                                if iou > max_iou:
                                    max_iou = iou
                                    person_max_iou = name
                            if max_iou > 0. and 'NoHaar' not in person:
                                names_to_remove.append(person_max_iou)
                                names_to_add[person] = box

                    for name in names_boxes:
                        if 'Unknown' in name:
                            if frame_count - names_boxes[name][1] > int(
                                    config_info.get('MAX_BUFFERS_UNKNOWN', config.MAX_BUFFERS_UNKNOWN)) * int(
                                    config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)):
                                names_to_remove.append(name)
                        else:
                            if frame_count - names_boxes[name][1] > int(
                                    config_info.get('MAX_BUFFERS_KNOWN', config.MAX_BUFFERS_KNOWN)) * int(
                                    config_info.get('FRAMES_SKIP_FACE_DETECTION', config.FRAMES_SKIP_FACE_DETECTION)):
                                names_to_remove.append(name)

                    for name in names_to_remove:
                        names_boxes.pop(name)

                    for name in names_to_add:
                        if 'NoHaar' in name:
                            continue
                        box = names_to_add[name]
                        names_boxes[name] = [box, frame_count, box, 1]
                    # for name in names_boxes:
                    # box = names_boxes[name][0]
                    names_to_remove = []
                    for name in names_boxes:
                        if 'Unknown' in name:
                            box_unknown, _, _, _ = names_boxes[name]
                            for _name in names_boxes:
                                if 'Unknown' not in _name:
                                    box_known, _, _, _ = names_boxes[_name]
                                    iou = getIoU(box_unknown, box_known)
                                    if iou > 0.4:
                                        print('overlap found between ' + name + ' and ' + _name)
                                    if iou > 0.8:
                                        names_to_remove.append(name)
                                        break
                    for name in names_to_remove:
                        names_boxes.pop(name)

                    if len(names_boxes) >= int(
                            config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)) and track_time == 0:
                        track_time = curr_time
                        count_current_face = len(final_faces.keys())
                    # Counter Reset
                    if len(names_boxes) <= 6 and abs(len(names_boxes) - count_current_face) >= 1:
                        count_current_face = len(names_boxes)
                        track_time = curr_time

            if config.TRACK_FACES:
                sayGreetingMessage(names_boxes)
                if greeting_unknown:
                    border_color = config.STOP_COLOR
                    msg_to_show_at_top = config_info.get('STOP_MSG', config.STOP_MSG)
                    track_time = curr_time

            # countdown is over and greetings for all known name are already announced
            if config.TRACK_FACES and (
                    len(names_boxes) >= int(config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)) and (
                    curr_time - track_time) > float(config_info.get('FACE_TRACK_TIME', config.FACE_TRACK_TIME))):
                config.TRACK_FACES = False
                config.TRACK_GESTURES = True
                config.VEHICLE_STATUS = 0
                final_names = names_boxes.keys()
                track_time = curr_time
                count_faces = len(names_boxes)

            if config.TRACK_FACES:
                delta_time = 0
                if len(names_boxes) > 0:
                    drawFaces(frame, names_boxes)

            # GESTURE TRACKING
            if config.TRACK_GESTURES:
                config.TRACK_FACES = False
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
                        final_faces = {}
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

                    if do_gesture_countdown:
                        if track_time3 == 0:
                            track_time3 = curr_time
                        # Counter Reset
                        if abs(final_gestures['thumbsup'] - count_current_thumbsup) >= 1:
                            count_current_thumbsup = final_gestures['thumbsup']
                            track_time3 = curr_time

                    if final_gestures['thumbsup'] >= int(
                            config_info.get('MIN_COUNT_THUMBSUP_FOR_START', config.MIN_COUNT_THUMBSUP_FOR_START)) or (
                            final_gestures['thumbsup'] >= int(config_info.get('MIN_COUNT_THUMBSUP_FOR_START',
                                                                              config.MIN_COUNT_THUMBSUP_FOR_START)) and curr_time - track_time3) >= float(
                            config_info.get('THUMBSUP_COUNTDOWN_TIME', config.THUMBSUP_COUNTDOWN_TIME)):

                        if track_time2 == 0:
                            track_time2 = curr_time
                            do_gesture_countdown = False
                            track_time3 = 0
                            count_current_face = 0

                        if not keypress == ord('x'):
                            # changing the state color instantly
                            border_color = config.READY_COLOR
                            msg_to_show_at_top = config_info.get('READY_MSG', config.READY_MSG)

                        if curr_time - track_time2 > config.MSG_SHOW_TIME:
                            # border_color = config.READY_COLOR
                            # msg_to_show_at_top = config_info.get('READY_MSG', config.READY_MSG)
                            track_time2 = curr_time

                        # Greeting photo 1
                        if len(names_boxes) < int(config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)):
                            photo_clicked = True
                            greeting_photo_1 = True
                            greeting_photo_2 = True

                        if len(names_boxes) >= int(config_info.get('MIN_NUMBER_FACES',
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

                            # changes for UI in the photo click
                            # Top bar rectangle

                            thickness = 30
                            cornerRadius = 20
                            horizontal_margin = 300
                            vertical_margin = 25

                            point1 = Point(horizontal_margin, int(float(h) * (1.0 - 0.15)))
                            point2 = Point(w - horizontal_margin, point1.y)
                            point3 = Point(point2.x, h - 1 - vertical_margin)
                            point4 = Point(point1.x, point3.y)

                            drawEllipseRectangle(frame_copy, point1, point2, point3, point4, cornerRadius, thickness,
                                                 border_color)

                            cv2.rectangle(frame_copy, (0, 0),
                                          (w, int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)),
                                          config.START_COLOR, -1)
                            # Bottom bar rectangle
                            cv2.rectangle(frame_copy, (0, int(float(h) * (1.0 - config.TOP_BORDER_WIDTH_FACTOR))),
                                          (w, h - 1), config.START_COLOR, -1)
                            frame_copy = drawUI(frame_copy, config.START_COLOR, w, h, trip_started)
                            _img_pil = Image.fromarray(frame_copy)
                            _draw = ImageDraw.Draw(_img_pil)
                            _draw.text((int(loc_msg_to_show_at_top * float(w) + 700),
                                        int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140),
                                       '#CES2019', font=font,
                                       fill=(255, 255, 255, 255))
                            _draw.text((point1.x + 120, point1.y + 5),
                                       'Your face is your ticket. \n YAMAHA PPM moves on your go',
                                       font=ImageFont.truetype('./font/Square721.TTF', 50),
                                       fill=(100, 100, 100, 255))
                            frame_copy = np.array(_img_pil)
                            # End of the changes
                            cv2.imshow(config.PIC_WIN_NAME, frame_copy)
                            cv2.waitKey(100 * config.SHOW_PIC_TIME)

                            cv2.imwrite(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.jpg', frame_copy)
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
                            config.VEHICLE_STATUS = 1
                            gesture_counts = {}
                            final_gestures = {}
                            for i in range(meta.classes):
                                final_gestures[gesture_class_names[i]] = 0
                            gesture_tracker = {}
                            try:
                                requests.post('http://127.0.0.1:5500/toggle/', timeout=0.0000000001)
                                break
                            except requests.exceptions.ReadTimeout:
                                pass

                if config.VEHICLE_STATUS == 1:
                    # frame_copy_gesture = copy.deepcopy(frame)
                    # cropped_frame_gesture_ = frame_copy_gesture[int(float(h) * config.TOP_BORDER_WIDTH_FACTOR):int(
                    #     float(h) * (1.0 - 0.15)), 0:w]
                    border_color = config.WAIT_COLOR

                    # Greeting photo 1
                    if len(names_boxes) < int(config_info.get('MIN_NUMBER_FACES', config.MIN_NUMBER_FACES)):
                        photo_clicked = True
                        greeting_photo_1 = True
                        greeting_photo_2 = True

                    if len(names_boxes) >= int(config_info.get('MIN_NUMBER_FACES',
                                                               config.MIN_NUMBER_FACES)) and not photo_clicked and not greeting_photo_1:
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
                        # changes for UI in the photo click
                        # Top bar rectangle

                        thickness = 30
                        cornerRadius = 20
                        horizontal_margin = 300
                        vertical_margin = 25

                        point1 = Point(horizontal_margin, int(float(h) * (1.0 - 0.15)))
                        point2 = Point(w - horizontal_margin, point1.y)
                        point3 = Point(point2.x, h - 1 - vertical_margin)
                        point4 = Point(point1.x, point3.y)

                        drawEllipseRectangle(frame_copy, point1, point2, point3, point4, cornerRadius, thickness,
                                             border_color)

                        cv2.rectangle(frame_copy, (0, 0),
                                      (w, int(float(h) * config.TOP_BORDER_WIDTH_FACTOR)),
                                      config.START_COLOR, -1)
                        # Bottom bar rectangle
                        cv2.rectangle(frame_copy, (0, int(float(h) * (1.0 - config.TOP_BORDER_WIDTH_FACTOR))),
                                      (w, h - 1), config.START_COLOR, -1)
                        frame_copy = drawUI(frame_copy, config.START_COLOR, w, h, trip_started)
                        _img_pil = Image.fromarray(frame_copy)
                        _draw = ImageDraw.Draw(_img_pil)
                        _draw.text((int(loc_msg_to_show_at_top * float(w) + 700),
                                    int(float(h) * 4. * config.TOP_BORDER_WIDTH_FACTOR / 3.) - 140),
                                   '#CES2019', font=font,
                                   fill=(255, 255, 255, 255))
                        _draw.text((point1.x + 120, point1.y + 5),
                                   'Your face is your ticket. \n YAMAHA PPM moves on your go',
                                   font=ImageFont.truetype('./font/Square721.TTF', 50),
                                   fill=(100, 100, 100, 255))
                        frame_copy = np.array(_img_pil)
                        # End of the changes

                        # while track_time - time.time() < config.SHOW_PIC_TIME:
                        cv2.imshow(config.PIC_WIN_NAME, frame_copy)
                        cv2.waitKey(100 * config.SHOW_PIC_TIME)

                        cv2.imwrite(config.TRIP_DIR + TRIP_ID + '/' + TRIP_ID + '.jpg', frame_copy)
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
                        ann1 = 'Thank you for choosing Yamaha PPM.\nA PPM stands for Public Personal Mobility.'
                        ann2 = 'If you want to stop in the middle of the \npathway, please raise your palm.'
                        if not announcements_done:
                            # msg_to_show = "Vehicle is starting.\nTo request a stop, please raise your palm"

                            if config.RFID_FLAG == 1 and not announcements_done_1:
                                msg_to_show = 'We are now approaching Central Hall Station'
                                msg_to_show_at_top = config_info.get('NEAR_ARRIVAL_STATION_1',
                                                                     config.NEAR_ARRIVAL_STATION_1)
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

                                if curr_time - stop_time_ann1 > 5.0 and stop_time_ann1 != 0.0:
                                    # announcement 2
                                    msg_to_show_at_top = config_info.get('WANT_TO_STOP_MSG', config.WANT_TO_STOP_MSG)
                                    msg_to_show = ann2
                                    tts.say(config.ANN2)
                                    announcements_done = True

                        if announcements_done:
                            msg_to_show_at_top = config_info.get('WANT_TO_STOP_MSG', config.WANT_TO_STOP_MSG)
                            msg_to_show = ann2
                        # ## END OF LATEST CHANGES

                        num, dets = gesture.detect(frame, gesture_obj)
                        # gesture_detection_palm = False
                        final_gestures, gesture_tracker = trackGestures(num, dets, meta, gesture_tracker, frame_count,
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
                                tts.say(config_info.get('STOP_MSG_BOT2', config.STOP_MSG_BOT2))
                            if greetings_done_4:
                                # giving the stop signal to CAN
                                try:
                                    requests.post('http://127.0.0.1:5500/toggle/', timeout=0.0000000001)
                                    break
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

        # Manual or Remote Control Operation (FOR STARTING THE VEHICLE ONLY)
        if keypress == ord('s'):
            # config.BRAKE_STATUS = 0
            config.TRACK_FACES = False
            config.TRACK_GESTURES = True
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

        # Manual or Remote Control Operation (FOR STOPPING THE VEHICLE ONLY)
        if keypress == ord('m'):
            # config.BRAKE_STATUS = 1
            config.TRACK_FACES = False
            config.TRACK_GESTURES = True
            config.VEHICLE_STATUS = 2
            msg_to_show = 'We will now make brief stop.\nPlease remain seated'
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

        # OBSTACLE DETECTION
        if config.VEHICLE_STATUS == 3:
            if keypress == ord('p'):
                config.OBSTACLE = 0

            if config.OBSTACLE == 0:
                if stop_time_object == 0:
                    border_color = config.WAIT_COLOR
                    if config.RFID_FLAG == 1:
                        msg_to_show_at_top = config_info.get('NEAR_ARRIVAL_STATION_1', config.NEAR_ARRIVAL_STATION_1)
                    elif config.RFID_FLAG == 3:
                        msg_to_show_at_top = config_info.get('NEAR_ARRIVAL_STATION_2', config.NEAR_ARRIVAL_STATION_2)
                    stop_time_object = time.time()
                    msg_to_show = 'Thank you for your patience.\nWe will now continue'
                    tts.say(config.START_MSG_OBSTACLE)
                if time.time() - stop_time_object > 3:
                    stop_time_object = 0
                    config.LAST_OBSTACLE = config.OBSTACLE
                    config.TRACK_GESTURES = True
                    config.VEHICLE_STATUS = 1
                    track_time = curr_time
                    gesture_counts = {}
                    final_gestures = {}
                    for i in range(meta.classes):
                        final_gestures[gesture_class_names[i]] = 0
                    gesture_tracker = {}
                    try:
                        requests.post('http://127.0.0.1:5500/toggle/', timeout=0.0000000001)
                        break
                    except requests.exceptions.ReadTimeout:
                        pass
                    silence_object_detected = True

        if keypress == ord('o'):
            config.OBSTACLE = 1
            if config.LAST_OBSTACLE == config.OBSTACLE:
                continue
            config.LAST_OBSTACLE = config.OBSTACLE

            config.VEHICLE_STATUS = 3
            if silence_object_detected:
                tts.silence()
                silence_object_detected = False
            msg_to_show = 'We have detected an obstacle.\nPlease remain seated.'
            tts.say(config_info.get('STOP_MSG_OBSTACLE', config.STOP_MSG_OBSTACLE))
            border_color = config.STOP_COLOR
            msg_to_show_at_top = config_info.get('STOP_MSG', config.STOP_MSG)
            # continue

        # END OF THE TRIP
        if config.VEHICLE_STATUS == 4:
            # Reset the application
            config.TRACK_FACES = True
            config.TRACK_GESTURES = False
            config.VEHICLE_STATUS = 0
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
            count_current_face = 0
            count_current_thumbsup = 0

            trip_started = False
            face_obj.unknown_count = 0

            silence_vehicle_status_0 = True
            silence_object_detected = True
            do_gesture_countdown = True

            frame_count = 0

            num_known_faces = 0
            num_unknown_faces = 0

            greeting_dict_unknown = {}

            # Bottom display parameters
            name_to_show = ''
            msg_to_show = ''

            # Initializing the variables
            face_boxes = []
            matching_names = []
            final_faces = {}
            face_tracker = {}
            final_names = []
            track_time = 0
            track_time2 = 0
            track_time3 = 0
            gesture_counts = {}
            final_gestures = {}
            for i in range(meta.classes):
                final_gestures[gesture_class_names[i]] = 0
            gesture_tracker = {}
            count_faces = 0
            count_current_face = 0
            vehicle_speed = 0

            stop_time_object = 0.0
            stop_time_end_trip = 0.0
            stop_time_photo_click = 0.0
            stop_time_photo_click2 = 0.0
            stop_time_ann = 0.0
            stop_time_ann1 = 0.0

            if config.DEBUG:
                print('Names boxes:', names_boxes)

            # Sending photo to the server
            if photo_clicked and len(names_boxes) >= config.MIN_NUMBER_FACES:
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
                        requests.post('http://127.0.0.1:6000/photo', data=json.dumps({'TripId': TRIP_ID}), verify=False,
                                      headers=headers, timeout=0.000001)
                    except requests.exceptions.ReadTimeout:
                        print('Photo sync api error')
                        pass
                except Exception as e:
                    print('error in photo sync')
                    pass

            photo_clicked = False
            # changes
            face_boxes = []
            matching_names = []
            names_boxes = {}
            # Creating new TRIP_ID for new trip
            TRIP_ID = str(uuid.uuid4())
            continue

        # For near RFID station
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

        # For toggling brake status
        if keypress == ord('m'):
            config.BRAKE_STATUS = int(not bool(config.BRAKE_STATUS))
        #     msg_to_show_at_top = config.WAIT_MSG

        # included keypress only for testing purpose
        if keypress == ord('b') or keypress == ord('d'):
            if config.LAST_RFID_FLAG == 0:
                config.LAST_RFID_FLAG = config.RFID_FLAG
            config.LAST_RFID_FLAG = config.RFID_FLAG
            tts.silence()
            if keypress == ord('b'):
                config.RFID_FLAG = 2
                msg_to_show_at_top = config.ARRIVAL_STATION_1
                tts.say(config_info.get('EXIT_MSG', config.EXIT_MSG))
            elif keypress == ord('d'):
                config.RFID_FLAG = 4
                msg_to_show_at_top = config.ARRIVAL_STATION_2
                tts.say(config_info.get('EXIT_MSG2', config.EXIT_MSG2))

            msg_to_show = 'Thank you for riding with us.\nWe will send this photo to your registered email'

            config.VEHICLE_STATUS = 4
            continue

        if keypress == ord('q'):
            break

        cv2.imshow(config.WIN_NAME, frame)
        frame_count += 1
    video.release()
    cv2.destroyAllWindows()
    try:
        requests.post('http://127.0.0.1:5500/teardown/', timeout=0.0000000001)
    except requests.exceptions.ReadTimeout:
        pass
