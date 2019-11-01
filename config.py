# vehicle id
VEHICLE_ID = 'Vehicle-1'
# application model to be used while starting the vehicle
MAIN_TYPE = 'API'  # or MAIN_TYPE = 'EDGE'
# To fetch encpdings from the server
FETCH_DATA = False
# run the application as Administrator
RUN_AS_ADMIN = False
# Is ssl required during API calls
SSL = False
# To control the vehicle via remote
REMOTE_CONTROL = True

# changing FR configs
# to specify whether to do face recognition or not
SKIP_FACE_RECOGNITION = False
# to specify whether to do face recognition after thumbsup or not - currently not in use
FACE_RECOGNITION_AFTER_THUMBSUP = True
# to specify whether to do face recognition after palmup or not - currently not in use
FACE_RECOGNITION_AFTER_PALMUP = True

# logging mode
DEBUG = False
# Flag to check whether Face model is loaded
IS_FACE_MODEL_LOADED = False
# Flag to check whether gesture model is loaded
IS_GESTURE_MODEL_LOADED = False
# Flag to check whether staff model is loaded
IS_STAFF_IDS_LOADED = False
# specifies the quality of image send to the FR api on server
PHOTO_QUALITY = 65

# Path to model files
# For Face
PATH_TO_FACE_RECOGNITION_MODEL = './face/face.pb'
HAARCASCADE_FACE_XML = './xml/haarcascade_frontalface_alt_tree.xml'
# For encoding calculation and updation
PATH_TO_FACE_ENCODINGS_FILE = './data/ppm_' + VEHICLE_ID + '_data.pickle'
PATH_TO_PHOTO_SEND_FILE = './data/photo_send.pickle'

PATH_TO_CONFIG_UPDATE_FILE = './data/config.json'
# For Gesture
PATH_TO_GESTURE_DETECTION_MODEL = './gesture/model/yolov3-tiny_3_33000.weights'
PATH_TO_GESTURE_DETECTION_CONFIG = './gesture/model/yolov3-tiny_3.cfg'
PATH_TO_GESTURE_DETECTION_CLASS = './gesture/model/custom.names'

# TRIP DIR
TRIP_DIR = './trip/'

# API calls
IP = 'yamaha-motor-ces2019.com/api'
# IP = '10.167.96.48'
# Calculate encoding api call
API_GET = 'https://' + IP + '/api/encoding/get-image-for-encoding'
API_PUT_SUCCESS = 'https://' + IP + '/api/encoding/update-encoding-status'
API_PUT_FAILED = 'https://' + IP + '/api/encoding/update-image-status'

# Get Staff Id
GET_STAFF_ID = 'https://' + IP + '/api/sync/get-staff?vehicleName=' + VEHICLE_ID

# Fetch encoding api call
API_GET_SAVE_ENCODING = 'https://' + IP + '/api/sync/get-encoding?vehicleName=' + VEHICLE_ID
API_GET_UPDATE_ENCODING = 'https://' + IP + '/api/sync/update-encoding'

# Fetch/update config api call
API_GET_SAVE_CONFIG = 'https://' + IP + '/api/sync/get-configuration?vehicleName=' + VEHICLE_ID
API_GET_UPDATE_CONFIG = 'https://' + IP + '/api/sync/update-configuration'

# Photo sync API calls
API_PHOTO_SAVE = 'https://' + IP + '/api/trip/save'

# Webcam frame parameters
FRAME_WIDTH = 1920  # width of the frame
FRAME_HEIGHT = 1080  # height of the frame
FRAME_FPS = 30  # FPS of the camera
WIN_NAME = "PPM"  # Name on the display window
PIC_WIN_NAME = "Picture"  # Name on the picture display window
VIDEO_SOURCE = 0  # to start the video from the camera feed
FLIP_VERTICAL = True  # flip the video horizonatally

# UI Settings
TOP_BORDER_WIDTH_FACTOR = 0.1
BOTTOM_BORDER_WIDTH_FACTOR = 0.1

# NEW SMALL STRIP COLORS
WHITE_COLOR = (255, 255, 255)
DARK_BLUE = (145, 115, 90)
DARK_RED = (25, 9, 136)
DARK_GREEN = (149, 142, 51)
# UI COLORS
WAIT_COLOR = (210, 200, 175)
STOP_COLOR = (25, 25, 235)
START_COLOR = (170, 190, 75)
READY_COLOR = (210, 188, 93)
PHOTO_CLICK_COLOR = (244, 134, 66)

# Parameters for face detection and recognition
DISTANCE_THRESHOLD = 0.85
UNKNOWN_DISTANCE_THRESHOLD = 1.0
FRAME_SCALE_FACTOR = 7  # greater value means less face detection
RESIZE_FACTOR_FACE_DETECTION = 0.5  # resize the frame before sending to the FR API - not in use
FRAMES_SKIP_FACE_DETECTION = 1  # frames to skip for FR
FACE_BOX_COLOR = (210, 200, 175)  # Fixed UI
UNKNOWN_FACE_BOX_COLOR = (25, 25, 235)  # fixed UI

# Parameters for gesture detection
THRESH_GESTURE_DETECTION = 0.75  # Minimum detection accuracy for each gesture
HIER_THRESH_GESTURE_DETECTION = 0.5  # Fixed
NMS_THRESH_GESTURE_DETECTION = 0.45  # Fixed
GESTURE_TRACKING_THRESH_IOU = 0.4  # Fixed
FRAMES_SKIP_GESTURE_DETECTION = 2

GESTURES_BOX_COLORS = {'palmup': (25, 25, 235),
                       'thumbsup': (170, 190, 75)}  # red for palmup --> stop, green for thumbsup --> go

# TRACK GESTURE
FRAMES_BUFFER_GESTURES = 1  # Minimum no. of frames each gesture is tracked before detection
FRAMES_BUFFER_SHOW_GESTURES = FRAMES_BUFFER_GESTURES - 1  # no of frames skipped
FRAMES_BUFFER_REMOVE_GESTURES = 2  # remove thumbs up gestures after detection
FRAMES_BUFFER_PALMUP_GESTURES = 2  # no of frames tracked before palmup is counted
FRAMES_BUFFER_SHOW_PALMUP_GESTURES = FRAMES_BUFFER_PALMUP_GESTURES - 1  # remove palmup gestures after detection
THUMBSUP_COUNTDOWN_TIME = 15.0  # tracking time for thumbs up
THUMBSUP_COOLDOWN_TIME = 10.0  # time after which thumbs-up count decrease to min count

# TRACK FACES -
TRACK_FACES = False
IS_FACE_RECOGNITION_DONE = False
FRAMES_BUFFER_FACES = 20  # Minimum no. of frames each face is tracked before detection
FRAMES_BUFFER_SHOW_FACES = FRAMES_BUFFER_GESTURES - 10  # Not in use right now
FRAMES_BUFFER_REMOVE_FACES = 15  # frames after which person face is removed
FRAMES_BUFFER_REMOVE_STAFF_FACES = 15  # frames after which staff face is removed
FRAMES_BUFFER_REMOVE_UNKNOWN_FACES = 10  # frames after which unknown face is removed
FRAMES_BUFFER_ADD_UNKNOWN_FACES = 20  # frames after which unknown faces are added
# UNKNOWN_FACE_BUFFER = 20
HAAR_BUFFER = 0  # not in use

# not in use right now
FACE_TRACK_TIME = 15.0
MIN_NUMBER_FACES = 1
MAX_NUMBER_FACES = 5
MAX_BUFFERS = 5
MAX_BUFFERS_KNOWN = 8
MAX_BUFFERS_UNKNOWN = 2

SHOW_PIC_TIME = 15
CAMERA_SHUTTER_CLICK = './data/camera-shutter-click.mp3'

# TRACK GESTURES
TRACK_GESTURES = False
MIN_COUNT_THUMBSUP_FOR_START = 2
MIN_COUNT_PALMUP_FOR_STOP = 1

# Messages
MSG_COLOR = (255, 255, 255)
GREETING_SPEAK = ['Welcome ', 'Hello ', 'Konnichiwa ', 'Good day ']

MSG_SHOW_TIME = 3  # stop message show time

GREETING_MSG1_LOC = 0.2

START_MSG = "START"
ARRIVING_MSG = 'ARRIVING'

WAIT_MSG = "WAITING FOR YOU"
WAIT_MSG_LOC = 0.1
FONT_SIZE = 3  # font size of the top message for OpenCV text
FACE_RECOGNITION_GREETING_FLAG = True
STOP_TIME_BUFFER = 3.0
# Face recognition start greeting
FACE_RECOGNITION_GREETING = 'We are going to start Face Authentication now. Please look at the screen.'
# Unknown message
UNKNOWN_MSG_DISPLAY = 'Unknown person detected'
UNKNOWN_MSG = 'We are sorry. We do not recognize your face. Please see an attendant.'
# Start the trip
READY_MSG_TOP = "THUMBS UP"
READY_MSG_BOT1 = "If you are ready to go please give a thumbs-up"
READY_MSG_BOT2 = "Vehicle is starting"
# Take a photo
TAKE_PHOTO_MSG_TOP = 'We will now take a photo. Please look at the screen and give a thumbs-up'
TAKE_PHOTO_MSG = 'We will now take a photo. Please look at the screen and give a thumbs-up. 3, 2, 1, cheese.'
FEEDBACK_PHOTO_MSG = 'Thank you for your cooperation. We will depart shortly. Please remain seated.'
# Stop the trip
ANN1 = 'Thank you for choosing Yamaha PPM. A PPM stands for Public Personal Mobility.'
ANN2 = "If you want to stop in the middle of the pathway, please raise your palm."
STOP_MSG_BOT1 = "To request a stop, please raise your palm"
STOP_MSG_BOT2 = "We will now make brief stop. Please remain seated"
STOP_MSG = "STOP NOW"
READY_MSG = 'READY'
WANT_TO_STOP_MSG = 'WANT TO STOP'
STATION_1 = 'SOUTH HALL'
STATION_2 = 'SOUTH HALL'
NEAR_ARRIVAL_STATION_1 = 'next CENTRALHALL STATION'
NEAR_ARRIVAL_STATION_2 = 'next SOUTHHALL STATION'
ARRIVAL_STATION_1 = 'CENTRALHALL STATION'
ARRIVAL_STATION_2 = 'SOUTHHALL STATION'
ARRIVING_STATION_MSG = 'Thank you for taking time with us.'
# Obstacle Detected
STOP_MSG_OBSTACLE = 'We have detected an obstacle. Please remain seated.'
START_MSG_OBSTACLE = 'Thank you for your patience. We will now continue'
# Continue the trip
CONTINUE_TRIP_MSG = 'If you are ready to continue, please give a thumbs-up'

# Exit message
EXIT_MSG = 'Otsukaresamdeshita. Thank you for riding with us. Please exit to ' + \
           'the right. MAIL_PHOTO_GREETING. Please be sure to visit our booth in the' + \
           'South Hall. Domo-Arigato'
EXIT_MSG2 = 'Otsukaresamdeshita. Thank you for riding with us. Please exit to ' + \
            'the left. MAIL_PHOTO_GREETING. Please be sure to visit our booth in the' + \
            'Central Hall. Domo-Arigato'

MAIL_PHOTO_GREETING = "We will send this photo to your registered email."
LEFT = 'left'
RIGHT = 'right'

VEHICLE_STATUS = 0
LAST_VEHICLE_STATUS = 0

# CAN
RFID_FLAG = 0
LAST_RFID_FLAG = 0
LAST_RFID_START = 0
LAST_RFID_STOP = 0
VEHICLE_SPEED = 0
BRAKE_STATUS = 1
LAST_BRAKE_STATUS = 1
OBSTACLE = 0
NO_OBSTACLE = 0
LAST_OBSTACLE = 0
TOGGLE_START = False
TOGGLE_STOP = False
TOGGLE_STATUS = False

# FACE RECOGNITION API
API_GET_FACE_DATA = "http://40.118.204.85:8181/api/face/recognize"
FACE_RECOGNITION_THRESHOLD = 0.50
FRAME_RESIZING_FACE = 3  # Resizing factor of the frame during api call
