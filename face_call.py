"""Face Recognition async call API"""
import config
import cv2
from requests_futures.sessions import FuturesSession
"""Creates the session for Async API Call"""
session = FuturesSession()


class AzureFaceApi:
    def __init__(self):
        self.url = config.API_GET_FACE_DATA
        self.matched_name = []
        self.matched_box = []
        self.matched_id = []
        self.matched_confidence = []
        self.data_dict = []
        self.resp = []
        self.frame = None
        self.flag = False

    def clearPreviousData(self):
        """Clearing the Previous Recorded Data."""
        self.matched_name[:] = []
        self.matched_box[:] = []
        self.matched_id[:] = []
        self.matched_confidence[:] = []

    def sendFrametoAzure(self):
        """Azure API Call Async."""
        self.flag = False
        try:
            img_str = cv2.imencode('.jpg', self.frame)[1].tostring()
            future = ((session.post(self.url, data=img_str, hooks={
                'response': self.response_hook,
            })))
        except Exception as e:
            print('Exception occurred')

    def response_hook(self, resp, *args, **kwargs):
        """Response Hook of Async Call."""
        try:
            resp.data = resp.json()
            if type(resp.data) is list:
                self.data_dict = resp.data
            # BUSY WAITING
            while self.flag is False:
                pass
            self.sendFrametoAzure()
        except Exception as e:
            self.sendFrametoAzure()

    def getStatus(self):
        """Returns the status of API Call can be made or not."""
        if len(self.data_dict):
            return 1
        else:
            return 0

    def getData(self):
        """Returns the Name, Box, Id and Confidence."""
        self.clearPreviousData()
        if self.data_dict is not None and not self.data_dict == []:
            for data in self.data_dict:
                if data['Confidence'] > config.FACE_RECOGNITION_THRESHOLD or data['Confidence'] == 0.0:
                    self.matched_name.append(data["Name"])
                    self.matched_box.append(
                        [data["Box"]["left"], data["Box"]["top"], data["Box"]["width"], data["Box"]["height"]])
                    self.matched_id.append(data['PersonId'])
                    self.matched_confidence.append(data['Confidence'])
            self.data_dict[:] = []
            return self.matched_name, self.matched_box, self.matched_id, self.matched_confidence

    def setFrame(self, frame, flag):
        """Sets the Frame to send to API , and flag for Busy waiting till the API receives previous response"""
        self.frame = frame
        self.flag = flag
