import config
from requests_futures.sessions import FuturesSession
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

session = FuturesSession()


class StaffApi:
    def __init__(self):
        self.url = config.GET_STAFF_ID
        self.staff_ids = []
        self.data_dict = []
        self.resp = []

    def clearPreviousData(self):
        self.staff_ids[:] = []

    def getStaffIds(self):

        future = ((session.get(self.url, verify=False, hooks={
            'response': self.response_hook,
        })))

    def response_hook(self, resp, *args, **kwargs):
        resp.data = resp.json()
        # print(resp.data)
        self.data_dict = resp.data

    def getStatus(self):
        if len(self.data_dict):
            return 1
        else:
            return 0

    def getData(self):
        self.clearPreviousData()
        if self.data_dict is not None:
            for data in self.data_dict:
                self.staff_ids.append(data['Id'])
            return self.staff_ids

