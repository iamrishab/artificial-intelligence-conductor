import urllib.request, json
import ssl
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import time
import config
import json
import os
from datetime import datetime
# from mongo_client import MongoDB

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# mongodb = MongoDB()
# col = mongodb.col


class MyAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_version=ssl.PROTOCOL_TLSv1)


def apiPutCallResponeSaveConfig():
    # s = requests.Session()
    # s.mount(config.API_GET_UPDATE_CONFIG, MyAdapter())
    headers = {"Content-Type": "application/json"}
    data = {'vehicleName': config.VEHICLE_ID, 'lastSynced': date}
    # s.put(config.API_GET_UPDATE_CONFIG, data=json.dumps(data), verify=False, headers=headers)
    requests.put(config.API_GET_UPDATE_CONFIG, data=json.dumps(data), verify=config.SSL, headers=headers)
    # s.close()


def updateConfig(data):
    if os.path.isfile(config.PATH_TO_CONFIG_UPDATE_FILE):
        with open(config.PATH_TO_CONFIG_UPDATE_FILE, 'r') as file:
            _data = json.load(file)
            with open(config.PATH_TO_CONFIG_UPDATE_FILE, 'w') as outfile:
                for row in data:
                    if row['Value'] is not None and not row['Value'] == '':
                        _data[row['Key']] = row['Value']
                        global date
                        date = row['CreatedDateTime']
                json.dump(_data, outfile)
        if config.DEBUG:
            print('Config updated')
        return True
    else:
        with open(config.PATH_TO_CONFIG_UPDATE_FILE, 'w') as outfile:
            _data = {}
            for row in data:
                _data[row['Key']] = row['Value']
            json.dump(_data, outfile)
        if config.DEBUG:
            print('Config synced')
        return True


def updateConfigMongo(data_dict):
    for row in data_dict:
        previous_value = col.find_one()
        col.update_one({row['Key']: previous_value[row['Key']]}, {"$set": {row['Key']: row['Value']}})
    return True


def createConfigMongo(data_dict):
    _temp_dict = {}
    for row in data_dict:
        _temp_dict[row['Key']] = row['Value']
    col.insert_one(_temp_dict)
    return True


def apiGetCallSaveConfig():
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(config.API_GET_SAVE_CONFIG, context=context) as response:
        if response.getcode() == 200:
            data = response.read().decode("utf-8")
            data_dict = json.loads(data)
            print('Fetched data', data_dict)
            if data_dict is not None and len(data_dict) > 0:
                is_saved = updateConfig(data_dict)
                if is_saved:
                    apiPutCallResponeSaveConfig()

                # Modifications for saving in MongoDB

                # value_exists = None
                # for row in data_dict:
                #     previous_value = col.find_one()
                #     print('Previous value', previous_value)
                #     if previous_value is not None:
                #         value_exists = col.find_one({row['Key']: previous_value[row['Key']]})
                #         print('Value exists', value_exists)
                #     break
                # if value_exists is not None:
                #     # Update the config in MongoDB
                #     is_saved = updateConfigMongo(data_dict)
                #     if is_saved:
                #         apiPutCallResponeSaveConfig()
                # else:
                #     is_saved = createConfigMongo(data_dict)
                #     if is_saved:
                #         apiPutCallResponeSaveConfig()


if __name__ == '__main__':
    try:
        apiGetCallSaveConfig()
    except Exception as e:
        print(e)
        time.sleep(5)
        pass
