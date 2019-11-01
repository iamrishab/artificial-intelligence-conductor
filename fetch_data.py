import urllib.request, json
import ssl
import logging
import pickle
import numpy as np
import os
import requests
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.poolmanager import PoolManager
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import time
import config
# from shutil import copyfile

user_ids, names, emails, encodings = [], [], [], []
date = ''

#
# class MyAdapter(HTTPAdapter):
#     def init_poolmanager(self, connections, maxsize, block=False):
#         self.poolmanager = PoolManager(num_pools=connections,
#                                        maxsize=maxsize,
#                                        block=block,
#                                        ssl_version=ssl.PROTOCOL_TLSv1)


def saveKnownEncodings(user_id, name, email, user_type, encoding):
    global user_ids
    global names
    global emails
    global encodings
    try:
        if os.path.isfile(config.PATH_TO_FACE_ENCODINGS_FILE) and os.path.getsize(config.PATH_TO_FACE_ENCODINGS_FILE) > 0:
            with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'rb') as fr:
                [user_ids, names, emails, user_types, encodings] = pickle.load(fr)
                if user_id in user_ids:
                    # Updating previous encoding
                    with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'wb') as fw:
                        get_index = user_ids.index(user_id)
                        encodings[get_index] = encoding
                        pickle.dump([user_ids, names, emails, user_types, encodings], fw, protocol=pickle.HIGHEST_PROTOCOL)
                        print('Encoding updated for user_id:', user_id)
                        return True
                else:
                    # Entering new encoding
                    with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'wb') as fw:
                        user_ids.append(user_id)
                        names.append(name)
                        emails.append(email)
                        user_types.append(user_type)
                        encodings.append(encoding)
                        pickle.dump([user_ids, names, emails, user_types, encodings], fw, protocol=pickle.HIGHEST_PROTOCOL)
                        print('Encoding inserted for user_id:', user_id)
                        return True
        else:
            with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'wb') as fw:
                pickle.dump([[user_id], [name], [email], [user_type], [encoding]], fw, protocol=pickle.HIGHEST_PROTOCOL)
                print('Encoding created for user_id:', user_id)
                return True
    except Exception as e:
        print('Exception occurred in reading from Pickle file:', e)
        time.sleep(5)
        apiGetCallSaveEncoding()


def apiPutCallResponeSaveEncoding(date):
    # s = requests.Session()
    # s.mount(config.API_GET_UPDATE_ENCODING, MyAdapter())
    data = {'vehicleName': config.VEHICLE_ID, 'lastSynced': date}
    headers = {"Content-Type": "application/json"}
    try:
        requests.put(config.API_GET_UPDATE_ENCODING, data=json.dumps(data), verify=config.SSL, headers=headers)
    except Exception as e:
        print('Fetch data error')
        time.sleep(5)
        pass

    # if url_response.status_code:
    #     logging.debug('Successfully updated Encoding API!')
    #     time.sleep(5)
    #     apiGetCallSaveEncoding()
    # s.close()


def apiGetCallSaveEncoding():
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(config.API_GET_SAVE_ENCODING, context=context) as response:
        if response.getcode() == 200:
            logging.debug('URL Hit successful')
            data = response.read().decode("utf-8")
            data_dict = json.loads(data)
            is_saved = False
            if data_dict is not None and not data_dict == []:
                for data in data_dict:
                    user_id = data['UserId']
                    name = data['Name']
                    email = data['Email']
                    encoding = data['ImageEncoding']
                    user_type = data['UserType']
                    # replacing square brackets
                    encoding = encoding.replace('[', '')
                    encoding = encoding.replace(']', '')
                    if config.DEBUG:
                        print(type(encoding))
                        print(encoding)
                    global date
                    date = data['EncodingCreatedDateTime']
                    # encoding_np_array = np.fromstring(encoding, dtype=np.float32, sep=',')
                    string_encoding_split = encoding.split(',')
                    arr = []
                    for _line in string_encoding_split:
                        arr.append(np.float64(_line))
                    arr_numpy = np.array(arr)
                    is_saved = saveKnownEncodings(user_id, name, email, user_type, arr_numpy)
                if is_saved:
                    apiPutCallResponeSaveEncoding(date)
                else:
                    time.sleep(5)
                    apiGetCallSaveEncoding()
            else:
                time.sleep(5)
                response.close()
                apiGetCallSaveEncoding()


if __name__ == '__main__':
    while True:
        try:
            apiGetCallSaveEncoding()
        except Exception as e:
            print(e)
            time.sleep(5)
            pass

