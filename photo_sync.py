import json
import ssl
import pickle
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import config
import base64
from flask import Flask, request, json

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

app = Flask(__name__)


class MyAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_version=ssl.PROTOCOL_TLSv1)


@app.route('/send', methods=['GET', 'POST'])
def send_image():
    data = request.get_json()
    print('json data', data)
    # data_dict = json.loads(data)
    # print('Data dict', data_dict)
    if data is not None and len(data) > 0:
        trip_id = data['TripId']
        start_time = data['TripStartTime']
        end_time = data['TripEndTime']
        vehicle_id = config.VEHICLE_ID
        persons = data['Users']
        current_user_ids = []
        if len(persons) > 0:
            with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'rb') as fr_pickle:
                [user_ids, names, _, _] = pickle.load(fr_pickle)
                for person in persons:
                    if 'person' in person:
                        index = names.index(person)
                        current_user_ids.append(user_ids[index])
        if config.DEBUG:
            print('User Ids', current_user_ids)
        if not os.path.exists(config.TRIP_DIR):
            os.mkdir(config.TRIP_DIR)
        if os.path.isfile(config.PATH_TO_PHOTO_SEND_FILE):
            with open(config.PATH_TO_PHOTO_SEND_FILE, 'rb') as fr:
                trip_id_pickle = pickle.load(fr)
                if trip_id in trip_id_pickle:
                    return 'True'
                else:
                    with open(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg", "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read())
                    # Encoded String modification
                    modified_encoded_string = str(encoded_string)[2:-1]
                    # s = requests.Session()
                    # s.mount(config.API_PHOTO_SAVE, MyAdapter())
                    data = {'Image': modified_encoded_string, 'TripId': trip_id, 'TripStartTime': start_time, 'TripEndTime': end_time,
                            'VehicleName': vehicle_id, 'Users': current_user_ids}
                    headers = {"Content-Type": "application/json"}
                    url_response = requests.post(config.API_PHOTO_SAVE, data=json.dumps(data), verify=False, headers=headers)
                    url_response_json = url_response.json()
                    reason_phrase = url_response_json['ReasonPhrase']
                    if config.DEBUG:
                        print('Reason Phrase', reason_phrase)
                    if 'Saved Successfully'.lower() in reason_phrase.lower():
                        with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                            trip_id_pickle.append(trip_id)
                            pickle.dump(trip_id_pickle, fw, protocol=pickle.HIGHEST_PROTOCOL)
                            print('Photo saved successfully:', trip_id)
                        return "True"

                    elif "Record already exists".lower() in reason_phrase.lower():
                        with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                            pickle.dump(trip_id_pickle.append(trip_id), fw, protocol=pickle.HIGHEST_PROTOCOL)
                        return "True"
                    else:
                        return 'False'

        else:
            with open(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            # Encoded String modification
            modified_encoded_string = str(encoded_string)[2:-1]
            # s = requests.Session()
            # s.mount(config.API_PHOTO_SAVE, MyAdapter())
            data = {'Image': modified_encoded_string, 'TripId': trip_id, 'TripStartTime': start_time, 'TripEndTime': end_time,
                    'VehicleName': vehicle_id, 'Users': current_user_ids}
            headers = {"Content-Type": "application/json"}
            url_response = requests.post(config.API_PHOTO_SAVE, data=json.dumps(data), verify=False, headers=headers)
            url_response_json = url_response.json()
            reason_phrase = url_response_json['ReasonPhrase']
            if config.DEBUG:
                print('URL response', url_response_json)
                # print('Reason Phrase', reason_phrase)
            # Record already exists
            # Vehicle not registered

            if 'Saved Successfully'.lower() in reason_phrase.lower():
                with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                    pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
            #         print('Photo saved successfully:', trip_id)
                return "True"
            elif "Record already exists".lower() in reason_phrase.lower():
                with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                    pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
                return "True"
            else:
                return "False"
    else:
        return "False"


@app.route('/photo', methods=['GET', 'POST'])
def send_image_new():
    data = request.get_json()
    trip_id = data['TripId']
    print('Trip to send', trip_id)
    if os.path.isfile(config.PATH_TO_PHOTO_SEND_FILE):
        with open(config.PATH_TO_PHOTO_SEND_FILE, 'rb') as fr:
            trip_id_pickle = pickle.load(fr)
            if trip_id not in trip_id_pickle:
                if os.path.isfile(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg") and os.path.isfile(config.TRIP_DIR + trip_id + "/" + trip_id + ".json"):
                    with open(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg", "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read())
                    # Encoded String modification
                    modified_encoded_string = str(encoded_string)[2:-1]
                    persons = []
                    with open(config.TRIP_DIR + trip_id + '/' + trip_id + '.json') as infile:
                        person_info = json.load(infile)
                        if person_info is not None and len(person_info) > 0:
                            for person in person_info:
                                if 'person' in person:
                                    persons.append(person_info[person])
                    current_user_ids = []
                    if len(persons) > 0:
                        with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'rb') as fr_pickle:
                            [user_ids, names, _, types, _] = pickle.load(fr_pickle)
                            for person in persons:
                                index = names.index(person)
                                if types[index] != 'Staff':
                                    current_user_ids.append(user_ids[index])
                        # s = requests.Session()
                        # s.mount(config.API_PHOTO_SAVE, MyAdapter())
                        data = {'Image': modified_encoded_string, 'TripId': trip_id,
                                'TripStartTime': person_info['TripStartTime'],
                                'TripEndTime': person_info['TripEndTime'],
                                'VehicleName': config.VEHICLE_ID, 'Users': current_user_ids}
                        headers = {"Content-Type": "application/json"}
                        try:
                            url_response = requests.post(config.API_PHOTO_SAVE, data=json.dumps(data), verify=config.SSL, headers=headers)
                            url_response_json = url_response.json()
                            reason_phrase = url_response_json['ReasonPhrase']
                            print('URL response', reason_phrase)
                            if 'Saved Successfully'.lower() in reason_phrase.lower():
                                with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                                    pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
                                #         print('Photo saved successfully:', trip_id)
                                return "True"
                            elif "Record already exists".lower() in reason_phrase.lower():
                                with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                                    pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
                            else:
                                return 'False'
                        except Exception as e:
                            print(e)
                            print('Photo sync error')
                            pass
                else:
                    return "False"
            else:
                return "False"

    else:
        if os.path.isfile(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg") and os.path.isfile(
                config.TRIP_DIR + trip_id + "/" + trip_id + ".json"):
            with open(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            # Encoded String modification
            modified_encoded_string = str(encoded_string)[2:-1]
            persons = []
            with open(config.TRIP_DIR + trip_id + '/' + trip_id + '.json') as infile:
                person_info = json.load(infile)
                if person_info is not None and len(person_info) > 0:
                    for person in person_info:
                        if 'person' in person:
                            persons.append(person_info[person])
            current_user_ids = []
            if len(persons) > 0:
                with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'rb') as fr_pickle:
                    [user_ids, names, _, types, _] = pickle.load(fr_pickle)
                    for person in persons:
                        index = names.index(person)
                        if types[index] != 'Staff':
                            current_user_ids.append(user_ids[index])
            # s = requests.Session()
            # s.mount(config.API_PHOTO_SAVE, MyAdapter())
            data = {'Image': modified_encoded_string, 'TripId': trip_id, 'TripStartTime': person_info['TripStartTime'],
                    'TripEndTime': person_info['TripEndTime'],
                    'VehicleName': config.VEHICLE_ID, 'Users': current_user_ids}
            headers = {"Content-Type": "application/json"}
            url_response = None
            try:
                url_response = requests.post(config.API_PHOTO_SAVE, data=json.dumps(data), verify=config.SSL, headers=headers)
            except Exception as e:
                print(e)
                print('Photo sync error')
                pass
            url_response_json = url_response.json()
            reason_phrase = url_response_json['ReasonPhrase']
            print('URL response', reason_phrase)
            if 'Saved Successfully'.lower() in reason_phrase.lower():
                with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                    pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
                #         print('Photo saved successfully:', trip_id)
                return "True"
            elif "Record already exists".lower() in reason_phrase.lower():
                with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                    pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
                return "True"
            else:
                return "False"
        else:
            return 'False'


@app.route('/photo_api', methods=['GET', 'POST'])
def send_image_new_api():
    data = request.get_json()
    trip_id = data['TripId']
    print('Trip to send', trip_id)
    if os.path.isfile(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg") and os.path.isfile(config.TRIP_DIR + trip_id + "/" + trip_id + ".json"):
        with open(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        # Encoded String modification
        modified_encoded_string = str(encoded_string)[2:-1]
        persons = []
        with open(config.TRIP_DIR + trip_id + '/' + trip_id + '.json') as infile:
            person_info = json.load(infile)
            if person_info is not None and len(person_info) > 0:
                for person in person_info:
                    if 'person' in person:
                        persons.append(person_info[person])

        data = {'Image': modified_encoded_string, 'TripId': trip_id,
                'TripStartTime': person_info['TripStartTime'],
                'TripEndTime': person_info['TripEndTime'],
                'VehicleName': config.VEHICLE_ID, 'Users': persons}
        headers = {"Content-Type": "application/json"}
        try:
            url_response = requests.post(config.API_PHOTO_SAVE, data=json.dumps(data), verify=config.SSL, headers=headers)
            url_response_json = url_response.json()
            reason_phrase = url_response_json['ReasonPhrase']
            print('URL response', reason_phrase)
        except Exception as e:
            print(e)
            pass
        finally:
            return "True"
    else:
        return "False"


@app.route('/resend', methods=['GET', 'POST'])
def resend_image():
    with open(config.PATH_TO_PHOTO_SEND_FILE, 'rb') as fr:
        trip_id_pickle = pickle.load(fr)
        files = os.listdir(config.TRIP_DIR)
        for trip_id in files:
            if trip_id not in trip_id_pickle:
                with open(config.TRIP_DIR + trip_id + "/" + trip_id + ".jpg", "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                # Encoded String modification
                modified_encoded_string = str(encoded_string)[2:-1]
                persons = []
                person_info = {}
                with open(config.TRIP_DIR + trip_id + '/' + trip_id + '.json') as infile:
                    person_info = json.load(infile)
                    if person_info is not None and len(person_info) > 0:
                        for person in person_info:
                            if 'person' in person:
                                persons.append(person_info[person])
                current_user_ids = []
                if len(persons) > 0:
                    with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'rb') as fr_pickle:
                        [user_ids, names, _, _] = pickle.load(fr_pickle)
                        for person in persons:
                            if person in names:
                                index = names.index(person)
                                current_user_ids.append(user_ids[index])
                    s = requests.Session()
                    s.mount(config.API_PHOTO_SAVE, MyAdapter())
                    data = {'Image': modified_encoded_string, 'TripId': trip_id, 'TripStartTime': person_info['TripStartTime'],
                            'TripEndTime': person_info['TripEndTime'],
                            'VehicleName': config.VEHICLE_ID, 'Users': current_user_ids}
                    headers = {"Content-Type": "application/json"}
                    url_response = s.post(config.API_PHOTO_SAVE, data=json.dumps(data), verify=True, headers=headers)
                    url_response_json = url_response.json()
                    reason_phrase = url_response_json['ReasonPhrase']
                    if config.DEBUG:
                        print('URL response', url_response_json)
                        # print('Reason Phrase', reason_phrase)
                    # Record already exists
                    # Vehicle not registered

                    if 'Saved Successfully'.lower() in reason_phrase.lower():
                        with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                            pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
                        #         print('Photo saved successfully:', trip_id)
                        return "True"
                    elif "Record already exists".lower() in reason_phrase.lower():
                        with open(config.PATH_TO_PHOTO_SEND_FILE, 'wb') as fw:
                            pickle.dump([trip_id], fw, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        return 'False'
            else:
                return "False"


@app.route('/test', methods=['GET', 'POST'])
def test1():
    data = request.get_json()
    print('Trip to send', data['text'])
    return 'True'


@app.route('/', methods=['GET', 'POST'])
def test2():
    return 'Working'


if __name__ == '__main__':
    app.run(port=6500, debug=False, threaded=True)
