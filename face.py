import sys, config, pickle
import cv2
import numpy as np
import tensorflow as tf
from align import detect_and_align
from sklearn.metrics import pairwise_distances
import os
import time


class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(self):
        # Initializing the parameters
        self.distance_threshold = np.float32(config.DISTANCE_THRESHOLD)
        self.unknown_distance_threshold = np.float32(config.UNKNOWN_DISTANCE_THRESHOLD)
        self.id_names = []
        self.unknown_names = []
        self.known_encodings = []
        self.unknown_encodings = []
        self.ids = []
        self.emails = []
        self.user_types = []
        # self.isException = False
        self.unknown_count = 0
        self.unknown_matching_distances = []
        self.unknown_matching_names = []
        self.unknown_matching_encodings = []

    def find_matching_names_ids_images(self, embs):
        matching_ids, matching_distances, matching_names, matching_emails, matching_types = [], [], [], [], []
        distance_matrix = pairwise_distances(embs, self.known_encodings)
        ind = 0
        # # print('Distance Matrix Known', distance_matrix)
        for distance_row in distance_matrix:
            min_index = np.argmin(distance_row)
            if np.float32(distance_row[min_index]) < self.distance_threshold:
                # # print('matching distaances:', str(distance_row[min_index]))
                matching_ids.append(self.ids[min_index])
                matching_names.append(self.id_names[min_index])
                # matching_emails.append(self.emails[min_index])
                matching_distances.append(distance_row[min_index])
                matching_types.append(self.user_types[min_index])
            else:
                if len(self.unknown_matching_distances) > 0:
                    distance_matrix = pairwise_distances(embs, self.unknown_matching_encodings)
                    # # print('Distance Matrix Unknown', distance_matrix)
                    for distance_row in distance_matrix:
                        min_index = np.argmin(distance_row)
                        if np.float32(distance_row[min_index]) < self.unknown_distance_threshold:
                            matching_ids.append('Unknown')
                            matching_names.append(self.unknown_matching_names[min_index])
                            # matching_emails.append('Unknown')
                            matching_distances.append(self.unknown_matching_names[min_index])
                            matching_types.append('Unknown')
                        else:
                            self.unknown_count += 1
                            unknown_name = 'Unknown' + str(self.unknown_count)
                            self.unknown_matching_names.append(unknown_name)
                            self.unknown_matching_distances.append('Unknown')
                            self.unknown_matching_encodings.append(embs[ind])
                            matching_distances.append(distance_row[min_index])
                            matching_ids.append('Unknown')
                            matching_names.append(unknown_name)
                            matching_types.append('Unknown')

                else:
                    self.unknown_count += 1
                    unknown_name = 'Unknown' + str(self.unknown_count)
                    self.unknown_matching_names.append(unknown_name)
                    self.unknown_matching_distances.append('Unknown')
                    self.unknown_matching_encodings.append(embs[ind])
                    matching_distances.append(distance_row[min_index])
                    matching_ids.append('Unknown')
                    matching_names.append(unknown_name)
                    matching_types.append('Unknown')
            ind += 1
        return matching_ids, matching_names, matching_distances, matching_types

    def fetchKnownEncodings(self):
        while True:
            try:
                if os.path.getsize(config.PATH_TO_FACE_ENCODINGS_FILE) > 0:
                    with open(config.PATH_TO_FACE_ENCODINGS_FILE, 'rb') as fr:
                        [self.ids, self.id_names, self.emails, self.user_types, self.known_encodings] = pickle.load(fr)
                    break
            except Exception as e:
                print('Pickle file not found')
                sys.exit(0)
                pass


def worker(input_q, output_q):
    # load embeddings
    id_data = IdData()
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config.PATH_TO_FACE_RECOGNITION_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
        mtcnn = detect_and_align.create_mtcnn(sess, None)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        config.IS_FACE_MODEL_LOADED = True
    face_cascade = cv2.CascadeClassifier(config.HAARCASCADE_FACE_XML)
    if face_cascade is None:
        pass
    global padded_bounding_boxes
    global matching_names
    global matching_ids
    global matching_types
    # global new_matching_names
    padded_bounding_boxes = []
    matching_ids = []
    matching_names = []
    matching_types = []

    # matching_distances = []
    id_data.fetchKnownEncodings()
    while True:
        if config.TRACK_FACES:
            frame = input_q.get()
            # if frame is not None:
            # Fetching the newly added data every time
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # detection
            face_patches, padded_bounding_boxes, face_status = detect_and_align.detect_faces(frame_rgb, mtcnn,
                                                                                             face_cascade,
                                                                                             config.FRAME_SCALE_FACTOR)
            if face_patches is not None and padded_bounding_boxes is not None:
                # recognition
                if len(face_patches) > 0:
                    face_patches = np.stack(face_patches)
                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                    # Actual detection.
                    embs = sess.run(embeddings, feed_dict=feed_dict)
                    _matching_ids, _matching_names, _matching_distances, _matching_types = id_data.find_matching_names_ids_images(
                        embs)
                    matching_names = []
                    matching_ids = []
                    matching_types = []
                    count = 0

                    for name, id, type in zip(_matching_names, _matching_ids, _matching_types):
                        if face_status[count] and type != 'Staff' and type != 'Unknown' and 'Unknown' not in name:
                            matching_names.append(name)
                            matching_ids.append(id)
                        else:
                            if 'Unknown' not in name and not face_status[count] and type != 'Staff' and type != 'Unknown':
                                matching_names.append(name)
                                matching_ids.append(id)
                                # matching_types.append(type)
                            else:
                                matching_names.append('NoHaar')
                                matching_ids.append('NoHaar')
                                # matching_types.append('NoHaar')

            # output = dict(face_boxes=padded_bounding_boxes, matching_names=matching_names, matching_ids=matching_ids)
            output = dict(face_boxes=padded_bounding_boxes, matching_names=matching_names, matching_ids=matching_ids)
            output_q.put(output)
