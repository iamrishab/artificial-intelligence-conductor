import config
import cv2
import tensorflow as tf
from align import detect_and_align

# setting the tensorflow session configs for better inference on CPU
tf_config = tf.ConfigProto()
tf_config.intra_op_parallelism_threads = 8  # 8 specifies the number of actual CPU cores(not logical cores)
tf_config.inter_op_parallelism_threads = 8  # 8 specifies the number of actual CPU cores(not logical cores)


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config.PATH_TO_FACE_RECOGNITION_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph, config=tf_config)
        mtcnn = detect_and_align.create_mtcnn(sess, None)
        config.IS_FACE_MODEL_LOADED = True
    global padded_bounding_boxes
    global matching_names
    padded_bounding_boxes = []
    matching_names = []
    while True:
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_patches, padded_bounding_boxes = detect_and_align._detect_faces(frame_rgb, mtcnn, config.FRAME_SCALE_FACTOR)
        output = dict(face_boxes=padded_bounding_boxes)
        output_q.put(output)
