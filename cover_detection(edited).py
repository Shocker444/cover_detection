import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial import distance as dist
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')

label_map = 'label_map.pbtxt'
config_path = r'model_config\pipeline.config'

configs = config_util.get_configs_from_pipeline_file(config_path)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('model_config', 'ckpt-6')).expect_partial()


@tf.function
def detect_image(image):
    images, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(images, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(label_map)


def midpoint(coords):
    return (coords[0] + coords[2]) * 0.5, (coords[1] + coords[3]) * 0.5


def midline(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def display(frame):
    img = np.array(frame)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    scores = []
    coordinates = []
    detections = detect_image(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    for i in range(num_detections):
        if detections['detection_scores'][i] > 0.7:
            scores.append(detections['detection_scores'][i])
            coordinates.append(detections['detection_boxes'][i])
    coords = {}
    midpoints = {}
    for i in range(len(coordinates)):
        coords[f"{i}"] = np.multiply(coordinates[i], [h, w, h, w])
        coords[f"{i}"] = coords[f"{i}"][::-1]
        midpoints[f"{i}"] = midpoint(coords[f"{i}"])

    pivot = midpoints['0']
    print(pivot)
    for i in range(len(coords)):
        d = dist.euclidean(pivot, midpoints[f"{i}"])
        if d > 1.0:
            d_in_centimeters = d * (28.57 / np.array(w))
    label_id_offset = 1
    image_np_with_detections = img.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.7,
        agnostic_mode=False
    )
    cv2.circle(image_np_with_detections, np.array(pivot).astype(int), 10, (240, 0, 159), -1)

    for i in range(len(midpoints) - 1):
        cv2.circle(image_np_with_detections, np.array(midpoints[f"{i + 1}"]).astype(int), 10, (240, 0, 159), -1)

        cv2.line(image_np_with_detections, np.array(pivot).astype(int),
                 np.array(midpoints[f"{i + 1}"]).astype(int), (240, 0, 159), 3)
        mx, my = midline(pivot, midpoints[f"{i + 1}"])
        cv2.putText(image_np_with_detections, "{:.1f}cm".format(d_in_centimeters), (int(mx), int(my + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.79, (240, 0, 159), 2)

    return image_np_with_detections


video_path = 'vid3.mp4'
src = cv2.VideoCapture(str(video_path))
video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

# get 150 frames
for i in range(150):
    ret, frame = src.read()
    if ret:
        image_np = display(frame)
        # Display output
        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

src.release()
cv2.destroyAllWindows()
