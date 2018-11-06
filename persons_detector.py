import tensorflow as tf
import cv2
import numpy as np

tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(
    per_process_gpu_memory_fraction=0.5))


class persons_detector():

    def __init__(self):
        self.model_sess, self.image_tensor, self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections = self.load_person_detection_trained_model()
        #self.param, self.model_params, self.all_human_parts, self.build_human_parts = self.get_config()

    def load_person_detection_trained_model(self):
        # weights tarined from scratch
        trained_model = '/workspace/models/multi_pose_estimation/trained_model/frozen_inference_graph.pb'

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(trained_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            sess = tf.Session(graph=detection_graph, config=tf_config)
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            return sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

    def detect_persons_in_image(self, image):
        #PATH_TO_TEST_IMAGES_DIR = '/Users/alontetro/Desktop/Work/Outernet/curr_repository/vision_2.0/vision_2.0/person_detection/test_images/'
        #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i)) for i in range(1900, 1930,10) ]
        # print(image.shape)
        im_height, im_width = image.shape[0], image.shape[1]
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
#          image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.model_sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        all_persons_bounding_boxes = []
        for idx, score in enumerate(scores[0]):
            if (score > 0.5 and classes[0][idx] == 1):
                ymin, xmin, ymax, xmax = boxes[0][idx]

                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                all_persons_bounding_boxes.append(
                    [int(left), int(top), int(right), int(bottom)])

        return np.array(all_persons_bounding_boxes, dtype="int32")

    def visualized_detect_boxes(self, img, all_persons_bounding_boxes):

        for idx, person_bounding_box in enumerate(all_persons_bounding_boxes):
            l_x, up_y, r_x, do_y = person_bounding_box

            img = cv2.rectangle(
                img, (l_x, up_y), (r_x, do_y), (255, 255, 255), 1)
        return img
