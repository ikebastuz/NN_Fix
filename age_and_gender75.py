import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import cv2
import os
import dlib

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(
    per_process_gpu_memory_fraction=0.5))
set_session(tf.Session(config=tf_config))

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class age_and_gender:
    def __init__(self, image_size=64, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"
        self.detector = dlib.get_frontal_face_detector()
        self.face_landmark_predictor = dlib.shape_predictor(
            '/workspace/models/face_landmarks/68_model.dat')

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.model = self.create_model()
        weight_file = os.path.join(
            "/workspace/models/age_and_gender_classifier/trained_model", "age_gender_model.hdf5")
        self.model.load_weights(weight_file)

    def draw_label(self, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
        # print (label)
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(
            image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale,
                    (255, 255, 255), thickness)

    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            # format of conv_params:
            #               [ [kernel_size=("kernel width", "kernel height"),
            #               strides="(stride_vertical,stride_horizontal)",
            #               padding="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(
                            axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                                  strides=stride,
                                  padding="same",
                                  kernel_initializer=self._weight_init,
                                  kernel_regularizer=l2(self._weight_decay),
                                  use_bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f

    # "Stacking Residual Units on the same stage"

    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    def create_model(self):
        logging.debug("Creating model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                       strides=(1, 1),
                       padding="same",
                       kernel_initializer=self._weight_init,
                       kernel_regularizer=l2(self._weight_decay),
                       use_bias=self._use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(
            block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(conv1)
        conv3 = self._layer(
            block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(conv2)
        conv4 = self._layer(
            block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(pool_size=(
            8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)
        predictions_a = Dense(units=101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        return model

    def detect_multi_age_and_gender(self, img):
        img_size = 64
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = self.detector(input_img, 0)
        faces = np.empty((len(detected), img_size, img_size, 3))
        faces_bounding_points = []
        faces_landmark_points = []
        age_and_genders = []

        h, w, c = img.shape
        for i, d in enumerate(detected):
            shape = self.face_landmark_predictor(input_img, d)
            faces_landmark_points.append(
                [(shape.part(i).x / w, shape.part(i).y / h) for i in range(0, 68)])

        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                1, d.bottom() + 1, d.width(), d.height()
            faces_bounding_points.append([x1, y1, x2, y2])
            xw1 = max(int(x1 - 0.9 * w), 0)
            yw1 = max(int(y1 - 0.9 * h), 0)
            xw2 = min(int(x2 + 0.9 * w), img_w - 1)
            yw2 = min(int(y2 + 0.9 * h), img_h - 1)
            faces[i, :, :, :] = cv2.resize(
                img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        if len(detected) > 0:
            # predict ages and genders of the detected faces
            results = self.model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            for i, d in enumerate(detected):
                age_and_genders.append(
                    [int(predicted_ages[i]), "F" if predicted_genders[i][0] > 0.25 else "M", np.max(predicted_genders[i])])

        h, w, c = img.shape
        faces = []
        for i, faces_bounding_point in enumerate(faces_bounding_points):
            temp = {}
            x, y, x1, y1 = faces_bounding_point
            temp['face_bounding_box'] = [x/w, y/h, x1/w, y1/h]
            temp['face_landmarks'] = faces_landmark_points[i]
            temp['age'] = age_and_genders[i][0]
            temp['gender'] = age_and_genders[i][1]
            temp['gender_score'] = age_and_genders[i][2]
            faces.append(temp)

        return faces

    def detect_single_age_and_gender(self, face):
        weight_file = os.path.join(
            "/workspace/models/age_and_gender_classifier/trained_model", "age_gender_model.hdf5")

        # load model and weights
        img_size = 64
        if (len(face.shape) > 1):
            if(face.shape[0] > 0 and face.shape[1] > 0):
                face = cv2.resize(face, (img_size, img_size))
            else:
                return 0, "None"
        else:
            return 0, "None"

        face = face[None, ...]

        self.model.load_weights(weight_file)

        # predict ages and genders of the detected faces
        results = self.model.predict(face)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_age = results[1].dot(ages).flatten()

        predicted_gender = "F" if predicted_genders[0][0] > 0.25 else "M"

        return int(predicted_age[0]), predicted_gender
