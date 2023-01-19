# chapter06.py
# -*- coding: utf-8 -*-
import csv
import copy
import itertools
import os
from collections import deque

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf


# ランドマークの画像上の位置を算出する関数
def calc_landmark_list(image, landmarks):
    landmark_point_x = []
    landmark_point_y = []
    landmark_point_z = []
    image_width, image_height = image.shape[1], image.shape[0]

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point_x.append(landmark_x)
        landmark_point_y.append(landmark_y)
        landmark_point_z.append(landmark_z)


    return landmark_point_x, landmark_point_y, landmark_point_z

def logging_csv(finger_num, gesture_id, number, csv_path, point_history_list_x, point_history_list_y, point_history_list_z):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesture_id, number, finger_num, *point_history_list_x, *point_history_list_y, *point_history_list_z])
    return

# 座標履歴を描画する関数
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                       (255, 0, 0), 2)
    return image


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # 1次元リストに変換
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    del temp_point_history[0:2]

    return temp_point_history

def pre_datashaping():
    # CSVファイル保存先
    csv_path = './point_history_add_label.csv'

    df = pd.read_csv(csv_path, index_col=0)
    df2 = df.query('fingernum == 13') - df.query('fingernum == 5')

    v_x1 = np.array([])
    for i in range(2, 35, 16):
        v_x1 = np.append(v_x1, df2.iat[0,i])


    df_fing_0 = df.query('fingernum == 0')
    df_fing_4 = df.query('fingernum == 4')
    df_fing_8 = df.query('fingernum == 8')
    df_fing_12 = df.query('fingernum == 12')
    df_fing_16 = df.query('fingernum == 16')
    df_fing_20 = df.query('fingernum == 20')

    df_4_0 = df_fing_4 - df_fing_0
    df_8_0 = df_fing_8 - df_fing_0
    df_12_0 = df_fing_12 - df_fing_0
    df_16_0 = df_fing_16 - df_fing_0
    df_20_0 = df_fing_20 - df_fing_0

    v_x2 = np.array([])

    for i in range(2, 35, 16):
        v_x2 = np.append(v_x2, df_4_0.iat[0, i])

    for i in range(2, 35, 16):
        v_x2 = np.append(v_x2, df_8_0.iat[0, i])

    for i in range(2, 35, 16):
        v_x2 = np.append(v_x2, df_12_0.iat[0, i])

    for i in range(2, 35, 16):
        v_x2 = np.append(v_x2, df_16_0.iat[0, i])

    for i in range(2, 35, 16):
        v_x2 = np.append(v_x2, df_20_0.iat[0, i])
    v = []
    v.append(v_x1)
    v.append(v_x2)

# カメラキャプチャ設定
camera_no = 0
video_capture = cv2.VideoCapture(camera_no)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,  # 最大手検出数：1
    min_detection_confidence=0.5,  # 検出信頼度閾値：0.5
    min_tracking_confidence=0.5  # トラッキング信頼度閾値：0.5
)

# ジェスチャー認識用モデルロード
tflite_save_path = './gesture_classifier.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 人差指のID
# ID_FINGER_TIP = 8

# 人差指の指先の座標履歴を保持するための変数
history_length = 16
point_history = deque(maxlen=history_length)
point_history_x = deque(maxlen=history_length)
point_history_y = deque(maxlen=history_length)
point_history_z = deque(maxlen=history_length)
point_history_num = [0]*21

csv_path = './point_history_detect.csv'

gesture_label = ['a', 'i', 'u', 'e', 'o']

while video_capture.isOpened():
    # カメラ画像取得
    ret, frame = video_capture.read()
    if ret is False:
        break
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # 鏡映しになるよう反転
    frame = cv2.flip(frame, 1)

    # MediaPipeで扱う画像は、OpenCVのBGRの並びではなくRGBのため変換
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 画像をリードオンリーにしてHands検出処理実施
    rgb_image.flags.writeable = False
    hands_results = hands.process(rgb_image)
    rgb_image.flags.writeable = True

    # 有効なランドマークが検出された場合、ランドマークを描画
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)

            # ランドマーク座標の計算
            landmark_list_x, landmark_list_y, landmark_list_z = calc_landmark_list(rgb_image, hand_landmarks)
            # 人差指の指先座標を履歴に追加
            for i in range(21):
                point_history_x.append(landmark_list_x[i])
                point_history_y.append(landmark_list_y[i])
                point_history_z.append(landmark_list_z[i])
                point_history_num[i] = point_history_num[i] + 1

                for i in range(21):
                    if point_history_num[i] == history_length:
                       point_history_list_x = list(point_history_x)
                       point_history_list_y = list(point_history_y)
                       point_history_list_z = list(point_history_z)
                       logging_csv(i,  csv_path,
                                   point_history_list_x, point_history_list_y, point_history_list_z)
                       point_history_num[i] = 0

    else:
        if len(point_history) > 0:
            point_history.popleft()

    gesture_id = 0
    if len(point_history) == history_length:
        temp_point_history = pre_process_point_history(rgb_image,
                                                       point_history)

        interpreter.set_tensor(
            input_details[0]['index'],
            np.array([temp_point_history]).astype(np.float32))
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])

        gesture_id = np.argmax(np.squeeze(tflite_results))

    # ディスプレイ表示
    cv2.putText(frame, gesture_label[gesture_id], (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)
    frame = draw_point_history(frame, point_history)
    cv2.imshow('chapter06', frame)

    # キー入力(ESC:プログラム終了)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        os.remove(csv_path)
        break

# リソースの解放
video_capture.release()
hands.close()
cv2.destroyAllWindows()
