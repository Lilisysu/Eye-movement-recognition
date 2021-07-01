# 这份代码的需要预先得到txt文件
# 而视频检测中需要与第一部分的网络相结合

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Concatenate, Flatten, Activation
from tensorflow.python.keras.models import Model, load_model
import os
import cv2
import keras.utils.np_utils
from tensorflow import ConfigProto, InteractiveSession
import math
from yolo3.model import darknet_body
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ---------------------------------------------------#
#   获取数据集
# ---------------------------------------------------#
def change_size(array):
    # 将arr进行reshape
    img = Image.fromarray(np.uint8(array))
    img = img.resize((64, 32))
    arr = np.array(img)
    return arr


def get_str(data):
    # 从字符串中获取数据
    data = data.strip()
    strs = data.split(',')
    nums = np.zeros(6)
    for i in range(0, 4):
        nums[i] = int(strs[i])
    nums[4] = float(strs[4])
    nums[5] = int(strs[5])
    return nums

def get_num(data):
    str1=data
    if str1[1]=='.':
        label=int(str1[0])
    elif str1[2] == '.':
        label = int(str1[0:1])
    elif str1[3]=='.':
        label=int(str1[0:2])
    else:
        label=int(str1[0:3])
    num=label
    return int(num)

# 数据预处理,获取图片，得到eye1，eye2，place，y，堆成数据集
from PIL import Image
import numpy as np

path = './Location_ImageSet'
count = len(open("boxes.TXT", 'rU').readlines())
# 获取imgs

y_train = np.zeros(shape=(count, 2))
x_train = np.zeros(shape=(count, 32, 64, 7))
line_count = 0  # 便于找到对应图片
with open('./boxes.TXT')as f:
    for line in f.readlines():
        data = line.split(' ')
        str1 = data[0][20:-4]
        num = int(str1)
        y_train[line_count][0] = (num % 200) // 20
        y_train[line_count][1] = (num % 200) % 20

        img = cv2.imread(data[0])

        [l1, t1, r1, b1, score1, class1] = get_str(data[1])
        [l1, t1, r1, b1] = [int(l1), int(t1), int(r1), int(b1)]
        # h=b1-t1
        # [l1, t1, r1, b1] =[l1-h, t1-h, r1+h, b1+h]
        [l2, t2, r2, b2, score2, class2] = get_str(data[2])
        [l2, t2, r2, b2] = [int(l2), int(t2), int(r2), int(b2)]
        # h2=b2-t2
        # [l2, t2, r2, b2]=[l2-h2, t2-h2, r2+h2, b2+h2]
        if l1 > l2:
            [l1, t1, r1, b1], [l2, t2, r2, b2] = [l2, t2, r2, b2], [l1, t1, r1, b1]
        eye1 = img[t1:b1, l1:r1, :]
        eye2 = img[t2:b2, l2:r2, :]
        eye1 = change_size(eye1)
        eye2 = change_size(eye2)
        place = [l1, t1, r1, b1, l2, t2, r2, b2]
        x_train[line_count, :, :, 0:3] = eye1
        x_train[line_count, :, :, 3:6] = eye2
        x_train[line_count, 15, 28:36, 6] = place
        line_count += 1

# 将数据转换成tensor格式
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# model = keras.models.Sequential()
# model.add(keras.layers.Convolution2D(20, (5, 5), input_shape=(32, 64, 7), activation='relu', padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(keras.layers.Conv2D(50, (5, 5), activation='relu', padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(keras.layers.Conv2D(50, (5, 5), activation='relu', padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(keras.layers.Conv2D(50, (5, 5), activation='relu', padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(1000, activation='relu', kernel_regularizer=keras.regularizers.l1(0.01)))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Dense(200, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Dense(20, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(0.01)))
# model.add(keras.layers.Dense(2))
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.fit(x_train, y_train, validation_split=0.15, epochs=100, batch_size=125, shuffle=True)
# model.save('./logs/000/model1.h5')

model = load_model('./logs/000/model.h5')
predict = model.predict(x_train)
m_s_e_y = 0
m_s_e_x = 0
for i in range(0, count):
    print(predict[i])
    print(y_train[i])
    m_s_e_y = m_s_e_y + math.fabs(y_train[i][0] - predict[i][0])
    m_s_e_x = m_s_e_x + math.fabs(y_train[i][1] - predict[i][1])
m_s_e_x, m_s_e_y = m_s_e_x / count, m_s_e_y / count
print(m_s_e_x)
print(m_s_e_y)