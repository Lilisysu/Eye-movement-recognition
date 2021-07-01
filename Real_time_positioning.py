import tensorflow as tf
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Concatenate, Flatten, Activation
from yolo import YOLO
from PIL import Image, ImageDraw
import numpy
import os
import cv2
import math
from timeit import default_timer as timer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_boxes_scores_classes(yolo):
    model = load_model('./logs/000/model.h5')
    x_predict = numpy.zeros(shape=(1, 32, 64, 7))
    vid = cv2.VideoCapture(0)
    dynamic_factor = numpy.zeros(shape=(1, 2))
    # count = 0
    # for i in range(0, 400):
    #     background = Image.open('./BACK.jpg')
    #     draw = ImageDraw.Draw(background)
    #     if i < 100:
    #         draw.rectangle([192, 216, 288, 324], outline='red', width=5)
    #     elif i < 200:
    #         draw.rectangle([1632, 216, 1728, 324], outline='red', width=5)
    #     elif i < 300:
    #         draw.rectangle([1632, 756, 1728, 864], outline='red', width=5)
    #     else:
    #         draw.rectangle([192, 756, 288, 864], outline='red', width=5)
    #     del draw
    #     result = numpy.asarray(background)
    #     b, g, r = cv2.split(result)
    #     result = cv2.merge([r, g, b])
    #     cv2.imshow("result", result)
    #     if i >= 40:
    #         if i <= 60:
    #             return_value, frame = vid.read()
    #             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #             box_, score_, class_ = yolo.get_box(image)
    #             Open_eyes = 0
    #             Open_eye = [-1, -1]
    #             nums = 0
    #             for k in range(0, len(class_)):
    #                 if class_[k] == 0:
    #                     Open_eyes = Open_eyes + 1
    #                     Open_eye[nums] = k
    #                     nums = nums + 1
    #                 if nums == 2:
    #                     break
    #             if Open_eyes >= 2:
    #                 count = count + 1
    #                 [l1, t1, r1, b1] = [int(box_[Open_eye[0]][0]), int(box_[Open_eye[0]][1]), int(box_[Open_eye[0]][2]),
    #                                     int(box_[Open_eye[0]][3])]
    #                 [l2, t2, r2, b2] = [int(box_[Open_eye[1]][0]), int(box_[Open_eye[1]][1]), int(box_[Open_eye[1]][2]),
    #                                     int(box_[Open_eye[1]][3])]
    #                 if l1 > l2:
    #                     [l1, t1, r1, b1], [l2, t2, r2, b2] = [l2, t2, r2, b2], [l1, t1, r1, b1]
    #                 eye1 = frame[l1:r1, t1:b1, :]
    #                 eye2 = frame[l2:r2, t2:b2, :]
    #                 eye1 = change_size(eye1)
    #                 eye2 = change_size(eye2)
    #                 place = [l1, t1, r1, b1, l2, t2, r2, b2]
    #                 x_predict[0, :, :, 0:3] = eye1
    #                 x_predict[0, :, :, 3:6] = eye2
    #                 x_predict[0, 15, 28:36, 6] = place
    #                 predict = model.predict(x_predict)
    #                 dynamic_factor[0][0] = predict[0][0] - 2 + dynamic_factor[0][0]
    #                 dynamic_factor[0][1] = predict[0][1] - 2 + dynamic_factor[0][1]
    #     if i >= 140:
    #         if i <= 160:
    #             return_value, frame = vid.read()
    #             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #             box_, score_, class_ = yolo.get_box(image)
    #             Open_eyes = 0
    #             Open_eye = [-1, -1]
    #             nums = 0
    #             for k in range(0, len(class_)):
    #                 if class_[k] == 0:
    #                     Open_eyes = Open_eyes + 1
    #                     Open_eye[nums] = k
    #                     nums = nums + 1
    #                 if nums == 2:
    #                     break
    #             if Open_eyes >= 2:
    #                 count = count + 1
    #                 [l1, t1, r1, b1] = [int(box_[Open_eye[0]][0]), int(box_[Open_eye[0]][1]), int(box_[Open_eye[0]][2]),
    #                                     int(box_[Open_eye[0]][3])]
    #                 [l2, t2, r2, b2] = [int(box_[Open_eye[1]][0]), int(box_[Open_eye[1]][1]), int(box_[Open_eye[1]][2]),
    #                                     int(box_[Open_eye[1]][3])]
    #                 if l1 > l2:
    #                     [l1, t1, r1, b1], [l2, t2, r2, b2] = [l2, t2, r2, b2], [l1, t1, r1, b1]
    #                 eye1 = frame[l1:r1, t1:b1, :]
    #                 eye2 = frame[l2:r2, t2:b2, :]
    #                 eye1 = change_size(eye1)
    #                 eye2 = change_size(eye2)
    #                 place = [l1, t1, r1, b1, l2, t2, r2, b2]
    #                 x_predict[0, :, :, 0:3] = eye1
    #                 x_predict[0, :, :, 3:6] = eye2
    #                 x_predict[0, 15, 28:36, 6] = place
    #                 predict = model.predict(x_predict)
    #                 dynamic_factor[0][0] = predict[0][0] - 2 + dynamic_factor[0][0]
    #                 dynamic_factor[0][1] = predict[0][1] - 17 + dynamic_factor[0][1]
    #     if i >= 240:
    #         if i <= 260:
    #             return_value, frame = vid.read()
    #             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #             box_, score_, class_ = yolo.get_box(image)
    #             Open_eyes = 0
    #             Open_eye = [-1, -1]
    #             nums = 0
    #             for k in range(0, len(class_)):
    #                 if class_[k] == 0:
    #                     Open_eyes = Open_eyes + 1
    #                     Open_eye[nums] = k
    #                     nums = nums + 1
    #                 if nums == 2:
    #                     break
    #             if Open_eyes >= 2:
    #                 count = count + 1
    #                 [l1, t1, r1, b1] = [int(box_[Open_eye[0]][0]), int(box_[Open_eye[0]][1]), int(box_[Open_eye[0]][2]),
    #                                     int(box_[Open_eye[0]][3])]
    #                 [l2, t2, r2, b2] = [int(box_[Open_eye[1]][0]), int(box_[Open_eye[1]][1]), int(box_[Open_eye[1]][2]),
    #                                     int(box_[Open_eye[1]][3])]
    #                 if l1 > l2:
    #                     [l1, t1, r1, b1], [l2, t2, r2, b2] = [l2, t2, r2, b2], [l1, t1, r1, b1]
    #                 eye1 = frame[l1:r1, t1:b1, :]
    #                 eye2 = frame[l2:r2, t2:b2, :]
    #                 eye1 = change_size(eye1)
    #                 eye2 = change_size(eye2)
    #                 place = [l1, t1, r1, b1, l2, t2, r2, b2]
    #                 x_predict[0, :, :, 0:3] = eye1
    #                 x_predict[0, :, :, 3:6] = eye2
    #                 x_predict[0, 15, 28:36, 6] = place
    #                 predict = model.predict(x_predict)
    #                 dynamic_factor[0][0] = predict[0][0] - 7 + dynamic_factor[0][0]
    #                 dynamic_factor[0][1] = predict[0][1] - 17 + dynamic_factor[0][1]
    #     if i >= 340:
    #         if i <= 360:
    #             return_value, frame = vid.read()
    #             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #             box_, score_, class_ = yolo.get_box(image)
    #             Open_eyes = 0
    #             Open_eye = [-1, -1]
    #             nums = 0
    #             for k in range(0, len(class_)):
    #                 if class_[k] == 0:
    #                     Open_eyes = Open_eyes + 1
    #                     Open_eye[nums] = k
    #                     nums = nums + 1
    #                 if nums == 2:
    #                     break
    #             if Open_eyes >= 2:
    #                 count = count + 1
    #                 [l1, t1, r1, b1] = [int(box_[Open_eye[0]][0]), int(box_[Open_eye[0]][1]), int(box_[Open_eye[0]][2]),
    #                                     int(box_[Open_eye[0]][3])]
    #                 [l2, t2, r2, b2] = [int(box_[Open_eye[1]][0]), int(box_[Open_eye[1]][1]), int(box_[Open_eye[1]][2]),
    #                                     int(box_[Open_eye[1]][3])]
    #                 if l1 > l2:
    #                     [l1, t1, r1, b1], [l2, t2, r2, b2] = [l2, t2, r2, b2], [l1, t1, r1, b1]
    #                 eye1 = frame[l1:r1, t1:b1, :]
    #                 eye2 = frame[l2:r2, t2:b2, :]
    #                 eye1 = change_size(eye1)
    #                 eye2 = change_size(eye2)
    #                 place = [l1, t1, r1, b1, l2, t2, r2, b2]
    #                 x_predict[0, :, :, 0:3] = eye1
    #                 x_predict[0, :, :, 3:6] = eye2
    #                 x_predict[0, 15, 28:36, 6] = place
    #                 predict = model.predict(x_predict)
    #                 dynamic_factor[0][0] = predict[0][0] - 7 + dynamic_factor[0][0]
    #                 dynamic_factor[0][1] = predict[0][1] - 2 + dynamic_factor[0][1]
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # dynamic_factor = dynamic_factor / count
    center_ = numpy.zeros(shape=(10, 2))
    pointer = 0
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        background = Image.open('./BACK.jpg')
        return_value, frame = vid.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        box_, score_, class_ = yolo.get_box(image)
        Open_eyes = 0
        Open_eye = [-1, -1]
        nums = 0
        for i in range(0, len(class_)):
            if class_[i] == 0:
                Open_eyes = Open_eyes + 1
                Open_eye[nums] = i
                nums = nums + 1
            if nums == 2:
                break
        if Open_eyes < 2:
            continue
        else:
            [l1, t1, r1, b1] = [int(box_[Open_eye[0]][0]), int(box_[Open_eye[0]][1]), int(box_[Open_eye[0]][2]),
                                int(box_[Open_eye[0]][3])]
            score1 = score_[Open_eye[0]]
            class1 = class_[Open_eye[0]]
            [l2, t2, r2, b2] = [int(box_[Open_eye[1]][0]), int(box_[Open_eye[1]][1]), int(box_[Open_eye[1]][2]),
                                int(box_[Open_eye[1]][3])]
            score2 = score_[Open_eye[1]]
            class2 = class_[Open_eye[1]]
            if l1 > l2:
                [l1, t1, r1, b1], [l2, t2, r2, b2] = [l2, t2, r2, b2], [l1, t1, r1, b1]
            eye1 = frame[l1:r1, t1:b1, :]
            eye2 = frame[l2:r2, t2:b2, :]
            eye1 = change_size(eye1)
            eye2 = change_size(eye2)
            place = [l1, t1, r1, b1, l2, t2, r2, b2]
            x_predict[0, :, :, 0:3] = eye1
            x_predict[0, :, :, 3:6] = eye2
            x_predict[0, 15, 28:36, 6] = place
            predict = model.predict(x_predict)
            predict = predict - dynamic_factor
            if predict[0][0] > 9:
                predict[0][0] = 9
            if predict[0][0] < 0:
                predict[0][0] = 0
            if predict[0][1] > 19:
                predict[0][1] = 19
            if predict[0][1] < 0:
                predict[0][1] = 0
            center = [int(predict[0][0] / 10 * 1080 + 1080 / 20), int(predict[0][1] / 20 * 1920 + 1920 / 40)]
            center_[pointer % 10] = center
            left = math.floor(center_.mean(axis=0)[1]) - 48
            right = math.floor(center_.mean(axis=0)[1]) + 48
            top = math.floor(center_.mean(axis=0)[0]) - 54
            bottom = math.floor(center_.mean(axis=0)[0]) + 54
            draw = ImageDraw.Draw(background)
            draw.rectangle([left, top, right, bottom], outline='red', width=5)
            del draw
            result = numpy.asarray(background)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(255, 255, 255), thickness=5)
            b, g, r = cv2.split(result)
            result = cv2.merge([r, g, b])
            cv2.imshow("result", result)
            pointer = pointer + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    yolo.close_session()


def change_size_416(array):
    img = Image.fromarray(array, "RGB")
    img = img.resize((416, 416))
    arr = numpy.array(img)
    return arr


def change_size(array):
    # 将arr进行reshape
    img = Image.fromarray(array, 'RGB')
    img = img.resize((64, 32))
    arr = numpy.array(img)
    return arr


if __name__ == '__main__':
    get_boxes_scores_classes(YOLO())
