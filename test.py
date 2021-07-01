import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import math
import glob
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def batch_detect_img(yolo):
    path = "./Location_ImageSet/*.jpg"  # your testset dir
    outdir = "./results/"  # save results dir
    for file in glob.glob(path):
        img = Image.open(file)
        img = yolo.detect_image(img)
        img.save(os.path.join(outdir, os.path.basename(file)))
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def get_box(yolo):
    path = "./Location_ImageSet/*.jpg"
    with open("boxes.txt", "w") as f:
        for jpg_file in glob.glob(path):
            image = Image.open(jpg_file)
            boxes, score, classes = yolo.get_box(image)
            count = 0
            for i in classes:
                if i == 0:
                    count = count + 1
            if count >= 2:
                box_nums = len(boxes)
                f.write(jpg_file)
                if box_nums != 0:
                    for i in range(0, box_nums):
                        f.write(' ')
                        f.write(str(round(boxes[i][1])))
                        f.write(',')
                        f.write(str(round(boxes[i][0])))
                        f.write(',')
                        f.write(str(round(boxes[i][3])))
                        f.write(',')
                        f.write(str(round(boxes[i][2])))
                        f.write(',')
                        f.write(str(score[i]))
                        f.write(',')
                        f.write(str(classes[i]))
                f.write('\n')
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    parser.add_argument(
        "--boxes", default=False, action="store_true",
    )

    parser.add_argument(
        "--image_batch_test", default=False, action='store_true'
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif FLAGS.boxes:
        get_box(YOLO(**vars(FLAGS)))
    elif FLAGS.image_batch_test:
        batch_detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
