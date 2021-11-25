import mot_class as mtc
import os              
import argparse
import yolo_object_detection as yolo_obj
import cv2
import imutils
from imutils.video import FPS

def read_user_input_info():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file")
    ap.add_argument("-t","--tracker", required=True,
        help="trackername to input tracker type")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    args = read_user_input_info()
    vs = cv2.VideoCapture(args["video"])
    tracker = args["tracker"]
    print("TRACKER:%s" % tracker)
    (grabbed, frame) = vs.read()
    resize_width = 3840
    frame = imutils.resize(frame, width=resize_width)
    (h, w) = frame.shape[:2]

    fps = FPS().start()
    yolo = yolo_obj.yolo_object_detection('person')
    bboxes = []
    bboxes = yolo.run_detection(frame)
    #print(bboxes)

    # bbox[x y w h]
    for i,box in enumerate(bboxes):
        if box[0] + box[2] >= w:
            dx = w - box[0]
            box[2] = dx
        if box[1] + box[3] >= h:
            dy = h - box[1]
            box[3] = dy


    cv2.imwrite('output.jpg', frame)

    fps.stop()
    print("[INFO] yolo elapsed time: {:.2f}".format(fps.elapsed()))
    
    #print(bboxes)
    # load our serialized model from disk                                   
    mc = mtc.mot_class_arch1(bboxes, tracker, frame, resize_width)
    mc.tracking(args)
