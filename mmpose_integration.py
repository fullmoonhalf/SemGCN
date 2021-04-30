import argparse
import cv2
import time



def parse_args():
    parser = argparse.ArgumentParser(description='SemGEN and MMpose combined demo.')

    parser.add_argument('--cv-camera-index', type=int)
    parser.add_argument('--cv-video-path')
    parser.add_argument('--cv-show', action='store_true')

    args = parser.parse_args()
    return args

def main(args):
    assert args.cv_camera_index is not None or args.cv_video_path is not None

    # initialize.
    cap = None
    if args.cv_camera_index is not None:
        cap = cv2.VideoCapture(args.cv_camera_index)
    else:
        cap = cv2.VideoCapture(args.cv_video_path)

    # main loop.
    frame_head = 0.0
    frame_tail = 0.0
    while (cap.isOpened()):
        last_elapsed = frame_tail - frame_head + 0.00001
        last_fps = 1.0 / last_elapsed
        frame_head = time.time()
        flag, img = cap.read()
        if not flag:
            break
        if args.cv_show:
            cv2.putText(img, "{} fps".format(int(last_fps)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_tail = time.time()




    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main(parse_args())
