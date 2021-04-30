import argparse
import cv2
import time

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result


def parse_args():
    parser = argparse.ArgumentParser(description='SemGEN and MMpose combined demo.')

    parser.add_argument('--cv-camera-index', type=int)
    parser.add_argument('--cv-video-path')
    parser.add_argument('--cv-show', action='store_true')

    parser.add_argument('--mmp-det-config', help='Config file for detection', required=True)
    parser.add_argument('--mmp-det-checkpoint', help='Checkpoint file for detection', required=True)
    parser.add_argument('--mmp-pose-config', help='Config file for pose', required=True)
    parser.add_argument('--mmp-pose-checkpoint', help='Checkpoint file for pose', required=True)
    parser.add_argument('--mmp-device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--mmp-bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--mmp-kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()
    return args

def main(args):
    print("Application Entry.")
    assert args.cv_camera_index is not None or args.cv_video_path is not None

    # initialize.
    print("Initialize.")
    cap = None
    if args.cv_camera_index is not None:
        cap = cv2.VideoCapture(args.cv_camera_index)
    else:
        cap = cv2.VideoCapture(args.cv_video_path)

    mmpose_driver = MMPoseDriver(args)
    semgcm_driver = SemGCMDriver()

    # main loop.
    print("Main Loop")
    frame_head = 0.0
    frame_tail = 0.0
    while (cap.isOpened()):
        last_elapsed = frame_tail - frame_head + 0.00001
        last_fps = 1.0 / last_elapsed
        frame_head = time.time()
        flag, img = cap.read()
        if not flag:
            break

        vis_img = mmpose_driver.update(img)
        semgcm_driver.update()

        if args.cv_show:
            cv2.putText(vis_img, "{} fps".format(int(last_fps)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('Image', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_tail = time.time()

    # Terminate.
    print("Exit Application.")
    cap.release()
    cv2.destroyAllWindows()



class MMPoseDriver:
    def __init__(self, args):
        print("Initialize MMPoseDriver - begin.")
        self.det_model = init_detector(args.mmp_det_config, args.mmp_det_checkpoint, device=args.mmp_device)
        self.pose_model = init_pose_model(args.mmp_pose_config, args.mmp_pose_checkpoint, device=args.mmp_device)
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.bbox_thr = args.mmp_bbox_thr
        self.kpt_thr = args.mmp_kpt_thr
        self.return_heatmap = False
        self.output_layer_names = None
        print("Initialize MMPoseDriver - end.")

    def update(self, img):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, img)
        # keep the person class bounding boxes.
        person_bboxes = self.process_mmdet_results(mmdet_results)
        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_bboxes,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            return_heatmap=self.return_heatmap,
            outputs=self.output_layer_names)

        # show the results
        vis_img = vis_pose_result(
            self.pose_model,
            img,
            pose_results,
            dataset=self.dataset,
            kpt_score_thr=self.kpt_thr,
            show=False)

        return vis_img

    def process_mmdet_results(self, mmdet_results, cat_id=0):
        """Process mmdet results, and return a list of bboxes.

        :param mmdet_results:
        :param cat_id: category id (default: 0 for human)
        :return: a list of detected bounding boxes
        """
        if isinstance(mmdet_results, tuple):
            det_results = mmdet_results[0]
        else:
            det_results = mmdet_results
        return det_results[cat_id]


class SemGCMDriver:
    def __init__(self):
        pass
    def update(self):
        pass



if __name__ == '__main__':
    main(parse_args())
