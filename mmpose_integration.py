import argparse
import cv2
import time
import torch
import os.path as path
import numpy as np


from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result


def parse_args():
    parser = argparse.ArgumentParser(description='SemGEN and MMpose combined demo.')

    parser.add_argument('--cv-camera-index', type=int)
    parser.add_argument('--cv-video-path', type=str)
    parser.add_argument('--cv-show', action='store_true')

    parser.add_argument('--mmp-det-config', help='Config file for detection', required=True)
    parser.add_argument('--mmp-det-checkpoint', help='Checkpoint file for detection', required=True)
    parser.add_argument('--mmp-pose-config', help='Config file for pose', required=True)
    parser.add_argument('--mmp-pose-checkpoint', help='Checkpoint file for pose', required=True)
    parser.add_argument('--mmp-device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--mmp-bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--mmp-kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--mmp-show', action='store_true')

    parser.add_argument('--sem-dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--sem-num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('--sem-hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('--sem-evaluate', default='', type=str, metavar='FILENAME', required=True, help='checkpoint to evaluate (file name)')
    parser.add_argument('--sem-show', action='store_true')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--render-score-threshold', type=float, default=0.5)

    args = parser.parse_args()
    return args


def main(args):
    app = Application(args)
    app.update()
    app.term()


class Application:
    def __init__(self, args):
        print("Application Entry.")
        self.args = args
        assert args.cv_camera_index is not None or args.cv_video_path is not None

        # initialize.
        print("Initialize.")
        self.cap = None
        if args.cv_camera_index is not None:
            self.cap = cv2.VideoCapture(args.cv_camera_index)
        else:
            self.cap = cv2.VideoCapture(args.cv_video_path)

        self.device_driver = DeviceDriver(args)
        self.mmpose_driver = MMPoseDriver(args, self.device_driver)
        self.semgcm_driver = SemGCMDriver(args, self.device_driver)

    def update(self):
        # main loop.
        print("Main Loop")
        frame_head = 0.0
        frame_tail = 0.0
        count = 0
        while (self.cap.isOpened()):
            count = count + 1
            last_elapsed = frame_tail - frame_head + 0.00001
            last_fps = 1.0 / last_elapsed
            frame_head = time.time()
            flag, img = self.cap.read()
            if not flag:
                break

            self.mmpose_driver.update(img)
            self.semgcm_driver.update(self.mmpose_driver)

            if self.args.cv_show:
                drawimg = img
                drawimg = self.mmpose_driver.render(drawimg) if self.args.mmp_show else drawimg
                drawimg = self.semgcm_driver.render(drawimg) if self.args.sem_show else drawimg
                cv2.putText(drawimg, "frame{} : {} fps".format(count, int(last_fps)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('Image', drawimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_tail = time.time()

    def term(self):
        print("Exit Application.")
        self.cap.release()
        cv2.destroyAllWindows()




class DeviceDriver:
    def __init__(self, args):
        print("Initialize DeviceDriver - begin.")
        self.name = args.device if torch.cuda.is_available() else "cpu"
        print("Initialize DeviceDriver - device {}".format(self.name))
        self.device = torch.device(self.name)
        print("Initialize DeviceDriver - end.")


class MMPoseDriver:
    def __init__(self, args, device):
        print("Initialize MMPoseDriver - begin.")
        self.det_model = init_detector(args.mmp_det_config, args.mmp_det_checkpoint, device=device.name)
        self.pose_model = init_pose_model(args.mmp_pose_config, args.mmp_pose_checkpoint, device=device.name)
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.bbox_thr = args.mmp_bbox_thr
        self.kpt_thr = args.mmp_kpt_thr
        self.return_heatmap = False
        self.output_layer_names = None
        self.last_pose_results = None
        self.last_returned_outputs = None
        self.last_converted_results = None
        self.last_scores = None
        self.coco_to_sem = [[12,13],[13],[15],[17],[12],[14],[16],[12,13],[6,7],[1],[7],[9],[11],[6],[8],[10]]
        self.device = device
        print("Initialize MMPoseDriver - end.")

    def update(self, img):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, img)
        # keep the person class bounding boxes.
        person_bboxes = self.process_mmdet_results(mmdet_results)
        # test a single image, with a list of bboxes.
        self.last_pose_results, self.last_returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_bboxes,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            return_heatmap=self.return_heatmap,
            outputs=self.output_layer_names)

    def render(self, img):
        vis_img = vis_pose_result(
            self.pose_model,
            img,
            self.last_pose_results,
            dataset=self.dataset,
            kpt_score_thr=self.kpt_thr,
            show=False)
        return vis_img
        

    def getLastPoseResult(self):
        self.last_converted_results = None
        self.last_scores = None
        if self.last_pose_results is None:
            return
        population = len(self.last_pose_results)
        self.last_converted_results = []
        self.last_scores = []
        for index in range(population):
            bbox = self.last_pose_results[index]['bbox']
            cx = (bbox[0][2] + bbox[0][0]) / 2
            cy = (bbox[0][3] + bbox[0][1]) / 2
            width = bbox[0][2] - bbox[0][0] / 2
            height = bbox[0][3] - bbox[0][1] / 2    
            keypoint = self.last_pose_results[index]['keypoints']
            result = []
            scores = []
            for cl in self.coco_to_sem:
                point = [0.0, 0.0]
                score = 0
                for c in cl:
                    point[0] = point[0] + keypoint[c][0]
                    point[1] = point[1] + keypoint[c][1]
                    score = score + keypoint[c][2]
                cn = len(cl)
                point[0] = ((point[0] / cn) - cx) / width
                point[1] = ((point[1] / cn) - cy) / height
                score = score / cn
                result.append(point)
                scores.append(score)
            result = torch.tensor(result).float().to(self.device.device)
            self.last_converted_results.append(result)
            self.last_scores.append(scores)
        return self.last_converted_results, self.last_scores


    def process_mmdet_results(self, mmdet_results, cat_id=0):
        if isinstance(mmdet_results, tuple):
            det_results = mmdet_results[0]
        else:
            det_results = mmdet_results
        return det_results[cat_id]


class SemGCMDriver:
    def __init__(self, args, device):
        print("Initialize SemGCMDriver - begin.")
        from models.sem_gcn import SemGCN
        from common.graph_utils import adj_mx_from_skeleton
        from common.skeleton import Skeleton
        self.hid_dim = args.sem_hid_dim
        self.num_layers = args.sem_num_layers
        self.p_dropout = (None if args.sem_dropout == 0.0 else args.sem_dropout)
        self.render_score_threshold = args.render_score_threshold
        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14 ],
            joints_left=[4, 5, 6, 10, 11, 12],
            joints_right=[1, 2, 3, 13, 14, 15]
        )
        self.skeleton._joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]
        adj = adj_mx_from_skeleton(self.skeleton)
        self.device = device
        self.model_pos = SemGCN(
            adj, 
            self.hid_dim, 
            num_layers=self.num_layers, 
            p_dropout=self.p_dropout,
            nodes_group=self.skeleton.joints_group()
        ).to(self.device.device)
        self.last_3d_positions = None
        self.last_scores = None

        # Resume from a checkpoint
        ckpt_path = args.sem_evaluate
        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            self.ckpt = torch.load(ckpt_path)
            self.rename_nonlocal_node(self.ckpt)
            start_epoch = self.ckpt['epoch']
            error_best = self.ckpt['error']
            self.model_pos.load_state_dict(self.ckpt['state_dict'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
        print("Initialize SemGCMDriver - end.")

    def update(self, mmpose):
        lastPoseResult, scores = mmpose.getLastPoseResult()
        self.last_3d_positions = None
        self.last_scores = scores
        if lastPoseResult is None:
            return
        self.last_3d_positions = []
        for input2d in lastPoseResult:
            result = self.model_pos(input2d).cpu()
            self.last_3d_positions.append(result)


    def render(self, img):
        if self.last_3d_positions is not None:
            for index, position in enumerate(self.last_3d_positions):
                for node, parent in enumerate(self.skeleton._parents):
                    if parent < 0:
                        continue
                    color = (255,0,0)
                    if self.last_scores[index][node] > self.render_score_threshold and self.last_scores[index][parent] > self.render_score_threshold:
                        color = (255,255,0)
                    nx = position[0][node][0] * 100 + 100
                    ny = position[0][node][1] * 100 + 100
                    px = position[0][parent][0] * 100 + 100
                    py = position[0][parent][1] * 100 + 100
                    cv2.line(img,(nx,ny),(px,py),color,1)
        return img


    def rename_nonlocal_node(self, dic):
        if 'items' not in dir(dic):
            return
        targets = []
        for key, value in dic.items():
            if type(key) == str:
                if '.nonlocal.' in key:
                    targets.append(key)
            self.rename_nonlocal_node(value)
        for key in targets:
            value = dic.pop(key)
            key = key.replace('.nonlocal.','._nonlocal.')
            dic[key] = value


if __name__ == '__main__':
    main(parse_args())
