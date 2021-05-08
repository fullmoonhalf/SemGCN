import argparse
import cv2
import time
import torch
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import sys

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
    parser.add_argument('--mmp-show-mmp', action='store_true')
    parser.add_argument('--mmp-show-2d', action='store_true')

    parser.add_argument('--sem-dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--sem-num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('--sem-hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('--sem-evaluate', default='', type=str, metavar='FILENAME', required=True, help='checkpoint to evaluate (file name)')
    parser.add_argument('--sem-show-3d', action='store_true')
    parser.add_argument('--sem-plot', action='store_true')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--render-score-threshold', type=float, default=0.5)
    parser.add_argument('--privacy', type=int)

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

        # setup skeleton
        from common.skeleton import Skeleton
        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14 ],
            joints_left=[4, 5, 6, 10, 11, 12],
            joints_right=[1, 2, 3, 13, 14, 15]
        )
        self.skeleton._joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]

        # setup drivers
        self.device_driver = DeviceDriver(args)
        self.mmpose_driver = MMPoseDriver(args, self.device_driver, self.skeleton)
        self.semgcm_driver = SemGCMDriver(args, self.device_driver, self.skeleton)

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

            self.semgcm_driver.plotting()

            if self.args.cv_show:
                drawimg = img
                drawimg = self.mmpose_driver.render(drawimg)
                drawimg = self.semgcm_driver.render(drawimg)
                cv2.putText(drawimg, "frame{} : {} fps".format(count, int(last_fps)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('Image', drawimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_tail = time.time()

    def term(self):
        print("Exit Application.")
        self.semgcm_driver.term()
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
    def __init__(self, args, device, skeleton):
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
        self.last_raw_results = None
        self.last_scores = None
        self.coco_to_sem = [
            [11,12],        # sem 00: 'Hip'
            [12],           # sem 01: 'RHip' 
            [14],           # sem 02: 'RKnee' 
            [16],           # sem 03: 'RFoot' 
            [11],           # sem 04: 'LHip' 
            [13],           # sem 05: 'LKnee' 
            [15],           # sem 06: 'LFoot' 
            [5,6,11,12],    # sem 07: 'Spine' 
            [5,6],          # sem 08: 'Thorax' 
            [0],            # sem 09: 'Head' 
            [5],            # sem 10: 'LShoulder' 
            [7],            # sem 11: 'LElbow' 
            [9],           # sem 12: 'LWrist' 
            [6],            # sem 13: 'RShoulder' 
            [8],            # sem 14: 'RElbow' 
            [10]             # sem 15: 'RWrist' 
        ]
        self.device = device
        self.render_mmp = args.mmp_show_mmp
        self.render_2d = args.mmp_show_2d
        self.skeleton = skeleton
        self.render_score_threshold = args.render_score_threshold
        self.privacy = args.privacy
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

        population = len(self.last_pose_results)
        self.last_raw_results = []
        self.last_converted_results = []
        self.last_scores = []
        for index in range(population):
            keypoint = self.last_pose_results[index]['keypoints']
            rawret = []
            scores = []
            minpos = [sys.float_info.max, sys.float_info.max]
            maxpos = [-sys.float_info.max, -sys.float_info.max]
            for cl in self.coco_to_sem:
                rawpt = [0.0, 0.0]
                score = 0
                for c in cl:
                    rawpt[0] = rawpt[0] + keypoint[c][0]
                    rawpt[1] = rawpt[1] + keypoint[c][1]
                    score = score + keypoint[c][2]
                cn = len(cl)
                rawpt[0] = rawpt[0] / cn
                rawpt[1] = rawpt[1] / cn
                score = score / cn
                rawret.append(rawpt)
                scores.append(score)
                minpos[0] = min(rawpt[0], minpos[0])
                minpos[1] = min(rawpt[1], minpos[1])
                maxpos[0] = max(rawpt[0], maxpos[0])
                maxpos[1] = max(rawpt[1], maxpos[1])
            
            cx = (maxpos[0] + minpos[0]) / 2
            cy = (maxpos[1] + minpos[1]) / 2
            width = (maxpos[0] - minpos[0]) / 2
            height = (maxpos[1] - minpos[1]) / 2
            scale = max(width, height)
            result = []
            for rawpt in rawret:
                point = [0.0, 0.0]
                point[0] = (rawpt[0] - cx) / scale
                point[1] = (rawpt[1] - cy) / scale
                result.append(point)

            result = torch.tensor(result).float().to(self.device.device)
            self.last_raw_results.append(rawret)
            self.last_converted_results.append(result)
            self.last_scores.append(scores)

    def render(self, img):
        img = self.__render_mmp(img)
        img = self.__render_2d(img)
        return img

    def __render_mmp(self, img):
        if self.render_mmp and self.last_pose_results:
            img = vis_pose_result(self.pose_model, img, self.last_pose_results, dataset=self.dataset, kpt_score_thr=self.kpt_thr, show=False)
            for result in self.last_pose_results:
                kp = result['keypoints']
                for i, p in enumerate(kp):
                    if i > 16:
                        break
                    x = int(p[0])
                    y = int(p[1])
                    cv2.putText(img, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
        return img

    def __render_2d(self, img):
        if self.render_2d and self.last_raw_results:
            for index, position in enumerate(self.last_raw_results):
                for node, parent in enumerate(self.skeleton._parents):
                    if parent < 0:
                        continue
                    color = (255,0,0)
                    trustable = False
                    if self.last_scores[index][node] > self.render_score_threshold and self.last_scores[index][parent] > self.render_score_threshold:
                        color = (255,255,0)
                        trustable = True
                    nx = int(position[node][0])
                    ny = int(position[node][1])
                    px = int(position[parent][0])
                    py = int(position[parent][1])
                    if node == 9 and self.privacy:
                        x = int(position[node][0])
                        y = int(position[node][1])
                        img = cv2.circle(img,(x,y), self.privacy, (190,190,190), -1)
                    cv2.line(img,(nx,ny),(px,py),color,1)
                    if trustable:
                        cv2.putText(img, "{}".format(node), (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA)
        return img

    def getLastPoseResult(self):
        return self.last_converted_results, self.last_scores

    def process_mmdet_results(self, mmdet_results, cat_id=0):
        if isinstance(mmdet_results, tuple):
            det_results = mmdet_results[0]
        else:
            det_results = mmdet_results
        return det_results[cat_id]


class SemGCMDriver:
    def __init__(self, args, device, skeleton):
        print("Initialize SemGCMDriver - begin.")
        from models.sem_gcn import SemGCN
        from common.graph_utils import adj_mx_from_skeleton
        self.hid_dim = args.sem_hid_dim
        self.num_layers = args.sem_num_layers
        self.p_dropout = (None if args.sem_dropout == 0.0 else args.sem_dropout)
        self.render_score_threshold = args.render_score_threshold
        self.skeleton = skeleton
        adj = adj_mx_from_skeleton(self.skeleton)
        self.device = device
        self.model_pos = SemGCN(
            adj, 
            self.hid_dim, 
            num_layers=self.num_layers, 
            p_dropout=self.p_dropout,
            nodes_group=self.skeleton.joints_group()
        ).to(self.device.device)
        self.last_2d_positions = None
        self.last_3d_positions = None
        self.last_scores = None
        self.render_3d = args.sem_show_3d
        self._plot = args.sem_plot
        self._plot_initalized = False
        self._plot_skeleton = None
        self.last_3d_skeletons = None

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

        if self._plot:
            size = 5
            radius = 1
            azim = 45
            title = "test"
            plt.ion()
            fig = plt.figure(figsize=(size, size))
            self._plot_ax = fig.add_subplot(1, 1 , 1, projection='3d')
            self._plot_ax.view_init(elev=15., azim=azim)
            self._plot_ax.set_xlim3d([-radius / 2, radius / 2])
            self._plot_ax.set_zlim3d([0, radius])
            self._plot_ax.set_ylim3d([-radius / 2, radius / 2])
            self._plot_ax.set_aspect('auto')
            self._plot_ax.set_xticklabels([])
            self._plot_ax.set_yticklabels([])
            self._plot_ax.set_zticklabels([])
            self._plot_ax.dist = 7.5
            self._plot_ax.set_title(title)  # , pad=35

        print("Initialize SemGCMDriver - end.")

    def update(self, mmpose):
        self.last_2d_positions, self.last_scores = mmpose.getLastPoseResult()
        self.last_3d_positions = None
        if self.last_2d_positions is None:
            return
        self.last_3d_positions = []
        for input2d in self.last_2d_positions:
            result = self.model_pos(input2d).cpu()
            self.last_3d_positions.append(result)

        self.last_3d_skeletons = []
        for index, position in enumerate(self.last_3d_positions):
            skeleton = []
            for node, parent in enumerate(self.skeleton._parents):
                if parent < 0:
                    continue
                n = [position[0][node][0], position[0][node][1], position[0][node][2]]
                p = [position[0][parent][0], position[0][parent][1], position[0][parent][2]]
                line = [n, p, [self.last_scores[index][node], self.last_scores[index][parent]]]
                skeleton.append(line)
            self.last_3d_skeletons.append(skeleton)

    def render(self, img):
        img = self.__render_3d(img)
        return img

    def plotting(self):
        if not self._plot:
            return
        if not self.last_3d_skeletons:
            return

        if not self._plot_skeleton:
            for skeleton in self.last_3d_skeletons:
                self._plot_skeleton = []
                for line in skeleton:
                    clr = '#a0a0a0'
                    if line[2][0] > self.render_score_threshold and line[2][1] > self.render_score_threshold:
                        clr = 'black'
                    self._plot_skeleton.append( self._plot_ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], zdir='z', color=clr))
                break
        else:
            for si, skeleton in enumerate(self.last_3d_skeletons):
                for li, line in enumerate(skeleton):
                    clr = '#a0a0a0'
                    if line[2][0] > self.render_score_threshold and line[2][1] > self.render_score_threshold:
                        clr = 'black'
                    self._plot_skeleton[li][0].set_xdata(np.array([line[0][0], line[1][0]]))
                    self._plot_skeleton[li][0].set_ydata(np.array([line[0][2], line[1][2]]))
                    self._plot_skeleton[li][0].set_3d_properties([line[0][1], line[1][1]], zdir='z')
                    self._plot_skeleton[li][0].set_color(clr)
                break

        plt.draw()
        plt.pause(0.01)

    def __render_3d(self, img):
        if self.render_3d and self.last_3d_skeletons:
            for skeleton in self.last_3d_skeletons:
                for line in skeleton:
                    color = (255,0,0)
                    if line[2][0] > self.render_score_threshold and line[2][1] > self.render_score_threshold:
                        color = (255,255,0)
                    nx = line[0][0] * 100 + 100
                    ny = line[0][1] * 100 + 100
                    px = line[1][0] * 100 + 100
                    py = line[1][1] * 100 + 100
                    cv2.line(img,(nx,ny),(px,py),color,1)
                break
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

    def term(self):
        if self._plot:
            plt.ioff()


if __name__ == '__main__':
    main(parse_args())
