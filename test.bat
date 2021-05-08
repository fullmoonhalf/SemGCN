SET DEMOPY=mmpose_integration.py

REM
REM CV
REM
SET OPTCV=--cv-camera-index 0 --cv-show
rem SET OPTCV=--cv-video-path ../mmpose/video/test3_Trim.mp4 --cv-show
rem SET OPTCV=--cv-video-path ../mmpose/video/test.mp4 --cv-show

REM
REM MMposeq
REM
SET MMDET_CFG=../mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py
SET MMDET_PTH=http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
SET MMPOSE_CFG=../mmpose/configs/top_down/resnet/coco-wholebody/res152_coco_wholebody_256x192.py
SET MMPOSE_PTH=../mmpose/checkpoints/res152_coco_wholebody_256x192-5de8ae23_20201004.pth
SET OPTMP=--mmp-det-config %MMDET_CFG% --mmp-det-checkpoint %MMDET_PTH% --mmp-pose-config %MMPOSE_CFG% --mmp-pose-checkpoint %MMPOSE_PTH%

REM
REM SemGCN
REM
SET SEM_EVALUATE=checkpoint/pretrained/ckpt_semgcn_nonlocal.pth.tar 
SET OPTSEM=--sem-evaluate %SEM_EVALUATE%

SET VIZOPT=--sem-show-3d --mmp-show-2d --sem-plot


REM
REM Execution
REM
python %DEMOPY% %OPTCV% %OPTMP% %OPTSEM% %VIZOPT%


