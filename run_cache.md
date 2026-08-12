## start
```bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam_calibrated.txt \
  --stride=1 \
  --disable_vis \
  --backend_thresh=28 \
  --backend_nms=2 \
  --reconstruction_path=outputs/my_map_06.pth

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --backend_thresh=28 \
  --backend_nms=2 \
  --reconstruction_path=outputs/my_map_06_1.pth

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --buffer=1024 \
  --filter_thresh=1.5 \
  --keyframe_thresh=3.0 \
  --frontend_window=32 \
  --frontend_thresh=18 \
  --frontend_nms=1 \
  --backend_thresh=24 \
  --backend_radius=3 \
  --backend_nms=2 \
  --reconstruction_path=outputs/my_map_06_3.pth

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --buffer=1024 \
  --filter_thresh=1.5 \
  --keyframe_thresh=3.0 \
  --frontend_window=40 \
  --frontend_thresh=20 \
  --frontend_nms=1 \
  --backend_thresh=30 \
  --backend_radius=3 \
  --backend_nms=1 \
  --reconstruction_path=outputs/my_map_06_4.pth

# best
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --buffer=1024 \
  --filter_thresh=2.0 \
  --keyframe_thresh=3.0 \
  --frontend_window=50 \
  --frontend_thresh=20 \
  --frontend_nms=1 \
  --backend_thresh=30 \
  --backend_radius=4 \
  --backend_nms=1 \
  --reconstruction_path=outputs/my_map_06_5.pth

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --buffer=1024 \
  --filter_thresh=1.2 \
  --keyframe_thresh=2.2 \
  --warmup=12 \
  --frontend_window=60 \
  --frontend_thresh=22 \
  --frontend_radius=4 \
  --frontend_nms=1 \
  --backend_thresh=32 \
  --backend_radius=5 \
  --backend_nms=1 \
  --reconstruction_path=outputs/my_map_06_6.pth

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --buffer=1536 \
  --filter_thresh=1.3 \
  --keyframe_thresh=2.4 \
  --warmup=14 \
  --frontend_window=60 \
  --frontend_thresh=22 \
  --frontend_radius=4 \
  --frontend_nms=1 \
  --backend_thresh=30 \
  --backend_radius=5 \
  --backend_nms=1 \
  --beta=0.4 \
  --reconstruction_path=outputs/my_map_06_8.pth

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python demo.py \
  --imagedir=data/my_map_06/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --buffer=1536 \
  --filter_thresh=1.5 \
  --keyframe_thresh=2.6 \
  --warmup=14 \
  --frontend_window=55 \
  --frontend_thresh=20 \
  --frontend_radius=4 \
  --frontend_nms=1 \
  --backend_thresh=28 \
  --backend_radius=5 \
  --backend_nms=1 \
  --beta=0.45 \
  --reconstruction_path=outputs/my_map_06_9.pth

```