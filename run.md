## start demo
```bash
source .venv/bin/activate
# nout
export CUDA_HOME="$PWD/.cuda/cuda-12.6"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# PC
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt --buffer 128 --stride 10

python demo.py \
    --imagedir=data/sfm_bench/rgb \
    --calib=calib/eth.txt \
    --buffer 128 \
    --stride 3

nvtop
htop

# fix artefacts (run without GUI)
mkdir -p reconstructions
# на мощном ПК
CUDA_VISIBLE_DEVICES=1 python demo.py \
    --imagedir=data/sfm_bench/rgb \
    --calib=calib/eth.txt \
    --disable_vis \
    --reconstruction_path=reconstructions/sfm_bench.pth
# но ноуте послабее
python demo.py \
    --imagedir=data/sfm_bench/rgb \
    --calib=calib/eth.txt \
    --buffer 128 \
    --stride 3 \
    --disable_vis \
    --reconstruction_path=reconstructions/sfm_bench.pth
# если ошибка, увеличить буффер
python demo.py \
    --imagedir=data/sfm_bench/rgb \
    --calib=calib/eth.txt \
    --buffer 160 \
    --stride 3 \
    --disable_vis \
    --reconstruction_path=reconstructions/sfm_bench.pth
python view_reconstruction.py reconstructions/sfm_bench.pth
```

## operationg video
```bash
# create folder
mkdir -p data/my_map_01/frames
mkdir -p outputs

# test
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,r_frame_rate:stream_tags=rotate \
  -show_entries format=duration \
  -of default=noprint_wrappers=1 \
  ~/Yandex.Disk/20260810_140239.mp4

# video => img (10 fps)
ffmpeg -i ~/Yandex.Disk/20260810_140239.mp4 \
  -vf "fps=10" \
  -q:v 2 \
  -start_number 0 \
  data/my_map_01/frames/%06d.jpg

# video => img (original fps)
ffmpeg -i ~/Videos/map_flight.mp4 \
  -vf "fps=10" \
  -start_number 0 \
  data/my_map_01/frames/%06d.png

# quantity frames
find data/my_map_01/frames -maxdepth 1 -type f | wc -l

# test reconstruction
python demo.py --help | grep reconstruction_path
# answer: --reconstruction_path RECONSTRUCTION_PATH

# 10 fps => --stride=1
python demo.py \
  --imagedir=data/my_map_01/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --reconstruction_path=outputs/my_map_01.pth

# without GUI
python demo.py \
  --imagedir=data/my_map_01/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --reconstruction_path=outputs/my_map_01.pth

# without GUI (if memory out)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python demo.py \
  --imagedir=data/my_map_04/frames \
  --calib=calib/sasung_cam.txt \
  --stride=1 \
  --disable_vis \
  --reconstruction_path=outputs/my_map_04.pth

# without GUI update
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --imagedir=data/my_map_kostya/frames \
  --calib=calib/sasung_cam_calibrated.txt \
  --stride=1 \
  --disable_vis \
  --backend_thresh=28 \
  --backend_nms=2 \
  --reconstruction_path=outputs/my_map_kostya.pth

# visualisation
python view_reconstruction.py outputs/my_map_01.pth
```

## operationg video 222222222222222222222222
```bash
# video => img (10 fps)
ffmpeg -i input/20260810_180126.mp4 \
  -vf "fps=10" \
  -q:v 2 \
  -start_number 0 \
  data/my_map_05/frames/%06d.jpg

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --imagedir=data/my_map_05/frames \
  --calib=calib/sasung_cam_calibrated.txt \
  --stride=1 \
  --disable_vis \
  --backend_thresh=28 \
  --backend_nms=2 \
  --reconstruction_path=outputs/my_map_05_2.pth

# visualisation
python view_reconstruction.py outputs/my_map_05.pth
```