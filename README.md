# DROID-SLAM


<!-- <center><img src="misc/DROID.png" width="640" style="center"></center> -->


[![IMAGE ALT TEXT HERE](misc/screenshot.png)](https://www.youtube.com/watch?v=GG78CSlSHSA)



[DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869)  
Zachary Teed and Jia Deng

```
@article{teed2021droid,
  title={{DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras}},
  author={Teed, Zachary and Deng, Jia},
  journal={Advances in neural information processing systems},
  year={2021}
}
```


## Requirements

To run the code you will need ...
* **Inference:** Running the demos will require a GPU with at least 11G of memory. 

* **Training:** Training requires a GPU with at least 24G of memory. We train on 4 x RTX-3090 GPUs.

## Getting Started
Clone the repo using the `--recursive` flag
```Bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
```

  If you forgot `--recursive`
  ```Bash
  git submodule update --init --recursive .
  ```

### Installing

Requires CUDA to be installed on your machine. If you run into issues, make sure the PyTorch and CUDA major versions match with the following check (minor version mismatch should be fine).

```Bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

```Bash
python3 -m venv .venv
source .venv/bin/activate

# install requirements (tested up to torch 2.7)
pip install -r requirements.txt

# optional (for visualization)
pip install moderngl moderngl-window

# install third-party modules (this will take a while)
pip install thirdparty/lietorch
pip install thirdparty/pytorch_scatter

# install droid-backends
pip install -e .
```

<!-- ### Deprecated Conda Installation

1. Creating a new anaconda environment using the provided .yaml file. Use `environment_novis.yaml` to if you do not want to use the visualization
```Bash
conda env create -f environment.yaml
pip install evo --upgrade --no-binary evo
pip install gdown
```

2. Compile the extensions (takes about 10 minutes)
```Bash
python setup.py install
``` -->


## Demos

1. Download the model from google drive: [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing) or with
    ```Bash
    ./tools/download_model.sh
    ```

2. Download some sample videos using the provided script.
    ```Bash
    ./tools/download_sample_data.sh
    ```

Run the demo on any of the samples (all demos can be run on a GPU with 11G of memory). To save the reconstruction with full resolution depth maps use the `--reconstruction_path` flag. If you ran with `--reconstruction_path my_reconstruction.pth`, you can view the reconstruction in high resolution by running
```Bash
python view_reconstruction.py my_reconstruction.pth
```

**Asynchronous and Multi-GPU Inference:** You can run the demos in asynchronous mode by running with `--asynchronous`. In this setting, the frontend and backend will run in seperate Python processes. You can additionally enable multi-GPU inference by setting the devices of the frontend and backend processes with the following arguments. 

**Visualization currently doesn't work multi-gpu setting**. You will need to run with ``--disable_vis``.


```Bash
python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt
```

```Bash
python demo.py --imagedir=data/mav0/cam0/data --calib=calib/euroc.txt --t0=150
```

```Bash
python demo.py --imagedir=data/rgbd_dataset_freiburg3_cabinet/rgb --calib=calib/tum3.txt
```


**Running on your own data:** All you need is a calibration file. Calibration files are in the form 
```
fx fy cx cy [k1 k2 p1 p2 [ k3 [ k4 k5 k6 ]]]
```
with parameters in brackets optional.

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, TUM, and ETH3D-SLAM. EuRoC and TUM can be run on a 1080Ti. The TartanAir and ETH3D-SLAM datasets will require 24G of memory. 

**Asynchronous and Multi-GPU Inference:** You can run evaluation in asynchronous mode by running with `--asynchronous`. In this setting, the frontend and backend will run in seperate Python processes. You can additionally enable multi-GPU inference by setting the devices of the frontend and backend processes with the following arguments. For example:
```
python evaluation_scripts/test_tartanair.py \
  --datapath data/tartanair_test/mono \
  --gt_path data/tartanair_test/mono_gt \
  --frontend_device cuda:0 \
  --backend_device cuda:1 \
  --asynchronous \
  --disable_vis
```


**Note:** Running with `--asynchronous` will typically produce better results, but this mode is not deterministic.

### TartanAir (Mono + Stereo)

Download the [TartanAir](https://theairlab.org/tartanair-dataset/) test set with this command.

```Bash
./tools/download_tartanair_test.sh
```

Or from these links: [Images](https://drive.google.com/file/d/1N8qoU-oEjRKdaKSrHPWA-xsnRtofR_jJ/view), [Groundtruth](https://cmu.box.com/shared/static/3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip) 



**Monocular evaluation:**
```bash
python evaluation_scripts/test_tartanair.py \
  --datapath datasets/tartanair_test/mono \
  --gt_path datasets/tartanair_test/mono_gt \
  --disable_vis
```

**Stereo evaluation:**
```bash
python evaluation_scripts/test_tartanair.py \
  --datapath datasets/tartanair_test/stereo \
  --gt_path datasets/tartanair_test/stereo_gt \
  --stereo --disable_vis
```


**Evaluating on the validation split:**

Download the [TartanAir](https://theairlab.org/tartanair-dataset/) dataset using the script `thirdparty/tartanair_tools/download_training.py` and put them in `datasets/TartanAir`
```Bash
# monocular eval
./tools/validate_tartanair.sh --plot_curve

# stereo eval
./tools/validate_tartanair.sh --plot_curve  --stereo
```

### EuRoC (Mono + Stereo)
Download the [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) sequences (ASL format):
```Bash
./tools/download_euroc.sh
```

Then run evaluation:
```Bash
# monocular eval (single gpu)
./tools/evaluate_euroc.sh

# monocular eval (multi gpu)
./tools/evaluate_euroc.sh --asynchronous --frontend_device cuda:0 --backend_device cuda:1

# stereo eval (single gpu)
./tools/evaluate_euroc.sh --stereo

# stereo eval (multi gpu)
./tools/evaluate_euroc.sh --stereo --asynchronous --frontend_device cuda:0 --backend_device cuda:1
```

### TUM-RGBD (Mono)
Download the [TUM-RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) sequences:
```
./tools/download_tum.sh
```
Then run evaluation:
```Bash
# monocular eval (single gpu)
./tools/evaluate_tum.sh

# monocular eval (multi gpu)
./tools/evaluate_tum.sh --asynchronous --frontend_device cuda:0 --backend_device cuda:1
```

### ETH3D (RGB-D)
Download the [ETH3D](https://www.eth3d.net/slam_datasets) dataset:
```Bash
./tools/download_eth3d.sh
```

```Bash
# RGB-D eval (single gpu)
./tools/evaluate_eth3d.sh > eth3d_results.txt
python evaluation_scripts/parse_results.py eth3d_results.txt

# RGB-D eval (multi gpu)
./tools/evaluate_eth3d.sh --asynchronous --frontend_device cuda:0 --backend_device cuda:1 > eth3d_results_async.txt
python evaluation_scripts/parse_results.py eth3d_results_async.txt
```

## Training

First download the TartanAir dataset. The download script can be found in `thirdparty/tartanair_tools/download_training.py`. You will only need the `rgb` and `depth` data.

```
python download_training.py --rgb --depth
```

You can then run the training script. We use 4x3090 RTX GPUs for training which takes approximatly 1 week. If you use a different number of GPUs, adjust the learning rate accordingly.

**Note:** On the first training run, covisibility is computed between all pairs of frames. This can take several hours, but the results are cached so that future training runs will start immediately. 


```
python train.py --datapath=<path to tartanair> --gpus=4 --lr=0.00025
```


## Localization in a saved map

`localize.py` estimates the camera pose for each frame of another video in the
coordinate system of an existing `--reconstruction_path` map. It uses SIFT map
landmarks, PnP/RANSAC, optical-flow tracking and periodic global matching. The
localizer runs on CPU and does not import DROID's CUDA extensions or model weights.

Install the video decoder in the existing environment:

```bash
source .venv/bin/activate
python -m pip install 'av>=14'
```

Example using the calibrated map in this workspace:

```bash
python localize.py \
  --map outputs/my_map_04_2.pth \
  --video input/20260810_162724.mp4 \
  --calib calib/sasung_cam_calibrated.txt \
  --output outputs/localization_my_map_04
```

For a second recording, replace `--video` and select a new output directory. Use
`--max-frames 90` for a short check, or `--start 10 --end 15` for an interval.
Frame indices and times still refer to the original video. Existing results are
protected; replacing a run requires `--overwrite`.

Outputs:

* `trajectory.csv`: one row per processed frame. `frame_index` starts at zero;
  `timestamp_sec` is presentation time relative to the first decoded frame.
  Original `pts`, `time_base` and `timestamp_source` are also retained.
* `x,y,z`: camera optical center in map coordinates. Monocular map units are
  arbitrary, not automatically metres. `--scale S` scales exported positions;
  determine S from an independent known distance if metres are required.
* `qx,qy,qz,qw`: camera-to-world orientation. Camera axes are right, down, forward.
* `status`: `localized`, `relocalized` after recovery, or `lost`. Lost frames have
  `nan` coordinates and orientation. No missing poses are silently interpolated.
* Quality columns: `inliers`, `matches`, `inlier_ratio`, median
  `reprojection_error_px`, image `coverage`, `reference_ids`, `method`, and `reason`.
* `metadata.json`: map/input hashes, calibration, conventions and run settings.
* `report.json`: coverage, lost intervals, reprojection errors, processing speed
  and completion/error state. Interrupted/failed runs retain their partial CSV.

Video decoding preserves variable frame-rate PTS. Missing or decreasing timestamps
are errors by default; `--allow-fps-fallback` explicitly permits labelled approximate
times when PTS is missing. Decode errors stop the run and mark it incomplete.

Calibration must correspond to the camera/lens, crop and zoom used for the new
recording. Parameters refer to the displayed image orientation; quarter-turn
display metadata is applied before calibration. `--rotation 0` ignores this
rotation, and `--rotation 90/180/270` overrides it counterclockwise. Mirrored
display transforms are rejected unless explicitly overridden. If only resolution
changed, pass `--calib-size WIDTH HEIGHT` for the calibration's original resolution.
The preprocessor corrects distortion, resizes using the demo's image-area rule,
crops the bottom/right to multiples of eight, and updates intrinsics accordingly.

The DROID loader reads full-resolution `disps` as inverse depth, multiplies stored
intrinsics by eight, and inverts world-to-camera poses when lifting landmarks.
SIFT features are cached in `.cache/localization`, keyed by map content, extractor
configuration and OpenCV version. `--no-cache` disables this cache.

`--independent` performs global matching on every frame, useful for evaluating
localization without tracking. The default tracks 2D observations of fixed 3D
landmarks, refreshes matches every 10 frames and globally checks every 60 frames.
Tracking failures trigger global search. Geometrically incompatible place
hypotheses with comparable support are rejected. Thresholds in `--help` are
starting values, not guarantees of physical accuracy; repetitive/changed scenes
can still produce incorrect poses.

Single-image check and frame diagnostics:

```bash
python localize.py --map outputs/my_map_04_2.pth \
  --image data/my_map_04/frames/000100.jpg \
  --calib calib/sasung_cam_calibrated.txt \
  --output outputs/localization_image --diagnostic-every 1
```

`--preview` opens a frame window; `--diagnostic-every 150` saves annotated JPEGs
without a GUI. Green points are geometric inliers and red segments show residuals
between observed and projected points.

View the original map (blue trajectory) and the localized video (red trajectory):

```bash
python view_localization.py \
  --trajectory outputs/localization_my_map_04/trajectory.csv --show-video
```

Use N/P to move between rows, Space to play/pause, and the mouse to inspect the
cloud. `--frame 300` chooses an initial original video frame; `--step 10` changes
the N/P increment. Lost intervals remain gaps. Both viewers use the **same**
reconstruction builder, CUDA multi-view depth filter, reference-camera frustums
and render settings. This viewer requires CUDA like `view_reconstruction.py`;
the localization calculation itself still runs on CPU. Defaults are
`--cloud-stride 2 --filter-threshold 0.005 --filter-count 3`, matching the original
viewer. For geometry export without a graphics window (CUDA is still required):

```bash
python view_localization.py \
  --trajectory outputs/localization_my_map_04/trajectory.csv \
  --no-window --export outputs/localization_my_map_04/geometry
```

To render and save a view for inspection, use `--screenshot result.png --frame 740`.
This renders through Open3D in a hidden window and exits; a working display/OpenGL
context is required. Old PLY exports made with the simplified point-cloud filter
should be regenerated with the command above.

### Localization checks

```bash
python -m unittest discover -s tests_localization -v
```

Tests cover DROID geometry conventions, PnP with outliers, degenerate/ambiguous
matches, optical flow, loss/reacquisition, calibration preprocessing, variable
video timestamps, CSV output and gaps in the displayed trajectory.

For a same-session held-out check, replay the **same FPS filter used for mapping**
and select only source frames whose decoded image was not selected by that filter:

```bash
python evaluation_scripts/validate_localization.py \
  --map outputs/my_map_04_2.pth \
  --video input/20260810_162724.mp4 \
  --calib calib/sasung_cam_calibrated.txt \
  --map-fps 10 --samples 30 \
  --output outputs/localization_my_map_04_heldout
```

This matches the mapping recipe `ffmpeg -vf fps=10` with default rounding and no
time trimming. It compares decoded YUV hashes before/after the filter, excluding
all selected mapping images, not only saved DROID keyframes. Each sampled query
is localized independently. The report records the selected frame indices and
annotated images. It does not measure cross-session robustness or metric pose
accuracy; those require another recording and independent reference measurements.

## Acknowledgements
Data from [TartanAir](https://theairlab.org/tartanair-dataset/) was used to train our model. We additionally use evaluation tools from [evo](https://github.com/MichaelGrupp/evo) and [tartanair_tools](https://github.com/castacks/tartanair_tools).
