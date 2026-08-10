## start demo
```bash
source .venv/bin/activate
export CUDA_HOME="$PWD/.cuda/cuda-12.6"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

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

