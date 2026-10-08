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
# jetson
source .venv/bin/activate
unset PYTHONPATH PYTHONHOME
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/home/test/droid-runtime/cuda-runtime-cu128"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ssh ni3nayka@MSI1.local

rsync -avzP /home/ni3nayka/Документы/GitHub/DROID-SLAM/input/ ni3nayka@MSI1.local:/home/ni3nayka/IRS-CAM-AI/DROID-SLAM/input/

rsync -avzP ni3nayka@MSI1.local:/home/ni3nayka/IRS-CAM-AI/DROID-SLAM/outputs/ /home/ni3nayka/Документы/GitHub/DROID-SLAM/outputs/

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
ffmpeg -i input/20261007_cam_image_rgb.mp4 \
  -vf "fps=30" \
  -q:v 2 \
  -start_number 0 \
  data/my_map_07/frames/%06d.jpg

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
# my_map_07: RealSense D435 RGB 640x480, factory calibration in demo.py text format.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python demo.py \
  --imagedir=data/my_map_07/frames \
  --calib=calib/realsense_d435_948122071094_color_640x480.txt \
  --stride=1 \
  --disable_vis \
  --reconstruction_path=outputs/my_map_07.pth

# Более удачная пересборка my_map_07 с настройками my_map_06_5:
# полная команда и сравнение находятся в draft.md, раздел "my_map_07 с настройками...".
.venv/bin/python view_reconstruction.py outputs/my_map_07_best.pth

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
python view_reconstruction.py outputs/my_map_06_5.pth
python view_reconstruction.py outputs/my_map_04_2.pth
```

## localisation
```bash
.venv/bin/python view_localization.py \
  --trajectory outputs/localization_my_map_04/trajectory.csv \
  --show-video
```

## localisation: 20261003_145134.mp4
```bash
# Локализация нового видео по существующей карте.
# Для повторной записи результата в ту же папку добавь --overwrite.
.venv/bin/python localize.py \
  --map outputs/my_map_04_2.pth \
  --video input/20261003_145134.mp4 \
  --calib calib/sasung_cam_calibrated.txt \
  --calib-size 1280 720 \
  --output outputs/localization_20261003_145134 \
  --diagnostic-every 30

# Просмотр новой траектории на карте с исходным видео.
.venv/bin/python view_localization.py \
  --trajectory outputs/localization_20261003_145134/trajectory.csv \
  --show-video
```

## realTime: видео в реальном темпе и камера

```bash
.venv/bin/python live_localize.py \
  --map outputs/my_map_04_2.pth \
  --video input/20261003_145134.mp4 \
  --calib calib/sasung_cam_calibrated.txt \
  --calib-size 1280 720 \
  --output outputs/live_20261003_145134 \
  --view --preview --overwrite


# Тест: кадры подаются с исходной частотой по PTS.
# Желтая камера движется по той же полной карте, что в view_reconstruction.
# Повторный запуск в ту же папку: добавь --overwrite.
.venv/bin/python live_localize.py \
  --map outputs/my_map_04_2.pth \
  --video input/20261003_145134.mp4 \
  --calib calib/sasung_cam_calibrated.txt \
  --calib-size 1280 720 \
  --output outputs/live_20261003_145134 \
  --view --preview

# Только вычисления и запись, без окон: убери --view --preview.
# Остановка: Ctrl-C, Esc в окне кадра или закрытие 3D-окна.

# Камера: вместо --video используй --camera 0 или --camera 'rtsp://HOST:PORT/PATH'.
# ВАЖНО: калибровка должна принадлежать подключенной камере при этом crop/zoom.
# --capture-size и --capture-fps — запрос параметров, зависящий от драйвера.
.venv/bin/python live_localize.py \
  --map outputs/my_map_04_2.pth \
  --camera 0 --capture-size 1280 720 --capture-fps 30 \
  --calib calib/sasung_cam_calibrated.txt --calib-size 1280 720 \
  --output outputs/live_camera --view --preview

# Просмотр сохраненного результата live-теста прежним просмотрщиком.
.venv/bin/python view_localization.py \
  --trajectory outputs/live_20261003_145134/trajectory.csv --show-video

# Все тесты, включая прежние режимы.
.venv/bin/python -m unittest discover -s tests_localization -v
```

Состояния: `INITIALIZING` → `TRACKING`; при потере — `LOST`, при отсутствии
свежего кадра — `STALE`. Первичная локализация требует времени на поиск по карте.
Координаты выдаются только для проверенных кадров, в единицах исходной карты.
Очередь захвата хранит один свежий кадр: пропуски при перегрузке считаются,
задержка не накапливается. Автоматическое восстановление после перекрытия обзора
работает через поиск по карте; после отключения источника запусти команду заново.

Результат: `trajectory.csv`, `poses.jsonl`, `latest_pose.json`,
`realtime_report.json`. Последний файл содержит задержки и счетчик пропусков.
При чтении `latest_pose.json` проверяй `valid` и срок
`valid_until_monotonic` через `time.monotonic()` на том же компьютере.
Задержка камеры до приема кадра компьютером в эти измерения не входит.

Проверено на `20261003_145134.mp4` и `my_map_04_2.pth` на текущем ноутбуке:

- 378/378 кадров обработаны, пропусков нет, подача по PTS — 29,7 кадра/с.
- Первые 9 кадров — начальный поиск; следующие 369 локализованы без потерь.
- Возраст кадра при получении позы: медиана 5,9 мс, p95 8,8 мс.
- Окна карты и видео работают параллельно; карта содержит 386 950 точек.
- При искусственном перекрытии обзора координаты становятся недействительными;
  после возвращения изображения локализация восстановилась за ~0,37 с.
- Офлайн-CSV повторного прогона совпал с прежним побайтно. Старый просмотрщик
  открывает live-CSV; проверка исходного видео на 3 отложенных кадрах — 3/3.

Основной отчет: `outputs/live_20261003_145134/realtime_report.json`.
Прогон с окнами: `outputs/live_20261003_145134_gui/`.
Проверка перекрытия: `outputs/live_occlusion_test/`.
Для этого standalone-прогона физическая USB/IP-камера не проверялась;
захват, отключение и зависание источника покрыты тестами с подмененным источником.
Проверка реальной D435 через ROS описана ниже.


```bash
.venv/bin/python live_localize.py \
  --map outputs/my_map_04_2.pth \
  --video input/20261003_145134.mp4 \
  --calib calib/sasung_cam_calibrated.txt \
  --calib-size 1280 720 \
  --output outputs/live_20261003_145134 \
  --view --preview --overwrite
```

## ROS 2 Jazzy: D435 и готовая карта

Полная инструкция, параметры и тесты: [ros/droid_slam/README.md](ros/droid_slam/README.md).
Команды ниже выполняются из корня этого репозитория.

```bash
# Один раз собрать пакет droid_slam.
source /opt/ros/jazzy/setup.bash
colcon --log-base ros/log build --base-paths ros/droid_slam \
  --build-base ros/build --install-base ros/install --symlink-install
```

```bash
# Терминал 1: существующий драйвер из IRS-DRIVERS.
# 948123023494 — ASIC serial; serial_no драйвера требует DEVICE serial ниже.
source /opt/ros/jazzy/setup.bash
source ../IRS-DRIVERS/install/setup.bash
ros2 launch realsense driver.launch.py model:=d435 serial_no:=948122071094
```

```bash
# Терминал 2: локализация + полная 3D-карта + текущий кадр.
source /opt/ros/jazzy/setup.bash
source ros/install/setup.bash
ros2 launch droid_slam localization.launch.py \
  map_path:="$PWD/outputs/my_map_04_2.pth" \
  image_topic:=/camera/color/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  save_calibration_path:="$PWD/outputs/d435_camera_info.yaml" \
  output_dir:="$PWD/outputs/ros_d435"
```

Калибровка автоматически приходит от самой камеры. Для явного файла добавьте
`calib_path:="$PWD/calib/realsense_d435_948122071094_color_640x480.yaml"`
при потоке 640×480. Заводской файл для 1280×720 также сохранён в `calib/`.
При повторной записи результата добавьте `overwrite:=true` или смените каталог.
Остановка — Ctrl-C или закрытие GUI. После перезапуска драйвера нода автоматически
продолжает обработку. Для позы камера должна видеть участок существующей карты.

По умолчанию координаты в единицах карты доступны в `/droid_slam/status` (JSON)
и GUI. Для метрического `/droid_slam/pose` задайте измеренный `scale_to_meters`;
TF включается отдельно через `publish_tf:=true`. Новая камера и depth-поток
не определяют метрический масштаб ранее построенной монокулярной карты.

Проверено: 42 автоматических теста; реальная D435 с GUI и перезапуском драйвера;
ROS-видео — 378 кадров без пропусков, 369 поз после начального поиска, p95 от
timestamp кадра до публикации 34,2 мс. Прежний офлайн-CSV совпал побайтно.
Отчёты: `outputs/ros_replay_final/ros_report.json` и
`outputs/ros_d435_reconnect_final/validation.json`.
Во время аппаратного теста D435 смотрела на помещение вне карты, поэтому
поза на физической камере оставалась неопределённой. Для её проверки в этой
карте камеру нужно перенести в картографированное место.
