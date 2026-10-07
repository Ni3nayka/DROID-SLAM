# droid_slam: локализация ROS 2 по готовой карте

Пакет для ROS 2 Jazzy принимает цветные кадры реальной камеры и показывает её
проверенную позу в готовой DROID-карте `.pth`. Использует существующие
`LiveTracker`, фоновый поиск по карте и GUI с полной плотной реконструкцией из
`view_reconstruction.py`. Карта остаётся фиксированной. Прежние `demo.py`,
`localize.py`, `live_localize.py` и просмотрщики запускаются как раньше.

## Сборка

Из корня DROID-SLAM, с настроенной проектной `.venv`:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base ros/log build --base-paths ros/droid_slam \
  --build-base ros/build --install-base ros/install --symlink-install
source ros/install/setup.bash
ros2 launch droid_slam localization.launch.py --show-args
```

Нужны ROS-пакеты из `package.xml`, `colcon`, а для вычислений — зависимости уже
работающего DROID-SLAM. GUI использует Open3D, графическую сессию и исходные
CUDA-модули реконструкции. Для запуска без окон задайте `show_gui:=false
show_preview:=false`.

Точка входа переключается на `<репозиторий>/.venv/bin/python`, сохраняя пути ROS.
На Jazzy это Python 3.12; версия Python должна совпадать с версией ROS-библиотек.
Другая среда задаётся через `DROID_SLAM_PYTHON`, перемещённый репозиторий — через
`DROID_SLAM_ROOT`. Сборка привязана к этому checkout; запуск возможен из другой
рабочей папки при абсолютных путях к данным. Нативный `cv_bridge` не требуется:
адаптер читает буфер Image напрямую, включая padding строк, поэтому не зависит
от несовместимости системного cv_bridge с NumPy 2 в проектной среде.

## Подключённая D435

У проверенного экземпляра **device serial — `948122071094`**, а
**ASIC serial — `948123023494`**. Параметр драйвера `serial_no` использует первый.
Пакет `realsense` в примере — существующий драйвер из соседнего `IRS-DRIVERS`.

Терминал 1, из корня DROID-SLAM:

```bash
source /opt/ros/jazzy/setup.bash
source ../IRS-DRIVERS/install/setup.bash
ros2 launch realsense driver.launch.py model:=d435 serial_no:=948122071094
```

Терминал 2, из корня DROID-SLAM:

```bash
source /opt/ros/jazzy/setup.bash
source ros/install/setup.bash
ros2 launch droid_slam localization.launch.py \
  map_path:="$PWD/outputs/my_map_04_2.pth" \
  image_topic:=/camera/color/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  save_calibration_path:="$PWD/outputs/d435_camera_info.yaml" \
  output_dir:="$PWD/outputs/ros_d435"
```

Откроются полная 3D-карта и окно текущего кадра. Завершение — Ctrl-C, закрытие
3D-окна или Esc в окне кадра. При остановке драйвера поза становится недействительной;
после его запуска обработка и поиск по карте возобновляются автоматически.
Для повторной записи результата задайте новую `output_dir` либо `overwrite:=true`.
Без `output_dir` запись на диск выключена. Драйвер запускается отдельно.

Чтобы получить позу, направьте камеру на **место, присутствующее в карте**.
Карта, построенная другой камерой, пригодна при достаточном совпадении видимых
деталей и корректной калибровке новой камеры. Цветной поток D435 используется
как монокулярный; depth не участвует и не делает масштаб старой карты метрическим.

## Калибровка

При пустом `calib_path` нода ожидает `sensor_msgs/CameraInfo` для выбранного
цветного потока. Проверяет размер изображения, optical frame, K, D и модель
искажений. Изменение калибровки сбрасывает трекинг и запускает локализацию заново.
`save_calibration_path` сохраняет снимок полученной калибровки в ROS YAML.

Файлы, прочитанные из подключённой камеры через librealsense:

- `calib/realsense_d435_948122071094_color_640x480.yaml`;
- `calib/realsense_d435_948122071094_color_1280x720.yaml`.

Они содержат происхождение, оба серийных номера, firmware и параметры профиля.
В обоих профилях заводские D равны нулю; это результат чтения камеры, а не
подстановка универсальных коэффициентов D435. Это заводская калибровка, без
новой оценки точности по шахматной доске. Фокусные расстояния и центр различаются
между профилями, поэтому используйте файл именно для текущего разрешения.

Явный файл имеет приоритет над CameraInfo:

```bash
ros2 launch droid_slam localization.launch.py \
  map_path:="$PWD/outputs/my_map_04_2.pth" \
  calib_path:="$PWD/calib/realsense_d435_948122071094_color_640x480.yaml"
```

Поддерживается стандартный ROS YAML и прежний текстовый `fx fy cx cy [D...]`.
Для `.txt` обязательны `calib_width` и `calib_height`. Калибровку Samsung нужно
оставить для видео Samsung, а не использовать для D435. Для уже выпрямленного
цветного изображения задайте `input_rectified:=true`: нода использует P и не
применяет D повторно. Поддерживаются plumb_bob/rational_polynomial; fisheye,
неучтённые ROI/binning и нетривиальная стереоректификация отклоняются явно.

Повторное чтение заводской калибровки (на этой машине SDK в системном Python):

```bash
/usr/bin/python3 evaluation_scripts/export_realsense_calibration.py \
  --serial 948123023494 --width 640 --height 480 --fps 30 \
  --output outputs/d435_factory.yaml
```

## Топики, координаты и задержки

| Топик | Тип | Содержание |
| --- | --- | --- |
| `/droid_slam/status` | `std_msgs/String`, JSON | Состояние, `valid`, поза в единицах карты, `ros_stamp_ns`, причины отказа и задержки |
| `/droid_slam/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Диагностика состояния, раз в секунду и при ошибке |
| `/droid_slam/pose` | `geometry_msgs/PoseStamped` | Проверенная поза в метрах, только при `scale_to_meters > 0` |
| `/tf` | `tf2_msgs/TFMessage` | Опционально, при `publish_tf:=true` и известном масштабе |

Положение — центр камеры, quaternion `xyzw` — поворот **camera optical → map**.
Оптические оси: x вправо, y вниз, z вперёд. Ориентация системы карты определяется
исходной реконструкцией: она не обязана иметь z вверх или совпадать с ROS ENU.
`map_frame:=droid_map` лишь задаёт имя системы; преобразование осей не выполняется.

Монокулярная карта имеет произвольный масштаб. По умолчанию `scale_to_meters:=0`,
так что GUI/JSON работают в единицах карты, а метрические PoseStamped/TF не
публикуются. После измерения масштаба задайте число метров на единицу карты.
Для TF используется отдельный `camera_frame:=droid_camera_optical_frame`, чтобы
не присваивать существующему optical frame драйвера второго родителя. Привязка
к `base_link` и общей TF-системе робота требует отдельно измеренной внешней калибровки.

Поза имеет исходный `Image.header.stamp`, а не время окончания вычисления.
Состояния: `WAITING_IMAGE`, `WAITING_CALIBRATION`, `INITIALIZING`, `TRACKING`,
`LOST`, `STALE`, `INVALID_INPUT`, `STOPPED`, `ERROR`. При потере координаты в
status становятся `null`; новые PoseStamped/TF не выдаются. ROS-потребитель должен
контролировать свежесть header и status: в TF может оставаться исторический transform.

Callback хранит только последний кадр, вычисления выполняются в рабочем потоке,
поиск по карте и GUI — в отдельных процессах. `max_frame_age` по умолчанию
150 мс: старые кадры/позы не выдаются как свежие. Проверяется ROS-время header,
а watchdog использует монотонные часы и продолжает работать при паузе `/clock`.
Для rosbag с `/clock` включайте `use_sim_time:=true`; при скачке timestamp назад
трекинг сбрасывается. `check_header_age:=false` отключает только сравнение с
ROS-временем, но сохраняет локальный watchdog. Нулевой stamp вне sim-time отклоняется.

По умолчанию QoS `RELIABLE`, depth 1: он проверен с текущим драйвером D435.
Для источника с BEST_EFFORT задайте `qos_reliability:=best_effort`. Для
CompressedImage укажите полный топик, например `/camera/color/image_raw/compressed`,
и `image_transport:=compressed`. Поддерживаются bgr8/rgb8/bgra8/rgba8/mono8,
цветные JPEG/PNG. Глубинный поток вместо цветного отклоняется.

`output_dir` содержит совместимые `trajectory.csv`, `metadata.json`, `report.json`,
а также `poses.jsonl`, `latest_pose.json`, `realtime_report.json`, `ros_report.json`.
CSV содержит обработанные кадры с исходными ROS timestamp; пропуски не интерполируются.
Просмотр: `view_localization.py --trajectory <output_dir>/trajectory.csv`.
Видео с физической камеры не записывается. JSONL фиксирует результат вычисления;
ROS-публикация дополнительно проверяет его срок действия. При чтении файлов на
том же компьютере проверяйте `valid_until_monotonic` через `time.monotonic()`.

`ros_report.json` отдельно считает пропуски прогрева и рабочего потока, время от
приёма до публикации и от header до публикации. Последнее включает доставку кадра,
но точность измерения зависит от часов и семантики timestamp драйвера. Процентили
считаются по последним 10 000 значениям. Производительность не оценивает точность
положения; для неё нужна независимая эталонная траектория.

Все параметры находятся в `config/localization.yaml`; их можно передать launch
аргументами или через `params_file`. Параметры локализации задаются при запуске.

## Воспроизводимые проверки

```bash
source /opt/ros/jazzy/setup.bash
source ros/install/setup.bash
.venv/bin/python -m unittest discover -s tests_localization -v
.venv/bin/python -m unittest discover -s ros/droid_slam/test -v
```

ROS-тесты проверяют декодирование/калибровку, часы и настоящий ROS-граф с поиском
по синтетической карте: поза/TF, исходный timestamp, потеря и восстановление,
остановка кадров без публикации застывшей позы. Прежние тесты проверяют офлайн
и live-локализацию.

Видео через ROS, терминал 1:

```bash
ros2 launch droid_slam localization.launch.py \
  map_path:="$PWD/outputs/my_map_04_2.pth" \
  image_topic:=/droid_test/image_raw camera_info_topic:=/droid_test/camera_info \
  output_dir:="$PWD/outputs/ros_video"
```

Терминал 2, после загрузки GUI:

```bash
source ros/install/setup.bash
ros2 run droid_slam video_publisher \
  --video "$PWD/input/20261003_145134.mp4" \
  --calib "$PWD/calib/sasung_cam_calibrated.txt" --calib-size 1280 720
```

Publisher передаёт кадры по исходным PTS с согласованными Image/CameraInfo и
ROS timestamp. После EOF нода остаётся в `STALE` до остановки или нового потока.
Для проверки JPEG добавьте `--compressed` publisher и `image_transport:=compressed`
ноде. Для ограниченного прогона доступны `duration` (секунды после загрузки) и
`max_frames` (обработанные кадры).

Физическая D435, GUI, отключение/повторный запуск драйвера и корректное завершение:

```bash
source ../IRS-DRIVERS/install/setup.bash
source ros/install/setup.bash
.venv/bin/python evaluation_scripts/validate_ros_realsense.py \
  --map outputs/my_map_04_2.pth --serial 948122071094 \
  --output outputs/ros_d435_validation --gui
```

Этот тест сам запускает и останавливает свои процессы драйвера/локализатора;
предварительно закройте другой драйвер камеры. Проверяет минимум 90 разных
кадров до и после перезапуска, переход в STALE, калибровку и чистое завершение.
Для проверки режима с явным файлом добавьте
`--calib calib/realsense_d435_948122071094_color_640x480.yaml` и выберите другую
папку `--output`.
Наличие корректных кадров не означает наличие позы: в комнате вне исходной
уличной карты ожидается `INITIALIZING/no_geometric_solution`.

### Результаты на этом компьютере, 2026-10-07

| Проверка | Результат |
| --- | --- |
| Прежние автоматические тесты | 30/30 |
| ROS-адаптеры и настоящий ROS-граф с позой/TF | 12/12 |
| Старый офлайн-режим, `20261003_145134.mp4` | 378/378 поз; CSV побайтно совпал с прежним |
| Старый live-режим, ограничение 90 кадров | 90 обработаны без пропусков; 80 поз после 10 кадров начального поиска |
| ROS raw RELIABLE, полное видео, GUI | 378 обработаны без пропусков; 369 поз после 9 кадров начального поиска |
| ROS CompressedImage, полное видео | 378 обработаны; 369 поз |
| ROS raw: p95 приём → вычисление / публикация | 8,7 / 16,2 мс |
| ROS raw: p95 timestamp изображения → публикация | 34,2 мс |
| Физическая D435, GUI и перезапуск драйвера | 90 + 90 кадров; STALE при остановке, 0 рабочих пропусков, чистый Ctrl-C |
| Физическая D435 с явным YAML калибровки | Ещё 90 + 90 кадров; источник калибровки подтверждён, перезапуск и Ctrl-C успешны |
| Прежний просмотрщик открывает ROS CSV | 378 строк, 369 поз, полная карта 386 950 точек |
| colcon, symlink и обычная установка | Сборка успешна; обычная установка запускается из `/tmp` |

Отчёты: `outputs/ros_replay_final/ros_report.json`,
`outputs/ros_d435_reconnect_final/validation.json`,
`outputs/ros_d435_file_calibration/validation.json`,
`outputs/regression_offline_after_ros/report.json`,
`outputs/regression_live_after_ros/realtime_report.json`.
Снимки полной карты: `outputs/ros_replay_final/map.png` и `legacy_viewer.png`.
Реальная D435 находилась в помещении вне уличной карты: тест аппаратуры не
подтверждает точность её позы на картографированном участке. Этот выездной тест
с той же D435 остаётся отдельной проверкой в нужном месте.
