'''
python calib/calibrate_camera_video.py \
  input/20260810_173854.mp4 \
  --step 15 \
  --max-frames 60 \
  --output calib/sasung_cam_calibrated.txt
'''


#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# ИЗМЕНИТЕ ПОД СВОЮ ДОСКУ
#
# Указывается количество ВНУТРЕННИХ углов.
# Например, для доски 10 x 7 клеток: (9, 6).
# ============================================================
BOARD_SIZE = (7, 7)

# Физический размер стороны клетки, например 25 мм.
SQUARE_SIZE = 25.0


def detect_corners(gray):
    """
    Ищет внутренние углы шахматной доски.

    findChessboardCornersSB обычно устойчивее классического метода,
    особенно при перспективных искажениях и дополнительном рисунке
    внутри клеток.
    """
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    found, corners = cv2.findChessboardCornersSB(
        gray,
        BOARD_SIZE,
        flags=flags,
    )

    return found, corners


def create_object_points():
    points = np.zeros(
        (BOARD_SIZE[0] * BOARD_SIZE[1], 3),
        dtype=np.float32,
    )

    points[:, :2] = (
        np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]]
        .T
        .reshape(-1, 2)
    )

    points *= SQUARE_SIZE
    return points


def calibrate(object_points, image_points, image_size):
    result = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=0,
    )

    (
        rms,
        camera_matrix,
        distortion,
        rotation_vectors,
        translation_vectors,
        intrinsic_std,
        extrinsic_std,
        per_view_errors,
    ) = result

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "rotation_vectors": rotation_vectors,
        "translation_vectors": translation_vectors,
        "intrinsic_std": intrinsic_std,
        "extrinsic_std": extrinsic_std,
        "per_view_errors": per_view_errors.reshape(-1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Калибровка камеры по видео с шахматной доской"
    )

    parser.add_argument(
        "video",
        help="Путь к видео",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="Проверять каждый N-й кадр, по умолчанию 15",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help="Максимальное количество принятых кадров",
    )

    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=80.0,
        help="Минимальная резкость кадра",
    )

    parser.add_argument(
        "--preview-dir",
        default="calibration_detected",
        help="Каталог с кадрами, на которых обнаружена доска",
    )

    parser.add_argument(
        "--output",
        default="camera_calibration.txt",
        help="Выходной файл для DROID-SLAM",
    )

    args = parser.parse_args()

    if args.step < 1:
        raise ValueError("--step должен быть не меньше 1")

    video = cv2.VideoCapture(args.video)

    if not video.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    object_template = create_object_points()

    object_points = []
    image_points = []
    accepted_frame_numbers = []

    frame_number = 0
    image_size = None

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print(f"Видео: {args.video}")
    print(f"Разрешение: "
          f"{int(video.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {fps:.3f}")
    print(f"Количество кадров: {total_frames}")
    print(f"Внутренние углы: {BOARD_SIZE}")
    print(f"Размер клетки: {SQUARE_SIZE}")
    print()

    while True:
        ok, frame = video.read()

        if not ok:
            break

        if frame_number % args.step != 0:
            frame_number += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        # Дисперсия лапласиана — простая оценка резкости.
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < args.min_sharpness:
            frame_number += 1
            continue

        found, corners = detect_corners(gray)

        if found:
            object_points.append(object_template.copy())
            image_points.append(corners.astype(np.float32))
            accepted_frame_numbers.append(frame_number)

            preview = frame.copy()
            cv2.drawChessboardCorners(
                preview,
                BOARD_SIZE,
                corners,
                found,
            )

            preview_path = (
                preview_dir / f"frame_{frame_number:06d}.jpg"
            )
            cv2.imwrite(str(preview_path), preview)

            print(
                f"Принят кадр {frame_number:6d}: "
                f"резкость={sharpness:8.1f}, "
                f"всего={len(image_points)}"
            )

            if len(image_points) >= args.max_frames:
                break

        frame_number += 1

    video.release()

    if len(image_points) < 10:
        raise RuntimeError(
            f"Обнаружено только {len(image_points)} подходящих кадров. "
            "Для надёжной калибровки желательно не менее 20–30."
        )

    print()
    print("Первичная калибровка...")

    initial = calibrate(
        object_points,
        image_points,
        image_size,
    )

    errors = initial["per_view_errors"]
    median_error = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median_error)))

    # Удаляем только явно плохие наблюдения.
    if mad > 1e-9:
        limit = median_error + 3.0 * 1.4826 * mad
    else:
        limit = median_error * 2.0

    # Не используем чрезмерно строгий предел.
    limit = max(limit, 0.5)

    good_indices = [
        i for i, error in enumerate(errors)
        if error <= limit
    ]

    removed_indices = [
        i for i, error in enumerate(errors)
        if error > limit
    ]

    print(f"Первичная RMS-ошибка: {initial['rms']:.4f} px")
    print(f"Медианная ошибка кадра: {median_error:.4f} px")
    print(f"Предел удаления кадров: {limit:.4f} px")

    if removed_indices and len(good_indices) >= 10:
        print("Удалены кадры с большой ошибкой:")

        for i in removed_indices:
            print(
                f"  кадр {accepted_frame_numbers[i]}: "
                f"{errors[i]:.4f} px"
            )

        filtered_object_points = [
            object_points[i] for i in good_indices
        ]
        filtered_image_points = [
            image_points[i] for i in good_indices
        ]

        result = calibrate(
            filtered_object_points,
            filtered_image_points,
            image_size,
        )
    else:
        result = initial

    camera_matrix = result["camera_matrix"]
    distortion = result["distortion"].reshape(-1)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    k1 = float(distortion[0]) if len(distortion) > 0 else 0.0
    k2 = float(distortion[1]) if len(distortion) > 1 else 0.0
    p1 = float(distortion[2]) if len(distortion) > 2 else 0.0
    p2 = float(distortion[3]) if len(distortion) > 3 else 0.0
    k3 = float(distortion[4]) if len(distortion) > 4 else 0.0

    droid_line = (
        f"{fx:.10f} {fy:.10f} "
        f"{cx:.10f} {cy:.10f} "
        f"{k1:.10f} {k2:.10f} "
        f"{p1:.10f} {p2:.10f} "
        f"{k3:.10f}"
    )

    Path(args.output).write_text(
        droid_line + "\n",
        encoding="utf-8",
    )

    print()
    print("============================================================")
    print("РЕЗУЛЬТАТ")
    print("============================================================")
    print(f"Использовано кадров: {len(result['per_view_errors'])}")
    print(f"Размер изображения: {image_size[0]} x {image_size[1]}")
    print(f"Итоговая RMS-ошибка: {result['rms']:.4f} px")
    print()
    print("Матрица камеры:")
    print(camera_matrix)
    print()
    print("Коэффициенты дисторсии:")
    print(distortion)
    print()
    print("Строка для DROID-SLAM:")
    print(droid_line)
    print()
    print(f"Результат сохранён в: {args.output}")
    print(f"Кадры для визуальной проверки: {preview_dir}")


if __name__ == "__main__":
    main()