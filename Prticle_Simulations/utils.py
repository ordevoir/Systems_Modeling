import subprocess
from pathlib import Path
from typing import Union
import imageio_ffmpeg
import os

Time = Union[float, int, str]  # секунды (число) или "HH:MM:SS[.ms]"

def _to_ffmpeg_ts(t: Time) -> str:
    """Привести время к строке HH:MM:SS.mmm для ffmpeg."""
    if isinstance(t, (int, float)):
        ms = int(round((t - int(t)) * 1000))
        s  = int(t) % 60
        m  = (int(t) // 60) % 60
        h  = int(t) // 3600
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return t  # уже строка

def trim_mp4(
    in_path: str,
    out_path: str,
    start: Time,
    end: Time,
    accurate: bool = False,
    reencode_video: str = "libx264",  # используется если accurate=True
    reencode_audio: str = "aac",      # используется если accurate=True
    crf: int = 18,                    # качество для H.264 (меньше = лучше)
    preset: str = "medium"            # скорость/сжатие для H.264
) -> None:
    """
    Обрезать видео [start, end). Если accurate=False — быстрое копирование потоков,
    но резка по ближайшим ключевым кадрам. Если accurate=True — точная резка с перекодированием.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    in_path = str(Path(in_path))
    out_path = str(Path(out_path))
    ss = _to_ffmpeg_ts(start)
    to = _to_ffmpeg_ts(end)

    # Для ffmpeg удобнее задавать длительность (-t), чем -to, когда используем -ss.
    # Посчитаем длительность в секундах для надёжности.
    def _to_seconds(t: Time) -> float:
        if isinstance(t, (int, float)):
            return float(t)
        h, m, s = t.split(":")
        s = float(s)
        return int(h)*3600 + int(m)*60 + s

    duration = _to_seconds(end) - _to_seconds(start)
    if duration <= 0:
        raise ValueError("end должно быть больше start")

    if accurate:
        # Точный вариант: -ss после -i и перекодирование
        cmd = [
            ffmpeg, "-y",
            "-i", in_path,
            "-ss", ss,
            "-t", f"{duration:.3f}",
            "-map", "0",
            "-c:v", reencode_video, "-crf", str(crf), "-preset", preset,
            "-c:a", reencode_audio, "-b:a", "192k",
            "-movflags", "+faststart",
            out_path
        ]
    else:
        # Быстрый вариант: -ss до -i и копирование потоков (режет по keyframe)
        cmd = [
            ffmpeg, "-y",
            "-ss", ss,
            "-i", in_path,
            "-t", f"{duration:.3f}",
            "-map", "0",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            out_path
        ]

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        # Вывести stderr для диагностики
        raise RuntimeError(f"ffmpeg error:\n{completed.stderr}")
    else:
        print(f"Saved: {out_path}")

# Примеры использования:

# 1) Быстро (без перекодирования), с 10.0 c до 25.0 c:
# trim_mp4("./simulation.mp4", "./simulation_trim_fast.mp4", start=10.0, end=25.0, accurate=False)

# 2) Точно (с перекодированием), с 00:00:10.000 до 00:00:25.000:
# trim_mp4("./simulation.mp4", "./simulation_trim_exact.mp4",
#          start="00:00:10.000", end="00:00:25.000", accurate=True, crf=20, preset="faster")


import subprocess, shlex, os
from typing import Optional
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"  # fallback, если imageio-ffmpeg недоступен


def compress_mp4(
    in_path: str,
    out_path: str,
    *,
    scale: float = 1.0,          # < 1 — даунскейл, > 1 — апскейл
    codec: str = "libx264",      # для HEVC: "libx265" или "hevc_nvenc"
    crf: int = 18,               # 18—20: визуально очень хорошо; 22—24: компактнее
    preset: str = "slow",        # ultrafast ... veryslow (чем медленнее — тем эффективнее)
    tune: Optional[str] = None,  # напр. "animation" или "film"; None — не задаём
    pix_fmt: str = "yuv420p",    # самый совместимый формат
    copy_audio: bool = True,     # True — не пережимать звук
    audio_bitrate: str = "192k", # если copy_audio=False
    faststart: bool = True,      # перенос moov-атомов в начало для веб-плееров
    fps: Optional[float] = None, # если нужно сменить частоту кадров
    extra_vf: Optional[str] = None,   # доп. видеофильтры, объединятся с scale
    overwrite: bool = True,
    loglevel: str = "error",
):
    """
    Сжимает видео CRF-режимом и масштабирует разрешение в `scale` раз.
    Пример: compress_mp4("in.mp4","out.mp4", scale=0.5, crf=20, preset="slow")
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    # Собираем фильтр масштабирования.
    # Делаем чётные ширину/высоту: trunc(.../2)*2 (важно для yuv420p)
    if scale != 1.0:
        vf_scale = f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"
    else:
        # даже без масштабирования гарантируем чётные размеры
        vf_scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    if extra_vf:
        vf = f"{vf_scale},{extra_vf}"
    else:
        vf = vf_scale

    # Базовая команда
    cmd = [FFMPEG_EXE, "-hide_banner", "-loglevel", loglevel, "-y" if overwrite else "-n", "-i", in_path]

    # Видео-настройки
    cmd += ["-c:v", codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", pix_fmt, "-vf", vf]
    if tune:
        cmd += ["-tune", tune]
    if fps is not None:
        cmd += ["-r", str(fps)]

    # Аудио
    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]

    # MP4-вкусности
    if faststart:
        cmd += ["-movflags", "faststart"]

    # Выходной файл
    cmd += [out_path]

    # Печать команды (удобно для отладки)
    print("Running:\n", " ".join(shlex.quote(x) for x in cmd))

    # Запуск
    subprocess.run(cmd, check=True)
    return out_path


import os, shlex, subprocess
try:
    import imageio_ffmpeg
    _FFMPEG_EXE_DEFAULT = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG_EXE_DEFAULT = "ffmpeg"

def crop_mp4(in_path: str,
             out_path: str,
             margins: tuple[int, int, int, int],
             *,
             order: str = "lrbt",          # порядок кортежа: left, right, top, bottom
             ffmpeg_path: str | None = None,
             codec: str = "libx264",
             crf: int = 18,
             preset: str = "slow",
             pix_fmt: str = "yuv420p",
             faststart: bool = True,
             audio_copy: bool = True,
             overwrite: bool = True) -> None:
    """
    Кадрирование MP4 по отступам от 4 границ (в пикселях).

    margins: кортеж из 4 целых. По умолчанию порядок (left, right, top, bottom).
             Если нужно (top, right, bottom, left), передайте order="trbl".

    Примечания:
      - Выходные ширина и высота округляются вниз до ближайших чётных.
      - Видео перекодируется (crop требует перекодирования). Аудио копируется (если есть).
    """
    if len(margins) != 4:
        raise ValueError("margins должен содержать ровно 4 значения")

    if order.lower() == "lrbt":
        left, right, top, bottom = margins
    elif order.lower() == "trbl":
        top, right, bottom, left = margins
    else:
        raise ValueError("order должен быть 'lrbt' или 'trbl'")

    for v in (left, right, top, bottom):
        if v < 0:
            raise ValueError("Отступы не могут быть отрицательными")

    ffmpeg = ffmpeg_path or _FFMPEG_EXE_DEFAULT
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Итоговые размеры как выражения ffmpeg:
    # crop=out_w:out_h:x:y
    # Гарантируем чётность размеров через floor(.../2)*2
    w_expr = f"floor((iw-{left}-{right})/2)*2"
    h_expr = f"floor((ih-{top}-{bottom})/2)*2"
    x_expr = f"{left}"
    y_expr = f"{top}"
    vf = f"crop={w_expr}:{h_expr}:{x_expr}:{y_expr}"

    cmd = [ffmpeg]
    cmd += ["-y" if overwrite else "-n"]
    cmd += ["-i", in_path]

    # Явно мапим первый видеопоток и (опц.) все аудиопотоки
    cmd += ["-map", "0:v:0", "-vf", vf, "-c:v", codec, "-crf", str(crf), "-preset", preset, "-pix_fmt", pix_fmt]
    if audio_copy:
        cmd += ["-map", "0:a?", "-c:a", "copy"]  # аудио если есть
    else:
        cmd += ["-an"]

    if faststart:
        cmd += ["-movflags", "faststart"]

    cmd += [out_path]

    # Запуск
    # print("Running:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


# 40 px слева/справа, 60 px сверху, 24 px снизу
# crop_mp4("input.mp4", "out/cropped.mp4", (40, 40, 60, 24))

# Если у тебя кортеж в порядке (top, right, bottom, left):
# crop_mp4("input.mp4", "out/cropped_trbl.mp4", (60, 40, 24, 40), order="trbl")
