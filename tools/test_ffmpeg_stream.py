#!/usr/bin/env python3
"""CI-тест стримингового скана на бандловом ffmpeg.

Проверяет, что скачанный (download_ffmpeg.py) ffmpeg:
1) умеет кодировать interlaced h264 в mpegts с GOP=12 (AVCHD-подобный контейнер,
   как реальные MTS с камер);
2) отдаёт I-кадры через тот же пайп, что использует _KeyframeStreamIter
   (-skip_frame nokey + showinfo + scale + rawvideo) - ровно то, что молча
   сломалось на Windows в v1.3.22 (0 кадров -> «разобран за 0 сек»).

Запуск: python tools/test_ffmpeg_stream.py   (негаснущий: любые зависания
ffmpeg убиваются таймером). Код возврата 0 - ffmpeg годен для стриминга.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_ffmpeg_path  # noqa: E402

W, H, FPS, SEC, GOP = 320, 240, 25, 5, 12
ENCODE_TIMEOUT = 180
DECODE_TIMEOUT = 180


def run(args, timeout):
    return subprocess.run(args, capture_output=True, timeout=timeout)


def main():
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg or not os.path.isfile(ffmpeg):
        print('FAIL: ffmpeg не найден (bin/)', flush=True)
        return 1
    ver = run([ffmpeg, '-version'], 30)
    if ver.returncode != 0:
        print('FAIL: ffmpeg -version упал:', ver.stderr.decode('utf-8', 'replace')[:300], flush=True)
        return 1
    print('ffmpeg:', ver.stdout.decode('utf-8', 'replace').splitlines()[0], flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        mts = os.path.join(tmp, 'test_gop12.mts')
        # AVCHD-подобный файл: interlaced h264 в mpegts, GOP=12 при 25 fps
        # (как MTS камер: I-кадры каждые 0.48с)
        enc = run([ffmpeg, '-hide_banner', '-nostdin',
                   '-f', 'lavfi', '-i', f'testsrc2=size={W}x{H}:rate={FPS}',
                   '-vf', 'tinterlace=mode=interleave_top',
                   '-c:v', 'libx264', '-g', str(GOP),
                   '-flags', '+ilme+ildct', '-x264opts', 'interlaced=1',
                   '-pix_fmt', 'yuv420p', '-t', str(SEC), '-f', 'mpegts', mts],
                  ENCODE_TIMEOUT)
        if enc.returncode != 0 or not os.path.isfile(mts):
            print('FAIL: ffmpeg не закодировал тестовый MTS:',
                  enc.stderr.decode('utf-8', 'replace')[-400:], flush=True)
            return 1
        print(f'encode OK: {os.path.getsize(mts)} байт', flush=True)

        # Тот же пайп, что _KeyframeStreamIter._start_ffmpeg.
        # stderr - в файл (не PIPE: переполнение буфера пайпа вешает ffmpeg),
        # чтение stdout - под ватчдогом (зависание = kill и FAIL)
        err_path = os.path.join(tmp, 'stderr.txt')
        cmd = [ffmpeg, '-hide_banner', '-nostdin', '-skip_frame', 'nokey',
               '-i', mts,
               '-vf', f'showinfo,scale={W}:{H}',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-fps_mode', 'passthrough',
               'pipe:1']
        frame_bytes = W * H * 3
        n_frames = 0
        with open(err_path, 'wb') as err_f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_f)
            watchdog = threading.Timer(DECODE_TIMEOUT, proc.kill)
            watchdog.start()
            try:
                t0 = time.time()
                while True:
                    chunk = proc.stdout.read(frame_bytes)
                    if not chunk:
                        break
                    if len(chunk) == frame_bytes:
                        n_frames += 1
                    if n_frames % 5 == 0:
                        print(f'  ...{n_frames} I-кадров ({time.time() - t0:.0f}с)', flush=True)
                    if n_frames > 60:   # страховка от бесконечного потока
                        proc.kill()
                        break
                proc.wait(timeout=60)
            except Exception as e:
                print('FAIL: сбой чтения пайпа:', e, flush=True)
                return 1
            finally:
                watchdog.cancel()
                if proc.poll() is None:
                    proc.kill()
                try:
                    proc.stdout.close()
                except Exception:
                    pass
        stderr_tail = open(err_path, 'rb').read() \
            .decode('utf-8', 'replace').strip().splitlines()[-5:]

        expect = int(SEC * FPS / GOP)  # ~10 I-кадров за 5с при GOP=12
        if proc.returncode not in (0, None) and n_frames == 0:
            print(f'FAIL: ffmpeg декодер вернул ошибку (код {proc.returncode}):', flush=True)
            for line in stderr_tail:
                print('  ', line, flush=True)
            return 1
        if n_frames < max(4, expect // 2):
            print(f'FAIL: -skip_frame nokey дал {n_frames} I-кадров '
                  f'(ожидалось ~{expect}) - стриминговый скан не работает.', flush=True)
            for line in stderr_tail:
                print('  ', line, flush=True)
            return 1
        print(f'OK: {n_frames} I-кадров из {SEC}с interlaced MTS (GOP={GOP}) '
              f'через стриминговый пайп', flush=True)
        return 0


if __name__ == '__main__':
    sys.exit(main())
