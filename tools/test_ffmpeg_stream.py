#!/usr/bin/env python3
"""CI-тест стримингового скана на бандловом ffmpeg.

Проверяет, что скачанный (download_ffmpeg.py) ffmpeg:
1) умеет кодировать interlaced h264 в mpegts с GOP=12 (AVCHD-подобный контейнер,
   как реальные MTS с камер);
2) отдаёт I-кадры через тот же пайп, что использует _KeyframeStreamIter
   (-skip_frame nokey + showinfo + scale + rawvideo) - ровно то, что молча
   сломалось на Windows в v1.3.22 (0 кадров -> «разобран за 0 сек»).

Запуск: python tools/test_ffmpeg_stream.py   (нужен bin/ffmpeg рядом)
Код возврата 0 - ffmpeg годен для стримингового скана.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_ffmpeg_path  # noqa: E402

W, H, FPS, SEC, GOP = 320, 240, 25, 5, 12


def run(args, timeout=120):
    return subprocess.run(args, capture_output=True, timeout=timeout)


def main():
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg or not os.path.isfile(ffmpeg):
        print('FAIL: ffmpeg не найден (bin/)')
        return 1
    ver = run([ffmpeg, '-version'])
    if ver.returncode != 0:
        print('FAIL: ffmpeg -version упал:', ver.stderr.decode('utf-8', 'replace')[:300])
        return 1
    print('ffmpeg:', ver.stdout.decode('utf-8', 'replace').splitlines()[0])

    with tempfile.TemporaryDirectory() as tmp:
        mts = os.path.join(tmp, 'test_gop12.mts')
        # AVCHD-подобный файл: interlaced h264 в mpegts, GOP=12 при 25 fps
        # (как MTS камер: I-кадры каждые 0.48с)
        enc = run([ffmpeg, '-hide_banner', '-nostdin',
                   '-f', 'lavfi', '-i', f'testsrc2=size={W}x{H}:rate={FPS}',
                   '-vf', 'tinterlace=mode=interleave_top',
                   '-c:v', 'libx264', '-g', str(GOP),
                   '-flags', '+ilme+ildct', '-x264opts', 'interlaced=1',
                   '-pix_fmt', 'yuv420p', '-t', str(SEC), '-f', 'mpegts', mts])
        if enc.returncode != 0 or not os.path.isfile(mts):
            print('FAIL: ffmpeg не закодировал тестовый MTS:',
                  enc.stderr.decode('utf-8', 'replace')[-400:])
            return 1

        # Тот же пайп, что _KeyframeStreamIter._start_ffmpeg
        out_h = H  # scale по высоте исходника, ширина уже маленькая
        cmd = [ffmpeg, '-hide_banner', '-nostdin', '-skip_frame', 'nokey',
               '-i', mts,
               '-vf', f'showinfo,scale={W}:{out_h}',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vsync', '0', 'pipe:1']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        frame_bytes = W * out_h * 3
        n_frames = 0
        stderr_tail = []
        try:
            while True:
                chunk = proc.stdout.read(frame_bytes)
                if not chunk:
                    break
                if len(chunk) == frame_bytes:
                    n_frames += 1
                if n_frames > 60:   # страховка от бесконечного потока
                    break
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
            stderr_tail = proc.stderr.read().decode('utf-8', 'replace') \
                .strip().splitlines()[-5:]

        expect = int(SEC * FPS / GOP)  # ~10 I-кадров за 5с при GOP=12
        if proc.returncode != 0:
            print('FAIL: ffmpeg декодер вернул ошибку (код %d):' % proc.returncode)
            for line in stderr_tail:
                print('  ', line)
            return 1
        if n_frames < max(4, expect // 2):
            print(f'FAIL: -skip_frame nokey дал {n_frames} I-кадров '
                  f'(ожидалось ~{expect}) - стриминговый скан не работает.')
            for line in stderr_tail:
                print('  ', line)
            return 1
        print(f'OK: {n_frames} I-кадров из {SEC}с interlaced MTS (GOP={GOP}) '
              f'через стриминговый пайп')
        return 0


if __name__ == '__main__':
    sys.exit(main())
