#!/usr/bin/env python3
"""
Смоук-тесты сборки FreestyleParser.

Запуск:
  env/bin/python tools/smoke_test.py                 # окружение сборки (быстро, до build)
  env/bin/python tools/smoke_test.py --bundle dist/FreestyleParser.app/Contents/MacOS/FreestyleParser
                                                     # содержимое собранного бандла (после build.py)

Без аргументов проверяется окружение сборки: все пиновые зависимости
импортируются, gdown читает свою метаданные (именно это падало в exe:
"No package metadata was found for gdown"), torchreid строит OSNet.
С --bundle дополнительно проверяется, что PyInstaller унёс в бандл
dist-info gdown (без него torchreid в exe падает с той же ошибкой),
а также прочие collect-all пакеты и модели.
"""

import glob
import importlib
import importlib.metadata as md
import os
import sys

PINNED = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'opencv-python': 'cv2',
    'pillow': 'PIL',
    'PyYAML': 'yaml',
    'requests': 'requests',
    'ultralytics': 'ultralytics',
    'supervision': 'supervision',
    'lap': 'lap',
    'matplotlib': 'matplotlib',
    'scikit-learn': 'sklearn',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchreid': 'torchreid',
    'gdown': 'gdown',
}

# Пакеты, импортируемые лениво из воркеров - PyInstaller сам их не видит,
# нужны collect-all в build.py и FreestyleParser.spec
COLLECT_REQUIRED = ['ultralytics', 'scipy', 'supervision', 'lap', 'torchreid', 'gdown']

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  ok: {name}")
    except Exception as e:
        failures.append((name, e))
        print(f"  FAIL: {name}: {e}")


def smoke_env():
    print("Окружение сборки:")
    for dist, mod in PINNED.items():
        def f(dist=dist, mod=mod):
            importlib.import_module(mod)
            assert md.version(dist), f"нет метаданных {dist}"
        check(f"import {mod}", f)

    def gdown_meta():
        v = md.version('gdown')
        assert v == '6.4.0', f"gdown {v} != 6.4.0 (пин из requirements.txt)"
    check("метаданные gdown читаются (importlib.metadata.version)", gdown_meta)

    def reid_model():
        import torchreid
        m = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=False).eval()
        assert sum(p.numel() for p in m.parameters()) > 0
    check("torchreid строит osnet_x1_0 (импорт-цепочка тянет gdown)", reid_model)


def collect_flags(path):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    return src


def smoke_build_scripts():
    print("Сборочные скрипты:")
    for script, marker in [('build.py', '--collect-all='), ('FreestyleParser.spec', "collect_all('")]:
        def f(script=script, marker=marker):
            src = collect_flags(script)
            for pkg in COLLECT_REQUIRED:
                assert f"{marker}{pkg}'" in src or f"{marker}{pkg})" in src, \
                    f"{script}: нет collect-all для {pkg}"
        check(f"{script}: collect-all для {', '.join(COLLECT_REQUIRED)}", f)


def find_internal(bundle):
    bundle = bundle.rstrip(os.sep)
    # 0) передан сам onedir-каталог (dist/FreestyleParser)
    if os.path.isdir(os.path.join(bundle, '_internal')):
        return os.path.join(bundle, '_internal')
    # 1) _internal рядом с бинарником (onedir: dist/FreestyleParser/FreestyleParser)
    d = os.path.dirname(bundle)
    if os.path.isdir(os.path.join(d, '_internal')):
        return os.path.join(d, '_internal')
    # 2) .app на macOS: Contents/MacOS/FreestyleParser -> _internal в onedir рядом с .app
    app_root = bundle[:bundle.find('.app') + len('.app')] if '.app' in bundle else None
    if app_root:
        onedir = os.path.join(os.path.dirname(app_root), os.path.splitext(os.path.basename(app_root))[0])
        if os.path.isdir(os.path.join(onedir, '_internal')):
            return os.path.join(onedir, '_internal')
    raise FileNotFoundError(f"не найден _internal под {bundle}")


def smoke_bundle(bundle):
    internal = find_internal(bundle)
    print(f"Бандл: {bundle}")
    print(f"  _internal: {internal}")

    def meta():
        infos = glob.glob(os.path.join(internal, 'gdown-*.dist-info'))
        assert infos, "gdown-*.dist-info отсутствует в бандле - torchreid в exe упадёт с 'No package metadata was found for gdown'"
        with open(os.path.join(infos[0], 'METADATA'), encoding='utf-8') as f:
            head = f.readline()
        print(f"    {os.path.basename(infos[0])} -> {head.strip()}")
    check("gdown dist-info внутри бандла", meta)

    for pkg in COLLECT_REQUIRED:
        def f(pkg=pkg):
            assert os.path.isdir(os.path.join(internal, pkg)), f"{pkg} не собран в бандл"
        check(f"пакет {pkg} внутри бандла", f)

    def models():
        for name in ['yolo11s-seg.pt', 'osnet_x1_0_combineall.pth', 'osnet_ain_x1_0_msmt17.pth']:
            p = os.path.join(internal, 'models', name)
            assert os.path.isfile(p), f"нет модели {name}"
            assert os.path.getsize(p) > 1_000_000, f"модель {name} подозрительно мала"
    check("модели YOLO и OSNet внутри бандла", models)

    def ffmpeg():
        for name in ('ffmpeg', 'ffprobe'):
            found = glob.glob(os.path.join(internal, 'bin', name + '*'))
            assert found, f"нет {name} в bin бандла"
    check("ffmpeg/ffprobe внутри бандла", ffmpeg)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    flags = [a for a in sys.argv[1:] if a.startswith('-')]
    if '--bundle' in flags and args:
        smoke_env()
        smoke_build_scripts()
        smoke_bundle(args[0])
    else:
        smoke_env()
        smoke_build_scripts()
    print()
    if failures:
        print(f"Смоук ПРОВАЛЕН: {len(failures)} проверок не прошли")
        sys.exit(1)
    print("Смоук пройден")
