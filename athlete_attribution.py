"""Атрибутика атлетов: профили попыток, банки, реестр, маппинг,
авто-привязки, сегменты скана атлетов и зонный профайлер.

Код вынесен из app.py и splitter.py (не удалён). Режимы с атрибутикой
сейчас отключены в UI (app.STRATEGY_MODES) - возвращаются после
стабилизации детекции границ попыток. app.py держит тонкие
делегирующие обёртки с прежними именами методов, UI не меняется.

Модуль лёгкий: splitter/attempts_v2 импортируются лениво (torch/ultralytics
не тянутся при старте приложения).
"""

import logging
import os
import threading

import numpy as np
import yaml

import i18n


class AthleteZoneProfiler:
    """Накопитель атрибутики атлета по сэмплам открытой зоны кейфрейм-скана:
    vest/helmet (HSV-гистограммы), reid (OSNet), boat (hue-бины), thumb_t
    (сэмпл, ближайший к центру ROI). Совместим по результату с attempts_v2
    (Bank-средние, те же функции hsv_hist/hue_bins)."""

    def __init__(self, start, roi_pct, reid_embed_rate=5.0, embedder=None):
        import attempts_v2
        self._av2 = attempts_v2
        self.start = start
        self.roi_pct = roi_pct  # (L, T, R, B) в процентах
        self.embedder = embedder
        self.reid_embed_rate = float(reid_embed_rate or 5.0)
        self.vest = attempts_v2.Bank()
        self.helmet = attempts_v2.Bank()
        self.reid = attempts_v2.Bank()
        self.boat_bins = np.zeros(18)
        self.thumb_dist = None
        self.thumb_t = None
        self.last_embed_t = None

    def on_sample(self, sec, frame, results):
        """frame - сэмпл (600px), results - вывод YOLO. bbox в координатах frame."""
        if results is None:
            return
        from splitter import YOLO_TARGET_CLASSES
        av2 = self._av2
        person, boat_dets = None, []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                name = self._model_names[cls] if hasattr(self, '_model_names') else ''
                if name not in YOLO_TARGET_CLASSES or conf < 0.3:
                    continue
                b = tuple(float(v) for v in box.xyxy[0].cpu().numpy())
                if name == 'person':
                    if person is None or conf > person[1]:
                        person = (b, conf)
                else:
                    boat_dets.append(b)
        if person is None:
            return
        (x1, y1, x2, y2), _conf = person
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        fh, fw = frame.shape[:2]
        x1, x2 = max(x1, 0), min(x2, fw)
        y1, y2 = max(y1, 0), min(y2, fh)

        def region(ry0, ry1, rx0, rx1):
            cy1, cy2 = int(y1 + bh * ry0), int(y1 + bh * ry1)
            cx1, cx2 = int(x1 + bw * rx0), int(x1 + bw * rx1)
            crop = frame[max(cy1, 0):cy2, max(cx1, 0):cx2]
            if crop.size == 0:
                return None, 0.0
            import cv2
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            sat = float(hsv[..., 1].mean()) / 255.0
            return crop, 0.5 + sat

        vc, vq = region(0.25, 0.75, 0.15, 0.85)
        self.vest.add(av2.hsv_hist(vc), vq)
        hc, hq = region(0.0, 0.22, 0.2, 0.8)
        self.helmet.add(av2.hsv_hist(hc), hq)

        L, T, R, B = self.roi_pct
        bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
        rw, rh = max(R - L, 1), max(B - T, 1)
        d = (((bcx - (L + R) / 2) / rw) ** 2
             + ((bcy - (T + B) / 2) / rh) ** 2) ** 0.5
        if self.thumb_dist is None or d < self.thumb_dist:
            self.thumb_dist = d
            self.thumb_t = sec

        if self.embedder is not None:
            interval = 1.0 / max(0.01, self.reid_embed_rate)
            if self.last_embed_t is None or sec - self.last_embed_t >= interval - 1e-6:
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size:
                    self.reid.add(self.embedder.embed(crop), vq)
                    self.last_embed_t = sec

        pcx, pcy = bcx, bcy
        best_b = None
        for bx1, by1, bx2, by2 in boat_dets:
            hb = av2.hue_bins(frame[int(by1):int(by2), int(bx1):int(bx2)])
            if hb is None:
                continue
            bins, sat = hb
            top = int(np.argmax(bins))
            if bins[top] < av2.BOAT_MIN_FRAC:
                continue
            dd = np.hypot(pcx - (bx1 + bx2) / 2, pcy - (by1 + by2) / 2)
            q = np.sqrt((bx2 - bx1) * (by2 - by1)) * (0.3 + sat) / (1 + dd / 200)
            if best_b is None or q > best_b[0]:
                best_b = (q, bins)
        if best_b is not None:
            self.boat_bins += best_b[1] * best_b[0]

    def result(self, start, end):
        boat = self.boat_bins / self.boat_bins.sum() if self.boat_bins.sum() > 0 else None
        return {'start': start, 'end': end,
                'thumb_t': self.thumb_t if self.thumb_t is not None else start,
                'vest': self.vest.mean(), 'helmet': self.helmet.mean(),
                'reid': self.reid.mean(),
                'boat': boat,
                'boat_top': int(np.argmax(boat)) if boat is not None else None}


class AthleteAttribution:
    """Состояние и логика атрибутики атлетов одной рабочей папки.

    Держит: маппинг атлетов (mapping.yaml), провизорные авто-привязки
    (assignments.yaml), сегменты скана атлетов (segments.yaml +
    segment_profiles.npz), профили попыток дня (attempt_profiles.npz),
    личный реестр атлетов между папками (~/.FreestyleParser) и базовые
    банки (read-only). app.py обращается через делегирующие обёртки.
    """

    REGISTRY_SAMPLES_MAX = 20  # сэмплов профилей на атлета
    ATTEMPT_SIM_THRESHOLD = 0.88   # минимальная похожесть для автоатрибуции
    ATTEMPT_SIM_MARGIN = 0.03      # отрыв от второго кандидата
    W_REID, W_BOAT = 0.7, 0.3      # вклад OSNet и лодки в похожесть

    def __init__(self, app):
        self._app = app  # доступ к output_folder/_app_folder/log (live-пути)
        self.athlete_mapping = {}
        self.attempt_assignments = {}  # авто-привязки: {attempt: {athlete, sim, second}}
        self.athlete_segments = []     # сегменты скана атлетов (без нарезки)
        self.segment_profiles = {}     # {key: {'vest', 'boat'}} профили сегментов
        self._assignments_lock = threading.Lock()
        self._registry_cache = None    # глобальный реестр (~/.FreestyleParser)
        self._base_banks_cache = None  # базовые банки (read-only, кеш сессии)

    # === Профили попыток дня ===

    def load_profile_store(self):
        """Профили попыток дня: {basename: {'vest': vec, 'boat': vec}}."""
        store_file = os.path.join(self._app.output_folder, 'attempt_profiles.npz')
        if not os.path.exists(store_file):
            return {}
        d = np.load(store_file, allow_pickle=True)
        return {str(n): {'vest': d['vest'][i], 'boat': d['boat'][i]}
                for i, n in enumerate(d['names'])}

    def save_profile_store(self, store):
        store_file = os.path.join(self._app.output_folder, 'attempt_profiles.npz')
        names = list(store.keys())
        np.savez(store_file,
                 names=np.array(names),
                 vest=np.array([store[n]['vest'] for n in names]),
                 boat=np.array([store[n]['boat'] for n in names]))

    # === Глобальный реестр атлетов (~/.FreestyleParser, живёт между папками) ===

    def registry_file(self):
        return os.path.join(self._app._app_folder, 'athlete_registry.npz')

    def load_registry(self):
        """Реестр: {имя: {'vest': [векторы 1664], 'boat': [векторы 18]}}.
        Кешируется; пополняется подтверждениями попыток и сегментов."""
        if self._registry_cache is not None:
            return self._registry_cache
        reg = {}
        f = self.registry_file()
        if os.path.exists(f):
            try:
                d = np.load(f, allow_pickle=True)
                for i, name in enumerate(d['names']):
                    name = str(name)
                    reg.setdefault(name, {'vest': [], 'boat': []})
                    reg[name]['vest'].append(d['vest'][i])
                    reg[name]['boat'].append(d['boat'][i])
            except Exception as e:
                self._app.log(f"Ошибка загрузки реестра атлетов: {e}", logging.WARNING)
        self._registry_cache = reg
        return reg

    def save_registry(self, reg):
        f = self.registry_file()
        try:
            names, vests, boats = [], [], []
            for name, s in reg.items():
                for v, b in zip(s['vest'], s['boat']):
                    names.append(name)
                    vests.append(v)
                    boats.append(b)
            if names:
                np.savez(f, names=np.array(names),
                         vest=np.array(vests), boat=np.array(boats))
            elif os.path.exists(f):
                os.remove(f)
            self._registry_cache = reg
        except Exception as e:
            self._app.log(f"Ошибка сохранения реестра атлетов: {e}", logging.ERROR)

    def registry_add_sample(self, name, vest, boat):
        """Добавляет профиль подтверждённой попытки/сегмента в реестр."""
        import attempts_v2
        if vest is None or float(np.linalg.norm(vest[attempts_v2.VEST_DIM:])) < 1e-6:
            return
        reg = self.load_registry()
        s = reg.setdefault(name, {'vest': [], 'boat': []})
        s['vest'].append(np.asarray(vest, dtype=float))
        s['boat'].append(np.asarray(boat if boat is not None else np.zeros(18),
                                    dtype=float))
        if len(s['vest']) > self.REGISTRY_SAMPLES_MAX:
            s['vest'] = s['vest'][-self.REGISTRY_SAMPLES_MAX:]
            s['boat'] = s['boat'][-self.REGISTRY_SAMPLES_MAX:]
        self.save_registry(reg)

    def registry_banks(self):
        """Банки из реестра: СЫРЫЕ сэмплы по атлетам (списки векторов).
        Усреднение не делаем: сэмплы разных скинов (лето/зима, куртки)
        в среднем не совпадут ни с чем - сравнение идёт max по сэмплам."""
        banks = {}
        for name, s in self.load_registry().items():
            if not s['vest']:
                continue
            banks[name] = {
                'vest': [v / max(np.linalg.norm(v), 1e-9) for v in s['vest']],
                'boat': [b / max(b.sum(), 1e-9) for b in s['boat']]}
        return banks

    # === Сегменты скана атлетов (без нарезки) ===

    def segments_file(self):
        return os.path.join(self._app.output_folder, 'segments.yaml')

    def segment_store_file(self):
        return os.path.join(self._app.output_folder, 'segment_profiles.npz')

    def load_segments(self):
        self.athlete_segments = []
        self.segment_profiles = {}
        f = self.segments_file()
        if os.path.exists(f):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh) or {}
                self.athlete_segments = data.get('segments', [])
            except Exception as e:
                self._app.log(f"Ошибка загрузки segments: {e}", logging.WARNING)
        sf = self.segment_store_file()
        if os.path.exists(sf):
            try:
                d = np.load(sf, allow_pickle=True)
                self.segment_profiles = {
                    str(n): {'vest': d['vest'][i], 'boat': d['boat'][i]}
                    for i, n in enumerate(d['names'])}
            except Exception as e:
                self._app.log(f"Ошибка загрузки профилей сегментов: {e}", logging.WARNING)

    def save_segments(self):
        try:
            with open(self.segments_file(), 'w', encoding='utf-8') as fh:
                yaml.dump({'segments': self.athlete_segments}, fh,
                          allow_unicode=True)
        except Exception as e:
            self._app.log(f"Ошибка сохранения segments: {e}", logging.ERROR)

    def save_segment_profiles(self):
        names = list(self.segment_profiles)
        try:
            if names:
                np.savez(self.segment_store_file(),
                         names=np.array(names),
                         vest=np.array([self.segment_profiles[n]['vest']
                                        for n in names]),
                         boat=np.array([self.segment_profiles[n]['boat']
                                        for n in names]))
            elif os.path.exists(self.segment_store_file()):
                os.remove(self.segment_store_file())
        except Exception as e:
            self._app.log(f"Ошибка сохранения профилей сегментов: {e}", logging.ERROR)

    # === Банки и сравнение ===

    def base_banks(self):
        """Базовые банки ~/.FreestyleParser/banks/*.npz (read-only, формат
        как у registry.npz: names/vest/boat, строки-сэмплы). Возвращаем
        СЫРЫЕ сэмплы списками - сравнение max по сэмплам (скины/сезоны).
        Кешируется на сессию (банки read-only); вызывается на каждую плитку."""
        if self._base_banks_cache is not None:
            return self._base_banks_cache
        samples = {}
        bdir = os.path.join(self._app._app_folder, 'banks')
        if os.path.isdir(bdir):
            for fn in sorted(os.listdir(bdir)):
                if not fn.endswith('.npz'):
                    continue
                try:
                    d = np.load(os.path.join(bdir, fn), allow_pickle=True)
                    for i, n in enumerate(d['names']):
                        samples.setdefault(str(n), []).append(
                            (d['vest'][i], d['boat'][i]))
                except Exception as e:
                    self._app.log(f"Ошибка загрузки банка {fn}: {e}", logging.WARNING)
        banks = {}
        for name, s in samples.items():
            banks[name] = {
                'vest': [v / max(np.linalg.norm(v), 1e-9) for v, _ in s],
                'boat': [b / max(b.sum(), 1e-9) for _, b in s]}
        self._base_banks_cache = banks
        return banks

    def athlete_banks(self, store):
        """Банки атлетов по приоритету: подтверждённый маппинг папки >
        личный реестр > базовые банки. Локальные затирают базовые
        при совпадении имён."""
        banks = {}
        for athlete, files in self.athlete_mapping.items():
            entries = [store[os.path.basename(f)] for f in files
                       if os.path.basename(f) in store]
            if not entries:
                continue
            # каждая попытка дня - отдельный сэмпл банка
            banks[athlete] = {
                'vest': [e['vest'] / max(np.linalg.norm(e['vest']), 1e-9)
                         for e in entries],
                'boat': [e['boat'] / max(e['boat'].sum(), 1e-9)
                         for e in entries]}
        merged = dict(self.base_banks())
        merged.update(self.registry_banks())
        merged.update(banks)
        return merged

    def attempt_athlete_sim(self, profile, bank):
        """Похожесть попытки и банка: MAX по сэмплам банка (скины, сезоны,
        ракурсы). Сэмпл = OSNet+жилет косинус + совпадение бина лодки."""
        # errstate: spurious overflow/divide-by-zero warning от Accelerate BLAS
        # (macOS) на матмуле таких размеров - результат корректен, проверено
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            vests = bank['vest'] if isinstance(bank['vest'], list) \
                else [bank['vest']]
            boats = bank['boat'] if isinstance(bank['boat'], list) \
                else [bank['boat']]
            best, best_reid = -1.0, 0.0
            for v, b in zip(vests, boats):
                s_reid = float(profile['vest'] @ v)
                b_hist = profile['boat']
                if b_hist.sum() > 0:
                    bp = b_hist / b_hist.sum()
                    top = int(np.argmax(bp))
                    bt = int(np.argmax(b))
                    dist = min(abs(top - bt), 18 - abs(top - bt))
                    s_boat = 1.0 if dist <= 1 else 0.0
                    s = self.W_REID * s_reid + self.W_BOAT * s_boat
                else:
                    s = s_reid * self.W_REID / (self.W_REID + self.W_BOAT)
                if s > best:
                    best, best_reid = s, s_reid
            return best, best_reid

    # === Автоатрибуция ===

    def try_auto_assign(self, store, name, banks):
        """Уверенное авто-присвоение одной попытки (под lock вызывающего).
        True - присвоена (провизорная привязка сохранена сразу)."""
        sims = {a: self.attempt_athlete_sim(store[name], b)[0]
                for a, b in banks.items()}
        srt = sorted(sims.items(), key=lambda kv: -kv[1])
        best, best_sim = srt[0]
        second_sim = srt[1][1] if len(srt) >= 2 else 0.0
        if best_sim >= self.ATTEMPT_SIM_THRESHOLD and best_sim - second_sim >= self.ATTEMPT_SIM_MARGIN:
            self.attempt_assignments[name] = {'athlete': best,
                                              'sim': round(best_sim, 3),
                                              'second': round(second_sim, 3)}
            self.save_assignments()
            self._app.need_update_attempts = True  # плитки обновятся постепенно
            self._app.log(i18n.tr("  Автоатрибуция: {name} → {best} (sim={sim:.2f}) "
                                  "- подтвердите на сетке превью",
                                  name=name, best=best, sim=best_sim))
            return True
        hint = ', '.join(f'{a}={s:.2f}' for a, s in srt[:2])
        self._app.log(i18n.tr("  {name}: требует подтверждения ({hint})", name=name, hint=hint))
        return False

    def auto_assign_attempts(self, store, banks):
        """Автоприсвоение новых попыток: уверенные - в провизорные привязки
        (assignments.yaml, до подтверждения пользователем), спорные - в лог.
        Банки строятся только из подтверждённого маппинга, поэтому ошибка
        авто-привязки не отравляет дальнейшую атрибуцию.
        Вызывается в конце прогона и после каждого подтверждения."""
        if not banks:
            return
        with self._assignments_lock:
            assigned_names = {os.path.basename(f)
                              for files in self.athlete_mapping.values() for f in files}
            fresh = [n for n in store
                     if n not in assigned_names and n not in self.attempt_assignments]
            n_auto = 0
            for name in fresh:
                if self.try_auto_assign(store, name, banks):
                    n_auto += 1
        if fresh:
            self._app.log(i18n.tr("Автоатрибуция: предложено {n} из {total}", n=n_auto, total=len(fresh)))

    def refresh_auto_assignments(self, store=None):
        """Пересобирает банки (по подтверждённому маппингу) и немедленно
        перепредлагает авто-привязки - подтверждение учитывается сразу.
        store=None -> загрузить с диска; иначе используется живой store
        (вызов из on_attempt_found в рабочем потоке)."""
        try:
            if store is None:
                store = self.load_profile_store()
            banks = self.athlete_banks(store)
            if banks:
                self.auto_assign_attempts(store, banks)
        except Exception as e:
            self._app.log(i18n.tr("Ошибка пересчёта автоатрибуции: {err}", err=e), logging.ERROR)

    # === Маппинг атлетов (mapping.yaml) ===

    def load_athlete_mapping(self):
        """Загружает маппинг атлетов из файла"""
        mapping_file = os.path.join(self._app.output_folder, "mapping.yaml")
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = yaml.safe_load(f)
                    if mapping is None:  # файл пустой или только комментарии
                        self.athlete_mapping = {}
                    else:
                        self.athlete_mapping = {
                            self.repair_mojibake(name): list(set(attempts))
                            for name, attempts in mapping.items()
                        }
            except Exception as e:
                self._app.log(i18n.tr("Ошибка при загрузке маппинга: {err}", err=str(e)))
                self.athlete_mapping = {}
        else:
            self.athlete_mapping = {}
        self.prune_athlete_mapping()

    @staticmethod
    def repair_mojibake(name):
        """Чинит имена, испорченные чтением UTF-8 как cp1251 (Windows-сборки
        до v1.3.1 писали mapping.yaml в utf-8, а читали без encoding - кириллица
        превращалась в 'ÃÂ¿ÃÂ¾ÃÂ...'). Если обратное преобразование однозначно
        удается - возвращаем чиненое имя, иначе как есть."""
        if not isinstance(name, str) or all(ord(c) < 128 for c in name):
            return name
        for enc in ('cp1251', 'cp1252'):
            try:
                return name.encode(enc).decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return name

    def prune_athlete_mapping(self):
        """Убирает из маппинга привязки к несуществующим файлам.
        Пустые атлеты не удаляются (атлет мог быть только что создан)."""
        pruned = []
        for athlete in list(self.athlete_mapping.keys()):
            kept = [name for name in self.athlete_mapping[athlete]
                    if os.path.exists(os.path.join(self._app.output_folder, name))]
            pruned += [f"{athlete}: {name}"
                       for name in self.athlete_mapping[athlete] if name not in kept]
            self.athlete_mapping[athlete] = kept
        if pruned:
            self._app.log(i18n.tr("Убраны мёртвые привязки маппинга: {names}", names=', '.join(pruned)))
            self.save_athlete_mapping()
        return pruned

    def save_athlete_mapping(self):
        mapping_file = os.path.join(self._app.output_folder, "mapping.yaml")
        try:
            # Преобразуем абсолютные пути в относительные
            mapping = {
                name: list(set([self._app.get_relative_path(path) for path in attempts]))
                for name, attempts in self.athlete_mapping.items()
            }
            with open(mapping_file, 'w', encoding='utf-8') as f:
                yaml.dump(mapping, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self._app.log(i18n.tr("Ошибка при сохранении маппинга: {err}", err=str(e)))

    # === Провизорные авто-привязки (assignments.yaml) ===

    def load_assignments(self):
        """Загружает авто-привязки (провизорные, до подтверждения пользователем)."""
        f = os.path.join(self._app.output_folder, "assignments.yaml")
        if os.path.exists(f):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    self.attempt_assignments = yaml.safe_load(fh) or {}
            except Exception as e:
                self._app.log(i18n.tr("Ошибка загрузки assignments: {err}", err=e))
                self.attempt_assignments = {}
        else:
            self.attempt_assignments = {}

    def save_assignments(self):
        f = os.path.join(self._app.output_folder, "assignments.yaml")
        try:
            with open(f, 'w', encoding='utf-8') as fh:
                yaml.dump(self.attempt_assignments, fh,
                          default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self._app.log(i18n.tr("Ошибка сохранения assignments: {err}", err=e))
