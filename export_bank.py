#!/usr/bin/env python
"""Одноразовый экспорт базового банка из папки дня.

Собирает профили попыток (attempt_profiles.npz) по привязкам из mapping.yaml
и сохраняет их как read-only базовый банк в ~/.FreestyleParser/banks/<ИМЯ>.npz
(формат идентичен athlete_registry.npz: names / vest / boat - по одному
сырому сэмплу на попытку, усреднение не выполняется).

Запуск:
    env/bin/python export_bank.py /путь/к/папке/дня RussianCup2026
"""
import argparse
import os
import sys

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser(
        description="Экспорт базового банка атлетов из папки дня")
    ap.add_argument("day_folder", help="папка дня с mapping.yaml и "
                                        "attempt_profiles.npz")
    ap.add_argument("bank_name", help="имя банка, например RussianCup2026")
    args = ap.parse_args()

    day = os.path.abspath(os.path.expanduser(args.day_folder))
    profiles_file = os.path.join(day, "attempt_profiles.npz")
    mapping_file = os.path.join(day, "mapping.yaml")
    for f in (profiles_file, mapping_file):
        if not os.path.exists(f):
            sys.exit(f"Не найден обязательный файл: {f}")

    d = np.load(profiles_file, allow_pickle=True)
    profiles = {str(n): (d["vest"][i], d["boat"][i])
                for i, n in enumerate(d["names"])}

    with open(mapping_file) as fh:
        mapping = yaml.safe_load(fh) or {}

    names, vests, boats = [], [], []
    skipped = []
    for athlete, clips in mapping.items():
        if not isinstance(clips, (list, tuple)):
            clips = [clips]
        for clip in clips:
            base = os.path.basename(str(clip))
            prof = profiles.get(base)
            if prof is None:
                skipped.append(f"{athlete}: {base} (нет профиля)")
                continue
            vest, boat = prof
            if float(np.linalg.norm(vest)) < 1e-6:
                skipped.append(f"{athlete}: {base} (пустой вектор)")
                continue
            names.append(athlete)
            vests.append(np.asarray(vest, dtype=np.float64))
            boats.append(np.asarray(boat if boat is not None
                                    else np.zeros(18), dtype=np.float64))

    if not names:
        sys.exit("Нет ни одного профиля для экспорта")

    home = os.path.expanduser("~")
    banks_dir = os.path.join(home, ".FreestyleParser", "banks")
    os.makedirs(banks_dir, exist_ok=True)
    out = os.path.join(banks_dir, f"{args.bank_name}.npz")
    np.savez(out, names=np.array(names),
             vest=np.array(vests), boat=np.array(boats))

    athletes = sorted(set(names))
    print(f"Банк сохранён: {out}")
    print(f"Атлетов: {len(athletes)}, сэмплов: {len(names)}")
    by_ath = {}
    for n in names:
        by_ath[n] = by_ath.get(n, 0) + 1
    for a in athletes:
        print(f"  {a}: {by_ath[a]}")
    if skipped:
        print(f"Пропущено ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
