#!/usr/bin/env python3
"""Чекер локализации: связка tr-ключей в коде <-> locales/en.json.

Ловит:
  1. Ключ есть в коде (i18n.tr("...")), но нет в en.json
     -> англ. пользователь увидит русский fallback (тихая потеря перевода).
  2. Ключ в en.json не используется ни в одном модуле
     -> мёртвый перевод (ключ переименовали в коде).
  3. Плейсхолдеры {name} в ключе и переводе не совпадают
     -> KeyError/молча неотформатированный шаблон в рантайме.
  4. Динамические ключи (переменные, MONTHS) проверяются отдельно:
     наличие известных "динамических" групп (месяцы, дни недели).

Выход: 0 если всё чисто, 1 если есть ошибки (для git-hook/CI).
Запуск: python3 tools/check_locales.py [--fix-dead]
"""

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_FILES = ["app.py", "Components.py", "utils.py", "splitter.py",
              "attempts_v2.py", "athlete_attribution.py"]
EN = os.path.join(ROOT, "locales", "en.json")

# Модули, в которых ищем tr-вызовы (относительно ROOT)
MODULES = ["i18n.py", "ui_tr.py"] + CODE_FILES

# Группы ключей, генерируемых в рантайме динамически (не видны AST)
DYNAMIC_GROUPS = {
    "месяцы": ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
               "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"],
    "дни недели": ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"],
}

# Ключи, используемые строковым доступом (table.get("...")) внутри i18n.py
WHITELIST = {"lang.name"}

PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")


def placeholders(s):
    return sorted(PLACEHOLDER_RE.findall(s))


def is_cyrillic(s):
    return any("\u0400" <= c <= "\u04FF" for c in s)


def extract_keys(path):
    """Все константные ключи i18n.tr("...") из файла (AST)."""
    keys = set()
    all_strs = set()
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and is_cyrillic(node.value):
            # строковые литералы тоже "использование": ключи могут лежать
            # в таблицах (STRATEGY_MODES и т.п.) и переводиться в рантайме
            all_strs.add(node.value)
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tr"):
            continue
        if not node.args:
            continue
        try:
            key = ast.literal_eval(node.args[0])
        except (ValueError, SyntaxError):
            continue  # динамический ключ (переменная) - обрабатываем группами
        if isinstance(key, str):
            keys.add(key)  # любые строковые ключи (в т.ч. без кириллицы)
    return keys, all_strs


def main():
    errors, warnings = [], []
    with open(EN, encoding="utf-8") as f:
        en = json.load(f)

    used = set()
    literals = set()
    for mod in CODE_FILES:
        path = os.path.join(ROOT, mod)
        ks, strs = extract_keys(path)
        used |= ks
        literals |= strs

    # 1. В коде, но нет перевода
    missing = used - set(en)
    for k in sorted(missing):
        errors.append(f"нет перевода в en.json: {k!r}")

    # 2. Мёртвые ключи (в en.json, но не в коде и не динамические)
    dynamic = set()
    for group in DYNAMIC_GROUPS.values():
        dynamic |= set(group)
    dead = set(en) - used - literals - dynamic - WHITELIST
    for k in sorted(dead):
        warnings.append(f"ключ в en.json не используется: {k!r}")

    # 3. Плейсхолдеры
    for k in sorted(used & set(en)):
        ph_key, ph_en = placeholders(k), placeholders(en[k])
        if ph_key != ph_en:
            errors.append(
                f"плейсхолдеры не совпадают: {k!r}: ключ={ph_key}, en={ph_en}")

    # 4. Динамические группы: хотя бы частично присутствуют
    for gname, group in DYNAMIC_GROUPS.items():
        absent = [k for k in group if k not in en]
        if absent:
            errors.append(f"динамическая группа '{gname}': нет в en.json: {absent}")

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print(f"\nитого: ключей в en.json={len(en)}, в коде={len(used)}, "
          f"ошибок={len(errors)}, предупреждений={len(warnings)}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
