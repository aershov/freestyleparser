"""Локализация: tr() + загрузка словарей из locales/.

Языки: JSON-файлы locales/<code>.json (ru.json - базовый, en.json - перевод).
Порядок поиска папки locales/ (первый найденный побеждает):
  1. $FREESTYLE_LOCALE_DIR  (для тестов/отладки)
  2. ~/.FreestyleParser/locales  (пользовательские/исправленные переводы)
  3. рядом с исходником или в бандле (sys._MEIPASS, как help.html)
Недостающие ключи - fallback на ru, затем сам ключ.
"""

import json
import logging
import os
import sys

log = logging.getLogger(__name__)

_BASE_LANGUAGE = "ru"
_languages = {}          # code -> {key: str}
_current = _BASE_LANGUAGE
_listeners = []          # колбэки при смене языка (для перерисовки UI)


def _locale_dirs():
    dirs = []
    env = os.environ.get("FREESTYLE_LOCALE_DIR")
    if env:
        dirs.append(env)
    home = os.path.expanduser("~/.FreestyleParser/locales")
    dirs.append(home)
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    dirs.append(os.path.join(base, "locales"))
    dirs.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales"))
    return dirs


def _load_all():
    global _languages
    _languages = {}
    for d in _locale_dirs():
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            code = fn[:-5]
            try:
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Пользовательские словари (home) перекрывают бандл
                    _languages.setdefault(code, {}).update(
                        {k: v for k, v in data.items() if isinstance(v, str)})
            except Exception as e:
                log.warning("Не удалось загрузить %s: %s", fn, e)


def available_languages():
    """[(код, название)], например [('ru', 'Русский'), ('en', 'English')]"""
    names = {}
    for code, table in _languages.items():
        name = table.get("lang.name", code)
        names[code] = name
    return sorted(names.items())


def get_language():
    return _current


def set_language(code, notify=True):
    global _current
    if code not in _languages:
        log.warning("Язык %r недоступен, остаёмся на %r", code, _current)
        return
    _current = code
    if notify:
        for cb in list(_listeners):
            try:
                cb()
            except Exception:
                log.exception("Ошибка в on_language_changed")


def on_language_changed(cb):
    """Подписка на смену языка (перерисовка UI)."""
    _listeners.append(cb)


class TrStr(str):
    """Строка-результат tr(): обычный str + исходный ключ и аргументы.
    ui_tr запоминает их у виджетов для перерисовки при смене языка."""

    def __new__(cls, value, key=None, args=None):
        obj = super().__new__(cls, value)
        obj.key = key if key is not None else value
        obj.args = dict(args) if args else {}
        return obj


def tr(key, **kwargs):
    """Перевод по ключу: tr('btn.save'). Плейсхолдеры {named} через str.format."""
    table = _languages.get(_current, {})
    s = table.get(key)
    if s is None:
        s = _languages.get(_BASE_LANGUAGE, {}).get(key)
    if s is None:
        s = key
    if kwargs:
        try:
            s = s.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return TrStr(s, key=key, args=kwargs)


_load_all()
