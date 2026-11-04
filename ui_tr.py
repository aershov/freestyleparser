"""Рантайм-смена языка UI: запоминаем у виджетов tr-ключи и перерисовываем.

Как работает: i18n.tr() возвращает TrStr (str + исходный ключ + аргументы).
install() патчит tk.Widget.__init__/config и Wm.title: любой text=/title=,
получивший TrStr, запоминает ключ у виджета (_tr_key/_tr_title).
set_language -> колбэк -> retext(root) обходит дерево виджетов и заново
ставит text=i18n.tr(ключ). Динамические тексты (лог, статусы) обновятся
сами при следующем событии - их перерисовывать не нужно.

В титлбар ОС Tkinter виджеты не помещает - поэтому переключатель языка
живёт в углу окна, а сам язык меняется без перезапуска.
"""

import tkinter as tk

import i18n

_installed = False


def _remember(widget, text):
    if isinstance(text, i18n.TrStr):
        widget._tr_key = text.key
        widget._tr_args = dict(text.args)
    else:
        # Динамический/нетр-текст: маркер снимаем (не перерисовывать)
        widget.__dict__.pop('_tr_key', None)
        widget.__dict__.pop('_tr_args', None)


def install():
    """Патчим Widget и Wm один раз при старте приложения."""
    global _installed
    if _installed:
        return
    _installed = True

    _orig_init = tk.Widget.__init__

    def _init(self, *args, **kwargs):
        # tk.Label: (master, cnf, **kw); ttk.Label: (master, name, cnf, kw)
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, dict) and 'text' in a:
                _remember(self, a['text'])
        if 'text' in kwargs and not isinstance(kwargs['text'], dict):
            _remember(self, kwargs['text'])
        return _orig_init(self, *args, **kwargs)

    tk.Widget.__init__ = _init

    _orig_config = tk.Widget.config

    def _config(self, cnf=None, **kw):
        if isinstance(cnf, dict):
            if 'text' in cnf:
                _remember(self, cnf['text'])
        if 'text' in kw:
            _remember(self, kw['text'])
        return _orig_config(self, cnf, **kw)

    tk.Widget.config = _config
    tk.Widget.configure = _config  # configure - отдельный атрибут, не алиас

    _orig_title = tk.Wm.title

    def _title(self, string=None):
        if isinstance(string, i18n.TrStr):
            self._tr_title = string.key
            self._tr_title_args = dict(string.args)
        return _orig_title(self, string)

    tk.Wm.title = _title


def retext(widget):
    """Перерисовка: обход дерева, text=/title= по запомненным ключам."""
    key = getattr(widget, '_tr_key', None)
    if key is not None:
        try:
            widget.config(text=i18n.tr(key, **getattr(widget, '_tr_args', {})))
        except tk.TclError:
            pass  # виджет уже уничтожен
    tkey = getattr(widget, '_tr_title', None)
    if tkey is not None:
        try:
            widget.title(i18n.tr(tkey, **getattr(widget, '_tr_title_args', {})))
        except tk.TclError:
            pass
    try:
        children = widget.winfo_children()
    except tk.TclError:
        return
    for child in children:
        retext(child)
