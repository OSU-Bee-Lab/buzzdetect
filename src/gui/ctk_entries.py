import tkinter as tk
from idlelib.tooltip import Hovertip
from tkinter import filedialog as filedialog

import customtkinter as ctk
from src.validation import ArgValid

from src.gui import config as cfg_gui


class AbstractEntry(ctk.CTkFrame):
    def __init__(self, master, label, var, tooltip=None, validation_function=None):
        super().__init__(master)
        self.label = ctk.CTkLabel(self, text=label, font=cfg_gui.font_textentry)
        self.name = label
        self.var = var
        self.invalidateframe = ctk.CTkFrame(self)
        self.validation_function = validation_function
        self.label_invalid = ctk.CTkLabel(self.invalidateframe, font=cfg_gui.font_hint, height=2, justify='left')
        self.valid = True

        if self.validation_function is None:
            self.validate = None
        else:
            self.validate='focusout'

        if tooltip:
            self.label.configure(text=f"{label}  ?⃝")
            Hovertip(
                self.label,
                tooltip, hover_delay=cfg_gui.hoverdelay,
                background='white'
            )

        self.label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.grid_columnconfigure(1, weight=1)


    def _validate_and_warn(self):
        argvalid: ArgValid = self.validation_function(self.var.get())

        if argvalid.message is None:
            self.invalidateframe.grid_forget()
            self.label_invalid.grid_forget()
        else:
            textcolor = 'darkorange' if argvalid.valid else 'darkred'
            self.label_invalid.configure(text=argvalid.message, text_color=textcolor)
            self.invalidateframe.grid(row=1, column=0, padx=5, pady=5, sticky="ew", columnspan=2)
            self.label_invalid.grid(row=0, column=0, padx=5, pady=5, sticky="ew", columnspan=2)

        self.argvalid = argvalid
        return argvalid.valid


class TextEntry(AbstractEntry):
    def __init__(self, master, label, var, tooltip=None, validation_function=None):
        super().__init__(master, label, var, tooltip, validation_function)
        self.entry = ctk.CTkEntry(self, textvariable=self.var, validate=self.validate, validatecommand=self._validate_and_warn)
        self.entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")



class FilePathEntry(TextEntry):
    def __init__(self, master, label, var, tooltip=None, validation_function=None, initialdir=None, browsetitle=None):
        super().__init__(master, label, var, tooltip, validation_function)
        self.var = var if var is not None else tk.StringVar()
        self.initialdir = initialdir
        self.browsetitle = browsetitle
        self.entry.configure(validatecommand=self._validate_and_warn)
        self.browse = ctk.CTkButton(self, text="📁", command=self._browse, width=10, height=5)
        self.browse.grid(row=0, column=2, padx=5, pady=0, sticky="w")

    def _browse(self):
        dir_selected = filedialog.askdirectory(initialdir=self.initialdir, title=self.browsetitle)
        if dir_selected == '':  # window was closed; do nothing
            return

        self.initialdir = dir_selected  # keep "memory" of initial dir
        self.var.set(dir_selected)


class DropDownEntry(AbstractEntry):
    def __init__(self, master, label, var, values, tooltip=None, command=None):
        super().__init__(master, label, var, tooltip)
        self.var = var if var is not None else tk.StringVar()

        self.dropdown = ctk.CTkOptionMenu(self, variable=self.var, values=values)
        if command is not None:
            self.dropdown.configure(command=command)

        self.dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

class CheckBoxEntry(AbstractEntry):
    def __init__(self, master, label, var, tooltip=None):
        super().__init__(master, label, var, tooltip)
        self.var = var if var is not None else tk.BooleanVar()

        self.checkbox = ctk.CTkCheckBox(self, text="", variable=self.var)
        self.checkbox.grid(row=0, column=1, padx=5, pady=5, sticky="w")