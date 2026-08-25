import inspect
import json
import os

import customtkinter as ctk
import src.validation as val
from src.analyze import analyze
from src.pipeline.loglevels import loglevels
from src.pipeline.manifest import read_manifest

import src.config as cfg
import src.gui.config as cfg_gui
import src.gui.ctk_entries as ent


def analysis_defaults():
    """
    Return default values for analysis parameters.
    If a config file exists, use those values. Otherwise, use the default values from analyze().
    """
    if os.path.exists(cfg_gui.path_settingscache):
        with open(cfg_gui.path_settingscache, 'r') as f:
            arguments = json.load(f)
    else:
        signature = inspect.signature(analyze)
        arguments = {
            k: v.default if v.default is not inspect.Parameter.empty else None
            for k, v in signature.parameters.items()
        }

    return arguments


def save_settings(vars_analysis):
    with open(cfg_gui.path_settingscache, 'w') as f:
        json.dump(vars_analysis, f, indent=2)

class AnalysisSettings(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.run = False
        self._locked_manifest = None
        self.vars_analysis = analysis_defaults()

        # drop q_gui and event_stopanalysis; handled by analysis window
        self.vars_analysis.pop('q_gui', None)
        self.vars_analysis.pop('event_stopanalysis', None)

        self.vars_tkinter = {
            'modelname': ctk.StringVar(self, ''),  # will be loaded at end of init; ctk coerces None to '' so just set to ''
            'precision': ctk.StringVar(self, self.vars_analysis['precision']),
            'framehop_prop': ctk.StringVar(self, self.vars_analysis['framehop_prop']),
            'chunklength': ctk.StringVar(self, self.vars_analysis['chunklength']),
            'analyzers_cpu': ctk.StringVar(self, self.vars_analysis['analyzers_cpu']),
            'analyzers_gpu': ctk.StringVar(self, self.vars_analysis['analyzers_gpu']),
            'n_streamers': ctk.StringVar(self, self.vars_analysis['n_streamers']),
            'stream_buffer_depth': ctk.StringVar(self, self.vars_analysis['stream_buffer_depth']),
            'dir_audio': ctk.StringVar(self, self.vars_analysis['dir_audio']),
            'dir_out': ctk.StringVar(self, self.vars_analysis['dir_out']),
            'verbosity_print': ctk.StringVar(self, self.vars_analysis['verbosity_print']),
            'verbosity_log': ctk.StringVar(self, self.vars_analysis['verbosity_log']),
            'log_progress': ctk.BooleanVar(self, self.vars_analysis['log_progress'])
        }


        self.title("buzzdetect")
        self.geometry("700x600")
        self.update_idletasks()
        self.minsize(self.winfo_width(), 400)


        # FRAME: main
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.frame_main = ctk.CTkScrollableFrame(self)
        self.frame_main.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_main.grid_columnconfigure(0, weight=1)
        self.frame_main.grid_rowconfigure(0, weight=1)

        # allow clicking on main frame to defocus widgets (lets you click out of text boxes, thereby validating them)
        self.frame_main.bind("<Button-1>", lambda e: self.frame_main.focus_set())


        # FRAME: Model
        self.frame_model = ctk.CTkFrame(self.frame_main)
        self.frame_model.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.frame_model.grid_columnconfigure(0, weight=1)

        self.available_models = []
        ctk.CTkLabel(self.frame_model, text="Model settings", font=cfg_gui.font_textheader).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_optionmenu = ent.DropDownEntry(
            master=self.frame_model, label='Model', var=self.vars_tkinter['modelname'],
            tooltip='Select a model to use for analysis.', command= lambda _: self._model_selected(),
            values=self.available_models
        )

        self.model_optionmenu.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.frame_output = ctk.CTkFrame(self.frame_model)
        self.frame_output.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.frame_output.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.frame_output, text="Output Format", font=cfg_gui.font_textheader).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # shown when the chosen output folder already has results: schema-defining
        # controls are locked to match, so resumed runs stay compatible.
        self.label_lock = ctk.CTkLabel(self.frame_output, text='', text_color='darkorange', wraplength=400, justify='left')

        self.tabview_format = ctk.CTkTabview(self.frame_output)
        self.tabview_format.add('Activations')
        self.tabview_format.add('Detections')
        self.tabview_format.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.tabview_format.tab("Activations").grid_columnconfigure(0, weight=1)
        self.tabview_format.tab("Detections").grid_columnconfigure(0, weight=1)

        self.frame_neurons = ctk.CTkFrame(self.tabview_format.tab("Activations"))
        self.frame_neurons.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.frame_neurons.grid_columnconfigure(0, weight=1)

        self.frame_neuron_checkboxes = ctk.CTkFrame(self.frame_neurons)
        self.frame_neuron_checkboxes.grid(row=0, column=0, sticky="ew")

        for i in range(cfg_gui.cols_neurons):
            self.frame_neuron_checkboxes.grid_columnconfigure(i, weight=1)

        self.neuron_checkboxes = {}  # class -> BooleanVar

        self.button_toggle_neurons = ctk.CTkButton(self.frame_neurons, text="Select All/None", command=lambda: self._toggle_neurons())
        self.button_toggle_neurons.grid(row=1, column=0, padx=2, sticky="ew")

        self.entry_precision = ent.TextEntry(
            master=self.tabview_format.tab('Detections'), label='Precision', var=self.vars_tkinter['precision'],
            tooltip='Precision with which to call detections, between 0 and 1.\nA value equal to or above 0.95 is recommended.',
            validation_function=lambda v: val.validate_precision(v if v != '' else None),
        )
        self.entry_precision.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # FRAME: Input/Output
        self.frame_io = ctk.CTkFrame(self.frame_main)
        self.frame_io.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.frame_io.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.frame_io, text="Input/Output Folders", font=cfg_gui.font_textheader).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # initial dirs will be buzzdetect root; assuming users don't want to start in the default folder
        # if they're changing the folder
        self.entry_dir_audio = ent.FilePathEntry(
            master=self.frame_io, label='Input Folder', var=self.vars_tkinter['dir_audio'],
            tooltip='Input folder containing audio files to analyze.', validation_function=val.validate_dir_audio,
            initialdir='.', browsetitle='Select input audio folder'
        )
        self.entry_dir_audio.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.entry_dir_out = ent.FilePathEntry(
            master=self.frame_io, label='Output Folder', var=self.vars_tkinter['dir_out'],
            tooltip='Output folder for analysis results.', validation_function=val.validate_dir_out,
            initialdir='.', browsetitle='Select output results folder'
        )
        self.entry_dir_out.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # FRAME: analysis parameters
        self.analysis_params_frame = ctk.CTkFrame(self.frame_main)
        self.analysis_params_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.analysis_params_frame.grid_columnconfigure(0, weight=1)

        # cpus
        self.entry_analyzers_cpu = ent.TextEntry(
            master=self.analysis_params_frame, label='CPU Analyzers', var=self.vars_tkinter['analyzers_cpu'],
            tooltip="The number of CPU-bazed workers to launch.\nUsually, 1 worker will efficiently use your system's resources, but try adding more.",
            validation_function=val.validate_analyzers_cpu,
        )
        self.entry_analyzers_cpu.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # gpu
        self.entry_analyzers_gpu = ent.TextEntry(
            master=self.analysis_params_frame, label='GPU Analyzers', var=self.vars_tkinter['analyzers_gpu'],
            tooltip="The number of GPU-bazed workers to launch.\nIf you're using GPU, you probably don't want any CPU analyzers.",
            validation_function=val.validate_analyzers_cpu,
        )
        self.entry_analyzers_gpu.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        # chunklength
        self.entry_chunklength = ent.TextEntry(
            master=self.analysis_params_frame, label='Chunk length', var=self.vars_tkinter['chunklength'],
            tooltip="The length of each chunk in seconds.",
            validation_function=val.validate_chunklength,
        )
        self.entry_chunklength.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.advanced_settings_button = ctk.CTkButton(
            self.analysis_params_frame,
            text="Open Advanced Settings",
            command=self._open_advanced_settings_window
        )
        self.advanced_settings_button.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        # Run button
        runframe = ctk.CTkFrame(self)
        runframe.grid(row=5, column=0, padx=0, pady=0, sticky="ew")
        runframe.grid_columnconfigure(0, weight=1)
        self.run_button = ctk.CTkButton(runframe, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=6, column=0, padx=20, pady=10, sticky="ew", columnspan=1)

        # ---- initializing values ----
        # Initial loading of models and classes
        self._load_available_models()
        saved_model = self.vars_analysis.get('modelname')
        if saved_model and saved_model in self.available_models:
            self.vars_tkinter['modelname'].set(saved_model)
        elif cfg.DEFAULT_MODEL in self.available_models:
            self.vars_tkinter['modelname'].set(cfg.DEFAULT_MODEL)
        else:
            self.vars_tkinter['modelname'].set(self.available_models[0])
        self._model_selected()
        self._load_neurons()

        # restore saved output mode (manifest will override if applicable)
        if self.vars_analysis.get('precision') is not None:
            self.tabview_format.set('Detections')

        # restore saved dir_out; _model_selected() overwrites it with the model default
        saved_dir_out = self.vars_analysis.get('dir_out')
        if saved_dir_out:
            self.vars_tkinter['dir_out'].set(saved_dir_out)

        # lock schema-defining controls if the output folder already has results,
        # and re-evaluate whenever the output folder changes
        self.vars_tkinter['dir_out'].trace_add('write', self._apply_manifest_lock)
        self._apply_manifest_lock()

    def _apply_manifest_lock(self, *_):
        """Read the manifest of the selected output folder (if any) and lock the
        controls that determine result schema to match it, so a resumed run can't
        write incompatible results into the folder."""
        dir_out = self.vars_tkinter['dir_out'].get()
        manifest = read_manifest(dir_out) if dir_out else None
        self._locked_manifest = manifest

        if manifest is None:
            self._set_schema_locked(False)
            self.label_lock.grid_forget()
            return

        self._apply_manifest_values(manifest)
        self._set_schema_locked(True)
        self.label_lock.configure(
            text="Results have already been written to this output folder."
                 "Model, output format, neurons, precision, and framehop are locked to match existing results."
                 "Choose a different output folder if you would like to run an analysis with different settings."
        )
        self.label_lock.grid(row=2, column=0, padx=5, pady=(0, 5), sticky="w")

    def _apply_manifest_values(self, manifest):
        """Force the controls to the manifest's values (before disabling them)."""
        model = manifest.get('modelname')
        if model and model in self.available_models and self.vars_tkinter['modelname'].get() != model:
            self.vars_tkinter['modelname'].set(model)
            self._load_neurons()

        if manifest.get('output_mode') == 'detections':
            self.tabview_format.set('Detections')
            precision = manifest.get('precision')
            self.vars_tkinter['precision'].set('' if precision is None else precision)
        else:
            self.tabview_format.set('Activations')
            locked_classes = set(manifest.get('classes_out') or [])
            for cls, var in self.neuron_checkboxes.items():
                var.set(cls in locked_classes)

        framehop = manifest.get('framehop_prop')
        if framehop is not None:
            self.vars_tkinter['framehop_prop'].set(framehop)

    def _set_schema_locked(self, locked: bool):
        state = 'disabled' if locked else 'normal'
        self.model_optionmenu.dropdown.configure(state=state)
        # gray the checked-box fill when locked so it doesn't read as still-active
        fg_color = 'gray50' if locked else ctk.ThemeManager.theme["CTkCheckBox"]["fg_color"]
        for widget in self.frame_neuron_checkboxes.winfo_children():
            widget.configure(state=state, fg_color=fg_color)
        self.button_toggle_neurons.configure(state=state)
        self.entry_precision.entry.configure(state=state)
        # disable switching between Activations/Detections tabs
        self.tabview_format._segmented_button.configure(state=state)

    def _open_advanced_settings_window(self):
        """Opens the Advanced Settings in a new window."""
        adv_window = AdvancedSettings(self, vars_analysis=self.vars_analysis, vars_tkinter= self.vars_tkinter)
        self.wait_window(adv_window)

    def _load_available_models(self):
        """Populates the model dropdown with available model directories."""
        self.available_models = [d for d in os.listdir(cfg.DIR_MODELS) if os.path.exists(os.path.join(cfg.DIR_MODELS, d, 'model.py'))]
        if not self.available_models:
            self._display_error(f"No valid models found in {cfg.DIR_MODELS}")
            self.model_optionmenu.dropdown.configure(values=[""])
        else:
            self.model_optionmenu.dropdown.configure(values=self.available_models)

    def _model_selected(self):
        """Called when a new model is selected in the dropdown."""
        modelname = self.vars_tkinter["modelname"].get()
        if modelname == '':
            return

        self._load_neurons()
        self.vars_tkinter['dir_out'].set(os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_OUTPUT))

    def _load_neurons(self):
        """Loads classes from config_model.json for the selected model and updates checkboxes."""
        # cache selected neurons; on first call (empty dict), fall back to saved settings
        if self.neuron_checkboxes:
            neurons_selected = [k for k, v in self.neuron_checkboxes.items() if v.get()]
        else:
            neurons_selected = self.vars_analysis.get('classes_out') or []

        # clear existing checkboxes
        for widget in self.frame_neuron_checkboxes.winfo_children():
            widget.destroy()
        self.neuron_checkboxes.clear()
        modelname = self.vars_tkinter['modelname'].get()
        if modelname == '':
            return

        config_path = os.path.join(cfg.DIR_MODELS, modelname, "config_model.json")
        if not os.path.exists(config_path):
            self._display_error(f"config_model.json not found for model: {modelname}")
            return

        with open(config_path, 'r') as f:
            config_model = json.load(f)
        available_classes = config_model.get('classes', [])
        if not available_classes:
            self._display_error(f"No 'classes' found in config_model.json for {modelname}")
            return

        available_classes.sort()

        if neurons_selected:
            values = [a in neurons_selected for a in available_classes]
        else:
            values = [True] * len(available_classes)

        for i, cls in enumerate(available_classes):
            var = ctk.BooleanVar(value=values[i])
            chk = ctk.CTkCheckBox(self.frame_neuron_checkboxes, text=cls, variable=var)
            row = i // cfg_gui.cols_neurons
            column = i % cfg_gui.cols_neurons
            chk.grid(row=row, column=column, padx=5, pady=2, sticky="w")
            self.neuron_checkboxes[cls] = var

    def _display_error(self, message):
        error_window = ctk.CTkToplevel(self)
        error_window.title("Error")
        error_window.geometry("400x100")
        error_window.transient(self)
        error_window.grab_set()
        error_window.grid_columnconfigure(1, weight=1)
        self.error_label = ctk.CTkLabel(error_window, text=message, text_color="red", wraplength=400)
        self.error_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

    def _toggle_neurons(self):
        """Selects or deselects all class checkboxes."""
        prop_selected = sum([v.get() for v in self.neuron_checkboxes.values()]) / len(self.neuron_checkboxes)
        if prop_selected < 1:
            value_to = True
        else:
            value_to = False

        for var in self.neuron_checkboxes.values():
            var.set(value_to)


    def _advanced_settings(self):
        """Opens the Advanced Settings in a new window."""
        adv_window = AdvancedSettings(self, self.vars_analysis, self.vars_tkinter)
        self.wait_window(adv_window)

    def _update_vars_analysis(self):
        vars_skip = ['classes_out', 'precision']
        for k, v in self.vars_tkinter.items():
            if k in vars_skip:
                continue

            self.vars_analysis[k] = v.get()

        # build skipped vars
        mode = self.tabview_format.get()
        if mode == 'Activations':
            self.vars_analysis['precision'] = None
        elif mode == 'Detections':
            val_precision = self.vars_tkinter['precision'].get()
            self.vars_analysis['precision'] = float(val_precision) if val_precision != '' else None
        else:
            raise ValueError(f"Invalid tabview mode: {mode}")

        self.vars_analysis['classes_out'] = [k for k, v in self.neuron_checkboxes.items() if v.get()]

        # interpret non-strings to their appropriate types
        self.vars_analysis['framehop_prop'] = float(self.vars_analysis['framehop_prop'])
        self.vars_analysis['chunklength'] = int(self.vars_analysis['chunklength'])
        self.vars_analysis['analyzers_cpu'] = int(self.vars_analysis['analyzers_cpu'])
        self.vars_analysis['analyzers_gpu'] = int(self.vars_analysis['analyzers_gpu'])
        self.vars_analysis['n_streamers'] = int(self.vars_analysis['n_streamers']) if self.vars_analysis['n_streamers'] != '' else None
        self.vars_analysis['stream_buffer_depth'] = int(self.vars_analysis['stream_buffer_depth']) if self.vars_analysis['stream_buffer_depth'] != '' else None

        self.vars_analysis = self.vars_analysis

    def _validate(self):
        entries = [self.entry_dir_audio, self.entry_dir_out, self.entry_analyzers_cpu, self.entry_analyzers_gpu, self.entry_chunklength]
        issues = [f"{entry.name}: {entry.argvalid.message}" for entry in entries if not entry.argvalid.valid]

        if not issues:
            return True
        else:
            self._display_error(f"Invalid settings!\n\n{'\n'.join(issues)}")
            return False

    def run_analysis(self):
        # force defocus any widget that the user is still clicked into to proc validation
        focused = self.focus_get()
        if focused and hasattr(focused, 'master'):
            # Trigger focusout event on the focused widget
            focused.event_generate('<FocusOut>')
            self.update_idletasks()
        if self._validate():
            self._update_vars_analysis()
            save_settings(self.vars_analysis)
            self.run = True
            self.destroy()


# --- Advanced Settings Window ---
class AdvancedSettings(ctk.CTkToplevel):
    def __init__(self, master: AnalysisSettings, vars_analysis, vars_tkinter):
        super().__init__(master)

        self.vars_analysis = vars_analysis
        self.vars_tkinter = vars_tkinter

        self.title("Advanced Settings")
        self.transient(master)
        self.grab_set()
        self.grid_columnconfigure(0, weight=1)

        # framehop_prop
        self.entry_framehop = ent.TextEntry(
            self, label='Framehop', var=self.vars_tkinter['framehop_prop'],
            tooltip='The spacing between frames, expressed as a proportion of the frame length.\n'
                    'E.g., a framehop of 1 produces contiguous frames, 0.50 produces frames with 50% overlap.',
            validation_function=val.validate_framehop
        )
        self.entry_framehop.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # framehop defines result schema; lock it if the output folder is locked
        if getattr(master, '_locked_manifest', None) is not None:
            self.entry_framehop.entry.configure(state='disabled')
            ctk.CTkLabel(
                self, text="Framehop is locked to match that of the existing results. Choose a new output folder to change framehop.",
                text_color='darkorange', wraplength=400, justify='left'
            ).grid(row=1, column=0, padx=5, pady=(0, 5), sticky="w")

        # concurrent_streamers
        self.entry_n_streamers = ent.TextEntry(
            self, label='Concurrent streamers', var=self.vars_tkinter['n_streamers'],
            tooltip='How many parallel audio streamers should be launched?\nIf you run into buffer bottlenecks, try increasing this number.\nLeave blank for automatic assignment.',
            validation_function=lambda v: val.validate_n_streamers(v if v != '' else None)  # ctk.StringVar can't hold None, coerces to ''
        )
        self.entry_n_streamers.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # stream_buffer_depth
        self.entry_stream_buffer_depth = ent.TextEntry(
            self, label='Stream Buffer Depth', var=self.vars_tkinter['stream_buffer_depth'],
            tooltip='How many audio chunks should be buffered in memory?\nLeave blank for automatic assignment.',
            validation_function=lambda v: val.validate_stream_buffer_depth(v if v != '' else None) # ctk.StringVar can't hold None, coerces to ''
        )
        self.entry_stream_buffer_depth.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # console Verbosity
        ent.DropDownEntry(
            self, label='Console Verbosity', var=self.vars_tkinter['verbosity_print'],
            tooltip='How verbose should the console output be?', values=list(loglevels.keys())
        ).grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        ent.DropDownEntry(
            self, label='Log File Verbosity', var=self.vars_tkinter['verbosity_log'],
            tooltip='How verbose should the log file output be?', values=list(loglevels.keys())
        ).grid(row=5, column=0, padx=5, pady=5, sticky="ew")

        ent.CheckBoxEntry(
            self, label='Log progress to file', var=self.vars_tkinter['log_progress'],
            tooltip='Should progress statements (e.g., reports from analyzers)\nbe written to the log file?\nCan produce very large log files.'
        ).grid(row=6, column=0, padx=5, pady=5, sticky="ew")

        # Close
        ctk.CTkButton(self, text="Close", command=self._close).grid(row=7, column=0, columnspan=2, pady=10)
        self.protocol("WM_DELETE_WINDOW", self._close)


    def _display_error(self, message):
        error_window = ctk.CTkToplevel(self)
        error_window.title("Error")
        error_window.geometry("400x100")
        error_window.transient(self)
        error_window.grab_set()
        error_window.grid_columnconfigure(1, weight=1)
        self.error_label = ctk.CTkLabel(error_window, text=message, text_color="red", wraplength=400)
        self.error_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

    def _validate(self):
        entries = [self.entry_framehop, self.entry_n_streamers, self.entry_stream_buffer_depth]
        issues = [f"{entry.name}: {entry.argvalid.message}" for entry in entries if not entry.valid]

        if not issues:
            return True
        else:
            self._display_error(f"Invalid settings!\n\n{'\n'.join(issues)}")
            return False

    def _close(self):
        focused = self.focus_get()
        focused.event_generate('<FocusOut>')
        if self._validate():
            self.destroy()



if __name__ == "__main__":
    root = AnalysisSettings()
    root.mainloop()