import multiprocessing
from tkinter import messagebox

import customtkinter as ctk

import buzzcode.gui.config as cfg_gui
from buzzcode.analysis.analysis import Analyzer
from buzzcode.analysis.assignments import AssignLog
from buzzcode.analysis.workers import Coordinator


def run_analysis(vars_analysis, q_gui, event_analysisdone, q_earlyexit):
    coordinator = Coordinator(
        analyzers_cpu=vars_analysis['analyzers_cpu'],
        analyzer_gpu=vars_analysis['analyzer_gpu'],
        streamers_total=vars_analysis['n_streamers'],
        depth=vars_analysis['stream_buffer_depth'],
        q_gui=q_gui,
        event_analysisdone=event_analysisdone,
        q_earlyexit=q_earlyexit
    )

    analyzer = Analyzer(
        modelname=vars_analysis['modelname'],
        classes_out=vars_analysis['classes_out'],
        precision=vars_analysis['precision'],
        framehop_prop=vars_analysis['framehop_prop'],
        chunklength=vars_analysis['chunklength'],
        dir_audio=vars_analysis['dir_audio'],
        dir_out=vars_analysis['dir_out'],
        verbosity_print=vars_analysis['verbosity_print'],
        verbosity_log=vars_analysis['verbosity_log'],
        log_progress=vars_analysis['log_progress'],
        coordinator=coordinator
    )
    analyzer.run()


class AnalysisWindow(ctk.CTk):
    def __init__(self, vars_analysis):
        super().__init__()
        self.vars_analysis = vars_analysis

        self.q_gui = multiprocessing.Queue()
        self.event_analysisdone = multiprocessing.Event()
        self.q_earlyexit = multiprocessing.Queue()

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.proc_analysis = multiprocessing.Process(target=lambda: None, name='gui_analysis_idle')

        self.new_analysis_requested = False

        window_width = 500
        window_height = 500

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f'{window_width}x{window_height}+{x}+{y}')
        self.title("buzzdetect - analysis")
        self.geometry(f"{window_width}x{window_height}")

        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.textbox = ctk.CTkTextbox(self, wrap="word")
        self.textbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame_button = ctk.CTkFrame(self)
        self.frame_button.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.frame_button.grid_columnconfigure(0, weight=1, uniform="buttons")
        self.frame_button.grid_columnconfigure(1, weight=1, uniform="buttons")
        self.frame_button.grid_columnconfigure(2, weight=1, uniform="buttons")

        self.button_stop = ctk.CTkButton(
            self.frame_button,
            text="Stop Analysis",
            command=self.stop_analysis,
            fg_color='pink',
            text_color='black'
        )

        self.button_stop.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.button_relaunch = ctk.CTkButton(
            self.frame_button,
            text="Re-run analysis",
            command=self.launch_analysis
        )
        self.button_relaunch.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.button_new = ctk.CTkButton(
            self.frame_button,
            text="New Analysis",
            command=self.new_analysis
        )
        self.button_new.grid(row=0, column=2, sticky="ew", padx=5, pady=5)


        self.after(cfg_gui.poll_interval_ms, self.poll_queue)
        self.launch_analysis()  # no need to wait; should only get here via settings window

    def _is_at_bottom(self) -> bool:
        self.update_idletasks()

        total_lines = int(self.textbox.index("end-1c").split('.')[0])

        # Index of the bottom-left visible point in the widget (uses pixel coords)
        last_vis_index = self.textbox.index(f"@0,{self.textbox.winfo_height()}")
        # Convert that index to a line number
        last_vis_line = int(last_vis_index.split('.')[0])

        return (total_lines - last_vis_line) <= cfg_gui.bottom_threshold

    def _trim_if_needed(self):
        end_index = self.textbox.index("end-1c")
        total_lines = int(end_index.split('.')[0])
        if total_lines > cfg_gui.max_lines:
            delete_to = total_lines - cfg_gui.max_lines
            self.textbox.delete("1.0", f"{delete_to + 1}.0")

    def launch_analysis(self):
        if self.proc_analysis.is_alive():
            return

        self.event_analysisdone.clear()
        self.proc_analysis = multiprocessing.Process(
            target=run_analysis,
            args=(self.vars_analysis, self.q_gui, self.event_analysisdone, self.q_earlyexit),
            name='gui_analysis'
        )
        self.proc_analysis.start()

        self.textbox.insert("end", "Launching analysis...\n")

    def stop_analysis(self):
        """Terminate the process and all its children."""
        if not self.proc_analysis.is_alive():
            return True

        if not messagebox.askyesno("Confirm Stop", "Stopping will leave partially-analyzed files.\nThe next analysis will pick up where this one left off.\nAre you sure you want to stop?"):
            return False

        self.q_earlyexit.put('Analysis stopped by user')
        self.configure_buttons()
        return True

    def new_analysis(self):
        self.new_analysis_requested = True
        self._close()

    def determine_state(self):
        if not self.proc_analysis.is_alive():
            return "idle"

        # if the event is set, but the process is still alive, we must be shutting down
        if self.event_analysisdone.is_set():
            return "stopping"

        return "running"

    def configure_buttons(self):
        state = self.determine_state()
        if state == 'idle':
            self.button_stop.configure(state="disabled", text='Stop Analysis')
            self.button_relaunch.configure(state="normal")
            self.button_new.configure(state="normal")
        elif state == 'stopping':
            self.button_stop.configure(state="disabled", text='Stopping...')
            self.button_relaunch.configure(state="disabled")
            self.button_new.configure(state="disabled")
        elif state == 'running':
            self.button_stop.configure(state="normal", text='Stop Analysis')
            self.button_relaunch.configure(state="disabled")
            self.button_new.configure(state="disabled")

    def poll_queue(self):
        while not self.q_gui.empty():
            a_log: AssignLog = self.q_gui.get()
            if a_log.terminate:
                break

            # Determine tag (based on level)
            color = cfg_gui.levelcolors.get(a_log.level_str, "black")

            # Make sure the tag exists and has a color
            if a_log.level_str not in self.textbox.tag_names():
                self.textbox.tag_config(a_log.level_str, foreground=color)

            # Insert text with tag
            at_bottom = self._is_at_bottom()
            self.textbox.insert("end", a_log.message + "\n", a_log.level_str)
            self._trim_if_needed()
            if at_bottom:
                self.textbox.see("end")

        self.configure_buttons()
        self.after(cfg_gui.poll_interval_ms, self.poll_queue)

    def _close(self):
        if self.stop_analysis():
            self.proc_analysis.join()
            self.destroy()


if __name__ == '__main__':
    vars_analysis = {
        'modelname': 'model_general_v3',
        'classes_out': 'all',
        'precision': None,
        'framehop_prop': 1,
        'chunklength': 200,
        'analyzers_cpu': 1,
        'analyzer_gpu': False,
        'n_streamers': None,
        'stream_buffer_depth': None,
        'dir_audio': 'audio_in',
        'dir_out': 'local/tmp',
        'verbosity_print': 'NOTSET',
        'verbosity_log': 'NOTSET',
        'log_progress': False,
        'q_gui': None,
        'event_stopanalysis': None
    }

    window = AnalysisWindow(vars_analysis)
    window.mainloop()