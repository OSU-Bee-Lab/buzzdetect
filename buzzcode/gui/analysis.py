import multiprocessing
from tkinter import messagebox

import customtkinter as ctk

import buzzcode.gui.config as cfg_gui
from buzzcode.analysis.analysis import Analyzer
from buzzcode.analysis.assignments import AssignLog
from buzzcode.analysis.workers import Coordinator
import threading
import time


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

    print('DEBUG: starting analysis')
    analyzer.run()
    time.sleep(10)
    print('DEBUG: analysis complete')
    print("Child threads still alive:", [t.name for t in threading.enumerate()])


# --- GUI main process ---
class AnalysisWindow(ctk.CTk):
    def __init__(self, vars_analysis):
        super().__init__()
        self.vars_analysis = vars_analysis

        self.q_gui = multiprocessing.Queue()
        self.event_analysisdone = multiprocessing.Event()
        self.q_earlyexit = multiprocessing.Queue()

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

        self.textbox = ctk.CTkTextbox(self, wrap="word")
        self.textbox.pack(expand=True, fill="both", padx=10, pady=10)

        self.start_btn = ctk.CTkButton(self, text="Re-run analysis", command=self.launch_analysis)
        self.start_btn.pack(side=ctk.LEFT, padx=5, pady=5)
        self.stop_btn = ctk.CTkButton(
            self,
            text="Stop Analysis",
            command=self.stop_analysis,
            fg_color=("pink", "pink"),
            text_color=("black", "black"),
        )
        self.stop_btn.pack(side=ctk.RIGHT, padx=5, pady=5)
        self.stop_btn.configure(state="disabled")

        self.after(cfg_gui.poll_interval_ms, self.poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.proc_analysis = None
        self.launch_analysis()  # no need to wait; should only get here via settings window

    # ---- smart autoscroll helpers ----
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
        if self.proc_analysis and self.proc_analysis.is_alive():
            return

        self.event_analysisdone.clear()

        self.proc_analysis = multiprocessing.Process(
            target=run_analysis,
            args=(self.vars_analysis, self.q_gui, self.event_analysisdone, self.q_earlyexit),
            name='gui_analysis'
        )

        self.proc_analysis.start()
        self.textbox.insert("end", "Launching analysis...\n")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")


    def stop_analysis(self):
        """Terminate the process and all its children."""
        if not self.proc_analysis or not self.proc_analysis.is_alive():
            return True

        if not messagebox.askyesno("Confirm Stop", "Stopping will leave partially-analyzed files.\nAre you sure you want to stop?"):
            return False

        self.q_earlyexit.put('Analysis stopped by user')
        self.start_btn.configure(state="normal")
        # self.stop_btn.configure(state="disabled")
        return True

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

        self.after(cfg_gui.poll_interval_ms, self.poll_queue)

    def _close(self):
        if self.stop_analysis():
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
        'dir_out': None,
        'verbosity_print': 'NOTSET',
        'verbosity_log': 'NOTSET',
        'log_progress': False,
        'q_gui': None,
        'event_stopanalysis': None
    }

    window = AnalysisWindow(vars_analysis)
    window.mainloop()