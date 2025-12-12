import multiprocessing
from tkinter import messagebox

import customtkinter as ctk

import buzzcode.gui.config as cfg_gui
from buzzcode.analysis.analyze import analyze
from buzzcode.analysis.assignments import AssignLog


# --- GUI main process ---
class AnalysisWindow(ctk.CTk):
    def __init__(self, vars_analysis):
        super().__init__()
        self.vars_analysis = vars_analysis

        self.vars_analysis['event_stopanalysis'] = multiprocessing.Event()
        self.vars_analysis['q_gui'] = multiprocessing.Queue()

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

        self.start_btn = ctk.CTkButton(self, text="Launch Analysis", command=self.launch_analysis)
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

        self.vars_analysis['event_stopanalysis'].clear()  # allows reruns
        self.proc_analysis = multiprocessing.Process(
            target=analyze,
            name='gui_analysis',
            kwargs=self.vars_analysis,
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

        self.vars_analysis['event_stopanalysis'].set()
        self.textbox.insert("end", "Stopping analysis...\n")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        return True

    def poll_queue(self):
        while not self.vars_analysis['q_gui'].empty():
            msg: AssignLog = self.vars_analysis['q_gui'].get()
            if not self.proc_analysis.is_alive or self.vars_analysis['event_stopanalysis'].is_set():
                break

            # Determine tag (based on level)
            color = cfg_gui.levelcolors.get(msg.level_str, "black")

            # Make sure the tag exists and has a color
            if msg.level_str not in self.textbox.tag_names():
                self.textbox.tag_config(msg.level_str, foreground=color)

            # Insert text with tag
            at_bottom = self._is_at_bottom()  # check BEFORE inserting
            self.textbox.insert("end", msg.message + "\n", msg.level_str)
            self._trim_if_needed()
            if at_bottom:
                self.textbox.see("end")

        self.after(cfg_gui.poll_interval_ms, self.poll_queue)

    def _close(self):
        if self.stop_analysis():
            self.destroy()
