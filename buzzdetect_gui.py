import customtkinter as ctk
from buzzcode.gui.Settings import AnalysisSettings
from buzzcode.gui.Analysis import AnalysisWindow
import multiprocessing

def analyze_gui():
    multiprocessing.set_start_method("spawn")
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

    settings = AnalysisSettings()
    settings.mainloop()

    analysis = AnalysisWindow(settings.vars_analysis)
    analysis.mainloop()

if __name__ == "__main__":
    analyze_gui()