import multiprocessing

import customtkinter as ctk

from src.gui.splashscreen import SplashScreen


def analyze_gui():
    multiprocessing.set_start_method("spawn")
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

    splash = SplashScreen()
    from src.gui.settings import AnalysisSettings
    from src.gui.analysis import AnalysisWindow
    splash.destroy()

    new_analysis_requested = True
    while new_analysis_requested:
        settings = AnalysisSettings()
        settings.mainloop()

        if not settings.run:
            return

        analysis = AnalysisWindow(settings.vars_analysis)
        analysis.mainloop()
        new_analysis_requested = analysis.new_analysis_requested

if __name__ == "__main__":
    analyze_gui()