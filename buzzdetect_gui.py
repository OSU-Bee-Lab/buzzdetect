import multiprocessing

import customtkinter as ctk

from buzzcode.gui.splashscreen import SplashScreen


def analyze_gui():
    multiprocessing.set_start_method("spawn")
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

    splash = SplashScreen()
    from buzzcode.gui.settings import AnalysisSettings
    from buzzcode.gui.analysis import AnalysisWindow
    splash.destroy()

    settings = AnalysisSettings()
    settings.mainloop()

    if settings.run:
        analysis = AnalysisWindow(settings.vars_analysis)
        analysis.mainloop()

if __name__ == "__main__":
    analyze_gui()