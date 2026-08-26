import customtkinter as ctk
from PIL import Image, ImageTk

import src.gui.config as cfg_gui


class SplashScreen(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.overrideredirect(True)
        self.attributes('-topmost', True)

        window_width = 500

        img_pil = Image.open('docs/source/_images/title_transparent.png')

        aspect_ratio = img_pil.width / img_pil.height
        window_height = int(window_width / aspect_ratio)
        img_pil = img_pil.resize((window_width, window_height))

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f'{window_width}x{window_height}+{x}+{y}')

        self.canvas = ctk.CTkCanvas(self, width=window_width, height=window_height, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(window_width//2, window_height//2, anchor='center', image=img_tk)

        self.message_x = window_width//2
        self.message_y = window_height - (cfg_gui.font_splash[1]*1.5)

        self.message_id_light = None
        self.message_id_dark = None
        self.update_text("Initializing...")

        self.update()

    def update_text(self, new_message):
        if self.message_id_light is not None:
            self.canvas.delete(self.message_id_light)
        if self.message_id_dark is not None:
            self.canvas.delete(self.message_id_dark)

        self.message_id_dark = self.canvas.create_text(self.message_x + 1, self.message_y + 1, text=new_message, font=cfg_gui.font_splash, fill='black')
        self.message_id_light = self.canvas.create_text(self.message_x, self.message_y, text=new_message, font=cfg_gui.font_splash, fill=cfg_gui.color_text_splash)


if __name__ == '__main__':
    import time
    splash = SplashScreen()
    time.sleep(3)