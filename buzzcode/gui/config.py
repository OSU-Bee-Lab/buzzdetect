font_textheader = ('TkDefaultFont', 16, "bold")
font_textentry = ('TkDefaultFont', 14)
font_hint = ('TkDefaultFont', 12, 'italic')

path_settingscache = "guisettings.json"

hoverdelay=300

levelcolors = {
    'DEBUG': 'darkgreen',
    'PROGRESS': 'black',
    'INFO': 'black',
    'WARNING': 'darkorange',
    'ERROR': 'darkred',
}

cols_neurons = 2

poll_interval_ms = 100
bottom_threshold = 1  # within this many lines of the bottom of the widget, follow updates (inexact)
max_lines = 100_000