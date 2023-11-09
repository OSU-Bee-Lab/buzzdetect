from subprocess import list2cmdline

def cmd_convert(path_in, path_out, band_low=200):
    cmdlist = [
        "ffmpeg",
        "-i", path_in,
        '-n', # don't overwrite
        "-v", "quiet", # low verbosity
        "-stats",
        "-sample_rate", "16000",  # Audio sample rate
        "-ac", "1",  # Audio channels
        "-af", "highpass = f = " + str(band_low),
        "-c:a", "pcm_s16le",  # Audio codec
        "-rf64", "always",  # use riff64 to allow large files
        path_out
    ]

    return list2cmdline(cmdlist)