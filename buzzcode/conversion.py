def cmd_convert(path_in, path_out, verbosity=1, band_low=200):
    cmdlist = [
        "ffmpeg",
        "-i", path_in,
        '-n'  # don't overwrite
    ]

    if verbosity == 0:
        cmdlist.extend(["-v", "quiet"])

    if verbosity == 1:
        cmdlist.extend(["-stats"])

    cmdlist.extend(
        [
            "-ar", "16000",  # Audio sample rate
            "-ac", "1",  # Audio channels
            "-af", "highpass = f = " + str(band_low),
            "-c:a", "pcm_s16le",  # Audio codec
            "-rf64", "always",  # use riff64 to allow large files
            path_out
        ]
    )

    return cmdlist
