"""Locating real MPEG audio frame boundaries in an mp3 by scanning for sync.

A fragment shim has to begin on a genuine frame header or libsndfile rejects
the file outright ("Format not recognised"). Bitrate arithmetic will not do it:
these files are CBR, but CBR frames still differ by one byte via the padding
bit, so frame_index * frame_size drifts. So: scan for the sync word and
validate the candidate by walking forward through several frames, each landing
exactly where the previous frame's computed size says it should.
"""

_BITRATES_V1_L3 = (None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
                   256, 320, None)
_BITRATES_V2_L3 = (None, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144,
                   160, None)
_SAMPLERATES = {
    3: (44100, 48000, 32000),   # MPEG 1
    2: (22050, 24000, 16000),   # MPEG 2
    0: (11025, 12000, 8000),    # MPEG 2.5
}


def parse_header(buf, offset):
    """Decode the MPEG audio frame header at `offset`, or None if invalid.

    Layer III only: it is the only layer these recorders emit, and accepting
    the others would widen what counts as a valid candidate for no gain.
    """
    if offset + 4 > len(buf):
        return None
    b0, b1, b2, b3 = buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None

    version = (b1 >> 3) & 0x03      # 3=MPEG1, 2=MPEG2, 0=MPEG2.5, 1=reserved
    layer = (b1 >> 1) & 0x03        # 1 == Layer III
    if version == 1 or layer != 1:
        return None

    bitrate_index = (b2 >> 4) & 0x0F
    samplerate_index = (b2 >> 2) & 0x03
    if samplerate_index == 3:
        return None
    table = _BITRATES_V1_L3 if version == 3 else _BITRATES_V2_L3
    bitrate = table[bitrate_index]
    if bitrate is None:      # 0 = "free format", 15 = reserved
        return None

    samplerate = _SAMPLERATES[version][samplerate_index]
    padding = (b2 >> 1) & 0x01
    channel_mode = (b3 >> 6) & 0x03
    channels = 1 if channel_mode == 3 else 2

    # MPEG 1 Layer III carries 1152 samples per frame; MPEG 2 and 2.5 carry 576.
    samples = 1152 if version == 3 else 576
    size = (samples // 8) * bitrate * 1000 // samplerate + padding

    return {
        'offset': offset,
        'version': version,
        'bitrate': bitrate,
        'samplerate': samplerate,
        'channels': channels,
        'padding': padding,
        'samples': samples,
        'size': size,
    }


def _chain_ok(buf, header, n_frames):
    """Do `n_frames` consecutive valid, consistent headers follow from here?

    This is what separates a real boundary from audio data that happens to
    contain 0xFF: each next header must land exactly at the previous frame's
    computed size and describe the same stream.
    """
    at = header
    for _ in range(n_frames - 1):
        nxt = parse_header(buf, at['offset'] + at['size'])
        if nxt is None:
            return False
        if (nxt['version'] != header['version']
                or nxt['samplerate'] != header['samplerate']
                or nxt['channels'] != header['channels']
                or nxt['bitrate'] != header['bitrate']):
            return False
        at = nxt
    return True


def find_frame(buf, start=0, chain=4):
    """First offset at or after `start` where `chain` valid frames begin.

    Returns the parsed header of that first frame, or None if the buffer runs
    out. `chain` frames must all validate, so the caller should hand in a
    buffer with room for them past the boundary it is looking for.
    """
    at = start
    limit = len(buf) - 4
    while at <= limit:
        index = buf.find(b'\xff', at)
        if index < 0 or index > limit:
            return None
        header = parse_header(buf, index)
        if header is not None and _chain_ok(buf, header, chain):
            return header
        at = index + 1
    return None
