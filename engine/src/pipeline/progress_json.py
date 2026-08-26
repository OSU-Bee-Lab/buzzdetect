import json
import sys

# Marker prefix lets a host process (buzzdetect2's Rust shell) pick structured
# progress events out of stdout without having to parse human-readable log
# text, which is free to change format. Anything without this prefix is
# ordinary log/print output and can be ignored by such a host.
MARKER = 'BDPROGRESS '


def emit_progress(event: str, **fields):
    payload = {'event': event, **fields}
    print(MARKER + json.dumps(payload), flush=True)
