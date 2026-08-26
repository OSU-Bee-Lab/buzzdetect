"""Cooperative stop for a running analysis.

Ctrl-C in a terminal and a host GUI's stop button both want the same thing: the
coordinator's early-exit path, so the streamers, analyzers and writer wind down
in order and say so on the way out, rather than the process vanishing
mid-chunk.

Two ways in, because a host process can't always send a signal: Windows has no
SIGTERM, and a windowed app has no console to raise a Ctrl-C event in. So a
line on stdin asks for the same thing. The stdin watcher only runs when stdin
is a pipe -- on a terminal it would swallow whatever the user types, and Ctrl-C
already covers that case.

A second stop request is taken as impatience and exits the process outright.
"""
import os
import signal
import sys
import threading

from src.pipeline.progress_json import emit_progress

# The one line a host process sends on stdin to ask for a tidy stop.
STOP_COMMAND = 'STOP'


class StopRequest:
    """Routes stop requests to the coordinator's early-exit queue, once."""
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.requested = threading.Event()
        self._lock = threading.Lock()

    def request(self, message='Analysis stopped by user'):
        with self._lock:
            first = not self.requested.is_set()
            self.requested.set()

        if not first:
            # Asked twice: stop waiting for the workers and go.
            os._exit(130)

        emit_progress('stage', name='stopping')
        # Read by Coordinator.wait_for_exit's watcher thread. If that hasn't
        # started yet the request sits in the queue until it does, and
        # Analyzer.run checks `requested` at the points before that.
        self.coordinator.q_earlyexit.put(message)


def install(coordinator):
    """Install the signal handlers and stdin watcher. Returns the StopRequest."""
    stop = StopRequest(coordinator)

    def handle_signal(signum, frame):
        stop.request()

    for signame in ('SIGINT', 'SIGTERM'):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, handle_signal)
        except ValueError:
            # Not the main thread -- nothing to install, and the host's stdin
            # channel below still works.
            pass

    def watch_stdin():
        for line in sys.stdin:
            if line.strip().upper() == STOP_COMMAND:
                stop.request()
                return
        # EOF: the host is gone. It kills us itself; nothing to do here.

    try:
        piped = sys.stdin is not None and not sys.stdin.isatty()
    except (ValueError, OSError):
        piped = False

    if piped:
        threading.Thread(target=watch_stdin, name='stop_watcher', daemon=True).start()

    return stop
