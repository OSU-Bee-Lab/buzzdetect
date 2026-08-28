"""src/pipeline/interrupt.py: the cooperative stop.

Stopping is a request, not a kill: the first one goes on the coordinator's
early-exit queue so the workers wind down and report themselves out. A second
one is impatience and ends the process.
"""

import io
import queue
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import tests._context as ctx  # noqa: F401

from src.pipeline import interrupt
from src.pipeline.progress_json import MARKER


class FakeCoordinator:
    def __init__(self):
        self.q_earlyexit = queue.Queue()


class TestStopRequest(unittest.TestCase):
    def setUp(self):
        self.coordinator = FakeCoordinator()
        self.stop = interrupt.StopRequest(self.coordinator)

    def request(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.stop.request()
        return out.getvalue()

    def test_starts_unrequested(self):
        self.assertFalse(self.stop.requested.is_set())

    def test_the_first_request_goes_on_the_early_exit_queue(self):
        self.request()
        self.assertTrue(self.stop.requested.is_set())
        self.assertEqual(self.coordinator.q_earlyexit.get_nowait(), 'Analysis stopped by user')

    def test_the_first_request_tells_the_host_it_is_stopping(self):
        line = self.request()
        self.assertIn(MARKER, line)
        self.assertIn('"name": "stopping"', line)

    def test_a_custom_message_is_carried_through(self):
        with redirect_stdout(io.StringIO()):
            self.stop.request('out of memory')
        self.assertEqual(self.coordinator.q_earlyexit.get_nowait(), 'out of memory')

    def test_a_second_request_exits_rather_than_queueing_again(self):
        self.request()
        # os._exit doesn't return, so the stand-in mustn't either, or the rest
        # of request() runs in the test and nowhere else.
        with mock.patch.object(interrupt.os, '_exit', side_effect=SystemExit) as exit_now, \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.stop.request()
        exit_now.assert_called_once_with(130)
        self.assertEqual(self.coordinator.q_earlyexit.qsize(), 1)  # no second request queued


class TestInstall(unittest.TestCase):
    def setUp(self):
        self.coordinator = FakeCoordinator()
        # A terminal by default, so install() doesn't start a watcher thread on
        # the real stdin and sit there reading it for the rest of the suite.
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        self.enterContext(mock.patch.object(sys, 'stdin', stdin))

    def test_installs_handlers_for_the_signals_this_platform_has(self):
        with mock.patch.object(interrupt.signal, 'signal') as signal_signal:
            interrupt.install(self.coordinator)
        installed = {call.args[0] for call in signal_signal.call_args_list}
        self.assertIn(interrupt.signal.SIGINT, installed)

    def test_a_signal_asks_for_the_same_tidy_stop(self):
        handlers = {}
        with mock.patch.object(interrupt.signal, 'signal',
                               side_effect=lambda sig, fn: handlers.setdefault(sig, fn)):
            stop = interrupt.install(self.coordinator)
        with redirect_stdout(io.StringIO()):
            handlers[interrupt.signal.SIGINT](interrupt.signal.SIGINT, None)
        self.assertTrue(stop.requested.is_set())
        self.assertFalse(self.coordinator.q_earlyexit.empty())

    def test_being_installed_off_the_main_thread_is_not_fatal(self):
        # A ValueError from signal.signal means we're not the main thread; the
        # host's stdin channel still works, so install must not raise.
        with mock.patch.object(interrupt.signal, 'signal', side_effect=ValueError):
            self.assertIsNotNone(interrupt.install(self.coordinator))

    def test_a_terminal_is_not_watched_for_a_stop_line(self):
        # On a terminal the watcher would swallow whatever the user types, and
        # Ctrl-C already covers that case.
        with mock.patch.object(interrupt.threading, 'Thread') as thread:
            interrupt.install(self.coordinator)
        thread.assert_not_called()

    def test_a_pipe_is_watched(self):
        stdin = mock.Mock()
        stdin.isatty.return_value = False
        with mock.patch.object(sys, 'stdin', stdin), \
                mock.patch.object(interrupt.threading, 'Thread') as thread:
            interrupt.install(self.coordinator)
        thread.assert_called_once()
        self.assertTrue(thread.call_args.kwargs['daemon'])

    def test_a_closed_stdin_is_not_watched(self):
        stdin = mock.Mock()
        stdin.isatty.side_effect = ValueError('I/O operation on closed file')
        with mock.patch.object(sys, 'stdin', stdin), \
                mock.patch.object(interrupt.threading, 'Thread') as thread:
            interrupt.install(self.coordinator)
        thread.assert_not_called()


class TestStdinWatcher(unittest.TestCase):
    """The line the desktop app writes on the engine's stdin."""

    def run_watcher(self, text):
        coordinator = FakeCoordinator()
        with mock.patch.object(sys, 'stdin', io.StringIO(text)), \
                mock.patch.object(interrupt.threading, 'Thread') as thread:
            stop = interrupt.install(coordinator)
            watch = thread.call_args.kwargs['target']
            with redirect_stdout(io.StringIO()):
                watch()
        return stop, coordinator

    def test_the_stop_command_asks_for_a_stop(self):
        stop, coordinator = self.run_watcher(interrupt.STOP_COMMAND + '\n')
        self.assertTrue(stop.requested.is_set())
        self.assertFalse(coordinator.q_earlyexit.empty())

    def test_case_and_whitespace_do_not_matter(self):
        stop, _ = self.run_watcher('  stop  \n')
        self.assertTrue(stop.requested.is_set())

    def test_other_lines_are_ignored(self):
        stop, _ = self.run_watcher('y\nhello\n')
        self.assertFalse(stop.requested.is_set())

    def test_the_manifest_prompts_y_does_not_stop_the_run(self):
        # start_analysis answers the manifest prompt with 'y' on the same pipe.
        stop, _ = self.run_watcher('y\n' + interrupt.STOP_COMMAND + '\n')
        self.assertTrue(stop.requested.is_set())

    def test_eof_alone_does_not_ask_for_a_stop(self):
        stop, coordinator = self.run_watcher('')
        self.assertFalse(stop.requested.is_set())
        self.assertTrue(coordinator.q_earlyexit.empty())


if __name__ == '__main__':
    unittest.main()
