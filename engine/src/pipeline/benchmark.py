"""Per-stage timing for finding out where a run's time goes.

Off unless the run asks for it. When on, every worker reports one line per
unit of work at the BENCHMARK log level (below PROGRESS, so it never reaches the
console) and the tally here is written out as a summary when the analysis ends.

A field named ``t_*`` is a duration in seconds; anything else is a count sampled
at that moment (e.g. a queue depth). The lines are ``BENCH <role> [<worker id>] k=v ...`` so a
log can be parsed with a split.
"""

import threading
import time
from collections import defaultdict

from src.pipeline.assignments import AssignLog

now = time.perf_counter


class Bench:
    def __init__(self, q_log, enabled: bool = False):
        self.enabled = enabled
        self.q_log = q_log
        self._lock = threading.Lock()
        self._n = defaultdict(int)
        self._sums = defaultdict(lambda: defaultdict(float))
        self._zeros = defaultdict(lambda: defaultdict(int))
        self._span = {}

    def record(self, role: str, ident='', **fields):
        if not self.enabled:
            return

        t = now()
        with self._lock:
            first, _ = self._span.get(role, (t, t))
            self._span[role] = (first, t)
            self._n[role] += 1
            for k, v in fields.items():
                self._sums[role][k] += v
                if v == 0:
                    self._zeros[role][k] += 1

        body = ' '.join(f'{k}={v:.5f}' if k.startswith('t_') else f'{k}={v}' for k, v in fields.items())
        self.q_log.put(AssignLog(message=f'BENCH {role} [{ident}] {body}', level_str='BENCHMARK'))

    def summary(self) -> str:
        with self._lock:
            lines = ['Benchmark summary (t_* in seconds; share = of the role\'s summed t_* time)']
            for role in sorted(self._n):
                n = self._n[role]
                sums = self._sums[role]
                first, last = self._span[role]
                total_t = sum(v for k, v in sums.items() if k.startswith('t_')) or 1.0
                lines.append(f'  {role}: n={n}, first-to-last {last - first:.2f}s')
                for k, v in sums.items():
                    if k.startswith('t_'):
                        lines.append(f'    {k:<14} total {v:8.2f}  mean {v / n * 1000:8.2f}ms  share {v / total_t:6.1%}')
                    else:
                        lines.append(f'    {k:<14} mean {v / n:8.2f}  at zero {self._zeros[role][k] / n:6.1%}')
            return '\n'.join(lines)
