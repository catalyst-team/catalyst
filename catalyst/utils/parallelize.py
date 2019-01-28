import sys
from threading import Thread, Lock
from typing import Callable, Iterable
from warnings import warn
import pandas as pd
import numpy as np
import multiprocessing


class StoppableThread(Thread):
    """
    Thread that can be stopped

    - code modified from https://github.com/rampeer/py-parallelize
    """

    def __init__(
        self,
        func: Callable,
        items: Iterable,
        callback: Callable = None,
        callback_each: int = 1,
        continue_on_exception: bool = False,
        exception_impute=None,
        exception_callback: Callable = None
    ):
        super().__init__()
        self.callback = callback
        self.callback_each = callback_each
        self.func = func
        self.items = items
        self.running = False
        self.current_index = 0
        self.results = []
        self.continue_on_exception = continue_on_exception
        self.exception_impute = exception_impute
        self.exception = None
        self.exception_callback = exception_callback

    def run(self):
        self.running = True
        self.results = []
        for self.current_index, item in enumerate(self.items):
            if not self.running:
                break
            try:
                self.results.append(self.func(item))
            except Exception as e:
                if not self.continue_on_exception:
                    self.exception = e
                    self.exception_callback()
                    break
                self.results.append(self.exception_impute)
                warn(
                    "%s processing element %s" %
                    (repr(sys.exc_info()[1]), str(item))
                )
            if self.callback is not None:
                if self.current_index % self.callback_each == 0:
                    self.callback()
        self.running = False


def parallelize(
    items: Iterable,
    func: Callable,
    thread_count: int = None,
    progressbar: bool = False,
    progressbar_tick: int = 1,
    continue_on_exception: bool = False,
    exception_impute=None
):
    """
    This function iterates (in multithreaded fashion)
    over `items` and calls `fun` for each item.

    - code modified from https://github.com/rampeer/py-parallelize

    Args:
        items: items to process.
        func: function to apply to each `items` element.
        progressbar: should progressbar be displayed?
        progressbar_tick: how often should we update progressbar?
        thread_count: how many threads should be allocated?
            If None, this parameter will be chosen automatically.
        continue_on_exception: if True, it will print warning
            if `func` fails on some element, instead of halting
        exception_impute: which value should be put into output
            if `func` throws an exception?
    """

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    lock = Lock()

    def _progressbar_callback():
        def report():
            lock.acquire()
            total = int(sum([len(t.items) for t in threads]))
            current = int(
                sum(
                    [
                        t.current_index + 1.0 if len(t.items) > 0 else 0
                        for t in threads
                    ]
                )
            )
            message = "[{0: <40}] {1} / {2} ({3: .2%})".format(
                "#" * int(current / total * 40), current, total,
                current / total
            )
            print(message, end="\r", file=sys.stderr, flush=True)
            lock.release()

        return report

    def _stop_all_threads():
        for t in threads:
            t.running = False

    items_split = np.array_split(items, thread_count)
    if progressbar:
        callback = _progressbar_callback()
    else:
        callback = None
    threads = [
        StoppableThread(
            func, x, callback, progressbar_tick, continue_on_exception,
            exception_impute, _stop_all_threads
        ) for x in items_split
    ]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupting threads...")
        _stop_all_threads()
        # We have to wait for all threads to process their current elements
        for t in threads:
            t.join()
    if callback is not None:
        callback()

    # Any exceptions?
    for t in threads:
        if t.exception is not None:
            raise t.exception

    collected_results = [item for thread in threads for item in thread.results]

    if isinstance(items, pd.Series):
        return pd.Series(list(collected_results), index=items.index)
    else:
        return collected_results
