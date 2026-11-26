"""Running the stages of a conversion side by side: reading ahead, encoding behind.

A conversion is a chain: decode a frame, compute it on the GPU, encode it. One after the
other, each stage waits for the two others. Here the first and the last get a thread of
their own, a few frames away from the computation in the middle, which stays in the
caller's thread (where progress is reported and cancelling is asked).

Nothing is reordered and nothing is dropped: the frames are the same, in the same order.
"""
import queue
import threading

import torch

DEPTH = 6                      # frames a stage may be ahead of the next one
_END = object()
_TICK = 0.05                   # seconds between two looks at "have we been told to stop?"


class Ahead:
    """Iterate over ``make()``, an iterator built and run in a thread of its own, a few items
    ahead of whoever consumes it.

    What the iterator raises is raised to the consumer, at the place in the sequence where
    it happened. :meth:`close` (or leaving the ``with`` block) stops the thread wherever it
    is and waits for it; the iterator is closed in the thread that ran it.
    """

    def __init__(self, make, depth: int = DEPTH):
        self._queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, args=(make,), name="hdr-blaster-reader", daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=_TICK)
                return True
            except queue.Full:
                pass
        return False

    def _run(self, make) -> None:
        try:
            with torch.inference_mode():          # per thread: the reader computes on the GPU too
                iterator = make()
                try:
                    for item in iterator:
                        if not self._put(item):
                            return
                finally:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
        except BaseException as exc:              # handed to the consumer, whatever it is
            self._error = exc
        finally:
            self._put(_END)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is _END:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Behind:
    """Hand items to ``work``, which runs in a thread of its own, a few items behind.

    :meth:`put` raises what ``work`` raised on an earlier item. :meth:`close` waits until
    everything handed over is done (and raises the same way); :meth:`abort` drops what is
    still waiting. One of the two must be called.
    """

    def __init__(self, work, depth: int = DEPTH):
        self._work = work
        self._queue = queue.Queue(maxsize=depth)
        self._drop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="hdr-blaster-encoder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if self._error is not None or self._drop.is_set():
                continue                           # keep emptying the queue: nobody must block on it
            try:
                self._work(item)
            except BaseException as exc:
                self._error = exc

    def put(self, item) -> None:
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._queue.put(item, timeout=_TICK)
                return
            except queue.Full:
                pass

    def close(self) -> None:
        self._queue.put(_END)                      # the thread always empties the queue: this returns
        self._thread.join()
        if self._error is not None:
            raise self._error

    def abort(self) -> None:
        self._drop.set()
        self._queue.put(_END)
        self._thread.join()
