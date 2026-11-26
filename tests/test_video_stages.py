"""Reading ahead and encoding behind: same items, same order, errors and stops carried across."""
import threading
import time

import pytest

from hdr_blaster.video.stages import Ahead, Behind


def live_threads(prefix="hdr-blaster"):
    return [t for t in threading.enumerate() if t.name.startswith(prefix)]


def test_reading_ahead_gives_every_item_in_order():
    with Ahead(lambda: iter(range(200)), depth=3) as items:
        assert list(items) == list(range(200))
    assert live_threads() == []


def test_the_reader_runs_ahead_but_never_further_than_its_depth():
    produced = []

    def source():
        for i in range(50):
            produced.append(i)
            yield i

    with Ahead(source, depth=4) as items:
        iterator = iter(items)
        assert next(iterator) == 0
        time.sleep(0.3)
        # one taken, four waiting, one in the reader's hands
        assert 5 <= len(produced) <= 6
        assert list(iterator) == list(range(1, 50))


def test_what_the_reader_raises_reaches_the_consumer_where_it_happened():
    def source():
        yield from range(5)
        raise OSError("disk gone")

    seen = []
    with pytest.raises(OSError, match="disk gone"):
        with Ahead(source) as items:
            for item in items:
                seen.append(item)
    assert seen == [0, 1, 2, 3, 4]
    assert live_threads() == []


def test_leaving_early_stops_the_reader_and_closes_its_iterator():
    closed = []

    def source():
        try:
            for i in range(10 ** 9):
                yield i
        finally:
            closed.append(threading.current_thread().name)

    with Ahead(source, depth=2) as items:
        for item in items:
            if item == 3:
                break
    assert closed == ["hdr-blaster-reader"], "the iterator is closed in the thread that ran it"
    assert live_threads() == []


def test_working_behind_does_every_item_in_order():
    done = []
    behind = Behind(lambda item: (time.sleep(0.001), done.append(item)), depth=3)
    for i in range(100):
        behind.put(i)
    behind.close()
    assert done == list(range(100))
    assert live_threads() == []


def test_a_failure_behind_is_raised_to_whoever_hands_the_next_items():
    def work(item):
        if item == 5:
            raise ValueError("encoder refused")

    behind = Behind(work, depth=2)
    with pytest.raises(ValueError, match="encoder refused"):
        for i in range(1000):
            behind.put(i)
    behind.abort()
    assert live_threads() == []


def test_a_failure_on_the_last_item_is_raised_when_closing():
    def work(item):
        if item == 9:
            raise ValueError("last one failed")

    behind = Behind(work)
    for i in range(10):
        behind.put(i)
    with pytest.raises(ValueError, match="last one failed"):
        behind.close()
    assert live_threads() == []


def test_aborting_drops_what_is_still_waiting():
    done = []
    gate = threading.Event()

    def work(item):
        gate.wait(5)
        done.append(item)

    behind = Behind(work, depth=4)
    for i in range(4):
        behind.put(i)
    gate.set()
    behind.abort()
    assert len(done) <= 2, "at most the item in hand (and one taken as the gate opened) is finished"
    assert live_threads() == []
