import threading
import time
import unittest

from router.breaker import CircuitBreaker
from router.dispatcher import Dispatcher, NoEngine, QueueFull, QueueTimeout
from router.engine import Engine, LatencyWindow
from router.policy import LeastLoaded, PrefixAffinity


def engines(*seats):
    return [Engine(f"e{i}", f"http://127.0.0.1:{9000 + i}", s) for i, s in enumerate(seats)]


class SeatTest(unittest.TestCase):
    def test_seats_are_reserved_and_returned(self):
        es = engines(1, 2)
        d = Dispatcher(es, LeastLoaded(), queue_limit=0)
        got = [d.acquire(timeout=0.1) for _ in range(3)]
        self.assertEqual(sorted(e.name for e in got), ["e0", "e1", "e1"])
        with self.assertRaises(QueueFull):
            d.acquire(timeout=0.1)
        d.release(got[0], ok=True, latency_s=0.5)
        self.assertIs(d.acquire(timeout=0.1), got[0])

    def test_release_without_acquire_is_an_error(self):
        es = engines(1)
        d = Dispatcher(es, LeastLoaded())
        with self.assertRaises(RuntimeError):
            d.release(es[0], ok=True)


class QueueTest(unittest.TestCase):
    def test_waiter_gets_the_seat_when_released(self):
        es = engines(1)
        d = Dispatcher(es, LeastLoaded(), queue_limit=4, wake_interval_s=0.01)
        first = d.acquire()
        result = {}

        def waiter():
            result["engine"] = d.acquire(timeout=2)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        self.assertEqual(d.snapshot()["queue_depth"], 1)
        d.release(first, ok=True)
        t.join(2)
        self.assertIs(result["engine"], es[0])
        self.assertEqual(d.snapshot()["queue_depth"], 0)

    def test_fifo_order(self):
        es = engines(1)
        d = Dispatcher(es, LeastLoaded(), queue_limit=8, wake_interval_s=0.01)
        held = d.acquire()
        order = []
        lock = threading.Lock()

        def waiter(i):
            e = d.acquire(timeout=5)
            with lock:
                order.append(i)
            d.release(e, ok=True)

        threads = []
        for i in range(4):
            t = threading.Thread(target=waiter, args=(i,))
            t.start()
            threads.append(t)
            while d.snapshot()["queue_depth"] < i + 1:
                time.sleep(0.005)
        d.release(held, ok=True)
        for t in threads:
            t.join(5)
        self.assertEqual(order, [0, 1, 2, 3])

    def test_queue_full_carries_retry_after(self):
        es = engines(1)
        d = Dispatcher(es, LeastLoaded(), queue_limit=0)
        es[0].latency.add(4.0)
        d.acquire()
        with self.assertRaises(QueueFull) as ctx:
            d.acquire()
        self.assertEqual(ctx.exception.retry_after, 4)
        self.assertEqual(d.snapshot()["counters"]["rejected_queue_full"], 1)

    def test_timeout(self):
        d = Dispatcher(engines(1), LeastLoaded(), queue_limit=1, wake_interval_s=0.01)
        d.acquire()
        t0 = time.monotonic()
        with self.assertRaises(QueueTimeout):
            d.acquire(timeout=0.1)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(d.snapshot()["queue_depth"], 0)


class BreakerIntegrationTest(unittest.TestCase):
    def test_all_open_raises_no_engine(self):
        es = engines(2)
        es[0].breaker = CircuitBreaker(failure_threshold=1, cooldown_s=60)
        d = Dispatcher(es, LeastLoaded())
        d.record_health(es[0], ok=False)
        with self.assertRaises(NoEngine) as ctx:
            d.acquire(timeout=0.1)
        self.assertEqual(ctx.exception.retry_after, 60)

    def test_open_engine_is_skipped_and_pins_are_dropped(self):
        es = engines(2, 2)
        for e in es:
            e.breaker = CircuitBreaker(failure_threshold=1, cooldown_s=60)
        policy = PrefixAffinity()
        d = Dispatcher(es, policy)
        e = d.acquire("task")
        self.assertEqual(policy.pinned("task"), e.name)
        d.release(e, ok=False, outcome="connect_error")
        self.assertIsNone(policy.pinned("task"))
        other = d.acquire("task")
        self.assertIsNot(other, e)

    def test_avoid_is_a_soft_preference(self):
        es = engines(1, 1)
        d = Dispatcher(es, LeastLoaded())
        self.assertIs(d.acquire(avoid=[es[0]]), es[1])
        self.assertIs(d.acquire(avoid=[es[0]]), es[0])  # only e0 has a seat left


class LatencyWindowTest(unittest.TestCase):
    def test_quantiles(self):
        w = LatencyWindow(size=100)
        self.assertIsNone(w.quantile(0.5))
        for v in range(1, 101):
            w.add(float(v))
        self.assertEqual(w.quantile(0.5), 50.0)
        self.assertEqual(w.quantile(0.95), 95.0)
        w.add(1000.0)  # evicts 1.0
        self.assertEqual(len(w), 100)


if __name__ == "__main__":
    unittest.main()
