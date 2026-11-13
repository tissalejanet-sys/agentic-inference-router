import unittest

from router.breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class BreakerTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.b = CircuitBreaker(failure_threshold=3, cooldown_s=10, clock=self.clock)

    def test_trips_after_consecutive_failures(self):
        self.assertFalse(self.b.record_failure())
        self.assertFalse(self.b.record_failure())
        self.assertTrue(self.b.record_failure())
        self.assertEqual(self.b.state, OPEN)
        self.assertFalse(self.b.allows())
        self.assertEqual(self.b.trips, 1)

    def test_success_resets_the_count(self):
        self.b.record_failure()
        self.b.record_failure()
        self.b.record_success()
        self.b.record_failure()
        self.b.record_failure()
        self.assertEqual(self.b.state, CLOSED)

    def test_half_open_allows_one_probe(self):
        for _ in range(3):
            self.b.record_failure()
        self.clock.t = 9.9
        self.assertEqual(self.b.state, OPEN)
        self.clock.t = 10.0
        self.assertEqual(self.b.state, HALF_OPEN)
        self.assertTrue(self.b.allows())
        self.b.on_dispatch()
        self.assertFalse(self.b.allows())

    def test_probe_success_closes(self):
        for _ in range(3):
            self.b.record_failure()
        self.clock.t = 11
        self.b.on_dispatch()
        self.b.record_success()
        self.assertEqual(self.b.state, CLOSED)
        self.assertTrue(self.b.allows())

    def test_probe_failure_reopens_for_a_full_cooldown(self):
        for _ in range(3):
            self.b.record_failure()
        self.clock.t = 11
        self.b.on_dispatch()
        self.assertTrue(self.b.record_failure())
        self.assertEqual(self.b.state, OPEN)
        self.clock.t = 20
        self.assertEqual(self.b.state, OPEN)
        self.clock.t = 21
        self.assertEqual(self.b.state, HALF_OPEN)

    def test_neutral_end_frees_the_probe_slot(self):
        for _ in range(3):
            self.b.record_failure()
        self.clock.t = 11
        self.b.on_dispatch()
        self.b.record_neutral()
        self.assertTrue(self.b.allows())

    def test_success_while_open_does_not_close_early(self):
        for _ in range(3):
            self.b.record_failure()
        self.b.record_success()
        self.assertEqual(self.b.state, OPEN)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)


if __name__ == "__main__":
    unittest.main()
