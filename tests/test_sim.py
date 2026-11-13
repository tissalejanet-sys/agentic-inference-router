import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))

import simulate  # noqa: E402


class SimulatorTest(unittest.TestCase):
    def setUp(self):
        self.ep = simulate.EngineParams(engines=2, seats=4, kv_tokens=60_000)
        self.w = simulate.WorkloadParams(tasks=30, rate=0.05)

    def test_every_request_completes_for_every_policy(self):
        expected = sum(len(c) for t in simulate.make_tasks(self.w, 7) for c in t.turns)
        for name in ("round-robin", "least-loaded", "prefix-affinity"):
            r = simulate.simulate(name, simulate.make_tasks(self.w, 7), self.ep, self.w)
            self.assertEqual(r["requests"], expected, name)
            self.assertGreaterEqual(r["wait_p95_s"], r["wait_p50_s"])

    def test_deterministic(self):
        a = simulate.simulate("least-loaded", simulate.make_tasks(self.w, 3), self.ep, self.w)
        b = simulate.simulate("least-loaded", simulate.make_tasks(self.w, 3), self.ep, self.w)
        self.assertEqual(a, b)

    def test_affinity_reuses_more_prefix_than_round_robin(self):
        res = simulate.run(["round-robin", "prefix-affinity"], self.ep, self.w, [1, 2])
        self.assertGreater(res["prefix-affinity"]["cache_hit"], res["round-robin"]["cache_hit"])

    def test_cli_runs(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            simulate.main(["--rate", "0.05", "--tasks", "10", "--seeds", "1"])
        self.assertIn("| prefix-affinity |", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
