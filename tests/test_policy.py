import unittest
from dataclasses import dataclass

from router.policy import LeastLoaded, PrefixAffinity, RoundRobin, make_policy, prefix_key


@dataclass
class E:
    name: str
    seats: int = 4
    inflight: int = 0


class PrefixKeyTest(unittest.TestCase):
    def test_same_head_same_key_regardless_of_tail(self):
        head = [{"role": "system", "content": "agent"}, {"role": "user", "content": "task 1"}]
        a = prefix_key(head + [{"role": "tool", "content": "x"}])
        b = prefix_key(head + [{"role": "tool", "content": "y"}, {"role": "user", "content": "more"}])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_key_order_inside_message_does_not_matter(self):
        a = prefix_key([{"role": "user", "content": "hi"}], n=1)
        b = prefix_key([{"content": "hi", "role": "user"}], n=1)
        self.assertEqual(a, b)

    def test_different_task_different_key(self):
        sys_msg = {"role": "system", "content": "agent"}
        a = prefix_key([sys_msg, {"role": "user", "content": "task 1"}])
        b = prefix_key([sys_msg, {"role": "user", "content": "task 2"}])
        self.assertNotEqual(a, b)

    def test_n_controls_how_much_is_hashed(self):
        sys_msg = {"role": "system", "content": "agent"}
        a = prefix_key([sys_msg, {"role": "user", "content": "task 1"}], n=1)
        b = prefix_key([sys_msg, {"role": "user", "content": "task 2"}], n=1)
        self.assertEqual(a, b)

    def test_empty(self):
        self.assertIsNone(prefix_key([]))
        self.assertIsNone(prefix_key([{"role": "user", "content": "x"}], n=0))


class LeastLoadedTest(unittest.TestCase):
    def test_lowest_utilisation_wins(self):
        a, b = E("a", seats=4, inflight=2), E("b", seats=8, inflight=3)
        self.assertIs(LeastLoaded().choose([a, b]), b)  # 0.375 < 0.5

    def test_tie_breaks_on_name(self):
        a, b = E("a"), E("b")
        self.assertIs(LeastLoaded().choose([b, a]), a)


class RoundRobinTest(unittest.TestCase):
    def test_cycles_in_configured_order(self):
        engines = [E("a"), E("b"), E("c")]
        rr = RoundRobin(["a", "b", "c"])
        self.assertEqual([rr.choose(engines).name for _ in range(5)], ["a", "b", "c", "a", "b"])

    def test_skips_engines_without_a_seat(self):
        rr = RoundRobin(["a", "b", "c"])
        self.assertEqual(rr.choose([E("b"), E("c")]).name, "b")
        self.assertEqual(rr.choose([E("a"), E("c")]).name, "c")
        self.assertEqual(rr.choose([E("a"), E("c")]).name, "a")


class PrefixAffinityTest(unittest.TestCase):
    def test_new_key_goes_least_loaded_and_sticks(self):
        a, b = E("a", inflight=2), E("b", inflight=0)
        pa = PrefixAffinity()
        self.assertIs(pa.choose([a, b], "k1"), b)
        b.inflight = 3  # b is now busier, but still has a seat
        self.assertIs(pa.choose([a, b], "k1"), b)
        self.assertEqual(pa.hits, 1)
        self.assertEqual(pa.new_keys, 1)

    def test_overflow_when_pinned_engine_is_full_keeps_pin(self):
        a, b = E("a"), E("b")
        pa = PrefixAffinity()
        self.assertIs(pa.choose([a, b], "k"), a)
        self.assertIs(pa.choose([b], "k"), b)  # a has no seat
        self.assertEqual(pa.overflows, 1)
        self.assertEqual(pa.pinned("k"), "a")
        self.assertIs(pa.choose([a, b], "k"), a)

    def test_forget_drops_pins_of_a_dead_engine(self):
        a, b = E("a"), E("b")
        pa = PrefixAffinity()
        pa.choose([a, b], "k")
        pa.forget("a")
        self.assertIsNone(pa.pinned("k"))
        self.assertIs(pa.choose([b], "k"), b)
        self.assertEqual(pa.pinned("k"), "b")

    def test_lru_bound(self):
        pa = PrefixAffinity(max_keys=2)
        e = [E("a")]
        for k in ("k1", "k2", "k3"):
            pa.choose(e, k)
        self.assertIsNone(pa.pinned("k1"))
        self.assertEqual(pa.stats()["affinity_keys"], 2)

    def test_no_key_uses_fallback(self):
        a, b = E("a", inflight=1), E("b")
        self.assertIs(PrefixAffinity().choose([a, b], None), b)


class MakePolicyTest(unittest.TestCase):
    def test_names(self):
        for name in ("least-loaded", "round-robin", "prefix-affinity"):
            self.assertEqual(make_policy(name, ["a"]).name, name)
        with self.assertRaises(ValueError):
            make_policy("random", ["a"])


if __name__ == "__main__":
    unittest.main()
