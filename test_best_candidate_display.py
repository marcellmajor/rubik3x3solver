"""Tests for the Space shortcut that shows the current best candidate in solver_deap_coevolution.py."""

import contextlib
import dataclasses
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import solver_deap_coevolution as s
from test_block_library import PRE_FEATURE_DIGESTS, search_digest
from test_manual_mitm import FakeTerminal, empty_search, terminal_mode

SOLVED = s.REFERENCE_SOLVED_KEY
FAR = "R1 U2 C3 D1 B2 L3 R2 C1"
_saved_environment = {}


def setUpModule():
    _saved_environment["directory"] = tempfile.mkdtemp()
    for name in ("XDG_CACHE_HOME", "XDG_DATA_HOME"):
        _saved_environment[name] = os.environ.get(name)
        os.environ[name] = os.path.join(_saved_environment["directory"], name.lower())


def tearDownModule():
    shutil.rmtree(_saved_environment["directory"], ignore_errors=True)
    for name in ("XDG_CACHE_HOME", "XDG_DATA_HOME"):
        if _saved_environment[name] is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = _saved_environment[name]


def after(key, moves):
    sequence = s.parse_moves(moves) if isinstance(moves, str) else tuple(moves)
    return s.state_key(s.replay(s.cube_from_key(key), sequence))


class DisplayTestCase(unittest.TestCase):
    def setUp(self):
        s._SOLVED_SIDE_TABLES.clear()
        self.directory = tempfile.mkdtemp()
        self.messages = []
        self.snapshots = []
        self.requests = s.ManualMitmRequests()
        self.display = s.DisplayRequests()

    def tearDown(self):
        s._SOLVED_SIDE_TABLES.clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    def make_run(self, key=None, **overrides):
        values = dict(fitness="manhattan", seed=3, generations=3, solution_population=20, block_population=20,
                      contexts=6, solution_capacity=16, block_capacity=8, min_block_gain=1.0, log_every=10 ** 6,
                      block_library=None, mitm=False, mitm_depth=4, mitm_attempt_seconds=30.0,
                      mitm_total_seconds=60.0, mitm_cache=os.path.join(self.directory, "cache"))
        values.update(overrides)
        config = s.SearchConfig(**values)
        key = after(SOLVED, FAR) if key is None else key
        return s.CoevolutionRun(config, key, s.validate_config(config), config.seed, log=self.messages.append,
                                manual_requests=self.requests, display_requests=self.display,
                                on_display=self.capture)

    def capture(self, snapshot):
        snapshot = dict(snapshot, archive_at_display=self.run.archive)
        self.snapshots.append(snapshot)

    def set_archive(self, run, active):
        active = tuple(s.parse_moves(active) if isinstance(active, str) else active)
        endpoint = s.replay_key(run.input_key, active)
        run.archive = s.ArchiveEntry(active, endpoint, run.objective.cost(endpoint), s.is_solved_key(endpoint),
                                     41, run.completed_generation, "fixture")
        return run.archive


class KeyTests(DisplayTestCase):
    def test_space_without_enter_queues_one_display_and_coalesces(self):
        control = s.KeyboardControl(self.requests, display_requests=self.display, stdin=FakeTerminal(),
                                    out=self.messages.append)
        control.handle_key(" ")
        self.assertTrue(self.display.pending)
        for _ in range(10):  # key repeat while pending
            control.handle_key(" ")
        self.assertTrue(self.display.take())
        self.assertFalse(self.display.take())  # one display, not eleven
        self.assertEqual(self.messages, [])  # the key reader never prints a candidate
        self.assertIn("Press Space to show the current best candidate; M to request MITM", control.status_line())

    def test_space_and_m_are_independent(self):
        control = s.KeyboardControl(self.requests, display_requests=self.display, stdin=FakeTerminal(),
                                    out=self.messages.append)
        control.handle_key(" ")
        control.handle_key("m")
        self.assertTrue(self.display.take())
        self.assertTrue(self.requests.pending)  # taking the display left the MITM request alone
        self.assertTrue(self.requests.take())
        control.handle_key(" ")  # Space still queues while a MITM attempt is running
        control.handle_key("M")  # ...while M is ignored then
        self.assertTrue(self.display.pending)
        self.requests.finish()
        self.assertFalse(self.requests.pending)
        self.assertEqual(self.messages, [s.MANUAL_MITM_ACK])

    @unittest.skipIf(sys.platform == "win32", "pseudo-terminals are POSIX only")
    def test_space_on_a_real_terminal(self):
        import pty
        master, slave = pty.openpty()
        terminal = os.fdopen(slave, "rb", buffering=0)
        self.addCleanup(terminal.close)
        self.addCleanup(os.close, master)
        before = terminal_mode(terminal.fileno())
        control = s.KeyboardControl(self.requests, display_requests=self.display, stdin=terminal,
                                    out=self.messages.append, poll_seconds=0.05)
        with control:
            os.write(master, b" ")
            self.assertTrue(self.display.wait_for_request(5))
            os.write(master, b"m")
            self.assertTrue(self.requests.wait_for_request(5))
        self.assertTrue(self.display.pending)
        self.assertEqual(terminal_mode(terminal.fileno()), before)


class SnapshotTests(DisplayTestCase):
    def test_latest_candidate_and_active_objective_at_servicing_time(self):
        self.run = run = self.make_run(fitness="grouped-alternating")
        self.display.request()
        entry = self.set_archive(run, "R1 U1 C1")  # changes after the key press
        run._switch_objective("grouped-corners", "test", 1)
        run.budget.check()
        snapshot = self.snapshots[-1]
        self.assertEqual(snapshot["moves"], run.archive.active)
        self.assertEqual((snapshot["fitness"], snapshot["objective"]), ("grouped-alternating", "grouped-corners"))
        self.assertEqual(snapshot["cost"], s.grouped_corners_cost(snapshot["endpoint"]))
        self.assertIn("current phase grouped-corners", s.best_candidate_lines(snapshot)[1])
        self.assertEqual(entry.active, s.parse_moves("R1 U1 C1"))

    def test_moves_cube_and_cost_describe_one_candidate(self):
        self.run = run = self.make_run()
        self.set_archive(run, "L1 D2 B3")
        self.display.request()
        run.budget.check()
        snapshot = self.snapshots[-1]
        replayed = s.state_key(s.replay(s.cube_from_key(run.input_key), snapshot["moves"]))  # cube.py replay
        self.assertEqual(snapshot["endpoint"], replayed)
        self.assertEqual(s.state_key(snapshot["cube"]), replayed)
        self.assertEqual(snapshot["cost"], s.manhattan_cost(replayed))
        self.assertTrue(snapshot["consistent"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            s._print_best_candidate(snapshot)
        text = output.getvalue()
        self.assertIn("=== Current best solution candidate", text)
        self.assertIn(f"H={snapshot['cost']}, solved: no", text)
        self.assertIn("moves (3): L1 D2 B3", text)
        self.assertIn("via S#41, fixture", text)
        cube = snapshot["cube"]
        self.assertIn(f'"{cube.top_face}"', text)  # the printed cube is the replayed one
        self.assertIn(f'"{cube.bottom_face}"', text)

    def test_empty_candidate_is_explicit(self):
        self.run = run = self.make_run()
        self.display.request()
        run.budget.check()
        snapshot = self.snapshots[-1]
        self.assertEqual(snapshot["moves"], ())
        self.assertEqual(snapshot["endpoint"], run.input_key)
        self.assertIn("    moves (0): (empty move list) (the unchanged input cube)", s.best_candidate_lines(snapshot))
        self.assertEqual(snapshot["when"], "before generation 1")

    def test_display_is_read_only(self):
        self.run = run = self.make_run()
        run.budget.start()
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        run.run_generation(1)
        state = (run.rng.getstate(), dataclasses.asdict(run.counters), run.archive,
                 [ind.uid for ind in run.population], [block.uid for block in run.blocks],
                 len(run.cost._values), run.last_improvement_generation, run.last_quality_improvement_generation,
                 dict(run._finisher_cache), run._finisher_last_attempt)
        self.display.request()
        run.budget.check()
        self.assertEqual(len(self.snapshots), 1)
        self.assertEqual((run.rng.getstate(), dataclasses.asdict(run.counters), run.archive,
                          [ind.uid for ind in run.population], [block.uid for block in run.blocks],
                          len(run.cost._values), run.last_improvement_generation,
                          run.last_quality_improvement_generation, dict(run._finisher_cache),
                          run._finisher_last_attempt), state)


class ServicingTests(DisplayTestCase):
    def test_served_during_a_generation_without_waiting_for_it(self):
        self.run = run = self.make_run(generations=2)
        run.on_improvement = lambda current, entry: self.display.request()  # a press mid-evaluation
        run.run()
        during = [snap for snap in self.snapshots if snap["when"].startswith("during generation")]
        self.assertTrue(during)
        for snapshot in self.snapshots:
            self.assertEqual(snapshot["moves"], snapshot["archive_at_display"].active)
            self.assertTrue(snapshot["consistent"])

    def test_served_during_mitm_without_disturbing_it(self):
        self.run = run = self.make_run()
        run.budget.start()
        self.requests.request()
        calls = []

        def search(*args, check=None, **kwargs):
            self.display.request()  # Space during the search
            check()  # the search's own cooperative check serves it
            calls.append(len(self.snapshots))
            return empty_search()

        with mock.patch.object(s, "mitm_search", side_effect=search):
            summary = run.finisher_boundary(1)
        self.assertEqual(calls, [1])
        self.assertTrue(self.snapshots[0]["when"].startswith("during a MITM attempt"))
        self.assertEqual((summary["trigger"], summary["outcome"]), ("manual", "no usable improvement"))

    def test_seeded_search_unchanged_with_and_without_presses(self):
        key = after(SOLVED, "R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        config = s.SearchConfig(fitness="manhattan", seed=5, generations=6, solution_population=20,
                                block_population=20, solution_capacity=16, block_capacity=8, contexts=6,
                                min_block_gain=1.0, log_every=10)
        for press in (False, True):
            display = s.DisplayRequests()
            shown = []
            run = s.CoevolutionRun(config, key, s.validate_config(config), 5, log=lambda message: None,
                                   display_requests=display, on_display=shown.append)
            if press:
                run.on_improvement = lambda current, entry: display.request()
            self.assertEqual(search_digest(run, run.run()), PRE_FEATURE_DIGESTS["manhattan"])
            self.assertEqual(bool(shown), press)


class CliTests(DisplayTestCase):
    def test_space_at_startup_prints_the_candidate(self):
        class SpaceKeyboard:
            def __init__(self, requests, display_requests=None):
                self.display_requests = display_requests

            def status_line(self):
                return "Keyboard control: fake"

            def __enter__(self):
                self.display_requests.request()
                return self

            def __exit__(self, *exc_info):
                return False

        args = ["--generations", "1", "--solution-population", "10", "--block-population", "10", "--contexts", "4",
                "--seed", "1", "--no-block-library", "--no-mitm"]
        with mock.patch.object(s, "default_input_cube", lambda: s.cube_from_key(after(SOLVED, FAR))):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = s.main(args, keyboard=SpaceKeyboard)
        text = output.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("=== Current best solution candidate (before generation 1) ===", text)
        self.assertIn("moves (0): (empty move list) (the unchanged input cube)", text)


if __name__ == "__main__":
    unittest.main()
