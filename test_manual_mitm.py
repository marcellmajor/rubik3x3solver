"""Tests for the interactive manual MITM trigger (pressing M) in solver_deap_coevolution.py."""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import solver_deap_coevolution as s
from test_block_library import PRE_FEATURE_DIGESTS, search_digest

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


def empty_search():
    return s.FinisherSearch("exhaustive", 2, 2, 0, [], [], [], 0.0)


class FakeTerminal:
    def isatty(self):
        return True


class ManualTestCase(unittest.TestCase):
    def setUp(self):
        s._SOLVED_SIDE_TABLES.clear()
        self.directory = tempfile.mkdtemp()
        self.messages = []
        self.requests = s.ManualMitmRequests()

    def tearDown(self):
        s._SOLVED_SIDE_TABLES.clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    def make_run(self, key=None, **overrides):
        values = dict(fitness="manhattan", seed=3, generations=3, solution_population=20, block_population=20,
                      contexts=6, solution_capacity=16, block_capacity=8, min_block_gain=1.0, log_every=10 ** 6,
                      block_library=None, mitm=False, mitm_depth=4, mitm_trigger_cubies=4, mitm_interval=50,
                      mitm_attempt_seconds=30.0, mitm_total_seconds=60.0,
                      mitm_cache=os.path.join(self.directory, "cache"))
        values.update(overrides)
        config = s.SearchConfig(**values)
        key = after(SOLVED, FAR) if key is None else key
        return s.CoevolutionRun(config, key, s.validate_config(config), config.seed, log=self.messages.append,
                                manual_requests=self.requests)

    def advanced(self, generations, **overrides):
        run = self.make_run(**overrides)
        run.budget.start()
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        for generation in range(1, generations + 1):
            run.run_generation(generation)
        return run

    def set_archive(self, run, active):
        active = tuple(s.parse_moves(active) if isinstance(active, str) else active)
        endpoint = s.replay_key(run.input_key, active)
        run.archive = s.ArchiveEntry(active, endpoint, run.objective.cost(endpoint), s.is_solved_key(endpoint),
                                     None, run.completed_generation, "fixture")
        return run.archive


class RequestQueueTests(ManualTestCase):
    def test_m_and_lowercase_m_queue_one_request_without_enter(self):
        control = s.KeyboardControl(self.requests, stdin=FakeTerminal(), out=self.messages.append)
        for key in "xq\n ":
            control.handle_key(key)
        self.assertFalse(self.requests.pending)
        control.handle_key("m")
        self.assertTrue(self.requests.pending)
        self.assertEqual(self.messages, [s.MANUAL_MITM_ACK])
        control.handle_key("M")  # coalesced while pending: no second request, no second message
        self.assertEqual(self.messages, [s.MANUAL_MITM_ACK])
        self.assertTrue(self.requests.take())
        self.requests.finish()
        control.handle_key("M")
        self.assertTrue(self.requests.pending)

    def test_no_backlog_during_an_active_attempt(self):
        self.assertTrue(self.requests.request())
        self.assertFalse(self.requests.request())
        self.assertTrue(self.requests.take())
        for _ in range(20):  # key repeat during the attempt
            self.assertFalse(self.requests.request())
        self.requests.finish()
        self.assertFalse(self.requests.take())  # nothing queued behind the finished attempt
        self.assertTrue(self.requests.request())

    def test_presses_during_any_attempt_are_ignored(self):
        control = s.KeyboardControl(self.requests, stdin=FakeTerminal(), out=self.messages.append)
        accepted = []

        def search(*args, **kwargs):
            control.handle_key("M")  # the user presses M while the search runs
            accepted.append(self.requests.pending)
            return empty_search()

        run = self.make_run(mitm=True, mitm_trigger_cubies=20)
        run.budget.start()
        with mock.patch.object(s, "mitm_search", side_effect=search):
            self.assertEqual(run.finisher_boundary(0)["trigger"], "automatic")
            self.requests.request()
            self.assertEqual(run.finisher_boundary(1)["trigger"], "manual")
        self.assertEqual(accepted, [False, False])  # neither attempt left a request behind
        self.assertNotIn(s.MANUAL_MITM_ACK, self.messages)
        self.assertFalse(self.requests.pending)
        control.handle_key("m")  # after the attempt, a press queues again
        self.assertTrue(self.requests.pending)

    def test_redirected_stdin_is_never_read(self):
        piped = io.StringIO("m\nM\n")
        control = s.KeyboardControl(self.requests, stdin=piped, out=self.messages.append)
        self.assertFalse(control.available)
        self.assertIn("stdin is not an interactive terminal", control.status_line())
        with control:
            pass
        self.assertEqual(piped.tell(), 0)
        self.assertFalse(self.requests.pending)


def terminal_mode(fd):
    """Terminal attributes without PENDIN, which macOS sets itself when line editing comes back on."""
    import termios
    mode = termios.tcgetattr(fd)
    mode[3] &= ~getattr(termios, "PENDIN", 0)
    return mode


@unittest.skipIf(sys.platform == "win32", "pseudo-terminals are POSIX only")
class TerminalTests(ManualTestCase):
    def open_terminal(self):
        import pty
        master, slave = pty.openpty()
        terminal = os.fdopen(slave, "rb", buffering=0)
        self.addCleanup(terminal.close)
        self.addCleanup(os.close, master)
        return master, terminal

    def test_keys_without_enter_on_a_real_terminal_and_mode_restored(self):
        import termios
        master, terminal = self.open_terminal()
        before = terminal_mode(terminal.fileno())
        control = s.KeyboardControl(self.requests, stdin=terminal, out=self.messages.append, poll_seconds=0.05)
        self.assertTrue(control.available)
        with control:
            mode = termios.tcgetattr(terminal.fileno())
            self.assertFalse(mode[3] & termios.ICANON)  # single keys, no Enter
            self.assertFalse(mode[3] & termios.ECHO)
            self.assertTrue(mode[3] & termios.ISIG)  # Ctrl+C still raises SIGINT
            for key in (b"m", b"M"):
                os.write(master, key)
                self.assertTrue(self.requests.wait_for_request(5))
                self.assertTrue(self.requests.take())
                self.requests.finish()
        self.assertEqual(terminal_mode(terminal.fileno()), before)
        self.assertEqual(self.messages, [s.MANUAL_MITM_ACK] * 2)
        self.assertIsNone(control._thread)

    def test_reader_start_failure_restores_the_terminal_and_keeps_solving(self):
        import threading
        _, terminal = self.open_terminal()
        before = terminal_mode(terminal.fileno())
        control = s.KeyboardControl(self.requests, stdin=terminal, out=self.messages.append, poll_seconds=0.05)
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("can't start new thread")):
            with control:  # does not raise: solving continues without the shortcut
                self.assertEqual(terminal_mode(terminal.fileno()), before)
        self.assertFalse(control.available)
        self.assertIn("cannot start the key reader", control.status_line())
        self.assertEqual(terminal_mode(terminal.fileno()), before)

    def test_mode_restored_after_errors_and_ctrl_c(self):
        _, terminal = self.open_terminal()
        before = terminal_mode(terminal.fileno())
        for exception in (RuntimeError("boom"), KeyboardInterrupt()):
            control = s.KeyboardControl(self.requests, stdin=terminal, out=self.messages.append, poll_seconds=0.05)
            with self.assertRaises(type(exception)):
                with control:
                    raise exception
            self.assertEqual(terminal_mode(terminal.fileno()), before)


class ManualAttemptTests(ManualTestCase):
    def test_runs_with_automatic_attempts_off_and_does_not_enable_them(self):
        run = self.make_run(generations=3)
        self.requests.request()
        result = run.run()
        attempts = result.finisher_report["attempts"]
        self.assertEqual([(attempt["generation"], attempt["trigger"]) for attempt in attempts], [(0, "manual")])
        self.assertEqual(result.finisher_report["manual_attempts"], 1)
        self.assertIsNone(run._finisher_last_attempt)  # automatic scheduling untouched
        self.assertTrue(any("manual MITM started" in line for line in self.messages))
        self.assertTrue(any(line.startswith("[finisher gen 0] manual attempt") for line in self.messages))

    def test_latest_archive_and_objective_at_the_boundary(self):
        run = self.make_run(fitness="grouped-alternating")
        run.budget.start()
        self.requests.request()
        entry = self.set_archive(run, "R1 U1")  # changes after the key press
        run._switch_objective("grouped-corners", "test", 1)
        captured = {}

        def search(start, table, depth, cost, prefix, capacity, **kwargs):
            captured.update(start=start, cost=cost, prefix=prefix, depth=depth)
            return empty_search()

        with mock.patch.object(s, "mitm_search", side_effect=search):
            summary = run.finisher_boundary(1)
        self.assertEqual((captured["start"], captured["prefix"]), (entry.endpoint, entry.active))
        self.assertEqual(captured["cost"](entry.endpoint), s.grouped_corners_cost(entry.endpoint))
        self.assertEqual(summary["initial_cost"], s.grouped_corners_cost(entry.endpoint))
        self.assertEqual(captured["depth"], run.config.mitm_depth)

    def test_bypasses_trigger_interval_and_endpoint_suppression_one_attempt_per_boundary(self):
        run = self.make_run(mitm=True, mitm_trigger_cubies=20, mitm_interval=50)
        run.budget.start()
        with mock.patch.object(s, "mitm_search", return_value=empty_search()) as search:
            self.assertEqual(run.finisher_boundary(0)["trigger"], "automatic")
            self.assertIsNone(run.finisher_boundary(1))  # interval and same endpoint suppress it
            self.requests.request()
            summary = run.finisher_boundary(1)
            self.assertEqual((summary["trigger"], summary["generation"]), ("manual", 1))
            self.assertEqual(search.call_count, 2)  # one attempt at boundary 1, the manual one
        run = self.make_run(mitm=True, mitm_trigger_cubies=0)  # the automatic trigger can never fire
        run.budget.start()
        self.requests.request()
        with mock.patch.object(s, "mitm_search", return_value=empty_search()):
            self.assertIsNone(run.finisher_attempt(0))
            self.assertEqual(run.finisher_boundary(0)["trigger"], "manual")

    def test_cumulative_allowance_and_overall_budget(self):
        run = self.make_run()
        run.budget.start()
        run.finisher_seconds = run.config.mitm_total_seconds
        self.requests.request()
        with mock.patch.object(s, "mitm_search") as search:
            self.assertIsNone(run.finisher_boundary(1))
        search.assert_not_called()
        self.assertFalse(self.requests.pending)  # consumed
        self.assertEqual(run.manual_finisher_events[-1]["status"], "skipped")
        self.assertTrue(any("manual MITM skipped: the cumulative finishing time" in line for line in self.messages))
        ticks = iter([0.0] + [100.0] * 10)
        run = self.make_run(time_limit=10.0)
        run.budget = s.Budget(10.0, clock=lambda: next(ticks))
        run.budget.start()
        self.requests.request()
        with mock.patch.object(s, "mitm_search") as search, self.assertRaises(s.BudgetExpired):
            run.finisher_boundary(1)
        search.assert_not_called()
        self.assertEqual(run.manual_finisher_events[-1], {"generation": 1, "status": "not run",
                                                          "reason": "the solver stopped first"})

    def test_ctrl_c_during_the_attempt_and_the_attempt_time(self):
        run = self.make_run(mitm_attempt_seconds=7.0)
        run.budget.start()
        self.requests.request()
        with mock.patch.object(s, "mitm_search", side_effect=s.SearchInterrupted("SIGINT received")):
            with self.assertRaises(s.SearchInterrupted):
                run.finisher_boundary(1)
        self.assertEqual(run.finisher_attempts[-1]["outcome"], "stopped")
        self.assertTrue(self.requests.request())  # the slot is free again
        seen = {}

        def search(*args, deadline=None, **kwargs):
            seen["left"] = deadline - time.monotonic()
            return empty_search()

        with mock.patch.object(s, "mitm_search", side_effect=search):
            run.finisher_boundary(2)
        self.assertTrue(0 < seen["left"] <= 7.0)
        run = self.make_run()
        run.budget.start()
        run.budget.request_interrupt()
        self.requests.request()
        with self.assertRaises(s.SearchInterrupted):
            run.finisher_boundary(1)
        self.assertEqual(run.finisher_attempts, [])

    def test_verified_solve_ends_the_run_normally(self):
        run = self.make_run(key=after(SOLVED, "R1 U1 C1"))
        self.requests.request()
        result = run.run()
        self.assertEqual(result.status, s.STATUS_SOLVED)
        self.assertEqual(result.archive.source, "finisher depth 4")
        final = s.finalize_result(result)
        self.assertTrue(final.solved)
        self.assertEqual(final.problems, [])
        self.assertEqual(result.finisher_report["attempts"][0]["outcome"], "solved")

    def test_improvement_enters_the_population_with_its_invariants(self):
        run = self.advanced(2, mitm_depth=3)
        before = list(run.population)
        fifth = len(before) // 5
        previous = self.set_archive(run, ())  # the empty baseline, which a short path improves
        self.requests.request()
        summary = run.finisher_boundary(2)
        self.assertEqual((summary["trigger"], summary["outcome"]), ("manual", "improved"))
        self.assertLess(run.archive.cost, previous.cost)
        population = run.population
        self.assertEqual(len(population), len(before))
        self.assertEqual([ind.uid for ind in population[:fifth]], [ind.uid for ind in before[:fifth]])
        self.assertTrue(all(ind.slot == "reset" and ind.active == () for ind in population[-fifth:]))
        inserted = [ind for ind in population if ind.origin == "finisher"]
        self.assertEqual(len(inserted), 1)
        record = inserted[0]
        self.assertEqual(record.slot, "offspring")
        self.assertEqual(record.active, run.archive.active)
        self.assertLessEqual(len(record.active), run.solution_capacity)
        self.assertEqual(s.simplify(record.active), record.active)
        self.assertEqual(list(record), list(record.active) + [s.NOOP] * (run.solution_capacity - len(record.active)))
        self.assertEqual(s.state_key(s.replay(s.cube_from_key(run.input_key), record.active)), record.endpoint)

    def test_no_usable_improvement_changes_nothing(self):
        run = self.advanced(2)
        archive, population = run.archive, [ind.uid for ind in run.population]
        self.requests.request()
        with mock.patch.object(s, "mitm_search", return_value=empty_search()):
            summary = run.finisher_boundary(2)
        self.assertEqual(summary["outcome"], "no usable improvement")
        self.assertIs(run.archive, archive)
        self.assertEqual([ind.uid for ind in run.population], population)
        self.assertIn("outcome: no usable improvement", s.finisher_summary_line(summary))

    def test_manual_improvement_postpones_recovery(self):
        run = self.make_run(adaptive_recovery=True, generations=1000)
        run.budget.start()
        record = {"available": True, "flags": ["LOW_UNIQUE_STATES", "NO_QUALITY_PROGRESS"],
                  "generations_without_quality_improvement": 150}
        run.recovery_checkpoint(100, dict(record), None)
        improved = {"trigger": "manual", "result": "improved", "initial_cost": 36, "result_cost": 30}
        run.recovery_checkpoint(150, dict(record), improved)
        self.assertIsNone(run._recovery_pending)
        self.assertEqual(run.recovery_postponements, 1)
        self.assertTrue(any("RECOVERY postponed" in line for line in self.messages))

    def test_no_request_leaves_the_seeded_search_unchanged(self):
        key = after(SOLVED, "R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        config = s.SearchConfig(fitness="manhattan", seed=5, generations=6, solution_population=20,
                                block_population=20, solution_capacity=16, block_capacity=8, contexts=6,
                                min_block_gain=1.0, log_every=10)
        run = s.CoevolutionRun(config, key, s.validate_config(config), 5, log=lambda message: None,
                               manual_requests=s.ManualMitmRequests())
        self.assertEqual(search_digest(run, run.run()), PRE_FEATURE_DIGESTS["manhattan"])


class FakeKeyboard:
    """Stands in for KeyboardControl: presses M once when the run starts."""

    instances = []

    def __init__(self, requests, display_requests=None, press=True):
        self.requests, self.display_requests, self.press, self.exited = requests, display_requests, press, False
        FakeKeyboard.instances.append(self)

    def status_line(self):
        return "Keyboard control: fake"

    def __enter__(self):
        if self.press:
            self.requests.request()
        return self

    def __exit__(self, *exc_info):
        self.exited = True
        return False


class CliTests(ManualTestCase):
    SMALL = ["--solution-population", "10", "--block-population", "10", "--contexts", "4", "--seed", "1",
             "--no-block-library"]

    def main(self, args, keyboard=FakeKeyboard, cube_moves="R1 U1 C1"):
        FakeKeyboard.instances.clear()
        with mock.patch.object(s, "default_input_cube", lambda: s.cube_from_key(after(SOLVED, cube_moves))):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = s.main(args + self.SMALL + ["--mitm-cache-dir", os.path.join(self.directory, "cache")],
                              keyboard=keyboard)
        return code, output.getvalue()

    def test_m_runs_one_attempt_with_no_mitm(self):
        code, text = self.main(["--no-mitm", "--mitm-depth", "4", "--generations", "2"])
        self.assertEqual(code, 0)
        self.assertIn("Finisher: automatic attempts disabled (--no-mitm)", text)
        self.assertIn("Keyboard control: fake", text)
        self.assertIn("manual MITM started on the best-ever candidate", text)
        self.assertIn("[finisher gen 0] manual attempt", text)
        self.assertIn("Finisher: 1 attempt(s) (1 manual)", text)
        self.assertTrue(FakeKeyboard.instances[0].exited)

    def test_redirected_stdin_reports_the_shortcut_unavailable(self):
        with mock.patch.object(sys, "stdin", io.StringIO("m\n")):
            code, text = self.main(["--generations", "0", "--no-mitm"], keyboard=None)
        self.assertEqual(code, 1)
        self.assertIn("Keyboard control: unavailable (stdin is not an interactive terminal)", text)

    def test_reader_closed_after_errors_and_ctrl_c(self):
        for raised, expected in ((s.ConsistencyError("boom"), 3), (KeyboardInterrupt(), 130)):
            with mock.patch.object(s.CoevolutionRun, "run", side_effect=raised):
                code, _ = self.main(["--generations", "2"])
            self.assertEqual(code, expected)
            self.assertTrue(FakeKeyboard.instances[0].exited)

    def test_keyboard_status_is_reported_after_setup(self):
        class FailingSetup(FakeKeyboard):
            available = True

            def status_line(self):
                return "Keyboard control: works" if self.available else "Keyboard control: unavailable (setup)"

            def __enter__(self):
                self.available = False  # e.g. the terminal mode could not be set
                return self

        code, text = self.main(["--generations", "0", "--no-mitm"], keyboard=FailingSetup)
        self.assertEqual(code, 1)
        self.assertIn("Keyboard control: unavailable (setup)", text)
        self.assertNotIn("Keyboard control: works", text)

    def test_request_left_pending_by_a_stop_is_reported(self):
        code, text = self.main(["--generations", "0"])
        self.assertEqual(code, 1)
        self.assertIn("Manual MITM request not run: the solver stopped", text)

    def test_help_explains_no_mitm_and_m(self):
        help_text = " ".join(s.build_arg_parser().format_help().split())
        self.assertIn("pressing M", help_text)
        self.assertIn("--no-mitm turns them off", help_text)


if __name__ == "__main__":
    unittest.main()
