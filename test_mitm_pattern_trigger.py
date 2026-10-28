"""Tests for the depth-12 pattern-distance trigger of automatic finisher attempts in solver_deap_coevolution.py."""

import contextlib
import io
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import solver_deap_coevolution as s

SOLVED = s.REFERENCE_SOLVED_KEY
# Real cubes. Their P and misplaced counts come from the real pattern tables (checked below).
P24 = "L1 B1 R1 C1 R1 C3 U3 L1"  # P = 24; 8 corners + 11 edges misplaced
P25 = "R1 B3 R1 C1 U1 C2 R1 B2 U2 C3"  # P = 25; 8 corners + 11 edges misplaced
U_PERM = "R2 U1 R1 U1 R3 U3 R3 U3 R3 U1 R3"  # P = 2; 0 corners + 3 edges: the cubie trigger
TWO_U_PERMS = U_PERM + " L2 D1 L1 D1 L3 D3 L3 D3 L3 D1 L3"  # P = 7; corners home, 6 edges
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


def moves(text):
    return tuple(s.parse_moves(text)) if text else ()


def inverse(text):
    return tuple(s.inverse_allele(allele) for allele in reversed(moves(text)))


def after(key, sequence):
    sequence = moves(sequence) if isinstance(sequence, str) else tuple(sequence)
    return s.state_key(s.replay(s.cube_from_key(key), sequence))


def search(termination="exhaustive", alternatives=(), solving=()):
    """A mocked depth-12 forward search result; the real result handling processes it."""
    return s.FinisherSearch(termination, 6, 6 if termination == "exhaustive" else 3, 10, list(solving),
                            list(alternatives), [], 0.0)


class PatternTriggerTestCase(unittest.TestCase):
    def setUp(self):
        s._SOLVED_SIDE_TABLES.clear()
        self.messages = []
        # The forward search is mocked; a small real table stands in for the 8.2M-state depth-6 one.
        table = s.prepare_solved_side_table(2, None)
        patcher = mock.patch.object(s, "prepare_solved_side_table", lambda *args, **kwargs: table)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        s._SOLVED_SIDE_TABLES.clear()

    def config(self, **overrides):
        values = dict(fitness="manhattan", seed=3, generations=10, solution_population=20, block_population=20,
                      contexts=4, solution_capacity=24, block_capacity=8, log_every=10 ** 6, block_library=None,
                      mitm=True, mitm_depth=12, mitm_pattern_trigger=24,  # explicit threshold for P24/P25 fixtures
                      mitm_attempt_seconds=30.0, mitm_total_seconds=60.0, mitm_cache=None)
        values.update(overrides)
        return s.SearchConfig(**values)

    def make_run(self, scramble, run_options=None, **overrides):
        config = self.config(**overrides)
        run = s.CoevolutionRun(config, after(SOLVED, scramble), s.validate_config(config), config.seed,
                               log=self.messages.append, **(run_options or {}))
        run.budget.start()
        run.population = run.initial_solution_population()
        return run

    def attempt(self, run, generation=0, result=None, **kwargs):
        """One automatic finisher check at ``generation`` with a mocked forward search."""
        with mock.patch.object(s, "mitm_search", return_value=result or search()) as searched:
            summary = run.finisher_attempt(generation, **kwargs)
        return summary, searched

    def boundary(self, run, generation, result=None):
        run.generation_in_progress = run.completed_generation = generation
        with mock.patch.object(s, "mitm_search", return_value=result or search("timed out")) as searched:
            summary = run.finisher_boundary(generation)
        return summary, searched

    @contextlib.contextmanager
    def pattern_spy(self, forbidden=False):
        """Count P computations; ``forbidden`` fails the test if P is computed at all."""
        calls = []
        real = s.pattern_distance_sum

        def spy(key):
            if forbidden:
                raise AssertionError("P computed although it was not needed")
            calls.append(key)
            return real(key)

        with mock.patch.object(s, "pattern_distance_sum", spy):
            yield calls

    def logged(self, text):
        return [message for message in self.messages if text in message]


class PatternDistanceTests(PatternTriggerTestCase):
    def test_real_fixtures_have_the_expected_pattern_sums(self):
        tables = s._grouped_cubie_tables()
        for scramble, expected, misplaced in ((P24, 24, (8, 11)), (P25, 25, (8, 11)), (U_PERM, 2, (0, 3)),
                                              (TWO_U_PERMS, 7, (0, 6)), ("", 0, (0, 0))):
            with self.subTest(scramble=scramble):
                key = after(SOLVED, scramble)
                distances = tables.distances(s._face_indices(key))
                self.assertEqual(len(distances), 5)  # two corner and three edge groups
                self.assertEqual(s.pattern_distance_sum(key), expected)
                self.assertEqual(sum(distances), expected)
                # Independent check: P is the top tier of the grouped-cubie cost (4096 * P + lower tiers).
                self.assertEqual(s.grouped_cubie_cost(key) // 4096, expected)
                self.assertEqual(s.misplaced_cubies(key), misplaced)

    def test_p_is_centre_relative_and_not_a_fitness_value(self):
        key = after(SOLVED, P24)
        colours = sorted(set(key))
        recoloured = key.translate(str.maketrans("".join(colours), "".join(colours[1:] + colours[:1])))
        self.assertNotEqual(recoloured, key)
        self.assertEqual(s.pattern_distance_sum(recoloured), 24)  # read relative to the face centres
        self.assertNotEqual(s.grouped_corners_cost(key) // 4096, 24)  # corner patterns weighted double there
        self.assertNotIn(24, (s.grouped_cubie_cost(key), s.mix_cost(key), s.manhattan_cost(key)))


class EligibilityTests(PatternTriggerTestCase):
    def test_depth_12_candidate_failing_both_original_triggers_qualifies(self):
        run = self.make_run(P24)
        with self.pattern_spy() as calls:
            summary, searched = self.attempt(run)
        self.assertEqual(len(calls), 1)
        self.assertEqual(searched.call_count, 1)
        self.assertEqual(searched.call_args.args[2], 12)  # the configured depth-12 search
        self.assertEqual((summary["trigger"], summary["eligibility"]), ("automatic", "pattern distance"))
        self.assertEqual((summary["pattern_distance"], summary["pattern_threshold"]), (24, 24))
        self.assertEqual((summary["depth"], summary["corners"], summary["edges"]), (12, 8, 11))

    def test_p_24_qualifies_and_p_25_does_not(self):
        summary, _ = self.attempt(self.make_run(P24))
        self.assertEqual(summary["eligibility"], "pattern distance")
        run = self.make_run(P25)
        summary, searched = self.attempt(run)
        self.assertIsNone(summary)
        self.assertEqual(searched.call_count, 0)
        self.assertEqual(run.finisher_skip_reason,
                         "trigger not met: 8 corners + 11 edges misplaced, pattern sum P=25 > 24")
        summary, _ = self.attempt(self.make_run(P25, mitm_pattern_trigger=25))
        self.assertEqual(summary["pattern_distance"], 25)

    def test_original_triggers_admit_without_computing_p(self):
        for scramble, fitness, eligibility in ((U_PERM, "manhattan", "misplaced cubies"),
                                               (TWO_U_PERMS, "corners-first", "corners-first edges")):
            for threshold in (1, 0):  # the pattern condition would fail (P = 2 or 7 > 1), or is off
                with self.subTest(fitness=fitness, threshold=threshold), self.pattern_spy(forbidden=True):
                    summary, searched = self.attempt(self.make_run(scramble, fitness=fitness,
                                                                   mitm_pattern_trigger=threshold))
                self.assertEqual(searched.call_count, 1)
                self.assertEqual(summary["eligibility"], eligibility)
                self.assertIsNone(summary["pattern_distance"])
                self.assertIsNone(summary["pattern_threshold"])
        # Without the corners-first fitness the same cube needs the pattern condition: P = 7.
        run = self.make_run(TWO_U_PERMS, mitm_pattern_trigger=1)
        self.assertIsNone(self.attempt(run)[0])
        self.assertTrue(run.finisher_skip_reason.endswith("pattern sum P=7 > 1"))
        self.assertEqual(self.attempt(self.make_run(TWO_U_PERMS))[0]["pattern_distance"], 7)

    def test_zero_restores_the_previous_eligibility(self):
        with self.pattern_spy(forbidden=True):
            run = self.make_run(P24, mitm_pattern_trigger=0)
            summary, searched = self.attempt(run)
            self.assertIsNone(summary)
            self.assertEqual(searched.call_count, 0)
            self.assertEqual(run.finisher_skip_reason, "trigger not met: 8 corners + 11 edges misplaced")
            self.assertEqual(self.attempt(self.make_run(U_PERM, mitm_pattern_trigger=0))[0]["eligibility"],
                             "misplaced cubies")
        self.assertFalse(s.pattern_trigger_active(self.config(mitm_pattern_trigger=0)))

    def test_other_depths_keep_their_previous_eligibility(self):
        for depth in (4, 10, 11):
            with self.subTest(depth=depth), self.pattern_spy(forbidden=True):
                run = self.make_run(P24, mitm_depth=depth)
                summary, searched = self.attempt(run)
                self.assertIsNone(summary)
                self.assertEqual(searched.call_count, 0)
                self.assertEqual(run.finisher_skip_reason, "trigger not met: 8 corners + 11 edges misplaced")
                summary, _ = self.attempt(self.make_run(U_PERM, mitm_depth=depth))
                self.assertEqual((summary["eligibility"], summary["depth"]), ("misplaced cubies", depth))
                self.assertFalse(s.pattern_trigger_active(self.config(mitm_depth=depth)))

    def test_p_is_the_same_whatever_the_fitness(self):
        for fitness in s.FITNESS_CHOICES:
            with self.subTest(fitness=fitness):
                summary, _ = self.attempt(self.make_run(P24, fitness=fitness))
                self.assertEqual((summary["eligibility"], summary["pattern_distance"]), ("pattern distance", 24))
                run = self.make_run(P25, fitness=fitness)
                self.assertIsNone(self.attempt(run)[0])
                self.assertIn("pattern sum P=25 > 24", run.finisher_skip_reason)


class ControlTests(PatternTriggerTestCase):
    """The pattern condition decides eligibility only; every other control still applies."""

    def test_no_mitm_runs_no_automatic_attempt(self):
        with self.pattern_spy(forbidden=True):
            run = self.make_run(P24, mitm=False)
            summary, searched = self.boundary(run, 0)
        self.assertIsNone(summary)
        self.assertEqual(searched.call_count, 0)
        self.assertEqual(run.finisher_skip_reason, "finisher disabled")
        self.assertEqual(run.lookup_passes, [])

    def test_interval_and_same_endpoint_suppression(self):
        run = self.make_run(P24)
        self.assertEqual(self.attempt(run, 0, search("exhaustive"))[0]["eligibility"], "pattern distance")
        summary, searched = self.attempt(run, 10)
        self.assertIsNone(summary)
        self.assertEqual(searched.call_count, 0)
        self.assertEqual(run.finisher_skip_reason, "interval: last attempt at generation 0")
        summary, searched = self.attempt(run, 50)  # due again, but this endpoint was already searched
        self.assertIsNone(summary)
        self.assertEqual(searched.call_count, 0)
        self.assertEqual(run.finisher_skip_reason, "endpoint already searched (exhausted the depth-12 search "
                                                   "without a solution)")

    def test_budgets_and_cancellation(self):
        run = self.make_run(P24)
        run.finisher_seconds = run.config.mitm_total_seconds
        self.assertIsNone(self.attempt(run)[0])
        self.assertEqual(run.finisher_skip_reason, "total finishing time spent")
        run = self.make_run(P24)
        self.assertIsNone(self.attempt(run, shared_deadline=time.monotonic() - 1.0)[0])
        self.assertEqual(run.finisher_skip_reason, "no attempt time left after this boundary's table lookup")
        run = self.make_run(P24)
        run.budget.request_interrupt()
        with self.assertRaises(s.SearchInterrupted):
            self.attempt(run)
        self.assertEqual(run.finisher_attempts, [])
        run = self.make_run(P24, mitm_attempt_seconds=5.0)
        before = time.monotonic()
        _, searched = self.attempt(run)
        self.assertLessEqual(searched.call_args.kwargs["deadline"], before + 5.0 + 1.0)  # per-attempt time

    def test_existing_result_handling_is_reused(self):
        run = self.make_run(P24)
        initial = run.archive.cost
        summary, _ = self.attempt(run, result=search("exhaustive", [inverse(P24)[:6]]))
        self.assertEqual(summary["result"], "improved")
        self.assertEqual(run.archive.source, "finisher depth 12")
        self.assertLess(run.archive.cost, initial)
        self.assertEqual(after(after(SOLVED, P24), run.archive.active), run.archive.endpoint)  # replay matches
        self.assertEqual(summary["population"], "queued for generation 1's offspring")
        run = self.make_run(P24, solution_capacity=1, block_capacity=1)  # the improvement does not fit
        self.assertEqual(self.attempt(run, result=search("exhaustive", [inverse(P24)[:6]]))[0]["result"],
                         "no feasible improving fallback")
        run = self.make_run(P24)
        with self.assertRaises(s.SolutionFound):
            self.attempt(run, result=search("solution found", solving=[inverse(P24)]))
        self.assertTrue(run.archive.solved)
        self.assertEqual(run.finisher_attempts[-1]["result"], "solved")

    def test_a_miss_or_timeout_never_restarts_the_ga(self):
        for termination in ("exhaustive", "timed out"):
            with self.subTest(termination=termination):
                run = self.make_run(P24, stagnation_restart=True, restart_stagnation_generations=500)
                summary, searched = self.boundary(run, 0, search(termination))
                self.assertEqual((summary["trigger"], summary["eligibility"]), ("automatic", "pattern distance"))
                self.assertEqual(summary["result"], "no feasible improving fallback")
                self.assertEqual(searched.call_count, 1)  # one forward attempt at this boundary
                self.assertEqual(len(run.lookup_passes), 1)  # the population lookups keep their schedule
                self.assertEqual(run.stagnation_events, [])


class PathTests(PatternTriggerTestCase):
    def test_manual_request_takes_the_boundary_without_computing_p(self):
        requests = s.ManualMitmRequests()
        requests.request()
        run = self.make_run(P24, run_options={"manual_requests": requests})
        with self.pattern_spy(forbidden=True):
            summary, searched = self.boundary(run, 0)
        self.assertEqual(searched.call_count, 1)
        self.assertEqual((summary["trigger"], summary["eligibility"], summary["pattern_distance"]),
                         ("manual", None, None))
        self.assertEqual(run.lookup_passes, [])  # the lookups wait for the next boundary, as before
        self.assertEqual(run.finisher_report()["manual_attempts"], 1)

    def test_stagnation_attempt_takes_the_boundary_without_computing_p(self):
        run = self.make_run(P24, mitm=False, stagnation_restart=True, restart_stagnation_generations=1)
        with self.pattern_spy(forbidden=True):
            summary, searched = self.boundary(run, 1)
        self.assertEqual(searched.call_count, 1)
        self.assertEqual((summary["trigger"], summary["eligibility"]), ("stagnation", None))
        self.assertEqual(run.stagnation_events[-1]["outcome"], "no restart")  # timed out: retried later
        run = self.make_run(P24, stagnation_restart=True, restart_stagnation_generations=1)
        with self.pattern_spy(forbidden=True):
            summary, searched = self.boundary(run, 1)  # the automatic attempt would also be due
        self.assertEqual(searched.call_count, 1)
        self.assertEqual(summary["trigger"], "stagnation")


class ConfigAndReportTests(PatternTriggerTestCase):
    def test_configuration_validation(self):
        for value in (0, 1, 24, 38, 100):
            s.validate_config(self.config(mitm_pattern_trigger=value))
        for value in (-1, True, False, 2.0, "24", None):
            with self.subTest(value=value), self.assertRaises(s.ConfigError):
                s.validate_config(self.config(mitm_pattern_trigger=value))
        self.assertEqual(s.SearchConfig().mitm_pattern_trigger, 12)

    def test_cli_option(self):
        parser = s.build_arg_parser()
        self.assertEqual(s.config_from_args(parser.parse_args([])).mitm_pattern_trigger, 12)
        self.assertEqual(s.config_from_args(parser.parse_args(["--mitm-pattern-trigger", "0"])).mitm_pattern_trigger,
                         0)
        text = " ".join(parser.format_help().split())
        self.assertIn("at --mitm-depth 12 only, also attempt when the sum P", text)
        self.assertIn("an experimental closeness heuristic, not a bound; 0 turns it off (default 12)", text)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = s.main(["--mitm-pattern-trigger", "-1", "--no-block-library"])
        self.assertEqual(code, s.EXIT_CODES[s.STATUS_INVALID_INPUT])

    def test_startup_line(self):
        cube = s.cube_from_key(after(SOLVED, P24))
        for overrides, present, absent in (
                ({}, "attempt at <=4 misplaced cubies or pattern sum P<=24 (experimental), every", "inactive"),
                ({"mitm_depth": 10}, "(pattern trigger: depth 12 only, inactive)", "pattern sum P<="),
                ({"mitm_pattern_trigger": 0}, "attempt at <=4 misplaced cubies, every", "pattern")):
            config = self.config(**overrides)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                s._print_startup(config, s.validate_config(config), 3, False, cube)
            line = next(line for line in output.getvalue().splitlines() if line.startswith("Finisher:"))
            self.assertIn(present, line)
            self.assertNotIn(absent, line)

    def test_admission_attempt_line_and_final_report(self):
        run = self.make_run(P24)
        self.attempt(run)
        self.assertEqual(self.logged("pattern trigger admits"),
                         ["[finisher gen 0] pattern trigger admits an automatic attempt: P=24 <= 24 at depth 12; "
                          "8 corners + 11 edges misplaced"])
        line = self.logged("[finisher gen 0] automatic attempt")[0]
        self.assertIn("automatic attempt (pattern trigger: P=24 <= 24), best-ever", line)
        self.assertIn("8 corners + 11 edges misplaced", line)
        self.assertIn("depth 12 = table 6 + forward 6/6", line)
        report = run.finisher_report()
        self.assertEqual((report["pattern_trigger"], report["pattern_trigger_active"]), (24, True))
        self.assertEqual(report["attempts"][0]["eligibility"], "pattern distance")
        result = run._result(s.STATUS_BUDGET_EXHAUSTED, "test")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            s._print_final(result, s.finalize_result(result), 0.0)
        self.assertIn("Finisher: 1 attempt(s) (0 manual, 1 admitted by the pattern trigger), finishing time",
                      output.getvalue())
        # An attempt admitted by the original trigger keeps the previous lines.
        self.messages.clear()
        run = self.make_run(U_PERM)
        self.attempt(run)
        self.assertEqual(self.logged("pattern trigger"), [])
        self.assertTrue(self.logged("[finisher gen 0] automatic attempt, best-ever"))
        result = run._result(s.STATUS_BUDGET_EXHAUSTED, "test")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            s._print_final(result, s.finalize_result(result), 0.0)
        self.assertIn("Finisher: 1 attempt(s) (0 manual), finishing time", output.getvalue())


if __name__ == "__main__":
    unittest.main()
