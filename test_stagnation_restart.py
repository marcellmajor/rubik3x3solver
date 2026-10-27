"""Tests for the stagnation-triggered MITM attempt and full GA restart in solver_deap_coevolution.py."""

import contextlib
import dataclasses
import gc
import io
import os
import shutil
import tempfile
import unittest
import weakref
from unittest import mock

import solver_deap_coevolution as s
from test_block_library import PRE_FEATURE_DIGESTS, search_digest

SOLVED = s.REFERENCE_SOLVED_KEY
SCRAMBLE = "R1 U2 C3 D1 B2 L3 R2 C1"
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


INPUT = after(SOLVED, SCRAMBLE)
CLOSER = inverse(SCRAMBLE)[:6]  # undoes six of the eight scramble moves: two moves from solved


def search(termination="exhaustive", alternatives=()):
    """A mocked forward search result; the real result handling processes it."""
    return s.FinisherSearch(termination, 2, 2 if termination == "exhaustive" else 1, 10, [], list(alternatives),
                            [], 0.0)


def idle_generation(self, generation, build_next_blocks=True):
    """A generation that changes nothing, so the stagnation count grows deterministically."""
    self.generation_in_progress = generation
    self.completed_generation = generation


class StagnationTestCase(unittest.TestCase):
    def setUp(self):
        s._SOLVED_SIDE_TABLES.clear()
        self.messages = []

    def tearDown(self):
        s._SOLVED_SIDE_TABLES.clear()

    def config(self, **overrides):
        values = dict(fitness="manhattan", seed=3, generations=10, solution_population=20, block_population=20,
                      contexts=4, solution_capacity=16, block_capacity=8, log_every=10 ** 6, block_library=None,
                      mitm=False, mitm_depth=4, mitm_attempt_seconds=30.0, mitm_total_seconds=60.0, mitm_cache=None,
                      stagnation_restart=True, restart_stagnation_generations=3)
        values.update(overrides)
        return s.SearchConfig(**values)

    def make_run(self, run_options=None, **overrides):
        config = self.config(**overrides)
        run = s.CoevolutionRun(config, INPUT, s.validate_config(config), config.seed, log=self.messages.append,
                               **(run_options or {}))
        run.budget.start()
        run.population = run.initial_solution_population()
        return run

    def session(self, budget=None, **overrides):
        config = self.config(**overrides)
        return s.SearchSession(config, INPUT, s.validate_config(config), config.seed, budget=budget,
                               log=self.messages.append)

    def boundary(self, run, generation):
        run.generation_in_progress = run.completed_generation = generation
        return run.finisher_boundary(generation)

    def candidate(self, run, active):
        return run.normalize_solution(tuple(active), check_budget=False)

    def logged(self, text):
        return [message for message in self.messages if text in message]


class CounterTests(StagnationTestCase):
    def test_trigger_at_exactly_the_configured_interval(self):
        run = self.make_run(restart_stagnation_generations=4)
        with mock.patch.object(s, "mitm_search", return_value=search("timed out")) as searched:
            for generation in range(1, 4):
                self.assertIsNone(self.boundary(run, generation))
            self.assertEqual(searched.call_count, 0)
            summary = self.boundary(run, 4)
        self.assertEqual(searched.call_count, 1)
        self.assertEqual(summary["trigger"], "stagnation")
        self.assertEqual((run.stagnation_events[0]["generation"], run.stagnation_events[0]["stalled_generations"]),
                         (4, 4))
        self.assertTrue(self.logged("[stagnation gen 4] 4 generations without a strict best-ever H improvement"))

    def test_ga_improvement_resets_but_equal_h_shortening_does_not(self):
        run = self.make_run()
        better = self.candidate(run, CLOSER)
        self.assertLess(better.cost, run.archive.cost)
        run.generation_in_progress = 5
        run.offer_to_archive(better, uid=None, source="GA")
        self.assertEqual(run._stagnation_since, 5)
        # The same endpoint and H with two redundant moves: replacing it is an archive update only.
        run.archive = dataclasses.replace(run.archive, active=better.active + moves("U1 U3"))
        run.generation_in_progress = 7
        self.assertTrue(run.offer_to_archive(better, uid=None, source="GA"))
        self.assertEqual(run.archive.active, better.active)
        self.assertEqual(run._stagnation_since, 5)
        self.assertFalse(run._stagnation_due(7))
        self.assertTrue(run._stagnation_due(8))

    def test_objective_switch_restarts_the_count(self):
        run = self.make_run(fitness="grouped-alternating")
        run._switch_objective("grouped-cubie", "test", 7)
        self.assertEqual(run._stagnation_since, 7)
        self.assertFalse(run._stagnation_due(9))
        self.assertTrue(run._stagnation_due(10))

    def test_disabled_makes_no_attempt(self):
        run = self.make_run(stagnation_restart=False)
        with mock.patch.object(s, "mitm_search") as searched:
            for generation in range(1, 8):
                self.boundary(run, generation)
        searched.assert_not_called()
        self.assertEqual(run.stagnation_events, [])


class AttemptTests(StagnationTestCase):
    def test_mitm_improvement_continues_and_resets_the_count(self):
        run = self.make_run()
        self.assertFalse(run.config.mitm)  # stagnation attempts run with --no-mitm
        with mock.patch.object(s, "mitm_search", return_value=search("exhaustive", [CLOSER])):
            summary = self.boundary(run, 3)
        self.assertEqual((summary["trigger"], summary["result"]), ("stagnation", "improved"))
        self.assertEqual(run.archive.active, CLOSER)
        self.assertEqual(run._stagnation_since, 3)
        self.assertEqual(run.stagnation_events[-1]["outcome"], "improved")
        self.assertTrue(self.logged("the population continues and the stagnation count restarts at 0"))
        self.assertFalse(run._stagnation_due(5))
        self.assertTrue(run._stagnation_due(6))

    def test_timed_out_improvement_still_resets_the_count(self):
        run = self.make_run()
        with mock.patch.object(s, "mitm_search", return_value=search("timed out", [CLOSER])):
            summary = self.boundary(run, 3)
        self.assertEqual(summary["result"], "improved")
        self.assertEqual(run._stagnation_since, 3)

    def test_exhausted_search_without_strict_improvement_requests_a_restart(self):
        run = self.make_run()
        closer = self.candidate(run, CLOSER)
        # An equal-H, shorter result (the redundant U1 U3 disappears in the join) does not prevent it.
        run.archive = s.ArchiveEntry(closer.active + moves("U1 U3"), closer.endpoint, closer.cost, False, 7, 0, "fixture")
        with mock.patch.object(s, "mitm_search", return_value=search("exhaustive", [moves("U1 U3")])):
            with self.assertRaises(s.RestartRequested) as raised:
                self.boundary(run, 3)
        self.assertIn("the depth-4 search was exhausted without a strict H improvement", str(raised.exception))
        self.assertEqual(run.stagnation_events[-1]["outcome"], "restart")
        self.assertEqual(run.finisher_attempts[-1]["search_termination"], "exhaustive")
        self.assertTrue(self.logged("restarting the GA"))

    def test_timeout_and_incomplete_processing_never_restart(self):
        run = self.make_run()
        with mock.patch.object(s, "mitm_search", return_value=search("timed out")) as searched:
            self.assertEqual(self.boundary(run, 3)["termination"], "timed out")
            event = run.stagnation_events[-1]
            self.assertEqual(event["outcome"], "no restart")
            self.assertIn("not exhaustive (search timed out)", event["detail"])
            self.assertNotIn("exhausted", " ".join(self.logged("[stagnation gen 3]")))
            for generation in range(4, 6):  # another full interval before the next attempt
                self.boundary(run, generation)
            self.assertEqual(searched.call_count, 1)
            self.boundary(run, 6)
            self.assertEqual(searched.call_count, 2)
        # Exhaustive search, but its results were not all processed before the attempt deadline.
        run = self.make_run(mitm_attempt_seconds=1e-9)
        with mock.patch.object(s, "mitm_search", return_value=search("exhaustive", [moves("U1")])):
            summary = self.boundary(run, 3)
        self.assertIsNotNone(summary["processing"])
        self.assertIn("not exhaustive (result processing stopped at the attempt deadline",
                      run.stagnation_events[-1]["detail"])

    def test_spent_finishing_time_is_reported_not_treated_as_exhausted(self):
        run = self.make_run(run_options={"finisher_seconds": 60.0})
        with mock.patch.object(s, "mitm_search") as searched:
            self.assertIsNone(self.boundary(run, 3))
        searched.assert_not_called()
        self.assertEqual(run.stagnation_events[-1]["outcome"], "no restart")
        self.assertIn("cumulative finishing time (60s, --mitm-total-seconds) is spent",
                      run.stagnation_events[-1]["detail"])
        self.assertEqual(run._stagnation_retry_at, 6)

    def test_manual_request_has_priority_and_the_stagnation_attempt_waits(self):
        requests = s.ManualMitmRequests()
        run = self.make_run(run_options={"manual_requests": requests}, mitm=True, mitm_trigger_cubies=20,
                            mitm_interval=1)
        with mock.patch.object(s, "mitm_search", return_value=search("timed out")):
            requests.request()
            self.assertEqual(self.boundary(run, 3)["trigger"], "manual")
            self.assertTrue(self.logged("[stagnation gen 3] MITM deferred to the next boundary: a manual request"))
            self.assertEqual(run.stagnation_events, [])
            summary = self.boundary(run, 4)  # automatic would be due too; one forward attempt only
        self.assertEqual(summary["trigger"], "stagnation")
        self.assertEqual([attempt["trigger"] for attempt in run.finisher_attempts if attempt["generation"] == 4],
                         ["stagnation"])


class SessionTests(StagnationTestCase):
    def run_session(self, searches, budget=None, **overrides):
        """Idle generations; stagnation attempts get ``searches`` in order."""
        session = self.session(budget=budget, **overrides)
        queue = list(searches)

        def next_search(*args, **kwargs):
            item = queue.pop(0)
            return item(*args, **kwargs) if callable(item) else item

        # A plain function, not a Mock: recorded call arguments would keep finished runs alive.
        with mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                mock.patch.object(s, "mitm_search", next_search):
            result = session.execute()
        return session, result

    def test_exhaustive_miss_restarts_once_from_a_fresh_start(self):
        config_before = dataclasses.asdict(self.config(restart_stagnation_generations=2, generations=6))
        # Attempt 1: improvement at 2, exhausted miss at 4; attempt 2 (generations 5-6): timeout at its 2.
        session, result = self.run_session([search("exhaustive", [CLOSER]), search("exhaustive"), search("timed out")],
                                           restart_stagnation_generations=2, generations=6)
        report = result.stagnation_report
        self.assertEqual(report["restarts"], 1)
        self.assertEqual([event["outcome"] for event in report["events"]], ["improved", "restart", "no restart"])
        first, second = report["attempts"]
        self.assertEqual((first["generations"], first["status"]), (4, s.STATUS_RESTARTED))
        self.assertEqual((second["first_generation"], second["generations"]), (5, 2))
        # The restart is a fresh run: same cube and configuration, nothing from the previous best.
        run = session.run
        self.assertEqual((run.attempt, run.generation_offset, run.input_key), (1, 4, INPUT))
        self.assertIs(run.config, session.config)
        self.assertEqual(dataclasses.asdict(run.config), config_before)
        self.assertEqual(run.history[0]["best_cost"], run.baseline.cost)
        self.assertTrue(all(individual.active == () for individual in run.population))
        self.assertEqual(second["best_moves"], ())
        # Generation limit is global: 4 + 2 generations, not 4 + 6.
        self.assertEqual(result.completed_generations, 6)
        self.assertEqual(result.status, s.STATUS_BUDGET_EXHAUSTED)
        self.assertTrue(self.logged("[restart 1] attempt 1 best:"))
        self.assertTrue(self.logged("[stagnation gen 2 of attempt 2, cumulative 6]"))

    def test_best_result_survives_a_worse_restart(self):
        session, result = self.run_session([search("exhaustive", [CLOSER]), search("exhaustive"), search("timed out")],
                                           restart_stagnation_generations=2, generations=6)
        first, second = result.stagnation_report["attempts"]
        self.assertLess(first["best_cost"], second["best_cost"])
        self.assertEqual(result.archive.active, CLOSER)
        self.assertEqual(result.archive.generation, 2)
        self.assertIn(f"(attempt 1, seed {session.seed})", result.archive.source)
        final = s.finalize_result(result)  # the normal replay verification
        self.assertEqual(final.problems, [])
        self.assertEqual(final.cost, first["best_cost"])

    def test_ctrl_c_after_a_restart_ends_the_session(self):
        session = self.session(restart_stagnation_generations=2, generations=20)

        def log(message):
            self.messages.append(message)
            if message.startswith("[restart 1] attempt 1 best:"):
                session.budget.request_interrupt()  # Ctrl+C while the restart is being set up

        session.log = session._run_log = log
        session.run.log = log
        with mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                mock.patch.object(s, "mitm_search", side_effect=[search("exhaustive", [CLOSER]), search("exhaustive")]):
            result = session.execute()
        self.assertEqual(result.status, s.STATUS_INTERRUPTED)
        self.assertEqual(result.stagnation_report["restarts"], 1)
        self.assertEqual(result.archive.active, CLOSER)
        self.assertEqual(s.finalize_result(result).problems, [])

    def test_cancellation_during_the_attempt_is_not_an_exhausted_search(self):
        session = self.session()

        def interrupted(*args, check=None, **kwargs):
            session.budget.request_interrupt()
            check()

        with mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                mock.patch.object(s, "mitm_search", side_effect=interrupted):
            result = session.execute()
        self.assertEqual(result.status, s.STATUS_INTERRUPTED)
        self.assertEqual(result.stagnation_report["restarts"], 0)
        self.assertEqual(session.finished, [])
        self.assertEqual(result.finisher_report["attempts"][-1]["result"], "abandoned")

    def test_restart_seeds_are_deterministic_and_different(self):
        self.assertEqual(s.restart_seed(3, 1), s.restart_seed(3, 1))
        self.assertEqual(len({3, s.restart_seed(3, 1), s.restart_seed(3, 2), s.restart_seed(4, 1)}), 4)
        searches = [search("exhaustive"), search("exhaustive"), search("timed out")]
        seeds = [self.run_session(list(searches), generations=8)[1].stagnation_report["seeds"] for _ in range(2)]
        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(seeds[0], [3, s.restart_seed(3, 1), s.restart_seed(3, 2)])
        for seed in seeds[0][1:]:
            self.assertTrue(self.logged(f"starts from the original cube with seed {seed};"))
        with mock.patch.object(s, "restart_seed", return_value=3):  # a derived seed never repeats an earlier one
            _, result = self.run_session([search("exhaustive"), search("timed out")], generations=5)
        self.assertEqual(result.stagnation_report["seeds"], [3, 4])

    def test_global_limits_survive_restarts(self):
        now = [1000.0]
        budget = s.Budget(500.0, clock=lambda: now[0])
        starts = []
        original_start = s.Budget.start

        def counted_start(self):
            starts.append(self.clock())
            original_start(self)

        def slow_search(*args, **kwargs):
            now[0] += 100.0  # time passes during attempt 1
            return search("exhaustive")

        with mock.patch.object(s.Budget, "start", counted_start):
            session, result = self.run_session([slow_search, search("timed out")], budget=budget, generations=6)
        self.assertEqual(starts, [1000.0])  # one clock start for the session
        self.assertEqual(budget.deadline, 1500.0)
        self.assertEqual(result.search_seconds, 100.0)
        first_seconds = session.finished[0][1].finisher_report["seconds"]
        self.assertGreater(first_seconds, 0.0)
        self.assertGreaterEqual(session.run.finisher_seconds, first_seconds)
        self.assertEqual(result.finisher_report["seconds"], session.run.finisher_seconds)
        self.assertEqual(len(result.finisher_report["attempts"]), 2)
        self.assertEqual(session.run.generation_limit, 3)
        self.assertEqual(result.completed_generations, 6)

    def test_previous_attempt_library_is_finalised_before_the_restart(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        calls = []
        original_finish = s.CoevolutionRun.finish_block_library

        def spy(self, allow_save=True):
            calls.append((self.attempt, allow_save))
            return original_finish(self, allow_save)

        config = self.config(generations=5, block_library=os.path.join(directory, "blocks.json"))
        with mock.patch.object(s.CoevolutionRun, "finish_block_library", spy), \
                mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                mock.patch.object(s, "mitm_search", side_effect=[search("exhaustive"), search("timed out")]):
            result = s.run_search(config, s.cube_from_key(INPUT), log=self.messages.append)
        self.assertEqual(calls, [(0, True), (1, True)])  # attempt 1 before attempt 2 existed, then the last one
        self.assertEqual(result.stagnation_report["restarts"], 1)

    def test_a_restart_releases_the_finished_run(self):
        runs = []
        original_init = s.CoevolutionRun.__init__

        def tracked(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            runs.append(weakref.ref(self))

        with mock.patch.object(s.CoevolutionRun, "__init__", tracked):
            session, result = self.run_session([search("exhaustive"), search("exhaustive")], generations=8)
        gc.collect()
        self.assertEqual(result.stagnation_report["restarts"], 2)
        self.assertEqual([ref() is None for ref in runs], [True, True, False])  # populations and caches freed
        self.assertIs(runs[-1](), session.run)
        self.assertEqual([attempt.seed for attempt, _ in session.finished], result.stagnation_report["seeds"][:2])
        self.assertEqual(len(result.stagnation_report["events"]), 2)

    def test_the_cli_releases_finished_runs_too(self):
        class NoKeyboard:
            def __init__(self, requests, display_requests=None):
                pass

            def status_line(self):
                return "Keyboard control: fake"

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        runs, alive = [], []
        original_init, original_run = s.CoevolutionRun.__init__, s.CoevolutionRun.run

        def tracked(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            runs.append(weakref.ref(self))

        def observed(self):
            gc.collect()
            alive.append([number for number, ref in enumerate(runs, 1) if ref() is not None])
            return original_run(self)

        searches = iter([search("exhaustive"), search("exhaustive")])
        args = ["--fitness", "manhattan", "--seed", "3", "--generations", "8", "--solution-population", "20",
                "--block-population", "20", "--contexts", "4", "--solution-capacity", "16", "--block-capacity", "8",
                "--no-block-library", "--no-mitm", "--mitm-depth", "4", "--no-mitm-cache",
                "--restart-stagnation-generations", "3"]
        with mock.patch.object(s, "default_input_cube", lambda: s.cube_from_key(INPUT)), \
                mock.patch.object(s.CoevolutionRun, "__init__", tracked), \
                mock.patch.object(s.CoevolutionRun, "run", observed), \
                mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                mock.patch.object(s, "mitm_search", lambda *args, **kwargs: next(searches)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = s.main(args, keyboard=NoKeyboard)
        self.assertEqual(code, s.EXIT_CODES[s.STATUS_BUDGET_EXHAUSTED])
        self.assertIn("Stagnation restart: 2 restart(s)", output.getvalue())
        self.assertEqual(alive, [[1], [2], [3]])  # each attempt runs alone

    def tampered_results(self, statuses):
        """Make the archived cost of results with these statuses disagree with their replay."""
        original = s.CoevolutionRun._result

        def tampered(run, status, reason):
            result = original(run, status, reason)
            if result.status in statuses:
                result = dataclasses.replace(result, archive=dataclasses.replace(result.archive,
                                                                                 cost=result.archive.cost - 1))
            return result
        return mock.patch.object(s.CoevolutionRun, "_result", tampered)

    def test_failed_replay_ends_the_session_instead_of_restarting(self):
        calls = []
        original_finish = s.CoevolutionRun.finish_block_library

        def spy(self, allow_save=True):
            calls.append((self.attempt, allow_save))
            return original_finish(self, allow_save)

        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        config = self.config(generations=12, block_library=os.path.join(directory, "blocks.json"))
        searches = iter([search("exhaustive"), search("exhaustive"), search("exhaustive")])
        with self.tampered_results({s.STATUS_RESTARTED}), \
                mock.patch.object(s.CoevolutionRun, "finish_block_library", spy), \
                mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                mock.patch.object(s, "mitm_search", lambda *args, **kwargs: next(searches)):
            result = s.run_search(config, s.cube_from_key(INPUT), log=self.messages.append)
        self.assertEqual(result.status, s.STATUS_ERROR)
        self.assertEqual(result.completed_generations, 3)
        self.assertEqual(result.stagnation_report["restarts"], 0)
        self.assertEqual(len(list(searches)), 2)  # no second attempt
        self.assertEqual(calls, [(0, False)])  # no save before the restart, none after the failure
        final = s.finalize_result(result)
        self.assertEqual(final.status, s.STATUS_ERROR)
        self.assertTrue(any("differs from the archived cost" in problem for problem in final.problems))
        self.assertTrue(self.logged("[restart 1] not started: attempt 1's best candidate failed replay"))

    def test_an_earlier_best_never_hides_a_last_candidate_that_fails_replay(self):
        with self.tampered_results({s.STATUS_BUDGET_EXHAUSTED}):
            session, result = self.run_session([search("exhaustive", [CLOSER]), search("exhaustive"),
                                                search("timed out")], restart_stagnation_generations=2,
                                               generations=6)
        self.assertEqual(result.stagnation_report["restarts"], 1)
        first, second = result.stagnation_report["attempts"]
        self.assertLess(first["best_cost"], second["best_cost"] + 1)  # better than attempt 2's true cost
        self.assertNotEqual(result.archive.active, CLOSER)
        self.assertIn("(attempt 2,", result.archive.source)
        self.assertEqual(s.finalize_result(result).status, s.STATUS_ERROR)

    def test_a_stop_while_finishing_the_attempt_starts_no_new_attempt(self):
        for stop, status, reason in (("interrupt", s.STATUS_INTERRUPTED, "SIGINT received; restart 1 not started"),
                                     ("deadline", s.STATUS_BUDGET_EXHAUSTED,
                                      "time limit reached; restart 1 not started")):
            with self.subTest(stop=stop):
                self.messages.clear()
                now = [0.0]
                budget = s.Budget(100.0, clock=lambda: now[0])
                session = self.session(budget=budget, generations=20)
                original_finish = s.CoevolutionRun.finish_block_library

                def stopping(run, allow_save=True):
                    if stop == "interrupt":
                        budget.request_interrupt()  # Ctrl+C during the library save
                    else:
                        now[0] = 100.0  # the time limit passes during the save
                    return original_finish(run, allow_save)

                def no_new_attempt(run, *args, **kwargs):
                    raise AssertionError("a new attempt was set up after the stop")

                with mock.patch.object(s.CoevolutionRun, "finish_block_library", stopping), \
                        mock.patch.object(s.CoevolutionRun, "__init__", no_new_attempt), \
                        mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                        mock.patch.object(s, "mitm_search", lambda *args, **kwargs: search("exhaustive", [CLOSER])
                                          if not session.run.archive.active else search("exhaustive")):
                    result = session.execute()
                self.assertEqual((result.status, result.reason), (status, reason))
                self.assertEqual(session.run.attempt, 0)
                self.assertEqual(result.stagnation_report["restarts"], 0)
                self.assertEqual(result.archive.active, CLOSER)
                self.assertEqual(s.finalize_result(result).problems, [])
                self.assertTrue(self.logged("[restart 1] not started:"))


class ReportingTests(StagnationTestCase):
    def test_cli_options_and_help(self):
        parser = s.build_arg_parser()
        defaults = s.config_from_args(parser.parse_args([]))
        self.assertEqual((defaults.stagnation_restart, defaults.restart_stagnation_generations), (True, 500))
        options = s.config_from_args(parser.parse_args(["--no-stagnation-restart",
                                                        "--restart-stagnation-generations", "7"]))
        self.assertEqual((options.stagnation_restart, options.restart_stagnation_generations), (False, 7))
        with self.assertRaises(s.ConfigError):
            s.validate_config(self.config(restart_stagnation_generations=0))
        text = " ".join(parser.format_help().split())
        self.assertIn("runs even with --no-mitm, which turns off only the automatic attempts", text)
        self.assertIn("pressing M still requests one attempt, and stagnation attempts still run unless", text)
        self.assertIn("restarts do not replenish them", text)

    def test_startup_and_final_report(self):
        cube = s.cube_from_key(INPUT)
        for enabled, expected in ((True, "Stagnation restart: after 3 generations without a strict best-ever H "
                                         "improvement, one MITM attempt (depth 4, also with --no-mitm)"),
                                  (False, "Stagnation restart: disabled (--no-stagnation-restart)")):
            config = self.config(stagnation_restart=enabled)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                s._print_startup(config, s.validate_config(config), 3, False, cube)
            self.assertIn(expected, output.getvalue())
        session = self.session(generations=5)  # attempt 2 (generations 4-5) never reaches the trigger
        with mock.patch.object(s.CoevolutionRun, "run_generation", idle_generation), \
                mock.patch.object(s, "mitm_search", side_effect=[search("exhaustive")]):
            result = session.execute()
            session.run.report_progress()
        self.assertTrue(self.logged("[gen 2 of attempt 2, cumulative 5 |"))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            s._print_final(result, s.finalize_result(result), 0.0)
        text = output.getvalue()
        self.assertIn(f"Stagnation restart: 1 restart(s) after 1 stagnation MITM request(s) (trigger 3 "
                      f"generations); seeds 3, {s.restart_seed(3, 1)}", text)
        self.assertIn("attempt 1 (seed 3, generations 1-3): best H=", text)
        self.assertIn("; ended: restart", text)
        self.assertIn(f"attempt 2 (seed {s.restart_seed(3, 1)}, generations 4-5)", text)
        self.assertIn("Completed generations: 5", text)


class UnchangedTests(StagnationTestCase):
    def test_seeded_search_unchanged_when_disabled_or_not_triggered(self):
        key = after(SOLVED, "R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        for enabled in (False, True):
            config = s.SearchConfig(fitness="manhattan", seed=5, generations=6, solution_population=20,
                                    block_population=20, solution_capacity=16, block_capacity=8, contexts=6,
                                    min_block_gain=1.0, log_every=10, stagnation_restart=enabled)
            session = s.SearchSession(config, key, s.validate_config(config), 5, log=lambda message: None)
            result = session.execute()
            self.assertEqual(search_digest(session.run, result), PRE_FEATURE_DIGESTS["manhattan"])
            self.assertEqual(result.stagnation_report["restarts"], 0)


if __name__ == "__main__":
    unittest.main()
