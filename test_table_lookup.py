"""Tests for the population-wide exact solved-side table checks (part of automatic MITM)."""

import contextlib
import dataclasses
import io
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import solver_deap_coevolution as s

SOLVED = s.REFERENCE_SOLVED_KEY
P1 = "R2 U2 R2 U2 R2 U2"  # (R2 U2)^3 == (U2 R2)^3: two reduced spellings of one cube state
P2 = "U2 R2 U2 R2 U2 R2"
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


# The input is chosen so that X + P1 lands on solved + "C1 U1", two moves from solved.
INPUT = after(SOLVED, moves("C1 U1") + inverse(P1))


class LookupTestCase(unittest.TestCase):
    def setUp(self):
        s._SOLVED_SIDE_TABLES.clear()
        self.directory = tempfile.mkdtemp()
        self.messages = []

    def tearDown(self):
        s._SOLVED_SIDE_TABLES.clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    def make_run(self, key=INPUT, **overrides):
        values = dict(fitness="manhattan", seed=3, generations=10, solution_population=20, block_population=20,
                      contexts=4, solution_capacity=16, block_capacity=8, log_every=10 ** 6, block_library=None,
                      mitm=True, mitm_depth=4, mitm_trigger_cubies=0, mitm_interval=50, mitm_attempt_seconds=30.0,
                      mitm_total_seconds=60.0, mitm_cache=os.path.join(self.directory, "cache"))
        values.update(overrides)
        config = s.SearchConfig(**values)
        run = s.CoevolutionRun(config, key, s.validate_config(config), config.seed, log=self.messages.append)
        run.budget.start()
        return run

    def individual(self, run, sequence, origin="fixture"):
        active = moves(sequence) if isinstance(sequence, str) else tuple(sequence)
        endpoint = s.replay_key(run.input_key, active)
        candidate = s.NormalizedCandidate(active, endpoint, run.objective.cost(endpoint), s.is_solved_key(endpoint))
        return run.make_solution(candidate, origin=origin)

    def far(self, run, count=3):
        """Candidates that are not within two moves of solved."""
        return [self.individual(run, text) for text in ("L1", "D2 B1", "B3 L2 D1")[:count]]

    def lookup_lines(self):
        return [line for line in self.messages if line.startswith("[table lookup gen")]


class LookupHitTests(LookupTestCase):
    def test_a_non_best_candidate_in_the_table_solves_the_cube(self):
        run = self.make_run()
        near = self.individual(run, P1)  # two moves from solved, worse ranked than the far ones may be
        run.population = self.far(run) + [near]
        self.assertGreater(s.misplaced_cubies(near.endpoint)[0] + s.misplaced_cubies(near.endpoint)[1],
                           run.config.mitm_trigger_cubies)  # the forward trigger would never pick it
        with self.assertRaises(s.SolutionFound):
            run.finisher_boundary(1)
        summary = run.lookup_passes[-1]
        self.assertEqual((summary["hits"], summary["solved"], summary["scan"]), (1, True, "completed"))
        self.assertIn(f"S#{near.uid}", summary["source"])
        self.assertTrue(run.archive.solved)
        self.assertIn(f"table lookup (radius 2) from S#{near.uid}", run.archive.source)
        self.assertEqual(summary["continuation_length"], 2)
        final = s.finalize_result(run._result(s.STATUS_SOLVED, "test"))  # independent cube.py replay
        self.assertTrue(final.solved)
        self.assertEqual(final.problems, [])
        self.assertTrue(any("via S#" in line and "2-move table continuation" in line for line in self.lookup_lines()))

    def test_the_continuation_is_the_tables_shortest_path(self):
        run = self.make_run()
        near = self.individual(run, P1)
        run.population = [near]
        table = s.prepare_solved_side_table(2, None)
        centres = "".join(INPUT[start + 4] for start in s.FACE_STARTS)
        state = near.endpoint.translate(str.maketrans(centres, "012345"))
        continuation = table.path_to_solved(state)
        self.assertEqual(continuation, moves("U3 C3"))
        self.assertEqual(after(near.endpoint, continuation), SOLVED)
        with self.assertRaises(s.SolutionFound):
            run.finisher_boundary(1)
        self.assertEqual(run._library_pending, {})  # no library here; the payload is tested separately

    def test_duplicate_endpoints_are_looked_up_once_and_each_source_is_tried(self):
        run = self.make_run()
        a, b = self.individual(run, P1), self.individual(run, P2)
        self.assertEqual(a.endpoint, b.endpoint)
        _, reduced_a = run._join_reduce(a.active, moves("U3 C3"))
        _, reduced_b = run._join_reduce(b.active, moves("U3 C3"))
        self.assertLess(len(reduced_a), len(reduced_b))  # only A's last move merges with the continuation
        run = self.make_run(solution_capacity=len(reduced_a), block_capacity=4)
        a, b = self.individual(run, P1), self.individual(run, P2)
        run.population = [b, b, a]  # B (infeasible) first, a copy of B, then A
        with self.assertRaises(s.SolutionFound):
            run.finisher_boundary(1)
        summary = run.lookup_passes[-1]
        self.assertEqual((summary["unique"], summary["hits"]), (2, 1))  # archive input + one shared endpoint
        self.assertEqual((summary["sources_tried"], summary["over_capacity"]), (2, 1))
        self.assertIn(f"S#{a.uid}", summary["source"])
        self.assertLess(summary["combined_after"], summary["combined_before"])  # fits only after the merge

    def test_an_infeasible_hit_then_a_feasible_one(self):
        run = self.make_run()
        b = self.individual(run, P2)
        one_away = self.individual(run, moves(P1) + moves("U3"))  # solved + "C1": one move away
        _, reduced_b = run._join_reduce(b.active, moves("U3 C3"))
        _, reduced_d = run._join_reduce(one_away.active, moves("C3"))
        capacity = len(reduced_d)
        self.assertGreater(len(reduced_b), capacity)
        run = self.make_run(solution_capacity=capacity, block_capacity=4)
        b = self.individual(run, P2)
        one_away = self.individual(run, moves(P1) + moves("U3"))
        run.population = [b, one_away]
        with self.assertRaises(s.SolutionFound):
            run.finisher_boundary(1)
        summary = run.lookup_passes[-1]
        self.assertEqual((summary["hits"], summary["over_capacity"]), (2, 1))
        self.assertIn(f"S#{one_away.uid}", summary["source"])

    def test_the_solving_continuation_goes_to_the_block_library(self):
        run = self.make_run(block_library=os.path.join(self.directory, "blocks.json"))
        run.population = [self.individual(run, P1)]
        with self.assertRaises(s.SolutionFound):
            run.finisher_boundary(1)
        self.assertIn(moves("U3 C3"), run._library_pending)


class LookupMissTests(LookupTestCase):
    def test_a_miss_changes_nothing_and_rules_out_only_the_table_radius(self):
        run = self.make_run()
        run.population = self.far(run)
        state = (run.rng.getstate(), dataclasses.asdict(run.counters), run.archive,
                 [ind.uid for ind in run.population])
        self.assertIsNone(run.finisher_boundary(1))
        summary = run.lookup_passes[-1]
        self.assertEqual((summary["hits"], summary["solved"], summary["scan"]), (0, False, "completed"))
        self.assertIn("no endpoint within 2 moves", summary["result"])
        self.assertIn("4-move MITM radius, are not ruled out", summary["result"])
        self.assertNotIn("more than", self.lookup_lines()[-1])
        self.assertEqual((run.rng.getstate(), dataclasses.asdict(run.counters), run.archive,
                          [ind.uid for ind in run.population]), state)

    def test_incomplete_preparation_and_scan(self):
        run = self.make_run()
        run.population = [self.individual(run, P1)]
        with mock.patch.object(s, "prepare_solved_side_table", side_effect=s._FinisherTimeout()):
            summary, deadline = run.table_lookup_pass(0)
        self.assertIsNone(deadline)
        self.assertIn("preparation incomplete", summary["result"])
        self.assertFalse(run._lookup_due(10))  # retried at the next scheduled pass, not earlier
        self.assertTrue(run._lookup_due(50))
        run = self.make_run(mitm_attempt_seconds=1e-9)  # the scan deadline has passed before it starts
        run.population = [self.individual(run, P1)]
        summary, _ = run.table_lookup_pass(0)
        self.assertEqual(summary["scan"], "interrupted after 0 of 2 candidates")
        self.assertIn("no match among the endpoints checked before the attempt deadline", summary["result"])
        self.assertFalse(run.archive.solved)


class SchedulingTests(LookupTestCase):
    def test_lookups_keep_their_own_schedule_without_suppressing_forward_attempts(self):
        run = self.make_run(mitm_trigger_cubies=20, mitm_interval=2)
        run.population = self.far(run)
        with mock.patch.object(s, "mitm_search",
                               return_value=s.FinisherSearch("exhaustive", 2, 2, 0, [], [], [], 0.0)) as search:
            for generation in range(5):
                run.finisher_boundary(generation)
        self.assertEqual([lookup["generation"] for lookup in run.lookup_passes], [0, 2, 4])
        self.assertEqual([attempt["generation"] for attempt in run.finisher_attempts], [0])  # one endpoint
        self.assertEqual(search.call_count, 1)

    def test_a_forward_attempt_shares_the_boundary_allowance(self):
        run = self.make_run(mitm_trigger_cubies=20, mitm_attempt_seconds=30.0)
        run.population = self.far(run)
        seen = {}
        original = run.table_lookup_pass

        def lookup(generation):
            summary, deadline = original(generation)
            seen["lookup_deadline"] = deadline
            return summary, deadline

        def search(*args, deadline=None, **kwargs):
            seen["search_deadline"] = deadline
            return s.FinisherSearch("exhaustive", 2, 2, 0, [], [], [], 0.0)

        with mock.patch.object(run, "table_lookup_pass", side_effect=lookup), \
                mock.patch.object(s, "mitm_search", side_effect=search):
            run.finisher_boundary(0)
        self.assertLessEqual(seen["search_deadline"], seen["lookup_deadline"])  # no second full allowance
        run = self.make_run(mitm_trigger_cubies=20)
        run.population = self.far(run)
        with mock.patch.object(run, "table_lookup_pass", return_value=({}, time.monotonic() - 1)), \
                mock.patch.object(s, "mitm_search") as search:
            self.assertIsNone(run.finisher_boundary(0))
        search.assert_not_called()
        self.assertIn("table lookup", run.finisher_skip_reason)
        self.assertIsNone(run._finisher_last_attempt)  # the forward attempt is not postponed

    def test_no_mitm_and_manual_priority(self):
        run = self.make_run(mitm=False)
        run.population = [self.individual(run, P1)]
        self.assertIsNone(run.finisher_boundary(0))
        self.assertEqual((run.lookup_passes, s._SOLVED_SIDE_TABLES), ([], {}))
        requests = s.ManualMitmRequests()
        run = self.make_run()
        run.manual_requests = requests
        run.population = self.far(run)
        requests.request()
        with mock.patch.object(s, "mitm_search", return_value=s.FinisherSearch("exhaustive", 2, 2, 0, [], [], [],
                                                                               0.0)):
            self.assertEqual(run.finisher_boundary(0)["trigger"], "manual")
            self.assertEqual(run.lookup_passes, [])  # deferred
            run.finisher_boundary(1)
        self.assertEqual([lookup["generation"] for lookup in run.lookup_passes], [1])

    def test_current_objective_space_and_ctrl_c(self):
        run = self.make_run(fitness="grouped-alternating")
        run._switch_objective("grouped-cubie", "test", 0)
        run.population = [self.individual(run, P1)]
        with self.assertRaises(s.SolutionFound):
            run.finisher_boundary(1)
        self.assertEqual(run.archive.cost, run.objective.cost(run.archive.endpoint))
        display = s.DisplayRequests()
        shown = []
        run = self.make_run()
        run.display_requests = display
        run.budget.display_requests, run.budget.on_display = display, run._service_display
        run.on_display = shown.append
        run.population = self.far(run)
        table = s.prepare_solved_side_table(2, None)

        def prepared_then_space(*args, **kwargs):
            display.request()  # Space pressed once the lookup is under way
            return table

        with mock.patch.object(s, "prepare_solved_side_table", side_effect=prepared_then_space):
            run.finisher_boundary(1)
        self.assertEqual(len(shown), 1)
        self.assertTrue(shown[0]["when"].startswith("during a table lookup"))
        run = self.make_run()
        run.population = self.far(run)
        before = [ind.uid for ind in run.population]
        run.budget.display_requests = display
        run.budget.on_display = run.budget.request_interrupt  # Ctrl+C arrives during the scan
        display.request()
        with self.assertRaises(s.SearchInterrupted):
            run.finisher_boundary(1)
        self.assertEqual([ind.uid for ind in run.population], before)
        self.assertFalse(run.archive.solved)

    def test_startup_text_mentions_the_population_checks(self):
        args = ["--generations", "0", "--no-block-library", "--mitm-depth", "4"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            s.main(args)
        self.assertIn("exact table checks of the whole population (within 2 moves, no misplaced-cubie trigger)",
                      output.getvalue())


if __name__ == "__main__":
    unittest.main()
