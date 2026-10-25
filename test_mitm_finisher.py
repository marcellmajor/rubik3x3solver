"""Tests for the optional bounded meet-in-the-middle finisher (--mitm) in solver_deap_coevolution.py."""

import contextlib
import io
import json
import os
import random
import shutil
import tempfile
import time
import unittest
from collections import deque
from unittest import mock

import solver_deap_coevolution as s
from test_block_library import PRE_FEATURE_DIGESTS, search_digest

SOLVED = s.REFERENCE_SOLVED_KEY
# Optimal 10-move solution of an edge 3-cycle (0 corners + 3 edges misplaced).
THREE_CYCLE_SOLUTION = "R1 C1 B3 D3 L2 D1 C3 B1 R3 D2"
_saved_environment = {}


def setUpModule():
    # Keep the real shared caches and block library untouched.
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
    """State after ``moves``, replayed independently through cube.py's rotations."""
    sequence = s.parse_moves(moves) if isinstance(moves, str) else tuple(moves)
    return s.state_key(s.replay(s.cube_from_key(key), sequence))


def scramble_of(solution):
    """The state that ``solution`` takes to solved."""
    moves = s.parse_moves(solution) if isinstance(solution, str) else tuple(solution)
    return after(SOLVED, tuple(s.inverse_allele(allele) for allele in reversed(moves)))


def raw_key(face_indices):
    """A face-index string (as stored in the table) in the reference colours."""
    return "".join("GRWOYB"[int(digit)] for digit in face_indices)


def bfs_distance(start, limit):
    """Independent optimal distance by plain breadth-first search (no pruning), or None beyond ``limit``."""
    if start == SOLVED:
        return 0
    seen, frontier = {start}, [start]
    for distance in range(1, limit + 1):
        following = []
        for key in frontier:
            for allele in s.REAL_MOVES:
                child = s.replay_key(key, (allele,))
                if child == SOLVED:
                    return distance
                if child not in seen:
                    seen.add(child)
                    following.append(child)
        frontier = following
    return None


def recolour(key, mapping="GRWOYB", to="OBYGRW"):
    return key.translate(str.maketrans(mapping, to))


def zero_cost(key):
    return 0


class FinisherTestCase(unittest.TestCase):
    def setUp(self):
        s._SOLVED_SIDE_TABLES.clear()
        self.directory = tempfile.mkdtemp()
        self.cache = os.path.join(self.directory, "cache")
        self.messages = []

    def tearDown(self):
        s._SOLVED_SIDE_TABLES.clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    def table(self, depth):
        return s.prepare_solved_side_table(depth, None)

    def make_run(self, key, **overrides):
        values = dict(fitness="manhattan", seed=3, generations=3, solution_population=20, block_population=20,
                      contexts=6, solution_capacity=16, block_capacity=8, min_block_gain=1.0, log_every=10 ** 6,
                      block_library=None, mitm=True, mitm_depth=4, mitm_trigger_cubies=20, mitm_interval=50,
                      mitm_attempt_seconds=30.0, mitm_total_seconds=60.0, mitm_cache=self.cache)
        values.update(overrides)
        config = s.SearchConfig(**values)
        objective = s.validate_config(config)
        return s.CoevolutionRun(config, key, objective, config.seed, log=self.messages.append)

    def set_archive(self, run, active):
        """Make ``active`` the best-ever entry directly (bypassing the strict-improvement rule)."""
        active = tuple(s.parse_moves(active) if isinstance(active, str) else active)
        endpoint = s.replay_key(run.input_key, active)
        run.archive = s.ArchiveEntry(active, endpoint, run.objective.cost(endpoint), s.is_solved_key(endpoint),
                                     None, run.completed_generation, "fixture")
        return run.archive

    def finisher_lines(self):
        return [message for message in self.messages if message.startswith("[finisher gen")]


class MisplacedCubieTests(FinisherTestCase):
    def test_position_and_orientation_are_counted_per_piece(self):
        twist = "R3 D3 R1 D1 " * 2 + "U1 " + "R3 D3 R1 D1 " * 4 + "U3"
        cases = {"": (0, 0), "R1": (4, 4), twist: (2, 0), "R3 C1 R3 B2 R1 C3 R3 B2 R2": (3, 0),
                 "R2 U2 " * 6: (0, 0)}
        for moves, expected in cases.items():
            self.assertEqual(s.misplaced_cubies(after(SOLVED, moves.strip())), expected, moves)
        self.assertEqual(s.misplaced_cubies(scramble_of(THREE_CYCLE_SOLUTION)), (0, 3))

    def test_counts_are_relative_to_the_centres(self):
        key = scramble_of(THREE_CYCLE_SOLUTION)
        self.assertEqual(s.misplaced_cubies(recolour(key)), (0, 3))
        self.assertEqual(s.misplaced_cubies(recolour(SOLVED)), (0, 0))

    def test_inconsistent_stickers_are_rejected(self):
        with self.assertRaises(ValueError):
            s.misplaced_cubies("G" * 54)


class SolvedSideTableTests(FinisherTestCase):
    def test_build_matches_the_exact_distance_counts_and_an_independent_bfs(self):
        table = self.table(3)
        counts = [0] * 4
        for value in table.entries.values():
            counts[value // 19] += 1
        self.assertEqual(tuple(counts), s.MITM_HALF_TURN_COUNTS[:4])
        # Independent: plain BFS over cube.py replays to depth 3 reaches exactly the stored states.
        seen, frontier = {SOLVED: 0}, [SOLVED]
        for distance in (1, 2, 3):
            following = []
            for key in frontier:
                for allele in s.REAL_MOVES:
                    child = after(key, (allele,))
                    if child not in seen:
                        seen[child] = distance
                        following.append(child)
            frontier = following
        stored = {raw_key(s._mitm_decode(code)): value // 19 for code, value in table.entries.items()}
        self.assertEqual(stored, seen)

    def test_paths_to_solved_replay_through_cube_py(self):
        table = self.table(3)
        rng = random.Random(4)
        for code in rng.sample(sorted(table.entries), 200):
            state = s._mitm_decode(code)
            path = table.path_to_solved(state)
            self.assertEqual(len(path), table.distance(code))
            self.assertEqual(after(raw_key(state), path), SOLVED)

    def test_inverse_allele_undoes_every_move(self):
        for allele in s.REAL_MOVES:
            self.assertEqual(after(SOLVED, (allele, s.inverse_allele(allele))), SOLVED)

    def test_a_stopped_build_resumes_to_the_same_table(self):
        calls = {"count": 0}

        def check():
            calls["count"] += 1
            if calls["count"] == 3:
                raise s.BudgetExpired("stop")

        table = s.SolvedSideTable(3)
        with self.assertRaises(s.BudgetExpired):
            table.extend(check)
        self.assertFalse(table.complete)
        with self.assertRaises(s.ConsistencyError):
            s.mitm_search(SOLVED, table, 6, zero_cost, (), 16)  # an incomplete table is never consulted
        table.extend()
        self.assertTrue(table.complete)
        fresh = s.SolvedSideTable(3)
        fresh.extend()
        self.assertEqual(table.entries, fresh.entries)

    def test_preparation_resumes_a_stopped_in_memory_build(self):
        with self.assertRaises(s._FinisherTimeout):
            s.prepare_solved_side_table(3, None, check=mock.Mock(side_effect=[None, s._FinisherTimeout()]))
        partial = s._SOLVED_SIDE_TABLES[3]
        self.assertFalse(partial.complete)
        table = s.prepare_solved_side_table(3, None)
        self.assertIs(table, partial)
        self.assertTrue(table.complete)
        self.assertEqual(len(table.entries), sum(s.MITM_HALF_TURN_COUNTS[:4]))


class SearchTests(FinisherTestCase):
    SCRAMBLES = ("R1", "U2 C3", "R1 U1 C1", "D2 L1 B3", "R1 L1 U2", "C1 R2 U3 B1", "R1 U1 R3 U3", "L2 D3 C1 B2 R1")

    def test_shortest_solutions_match_an_independent_bfs(self):
        keys = {moves: after(SOLVED, moves) for moves in self.SCRAMBLES}
        optimum = {moves: bfs_distance(key, 4) for moves, key in keys.items()}
        tables = {depth: self.table(depth) for depth in (0, 1, 2)}
        for depth in (1, 2, 3, 4):
            table = tables[depth // 2]
            for moves, key in keys.items():
                search = s.mitm_search(key, table, depth, zero_cost, (), 16)
                expected = optimum[moves] if optimum[moves] is not None and optimum[moves] <= depth else None
                if expected is None:
                    self.assertEqual(search.solving, [], (depth, moves))
                    self.assertEqual(search.termination, "exhaustive")
                    self.assertEqual(search.forward_depth, depth - depth // 2)
                else:
                    self.assertEqual(len(search.solving[0]), expected, (depth, moves))
                    self.assertIn(search.termination, ("exhaustive", "solution found"))
                for path in search.solving:
                    self.assertLessEqual(len(path), depth)
                    self.assertEqual(after(key, path), SOLVED)  # forward part + inverted table part

    def test_odd_and_even_depths_split_into_table_floor_and_forward_ceil(self):
        key = after(SOLVED, "C1 R2 U3 B1 D1")
        for depth in (3, 4, 5):
            search = s.mitm_search(key, self.table(depth // 2), depth, zero_cost, (), 16)
            self.assertEqual(search.forward_target, depth - depth // 2)
        self.assertEqual(len(s.mitm_search(key, self.table(2), 5, zero_cost, (), 16).solving[0]), 5)

    def test_recoloured_cubes_find_the_same_paths(self):
        table = self.table(2)
        for moves in self.SCRAMBLES[:6]:
            key = after(SOLVED, moves)
            plain = s.mitm_search(key, table, 4, zero_cost, (), 16)
            coloured = s.mitm_search(recolour(key), table, 4, zero_cost, (), 16)
            self.assertEqual(plain.solving, coloured.solving)
            self.assertEqual((plain.states, plain.termination), (coloured.states, coloured.termination))

    def test_temporary_fitness_regressions_are_not_pruned(self):
        key = after(SOLVED, "R1 U1 C1 L1")

        def cost(state):  # every intermediate state looks far worse than the start
            return 0 if s.is_solved_key(state) else 1 if state == key else 100

        search = s.mitm_search(key, self.table(2), 4, cost, (), 16)
        self.assertEqual(len(search.solving[0]), 4)
        self.assertEqual(search.alternatives, [])

    def test_early_stop_once_no_shorter_solution_is_possible(self):
        key = after(SOLVED, "R1 U1")
        search = s.mitm_search(key, self.table(3), 6, zero_cost, (), 16)
        self.assertEqual(search.termination, "solution found")
        self.assertEqual(search.forward_depth, 1)
        self.assertEqual(len(search.solving[0]), 2)

    def test_fallback_keeps_the_best_strictly_improving_forward_paths(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        start_cost = s.manhattan_cost(key)
        search = s.mitm_search(key, self.table(1), 3, s.manhattan_cost, (), 16)
        self.assertEqual(search.solving, [])
        self.assertTrue(search.alternatives)
        self.assertLessEqual(len(search.alternatives), s.MITM_FALLBACK_ALTERNATIVES)
        costs = [s.manhattan_cost(s.replay_key(key, path)) for path in search.alternatives]
        self.assertTrue(all(cost < start_cost for cost in costs))
        self.assertEqual(costs, sorted(costs))
        self.assertTrue(all(len(path) <= 2 for path in search.alternatives))
        # Independent check: no forward path of at most two moves beats the best kept one.
        best = min(s.manhattan_cost(s.replay_key(key, (a, b) if b else (a,)))
                   for a in s.REAL_MOVES for b in (None,) + s.REAL_MOVES)
        self.assertEqual(costs[0], best)

    def test_no_improvement_is_reported_without_alternatives(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        search = s.mitm_search(key, self.table(1), 3, zero_cost, (), 16)
        self.assertEqual((search.solving, search.alternatives, search.over_capacity), ([], [], []))
        self.assertEqual(search.termination, "exhaustive")

    def test_fallback_paths_are_split_by_capacity_after_simplification(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        prefix = s.parse_moves("L1 D2 B1 C3")  # unrelated prefix: only cancelling paths stay within 4 moves
        search = s.mitm_search(key, self.table(1), 3, s.manhattan_cost, prefix, 4)
        self.assertTrue(search.over_capacity)
        self.assertLessEqual(len(search.over_capacity), s.MITM_OVER_CAPACITY_ALTERNATIVES)
        for path in search.alternatives:
            self.assertLessEqual(len(s.simplify(prefix + path)), 4)
        for path in search.over_capacity:
            self.assertGreater(len(s.simplify(prefix + path)), 4)

    def test_deadline_and_overall_stop(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        table = self.table(1)
        search = s.mitm_search(key, table, 3, zero_cost, (), 16, deadline=0.0, clock=lambda: 1.0)
        self.assertEqual((search.termination, search.forward_depth, search.solving), ("timed out", 0, []))
        with self.assertRaises(s.SearchInterrupted):
            s.mitm_search(key, table, 3, zero_cost, (), 16, check=mock.Mock(side_effect=s.SearchInterrupted("x")))

    def test_depth_ten_finds_the_optimal_ten_move_three_cycle(self):
        key = scramble_of(THREE_CYCLE_SOLUTION)
        table = s.prepare_solved_side_table(5, self.cache)
        self.assertEqual(len(table.entries), sum(s.MITM_HALF_TURN_COUNTS))
        search = s.mitm_search(key, table, 10, zero_cost, (), 72)
        self.assertIn(search.termination, ("solution found", "exhaustive"))
        self.assertEqual(len(search.solving[0]), 10)
        self.assertEqual(after(key, search.solving[0]), SOLVED)
        self.assertEqual(s.mitm_search(key, table, 9, zero_cost, (), 72).solving, [])  # no 9-move solution


class DiskCacheTests(FinisherTestCase):
    def path(self, depth=3):
        return s.mitm_cache_file(self.cache, depth)

    def test_round_trip_and_atomic_write(self):
        built = s.prepare_solved_side_table(3, self.cache)
        self.assertEqual(built.source, "built")
        self.assertEqual(os.listdir(self.cache), [os.path.basename(self.path())])  # no temporary files left
        loaded = s.load_solved_side_table(self.cache, 3)
        self.assertEqual((loaded.entries, loaded.source, loaded.complete), (built.entries, "disk cache", True))
        s._SOLVED_SIDE_TABLES.clear()
        self.assertEqual(s.prepare_solved_side_table(3, self.cache).source, "disk cache")
        self.assertEqual(s.prepare_solved_side_table(3, self.cache).source, "memory")
        with open(self.path(), "rb") as handle:
            header = json.loads(handle.readline())
        self.assertEqual((header["format"], header["version"], header["depth"]),
                         (s.MITM_CACHE_FORMAT, s.MITM_CACHE_VERSION, 3))

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(s.load_solved_side_table(self.cache, 3))

    def rewrite(self, transform):
        s.prepare_solved_side_table(3, self.cache)
        s._SOLVED_SIDE_TABLES.clear()
        with open(self.path(), "rb") as handle:
            header, body = handle.read().split(b"\n", 1)
        header, body = transform(json.loads(header), bytearray(body))
        with open(self.path(), "wb") as handle:
            handle.write(json.dumps(header, sort_keys=True).encode() + b"\n" + bytes(body))

    def assert_rebuilt(self, message):
        with self.assertRaisesRegex(ValueError, message):
            s.load_solved_side_table(self.cache, 3)
        table = s.prepare_solved_side_table(3, self.cache, report=self.messages.append)
        self.assertEqual(table.source, "built")
        self.assertTrue(any("rebuilding" in line for line in self.messages), self.messages)
        s._SOLVED_SIDE_TABLES.clear()
        self.assertEqual(s.load_solved_side_table(self.cache, 3).entries, table.entries)  # repaired on disk

    def test_stale_version_or_move_model_is_rebuilt(self):
        def stale(header, body):
            header["fingerprint"] = "0" * 64
            return header, body

        self.rewrite(stale)
        self.assert_rebuilt("another format version or move model")

    def test_corrupt_body_is_rebuilt(self):
        def corrupt(header, body):
            body[100] ^= 1
            return header, body

        self.rewrite(corrupt)
        self.assert_rebuilt("truncated or corrupt")

    def test_truncated_and_garbage_files_are_rebuilt(self):
        self.rewrite(lambda header, body: (header, body[:-19]))
        self.assert_rebuilt("truncated or corrupt")
        with open(self.path(), "wb") as handle:
            handle.write(b"not a table")
        s._SOLVED_SIDE_TABLES.clear()
        self.assert_rebuilt("wrong size or no header")

    def test_wrong_move_links_are_rebuilt(self):
        record = s.MITM_CODE_BYTES + 1

        def relink(header, body):
            offset = 997 * record
            code = int.from_bytes(body[offset:offset + s.MITM_CODE_BYTES], "big")
            distance = body[offset + s.MITM_CODE_BYTES] // 19
            table = self.table(3)
            for allele in s.REAL_MOVES:  # a move whose undo does not reach distance - 1
                parent = "".join(s._MOVE_TABLES[s.inverse_allele(allele)](s._mitm_decode(code)))
                if table.distance(int(parent, 6)) != distance - 1:
                    body[offset + s.MITM_CODE_BYTES] = distance * 19 + allele
                    break
            header["body_sha256"] = s.hashlib.sha256(bytes(body)).hexdigest()
            return header, body

        self.rewrite(relink)
        s._SOLVED_SIDE_TABLES.clear()
        self.assert_rebuilt("move links")

    def test_unwritable_cache_keeps_the_table_in_memory(self):
        blocked = os.path.join(self.directory, "file")
        with open(blocked, "w"):
            pass
        table = s.prepare_solved_side_table(2, blocked, report=self.messages.append)
        self.assertTrue(table.complete)
        self.assertTrue(any("cannot write" in line for line in self.messages), self.messages)

    def test_a_stopped_cache_write_leaves_nothing_and_is_retried(self):
        s.prepare_solved_side_table(3, None)  # complete in memory, never written
        with self.assertRaises(s._FinisherTimeout):
            s.prepare_solved_side_table(3, self.cache, check=mock.Mock(side_effect=s._FinisherTimeout()))
        self.assertFalse(os.path.exists(self.cache))  # no file, no temporary file, no directory
        table = s.prepare_solved_side_table(3, self.cache)
        self.assertEqual(table.source, "memory")
        self.assertEqual(s.load_solved_side_table(self.cache, 3).entries, table.entries)
        with mock.patch.object(s, "save_solved_side_table") as save:
            s.prepare_solved_side_table(3, self.cache)
        save.assert_not_called()  # written once per directory

    def test_cache_files_are_deterministic_without_sorting(self):
        s.prepare_solved_side_table(3, self.cache)
        with open(self.path(), "rb") as handle:
            first = handle.read()
        s._SOLVED_SIDE_TABLES.clear()
        shutil.rmtree(self.cache)
        s.prepare_solved_side_table(3, self.cache)
        with open(self.path(), "rb") as handle:
            self.assertEqual(handle.read(), first)

    def test_memory_only_mode_writes_nothing(self):
        s.prepare_solved_side_table(2, None)
        self.assertFalse(os.path.exists(self.cache))


class RunFinisherTests(FinisherTestCase):
    FAR = "R1 U2 C3 D1 B2 L3 R2 C1"

    def test_solves_after_setup_through_the_normal_archive_path(self):
        run = self.make_run(after(SOLVED, "R1 U1 C1 L1"))
        result = run.run()
        self.assertEqual(result.status, s.STATUS_SOLVED)
        self.assertEqual(result.completed_generations, 0)
        self.assertEqual(result.archive.source, "finisher depth 4")
        final = s.finalize_result(result)
        self.assertTrue(final.solved)
        self.assertEqual(final.problems, [])
        attempt = result.finisher_report["attempts"][0]
        self.assertEqual((attempt["generation"], attempt["result"], attempt["result_solved"]), (0, "solved", True))
        self.assertEqual(len(self.finisher_lines()), 1)
        self.assertIn("solved=yes", self.finisher_lines()[0])
        self.assertGreater(result.finisher_report["preparation_seconds"], 0)
        self.assertEqual(result.finisher_report["table_source"], "built")

    def test_simplification_across_the_join(self):
        run = self.make_run(after(SOLVED, "R1 U1 C1"))
        run.budget.start()
        self.set_archive(run, "C3 L1")  # overshoots: the suffix starts by undoing L1
        with self.assertRaises(s.SolutionFound):
            run.finisher_attempt(0)
        attempt = run.finisher_attempts[-1]
        self.assertEqual((attempt["suffix_length"], attempt["combined_before"], attempt["combined_after"]), (3, 5, 3))
        self.assertEqual(run.archive.active, s.parse_moves("C3 U3 R3"))

    def test_raw_proposal_over_capacity_fits_after_reduction(self):
        run = self.make_run(after(SOLVED, "R1 U1 C1"), solution_capacity=3, block_capacity=2)
        run.budget.start()
        self.set_archive(run, "C3 L1")
        with self.assertRaises(s.SolutionFound):
            run.finisher_attempt(0)
        self.assertEqual(len(run.archive.active), 3)

    def test_boundary_reduction_removes_cancellations_and_state_loops(self):
        run = self.make_run(after(SOLVED, self.FAR))
        entry = self.set_archive(run, "U1 R1")
        proposal, reduced = run._finisher_reduce(entry, s.parse_moves("R3 C2"))
        self.assertEqual((len(proposal), reduced), (4, s.parse_moves("U1 C2")))
        entry = self.set_archive(run, ())
        _, reduced = run._finisher_reduce(entry, s.parse_moves("R2 U2 " * 6 + "C1"))
        self.assertEqual(reduced, s.parse_moves("C1"))

    def fabricated(self, solving=(), alternatives=(), over=(), termination="exhaustive"):
        return s.FinisherSearch(termination, 2, 2, 0, [s.parse_moves(p) for p in solving],
                                [s.parse_moves(p) for p in alternatives], [s.parse_moves(p) for p in over], 0.0)

    def blank_summary(self):
        return {"termination": "exhaustive", "result": None, "suffix_length": None, "combined_before": None,
                "combined_after": None, "result_cost": None, "result_solved": False, "archive": False,
                "population": "-", "candidates": 0, "candidates_checked": 0, "processing": None}

    def overshooting_run(self, capacity=4):
        """Best-ever ``L3 B3 R1`` overshoots: a fitting path reaches H 150, an oversized one H 96."""
        run = self.make_run(after(SOLVED, "R1 U1 C1 D1 B1 L1"), solution_capacity=capacity, block_capacity=2)
        run.budget.start()
        entry = self.set_archive(run, "L3 B3 R1")
        self.assertEqual(entry.cost, 156)
        return run, entry

    def test_best_final_result_wins_across_fitting_and_oversized_paths(self):
        run, entry = self.overshooting_run()
        # "R3" fits at once (L3 B3, H 150); "R3 D3 C3" is 6 raw moves but reduces to 4 (H 96).
        search = self.fabricated(alternatives=["R3"], over=["R3 D3 C3"])
        with mock.patch.object(s, "mitm_search", return_value=search):
            summary = run.finisher_attempt(0)
        self.assertEqual(summary["result"], "improved")
        self.assertEqual((run.archive.active, run.archive.cost), (s.parse_moves("L3 B3 D3 C3"), 96))
        self.assertEqual((summary["combined_before"], summary["combined_after"]), (6, 4))
        self.assertEqual((summary["candidates"], summary["candidates_checked"]), (2, 2))
        replayed = s.state_key(s.replay(s.cube_from_key(run.input_key), run.archive.active))
        self.assertEqual(run.objective.cost(replayed), 96)

    def test_processing_stops_at_the_attempt_deadline_with_the_best_so_far(self):
        run, entry = self.overshooting_run()
        search = self.fabricated(alternatives=["R3", "R3 D3 C3"])
        summary = self.blank_summary()
        check = mock.Mock(side_effect=[None, s._FinisherTimeout()])
        self.assertIsNone(run._finisher_apply(entry, search, summary, check))
        self.assertEqual(summary["processing"], "stopped at the attempt deadline after 1 of 2 candidates")
        self.assertEqual((summary["result"], run.archive.cost), ("improved", 150))
        run, entry = self.overshooting_run()
        summary = self.blank_summary()
        run._finisher_apply(entry, search, summary, mock.Mock(side_effect=s._FinisherTimeout()))
        self.assertIn("no usable result among the 0 candidates", summary["result"])
        self.assertIs(run.archive, entry)
        self.assertIn("result processing stopped", s.finisher_summary_line(dict(
            summary, generation=1, source="x", uid=None, active_length=3, corners=0, edges=0, initial_cost=156,
            depth=4, table_depth=2, forward_target=2, forward_depth=2, states=0, search_seconds=0.0,
            preparation_seconds=0.0, table_source="memory")))

    def test_an_overall_stop_keeps_only_a_verified_solve(self):
        run = self.make_run(after(SOLVED, "R1 U1 C1"))
        run.budget.start()
        entry = self.set_archive(run, "C3 L1")
        search = self.fabricated(solving=["L3 U3 R3"], alternatives=["L3"])
        stop = mock.Mock(side_effect=[None, s.SearchInterrupted("SIGINT received")])
        candidate, suffix = run._finisher_apply(entry, search, self.blank_summary(), stop)
        self.assertTrue(candidate.solved)
        run, entry = self.overshooting_run()
        stop = mock.Mock(side_effect=[None, s.SearchInterrupted("SIGINT received")])
        with self.assertRaises(s.SearchInterrupted):
            run._finisher_apply(entry, self.fabricated(alternatives=["R3", "R3 D3 C3"]), self.blank_summary(), stop)
        self.assertIs(run.archive, entry)  # an unsolved improvement is dropped after an overall stop

    def test_attempt_time_excludes_preparation_but_covers_processing(self):
        run = self.make_run(after(SOLVED, self.FAR), mitm_attempt_seconds=0.2)
        run.budget.start()
        table = self.table(2)
        seen = {}

        def slow_preparation(*args, **kwargs):
            time.sleep(0.3)
            return table

        def slow_search(*args, deadline=None, **kwargs):
            seen["left"] = deadline - time.monotonic()
            time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)
            return self.fabricated(alternatives=["R3"])

        with mock.patch.object(s, "prepare_solved_side_table", side_effect=slow_preparation), \
                mock.patch.object(s, "mitm_search", side_effect=slow_search):
            summary = run.finisher_attempt(0)
        self.assertGreater(summary["preparation_seconds"], 0.3)
        self.assertGreater(seen["left"], 0.1)  # 0.3 s of preparation did not use up the 0.2 s attempt
        self.assertEqual(summary["processing"], "stopped at the attempt deadline after 0 of 1 candidates")
        self.assertEqual(run.archive.source, "empty baseline")

    def test_shortening_table_is_built_during_preparation(self):
        saved = s._SHORTEN_INDEX
        try:
            s._SHORTEN_INDEX = None
            with self.assertRaises(s._FinisherTimeout):
                s._shorten_index(mock.Mock(side_effect=[None] * 10 + [s._FinisherTimeout()]))
            self.assertIsNone(s._SHORTEN_INDEX)  # nothing is kept from a stopped build
            run = self.make_run(after(SOLVED, self.FAR))
            run.budget.start()
            with mock.patch.object(s, "mitm_search", return_value=self.fabricated()):
                run.finisher_attempt(0)
            self.assertIsNotNone(s._SHORTEN_INDEX)
        finally:
            if s._SHORTEN_INDEX is None:
                s._SHORTEN_INDEX = saved

    def test_solving_path_still_over_capacity_is_reported(self):
        run = self.make_run(after(SOLVED, "R1 U1 C1"), solution_capacity=2, block_capacity=2)
        run.budget.start()
        self.set_archive(run, "C3 L1")
        with mock.patch.object(s, "mitm_search", return_value=self.fabricated(solving=["L3 U3 R3"])):
            summary = run.finisher_attempt(0)
        self.assertEqual(summary["result"], "solving path exceeds the solution capacity after reductions")
        self.assertFalse(run.archive.solved)

    def test_improvement_at_a_generation_boundary_replaces_the_worst_offspring(self):
        run = self.make_run(after(SOLVED, self.FAR), mitm_depth=3)
        run.budget.start()
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        run.run_generation(1)
        before = list(run.population)
        fifth = len(before) // 5
        previous = self.set_archive(run, ())  # the empty baseline, which a 2-move path improves
        summary = run.finisher_attempt(1)
        self.assertEqual(summary["result"], "improved")
        self.assertLess(run.archive.cost, previous.cost)
        self.assertEqual(run.archive.source, "finisher depth 3")
        after_population = run.population
        self.assertEqual(len(after_population), len(before))
        self.assertEqual([ind.uid for ind in after_population[:fifth]], [ind.uid for ind in before[:fifth]])
        self.assertEqual([ind.uid for ind in after_population[-fifth:]], [ind.uid for ind in before[-fifth:]])
        inserted = [ind for ind in after_population if ind.origin == "finisher"]
        self.assertEqual(len(inserted), 1)
        record = inserted[0]
        self.assertEqual(record.slot, "offspring")
        self.assertEqual(record.active, run.archive.active)
        self.assertEqual(list(record), list(record.active) + [s.NOOP] * (run.solution_capacity - len(record.active)))
        candidates = [ind for ind in before if ind.slot in ("offspring", "baseline-start")]
        worst = max(candidates, key=lambda ind: (ind.rank, ind.uid))
        self.assertNotIn(worst.uid, {ind.uid for ind in after_population})
        self.assertIn(f"replaced offspring S#{worst.uid}", summary["population"])
        run.run_generation(2)  # the next generation selects from the modified population
        self.assertEqual(len(run.population), len(before))

    def test_results_after_setup_wait_for_generation_one(self):
        run = self.make_run(after(SOLVED, self.FAR), mitm_depth=3, generations=1)
        result = run.run()
        attempt = result.finisher_report["attempts"][0]
        self.assertEqual((attempt["generation"], attempt["result"]), (0, "improved"))
        self.assertEqual(attempt["population"], "queued for generation 1's offspring")
        self.assertEqual([ind.origin for ind in run.population].count("finisher"), 1)
        self.assertTrue(any(message.startswith("[finisher] S#") for message in self.messages))

    def test_initial_population_stays_all_noop(self):
        run = self.make_run(after(SOLVED, self.FAR), mitm_depth=3)
        run.budget.start()
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        run.finisher_attempt(0)
        self.assertEqual(len(run._finisher_queue), 1)
        self.assertTrue(all(ind.active == () for ind in run.population))

    def test_suffix_becomes_a_pending_library_candidate(self):
        library = os.path.join(self.directory, "blocks.json")
        run = self.make_run(after(SOLVED, self.FAR), mitm_depth=3, block_library=library)
        run.budget.start()
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        summary = run.finisher_attempt(0)
        self.assertEqual(summary["result"], "improved")
        self.assertEqual([len(payload) for payload in run._library_pending], [summary["suffix_length"]])

    def test_exhausted_and_timed_out_attempts_are_not_repeated_on_the_same_endpoint(self):
        for termination, reported in (("exhaustive", "exhausted the depth-4 search without a solution"),
                                      ("timed out", "timed out")):
            run = self.make_run(after(SOLVED, self.FAR))
            run.budget.start()
            with mock.patch.object(s, "mitm_search", return_value=self.fabricated(termination=termination)) as search:
                summary = run.finisher_attempt(0)
                self.assertEqual((summary["termination"], summary["result"]),
                                 (reported, "no feasible improving fallback"))
                self.assertIsNone(run.finisher_attempt(50))
                self.assertEqual(search.call_count, 1)
            cached = next(iter(run._finisher_cache.values()))
            self.assertEqual((cached["termination"], cached["retryable"]), (reported, False))
            if termination == "timed out":
                line = self.finisher_lines()[-1]
                self.assertNotIn("exhausted", line)
                self.assertNotIn("more than", line)

    def test_incomplete_preparation_is_retried(self):
        run = self.make_run(after(SOLVED, self.FAR))
        run.budget.start()
        table = self.table(2)
        with mock.patch.object(s, "prepare_solved_side_table", side_effect=[s._FinisherTimeout(), table]), \
                mock.patch.object(s, "mitm_search", return_value=self.fabricated()) as search:
            first = run.finisher_attempt(0)
            self.assertEqual(first["termination"], "preparation incomplete")
            self.assertIsNone(run.finisher_attempt(10))  # still respects the interval
            second = run.finisher_attempt(50)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(second["generation"], 50)

    def test_sigint_stops_the_attempt_and_the_run(self):
        run = self.make_run(after(SOLVED, self.FAR))
        run.budget.start()
        with mock.patch.object(s, "mitm_search", side_effect=s.SearchInterrupted("SIGINT received")):
            with self.assertRaises(s.SearchInterrupted):
                run.finisher_attempt(0)
        attempt = run.finisher_attempts[-1]
        self.assertEqual(attempt["termination"], "stopped by the run's budget or SIGINT")
        self.assertNotEqual(attempt["termination"], "exhaustive")
        run = self.make_run(after(SOLVED, self.FAR))
        with mock.patch.object(s, "mitm_search", side_effect=s.SearchInterrupted("SIGINT received")):
            result = run.run()
        self.assertEqual(result.status, s.STATUS_INTERRUPTED)
        run = self.make_run(after(SOLVED, self.FAR))
        run.budget.start()
        run.budget.request_interrupt()
        with self.assertRaises(s.SearchInterrupted):
            run.finisher_attempt(0)
        self.assertEqual(run.finisher_attempts, [])  # no finishing work after an overall stop

    def test_trigger_interval_budget_and_solved_endpoints(self):
        key = scramble_of(THREE_CYCLE_SOLUTION)  # 0 corners + 3 edges, more than 4 moves away
        run = self.make_run(key, mitm_trigger_cubies=2)
        run.budget.start()
        self.assertIsNone(run.finisher_attempt(0))  # 3 > 2 misplaced cubies
        run = self.make_run(key, fitness="corners-first", mitm_trigger_cubies=2, mitm_corners_first_edge_trigger=3)
        run.budget.start()
        self.assertIsNotNone(run.finisher_attempt(0))  # corners home and 3 <= 3 edges
        run = self.make_run(key, fitness="corners-first", mitm_trigger_cubies=2, mitm_corners_first_edge_trigger=2)
        run.budget.start()
        self.assertIsNone(run.finisher_attempt(0))
        run = self.make_run(key, mitm_interval=50)
        run.budget.start()
        self.assertIsNotNone(run.finisher_attempt(0))
        self.set_archive(run, "R1")  # a new endpoint still waits for the interval
        self.assertIsNone(run.finisher_attempt(49))
        self.assertIsNotNone(run.finisher_attempt(50))
        self.set_archive(run, "U1")
        run.finisher_seconds = run.config.mitm_total_seconds
        self.assertIsNone(run.finisher_attempt(100))  # total finishing budget spent
        run = self.make_run(after(SOLVED, "R1"))
        run.archive = s.ArchiveEntry((), SOLVED, 0, True, None, 0, "fixture")
        self.assertIsNone(run.finisher_attempt(0))  # solved endpoints are skipped

    def test_disabled_mode_does_no_table_work_and_leaves_the_search_unchanged(self):
        key = after(SOLVED, "R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        common = dict(fitness="manhattan", seed=5, generations=6, solution_population=20, block_population=20,
                      solution_capacity=16, block_capacity=8, contexts=6, min_block_gain=1.0, log_every=10,
                      block_library=None)
        for overrides in (dict(mitm=False), dict(mitm=True, mitm_trigger_cubies=0)):
            config = s.SearchConfig(**common, mitm_cache=self.cache, **overrides)
            run = s.CoevolutionRun(config, key, s.validate_config(config), 5, log=lambda message: None)
            result = run.run()
            self.assertEqual(search_digest(run, result), PRE_FEATURE_DIGESTS["manhattan"])
            self.assertEqual(result.finisher_report["attempts"], [])
            self.assertEqual(s._SOLVED_SIDE_TABLES, {})
            self.assertFalse(os.path.exists(self.cache))

    def test_an_unproductive_attempt_does_not_touch_the_ga_random_stream(self):
        key = after(SOLVED, "R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        config = s.SearchConfig(fitness="manhattan", seed=5, generations=6, solution_population=20,
                                block_population=20, solution_capacity=16, block_capacity=8, contexts=6,
                                min_block_gain=1.0, log_every=10, block_library=None, mitm=True, mitm_depth=4,
                                mitm_trigger_cubies=20, mitm_interval=1, mitm_cache=None)
        run = s.CoevolutionRun(config, key, s.validate_config(config), 5, log=lambda message: None)
        with mock.patch.object(s, "mitm_search", return_value=self.fabricated()):
            result = run.run()
        self.assertGreater(len(result.finisher_report["attempts"]), 0)
        self.assertEqual(search_digest(run, result), PRE_FEATURE_DIGESTS["manhattan"])


class ConfigAndCliTests(FinisherTestCase):
    def test_validation(self):
        bad = [dict(mitm="yes"), dict(mitm_depth=0), dict(mitm_depth=11), dict(mitm_depth=2.0),
               dict(mitm_trigger_cubies=-1), dict(mitm_trigger_cubies=21), dict(mitm_corners_first_edge_trigger=13),
               dict(mitm_interval=0), dict(mitm_attempt_seconds=0.0), dict(mitm_attempt_seconds=float("nan")),
               dict(mitm_total_seconds=-1.0), dict(mitm_total_seconds=float("inf")), dict(mitm_cache="")]
        for overrides in bad:
            with self.assertRaises(s.ConfigError, msg=overrides):
                s.validate_config(s.SearchConfig(**overrides))
        with self.assertRaisesRegex(s.ConfigError, "deeper searches are not supported"):
            s.validate_config(s.SearchConfig(mitm_depth=12))
        s.validate_config(s.SearchConfig(mitm=True, mitm_depth=1, mitm_trigger_cubies=0, mitm_cache="x"))

    def test_cli_defaults_and_cache_options(self):
        parser = s.build_arg_parser()
        config = s.config_from_args(parser.parse_args([]))
        self.assertEqual((config.mitm, config.mitm_depth, config.mitm_trigger_cubies,
                          config.mitm_corners_first_edge_trigger, config.mitm_interval,
                          config.mitm_attempt_seconds, config.mitm_total_seconds),
                         (True, 10, 4, 6, 50, 10.0, 30.0))  # the CLI runs the finisher by default
        self.assertEqual(config.mitm_cache, os.path.join(os.environ["XDG_CACHE_HOME"], "rubik3x3solver"))
        self.assertFalse(s.config_from_args(parser.parse_args(["--no-mitm"])).mitm)
        self.assertFalse(s.SearchConfig().mitm)  # programmatic callers still opt in
        config = s.config_from_args(parser.parse_args(["--mitm", "--mitm-depth", "6", "--no-mitm-cache"]))
        self.assertEqual((config.mitm, config.mitm_depth, config.mitm_cache), (True, 6, None))
        config = s.config_from_args(parser.parse_args(["--mitm", "--no-mitm", "--mitm-cache-dir", "rel"]))
        self.assertEqual((config.mitm, config.mitm_cache), (False, os.path.abspath("rel")))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--mitm-cache-dir", "x", "--no-mitm-cache"])
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "relative"}):
            self.assertEqual(s.default_mitm_cache_dir(),
                             os.path.join(os.path.expanduser("~"), ".cache", "rubik3x3solver"))

    def test_cli_solve_reports_startup_attempt_and_totals(self):
        with mock.patch.object(s, "default_input_cube", lambda: s.cube_from_key(after(SOLVED, "R1 U1 C1"))):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = s.main(["--mitm", "--mitm-depth", "4", "--mitm-trigger-cubies", "20",
                               "--mitm-cache-dir", self.cache, "--no-block-library",
                               "--solution-population", "10", "--block-population", "10", "--contexts", "4",
                               "--seed", "1"])
        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Finisher: meet-in-the-middle depth 4 (table 2 + forward 2); attempt at <=20", text)
        self.assertIn("[finisher gen 0]", text)
        self.assertIn("Finisher: 1 attempt(s)", text)
        self.assertIn("table built", text)
        self.assertTrue(os.path.exists(s.mitm_cache_file(self.cache, 2)))

    def test_cli_default_and_disabled_lines(self):
        common = ["--no-block-library", "--generations", "0", "--solution-population", "10",
                  "--block-population", "10", "--contexts", "4", "--seed", "1"]
        with mock.patch.object(s, "default_input_cube", lambda: s.cube_from_key(after(SOLVED, "R1 U1 C1"))):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                s.main(common)
            self.assertIn("Finisher: meet-in-the-middle depth 10", output.getvalue())
            self.assertIn("(--no-mitm turns it off)", output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                s.main(common + ["--no-mitm"])
        self.assertIn("Finisher: automatic attempts disabled (--no-mitm)", output.getvalue())
        self.assertNotIn("[finisher", output.getvalue())

    def test_cli_runs_the_finisher_by_default(self):
        with mock.patch.object(s, "default_input_cube", lambda: s.cube_from_key(after(SOLVED, "R1 U1 C1"))):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = s.main(["--no-block-library", "--mitm-depth", "4", "--mitm-trigger-cubies", "20",
                               "--solution-population", "10", "--block-population", "10", "--contexts", "4",
                               "--seed", "1"])
        self.assertEqual(code, 0)
        self.assertIn("[finisher gen 0]", output.getvalue())
        self.assertIn("via finisher depth 4", output.getvalue())


if __name__ == "__main__":
    unittest.main()
