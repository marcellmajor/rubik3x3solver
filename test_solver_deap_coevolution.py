"""Deterministic tests for solver_deap_coevolution.py (standard-library unittest).

Run with:  uv run python -m unittest test_solver_deap_coevolution -v
"""

import ast
import contextlib
import dataclasses
import io
import os
import random
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import solver_deap_coevolution as s

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVED_KEY = "G" * 9 + "R" * 9 + "W" * 9 + "O" * 9 + "Y" * 9 + "B" * 9
LABELED_KEY = (string.ascii_letters + "01")[:54]


def key_after(start_key, moves):
    sequence = s.parse_moves(moves) if isinstance(moves, str) else moves
    return s.state_key(s.replay(s.cube_from_key(start_key), sequence))


def scrambled(moves):
    return s.cube_from_key(key_after(SOLVED_KEY, moves))


def make_run(input_cube=None, *, seed=1, budget=None, **overrides):
    config = s.SearchConfig(seed=seed, log_every=10 ** 6, **overrides)
    objective = s.validate_config(config)
    cube = s.default_input_cube() if input_cube is None else input_cube
    key = s.validate_input_cube(cube)
    return s.CoevolutionRun(config, key, objective, seed, budget=budget, log=lambda message: None)


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        value = function(*args, **kwargs)
    return value, out.getvalue()


class ScriptedRng:
    """RNG stub returning scripted randrange values."""

    def __init__(self, values):
        self.values = list(values)

    def randrange(self, stop):
        value = self.values.pop(0)
        assert 0 <= value < stop
        return value


class SimplifierTests(unittest.TestCase):
    def test_spec_examples(self):
        examples = {
            "R1 R1": "R2",
            "R1 R2": "R3",
            "R1 R3": "",
            "R2 R2": "",
            "R1 R1 R1 R1": "",
            "R1 U1 U3 R2": "R3",
            "R1 NOOP R2": "R3",
            "R1 L1 R3": "L1",
            "L1 R2": "R2 L1",
        }
        for source, expected in examples.items():
            with self.subTest(source=source):
                self.assertEqual(s.simplify(s.parse_moves(source)), s.parse_moves(expected))

    def test_cascading_cancellations_and_axis_runs(self):
        self.assertEqual(s.simplify(s.parse_moves("U1 R1 L1 R3 L3 U3")), ())
        self.assertEqual(s.simplify(s.parse_moves("C1 U1 D1 U3 D3 C3 B2")), s.parse_moves("B2"))
        self.assertEqual(s.simplify(s.parse_moves("B1 C1 B1")), s.parse_moves("C1 B2"))
        self.assertEqual(s.simplify(s.parse_moves("D1 U1 R1 L1")), s.parse_moves("U1 D1 R1 L1"))

    def test_random_sequences_preserve_the_labeled_permutation(self):
        rng = random.Random(11)
        for _ in range(400):
            sequence = tuple(rng.randrange(19) for _ in range(rng.randint(0, 40)))
            reduced = s.simplify(sequence)
            self.assertEqual(key_after(LABELED_KEY, sequence), key_after(LABELED_KEY, reduced))
            self.assertEqual(s.simplify(reduced), reduced)
            self.assertLessEqual(len(reduced), len(s.active_moves(sequence)))
            self.assertNotIn(s.NOOP, reduced)
            for left, right in zip(reduced, reduced[1:]):
                left_face, right_face = (left - 1) // 3, (right - 1) // 3
                self.assertNotEqual(left_face, right_face)
                if left_face // 2 == right_face // 2:
                    self.assertEqual((left_face % 2, right_face % 2), (0, 1))
            for first, second, third in zip(reduced, reduced[1:], reduced[2:]):
                self.assertFalse((first - 1) // 6 == (second - 1) // 6 == (third - 1) // 6)

    def test_engine_semantics_used_by_the_simplifier(self):
        for face in s.FACE_LETTERS:
            self.assertEqual(key_after(LABELED_KEY, f"{face}1 {face}1 {face}1 {face}1"), LABELED_KEY)
            self.assertEqual(key_after(LABELED_KEY, f"{face}2"), key_after(LABELED_KEY, f"{face}1 {face}1"))
        for first, second in ("RL", "UD", "CB"):
            self.assertEqual(key_after(LABELED_KEY, f"{first}1 {second}1"),
                             key_after(LABELED_KEY, f"{second}1 {first}1"))
        for first, second in ("RU", "RC", "UC", "LD", "DB"):
            self.assertNotEqual(key_after(LABELED_KEY, f"{first}1 {second}1"),
                                key_after(LABELED_KEY, f"{second}1 {first}1"))

    def test_cube_inverse_is_not_reversal(self):
        sequence = s.parse_moves("R1 U2 C3")
        self.assertEqual(s.cube_inverse(sequence), s.parse_moves("C1 U2 R3"))
        self.assertEqual(key_after(LABELED_KEY, sequence + s.cube_inverse(sequence)), LABELED_KEY)

    def test_invalid_alleles_rejected(self):
        for bad in ((19,), (-1,), (1.0,), (True,), ("R1",)):
            with self.assertRaises(ValueError):
                s.simplify(bad)

    def test_canonical_enumeration_is_lexicographic_and_reduced(self):
        sequences = list(s.iter_canonical_sequences(2))
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertTrue(all(s.simplify(sequence) == sequence for sequence in sequences))
        self.assertEqual(len(list(s.iter_canonical_sequences(1))), 18)


class NormalizationTests(unittest.TestCase):
    def test_empty_baseline_has_length_zero(self):
        run = make_run()
        self.assertEqual(run.baseline.active, ())
        self.assertEqual(run.baseline.rank, (1, s.manhattan_cost(run.input_key), 0))
        record = run.baseline_record("reset")
        self.assertEqual(list(record), [s.NOOP] * run.solution_capacity)

    def test_solve_at_first_move_then_scramble_stores_only_the_solving_move(self):
        run = make_run(scrambled("R3"))
        candidate = run.normalize_solution(s.parse_moves("R1 U1 C2 D3"))
        self.assertEqual(candidate.active, s.parse_moves("R1"))
        self.assertTrue(candidate.solved)
        self.assertTrue(s.is_solved_key(key_after(run.input_key, candidate.active)))

    def test_prefix_useful_before_same_face_compression(self):
        run = make_run(scrambled("R3"))
        candidate = run.normalize_solution(s.parse_moves("R1 R1"))
        self.assertEqual(candidate.active, s.parse_moves("R1"))
        self.assertTrue(candidate.solved)

    def test_normalization_is_idempotent_and_pads_with_noop(self):
        run = make_run(scrambled("R1 U2 C3 D1 B2 L3"))
        rng = random.Random(5)
        for _ in range(100):
            proposal = tuple(rng.randrange(19) for _ in range(rng.randint(0, 30)))
            candidate = run.normalize_solution(proposal)
            self.assertEqual(run.normalize_solution(candidate.active), dataclasses.replace(candidate,
                                                                                        capacity_rejected=False))
            record = run.make_solution(candidate, origin="test")
            active_length = len(candidate.active)
            self.assertEqual(tuple(record[:active_length]), candidate.active)
            self.assertTrue(all(gene == s.NOOP for gene in record[active_length:]))
            self.assertEqual(len(record), run.solution_capacity)
            self.assertEqual(record.fitness.values, tuple(float(value) for value in candidate.rank))

    def test_capacity_boundary_and_cancellation_to_fit(self):
        run = make_run(scrambled("R1 U1 C1"), solution_capacity=2, block_capacity=2)
        # U1 U2 merges, so the improving prefix fits capacity 2; the solving prefix needs 3 moves.
        candidate = run.normalize_solution(s.parse_moves("C3 U1 U2 R3"))
        self.assertEqual(candidate.active, s.parse_moves("C3 U3"))
        self.assertFalse(candidate.solved)
        self.assertTrue(candidate.capacity_rejected)
        self.assertEqual(run.counters.solution_capacity_rejections, 1)

    def test_evaluation_order_does_not_change_results(self):
        proposal = s.parse_moves("C3 U3 D1 R2")
        first_run = make_run(scrambled("R1 U1 C1"))
        expected = first_run.normalize_solution(proposal)
        second_run = make_run(scrambled("R1 U1 C1"))
        rng = random.Random(3)
        for _ in range(50):
            candidate = second_run.normalize_solution(tuple(rng.randrange(1, 19) for _ in range(6)))
            try:
                second_run.offer_to_archive(candidate, uid=None, source="test")
            except s.SolutionFound:
                pass
        self.assertEqual(second_run.normalize_solution(proposal), expected)

    def test_already_solved_input_returns_immediately(self):
        result = s.run_search(s.SearchConfig(seed=1, generations=0), s.cube_from_key(SOLVED_KEY),
                              log=lambda message: None)
        self.assertEqual(result.status, s.STATUS_SOLVED)
        self.assertEqual(result.archive.active, ())
        self.assertFalse(result.block_population_created)
        self.assertEqual(s.finalize_result(result).status, s.STATUS_SOLVED)


class ObjectiveTests(unittest.TestCase):
    @staticmethod
    def reference_manhattan(key):
        offsets = s.COLOR_OFFSETS
        total = 0
        for start in s.FACE_STARTS:
            face = key[start:start + 9]
            center = face[4]
            for index, sticker in enumerate(face):
                if sticker == center:
                    continue
                current = (offsets[center][0] + index // 3, offsets[center][1] + index % 3)
                goal = (offsets[sticker][0] + index // 3, offsets[sticker][1] + index % 3)
                total += abs(current[0] - goal[0]) + abs(current[1] - goal[1])
        return total

    def test_manhattan_matches_the_historical_formula(self):
        rng = random.Random(2)
        for _ in range(200):
            key = key_after(SOLVED_KEY, s.generate_reduced_block(rng.randint(1, 16), rng))
            self.assertEqual(s.manhattan_cost(key), self.reference_manhattan(key))
        self.assertEqual(s.manhattan_cost(s.validate_input_cube(s.default_input_cube())), 24)

    def test_every_objective_identifies_the_solved_state(self):
        objectives = [s.make_objective(name) for name in ("manhattan", "misplaced", "manhattan-min")]
        objectives.append(s.make_objective("manhattan-misplaced", 0.5, 2.0))
        objectives.append(s.make_objective("manhattan-misplaced", 0.0, 1.0))
        rng = random.Random(4)
        for objective in objectives:
            self.assertEqual(objective.cost(SOLVED_KEY), 0)
            for _ in range(30):
                key = key_after(SOLVED_KEY, s.generate_reduced_block(rng.randint(1, 8), rng))
                self.assertEqual(objective.cost(key) == 0, s.is_solved_key(key))

    def test_names_resolve_as_documented(self):
        self.assertEqual(s.DEFAULT_FITNESS, "manhattan")
        args = s.build_arg_parser().parse_args([])
        self.assertEqual(s.validate_config(s.config_from_args(args)).state_function, "manhattan")
        self.assertEqual(s.make_objective("manhattan-min").state_function, "manhattan")
        self.assertEqual(s.make_objective("misplaced").state_function, "misplaced")
        mixed = s.make_objective("manhattan-misplaced")
        self.assertEqual((mixed.manhattan_weight, mixed.misplaced_weight), (1.0, 1.0))
        self.assertFalse(mixed.integer_valued)

    def test_invalid_weights_rejected(self):
        for weights in ((0.0, 0.0), (-1.0, 1.0), (float("nan"), 1.0), (float("inf"), 1.0), (1e308, 1e308)):
            with self.subTest(weights=weights), self.assertRaises(s.ConfigError):
                s.make_objective("manhattan-misplaced", *weights)
        with self.assertRaises(s.ConfigError):
            s.make_objective("manhattan", 1.0, None)
        tiny = s.make_objective("manhattan-misplaced", 1e-300, 1e-300)
        self.assertGreater(tiny.cost(key_after(SOLVED_KEY, "R1")), 0)

    def test_gain_threshold_regressions(self):
        for integer_valued in (True, False):
            self.assertFalse(s.gain_qualifies(0.0, 1e-12, integer_valued))
            self.assertFalse(s.gain_qualifies(-1e-10, 1e-12, integer_valued))
            self.assertTrue(s.gain_qualifies(1.0, 1.0, integer_valued))
            self.assertFalse(s.gain_qualifies(0.5, 1.0, integer_valued))
        self.assertTrue(s.gain_qualifies(1e-300, 1e-300, False))
        self.assertTrue(s.gain_qualifies(1.0 - 1e-15, 1.0, False))
        self.assertFalse(s.gain_qualifies(1.0 - 1e-15, 1.0, True))
        with self.assertRaises(s.NumericalError):
            s.gain_qualifies(float("nan"), 1.0, False)

    def test_explicit_weight_flags_need_the_mixed_objective(self):
        code, output = quiet(s.main, ["--fitness", "manhattan", "--misplaced-weight", "1"])
        self.assertEqual(code, 2)
        self.assertIn("Status: INVALID_INPUT", output)
        code, _ = quiet(s.main, ["--fitness", "manhattan-misplaced", "--manhattan-weight", "nan"])
        self.assertEqual(code, 2)

    def test_repeated_initialisation_is_safe(self):
        self.assertEqual(s.ensure_deap_types(), s.ensure_deap_types())
        first = s.run_search(s.SearchConfig(seed=9, generations=2, solution_population=10,
                                            block_population=10, solution_capacity=8, block_capacity=4,
                                            contexts=3), log=lambda message: None)
        second = s.run_search(s.SearchConfig(seed=9, generations=2, solution_population=10,
                                             block_population=10, solution_capacity=8, block_capacity=4,
                                             contexts=3), log=lambda message: None)
        self.assertEqual(first.archive, second.archive)
        self.assertEqual(first.counters.quarter_turns, second.counters.quarter_turns)

    def test_config_validation(self):
        bad = [
            dict(solution_population=7), dict(solution_population=0), dict(block_population=15),
            dict(block_capacity=40), dict(solution_capacity=0), dict(contexts=0), dict(log_every=0),
            dict(generations=-1), dict(min_block_gain=0.0), dict(min_block_gain=float("inf")),
            dict(time_limit=0.0), dict(time_limit=float("nan")),
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(s.ConfigError):
                s.validate_config(s.SearchConfig(**overrides))
        with self.assertRaises(s.ConfigError):
            s.validate_operator_probabilities((("a", 0.5), ("b", 0.4)))

    def test_input_validation(self):
        bad_cubes = [
            s.Cube("GGGGGGGG", ["RRRRRRRRR", "WWWWWWWWW", "OOOOOOOOO", "YYYYYYYYY"], "BBBBBBBBB"),
            s.Cube("GGGGGGGGX", ["RRRRRRRRR", "WWWWWWWWW", "OOOOOOOOO", "YYYYYYYYY"], "BBBBBBBBB"),
            s.Cube("GGGGGGGGR", ["RRRRRRRRR", "WWWWWWWWW", "OOOOOOOOO", "YYYYYYYYY"], "BBBBBBBBB"),
            s.Cube("GGGGRGGGG", ["RRRRRRRRG", "WWWWWWWWW", "OOOOOOOOO", "YYYYYYYYY"], "BBBBBBBBB"),
            s.Cube("GGGGGGGGG", ["RRRRRRRRR", "WWWWWWWWW", "OOOOOOOOO"], "BBBBBBBBB"),
        ]
        for cube in bad_cubes:
            with self.assertRaises(s.InvalidInputError):
                s.validate_input_cube(cube)


class PopulationPolicyTests(unittest.TestCase):
    def test_initial_population_is_all_noop(self):
        run = make_run(solution_population=10)
        population = run.initial_solution_population()
        self.assertEqual(len({individual.uid for individual in population}), 10)
        for individual in population:
            self.assertEqual(list(individual), [s.NOOP] * run.solution_capacity)
            self.assertEqual(individual.active, ())

    def test_baseline_start_quota(self):
        expected = {5: 1, 10: 1, 15: 2, 20: 2, 25: 3, 50: 6, 200: 24}
        for population, quota in expected.items():
            self.assertEqual(s.baseline_start_quota(3 * population // 5), quota)

    def test_twenty_sixty_twenty_replacement_with_160_improved(self):
        run = make_run(scrambled("R1 U2 C3 D1 B2 L3 R2 C1 U3 B1 D2 L1 R3 C2 U1 B3 D3 L2 R1 C3"),
                       solution_population=200, seed=12)
        rng = random.Random(8)
        improved, seen = [], set()
        while len(improved) < 160:
            candidate = run.normalize_solution(tuple(rng.randrange(1, 19) for _ in range(rng.randint(1, 4))))
            if candidate.rank < run.baseline_rank and candidate.active not in seen:
                seen.add(candidate.active)
                improved.append(run.make_solution(candidate, origin="fixture"))
        empties = [run.baseline_record("fixture-empty") for _ in range(40)]
        population = improved + empties
        run.population = population
        run.blocks = run.initial_block_population()
        ranked = sorted(population, key=lambda individual: (individual.rank, individual.uid))
        max_uid = max(individual.uid for individual in population)
        run.run_generation(1, build_next_blocks=False)
        new_population = run.population
        self.assertEqual(len(new_population), 200)

        elites = new_population[:40]
        for copy, original in zip(elites, ranked[:40]):
            self.assertIsNot(copy, original)
            self.assertEqual((copy.uid, copy.active, copy.endpoint, copy.rank, list(copy)),
                             (original.uid, original.active, original.endpoint, original.rank, list(original)))
            self.assertEqual(copy.fitness.values, original.fitness.values)

        offspring = new_population[40:160]
        self.assertEqual(Counter(child.slot for child in offspring), {"offspring": 96, "baseline-start": 24})
        self.assertTrue(all(child.uid > max_uid for child in offspring))
        resets = new_population[160:]
        self.assertTrue(all(reset.slot == "reset" and reset.active == () for reset in resets))
        self.assertTrue(all(list(reset) == [s.NOOP] * run.solution_capacity for reset in resets))
        self.assertTrue(all(reset.uid > max(child.uid for child in offspring) for reset in resets))
        discarded = {individual.uid for individual in ranked[160:]}
        self.assertFalse(discarded & {individual.uid for individual in new_population})
        self.assertEqual(run.counters.baseline_start_attempts, 24)

    def test_equal_fitness_tournament_ties_are_random(self):
        candidates = [SimpleNamespace(uid=uid, rank=(1, 5, 2)) for uid in (1, 2, 3)]
        key = lambda candidate: candidate.rank
        first = s.tournament_select(candidates, key, ScriptedRng([0, 1, 2, 0]))
        last = s.tournament_select(candidates, key, ScriptedRng([0, 1, 2, 2]))
        self.assertEqual((first.uid, last.uid), (1, 3))
        better = SimpleNamespace(uid=9, rank=(1, 4, 9))
        self.assertIs(s.tournament_select(candidates + [better], key, ScriptedRng([0, 3, 1])), better)

    def test_best_ever_never_worsens(self):
        result = s.run_search(s.SearchConfig(seed=3, generations=12, solution_population=20,
                                             block_population=20, contexts=6),
                              scrambled("R2 B2 D1 R2 U1 C2 D2 C1"), log=lambda message: None)
        ranks = [(0 if record["solved"] else 1, record["best_cost"], record["best_length"])
                 for record in result.history]
        self.assertEqual(ranks, sorted(ranks, reverse=True))
        self.assertLess(ranks[-1], ranks[0])

    def test_mutation_positions_are_all_reachable(self):
        rng = random.Random(1)
        sequence = s.parse_moves("R1 U1 C1 L1")
        insert_positions, deleted, spans = set(), set(), set()
        for _ in range(3000):
            insert_positions.add(s.op_insert_block(sequence, (99,), rng).index(99))
            remaining = s.op_delete_gene(sequence, rng)
            deleted.add(next(index for index, gene in enumerate(sequence) if gene not in remaining))
            spans.add(s.choose_span(len(sequence), rng))
        self.assertEqual(insert_positions, set(range(len(sequence) + 1)))
        self.assertEqual(deleted, set(range(len(sequence))))
        self.assertIn((3, 1), spans)
        self.assertIn((0, 3), spans)
        self.assertEqual(s.op_insert_block((), (7,), rng), (7,))

    def test_empty_and_one_move_parents_generate_active_offspring(self):
        run = make_run(scrambled("R1 U1 C1"), solution_population=10)
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        batch = run.build_context_batch(run.population)
        run.evaluate_blocks(run.blocks, batch)
        run._index_blocks(run.blocks, batch)
        one_move = run.make_solution(run.normalize_solution(s.parse_moves("C3")), origin="fixture")
        for pool in ([run.population[0]], [one_move]):
            children = [run._ordinary_offspring(pool) for _ in range(200)]
            self.assertTrue(any(child.active and child.active != pool[0].active for child in children))
            self.assertTrue(all(len(child.active) <= run.solution_capacity for child in children))
        operators = {child.operator for child in children}
        self.assertTrue(operators & {"delete-gene", "substitute-gene", "cube-inverse", "span-replace"})

    def test_no_mutable_aliasing(self):
        run = make_run(scrambled("R1 U1 C1"), solution_population=10)
        record = run.make_solution(run.normalize_solution(s.parse_moves("C3")), origin="fixture")
        copy = run.copy_solution(record)
        copy[0] = s.NOOP
        self.assertNotEqual(record[0], s.NOOP)
        batch = run.build_context_batch([record] + [run.baseline_record("x") for _ in range(9)])
        for context in batch.contexts:
            self.assertIsNot(context.cube, run._original)
        before = s.state_key(batch.contexts[0].cube)
        s.apply_move(batch.contexts[0].cube.clone(), 1)
        self.assertEqual(s.state_key(batch.contexts[0].cube), before)
        self.assertEqual(s.state_key(run._original), run.input_key)


def population_with(run, actives, size):
    records = [run.make_solution(run.normalize_solution(s.parse_moves(active)), origin="fixture")
               for active in actives]
    records += [run.baseline_record("fixture-empty") for _ in range(size - len(records))]
    return sorted(records, key=lambda individual: (individual.rank, individual.uid))


class ContextBatchTests(unittest.TestCase):
    @staticmethod
    def fake_population(endpoints, actives=None):
        population = []
        for index, endpoint in enumerate(endpoints):
            active = actives[index] if actives else (1,) * (index % 3 + 1)
            population.append(SimpleNamespace(uid=index + 1, endpoint=endpoint, active=active,
                                              rank=(1, index, len(active))))
        return population

    def test_full_batch_composition(self):
        population = self.fake_population([f"state{index:03d}" for index in range(200)])
        states = s.select_context_states(population, 16, random.Random(1), "original")
        groups = Counter(group for _, _, _, group in states)
        self.assertEqual(groups, {"original": 1, "elite": 8, "non-elite": 7})
        self.assertEqual([key for key, _, _, group in states if group == "elite"],
                         [f"state{index:03d}" for index in range(8)])
        non_elite = [key for key, _, _, group in states if group == "non-elite"]
        self.assertTrue(all(40 <= int(key[5:]) < 160 for key in non_elite))
        self.assertEqual(len({key for key, _, _, _ in states}), 16)

    def test_shortage_redistribution_and_single_context(self):
        endpoints = ["e1", "e2", "e3"] * 13 + ["e1"] + [f"n{index}" for index in range(10)] * 16
        population = self.fake_population(endpoints[:200])
        states = s.select_context_states(population, 16, random.Random(2), "original")
        groups = Counter(group for _, _, _, group in states)
        self.assertEqual(groups["elite"], 3)
        self.assertEqual(groups["non-elite"], 7)
        self.assertEqual(groups["fill"], 3)
        self.assertEqual(len(states), 14)  # only thirteen distinct non-original states exist
        self.assertEqual(s.select_context_states(population, 1, random.Random(2), "original"),
                         [("original", (), None, "original")])

    def test_source_tie_breaks(self):
        population = [
            SimpleNamespace(uid=5, endpoint="x", active=(4, 2), rank=(1, 3, 2)),
            SimpleNamespace(uid=3, endpoint="x", active=(4, 1), rank=(1, 3, 2)),
            SimpleNamespace(uid=1, endpoint="x", active=(4, 1, 7), rank=(1, 3, 3)),
            SimpleNamespace(uid=2, endpoint="x", active=(4, 1), rank=(1, 3, 2)),
            SimpleNamespace(uid=9, endpoint="original", active=(), rank=(1, 9, 0)),
        ]
        states = s.select_context_states(population, 4, random.Random(0), "original")
        self.assertEqual(states[1][:3], ("x", (4, 1), 2))

    def test_identical_contexts_are_deduplicated(self):
        run = make_run(scrambled("R1 U1 C1"), solution_population=10)
        batch = run.build_context_batch(population_with(run, [], 10))
        self.assertEqual(len(batch.contexts), 1)
        batch = run.build_context_batch(population_with(run, ["C3", "C3", "C3"], 10))
        self.assertEqual(len(batch.contexts), 2)


class BlockTests(unittest.TestCase):
    def evaluate(self, run, population, blocks):
        batch = run.build_context_batch(population)
        run.evaluate_blocks(blocks, batch)
        run._index_blocks(blocks, batch)
        return batch

    def test_worsening_first_move_then_useful_prefix_qualifies(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        block = next(sequence for sequence in s.iter_canonical_sequences(2)
                     if s.manhattan_cost(key_after(original, sequence[:1])) > s.manhattan_cost(original)
                     and s.manhattan_cost(original) - s.manhattan_cost(key_after(original, sequence)) >= 1
                     and not s.is_solved_key(key_after(original, sequence)))
        run = make_run(s.cube_from_key(original), solution_population=10)
        record = run.make_block(block, origin="fixture")
        self.evaluate(run, population_with(run, [], 10), [record])
        outcome = record.evaluation.outcomes[0]
        self.assertIsNotNone(outcome.best)
        self.assertEqual(outcome.best.k, 2)
        self.assertEqual(record.evaluation.useful_contexts, 1)

    def test_block_helpful_on_one_context_and_harmful_on_another(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        context_a = key_after(original, "R3")
        block = s.parse_moves("D3")
        cost = s.manhattan_cost
        self.assertLess(cost(context_a), cost(original))
        self.assertGreaterEqual(cost(context_a) - cost(key_after(context_a, block)), 1)
        self.assertGreater(cost(key_after(original, block)), cost(original))
        run = make_run(s.cube_from_key(original), solution_population=5, contexts=2)
        record = run.make_block(block, origin="fixture")
        batch = self.evaluate(run, population_with(run, ["R3"], 5), [record])
        self.assertEqual([context.group for context in batch.contexts], ["original", "elite"])
        original_outcome, context_outcome = record.evaluation.outcomes
        self.assertIsNone(original_outcome.best)
        self.assertEqual(original_outcome.qualifying, ())
        self.assertEqual(context_outcome.best.endpoint, key_after(context_a, block))
        self.assertEqual(context_outcome.best.gain,
                         s.manhattan_cost(context_a) - s.manhattan_cost(key_after(context_a, block)))
        self.assertEqual(record.evaluation.useful_contexts, 1)

    def test_context_change_invalidates_and_order_does_not_matter(self):
        original = s.cube_from_key(key_after(SOLVED_KEY, "R1 U1 C1"))
        blocks_text = ["U3", "C3 U3", "D1 C3", "R2", "B1 L2"]
        run = make_run(original, solution_population=5, contexts=2)
        records = [run.make_block(s.parse_moves(text), origin="fixture") for text in blocks_text]
        first = self.evaluate(run, population_with(run, [], 5), records)
        first_scores = [record.evaluation.score for record in records]
        second = self.evaluate(run, population_with(run, ["C3"], 5), records)
        self.assertNotEqual(first.identity, second.identity)
        self.assertTrue(all(record.batch_serial == second.serial for record in records))
        self.assertNotEqual(first_scores, [record.evaluation.score for record in records])

        reversed_run = make_run(original, solution_population=5, contexts=2)
        reversed_records = [reversed_run.make_block(s.parse_moves(text), origin="fixture")
                            for text in reversed(blocks_text)]
        self.evaluate(reversed_run, population_with(reversed_run, ["C3"], 5), reversed_records)
        self.assertEqual({tuple(record.active): record.evaluation.score for record in records},
                         {tuple(record.active): record.evaluation.score for record in reversed_records})

    def test_solution_found_during_block_scoring_is_rebuilt_from_its_source(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        run = make_run(s.cube_from_key(original), solution_population=5, contexts=2)
        record = run.make_block(s.parse_moves("U3 R3"), origin="fixture")
        with self.assertRaises(s.SolutionFound) as caught:
            self.evaluate(run, population_with(run, ["C3"], 5), [record])
        entry = caught.exception.entry
        self.assertEqual(entry.active, s.parse_moves("C3 U3 R3"))
        self.assertEqual(entry.block_id, record.uid)
        self.assertTrue(s.is_solved_key(key_after(original, entry.active)))

    def test_targeted_append_creates_a_verified_improvement(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        run = make_run(s.cube_from_key(original), solution_population=5, contexts=1)
        record = run.make_block(s.parse_moves("C3 U3"), origin="fixture")
        self.evaluate(run, population_with(run, [], 5), [record])
        payload, block_id, targeted = run.choose_append_payload((), original)
        self.assertTrue(targeted)
        self.assertEqual((payload, block_id), (s.parse_moves("C3 U3"), record.uid))
        candidate = run.normalize_solution(payload)
        self.assertLess(candidate.rank, run.baseline_rank)
        self.assertEqual(key_after(original, candidate.active), candidate.endpoint)
        # Appending the exploratory block (R3) to that candidate solves, and the archive stops the run.
        solver = run.make_block(s.parse_moves("R3"), origin="fixture")
        run._block_representatives = [solver]
        run._block_index = {}
        follow_up, block_id, targeted = run.choose_append_payload(candidate.active, candidate.endpoint)
        self.assertFalse(targeted)
        solved = run.normalize_solution(candidate.active + follow_up)
        with self.assertRaises(s.SolutionFound):
            run.offer_to_archive(solved, uid=None, source="test", block_id=block_id)
        self.assertTrue(s.is_solved_key(key_after(original, run.archive.active)))

    def test_short_representative_fit_is_rechecked_for_a_longer_parent(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        long_parent = s.parse_moves("R2 U2 " * 6)
        self.assertEqual(key_after(original, long_parent), original)
        run = make_run(s.cube_from_key(original), solution_population=5, contexts=1,
                       solution_capacity=12, block_capacity=2)
        record = run.make_block(s.parse_moves("C3 U3"), origin="fixture")
        self.evaluate(run, population_with(run, [], 5), [record])
        self.assertTrue(run.choose_append_payload((), original)[2])
        payload, _, targeted = run.choose_append_payload(long_parent, original)
        self.assertFalse(targeted)
        self.assertEqual(payload, s.parse_moves("C3 U3"))

    def test_append_scores_each_block_once_then_falls_back(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        long_parent = s.parse_moves("R2 U2 " * 6)  # ends in U2, so only U-first blocks could merge and fit
        run = make_run(s.cube_from_key(original), solution_population=5, contexts=1,
                       solution_capacity=12, block_capacity=4)
        records = [run.make_block(s.parse_moves(text), origin="fixture")
                   for text in ("C3 U3", "C3", "D1 C3", "C3 B1", "B1 L2")]
        self.evaluate(run, population_with(run, [], 5), records)
        entries = run._block_index[original]
        self.assertGreater(len(entries), 1)
        self.assertFalse(any(s.MOVE_NAMES[sequence[0]][0] == "U" for sequence, _, _ in entries))
        scored = Counter()
        score = run._score_append_entry

        def counting(entry, base_reducer):
            scored[entry[1]] += 1
            return score(entry, base_reducer)

        run._score_append_entry = counting
        _, block_id, targeted = run.choose_append_payload(long_parent, original)
        self.assertFalse(targeted)
        self.assertEqual(set(scored), {uid for _, uid, _ in entries})
        self.assertEqual(set(scored.values()), {1})
        self.assertIn(block_id, {block.uid for block in records})

    def test_append_shortcut_for_context_source_parents_is_exact(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        run = make_run(s.cube_from_key(original), solution_population=5, contexts=2)
        records = [run.make_block(s.parse_moves(text), origin="fixture")
                   for text in ("C3 U3", "U3", "D3", "D1 C3", "B1 L2")]
        batch = self.evaluate(run, population_with(run, ["R3"], 5), records)
        self.assertEqual([context.source for context in batch.contexts], [(), s.parse_moves("R3")])
        for context in batch.contexts:
            entries = run._block_index[context.key]
            self.assertTrue(entries)
            replayed = s.MoveReducer.from_moves(context.source)
            for entry in entries:
                self.assertEqual(run._score_append_entry(entry, None), run._score_append_entry(entry, replayed))
        with mock.patch.object(s.MoveReducer, "from_moves", side_effect=AssertionError("reducer replayed")):
            for context in batch.contexts:
                self.assertTrue(run.choose_append_payload(context.source, context.key)[2])

    def test_sampled_append_matches_the_full_tournament(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        run = make_run(s.cube_from_key(original), solution_population=5, block_population=30, contexts=1)
        blocks = run.initial_block_population()
        self.evaluate(run, population_with(run, [], 5), blocks)
        empty = s.MoveReducer()
        options = [(run._score_append_entry(entry, empty), entry) for entry in run._block_index[original]]
        self.assertGreater(len(options), 5)
        self.assertTrue(all(score is not None for score, _ in options))
        for seed in range(100):
            run.rng = random.Random(seed)
            payload, uid, targeted = run.choose_append_payload((), original)
            (_, k), (sequence, expected_uid, _) = s.tournament_select(options, lambda option: option[0][0],
                                                                    random.Random(seed))
            self.assertTrue(targeted)
            self.assertEqual((payload, uid), (sequence[:k], expected_uid))

    def test_fresh_blocks_follow_the_reduced_length_schedule(self):
        run = make_run(block_population=20, block_capacity=4, solution_capacity=8)
        blocks = run.initial_block_population()
        self.assertEqual(Counter(len(block.active) for block in blocks), {1: 5, 2: 5, 3: 5, 4: 5})
        self.assertEqual(run.counters.target_length_fallbacks, 0)
        rng = random.Random(0)
        schedule = s.reduced_length_schedule(37, 5, rng)
        self.assertTrue(set(Counter(schedule).values()) <= {7, 8})

    def test_duplicate_cap_holds_at_block_capacity_one(self):
        run = make_run(block_population=100, block_capacity=1, solution_capacity=1)
        cap = max(2, -(-100 // 18))
        blocks = run.initial_block_population()
        self.assertLessEqual(max(Counter(block.active for block in blocks).values()), cap)
        run.population = run.initial_solution_population()
        batch = run.build_context_batch(run.population)
        run.evaluate_blocks(blocks, batch)
        run._index_blocks(blocks, batch)
        next_blocks = run.next_block_population(blocks)
        self.assertEqual(len(next_blocks), 100)
        self.assertLessEqual(max(Counter(block.active for block in next_blocks).values()), cap)

    def test_enumeration_fallback_fills_when_resampling_fails(self):
        run = make_run(block_population=10, block_capacity=2, solution_capacity=4)
        counts = {(allele,): run.max_block_copies for allele in s.REAL_MOVES if allele != 18}
        block = run.fresh_block(1, counts, origin="test")
        self.assertEqual(block.active, (18,))
        full = {(allele,): run.max_block_copies for allele in s.REAL_MOVES}
        block = run.fresh_block(1, full, origin="test")
        self.assertEqual(len(block.active), 2)
        self.assertEqual(run.counters.target_length_fallbacks, 1)

    def test_neutral_regime_keeps_generating_fresh_and_longer_blocks(self):
        run = make_run(block_population=30, min_block_gain=1e9, solution_population=10)
        run.population = run.initial_solution_population()
        blocks = run.initial_block_population()
        batch = run.build_context_batch(run.population)
        run.evaluate_blocks(blocks, batch)
        run._index_blocks(blocks, batch)
        self.assertFalse(any(block.evaluation.useful for block in blocks))
        next_blocks = run.next_block_population(blocks)
        self.assertEqual(Counter(block.slot for block in next_blocks)["elite"], 3)
        self.assertEqual(sum(1 for block in next_blocks if block.slot == "fresh"), 9)
        self.assertTrue(all(block.active for block in next_blocks))
        self.assertGreater(max(len(block.active) for block in next_blocks), 1)
        self.assertTrue(all(block.evaluation is None for block in next_blocks))

    def test_block_operators_and_fallbacks(self):
        run = make_run(block_population=10, block_capacity=4, solution_capacity=8)
        first = run.make_block(s.parse_moves("R1 U1"), origin="fixture")
        second = run.make_block(s.parse_moves("C2 D1 B3"), origin="fixture")
        for operator in s.BLOCK_OPERATORS:
            proposal, bound = run.vary_block(operator, [first, second])
            self.assertLessEqual(len(proposal), bound)
        self.assertEqual(run.vary_block("inverse", [first])[0], s.parse_moves("U3 R3"))
        self.assertEqual(run.vary_block("concatenate", [first, second])[0], s.parse_moves("R1 U1 C2 D1 B3"))

        representatives = [first, second]
        run.vary_block = lambda operator, parents: ((), run.block_capacity)
        empty = run.block_offspring(representatives, False, {})
        self.assertTrue(empty.active)
        self.assertEqual(empty.fallback_reason, "empty-result")

        run.vary_block = lambda operator, parents: (s.parse_moves("R1 U1 C1 D1 B1 L1"), 6)
        oversized = run.block_offspring(representatives, False, {})
        self.assertEqual(oversized.fallback_reason, "variation-attempts-exhausted")
        self.assertEqual(run.counters.block_oversized_rejections, s.BLOCK_VARIATION_ATTEMPTS)
        self.assertLessEqual(len(oversized.active), run.block_capacity)
        # Empty solution candidates stay empty: no random refill.
        self.assertEqual(run.normalize_solution(()).active, ())

    def test_variation_created_block_gets_an_attributed_improvement(self):
        original = key_after(SOLVED_KEY, "R1 U1 C1")
        run = make_run(s.cube_from_key(original), solution_population=5, contexts=1)
        first = run.make_block(s.parse_moves("C3"), origin="fixture")
        second = run.make_block(s.parse_moves("U3"), origin="fixture")
        proposal, _ = run.vary_block("concatenate", [first, second])
        evolved = run.make_block(s.simplify(proposal), origin="offspring", operator="concatenate",
                                 parents=(first.uid, second.uid))
        self.evaluate(run, population_with(run, [], 5), [evolved])
        self.assertTrue(evolved.evaluation.useful)
        child = run._baseline_start_offspring()
        self.assertEqual(child.block_id, evolved.uid)
        self.assertLess(child.rank, run.baseline_rank)
        self.assertEqual(run.counters.attributed_improvements[evolved.uid], 1)
        self.assertIn(f"B{evolved.uid}(offspring/concatenate", run.describe_attributions())

    def test_budget_expiry_inside_context_evaluation_preserves_the_archive(self):
        holder = {}

        def clock():
            run = holder.get("run")
            if run is not None and run.generation_in_progress == 3 and run.partial_phase == "block evaluation":
                return 100.0
            return 0.0

        budget = s.Budget(1.0, clock)
        run = make_run(scrambled("R2 B2 D1 R2 U1 C2 D2 C1"), budget=budget, solution_population=10,
                       block_population=10, contexts=4, generations=10)
        holder["run"] = run
        result = run.run()
        self.assertEqual(result.status, s.STATUS_BUDGET_EXHAUSTED)
        self.assertEqual((result.completed_generations, result.partial_generation, result.partial_phase),
                         (2, 3, "block evaluation"))
        self.assertEqual(s.finalize_result(result).status, s.STATUS_BUDGET_EXHAUSTED)
        self.assertLessEqual(result.archive.rank, run.baseline_rank)


class IntegrationTests(unittest.TestCase):
    SMALL = ["--solution-population", "10", "--block-population", "10", "--solution-capacity", "8",
             "--block-capacity", "4", "--contexts", "4", "--seed", "5"]

    def test_import_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ("import solver_deap_coevolution as m; from deap import creator; "
                    "print(hasattr(creator, m.SOLUTION_INDIVIDUAL_NAME))")
            completed = subprocess.run([sys.executable, "-c", code], cwd=directory, capture_output=True,
                                       text=True, env={**os.environ, "PYTHONPATH": REPO_DIR})
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "False\n")
            self.assertEqual(os.listdir(directory), [])

    def test_reduced_run_exercises_both_populations(self):
        result = s.run_search(s.SearchConfig(seed=4, generations=5, solution_population=10,
                                             block_population=10, solution_capacity=8, block_capacity=4,
                                             contexts=4), log=lambda message: None)
        self.assertEqual(result.status, s.STATUS_BUDGET_EXHAUSTED)
        self.assertEqual(result.completed_generations, 5)
        self.assertTrue(result.block_population_created)
        self.assertEqual([record["gen"] for record in result.history], [0, 1, 2, 3, 4, 5])
        self.assertGreater(result.counters.block_sequence_evaluations, 0)
        self.assertEqual(result.counters.baseline_start_attempts, 5)

    def test_short_scrambles_are_solved_and_replay_verified(self):
        for moves in ("R1", "C2", "R1 U1", "B3 D2"):
            with self.subTest(moves=moves):
                result = s.run_search(s.SearchConfig(seed=2, generations=30, solution_population=20,
                                                     block_population=20), scrambled(moves),
                                      log=lambda message: None)
                final = s.finalize_result(result)
                self.assertEqual(final.status, s.STATUS_SOLVED)
                self.assertTrue(s.is_solved_key(key_after(result.input_key, final.moves)))

    def test_zero_generations(self):
        code, output = quiet(s.main, ["--generations", "0"] + self.SMALL)
        self.assertEqual(code, 1)
        self.assertIn("Block population: not created", output)
        self.assertIn("Moves (0, reduced): (empty move list)", output)
        self.assertTrue(output.rstrip().splitlines()[-2].startswith("Status: BUDGET_EXHAUSTED"))

    def test_time_expiry_returns_budget_exhausted(self):
        ticks = iter(range(10 ** 9))
        code, output = quiet(s.main, ["--time-limit", "500"] + self.SMALL, clock=lambda: float(next(ticks)))
        self.assertEqual(code, 1)
        self.assertIn("time limit reached", output)

    def test_sigint_returns_interrupted(self):
        before = signal.getsignal(signal.SIGINT)
        calls = {"count": 0}

        def clock():
            calls["count"] += 1
            if calls["count"] == 50:
                os.kill(os.getpid(), signal.SIGINT)
            return 0.0

        code, output = quiet(s.main, ["--time-limit", "100"] + self.SMALL, clock=clock)
        self.assertEqual(code, 130)
        self.assertIn("Status: INTERRUPTED", output)
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_plotting_never_overwrites(self):
        directory = tempfile.mkdtemp()
        previous = os.getcwd()
        try:
            os.chdir(directory)
            for _ in range(2):
                code, output = quiet(s.main, ["--generations", "0", "--plot"] + self.SMALL)
                self.assertEqual(code, 1)
            code, output = quiet(s.main, ["--generations", "3", "--plot"] + self.SMALL)
            self.assertEqual(sorted(os.listdir(directory)),
                             ["coevolution_progress.png", "coevolution_progress_1.png",
                              "coevolution_progress_2.png"])
            self.assertIn("coevolution_progress_2.png", output)
        finally:
            os.chdir(previous)
            shutil.rmtree(directory)

    def test_replay_mismatch_is_an_error(self):
        result = s.run_search(s.SearchConfig(seed=1, generations=1, solution_population=10,
                                             block_population=10, contexts=2), scrambled("R1 U1 C1 D2"),
                              log=lambda message: None)
        self.assertEqual(s.finalize_result(result).status, result.status)
        result.archive = dataclasses.replace(result.archive, cost=result.archive.cost + 1)
        final = s.finalize_result(result)
        self.assertEqual(final.status, s.STATUS_ERROR)
        self.assertEqual(s.EXIT_CODES[final.status], 3)

    def test_benchmark_overrides_are_in_memory_only(self):
        import benchmark_coevolution as benchmark

        with open(benchmark.OLD_SOLVER_PATH, "rb") as handle:
            before = handle.read()
        setup, ga_run, overrides = benchmark.prepare_old_solver(before.decode(), 42, 7)
        source = ast.unparse(setup)
        for expected in ("random.seed(42)", "POPULATION_SIZE = 200", "GENERATIONS = 7", "MAX_GENES = 32"):
            self.assertIn(expected, source)
        self.assertNotIn("plt.show", source + ast.unparse(ga_run))
        self.assertIn("eaSimpleWithElitism", ast.unparse(ga_run))
        self.assertEqual(len(overrides), 5)
        with open(benchmark.OLD_SOLVER_PATH, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_existing_sources_are_unchanged(self):
        if shutil.which("git") is None:
            self.skipTest("git unavailable")
        completed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "cube.py", "solver_deap.py"],
                                   cwd=REPO_DIR)
        self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
