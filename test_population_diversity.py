"""Acceptance tests for solution-population diversity and stagnation warnings
(solution_population_diversity_specs.md, section 7).

Run with:  uv run python -m unittest test_population_diversity -v
"""

import contextlib
import dataclasses
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from itertools import combinations
from types import SimpleNamespace
from unittest import mock

import solver_deap_coevolution as s

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVED = s.REFERENCE_SOLVED_KEY
CORNERS = [slot for slot in s._CUBIE_SLOTS if len(slot) == 3]
EDGES = [slot for slot in s._CUBIE_SLOTS if len(slot) == 2]
DEFAULTS = {"min_unique_ratio": 0.10, "max_dominance": 0.80, "min_cubie_distance": 2.0, "stagnation_generations": 100}
_saved_environment = {}


def setUpModule():
    _saved_environment["XDG_DATA_HOME"] = os.environ.get("XDG_DATA_HOME")
    _saved_environment["directory"] = tempfile.mkdtemp()
    os.environ["XDG_DATA_HOME"] = _saved_environment["directory"]
    # The CLI's finisher is on by default and caches its table under XDG_CACHE_HOME.
    _saved_environment["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = os.path.join(_saved_environment["directory"], "cache")


def tearDownModule():
    shutil.rmtree(_saved_environment["directory"], ignore_errors=True)
    if _saved_environment["XDG_DATA_HOME"] is None:
        os.environ.pop("XDG_DATA_HOME", None)
    else:
        os.environ["XDG_DATA_HOME"] = _saved_environment["XDG_DATA_HOME"]
    if _saved_environment["XDG_CACHE_HOME"] is None:
        os.environ.pop("XDG_CACHE_HOME", None)
    else:
        os.environ["XDG_CACHE_HOME"] = _saved_environment["XDG_CACHE_HOME"]


def rotated(key, slot):
    """Turn one cubie in place: a corner twist or an edge flip (one differing cubie)."""
    stickers = list(key)
    colours = [stickers[p] for p in slot]
    for position, colour in zip(slot, colours[1:] + colours[:1]):
        stickers[position] = colour
    return "".join(stickers)


def distinct_states(count, seed=1, length=15):
    rng, states = random.Random(seed), []
    while len(states) < count:
        key = s.replay_key(SOLVED, s.generate_reduced_block(length, rng))
        if key not in states:
            states.append(key)
    return states


def build(non_reset, reset_key=SOLVED, slot="offspring"):
    """A population with len(non_reset) // 4 explicit reset slots (the 20/80 split)."""
    individuals = [SimpleNamespace(endpoint=key, slot=slot, origin="offspring", active=(1,)) for key in non_reset]
    individuals += [SimpleNamespace(endpoint=reset_key, slot="reset", origin="reset", active=())
                    for _ in range(len(non_reset) // 4)]
    return individuals


def measure(population, generations_without=0, thresholds=DEFAULTS):
    record = s.measure_population_diversity(population, len(population))
    record["generation"] = 50
    record["generations_without_quality_improvement"] = generations_without
    active = s.diversity_conditions(record, generations_without, thresholds)
    record["flags"] = [flag for flag, _ in active]
    record["messages"] = active
    return record


def small_config(**overrides):
    values = dict(seed=1, generations=101, solution_population=20, block_population=20, solution_capacity=16,
                  block_capacity=8, contexts=6, min_block_gain=1.0, log_every=1000)
    values.update(overrides)
    return s.SearchConfig(**values)


def make_run(config, cube=None, log=None):
    cube = s.default_input_cube() if cube is None else cube
    return s.CoevolutionRun(config, s.validate_input_cube(cube), s.validate_config(config), config.seed,
                            log=log if log is not None else (lambda message: None))


def brute_force_mean(keys):
    decoded = [dict(s.decode_cubie_poses(key)) for key in keys]
    distances = [sum(a[cubie] != b[cubie] for cubie in a) for a, b in combinations(decoded, 2)]
    return sum(distances) / len(distances)


class ScheduleTests(unittest.TestCase):
    def test_checks_run_only_at_completed_fifty_generation_boundaries(self):
        directory = tempfile.mkdtemp()
        try:
            for overrides in ({}, {"log_every": 1}, {"block_library": os.path.join(directory, "b.json")}):
                with self.subTest(**overrides):
                    result = s.run_search(small_config(generations=123, **overrides), log=lambda message: None)
                    self.assertEqual([r["generation"] for r in result.diversity_records], [50, 100])
                    self.assertEqual([row["gen"] for row in result.history if "diversity" in row], [50, 100])
                    rows = {row["gen"]: row for row in result.history}
                    self.assertIs(rows[50]["diversity"], result.diversity_records[0])
        finally:
            shutil.rmtree(directory)

    def test_zero_solved_partial_and_terminating_runs(self):
        self.assertEqual(s.run_search(small_config(generations=0), log=lambda m: None).diversity_records, [])
        solved = s.run_search(small_config(), s.cube_from_key(SOLVED), log=lambda m: None)
        self.assertEqual(solved.diversity_records, [])
        one_move = s.run_search(small_config(), s.replay(s.cube_from_key(SOLVED), s.parse_moves("R1")),
                                log=lambda m: None)
        self.assertEqual((one_move.status, one_move.diversity_records), (s.STATUS_SOLVED, []))

        holder = {}

        def clock():
            run = holder.get("run")
            if run is not None and run.generation_in_progress == 150 and run.partial_phase == "block evaluation":
                return 100.0
            return 0.0

        config = small_config(generations=300)
        run = s.CoevolutionRun(config, s.validate_input_cube(s.default_input_cube()), s.validate_config(config), 1,
                               budget=s.Budget(1.0, clock), log=lambda m: None)
        holder["run"] = run
        result = run.run()
        self.assertEqual((result.partial_generation, [r["generation"] for r in result.diversity_records]),
                         (150, [50, 100]))

    def test_library_timeout_does_not_suppress_a_completed_check(self):
        directory = tempfile.mkdtemp()
        try:
            run = make_run(small_config(generations=60, block_library=os.path.join(directory, "b.json")))

            def checkpoint(include_snapshot):
                if run.completed_generation == 50:
                    raise s.BudgetExpired("time limit reached")

            run._library_checkpoint = checkpoint
            result = run.run()
            self.assertEqual((result.status, [r["generation"] for r in result.diversity_records]),
                             (s.STATUS_BUDGET_EXHAUSTED, [50]))
        finally:
            shutil.rmtree(directory)


class ConditionTests(unittest.TestCase):
    def test_each_condition_alone_and_all_together(self):
        base = distinct_states(80)
        low_unique = build([base[0]] * 40 + [base[1]] * 40)
        dominant = build([base[0]] * 65 + base[1:16])
        twists = [rotated(SOLVED, slot) for slot in CORNERS] + [rotated(SOLVED, slot) for slot in EDGES]
        low_spread = build([SOLVED] * 60 + twists + twists[:0])
        cases = {
            "LOW_UNIQUE_STATES": (low_unique, 0),
            "DOMINANT_STATE": (dominant, 0),
            "LOW_CUBIE_SPREAD": (low_spread, 0),
            "NO_QUALITY_PROGRESS": (build(base), 117),
        }
        for expected, (population, stalled) in cases.items():
            with self.subTest(condition=expected):
                record = measure(population, stalled)
                self.assertEqual(record["flags"], [expected])
                text = "\n".join(s.diversity_report_lines(record, False))
                self.assertEqual(text.count("WARNING"), 1)
        everything = measure(build([SOLVED] * 76 + twists[:4]), 117)
        self.assertEqual(everything["flags"], list(s.DIVERSITY_CONDITIONS))
        text = "\n".join(s.diversity_report_lines(everything, False))
        self.assertEqual(text.count("WARNING"), 1)
        self.assertIn("low diversity with stalled cube-quality progress", text)
        for condition in s.DIVERSITY_CONDITIONS:
            self.assertIn(f"  {condition}: ", text)
        self.assertIn("Intentional reset slots excluded from thresholds: 20/100.", text)

    def test_boundaries_use_unrounded_values(self):
        def synthetic(unique_ratio=0.5, dominance=0.5, mean=10.0):
            subset = {"count": 80, "unique": 1, "unique_ratio": unique_ratio, "dominant": 1,
                      "dominant_ratio": dominance, "cubie_mean": mean, "cubie_reason": None}
            return {"available": True, "non_reset": subset, "whole": subset}

        def flags(record, stalled=0):
            return [flag for flag, _ in s.diversity_conditions(record, stalled, DEFAULTS)]

        self.assertEqual(flags(synthetic(unique_ratio=0.10)), [])
        self.assertEqual(flags(synthetic(unique_ratio=0.0999999)), ["LOW_UNIQUE_STATES"])  # displays 10.0%
        self.assertEqual(flags(synthetic(unique_ratio=0.1000001)), [])
        self.assertEqual(flags(synthetic(dominance=0.80)), [])
        self.assertEqual(flags(synthetic(dominance=0.8000001)), ["DOMINANT_STATE"])  # displays 80.00%
        self.assertEqual(flags(synthetic(dominance=0.7999999)), [])
        self.assertEqual(flags(synthetic(mean=2.0)), [])
        self.assertEqual(flags(synthetic(mean=1.9999999)), ["LOW_CUBIE_SPREAD"])  # displays 2.00
        self.assertEqual(flags(synthetic(mean=2.0000001)), [])
        self.assertEqual(flags(synthetic(), 99), [])
        self.assertEqual(flags(synthetic(), 100), ["NO_QUALITY_PROGRESS"])
        self.assertEqual(flags(synthetic(), 101), ["NO_QUALITY_PROGRESS"])
        # Exact boundaries on real populations: 8/80 unique, 64/80 dominant, mean exactly 2.0.
        base = distinct_states(16)
        self.assertNotIn("LOW_UNIQUE_STATES", measure(build([base[i % 8] for i in range(80)]))["flags"])
        self.assertIn("LOW_UNIQUE_STATES", measure(build([base[i % 7] for i in range(80)]))["flags"])
        self.assertEqual(measure(build([base[0]] * 64 + base[1:16] + [base[1]]))["flags"], [])
        self.assertEqual(measure(build([base[0]] * 65 + base[1:16]))["flags"], ["DOMINANT_STATE"])
        flips = [rotated(SOLVED, slot) for slot in EDGES]
        exactly_two = measure(build(flips[:8]))
        self.assertEqual(exactly_two["non_reset"]["cubie_mean"], 2.0)
        self.assertNotIn("LOW_CUBIE_SPREAD", exactly_two["flags"])
        below = measure(build([SOLVED] + flips[:7]))
        self.assertEqual(below["non_reset"]["cubie_mean"], 1.75)
        self.assertIn("LOW_CUBIE_SPREAD", below["flags"])


class PopulationRoleTests(unittest.TestCase):
    def test_only_current_reset_slots_are_excluded(self):
        others = distinct_states(19)
        population = build([SOLVED] * 61 + others)  # 61 + 19 non-reset; 20 resets also at SOLVED
        record = measure(population)
        self.assertEqual(record["sizes"], {"whole": 100, "non_reset": 80, "reset": 20})
        self.assertGreater(record["whole"]["dominant_ratio"], 0.80)  # whole population alone would trigger
        self.assertEqual(record["flags"], [])
        population[0].slot, population[0].origin = "elite", "reset"  # elite descended from an old reset
        population[1].active = ()  # empty offspring
        population[2].slot = "baseline-start"
        self.assertEqual(measure(population)["sizes"]["non_reset"], 80)

    def test_endpoint_identity(self):
        one = s.replay_key(SOLVED, s.parse_moves("R1"))
        three = s.replay_key(SOLVED, s.parse_moves("R3"))
        self.assertEqual(s.manhattan_cost(one), s.manhattan_cost(three))  # equal H, different states
        population = build([one, one, three, three])
        population[0].active, population[1].active = s.parse_moves("R1"), s.parse_moves("R3 R2")  # same endpoint
        record = measure(population)
        self.assertEqual((record["non_reset"]["unique"], record["non_reset"]["count"]), (2, 4))


class CubieMeanTests(unittest.TestCase):
    def test_frequency_formula_matches_brute_force(self):
        rng = random.Random(9)
        states = distinct_states(6, seed=3)
        variants = [rotated(states[0], CORNERS[0]), rotated(rotated(states[0], CORNERS[0]), CORNERS[0]),
                    rotated(states[1], EDGES[3]), s.replay_key(states[2], s.parse_moves("R1"))]
        pool = states + variants
        for _ in range(60):
            keys = [rng.choice(pool) for _ in range(rng.randint(2, 12))]
            multiplicities = {}
            for key in keys:
                multiplicities[key] = multiplicities.get(key, 0) + 1
            poses = {key: s.decode_cubie_poses(key) for key in multiplicities}
            mean, distance_sum, pairs = s.mean_cubie_difference(multiplicities, poses)
            self.assertEqual(pairs, len(keys) * (len(keys) - 1) // 2)
            self.assertAlmostEqual(mean, brute_force_mean(keys), places=12)
            self.assertTrue(0 <= mean <= 20)
        self.assertEqual(s.mean_cubie_difference({SOLVED: 5}, {SOLVED: s.decode_cubie_poses(SOLVED)})[0], 0.0)
        self.assertEqual(brute_force_mean([SOLVED, rotated(SOLVED, CORNERS[0])]), 1.0)  # orientation only
        self.assertEqual(brute_force_mean([SOLVED, s.replay_key(SOLVED, s.parse_moves("R1"))]), 8.0)


class StagnationTests(unittest.TestCase):
    def test_only_strict_quality_improvements_reset_the_clock(self):
        cube = s.replay(s.cube_from_key(SOLVED), s.parse_moves("R1 U1 C1"))
        run = make_run(small_config(), cube)
        run.generation_in_progress = 7
        better = run.normalize_solution(s.parse_moves("C3"))
        self.assertTrue(run.offer_to_archive(better, uid=None, source="block B1 prefix 1 on context 0"))
        self.assertEqual(run.last_quality_improvement_generation, 7)
        # Same H, shorter sequence: archive improves, quality clock does not.
        run.archive = dataclasses.replace(run.archive, active=s.parse_moves("C3 R2 R2"))
        run.generation_in_progress = 30
        self.assertTrue(run.offer_to_archive(better, uid=None, source="length only"))
        self.assertEqual((run.last_improvement_generation, run.last_quality_improvement_generation), (30, 7))
        run.population = build(distinct_states(16))
        run.config = dataclasses.replace(run.config, solution_population=20)
        record = run.diversity_checkpoint(150)
        self.assertEqual(record["generations_without_quality_improvement"], 143)
        self.assertEqual(record["flags"], ["NO_QUALITY_PROGRESS"])

    def test_stagnation_alone_is_not_called_homogeneity(self):
        record = measure(build(distinct_states(80)), 120)
        text = "\n".join(s.diversity_report_lines(record, False))
        self.assertIn("WARNING: no cube-quality progress", text)
        self.assertNotIn("low solution-population diversity", text)
        self.assertNotIn("low diversity", text)
        self.assertNotIn("Intentional reset slots", text)
        diversity_only = "\n".join(s.diversity_report_lines(measure(build([SOLVED] * 80)), False))
        self.assertIn("WARNING: low solution-population diversity", diversity_only)
        self.assertNotIn("stalled", diversity_only)


class UnavailableDataTests(unittest.TestCase):
    def test_undecodable_states_and_small_subsets_never_report_zero_spread(self):
        stickers = list(SOLVED)
        stickers[0], stickers[10] = stickers[10], stickers[0]
        broken = "".join(stickers)
        record = measure(build([broken] * 76 + distinct_states(4)))
        subset = record["non_reset"]
        self.assertIsNone(subset["cubie_mean"])
        self.assertIn("consistent set of cubies", subset["cubie_reason"])
        self.assertEqual(record["flags"], ["LOW_UNIQUE_STATES", "DOMINANT_STATE"])
        self.assertIn("unavailable non-reset", "\n".join(s.diversity_report_lines(record, False)))
        self.assertEqual(s.mean_cubie_difference({SOLVED: 1}, {SOLVED: s.decode_cubie_poses(SOLVED)})[0], None)

    def test_inconsistent_population_roles_are_reported(self):
        population = build([SOLVED] * 80)
        population[-1].slot = "offspring"  # one reset slot missing
        record = measure(population, 120)
        self.assertFalse(record["available"])
        self.assertIn("inconsistent", record["reason"])
        self.assertEqual(record["flags"], ["NO_QUALITY_PROGRESS"])  # stagnation is still measured
        self.assertIn("diversity metrics unavailable", "\n".join(s.diversity_report_lines(record, False)))
        missing = s.measure_population_diversity(None, 100)
        self.assertFalse(missing["available"])


class LifecycleTests(unittest.TestCase):
    def test_warning_repeats_while_active_then_one_recovery(self):
        lines = []
        run = make_run(small_config(solution_population=100, block_population=20, stagnation_generations=10 ** 6),
                       log=lines.append)
        flagged, healthy = build([SOLVED] * 80), build(distinct_states(80))
        sequence = [(50, flagged), (100, flagged), (150, healthy), (200, healthy)]
        per_checkpoint = []
        for generation, population in sequence:
            run.population = population
            start = len(lines)
            run.diversity_checkpoint(generation)
            per_checkpoint.append("\n".join(lines[start:]))
        self.assertEqual([text.count("WARNING") for text in per_checkpoint], [1, 1, 0, 0])
        self.assertEqual([text.count("recovered") for text in per_checkpoint], [0, 0, 1, 0])
        self.assertEqual([r["consecutive_flagged"] for r in run.diversity_records], [1, 2, 0, 0])

    def test_final_report_does_not_replay_the_warning(self):
        args = ["--generations", "51", "--solution-population", "10", "--block-population", "10",
                "--solution-capacity", "8", "--block-capacity", "4", "--contexts", "4", "--seed", "5",
                "--no-block-library", "--diversity-min-unique-ratio", "1.0", "--no-mitm"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(s.main(args), 1)
        text = output.getvalue()
        self.assertEqual(text.count("WARNING: low solution-population diversity"), 1)
        self.assertIn("Diversity checks: 1 (last at generation 50; 1 flagged)", text)


class NoAlgorithmChangeTests(unittest.TestCase):
    def test_enabled_and_disabled_runs_are_identical(self):
        runs = {}
        for enabled in (True, False):
            run = make_run(small_config(generations=101, diversity_checks=enabled))
            result = run.run()
            runs[enabled] = (run, result)
        (on, on_result), (off, off_result) = runs[True], runs[False]
        self.assertEqual(len(on_result.diversity_records), 2)
        self.assertEqual(off_result.diversity_records, [])
        self.assertEqual(on_result.archive, off_result.archive)
        self.assertEqual(dataclasses.asdict(on_result.counters), dataclasses.asdict(off_result.counters))
        strip = lambda history: [{k: v for k, v in row.items() if k != "diversity"} for row in history]
        self.assertEqual(strip(on_result.history), strip(off_result.history))
        self.assertEqual(on.rng.getstate(), off.rng.getstate())
        self.assertEqual([i.active for i in on.population], [i.active for i in off.population])

    def test_standalone_check_leaves_rng_population_and_archive_untouched(self):
        run = make_run(small_config(generations=3))
        run.run()
        state, archive = run.rng.getstate(), run.archive
        snapshot = [(i.uid, i.active, i.endpoint, i.slot, i.fitness.values) for i in run.population]
        run.diversity_checkpoint(50)
        self.assertEqual(run.rng.getstate(), state)
        self.assertIs(run.archive, archive)
        self.assertEqual([(i.uid, i.active, i.endpoint, i.slot, i.fitness.values) for i in run.population], snapshot)


class SideEffectTests(unittest.TestCase):
    def test_manhattan_checks_build_no_pattern_tables(self):
        code = ("import sys; sys.path.insert(0, %r); import solver_deap_coevolution as s\n"
                "config = s.SearchConfig(seed=1, generations=50, solution_population=20, block_population=20,\n"
                "                        solution_capacity=16, block_capacity=8, contexts=6)\n"
                "result = s.run_search(config, log=lambda m: None)\n"
                "assert len(result.diversity_records) == 1\n"
                "assert s._grouped_cubie_tables.cache_info().currsize == 0\n") % REPO_DIR
        directory = tempfile.mkdtemp()
        try:
            completed = subprocess.run([sys.executable, "-c", code], cwd=directory, capture_output=True, text=True,
                                       env={**os.environ, "XDG_DATA_HOME": directory})
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(os.listdir(directory), [])  # no files written
        finally:
            shutil.rmtree(directory)

    def test_checkpoint_replays_nothing_scores_nothing_and_keeps_no_cache(self):
        run = make_run(small_config(generations=3))
        run.run()
        attributes = set(vars(run))
        forbidden = mock.Mock(side_effect=AssertionError("not allowed during a diversity check"))
        with mock.patch.object(s, "replay", forbidden), mock.patch.object(s, "replay_key", forbidden), \
                mock.patch.object(run, "cost", forbidden), mock.patch.object(run.objective.__class__, "cost", forbidden), \
                mock.patch("builtins.open", forbidden):
            run.diversity_checkpoint(50)
        self.assertEqual(set(vars(run)), attributes)


class ConfigurationTests(unittest.TestCase):
    def test_invalid_thresholds_are_rejected(self):
        bad = [dict(diversity_min_unique_ratio=-0.1), dict(diversity_min_unique_ratio=1.1),
               dict(diversity_min_unique_ratio=float("nan")), dict(diversity_max_dominance=2.0),
               dict(diversity_max_dominance=True), dict(diversity_min_cubie_distance=20.5),
               dict(diversity_min_cubie_distance=float("inf")), dict(stagnation_generations=0),
               dict(stagnation_generations=1.5), dict(diversity_checks="yes")]
        for overrides in bad:
            with self.subTest(**{k: repr(v) for k, v in overrides.items()}), self.assertRaises(s.ConfigError):
                s.validate_config(s.SearchConfig(**overrides))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(s.main(["--diversity-max-dominance", "1.5"]), 2)

    def test_cli_options_and_logger_capture(self):
        args = s.build_arg_parser().parse_args(["--diversity-min-unique-ratio", "0.2", "--diversity-max-dominance",
                                                "0.5", "--diversity-min-cubie-distance", "3", "--stagnation-generations",
                                                "40"])
        config = s.config_from_args(args)
        self.assertEqual((config.diversity_checks, config.diversity_min_unique_ratio, config.diversity_max_dominance,
                          config.diversity_min_cubie_distance, config.stagnation_generations), (True, 0.2, 0.5, 3.0, 40))
        self.assertFalse(s.config_from_args(s.build_arg_parser().parse_args(["--no-diversity-checks"])).diversity_checks)
        lines = []
        run = make_run(small_config(generations=50), log=lines.append)
        result = run.run()
        record = result.diversity_records[0]
        self.assertTrue(any(line.startswith("[gen 50] solution-population diversity") for line in lines))
        self.assertEqual(record["generation"], 50)
        self.assertEqual(record["sizes"], {"whole": 20, "non_reset": 16, "reset": 4})
        self.assertEqual(record["thresholds"], DEFAULTS)
        self.assertEqual(record["generations_without_quality_improvement"], 50 - run.last_quality_improvement_generation)
        self.assertGreaterEqual(result.diversity_seconds, record["seconds"])

    def test_disabled_checks_do_no_diagnostic_work(self):
        with mock.patch.object(s, "decode_cubie_poses", side_effect=AssertionError("diagnostic work")), \
                mock.patch.object(s, "measure_population_diversity", side_effect=AssertionError("diagnostic work")):
            result = s.run_search(small_config(generations=101, diversity_checks=False), log=lambda m: None)
        self.assertEqual((result.diversity_records, result.diversity_seconds), ([], 0.0))


if __name__ == "__main__":
    unittest.main()
