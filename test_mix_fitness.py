"""Tests for --fitness mix: the five state primitives, each scaled by its upper bound."""

import math
import random
import unittest

import solver_deap_coevolution as s
from grouped_cubie_fitness import UNREACHED

SOLVED = s.REFERENCE_SOLVED_KEY


def scrambled(rng, length):
    return s.replay_key(SOLVED, [rng.choice(s.REAL_MOVES) for _ in range(length)])


def primitives(key):
    """The five primitives from their own functions, independent of mix_cost's single decoding."""
    tables = s._grouped_cubie_tables()
    return {"manhattan": s.manhattan_cost(key), "misplaced": s.misplaced_cost(key), "cubie": s.cubie_cost(key),
            "edges": s.make_objective("edges").cost(key), "pattern": sum(tables.distances(s._face_indices(key)))}


class MixFitnessTests(unittest.TestCase):
    def test_bounds_and_equal_scaling(self):
        multipliers, maxima, scale = s.mix_multipliers()
        self.assertEqual(maxima, {"manhattan": 336, "misplaced": 48, "cubie": 52, "edges": 20, "pattern": 38})
        self.assertEqual(scale, math.lcm(336, 48, 52, 20, 38))
        for name, maximum in maxima.items():
            self.assertEqual(multipliers[name] * maximum, scale)  # every term spans 0..scale
        tables = s._grouped_cubie_tables()
        self.assertEqual(sorted(max(d for d in table if d != UNREACHED) for table in tables._distance_tables),
                         [7, 7, 8, 8, 8])
        rng = random.Random(11)
        for _ in range(2000):  # no reachable state exceeds a bound
            values = primitives(scrambled(rng, rng.randint(0, 30)))
            for name, value in values.items():
                self.assertLessEqual(value, maxima[name], name)

    def test_cost_is_the_scaled_sum_of_independent_primitives(self):
        multipliers, _, _ = s.mix_multipliers()
        rng = random.Random(12)
        for _ in range(500):
            key = scrambled(rng, rng.randint(0, 25))
            values = primitives(key)
            expected = sum(multipliers[name] * value for name, value in values.items())
            self.assertEqual(s.mix_cost(key), expected)
            self.assertIsInstance(s.mix_cost(key), int)
            components = s._grouped_cubie_tables().components(s._face_indices(key))
            self.assertEqual(components, (values["pattern"], values["cubie"], values["misplaced"]))

    def test_zero_exactly_when_solved(self):
        self.assertEqual(s.mix_cost(SOLVED), 0)
        for allele in s.REAL_MOVES:
            self.assertGreater(s.mix_cost(s.replay_key(SOLVED, (allele,))), 0)
        rng = random.Random(13)
        for _ in range(300):
            key = scrambled(rng, rng.randint(1, 25))
            self.assertEqual(s.mix_cost(key) == 0, s.is_solved_key(key))

    def test_cubie_based_terms_ignore_colour_names(self):
        rng = random.Random(14)
        key = scrambled(rng, 12)
        recoloured = key.translate(str.maketrans("GRWOYB", "RWOYBG"))
        tables = s._grouped_cubie_tables()
        self.assertEqual(tables.components(s._face_indices(key)), tables.components(s._face_indices(recoloured)))

    def test_objective_cli_and_validation(self):
        objective = s.make_objective("mix")
        self.assertEqual((objective.state_function, objective.integer_valued), ("mix", True))
        args = s.build_arg_parser().parse_args(["--fitness", "mix"])
        self.assertEqual(s.validate_config(s.config_from_args(args)).state_function, "mix")
        self.assertIn("manhattan/336 + misplaced/48 + cubie/52 + wrong cubies/20 + pattern/38", objective.describe())
        with self.assertRaises(s.ConfigError):
            s.make_objective("mix", manhattan_weight=1.0)
        stickers = list(SOLVED)
        stickers[0], stickers[10] = stickers[10], stickers[0]
        with self.assertRaises(s.InvalidInputError):
            s.run_search(s.SearchConfig(fitness="mix", generations=0), s.cube_from_key("".join(stickers)),
                         log=lambda message: None)

    def test_short_run_finalises_with_mix_costs(self):
        config = s.SearchConfig(fitness="mix", seed=3, generations=5, solution_population=20, block_population=20,
                                contexts=4, solution_capacity=16, block_capacity=8, block_library=None, mitm=False)
        rng = random.Random(15)
        result = s.run_search(config, s.cube_from_key(scrambled(rng, 6)), log=lambda message: None)
        final = s.finalize_result(result)
        self.assertEqual(final.problems, [])
        self.assertEqual(result.archive.cost, s.mix_cost(result.archive.endpoint))


if __name__ == "__main__":
    unittest.main()
