"""Independent replay checks for the optional grouped-cubie objective."""

import contextlib
import io
import random
import string
import subprocess
import sys
import unittest

import solver_deap_coevolution as s


SOLVED = "".join(color * 9 for color in "GRWOYB")
LABELLED = (string.ascii_letters + "01")[:54]


def moved(key, allele):
    # Reference rotations deliberately go through cube.py, not the fast tables.
    return s.state_key(s.replay(s.cube_from_key(key), (allele,)))


class GroupedCubieTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = s._grouped_cubie_tables()

    def distances(self, key):
        return self.tables.distances(s._face_indices(key))

    def test_patterns_cover_every_piece_and_all_reachable_placements(self):
        self.assertEqual(sorted(m for p in self.tables.patterns for m in p.members), list(range(20)))
        self.assertEqual([p.reachable_count for p in self.tables.patterns],
                         [136080, 136080, 190080, 190080, 190080])
        for pattern in self.tables.patterns:
            self.assertEqual(len(pattern.members), 4)
            self.assertEqual(len({len(s._CUBIE_SLOTS[m]) for m in pattern.members}), 1)
            self.assertEqual(len(pattern.distances) - pattern.distances.count(255), pattern.reachable_count)
            self.assertEqual(pattern.distances.count(0), 1)
        self.assertEqual(self.distances(SOLVED), (0, 0, 0, 0, 0))

    def test_exact_distances_against_independent_shallow_bfs(self):
        # Enumerate EVERY full-cube path through depth two using real rotations.
        # Minimizing depth for each labelled-piece projection gives independent
        # exact abstract distances for all the projections reached here.
        projections = [tuple(p for m in pattern.members for p in s._CUBIE_SLOTS[m])
                       for pattern in self.tables.patterns]
        reference = [{} for _ in projections]
        observations = []
        frontier, seen = {LABELLED}, {LABELLED}
        recolour = str.maketrans(LABELLED, SOLVED)
        for depth in range(3):
            next_frontier = set()
            for key in frontier:
                signatures = [tuple(key.index(LABELLED[p]) for p in projection) for projection in projections]
                for table, signature in zip(reference, signatures):
                    table.setdefault(signature, depth)
                observations.append((signatures, self.distances(key.translate(recolour))))
                if depth < 2:
                    for allele in s.REAL_MOVES:
                        neighbor = moved(key, allele)
                        if neighbor not in seen:
                            seen.add(neighbor)
                            next_frontier.add(neighbor)
            frontier = next_frontier
        for signatures, distances in observations:
            self.assertEqual(distances, tuple(table[signature] for table, signature in zip(reference, signatures)))

    def test_random_states_have_exact_pattern_descent_and_safe_priorities(self):
        rng = random.Random(171)
        keys = [SOLVED]
        key = SOLVED
        for _ in range(30):
            for _ in range(7):
                key = moved(key, rng.choice(s.REAL_MOVES))
            keys.append(key)
            distances = self.distances(key)
            neighbors = [self.distances(moved(key, a)) for a in s.REAL_MOVES]
            for i, distance in enumerate(distances):
                self.assertLess(distance, 255)
                self.assertTrue(all(abs(n[i] - distance) <= 1 for n in neighbors))
                if distance:
                    self.assertEqual(min(n[i] for n in neighbors), distance - 1)
        objective = s.make_objective("grouped-cubie")
        recolour = str.maketrans("GRWOYB", "BYOWRG")
        for key in keys:
            self.assertEqual(objective.cost(key) == 0, s.is_solved_key(key))
            self.assertEqual(objective.cost(key), objective.cost(key.translate(recolour)))
        by_tiers = sorted(keys, key=lambda k: (sum(self.distances(k)), s.cubie_cost(k), s.misplaced_cost(k)))
        self.assertEqual([objective.cost(k) for k in by_tiers], sorted(objective.cost(k) for k in keys))

    def test_cli_selection_and_verified_search(self):
        args = s.build_arg_parser().parse_args(["--fitness", "grouped-cubie"])
        objective = s.validate_config(s.config_from_args(args))
        self.assertEqual(objective.state_function, "grouped-cubie")
        self.assertTrue(objective.integer_valued)
        self.assertIn("4096", objective.describe())
        for sequence in ((), s.parse_moves("R1"), s.parse_moves("R1 U1")):
            cube = s.replay(s.cube_from_key(SOLVED), sequence)
            result = s.run_search(s.SearchConfig(fitness="grouped-cubie", seed=2, generations=30,
                                                solution_population=20, block_population=20),
                                  cube, log=lambda message: None)
            final = s.finalize_result(result)
            self.assertEqual(final.status, s.STATUS_SOLVED)
            self.assertEqual(final.cost, 0)
            self.assertFalse(final.problems)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = s.main(["--fitness", "grouped-cubie", "--generations", "0", "--seed", "2"])
        self.assertEqual(code, 1)  # valid input, zero search budget
        self.assertIn("grouped-cubie", output.getvalue())
        self.assertIn("Replayed cost", output.getvalue())

    def test_invalid_cubies_and_inapplicable_weights_are_rejected(self):
        stickers = list(SOLVED)
        stickers[0], stickers[10] = stickers[10], stickers[0]
        with self.assertRaises(s.InvalidInputError):
            s.run_search(s.SearchConfig(fitness="grouped-cubie", generations=0),
                         s.cube_from_key("".join(stickers)), log=lambda message: None)
        with self.assertRaises(s.ConfigError):
            s.make_objective("grouped-cubie", manhattan_weight=1)

    def test_import_and_objective_selection_do_not_build_tables(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import solver_deap_coevolution as s; "
                                 "s.make_objective('grouped-cubie'); "
                                 "assert s._grouped_cubie_tables.cache_info().currsize == 0"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
