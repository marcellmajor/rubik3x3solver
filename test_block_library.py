"""Acceptance tests for the persistent move-block library (persistent_move_block_library_specs.md).

Run with:  uv run python -m unittest test_block_library -v
"""

import dataclasses
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest import mock

import solver_deap_coevolution as s

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVED = s.REFERENCE_SOLVED_KEY
LABELLED = "".join(chr(0x100 + i) for i in range(54))
_saved_environment = {}

# Frozen version-1 reference panel (section 4.1).
PANEL_CHECKSUM = '5bc6ec8cb59afdcd794c169b2d45e0780eb860095386ed99af06ca647ada6633'
REFERENCE_PANEL = (
    "OGGOGGYYYRRGRRGRRGWWGWWGWWOOOOOOOBBBBYYBYYRYYWWWRBBRBB",
    "GGGGGGRYYWWWRRRRRROOGWWGWWGYOOYOOYOOBRRBYYBYYOWWBBBBBB",
    "BBBGGGGGGRRRRRRYYYYRRYWWOWWWWWOOOOOOOOWYYWYYRBBBBBBGGG",
    "WWWGGBGGBRRRRRROOOBWWBWWGWWOOROOROORYYGYYGYYBBBGBBGYYY",
    "RYYRGGRGGRRWRRBRRBGGGWWWWWWYOOGOOGOOYYBYYBYYBOBBOBBOWW",
    "GGRGGRGGWBWWBRRBRROOOWWWWWWYYGOOGOOGRYYRYYRYYBBYBBOBBO",
    "GGOGGOBBOBRRGRRGRRWWYWWYWWYOOGOOBOOBWWWYYYYYYGGRBBRBBR",
    "YYYGGRGGRBBBRRRRRRRWWGWWGWWOOGOOGOOGYYOYYBYYBBBOBBOWWW",
    "GGYGGYOOYOWWRRRRRROOBGWWGWWYYROOBOOBGGGYYBYYBBBRBBRWWW",
    "YGGRGGRBBRRGRRGOYYOWWOWWBYYGOOGOORRRWYYWYYWWGBBBBBBOOW",
    "RGGRGGOGGOOBRRBRRBWWYWWYWWYGRRGOOGOOWWWYYYYYYRBBOBBOBB",
    "GGGGGGYYYRRRRRRRRRWWGWWGWWGOOOOOOOOOBYYBYYBYYWWWBBBBBB",
    "BBYGGOGBOBRRGRRBWWYWRYWYOOYWYGOOGOOGWWORYYRWWGGYBBRBBR",
    "RGGRGGRGGRRYRRBOOGBBBWWWWWBWOOGOOWOOYYYYYYYYGOBBRBBRWW",
    "GGOGGOGGOGRRGRRWWWWWWWWWOOBOOBOOBYYYYYYYYYGRRRRRBBBBBB",
    "GGRGGRGGOBOOBRRBRRYYYWWWWWWRRGOOGOOGWYYWYYWYYBBRBBOBBO",
    "YYYGGGOWWRRRRRROOWGWBGWBBOOYOOYOOGRRGYBGYBGWWYWWYBBRBB",
    "BRYRGYWOOGWWGROOOWRGGWWOOBBRYYGOBYBOGRRYYYWWBRWGRBGBBY",
    "WRRGGYBGYGGBRRBRRROWYOWYYWYOYBOOOOOORRWWYBWYBGGGBBBGWW",
    "GGRGGBBBOYWWORRWYYOOOWWYBRBWWYGOBYYWGOGYYWRRRRRGBBGOOB",
    "RGGRGGBBBOOGBROYOGYYOWWWYYYWWWROGBRWRWWYYYRRBOBGOBGOBR",
    "OYBGGGRRRWRYWRYWGYGOBGWWGOOWWWBOOYYYGROYYORROBBBBBBRWG",
    "GBBGGBGGBRRORROYWWYWWYWWOOROOROORYYWYYWYYWRROGGGBBGBBB",
    "YYOGGOYWWYRRBRRBRRGWBGWBGYOROOROOGGWGBBGYWOYWYYBBBOWWR",
    "YYWOGBOYGGRRORGWBWGWWWWWBRRBOOBORBYBYYOWYBOOGYGYGBGRRR",
    "YBOYGBGBWBWBRRGRRGRRRWWWYYYWOROORBOGBYWBYGOYGOGWOBGOWY",
    "RRBGGGYYGRGGRRROOOYYBWWGBOORRROOOGYOWWWBYBYBGYBBYBWWWW",
    "BWWBGBGOOGGOBRORYWWWYWWGRRBRYYOOYRROGRORYOYYYWWBGBGGBB",
    "RRORGGOBYBWWBROYOWGYBWWGGGRYRRWOBWRBGOWWYYYBGBGRYBOOYO",
    "RGORGOOGGBOBORGBBRWWGWWBBYWWROYORRORYWWYYYYRYGBGGBBYWO",
    "YYWGGRGOYOBBBROOWOROOYWGYBRYWRWOBGYBGGBOYRRYGWRWWBGBRW",
    "WWBYGGYGRWGGOROWYOROOGWWGWWBYYBOOBYGBROWYBRRORBYRBBYRG",
    "BYGBGROBBYOWGRWOYRRRWGWBGYOBORWOWYRYYYOOYOGWGBGRGBRWBW",
    "YROOGBROGYBRYRGWWOGGBWWGBGBWYRYORYRGWOBYYBYWORWORBOWBG",
    "BGRYGWGRRGWYORRBBROBYWWBWRGRYYOOOYWGBOWYYGOGOOBWGBYBRW",
    "GGOWGOBYYBYRBRWOYYWRWBWWGRBROBGOBOWORGWOYRWYYYOGGBBRRG",
    "YOOYGBOWOYGRGRROWYGRYGWBRYBGRGWOORORWRBBYYGGWWYWBBWBOB",
    "BWROGGBOGWBYGRBGYBOGOOWWYBYWWOROOGRRYYBYYRGGWRBWYBWRRO",
    "GGOYGGWYWYOYYRRGRYRRGGWBOOYRORWORRWOBYBWYBBOWBGWBBWGBO",
    "RYBRGWOGWYOWORRBGBGBYYWBOOBGOOWOBWYYGRRYYBGGWOGRWBWYRR",
    "BOWBGOGYBRYYGRWGGWORYOWYOORRRWBOWBWYRGGRYWOGWYBGBBRBYO",
    "RYOWGRWGBYOGBRGGOOYGGRWYBWGRYWROBYGBRWBOYYYRWOORBBBWWO",
    "YGGWGOWGYWOOYRBWBWBGOOWBBGGGYBYOWRWGRYROYRYWBYRORBBRRO",
    "BWBWGRWORWOWBROGYRRGOGWWBGYGBGROROBYYYOBYWBGRGYOYBRYOW",
    "OYWOGYRWOOBYRRBYRWGWBRWRRBGWBGGOGRGOWGBOYYYYBYWBOBOGWR",
    "BGYWGYGOBOWRORYOWOYOYBWYBRBRGWROWORWROGBYBRYGYGGBBGWRW",
    "ORGBGBYOOYYGBROBWYWRBWWGGGBRYBYOGWWGYYORYOWWWOBROBGRRR",
    "BGBYGRYOWRROORBYWGWGROWRWRYGYOWOBROYGBWYYWBWGBGOYBGRBO",
)

# Search fingerprints for the library-disabled search (section 10.7);
# re-recorded after candidate state-loop removal changed stored actives.
PRE_FEATURE_DIGESTS = {
    "manhattan": "7d6d332531a99cabdcb18079b4fdb6eabe4fd0cb69069f52d675e9f5e410f0b7",
    "grouped-cubie": "aa9fddf4ec607a55081dde80930d8faa2b7182d431e3f94e595d38b55f8c5732",
}


def setUpModule():
    # Never touch the real shared library, even if a test forgets a path.
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


def scrambled(moves):
    return s.replay(s.cube_from_key(SOLVED), s.parse_moves(moves))


def scrambled_default_cube():
    """A fully scrambled input, so CLI tests do not depend on the module's default cube."""
    return s.Cube("WYRYGRGRY", ["WOGBRGYOR", "RGOWWOYYG", "YWOBOBOOO", "BGBWYRBYB"], "WGWRBBGWR")


def small_config(path, **overrides):
    values = dict(seed=3, generations=3, solution_population=20, block_population=20, contexts=6,
                  solution_capacity=16, block_capacity=8, min_block_gain=1.0, log_every=10, block_library=path)
    values.update(overrides)
    return s.SearchConfig(**values)


def make_run(config, cube, *, budget=None):
    return s.CoevolutionRun(config, s.validate_input_cube(cube), s.validate_config(config), config.seed,
                            budget=budget, log=lambda message: None)


def search_digest(run, result):
    counters = {k: v for k, v in dataclasses.asdict(result.counters).items() if not k.startswith("library_")}
    payload = repr((result.status, result.reason, result.archive, [dict(r) for r in result.history],
                    sorted(counters.items()), [ind.active for ind in run.population], [b.active for b in run.blocks]))
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical(length):
    return list(s.iter_canonical_sequences(length))


def expected_top(candidates):
    """Independent top-100: score every eligible distinct transformation and sort by the spec order."""
    best = {}
    for moves in candidates:
        reduced = s.simplify(moves)
        permutation = s.block_permutation(reduced)
        if permutation == tuple(range(54)) or len(reduced) > 64:
            continue
        if permutation not in best or (len(reduced), reduced) < (len(best[permutation]), best[permutation]):
            best[permutation] = reduced
    entries = []
    for permutation, moves in best.items():
        score = s.reference_score(permutation)
        if score[1] >= 1:
            entries.append(s.LibraryEntry(permutation, moves, score, s.transformation_id(permutation)))
    entries.sort(key=s.LibraryEntry.rank_key)
    return entries[:100], len(entries)


def file_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


class LibraryTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "lib", "blocks.json")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def saved_library(self, candidates):
        library = s.BlockLibrary(self.path)
        library.submit(candidates)
        self.assertTrue(library.save())
        return library


class StableRankingTests(LibraryTestCase):
    def test_panel_matches_frozen_fixture_and_independent_regeneration(self):
        self.assertEqual(s.reference_panel(), REFERENCE_PANEL)
        self.assertEqual(hashlib.sha256("\n".join(REFERENCE_PANEL).encode()).hexdigest(), PANEL_CHECKSUM)
        self.assertEqual(len(set(REFERENCE_PANEL)), 48)
        # Regenerate through cube.py's own rotations, not the move tables.
        states = []
        for length in (2, 4, 8, 12, 20, 30):
            for sample in range(8):
                attempt = 0
                while True:
                    moves = []
                    for position in range(length):
                        text = f"rubik-block-library-v1|{length}|{sample}|{attempt}|{position}"
                        moves.append(1 + int.from_bytes(hashlib.sha256(text.encode("ascii")).digest(), "big") % 18)
                    key = s.state_key(s.replay(s.cube_from_key(SOLVED), moves))
                    if not s.is_solved_key(key) and key not in states:
                        break
                    attempt += 1
                states.append(key)
        self.assertEqual(tuple(states), REFERENCE_PANEL)

    def test_scores_and_order_are_independent_of_rng_and_objective(self):
        candidates = canonical(1) + canonical(2)[:60]
        first, _ = expected_top(candidates)
        random.seed(12345)
        state = random.getstate()
        second, _ = expected_top(list(reversed(candidates)))
        self.assertEqual(random.getstate(), state)  # scoring draws no randomness
        self.assertEqual(first, second)
        for fitness in ("manhattan", "misplaced", "grouped-cubie"):  # the run objective plays no part
            library = s.BlockLibrary(os.path.join(self.directory, fitness + ".json"))
            library.submit(candidates)
            self.assertEqual(library.ranked(), first[:len(library.ranked())])


class DistinctnessTests(LibraryTestCase):
    def test_equivalent_sequences_share_one_slot_and_identity_is_excluded(self):
        library = s.BlockLibrary(self.path)
        self.assertEqual(library.fingerprint(s.parse_moves("R1 R1"))[1], library.fingerprint(s.parse_moves("R2"))[1])
        self.assertEqual(library.fingerprint(s.parse_moves("R1 L1"))[1], library.fingerprint(s.parse_moves("L1 R1"))[1])
        longer = s.parse_moves("U2 R2 U2 R2 U2 R2")
        smaller = s.parse_moves("R2 U2 R2 U2 R2 U2")
        self.assertNotEqual(s.simplify(longer), s.simplify(smaller))
        self.assertEqual(s.block_permutation(longer), s.block_permutation(smaller))
        library.submit([longer, s.parse_moves("R2 U2 " * 6), (s.NOOP,), s.parse_moves("R1 R3")])
        library.submit([smaller])
        self.assertEqual([entry.moves for entry in library.entries.values()], [smaller])
        self.assertEqual(library.counters.library_representative_updates, 1)
        self.assertEqual(library.counters.library_rejected_candidates, 3)  # identity cycle, NOOP, R1 R3
        entry = next(iter(library.entries.values()))
        self.assertEqual(entry.score, s.reference_score(s.block_permutation(longer)))

    def test_distinct_permutations_with_equal_scores_stay_separate(self):
        # Real ties are rare on the 48-state panel, so pin every score to one value.
        moves = canonical(1)[:6]
        library = s.BlockLibrary(self.path)
        with mock.patch.object(library, "_score", return_value=(0, 3, 12345)):
            library.submit(moves)
        self.assertEqual(len(library.entries), 6)
        self.assertEqual([entry.moves for entry in library.ranked()], sorted(moves))  # ties fall to allele order

    def test_permutation_convention_matches_cube_py(self):
        rng = random.Random(4)
        for _ in range(50):
            moves = s.generate_reduced_block(rng.randint(1, 20), rng)
            permutation = s.block_permutation(moves)
            moved = s.state_key(s.replay(s.cube_from_key(LABELLED), moves))
            self.assertEqual(moved, "".join(LABELLED[p] for p in permutation))


class PrefixTests(LibraryTestCase):
    def test_full_block_never_inherits_a_prefix_score(self):
        block = s.parse_moves("C1 U1 B3 U3 R2 D1")
        prefix = block[:4]
        library = s.BlockLibrary(self.path)
        library.submit([block])
        full_entry = library.entries[s.block_permutation(block)]
        self.assertEqual(full_entry.score, s.reference_score(s.block_permutation(block)))
        self.assertNotEqual(full_entry.score, s.reference_score(s.block_permutation(prefix)))
        library.submit([prefix])
        self.assertIn(s.block_permutation(prefix), library.entries)
        self.assertEqual(library.entries[s.block_permutation(block)], full_entry)


class TopHundredTests(LibraryTestCase):
    def test_file_holds_exactly_the_reference_top_100(self):
        candidates = canonical(1) + canonical(2)
        expected, eligible = expected_top(candidates)
        self.assertGreaterEqual(eligible, 120)
        self.saved_library(candidates)
        document = json.loads(file_bytes(self.path))
        self.assertEqual(document["entries"], [entry.record() for entry in expected])
        before = file_bytes(self.path)
        again = s.BlockLibrary(self.path)
        again.submit(candidates)
        again.submit(candidates)  # repeated submissions earn nothing
        self.assertFalse(again.save())
        self.assertEqual(file_bytes(self.path), before)

    def test_fewer_eligible_candidates_give_fewer_entries_without_filler(self):
        candidates = canonical(1)[:5] + canonical(1)[:5]
        expected, eligible = expected_top(candidates)
        self.saved_library(candidates)
        records = json.loads(file_bytes(self.path))["entries"]
        self.assertEqual(len(records), eligible)
        self.assertLess(len(records), 100)
        self.assertEqual(len({record["transformation_id"] for record in records}), len(records))


class PopulationTests(LibraryTestCase):
    def test_seeding_rules(self):
        self.saved_library(canonical(1) + canonical(2) + canonical(3)[:400])
        stored = json.loads(file_bytes(self.path))["entries"]
        long_entries = [record for record in stored if len(record["moves"]) > 2]
        self.assertTrue(long_entries)
        config = small_config(self.path, block_population=20, block_capacity=2, solution_capacity=8, generations=1)
        run = make_run(config, scrambled("R2 B2 D1 R2 U1 C2 D2 C1"))
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        seeded = [block for block in run.blocks if block.origin == "library"]
        self.assertEqual(len(seeded), 5)  # floor(20 / 4)
        self.assertTrue(all(len(block.active) <= 2 and block.library_id for block in seeded))
        self.assertEqual([block.library_id for block in seeded],
                         [entry.transformation_id for entry in run.library.ranked()
                          if len(entry.moves) <= 2][:5])
        self.assertEqual(len(run.blocks), 20)
        self.assertLessEqual(max(Counter(block.active for block in run.blocks).values()), run.max_block_copies)
        self.assertTrue(all(list(ind) == [s.NOOP] * 8 for ind in run.population))
        run.budget.start()
        run.run_generation(1, build_next_blocks=False)
        for block in seeded:  # fresh runtime fitness on the current batch, never the reference score
            self.assertEqual(block.batch_serial, run.batch.serial)
            self.assertEqual(block.fitness.values, tuple(float(v) for v in block.evaluation.score))
        run.finish_block_library()
        records = json.loads(file_bytes(self.path))["entries"]
        self.assertTrue(all(record in records for record in long_entries if record in stored[:100]))
        self.assertTrue(any(len(record["moves"]) > 2 for record in records))  # never truncated or deleted


class CrossCubeTests(LibraryTestCase):
    def test_block_saved_on_cube_a_is_reused_on_cube_b_with_another_objective(self):
        result_a = s.run_search(small_config(self.path, generations=2), scrambled("R2 B2 D1 R2 U1 C2 D2 C1 U3 L1"),
                                log=lambda message: None)
        self.assertGreater(result_a.library_report["entries"], 0)
        saved = {record["transformation_id"] for record in json.loads(file_bytes(self.path))["entries"]}
        config_b = small_config(self.path, fitness="misplaced", seed=9, generations=1)
        run = make_run(config_b, scrambled("C1 U2 L3 B1 R2 D3 C2 U1"))
        self.assertEqual(run.library_status, "enabled")
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        seeded = [block for block in run.blocks if block.origin == "library"]
        self.assertTrue(seeded)
        self.assertTrue({block.library_id for block in seeded} <= saved)
        self.assertTrue(all(block.evaluation is None for block in seeded))  # no persisted score copied in
        run.budget.start()
        run.run_generation(1, build_next_blocks=False)
        self.assertTrue(all(block.evaluation is not None for block in seeded))


class DisabledTests(LibraryTestCase):
    def test_disabled_search_matches_the_pre_feature_fingerprints(self):
        cube = scrambled("R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        for fitness, digest in PRE_FEATURE_DIGESTS.items():
            config = s.SearchConfig(fitness=fitness, seed=5, generations=6, solution_population=20,
                                    block_population=20, solution_capacity=16, block_capacity=8, contexts=6,
                                    min_block_gain=1.0, log_every=10)
            run = make_run(config, cube)
            result = run.run()
            self.assertEqual(search_digest(run, result), digest)
            library_counters = {k: v for k, v in dataclasses.asdict(result.counters).items() if k.startswith("library_")}
            self.assertTrue(all(value == 0 for value in library_counters.values()))

    def test_enabled_with_an_empty_library_runs_the_same_search(self):
        cube = scrambled("R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        config = s.SearchConfig(fitness="manhattan", seed=5, generations=6, solution_population=20,
                                block_population=20, solution_capacity=16, block_capacity=8, contexts=6,
                                min_block_gain=1.0, log_every=10, block_library=self.path)
        run = make_run(config, cube)
        self.assertEqual(search_digest(run, run.run()), PRE_FEATURE_DIGESTS["manhattan"])

    @mock.patch.object(s, "default_input_cube", scrambled_default_cube)
    def test_cli_flags(self):
        args = ["--seed", "7", "--generations", "2", "--solution-population", "20", "--block-population", "20",
                "--contexts", "6", "--solution-capacity", "16", "--block-capacity", "8"]
        xdg = os.environ["XDG_DATA_HOME"]
        default = os.path.join(xdg, "rubik3x3solver", "move_blocks_v1.json")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(s.main(args + ["--no-block-library"]), 1)
        self.assertFalse(os.path.exists(os.path.join(xdg, "rubik3x3solver")))
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(s.main(args + ["--no-block-library", "--block-library", "x.json"]), 2)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(s.main(args), 1)
        self.assertTrue(os.path.exists(default))
        self.assertIn("Block library: " + default + " (enabled)", output.getvalue())
        previous = os.getcwd()
        try:
            os.chdir(self.directory)
            with redirect_stdout(io.StringIO()):
                s.main(args + ["--block-library", "relative/blocks.json"])
        finally:
            os.chdir(previous)
        self.assertTrue(os.path.exists(os.path.join(self.directory, "relative", "blocks.json")))
        completed = subprocess.run([sys.executable, "solver_deap_coevolution.py", "--help"], cwd=REPO_DIR,
                                   capture_output=True, text=True, env={**os.environ, "XDG_DATA_HOME": self.directory + "/h"})
        self.assertEqual(completed.returncode, 0)
        self.assertFalse(os.path.exists(self.directory + "/h"))


class StoredDataSafetyTests(LibraryTestCase):
    def valid_document(self):
        self.saved_library(canonical(1))
        return json.loads(file_bytes(self.path))

    def write(self, document, raw=None):
        with open(self.path, "wb") as handle:
            handle.write(raw if raw is not None else json.dumps(document).encode())

    def assert_preserved_and_unused(self):
        before = file_bytes(self.path)
        result = s.run_search(small_config(self.path, generations=2), scrambled("R1"), log=lambda message: None)
        self.assertEqual(result.status, s.STATUS_SOLVED)
        self.assertEqual(result.library_report["status"], "unusable")
        self.assertTrue(result.library_report["diagnostics"])
        self.assertEqual(file_bytes(self.path), before)

    def test_bad_files_never_run_and_are_never_overwritten(self):
        document = self.valid_document()

        def mutated(change, fix_revision=True):
            copy = json.loads(json.dumps(document))
            change(copy)
            if fix_revision:
                copy["revision"] = s._canonical_digest(copy["entries"])
            return copy

        cases = {
            "malformed": (None, b"{not json"),
            "oversized": (None, b" " * (s.BLOCK_LIBRARY_MAX_BYTES + 1)),
            "empty file": (None, b""),
            "NaN": (None, json.dumps(document).replace('"gain_ppm_sum": %d' % document["entries"][0]["reference_score"]["gain_ppm_sum"], '"gain_ppm_sum": NaN', 1).encode()),
            "schema": (mutated(lambda d: d.update(schema_version=2)), None),
            "convention": (mutated(lambda d: d.update(move_convention_id="0" * 64)), None),
            "evaluator": (mutated(lambda d: d.update(evaluator_version="other")), None),
            "token": (mutated(lambda d: d["entries"][0].update(moves=["X9"])), None),
            "fingerprint": (mutated(lambda d: d["entries"][0].update(transformation_id="0" * 64)), None),
            "bijection": (mutated(lambda d: d["entries"][0]["permutation"].__setitem__(0, 1)), None),
            "boolean": (mutated(lambda d: d["entries"][0]["reference_score"].update(solved_count=True)), None),
            "score": (mutated(lambda d: d["entries"][0]["reference_score"].update(gain_ppm_sum=1)), None),
            "revision": (mutated(lambda d: d["entries"].pop(), fix_revision=False), None),
            "code-like token": (mutated(lambda d: d["entries"][0].update(moves=["__import__('os').system('x')"])), None),
        }
        for name, (bad, raw) in cases.items():
            with self.subTest(case=name):
                self.write(bad, raw)
                self.assert_preserved_and_unused()
        self.write(document)  # the untouched original still loads
        library = s.BlockLibrary(self.path)
        self.assertTrue(library.load())


class LifecycleTests(LibraryTestCase):
    def test_failed_load_disables_all_later_library_work(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "wb") as handle:
            handle.write(b"not valid JSON")
        before = file_bytes(self.path)
        messages = []
        config = small_config(self.path, generations=3)
        cube = scrambled("R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        run = s.CoevolutionRun(config, s.validate_input_cube(cube), s.validate_config(config),
                               config.seed, log=messages.append)
        self.assertFalse(run.library.usable)
        with mock.patch.object(s, "reference_score", side_effect=AssertionError("unusable library scored")), \
                mock.patch.object(s, "_atomic_write", side_effect=AssertionError("unusable library written")):
            run._library_note_payload(s.parse_moves("R1"), (1, 1, 1))
            run._library_checkpoint(include_snapshot=True)
            run.library.submit([s.parse_moves("R1")])
            result = run.run()
            run.finish_block_library()
        self.assertFalse(run._library_pending)
        self.assertIsNone(run._library_snapshot)
        for name in ("library_candidates_submitted", "library_pending_payloads", "library_checkpoints",
                     "library_reference_evaluations", "library_additions", "library_saves"):
            self.assertEqual(getattr(run.counters, name), 0, name)
        self.assertFalse(any("Block library checkpoint" in message for message in messages))
        self.assertEqual(file_bytes(self.path), before)
        baseline = s.run_search(small_config(None, generations=3), cube, log=lambda message: None)
        self.assertEqual(result.archive, baseline.archive)

    def test_final_save_defers_uncached_concurrent_entries_after_stop(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                path = os.path.join(self.directory, f"concurrent-{interrupted}.json")
                now = [0.0]
                budget = s.Budget(1.0, clock=lambda: now[0])
                run = make_run(small_config(path), scrambled("R2 B2 D1 R2 U1"), budget=budget)
                run.library.submit([s.parse_moves("R1")])
                other = s.BlockLibrary(path)
                other.submit([s.parse_moves("U1")])
                self.assertTrue(other.save())
                before = file_bytes(path)
                local_entries = dict(run.library.entries)
                budget.start()
                if interrupted:
                    budget.request_interrupt()
                else:
                    now[0] = 2.0
                evaluations = run.counters.library_reference_evaluations
                with mock.patch.object(s, "reference_score", side_effect=AssertionError("scored after stop")):
                    report = run.finish_block_library()
                self.assertEqual(file_bytes(path), before)
                self.assertEqual(run.library.entries, local_entries)
                self.assertEqual(run.counters.library_reference_evaluations, evaluations)
                self.assertTrue(any("save deferred" in message for message in report["diagnostics"]))
                self.assertTrue(run.library.writable)

    def test_stopped_save_can_merge_cached_scores_without_evaluation(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                path = os.path.join(self.directory, f"cached-{interrupted}.json")
                now = [0.0]
                budget = s.Budget(1.0, clock=lambda: now[0])
                run = make_run(small_config(path), scrambled("R2 B2 D1 R2 U1"), budget=budget)
                run.library.submit([s.parse_moves("R1")])
                self.assertTrue(run.library.save())
                run.library.submit([s.parse_moves("U1")])
                budget.start()
                if interrupted:
                    budget.request_interrupt()
                else:
                    now[0] = 2.0
                with mock.patch.object(s, "reference_score", side_effect=AssertionError("cached score recomputed")):
                    run.finish_block_library()
                self.assertEqual(len(json.loads(file_bytes(path))["entries"]), 2)

    def test_missing_file_is_created_only_with_eligible_entries(self):
        library = s.BlockLibrary(self.path)
        self.assertTrue(library.load())
        self.assertFalse(library.save())
        self.assertFalse(os.path.exists(os.path.dirname(self.path)))
        library.submit([s.parse_moves("R1")])
        self.assertTrue(library.save())
        self.assertTrue(os.path.exists(self.path))

    def test_solved_input_and_zero_generations_leave_the_library_untouched(self):
        self.saved_library(canonical(1))
        before = file_bytes(self.path)
        modified = os.stat(self.path).st_mtime_ns
        solved = s.run_search(small_config(self.path), s.cube_from_key(SOLVED), log=lambda message: None)
        self.assertEqual(solved.library_report["status"], "not needed")
        zero = s.run_search(small_config(self.path, generations=0), scrambled("R1 U1"), log=lambda message: None)
        self.assertEqual(zero.library_report["status"], "not needed")
        self.assertEqual((file_bytes(self.path), os.stat(self.path).st_mtime_ns), (before, modified))
        other = os.path.join(self.directory, "never.json")
        s.run_search(small_config(other, generations=0), scrambled("R1 U1"), log=lambda message: None)
        self.assertFalse(os.path.exists(other))

    def test_budget_expiry_mid_evaluation_keeps_only_complete_scores(self):
        calls = {"n": 0}

        def check():
            calls["n"] += 1
            if calls["n"] > 48 + 20:  # first candidate completes, second stops mid-panel
                raise s.BudgetExpired("time limit reached")

        library = s.BlockLibrary(self.path)
        with self.assertRaises(s.BudgetExpired):
            library.submit([s.parse_moves("C1"), s.parse_moves("D1")], check=check)
        self.assertEqual(list(library.entries), [s.block_permutation(s.parse_moves("C1"))])
        self.assertEqual(len(library._scores), 1)
        self.assertTrue(library.save())
        self.assertEqual(len(json.loads(file_bytes(self.path))["entries"]), 1)

    def test_clean_interrupt_saves_completed_work_without_new_evaluation(self):
        holder = {}
        budget = s.Budget()
        config = small_config(self.path, generations=5)
        run = s.CoevolutionRun(config, s.validate_input_cube(scrambled("R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")),
                               s.validate_config(config), 3, budget=budget, log=lambda message: None)
        original = run._library_checkpoint

        def checkpoint_then_interrupt(include_snapshot):
            original(include_snapshot)
            holder["evaluations"] = run.counters.library_reference_evaluations
            budget.request_interrupt()

        run._library_checkpoint = checkpoint_then_interrupt
        result = run.run()
        self.assertEqual(result.status, s.STATUS_INTERRUPTED)
        self.assertEqual(run.counters.library_reference_evaluations, holder["evaluations"])
        run.finish_block_library()
        self.assertEqual(len(json.loads(file_bytes(self.path))["entries"]), len(run.library.entries))

    def test_consistency_failure_persists_nothing_new(self):
        config = small_config(self.path, generations=2)
        with mock.patch.object(s.CoevolutionRun, "_index_blocks", side_effect=s.ConsistencyError("injected")):
            with self.assertRaises(s.ConsistencyError):
                s.run_search(config, scrambled("R2 B2 D1 R2 U1"), log=lambda message: None)
        self.assertFalse(os.path.exists(self.path))
        with mock.patch.object(s, "finalize_result",
                               return_value=s.FinalReport(s.STATUS_ERROR, (), None, None, None, ["injected"])):
            s.run_search(small_config(self.path, generations=0 + 1), scrambled("R2 B2 D1 R2 U1"),
                         log=lambda message: None)
        checkpoint_only = json.loads(file_bytes(self.path))["entries"]
        result = s.run_search(small_config(self.path, generations=1), scrambled("R2 B2 D1 R2 U1"),
                              log=lambda message: None)
        self.assertGreaterEqual(result.library_report["entries"], len(checkpoint_only))

    def test_library_errors_never_turn_a_solve_into_an_error(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as handle:
            handle.write("garbage")
        result = s.run_search(small_config(self.path, generations=10), scrambled("R1 U1"), log=lambda message: None)
        self.assertEqual(s.finalize_result(result).status, s.STATUS_SOLVED)


class ConcurrencyTests(LibraryTestCase):
    def test_budget_expiring_during_merge_preserves_disk_and_local_archive(self):
        self.saved_library([s.parse_moves("R1"), s.parse_moves("U1")])
        before = file_bytes(self.path)
        library = s.BlockLibrary(self.path)
        library.submit([s.parse_moves("C1")])
        local = dict(library.entries)
        checks = 0

        def check():
            nonlocal checks
            checks += 1
            if checks > 10:
                raise s.BudgetExpired("time limit reached")

        self.assertFalse(library.save(check=check))
        self.assertEqual(file_bytes(self.path), before)
        self.assertEqual(library.entries, local)
        self.assertTrue(library.writable)
        self.assertIn("save deferred", library.diagnostics[0])
        # A later explicit retry with time available still merges both writers.
        self.assertTrue(library.save())
        self.assertEqual(len(json.loads(file_bytes(self.path))["entries"]), 3)

    def test_concurrent_writers_keep_the_top_100_of_both_archives(self):
        candidates = canonical(1) + canonical(2)
        halves = [candidates[0::2], candidates[1::2]]
        script = ("import json, sys; sys.path.insert(0, %r); import solver_deap_coevolution as s\n"
                  "path, half = sys.argv[1], json.loads(sys.argv[2])\n"
                  "library = s.BlockLibrary(path)\n"
                  "for start in range(0, len(half), 20):\n"
                  "    library.submit([tuple(m) for m in half[start:start + 20]])\n"
                  "    library.save()\n"
                  "assert not library.diagnostics, library.diagnostics\n") % REPO_DIR
        os.makedirs(os.path.dirname(self.path))
        processes = [subprocess.Popen([sys.executable, "-c", script, self.path, json.dumps(half)],
                                      stderr=subprocess.PIPE, text=True) for half in halves]
        for process in processes:
            _, error = process.communicate(timeout=120)
            self.assertEqual(process.returncode, 0, error)
        expected, _ = expected_top(candidates)
        self.assertEqual(json.loads(file_bytes(self.path))["entries"], [entry.record() for entry in expected])

    def test_failed_write_preserves_the_old_file_and_cleans_up(self):
        self.saved_library(canonical(1)[:3])
        before = file_bytes(self.path)
        library = s.BlockLibrary(self.path)
        library.submit(canonical(2)[:30])
        with mock.patch("os.replace", side_effect=OSError(28, "No space left on device")):
            self.assertFalse(library.save())
        self.assertEqual(file_bytes(self.path), before)
        self.assertEqual(sorted(os.listdir(os.path.dirname(self.path))), ["blocks.json", "blocks.json.lock"])
        self.assertIn("No space left on device", library.diagnostics[0])

    def test_lock_timeout_preserves_file_and_solver_result(self):
        import fcntl
        self.saved_library(canonical(1)[:3])
        before = file_bytes(self.path)
        with open(self.path + ".lock", "w") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            library = s.BlockLibrary(self.path, lock_timeout=0.2)
            library.submit(canonical(2)[:10])
            started = time.monotonic()
            self.assertFalse(library.save())
            self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("timed out", library.diagnostics[0])
        self.assertEqual(file_bytes(self.path), before)


class BoundedResourceTests(LibraryTestCase):
    def test_caches_archive_and_pending_queue_stay_bounded(self):
        library = s.BlockLibrary(self.path)
        library.submit(canonical(2) + canonical(3) + canonical(4)[:2000])
        self.assertLessEqual(len(library._scores), s.BLOCK_LIBRARY_CACHE_LIMIT)
        self.assertLessEqual(len(library._fingerprints), s.BLOCK_LIBRARY_CACHE_LIMIT)
        self.assertLessEqual(len(library.entries), 100)
        run = make_run(small_config(self.path), scrambled("R2 B2 D1 R2 U1"))
        for index, moves in enumerate(canonical(2)[:150]):
            run._library_note_payload(moves, (1, index, 3))
        self.assertEqual(len(run._library_pending), s.BLOCK_LIBRARY_PENDING_LIMIT)
        self.assertTrue(run._library_flush_requested)
        kept = run._drain_library_pending()
        self.assertEqual(kept, canonical(2)[:100])  # best resulting solution ranks win

    def test_state_fitness_never_touches_the_library(self):
        reads, scores = Counter(), Counter()
        real_read, real_score = s._read_library_file, s.reference_score

        def counted_read(path):
            reads["n"] += 1
            return real_read(path)

        def counted_score(permutation, check=None):
            scores["n"] += 1
            return real_score(permutation, check)

        with mock.patch.object(s, "_read_library_file", counted_read), mock.patch.object(s, "reference_score", counted_score):
            result = s.run_search(small_config(self.path, generations=4), scrambled("R2 B2 D1 R2 U1 C2 D2 C1"),
                                  log=lambda message: None)
        counters = result.counters
        self.assertEqual(scores["n"], counters.library_reference_evaluations)
        self.assertEqual(reads["n"], 1 + counters.library_checkpoints + 1)  # load, checkpoint saves, final save
        self.assertGreater(counters.cost_computations, 20 * counters.library_reference_evaluations)


class ReplayTests(LibraryTestCase):
    def test_saved_sequences_replay_through_cube_py(self):
        s.run_search(small_config(self.path, generations=2), scrambled("R2 B2 D1 R2 U1 C2 D2 C1"), log=lambda m: None)
        for record in json.loads(file_bytes(self.path))["entries"]:
            moves = tuple(s.ALLELE_BY_NAME[token] for token in record["moves"])
            moved = s.state_key(s.replay(s.cube_from_key(LABELLED), moves))
            self.assertEqual(moved, "".join(LABELLED[p] for p in record["permutation"]))
            self.assertEqual(record["transformation_id"], s.transformation_id(tuple(record["permutation"])))


if __name__ == "__main__":
    unittest.main()
