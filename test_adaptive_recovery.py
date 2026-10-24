"""Tests for the optional adaptive population recovery (--adaptive-recovery) in solver_deap_coevolution.py."""

import contextlib
import dataclasses
import hashlib
import io
import os
import random
import shutil
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import solver_deap_coevolution as s
from test_block_library import PRE_FEATURE_DIGESTS, search_digest

SOLVED = s.REFERENCE_SOLVED_KEY
# A near-solved cube (four edges wrong) whose small-population search stays on the empty
# baseline: every candidate shares one endpoint, so the health checks flag low diversity.
PLATEAU = s.validate_input_cube(s.Cube("GGGYGWGGG", ["RORRRRRRR", "WGWWWWWWW", "OROOOOOOO", "YGYYYYYYY"],
                                       "BBBBBBBBB"))
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


def fake_blocks(actives, useful, serial=7):
    return [SimpleNamespace(active=tuple(active), batch_serial=serial,
                            evaluation=SimpleNamespace(useful=useful, improving_contexts=int(useful)))
            for active in actives]


def timeless_digest(run, result):
    """``search_digest`` without measured durations, which differ between identical runs."""
    history = []
    for row in result.history:
        row = dict(row)
        if "diversity" in row:
            row["diversity"] = {k: v for k, v in row["diversity"].items() if k != "seconds"}
        history.append(row)
    counters = {k: v for k, v in dataclasses.asdict(result.counters).items() if not k.startswith("library_")}
    payload = repr((result.status, result.reason, result.archive, history, sorted(counters.items()),
                    [ind.active for ind in run.population], [b.active for b in run.blocks]))
    return hashlib.sha256(payload.encode()).hexdigest()


UNHEALTHY = fake_blocks([(s.parse_moves("R1")[0],)] * 10, useful=False)
HEALTHY = fake_blocks([(allele,) for allele in s.REAL_MOVES[:10]], useful=True)


class RecoveryTestCase(unittest.TestCase):
    def setUp(self):
        s._SOLVED_SIDE_TABLES.clear()
        self.directory = tempfile.mkdtemp()
        self.messages = []

    def tearDown(self):
        s._SOLVED_SIDE_TABLES.clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    def make_run(self, key=PLATEAU, **overrides):
        values = dict(fitness="manhattan", seed=3, generations=1000, solution_population=20, block_population=20,
                      contexts=6, solution_capacity=16, block_capacity=8, min_block_gain=1.0, log_every=10 ** 6,
                      block_library=None, adaptive_recovery=True)
        values.update(overrides)
        config = s.SearchConfig(**values)
        return s.CoevolutionRun(config, key, s.validate_config(config), config.seed, log=self.messages.append)

    def advanced(self, generations, key=PLATEAU, **overrides):
        run = self.make_run(key, **overrides)
        run.budget.start()
        run.population = run.initial_solution_population()
        run.blocks = run.initial_block_population()
        for generation in range(1, generations + 1):
            run.run_generation(generation)
        return run

    def record(self, run, generation, flags, available=True, stalled=None):
        size = run.config.solution_population
        population = run.population or [SimpleNamespace(endpoint=PLATEAU, slot="reset" if index < size // 5
                                                         else "offspring") for index in range(size)]
        record = s.measure_population_diversity(population, size)
        record.update(generation=generation, flags=list(flags), messages=[],
                      generations_without_quality_improvement=generation if stalled is None else stalled)
        if not available:
            record.update(available=False, reason="fixture: unavailable")
        return record

    def checkpoint(self, run, generation, flags, blocks=None, finisher=None, **options):
        if blocks is not None:
            run._evaluated_blocks = (blocks[0].batch_serial, generation, blocks)
        run.recovery_checkpoint(generation, self.record(run, generation, flags, **options), finisher)

    def qualify(self, run, *generations, blocks=None):
        for generation in generations:
            self.checkpoint(run, generation, ["LOW_UNIQUE_STATES", "NO_QUALITY_PROGRESS"], blocks=blocks)

    def applied(self, run, generation):
        """Stand in for the generation that applies the pending intervention."""
        self.assertIsNotNone(run._recovery_pending)
        run._recovery_pending = None
        run._recovery_last_generation = generation

    def recovery_lines(self):
        return [line for line in self.messages if "RECOVERY" in line]


class TriggerTests(RecoveryTestCase):
    def test_disabled_mode_is_the_unchanged_search(self):
        key = after(SOLVED, "R2 B2 D1 R2 U1 C2 D2 C1 U3 L1")
        config = s.SearchConfig(fitness="manhattan", seed=5, generations=6, solution_population=20,
                                block_population=20, solution_capacity=16, block_capacity=8, contexts=6,
                                min_block_gain=1.0, log_every=10)
        run = s.CoevolutionRun(config, key, s.validate_config(config), 5, log=lambda message: None)
        result = run.run()
        self.assertEqual(search_digest(run, result), PRE_FEATURE_DIGESTS["manhattan"])
        self.assertEqual((result.recovery_report["enabled"], result.recovery_report["interventions"]), (False, 0))
        run = self.make_run(adaptive_recovery=False, generations=160)
        result = run.run()
        self.assertEqual(run.recovery_records, [])
        self.assertEqual(self.recovery_lines(), [])
        self.assertFalse(any("block-population health" in line for line in self.messages))
        self.assertTrue(any("NO_QUALITY_PROGRESS" in line for line in self.messages))  # it would have qualified

    def test_low_diversity_alone_or_stagnation_alone_never_qualifies(self):
        run = self.make_run()
        for generation, flags in enumerate((["LOW_UNIQUE_STATES", "DOMINANT_STATE", "LOW_CUBIE_SPREAD"],
                                            ["NO_QUALITY_PROGRESS"], ["DOMINANT_STATE"], ["NO_QUALITY_PROGRESS"]), 1):
            self.checkpoint(run, 50 * generation, flags)
            self.assertEqual(run._recovery_solution_streak, 0)
        self.assertIsNone(run._recovery_pending)

    def test_consecutive_checks_and_streak_reset(self):
        run = self.make_run(recovery_consecutive_checks=3)
        self.qualify(run, 100, 150)
        self.assertEqual(run._recovery_solution_streak, 2)
        self.checkpoint(run, 200, ["NO_QUALITY_PROGRESS"])  # combined condition absent
        self.assertEqual(run._recovery_solution_streak, 0)
        self.qualify(run, 250, 300)
        self.assertIsNone(run._recovery_pending)
        self.qualify(run, 350)
        pending = run._recovery_pending
        self.assertEqual((pending["scheduled_generation"], pending["apply_generation"], pending["streak"]),
                         (350, 351, 3))
        self.assertEqual(pending["triggers"], ["NO_QUALITY_PROGRESS", "LOW_UNIQUE_STATES"])
        self.assertEqual(run._recovery_solution_streak, 0)  # the evidence is used up

    def test_unavailable_measurements_break_the_streak(self):
        run = self.make_run()
        self.qualify(run, 100)
        self.checkpoint(run, 150, ["LOW_UNIQUE_STATES", "NO_QUALITY_PROGRESS"], available=False)
        self.assertEqual(run._recovery_solution_streak, 0)
        self.qualify(run, 200)
        self.assertIsNone(run._recovery_pending)

    def test_cooldown_is_enforced_from_the_applied_generation(self):
        run = self.make_run(recovery_cooldown_generations=150)
        self.qualify(run, 100, 150)
        self.applied(run, 151)
        self.qualify(run, 200, 250)
        self.assertIsNone(run._recovery_pending)
        self.assertTrue(any("cooling down" in line for line in self.messages))
        self.qualify(run, 300)  # 301 - 151 = 150 completed generations
        self.assertIsNotNone(run._recovery_pending)

    def test_no_intervention_without_a_generation_left(self):
        run = self.make_run(generations=150)
        self.qualify(run, 100, 150)
        self.assertIsNone(run._recovery_pending)

    def test_no_recovery_after_the_budget_expired(self):
        run = self.make_run()
        run.budget.start()
        self.qualify(run, 100)
        run.budget.request_interrupt()
        with self.assertRaises(s.SearchInterrupted):
            self.qualify(run, 150)
        self.assertIsNone(run._recovery_pending)


class FinisherIntegrationTests(RecoveryTestCase):
    def test_a_finisher_improvement_postpones_recovery(self):
        run = self.make_run(mitm=True, mitm_cache=None)
        self.qualify(run, 100)
        improved = {"result": "improved", "initial_cost": 36, "result_cost": 30}
        self.checkpoint(run, 150, ["LOW_UNIQUE_STATES", "NO_QUALITY_PROGRESS"], finisher=improved)
        self.assertIsNone(run._recovery_pending)
        self.assertEqual((run.recovery_postponements, run._recovery_solution_streak), (1, 0))
        self.assertTrue(any("RECOVERY postponed" in line for line in self.messages))

    def test_an_unproductive_attempt_or_skip_lets_recovery_proceed(self):
        run = self.make_run(mitm=True, mitm_cache=None)
        self.qualify(run, 100)
        attempt = {"result": "no feasible improving fallback"}
        self.checkpoint(run, 150, ["LOW_UNIQUE_STATES", "NO_QUALITY_PROGRESS"], finisher=attempt)
        self.assertEqual(run._recovery_pending["finisher"], "attempted: no feasible improving fallback")
        run = self.make_run()
        self.qualify(run, 100, 150)
        self.assertEqual(run._recovery_pending["finisher"], "finisher disabled")

    def test_real_finisher_runs_first_and_its_own_cache_is_respected(self):
        run = self.make_run(generations=160, mitm=True, mitm_depth=4, mitm_trigger_cubies=20, mitm_interval=1,
                            mitm_cache=None)
        result = run.run()
        attempts = result.finisher_report["attempts"]
        self.assertEqual([attempt["generation"] for attempt in attempts], [0])  # one baseline endpoint
        record = run.recovery_records[0]
        self.assertTrue(record["finisher"].startswith("not attempted (endpoint already searched"), record["finisher"])

    def test_a_finisher_solve_ends_the_run_normally(self):
        run = self.make_run(key=after(SOLVED, "R1 U1 C1"), generations=5, mitm=True, mitm_depth=4,
                            mitm_trigger_cubies=20, mitm_cache=None)
        result = run.run()
        self.assertEqual(result.status, s.STATUS_SOLVED)
        self.assertEqual(run.recovery_records, [])


class EscalationTests(RecoveryTestCase):
    def first_intervention(self, run, blocks=UNHEALTHY):
        self.qualify(run, 100, 150, blocks=blocks)
        self.assertEqual(run._recovery_pending["type"], "solution-only")  # never both on the first one
        self.assertEqual(run._recovery_pending["block_note"], "not considered (first intervention)")
        self.applied(run, 151)

    def test_escalation_needs_no_quality_improvement_and_two_unhealthy_block_checks(self):
        run = self.make_run()
        self.first_intervention(run)
        self.qualify(run, 200, 250, blocks=UNHEALTHY)
        pending = run._recovery_pending
        self.assertEqual(pending["type"], "solution plus partial block refresh")
        self.assertEqual(pending["block_refresh_limit"], 5)  # floor(20 * 0.25)
        self.assertEqual(run._recovery_block_streak, 4)  # the solution-only intervention left it unused

    def test_block_refresh_requires_its_own_evidence(self):
        cases = ((HEALTHY, "skipped (block health acceptable)"),
                 (None, "skipped (block health unavailable"))
        for blocks, note in cases:
            run = self.make_run()
            self.first_intervention(run, blocks=HEALTHY)
            self.qualify(run, 200, 250, blocks=blocks)
            self.assertEqual(run._recovery_pending["type"], "solution-only")
            self.assertTrue(run._recovery_pending["block_note"].startswith(note), run._recovery_pending["block_note"])
        run = self.make_run()
        self.first_intervention(run, blocks=HEALTHY)
        self.qualify(run, 200, blocks=HEALTHY)
        self.qualify(run, 250, blocks=UNHEALTHY)  # one unhealthy check is not enough
        self.assertIn("1 of 2 required consecutive checks", run._recovery_pending["block_note"])
        run = self.make_run(recovery_block_refresh_fraction=0.0)
        self.first_intervention(run)
        self.qualify(run, 200, 250, blocks=UNHEALTHY)
        self.assertIn("allows 0 blocks", run._recovery_pending["block_note"])

    def test_genuine_quality_improvement_resets_escalation(self):
        run = self.make_run()
        self.first_intervention(run)
        archive = run.archive
        run.archive = s.ArchiveEntry(archive.active, archive.endpoint, archive.cost - 1, False, None, 160, "fixture")
        self.qualify(run, 200, 250, blocks=UNHEALTHY)
        self.assertEqual(run._recovery_pending["type"], "solution-only")
        self.assertIn("cube quality improved", run._recovery_pending["block_note"])

    def test_length_only_improvement_resets_neither_stagnation_nor_escalation(self):
        key = after(SOLVED, "R3 U3 C2 L1")
        run = self.make_run(key=key)
        candidate = run.normalize_solution(s.parse_moves("L3 C2"), check_budget=False)
        self.assertEqual(candidate.active, s.parse_moves("L3 C2"))
        run.archive = s.ArchiveEntry(s.parse_moves("L3 C2 U1 U3"), candidate.endpoint, candidate.cost,
                                     candidate.solved, None, 0, "fixture: longer spelling")
        stalled = run.last_quality_improvement_generation
        self.first_intervention(run)
        run.generation_in_progress = 180
        self.assertTrue(run.offer_to_archive(candidate, uid=None, source="shorter"))
        self.assertEqual(run.last_quality_improvement_generation, stalled)
        self.qualify(run, 200, 250, blocks=UNHEALTHY)
        self.assertEqual(run._recovery_pending["type"], "solution plus partial block refresh")


class BlockHealthTests(RecoveryTestCase):
    def test_identity_is_the_full_transformation(self):
        equivalent = [("R1", "L1"), ("L1", "R1"), ("U1",), ("U1", "U1", "U1", "U1", "U1")]
        record = s.measure_block_health(fake_blocks([s.parse_moves(" ".join(a)) for a in equivalent], True), 7)
        self.assertEqual((record["distinct"], record["count"], record["flags"]), (2, 4, []))
        more = equivalent + [("R2", "U2") * 6 + ("L1", "R1")]
        record = s.measure_block_health(fake_blocks([s.parse_moves(" ".join(a)) for a in more], True), 7)
        self.assertEqual((record["distinct"], record["flags"]), (2, ["LOW_DISTINCT_TRANSFORMATIONS"]))

    def test_no_improving_block_and_unavailable_evaluations(self):
        record = s.measure_block_health(fake_blocks([(a,) for a in s.REAL_MOVES], False), 7)
        self.assertEqual(record["flags"], ["NO_IMPROVING_BLOCKS"])
        stale = fake_blocks([("R1",), ("U1",)], True)
        stale[1].batch_serial = 6
        record = s.measure_block_health(stale, 7)
        self.assertFalse(record["available"])
        self.assertFalse(record["unhealthy"])  # missing evidence is not unhealthy evidence
        run = self.make_run()
        run._evaluated_blocks = (7, 99, UNHEALTHY)
        self.assertFalse(run.block_health(100)["available"])
        self.assertTrue(run.block_health(99)["unhealthy"])

    def test_any_strict_improvement_counts_whatever_the_gain_threshold(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        for gain in (100.0, 1.0):
            run = self.advanced(1, key=key, min_block_gain=gain)
            blocks = [run.make_block((allele,), origin="fixture") for allele in s.REAL_MOVES]
            batch = run.build_context_batch(sorted(run.population, key=s._solution_sort_key))
            run.evaluate_blocks(blocks, batch)
            improving = [block for block in blocks if block.evaluation.improving_contexts]
            # Independent: a one-move block improves when that move lowers some context's cost.
            expected = [block for block in blocks
                        if any(run.objective.cost(s.replay_key(context.key, block.active)) < context.cost
                               for context in batch.contexts)]
            self.assertEqual(improving, expected)
            self.assertTrue(improving)
            if gain == 100.0:
                self.assertFalse(any(block.evaluation.useful for block in blocks))  # none meets the threshold
            else:  # integer fitness at the default gain: both notions agree
                self.assertEqual([block.evaluation.useful for block in blocks],
                                 [bool(block.evaluation.improving_contexts) for block in blocks])
            run._evaluated_blocks = (batch.serial, 1, blocks)
            self.assertNotIn("NO_IMPROVING_BLOCKS", run.block_health(1)["flags"])

    def test_diagnostics_use_no_randomness(self):
        run = self.advanced(3)
        state = run.rng.getstate()
        run.block_health(3)
        s.measure_population_diversity(run.population, 20)
        self.qualify(run, 100)
        self.assertEqual(run.rng.getstate(), state)


class RecoveryGenerationTests(RecoveryTestCase):
    def schedule(self, run, generation, refresh=False, limit=0):
        run._recovery_pending = {"scheduled_generation": generation - 1, "apply_generation": generation,
                                 "type": "solution-only", "refresh_blocks": refresh, "block_refresh_limit": limit,
                                 "triggers": ["NO_QUALITY_PROGRESS", "LOW_UNIQUE_STATES"], "streak": 2,
                                 "generations_without_quality_improvement": 150, "finisher": "finisher disabled",
                                 "block_health": s.measure_block_health((), None), "block_note": "fixture",
                                 "before": "fixture", "quality": run._quality(), "seconds": 0.0}

    def test_invariants_and_valid_recovery_offspring(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        run = self.advanced(3, key=key)
        before = sorted(run.population, key=s._solution_sort_key)
        archive_rank = run.archive.rank
        self.schedule(run, 4)
        run.run_generation(4)
        population = run.population
        self.assertEqual(len(population), 20)
        elites, offspring, resets = population[:4], population[4:16], population[16:]
        self.assertEqual([ind.uid for ind in elites], [ind.uid for ind in before[:4]])
        self.assertTrue(all(ind.slot == "elite" for ind in elites))
        self.assertTrue(all(ind.slot == "reset" and ind.active == () for ind in resets))
        self.assertTrue(all(ind.origin == "recovery" and ind.slot == "offspring" for ind in offspring))
        self.assertEqual(Counter(ind.operator.split("-x")[0] for ind in offspring),
                         {"recovery-fresh": 4, "recovery-mutation": 4, "recovery-blocks": 4})
        for ind in offspring:
            self.assertLessEqual(len(ind.active), run.solution_capacity)
            self.assertEqual(s.simplify(ind.active), ind.active)
            self.assertNotIn(s.NOOP, ind.active)
            self.assertEqual(list(ind), list(ind.active) + [s.NOOP] * (run.solution_capacity - len(ind.active)))
            replayed = s.state_key(s.replay(s.cube_from_key(key), ind.active))  # independent cube.py replay
            self.assertEqual(replayed, ind.endpoint)
            self.assertEqual(run.objective.cost(replayed), ind.cost)
        self.assertLessEqual(run.archive.rank, archive_rank)
        record = run.recovery_records[-1]
        self.assertEqual((record["solution"]["elites"], record["solution"]["offspring"], record["solution"]["resets"]),
                         (4, 12, 4))
        self.assertTrue(any("endpoint diversity after publication" in line for line in self.messages))
        self.assertEqual(run._recovery_last_generation, 4)
        recovery_uids = {ind.uid for ind in offspring}
        run.run_generation(5)  # the recovery candidates are parents and survivors of the evolving population
        used = {parent for ind in run.population for parent in ind.parents} | {ind.uid for ind in run.population}
        self.assertTrue(recovery_uids & used)

    def test_complete_proposals_allow_temporary_regressions(self):
        key = after(SOLVED, "R1 U1 L1 D1")
        run = self.advanced(1, key=key)
        detour = s.parse_moves("C1 C3 D3 L3")
        self.assertGreater(run.objective.cost(after(key, "C1")), run.objective.cost(key))  # premise
        calls = []
        original = run.normalize_solution

        def normalize(proposal, **options):
            calls.append(tuple(proposal))
            return original(proposal, **options)

        self.schedule(run, 2)
        with mock.patch.object(run, "_recovery_proposal", return_value=(detour, (), "fixture", None)), \
                mock.patch.object(run, "normalize_solution", side_effect=normalize):
            population = run.recovery_solution_population(run.population, run._recovery_pending)
        self.assertTrue(all(call == detour for call in calls))  # normalised once per complete proposal
        expected = after(key, "D3 L3")
        self.assertTrue(all(ind.endpoint == expected for ind in population[4:16]))

    def test_novelty_after_normalisation_and_bounded_retries(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        run = self.advanced(2, key=key)
        r1 = s.parse_moves("R1")[0]
        spellings = [(r1,), (r1, s.NOOP), (s.NOOP, r1), (s.NOOP, r1, s.NOOP)]  # one move, four chromosomes
        cycle = iter(spellings * 100)
        self.schedule(run, 3)
        with mock.patch.object(run, "_recovery_proposal",
                               side_effect=lambda kind, parent: (next(cycle), (), "fixture", None)) as proposal:
            population = run.recovery_solution_population(run.population, run._recovery_pending)
        endpoint = run.normalize_solution((r1,), check_budget=False).endpoint
        elite_endpoints = {ind.endpoint for ind in population[:4]}
        novel = 0 if endpoint in elite_endpoints or endpoint == key else 1
        stats = run._recovery_pending["solution"]
        self.assertEqual(stats["shortfall"], 12 - novel)
        self.assertEqual(stats["retries"], (12 - novel) * s.RECOVERY_NOVELTY_RETRIES)
        self.assertEqual(proposal.call_count, novel + (12 - novel) * (1 + s.RECOVERY_NOVELTY_RETRIES))
        self.assertEqual(len(population), 20)

    def test_parents_cycle_through_distinct_endpoints(self):
        key = after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1")
        run = self.advanced(3, key=key)
        ranked = sorted(run.population, key=s._solution_sort_key)
        distinct = {ind.endpoint for ind in ranked[:16]}
        chosen = []

        def proposal(kind, next_parent):
            if kind == "fresh":
                return (s.REAL_MOVES[0],), (), "fixture", None
            parent = next_parent()
            chosen.append(parent.endpoint)
            return parent.active + (s.REAL_MOVES[len(chosen) % 18],), (parent.uid,), "fixture", None

        self.schedule(run, 4)
        with mock.patch.object(run, "_recovery_proposal", side_effect=proposal):
            run.recovery_solution_population(run.population, run._recovery_pending)
        self.assertGreater(len(distinct), 1)
        first = chosen[:len(distinct)]
        self.assertEqual(sorted(first), sorted(distinct))  # every distinct endpoint once before any repeat

    def test_an_improving_proposal_leaves_the_empty_baseline(self):
        run = self.advanced(1)  # every candidate is the empty baseline
        self.assertEqual({ind.endpoint for ind in run.population}, {PLATEAU})
        improving = s.parse_moves("B3 U3 R2 U1 R1 U1 R3 U3 R3 U3 R3 U1 R3 U1 B1")
        original = run._recovery_proposal

        def proposal(kind, next_parent):
            if kind == "fresh":
                return improving, (), "recovery-fresh", None
            return original(kind, next_parent)

        self.schedule(run, 2)
        with mock.patch.object(run, "_recovery_proposal", side_effect=proposal):
            run.run_generation(2)
        offspring = [ind for ind in run.population if ind.origin == "recovery"]
        self.assertIn(improving, [ind.active for ind in offspring])  # kept unchanged by normalisation
        self.assertEqual((run.archive.cost, run.archive.active), (30, improving))
        self.assertGreater(len({ind.endpoint for ind in run.population}), 1)

    def test_interrupted_recovery_time_is_still_reported(self):
        for refresh in (False, True):
            # On the plateau every block is removable, so the third fresh block is drawn inside the refresh.
            run = self.advanced(2, key=PLATEAU if refresh else after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1"))
            self.schedule(run, 3, refresh=refresh, limit=5)
            target = run.fresh_block if refresh else run.normalize_solution
            calls = {"count": 0}

            def interrupting(*args, _target=target, **options):
                calls["count"] += 1
                if calls["count"] == 3:
                    run.budget.request_interrupt()
                return _target(*args, **options)

            name = "fresh_block" if refresh else "normalize_solution"
            with mock.patch.object(run, name, side_effect=interrupting):
                with self.assertRaises(s.SearchInterrupted):
                    run.run_generation(3)
            part = "blocks" if refresh else "solution"
            self.assertGreater(run._recovery_pending[part]["seconds"], 0)
            self.assertEqual(run._recovery_pending["seconds"], run._recovery_pending[part]["seconds"])
            self.assertEqual(run.recovery_seconds, run._recovery_pending["seconds"])

    def test_interruption_keeps_the_published_population(self):
        run = self.advanced(2, key=after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1"))
        published, blocks, generation = run.population, run.blocks, run.completed_generation
        self.schedule(run, 3)
        original = run.normalize_solution
        calls = {"count": 0}

        def normalize(proposal, **options):
            calls["count"] += 1
            if calls["count"] == 5:
                run.budget.request_interrupt()
            return original(proposal, **options)

        with mock.patch.object(run, "normalize_solution", side_effect=normalize):
            with self.assertRaises(s.SearchInterrupted):
                run.run_generation(3)
        self.assertIs(run.population, published)
        self.assertIs(run.blocks, blocks)
        self.assertEqual((run.completed_generation, run.partial_generation), (generation, 3))
        self.assertIsNotNone(run._recovery_pending)
        self.assertEqual(run.recovery_records, [])

    def test_run_level_interruption_reports_the_pending_intervention(self):
        run = self.make_run(generations=400)
        original = run.recovery_solution_population

        def interrupted(population, pending):
            run.budget.request_interrupt()
            return original(population, pending)

        with mock.patch.object(run, "recovery_solution_population", side_effect=interrupted):
            result = run.run()
        self.assertEqual(result.status, s.STATUS_INTERRUPTED)
        self.assertTrue(result.recovery_report["pending"])
        self.assertEqual(result.recovery_report["interventions"], 0)


class BlockRefreshTests(RecoveryTestCase):
    def crafted(self, run):
        rng = random.Random(9)
        sequences = [s.parse_moves("C1"), s.parse_moves("B1"),
                     s.parse_moves("R1 L1 U2 D2"), s.parse_moves("L1 R1 U2 D2"),
                     s.parse_moves("U1 D1 C2 B2"), s.parse_moves("D1 U1 C2 B2")]
        while len(sequences) < 20:
            candidate = s.generate_reduced_block(3, rng)
            if candidate not in sequences:
                sequences.append(candidate)
        return [run.make_block(sequence, origin="fixture") for sequence in sequences]

    def test_redundant_weak_blocks_go_first_and_replacements_share_the_batch(self):
        run = self.advanced(2)
        blocks = self.crafted(run)
        batch = run.build_context_batch(sorted(run.population, key=s._solution_sort_key))
        run.evaluate_blocks(blocks, batch)
        self.assertFalse(any(block.evaluation.useful for block in blocks))  # the plateau premise
        pending = {"block_refresh_limit": 2, "seconds": 0.0}
        library = mock.Mock()
        run.library = library
        refreshed = run._refresh_blocks(blocks, batch, pending)
        self.assertEqual(library.mock_calls, [])  # the persistent library is never touched
        removed = {block.uid for block in blocks} - {block.uid for block in refreshed}
        self.assertEqual(removed, {blocks[3].uid, blocks[5].uid})  # the later spelling of each equivalent pair
        self.assertEqual(len(refreshed), 20)
        protected = sorted(blocks, key=s._block_sort_key)[:2]
        self.assertTrue(all(any(block is kept for kept in refreshed) for block in protected))
        for block in refreshed:
            self.assertEqual(block.batch_serial, batch.serial)  # every score comes from the same batch
            self.assertLessEqual(len(block.active), run.block_capacity)
        self.assertLessEqual(max(Counter(block.active for block in refreshed).values()), run.max_block_copies)
        self.assertEqual(pending["blocks"]["refreshed"], 2)
        self.assertEqual(pending["blocks"]["redundant"], 2)
        self.assertGreaterEqual(pending["blocks"]["after"]["distinct"], pending["blocks"]["before"]["distinct"])

    def test_useful_distinct_blocks_are_retained(self):
        run = self.advanced(2, key=after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1"))
        batch = run.build_context_batch(sorted(run.population, key=s._solution_sort_key))
        blocks = run.blocks
        run.evaluate_blocks(blocks, batch)
        refreshed = run._refresh_blocks(blocks, batch, {"block_refresh_limit": 10, "seconds": 0.0})
        kept = {id(block) for block in refreshed}
        best_of = {}
        for block in sorted(blocks, key=s._block_sort_key):
            best_of.setdefault(s.block_permutation(block.active), block)
        for block in best_of.values():
            if block.evaluation.useful:
                self.assertIn(id(block), kept)

    def test_improvement_during_block_evaluation_cancels_the_scheduled_refresh(self):
        run = self.advanced(2)
        RecoveryGenerationTests.schedule(self, run, 3, refresh=True, limit=5)
        run._recovery_pending["type"] = "solution plus partial block refresh"
        run._recovery_block_streak = 2
        original = run.evaluate_blocks

        def improving_evaluation(blocks, batch):
            original(blocks, batch)
            archive = run.archive  # stands in for a block discovery verified during evaluation
            run.archive = s.ArchiveEntry(archive.active, archive.endpoint, archive.cost - 6, False, None, 3,
                                         "fixture")

        with mock.patch.object(run, "evaluate_blocks", side_effect=improving_evaluation):
            run.run_generation(3)
        record = run.recovery_records[-1]
        self.assertEqual((record["type"], record["refresh_blocks"]), ("solution-only", False))
        self.assertIn("cancelled (cube quality improved", record["block_note"])
        self.assertNotIn("blocks", record)
        self.assertFalse(any(block.origin == "recovery" for block in run._evaluated_blocks[2]))
        self.assertEqual(run._recovery_block_streak, 2)  # unused block evidence is kept
        self.assertTrue(any("cancelled (cube quality improved" in line for line in self.messages))

    def test_escalated_generation_uses_the_refreshed_blocks(self):
        run = self.advanced(2)
        RecoveryGenerationTests.schedule(self, run, 3, refresh=True, limit=5)
        run.run_generation(3)
        serial, generation, evaluated = run._evaluated_blocks
        self.assertEqual((serial, generation), (run.batch.serial, 3))
        self.assertTrue(all(block.batch_serial == serial for block in evaluated))
        self.assertEqual(sum(1 for block in evaluated if block.origin == "recovery"),
                         run.recovery_records[-1]["blocks"]["refreshed"])
        indexed = {block.uid for block in run._block_representatives}
        self.assertTrue(indexed <= {block.uid for block in evaluated})
        self.assertEqual(len(run.blocks), 20)


class RunTests(RecoveryTestCase):
    def test_plateau_run_intervenes_then_escalates(self):
        run = self.make_run(generations=260)
        result = run.run()
        records = run.recovery_records
        self.assertEqual([record["apply_generation"] for record in records], [151, 251])
        self.assertEqual([record["type"] for record in records],
                         ["solution-only", "solution plus partial block refresh"])
        self.assertEqual(result.recovery_report["interventions"], 2)
        # Recovery never pretends to improve: the stagnation count keeps growing.
        checks = {record["generation"]: record["generations_without_quality_improvement"]
                  for record in run.diversity_records}
        self.assertEqual(checks[200], 200 - run.last_quality_improvement_generation)
        self.assertEqual(checks[250], 250 - run.last_quality_improvement_generation)

    def test_fixed_seed_is_deterministic(self):
        digests, reports = [], []
        for _ in range(2):
            run = self.make_run(generations=260)
            result = run.run()
            digests.append(timeless_digest(run, result))
            timed = ("seconds", "after_record", "before_record", "solution", "blocks")
            reports.append([({k: v for k, v in record.items() if k not in timed},
                             {k: v for k, v in record["solution"].items() if k != "seconds"})
                            for record in run.recovery_records])
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(reports[0], reports[1])

    def test_library_and_archive_are_preserved(self):
        path = os.path.join(self.directory, "blocks.json")
        seeded = self.make_run(key=after(SOLVED, "R1 U2 C3 D1 B2 L3 R2 C1"), generations=3, block_library=path,
                               adaptive_recovery=False)
        seeded.run()
        seeded.finish_block_library()
        before = s.BlockLibrary(path, counters=s.SearchCounters())
        self.assertTrue(before.load())
        run = self.make_run(generations=260, block_library=path)
        result = run.run()
        run.finish_block_library()
        self.assertEqual(result.recovery_report["with_block_refresh"], 1)
        after_library = s.BlockLibrary(path, counters=s.SearchCounters())
        self.assertTrue(after_library.load())
        self.assertTrue(before.entries)
        self.assertTrue(set(before.entries) <= set(after_library.entries) or len(after_library.entries) == 100)
        self.assertLessEqual(result.archive.rank, run.baseline.rank)


class ConfigAndCliTests(RecoveryTestCase):
    def test_validation(self):
        bad = [dict(adaptive_recovery="yes"), dict(recovery_consecutive_checks=0), dict(recovery_consecutive_checks=1.5),
               dict(recovery_cooldown_generations=0), dict(recovery_block_refresh_fraction=-0.1),
               dict(recovery_block_refresh_fraction=0.51), dict(recovery_block_refresh_fraction=float("nan"))]
        for overrides in bad:
            with self.assertRaises(s.ConfigError, msg=overrides):
                s.validate_config(s.SearchConfig(**overrides))
        with self.assertRaisesRegex(s.ConfigError, "needs the diversity checks"):
            s.validate_config(s.SearchConfig(adaptive_recovery=True, diversity_checks=False))
        s.validate_config(s.SearchConfig(adaptive_recovery=True, recovery_block_refresh_fraction=0.5))

    def test_cli_options(self):
        parser = s.build_arg_parser()
        config = s.config_from_args(parser.parse_args([]))
        self.assertEqual((config.adaptive_recovery, config.recovery_consecutive_checks,
                          config.recovery_cooldown_generations, config.recovery_block_refresh_fraction),
                         (False, 2, 100, 0.25))
        config = s.config_from_args(parser.parse_args(["--adaptive-recovery", "--recovery-consecutive-checks", "3",
                                                       "--recovery-cooldown-generations", "200",
                                                       "--recovery-block-refresh-fraction", "0.1"]))
        self.assertEqual((config.adaptive_recovery, config.recovery_consecutive_checks,
                          config.recovery_cooldown_generations, config.recovery_block_refresh_fraction),
                         (True, 3, 200, 0.1))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = s.main(["--adaptive-recovery", "--no-diversity-checks", "--generations", "1"])
        self.assertEqual(code, 2)
        self.assertIn("needs the diversity checks", output.getvalue())

    def test_cli_reports_startup_interventions_and_totals(self):
        with mock.patch.object(s, "default_input_cube", lambda: s.cube_from_key(PLATEAU)):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                s.main(["--adaptive-recovery", "--generations", "160", "--solution-population", "20",
                        "--block-population", "20", "--contexts", "6", "--solution-capacity", "16",
                        "--block-capacity", "8", "--seed", "3", "--no-block-library"])
            text = output.getvalue()
            self.assertIn("Adaptive recovery: after 2 consecutive health checks", text)
            self.assertIn("[gen 151] RECOVERY applied (solution-only)", text)
            self.assertIn("WARNING: low diversity with stalled cube-quality progress", text)  # warnings stay distinct
            self.assertIn("Adaptive recovery: 1 intervention(s) (1 solution-only, 0 with block refresh)", text)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                s.main(["--generations", "0", "--no-block-library"])
            self.assertIn("Adaptive recovery: disabled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
