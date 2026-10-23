"""Bounded comparison (spec section 13): cooperative coevolution vs the existing prefix GA.

Each run executes serially in its own subprocess with the same budget:

* new solver: ``solver_deap_coevolution`` with its block defaults, solution
  capacity 32, population 200, objective ``manhattan`` and a time limit;
* baseline: ``solver_deap.py`` read from disk, compiled in memory with AST
  overrides (seed, MAX_GENES=32, POPULATION_SIZE=200, generation ceiling,
  headless) plus read-only instrumentation. Its source is never written and it
  is never imported by the new solver.

The common outcome is the independently replayed endpoint cost/solved status of
each run's best verified prefix, not the two programs' internal fitness values.

Usage::

    uv run python benchmark_coevolution.py                 # seeds 42 43 44, 120 s per run
    uv run python benchmark_coevolution.py --seconds 5 --output /tmp/smoke.txt

The report goes to ``experiment_coevolution.txt`` (numbered if that file exists).
"""

from __future__ import annotations

import time

_PROCESS_START = time.monotonic()

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_SOLVER_PATH = os.path.join(REPO_DIR, "solver_deap.py")
PROTECTED_FILES = ("cube.py", "solver_deap.py")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SECONDS = 120.0
SOLUTION_CAPACITY = 32
SOLUTION_POPULATION = 200
GENERATION_CEILING = 1_000_000
NEUTRALITY_GENERATIONS = 5
REPORT_FILENAME = "experiment_coevolution.txt"
LIBRARIES = ("deap", "numpy", "matplotlib", "colorama")
OBSERVATION_CACHE_LIMIT = 200_000

OLD_INSTRUMENTATION_RULES = (
    "toolbox.evaluate is wrapped: a budget check runs before each evaluation, then the original "
    "evaluator runs unchanged (including its global min_moves/best_fitness_ever updates).",
    "rotate() is wrapped: a budget check runs before every face turn, so the time limit can stop the "
    "script inside an evaluation instead of after a whole population; it also counts calls and quarter turns.",
    "manhattan_distance() and misplaced_stickers() are wrapped only to count calls.",
    "After each completed evaluation the harness replays the individual's genes from the original cube "
    "with solver_deap_coevolution's reducer and best-prefix normalisation (capacity 32), verifies the "
    "chosen prefix by independent replay and keeps it if its (unsolved, H, length) rank is strictly better. "
    "This also covers partially completed generations before the hall-of-fame update.",
    "toolbox.select is wrapped only to mark generation boundaries and to count distinct best-prefix "
    "endpoints of each fully evaluated population; it passes the arguments through unchanged.",
    "Observation never touches the global random module, the individuals or their fitness values, so "
    "selection decisions are unchanged (checked by the neutrality run below).",
    "Observation time is paused out of the baseline's search budget and reported separately, so harness "
    "overhead does not shorten the baseline's own 120 s of search.",
    "A verified feasible solve stops the run early.",
)


class HarnessStop(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status
        self.reason = reason


class PausableBudget:
    """Monotonic search budget that excludes paused harness-observation time."""

    def __init__(self, seconds, clock=time.monotonic):
        self.seconds = seconds
        self.clock = clock
        self.started_at = None
        self.excluded = 0.0
        self._pause_started = None

    def start(self):
        self.started_at = self.clock()

    def used(self):
        now = self.clock()
        excluded = self.excluded
        if self._pause_started is not None:
            excluded += now - self._pause_started
        return now - self.started_at - excluded

    def check(self):
        if self._pause_started is None and self.used() >= self.seconds:
            raise HarnessStop("BUDGET_EXHAUSTED", "time limit reached")

    @contextlib.contextmanager
    def paused(self):
        if self._pause_started is not None:
            yield
            return
        self._pause_started = self.clock()
        try:
            yield
        finally:
            self.excluded += self.clock() - self._pause_started
            self._pause_started = None


# ---------------------------------------------------------------------------
# Baseline preparation (AST overrides) and instrumentation
# ---------------------------------------------------------------------------


def prepare_old_solver(source, seed, generation_ceiling):
    """Return (setup module, GA-call module, override descriptions) for solver_deap.py."""
    tree = ast.parse(source, OLD_SOLVER_PATH)
    assignments = {
        "POPULATION_SIZE": SOLUTION_POPULATION,
        "GENERATIONS": generation_ceiling,
        "MAX_GENES": SOLUTION_CAPACITY,
    }
    applied = Counter()
    overrides = []
    ga_index = None
    for index, node in enumerate(tree.body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in assignments):
            name = node.targets[0].id
            overrides.append(f"line {node.lineno}: {name} = {ast.unparse(node.value)} -> {assignments[name]}")
            node.value = ast.Constant(assignments[name])
            applied[name] += 1
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and ast.unparse(node.value.func) == "random.seed"):
            overrides.append(f"line {node.lineno}: {ast.unparse(node)} -> random.seed({seed})")
            node.value.args = [ast.Constant(seed)]
            node.value.keywords = []
            applied["random.seed"] += 1
        elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == "eaSimpleWithElitism"):
            if ga_index is not None:
                raise RuntimeError("solver_deap.py has more than one eaSimpleWithElitism run")
            ga_index = index
    for name in (*assignments, "random.seed"):
        if applied[name] != 1:
            raise RuntimeError(f"expected exactly one {name} statement in solver_deap.py, found {applied[name]}")
    if ga_index is None:
        raise RuntimeError("solver_deap.py has no module-level eaSimpleWithElitism run")
    trailing = tree.body[ga_index + 1:]
    if trailing:
        overrides.append(
            f"lines {trailing[0].lineno}-{trailing[-1].end_lineno}: {len(trailing)} statements after the GA run "
            "(full-chromosome final printout, print_best, matplotlib plot and plt.show) are not executed; "
            "the harness replays and reports the best verified prefix instead")
    setup = ast.fix_missing_locations(ast.Module(tree.body[:ga_index], []))
    ga_run = ast.fix_missing_locations(ast.Module([tree.body[ga_index]], []))
    return setup, ga_run, overrides


class OldSolverInstrumentation:
    """Read-only budget, counting and best-prefix observation hooks for the baseline."""

    def __init__(self, namespace, observer, budget, observe=True):
        import solver_deap_coevolution as s

        self.s = s
        self.namespace = namespace
        self.observer = observer
        self.budget = budget
        self.observe = observe
        self.allele_of = [s.ALLELE_BY_NAME[name] for name in namespace["genes"]]
        self.cache = {}
        self.best = observer.baseline
        self.best_found = {"seconds": 0.0, "evaluation": 0, "generation": 0}
        self.evaluations = 0
        self.rotate_calls = 0
        self.quarter_turns = 0
        self.manhattan_calls = 0
        self.misplaced_calls = 0
        self.observations = 0
        self.observation_cache_hits = 0
        self.select_calls = 0
        self.last_population = None

    def install(self):
        namespace = self.namespace
        toolbox = namespace["toolbox"]
        rotate = namespace["rotate"]
        manhattan = namespace["manhattan_distance"]
        misplaced = namespace["misplaced_stickers"]
        evaluate = toolbox.evaluate
        select = toolbox.select

        def counted_rotate(actual_cube, gene):
            self.budget.check()
            self.rotate_calls += 1
            self.quarter_turns += int(gene[1])
            return rotate(actual_cube, gene)

        def counted_manhattan(cube):
            self.manhattan_calls += 1
            return manhattan(cube)

        def counted_misplaced(cube):
            self.misplaced_calls += 1
            return misplaced(cube)

        def observed_evaluate(individual):
            self.budget.check()
            fitness = evaluate(individual)
            self.evaluations += 1
            if self.observe:
                with self.budget.paused():
                    self._observe(individual)
            return fitness

        def observed_select(population, k):
            with self.budget.paused():
                self.select_calls += 1
                self.last_population = self.describe_population(population)
            return select(population, k)

        namespace["rotate"] = counted_rotate
        namespace["manhattan_distance"] = counted_manhattan
        namespace["misplaced_stickers"] = counted_misplaced
        toolbox.register("evaluate", observed_evaluate)
        toolbox.register("select", observed_select)

    def candidate(self, genes):
        candidate = self.cache.get(genes)
        if candidate is not None:
            self.observation_cache_hits += 1
            return candidate
        moves = tuple(self.allele_of[gene] for gene in genes)
        candidate = self.observer.normalize_solution(moves, check_budget=False)
        self.observations += 1
        if len(self.cache) >= OBSERVATION_CACHE_LIMIT:
            self.cache.clear()
        self.cache[genes] = candidate
        return candidate

    def _observe(self, individual):
        candidate = self.candidate(tuple(individual))
        if candidate.rank < self.best.rank:
            self.best = candidate
            self.best_found = {"seconds": self.budget.used(), "evaluation": self.evaluations,
                               "generation": self.select_calls}
        if candidate.solved:
            raise HarnessStop("SOLVED", "verified solve")

    def describe_population(self, population):
        genomes = [tuple(individual) for individual in population]
        description = {
            "size": len(genomes),
            "digest": hashlib.sha256(repr(genomes).encode()).hexdigest(),
        }
        if self.observe:
            description["unique_endpoints"] = len({self.candidate(genes).endpoint for genes in genomes})
        return description


def run_old_worker(seed, seconds, generation_ceiling, log_path, observe=True):
    import solver_deap_coevolution as s

    with open(OLD_SOLVER_PATH, "rb") as handle:
        source_bytes = handle.read()
    setup_module, ga_module, overrides = prepare_old_solver(source_bytes.decode("utf-8"), seed, generation_ceiling)
    namespace = {"__name__": "solver_deap_benchmark", "__file__": OLD_SOLVER_PATH}
    with open(log_path or os.devnull, "w") as log:
        with contextlib.redirect_stdout(log):
            exec(compile(setup_module, OLD_SOLVER_PATH, "exec"), namespace)
        input_key = s.state_key(namespace["my_cube"])
        if input_key != s.validate_input_cube(s.default_input_cube()):
            raise RuntimeError("solver_deap.py's active my_cube differs from the specified input literal")
        config = s.SearchConfig(seed=seed, solution_population=SOLUTION_POPULATION,
                                solution_capacity=SOLUTION_CAPACITY, generations=0)
        observer = s.CoevolutionRun(config, input_key, s.make_objective("manhattan"), seed, log=lambda message: None)
        budget = PausableBudget(seconds)
        hooks = OldSolverInstrumentation(namespace, observer, budget, observe)
        hooks.install()

        budget.start()
        search_started = time.monotonic()
        status, reason = "BUDGET_EXHAUSTED", "generation ceiling reached"
        try:
            with contextlib.redirect_stdout(log):
                exec(compile(ga_module, OLD_SOLVER_PATH, "exec"), namespace)
        except HarnessStop as stop:
            status, reason = stop.status, stop.reason
        search_wall = time.monotonic() - search_started

    finalization_started = time.monotonic()
    completed = hooks.select_calls - 1 if hooks.select_calls else None
    final_population = hooks.last_population
    if "population" in namespace and reason == "generation ceiling reached":
        completed = generation_ceiling
        final_population = hooks.describe_population(namespace["population"])
    best = hooks.best
    replay = verify_replay(s, input_key, best.active, best.endpoint, best.cost, best.solved)
    if replay["problems"]:
        status = "ERROR"
    finalization = time.monotonic() - finalization_started
    if source_bytes != open(OLD_SOLVER_PATH, "rb").read():
        raise RuntimeError("solver_deap.py changed during the benchmark")

    return {
        "solver": "old",
        "seed": seed,
        "status": status,
        "stop_reason": reason,
        "best": {
            "moves": s.format_moves(best.active),
            "length": len(best.active),
            "cost": best.cost,
            "solved": best.solved,
            "endpoint": best.endpoint,
            "found": hooks.best_found,
        },
        "replay": replay,
        "input_cost": observer.baseline.cost,
        "unique_endpoints": final_population.get("unique_endpoints") if final_population else None,
        "population_size": final_population["size"] if final_population else None,
        "population_digest": final_population["digest"] if final_population else None,
        "generations_completed": completed,
        "timing": {
            "setup": search_started - _PROCESS_START,
            "search": budget.used(),
            "observation_overhead": budget.excluded,
            "search_wall": search_wall,
            "finalization": finalization,
        },
        "work": {
            "evaluations (one my_cube.clone() each)": hooks.evaluations,
            "manhattan_distance calls": hooks.manhattan_calls,
            "misplaced_stickers calls": hooks.misplaced_calls,
            "rotate calls": hooks.rotate_calls,
            "quarter turns": hooks.quarter_turns,
            "harness normalisations (excluded from budget)": hooks.observations,
            "harness observation cache hits": hooks.observation_cache_hits,
        },
        "overrides": overrides,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "observe": observe,
    }


def verify_replay(s, input_key, active, endpoint, cost, solved):
    """Replay the best active moves from a fresh original cube and compare stored metadata."""
    cube = s.replay(s.cube_from_key(input_key), active)
    key = s.state_key(cube)
    replayed_cost = s.manhattan_cost(key)
    replayed_solved = s.is_solved_key(key)
    problems = []
    if key != endpoint:
        problems.append("replayed endpoint differs")
    if replayed_cost != cost:
        problems.append(f"replayed cost {replayed_cost} differs from stored {cost}")
    if replayed_solved != solved:
        problems.append("replayed solved flag differs")
    if s.simplify(active) != tuple(active) or len(active) > SOLUTION_CAPACITY:
        problems.append("best moves are not reduced or exceed capacity 32")
    return {"cost": replayed_cost, "solved": replayed_solved, "endpoint": key, "problems": problems}


# ---------------------------------------------------------------------------
# New-solver worker
# ---------------------------------------------------------------------------


def run_new_worker(seed, seconds, generation_ceiling, log_path):
    import solver_deap_coevolution as s

    config = s.SearchConfig(fitness="manhattan", seed=seed, solution_population=SOLUTION_POPULATION,
                            solution_capacity=SOLUTION_CAPACITY, generations=generation_ceiling,
                            time_limit=seconds, log_every=100)
    objective = s.validate_config(config)
    input_key = s.validate_input_cube(s.default_input_cube())
    budget = s.Budget(seconds)
    with open(log_path or os.devnull, "w") as log:
        run = s.CoevolutionRun(config, input_key, objective, seed, budget=budget,
                               log=lambda message: print(message, file=log))
        result = run.run()
        search_wall = time.monotonic() - budget.started_at
        run.report_progress_if_stale()

    finalization_started = time.monotonic()
    final = s.finalize_result(result)
    entry = result.archive
    replay = verify_replay(s, input_key, entry.active, entry.endpoint, entry.cost, entry.solved)
    finalization = time.monotonic() - finalization_started
    status = final.status if not replay["problems"] else "ERROR"

    counters = result.counters
    blocks = run.blocks or []
    summary = run.last_block_summary or {}
    attributions = counters.attributed_improvements
    return {
        "solver": "new",
        "seed": seed,
        "status": status,
        "stop_reason": result.reason,
        "best": {
            "moves": s.format_moves(entry.active),
            "length": len(entry.active),
            "cost": entry.cost,
            "solved": entry.solved,
            "endpoint": entry.endpoint,
            "found": {"generation": entry.generation, "source": entry.source},
        },
        "replay": {**replay, "problems": replay["problems"] + final.problems},
        "input_cost": run.baseline.cost,
        "unique_endpoints": len({individual.endpoint for individual in run.population or []}),
        "population_size": len(run.population or []),
        "generations_completed": result.completed_generations,
        "partial_generation": result.partial_generation,
        "partial_phase": result.partial_phase,
        "timing": {
            "setup": budget.started_at - _PROCESS_START,
            "search": result.search_seconds,
            "search_wall": search_wall,
            "finalization": finalization,
        },
        "work": {
            "H computations": counters.cost_computations,
            "H cache hits": counters.cost_cache_hits,
            "quarter turns": counters.quarter_turns,
            "move applications": counters.move_applications,
            "cube clones": counters.cube_clones,
            "normalisations": counters.normalizations,
            "independent replays": counters.independent_replays,
            "block sequence evaluations": counters.block_sequence_evaluations,
            "block evaluation cache hits": counters.block_evaluation_cache_hits,
            "solution capacity rejections": counters.solution_capacity_rejections,
            "block oversized rejections": counters.block_oversized_rejections,
            "block-prefix capacity exclusions": counters.block_prefix_capacity_exclusions,
            "no-change offspring": counters.no_change_offspring,
        },
        "blocks": {
            "population_size": len(blocks),
            "distinct_sequences": len({block.active for block in blocks}),
            "length_histogram": dict(sorted(Counter(len(block.active) for block in blocks).items())),
            "origins": dict(Counter(block.origin for block in blocks)),
            "last_batch": {key: value for key, value in summary.items() if key != "lengths"},
            "fallbacks": dict(counters.block_fallbacks),
            "baseline_start_attempts": counters.baseline_start_attempts,
            "baseline_start_improvements": counters.baseline_start_improvements,
            "targeted_appends": counters.targeted_appends,
            "exploratory_appends": counters.exploratory_appends,
            "attributed_improvements_total": sum(attributions.values()),
            "attributed_block_ids": len(attributions),
            "top_attributions": run.describe_attributions(),
            "block_discoveries_archived": counters.block_discoveries_archived,
        },
        "config": {
            "fitness": config.fitness, "solution_population": config.solution_population,
            "block_population": config.block_population, "solution_capacity": config.solution_capacity,
            "block_capacity": config.block_capacity, "contexts": config.contexts,
            "generations (ceiling)": config.generations, "min_block_gain": config.min_block_gain,
            "time_limit": config.time_limit, "log_every": config.log_every, "seed": config.seed,
            "solution operators": dict(s.SOLUTION_OPERATOR_PROBABILITIES),
            "block operators": list(s.BLOCK_OPERATORS),
        },
    }


# ---------------------------------------------------------------------------
# Runner and report
# ---------------------------------------------------------------------------


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _command(args):
    try:
        completed = subprocess.run(args, cwd=REPO_DIR, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed


def protected_file_state():
    state = {name: _sha256(os.path.join(REPO_DIR, name)) for name in PROTECTED_FILES}
    git = _command(["git", "diff", "--quiet", "HEAD", "--", *PROTECTED_FILES])
    state["git_clean"] = None if git is None else git.returncode == 0
    return state


def environment_details():
    cpu = platform.processor() or platform.machine()
    brand = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
    if brand is not None and brand.returncode == 0 and brand.stdout.strip():
        cpu = brand.stdout.strip()
    commit = _command(["git", "rev-parse", "--short", "HEAD"])
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "libraries": versions,
        "git_commit": commit.stdout.strip() if commit is not None and commit.returncode == 0 else "unknown",
    }


def run_worker_subprocess(solver, seed, seconds, generation_ceiling, log_dir, observe=True):
    with tempfile.TemporaryDirectory() as directory:
        result_path = os.path.join(directory, "result.json")
        command = [sys.executable, os.path.abspath(__file__), "--worker", solver, "--worker-seed", str(seed),
                   "--seconds", repr(seconds), "--generation-ceiling", str(generation_ceiling),
                   "--result-file", result_path]
        if not observe:
            command.append("--no-observe")
        if log_dir:
            suffix = "" if observe else "_unobserved"
            command += ["--log-file", os.path.join(log_dir, f"{solver}_seed{seed}_g{generation_ceiling}{suffix}.log")]
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
        started = time.monotonic()
        completed = subprocess.run(command, cwd=REPO_DIR, env=env, capture_output=True, text=True,
                                   timeout=min(seconds * 4 + 600, 6 * 3600))
        wall = time.monotonic() - started
        if completed.returncode != 0 or not os.path.exists(result_path):
            return {"solver": solver, "seed": seed, "status": "ERROR", "stop_reason": "worker failed",
                    "error": (completed.stderr or completed.stdout)[-4000:], "process_wall": wall}
        with open(result_path) as handle:
            result = json.load(handle)
    result["process_wall"] = wall
    return result


def neutrality_check(seed, log_dir):
    observed = run_worker_subprocess("old", seed, 1e9, NEUTRALITY_GENERATIONS, log_dir, observe=True)
    unobserved = run_worker_subprocess("old", seed, 1e9, NEUTRALITY_GENERATIONS, log_dir, observe=False)
    digests = (observed.get("population_digest"), unobserved.get("population_digest"))
    return {
        "seed": seed,
        "generations": NEUTRALITY_GENERATIONS,
        "identical": digests[0] is not None and digests[0] == digests[1],
        "digests": digests,
        "errors": [run.get("error") for run in (observed, unobserved) if run.get("error")],
    }


def free_path(path):
    stem, extension = os.path.splitext(path)
    candidate, number = path, 1
    while os.path.exists(candidate):
        candidate = f"{stem}_{number}{extension}"
        number += 1
    return candidate


def _table(headers, rows):
    widths = [max(len(str(item)) for item in column) for column in zip(headers, *rows)]
    line = lambda items: "  ".join(str(item).ljust(width) for item, width in zip(items, widths)).rstrip()
    return [line(headers), line(["-" * width for width in widths]), *[line(row) for row in rows]]


def _value(value):
    return "-" if value is None else value


def _seconds(value):
    return "-" if value is None else f"{value:.1f}"


def outcome_lines(results, input_cost):
    lines = []
    solved = [run for run in results if run.get("replay", {}).get("solved")]
    improved = [run for run in results if run.get("replay") and run["replay"]["cost"] < input_cost]
    errors = [run for run in results if run["status"] == "ERROR"]
    if errors:
        lines.append(f"{len(errors)} run(s) ended in ERROR: "
                     + ", ".join(f"{run['solver']}/seed {run['seed']}" for run in errors) + ".")
    if solved:
        lines.append("Verified solves: " + ", ".join(
            f"{run['solver']} solver seed {run['seed']} ({run['best']['length']} moves)" for run in solved) + ".")
    else:
        lines.append("No run of either solver solved the active cube within its budget.")
    if improved:
        lines.append("Runs whose best verified prefix improved on the input cost "
                     f"H={input_cost}: " + ", ".join(
                         f"{run['solver']} seed {run['seed']} (H={run['replay']['cost']}, "
                         f"{run['best']['length']} moves)" for run in improved) + ".")
    else:
        lines.append(f"No run of either solver improved on the input cost H={input_cost}: every best "
                     "verified attempt is the empty move list.")
    lines.append("These are the only runs made for this report; no seed was rerun, dropped or selected.")
    lines.append("Three seeds per solver describe observed behaviour on one cube. They are not a "
                 "solve-probability estimate.")
    return lines


def render_report(results, context):
    environment = context["environment"]
    new_runs = [run for run in results if run["solver"] == "new"]
    old_runs = [run for run in results if run["solver"] == "old"]
    input_cost = next((run["input_cost"] for run in results if "input_cost" in run), None)
    out = []
    add = out.append
    add("Bounded comparison: cooperative coevolution vs existing prefix GA")
    add("=" * 66)
    add(f"Generated: {context['finished']} (started {context['started']})")
    add(f"Command: {context['command']}")
    add("Specification: new_deap_implementation_specs_fixed.md, section 13 (bounded first-delivery comparison)")
    add("")
    add("1. Outcome")
    add("----------")
    for line in outcome_lines(results, input_cost):
        add(f"- {line}")
    add("")

    add("2. Results per run (endpoint cost/solved status replayed independently from the original cube)")
    add("-" * 96)
    rows = []
    for run in results:
        replay = run.get("replay") or {}
        timing = run.get("timing") or {}
        rows.append([
            "new (coevolution)" if run["solver"] == "new" else "old (prefix GA)",
            run["seed"], run["status"], run.get("stop_reason", "-"),
            replay.get("cost", "-"), "yes" if replay.get("solved") else "no",
            (run.get("best") or {}).get("length", "-"),
            f"{_value(run.get('unique_endpoints'))}/{_value(run.get('population_size'))}",
            _value(run.get("generations_completed")),
            _seconds(timing.get("search")), _seconds(timing.get("setup")),
            f"{timing.get('finalization', 0) * 1000:.1f}ms" if timing else "-",
        ])
    out.extend(_table(["solver", "seed", "status", "stop reason", "best H", "solved", "moves",
                       "unique endpoints", "generations", "search s", "setup s", "finalisation"], rows))
    add("")
    add(f"Input cube cost H (manhattan compatibility heuristic): {input_cost}. Move counts are reduced active")
    add("lengths (a half turn counts as one move). Unique endpoints: distinct endpoint states in the last fully")
    add("evaluated solution population (old solver: distinct best-prefix endpoints of its individuals).")
    add("Search time is measured from immediately before population creation; old-solver search time")
    add("excludes paused harness observation (see section 4).")
    add("")

    add("3. Best verified move sequences")
    add("-------------------------------")
    for run in results:
        best = run.get("best") or {}
        found = best.get("found", {})
        where = (f"generation {found.get('generation')} via {found.get('source')}" if run["solver"] == "new"
                 else f"after {found.get('seconds', 0):.1f}s, evaluation {found.get('evaluation')}, "
                      f"generation {found.get('generation')}")
        if best.get("length") == 0 and not best.get("solved"):
            where = "empty baseline; never improved"
        else:
            where = f"found {where}"
        add(f"{run['solver']} seed {run['seed']}: {best.get('moves', '-')} [{where}]")
        problems = (run.get("replay") or {}).get("problems")
        if problems:
            add(f"  REPLAY PROBLEMS: {'; '.join(problems)}")
        if run.get("error"):
            add(f"  WORKER ERROR: {run['error']}")
    add("")

    add("4. Work counters and timing diagnostics")
    add("---------------------------------------")
    for run in results:
        timing = run.get("timing") or {}
        add(f"{run['solver']} seed {run['seed']}:")
        add(f"  timing: setup {_seconds(timing.get('setup'))}s, search {timing.get('search', 0):.2f}s, "
            f"search wall {timing.get('search_wall', 0):.2f}s, "
            + (f"observation overhead {timing['observation_overhead']:.2f}s (excluded from budget), "
               if "observation_overhead" in timing else "")
            + f"finalisation {timing.get('finalization', 0) * 1000:.1f}ms, process wall {run.get('process_wall', 0):.1f}s")
        for name, value in (run.get("work") or {}).items():
            add(f"  {name}: {value}")
    add("")

    add("5. New solver: block diversity and attributed block gains")
    add("--------------------------------------------------------")
    for run in new_runs:
        blocks = run.get("blocks")
        if not blocks:
            continue
        batch = blocks["last_batch"]
        add(f"seed {run['seed']}: {blocks['distinct_sequences']} distinct sequences in a block population of "
            f"{blocks['population_size']}; origins {blocks['origins']}")
        histogram = " ".join(f"{length}:{count}" for length, count in blocks["length_histogram"].items())
        add(f"  reduced length histogram (length:count): {histogram}")
        if batch:
            add(f"  last completed batch #{batch['batch']} ({batch['contexts']} distinct contexts): useful blocks "
                f"{batch['useful_blocks']}, best mean relative gain {batch['best_mean_gain']:.4f}, best max "
                f"relative gain {batch['best_max_gain']:.4f}, best absolute gain {batch['best_abs_gain']}")
        add(f"  baseline-start attempts {blocks['baseline_start_attempts']}, improvements "
            f"{blocks['baseline_start_improvements']}; targeted appends {blocks['targeted_appends']}, "
            f"exploratory appends {blocks['exploratory_appends']}")
        add(f"  attributed improvements: {blocks['attributed_improvements_total']} across "
            f"{blocks['attributed_block_ids']} block IDs; top: {blocks['top_attributions']}")
        add(f"  block discoveries archived: {blocks['block_discoveries_archived']}; fallbacks: "
            f"{blocks['fallbacks'] or 'none'}")
    add("Block gains are measured on each generation's changing context batch, not on a fixed objective.")
    add("")

    add("6. Protocol and effective configuration")
    add("---------------------------------------")
    protocol = [
        ("Input", "active cube literal from spec section 3.1 (also the last my_cube in solver_deap.py)"),
        ("Seeds", " ".join(str(seed) for seed in context["seeds"])),
        ("Objective", "pure manhattan compatibility state cost for the comparison outcome"),
        ("Solution capacity", f"{SOLUTION_CAPACITY} face-turn genes"),
        ("Solution population", str(SOLUTION_POPULATION)),
        ("Search budget", f"{context['seconds']:g} s monotonic time per run, generation ceiling "
                          f"{context['generation_ceiling']} so the clock binds"),
        ("Execution", "same machine, serial single-process runs, each in a fresh subprocess, "
                      "MPLBACKEND=Agg, PYTHONHASHSEED=0, no GUI plots"),
    ]
    out.extend(_table(["setting", "both solvers"], protocol))
    add("")
    if new_runs and new_runs[0].get("config"):
        add("New solver configuration (identical for every seed except the seed itself):")
        for name, value in new_runs[0]["config"].items():
            if name != "seed":
                add(f"  {name}: {value}")
        add("")
    if old_runs and old_runs[0].get("overrides"):
        add("Old solver: solver_deap.py read from disk and compiled in memory with these AST overrides")
        add(f"(seed shown for {old_runs[0]['seed']}; each run substitutes its own seed). Source sha256 "
            f"{old_runs[0].get('source_sha256')}.")
        for override in old_runs[0]["overrides"]:
            add(f"  - {override}")
        add("  Unchanged: MUTATION_PROB 0.9, CROSSOVER_PROB 0.001, HALL_OF_FAME_SIZE 1, selTournament(5), "
            "mutate_individual_mixed, cxOnePoint,")
        add("    evaluate_individual_manhattan_misplaced with weight_manhattan 1.0 / weight_misplaced 0.0, "
            "verbose logbook output (sent to a log sink).")
        add("Old solver instrumentation rules:")
        for rule in OLD_INSTRUMENTATION_RULES:
            add(f"  - {rule}")
        add("")

    add("7. Remaining differences between the two harnesses")
    add("-------------------------------------------------")
    for line in (
        "The old solver keeps its own fitness: min over prefixes of manhattan + 0*misplaced + length/MAX_GENES, "
        "seeded by the module-global min_moves/best_fitness_ever, which change during the run. The new solver "
        "ranks (unsolved, H, reduced length) with a pure H. Only the replayed endpoint cost/solved status is compared.",
        "The old evaluator applies all 32 genes and computes misplaced_stickers at every prefix even though its "
        "weight is 0; the new solver normalises to the best reduced prefix and caches pure H values.",
        "The old GA uses the module-global random generator; the new solver uses one seeded random.Random per run. "
        "Equal seed numbers do not imply comparable random streams.",
        "Old-solver harness observation (best-prefix replay) is paused out of its budget; the new solver's own "
        "normalisation and verification are part of its search and count against its budget.",
        "The old solver still prints its logbook every generation and print_best every 25 generations (into a "
        "log sink); the new solver reports every 100 generations.",
    ):
        add(f"- {line}")
    add("")

    add("8. Environment and integrity checks")
    add("----------------------------------")
    add(f"Machine: {environment['platform']}; CPU {environment['cpu']} ({environment['cpu_count']} logical cores)")
    add(f"Python: {environment['implementation']} {environment['python']}")
    add("Libraries: " + ", ".join(f"{name} {version}" for name, version in environment["libraries"].items()))
    add(f"Git commit: {environment['git_commit']}")
    before, after = context["protected_before"], context["protected_after"]
    for name in PROTECTED_FILES:
        add(f"{name}: sha256 {before[name]} before, {'unchanged' if before[name] == after[name] else 'CHANGED'} after")
    add(f"git diff HEAD on {', '.join(PROTECTED_FILES)}: "
        f"{'clean' if after['git_clean'] else 'not clean' if after['git_clean'] is False else 'git unavailable'}")
    check = context.get("neutrality")
    if check:
        verdict = "identical" if check["identical"] else "DIFFERENT"
        add(f"Observation neutrality: seed {check['seed']}, {check['generations']} generations with and without "
            f"best-prefix observation -> final populations {verdict} (sha256 {check['digests'][0]})")
        for error in check["errors"]:
            add(f"  neutrality worker error: {error}")
    if context.get("log_dir"):
        add(f"Subprocess logs: {context['log_dir']}")
    add("")
    return "\n".join(out)


def build_parser():
    parser = argparse.ArgumentParser(description="Bounded comparison of the coevolution solver and the prefix GA.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="search budget per run")
    parser.add_argument("--generation-ceiling", type=int, default=GENERATION_CEILING)
    parser.add_argument("--output", default=os.path.join(REPO_DIR, REPORT_FILENAME))
    parser.add_argument("--log-dir", default=None, help="keep each subprocess's stdout log here")
    parser.add_argument("--skip-neutrality-check", action="store_true")
    parser.add_argument("--worker", choices=("new", "old"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--log-file", help=argparse.SUPPRESS)
    parser.add_argument("--no-observe", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.worker == "new":
            result = run_new_worker(args.worker_seed, args.seconds, args.generation_ceiling, args.log_file)
        else:
            result = run_old_worker(args.worker_seed, args.seconds, args.generation_ceiling, args.log_file,
                                    observe=not args.no_observe)
        with open(args.result_file, "w") as handle:
            json.dump(result, handle)
        return 0

    if args.seconds <= 0 or args.generation_ceiling < 1:
        print("--seconds must be positive and --generation-ceiling at least 1", file=sys.stderr)
        return 2
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    context = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["benchmark_coevolution.py", *(argv if argv is not None else sys.argv[1:])]),
        "seeds": args.seeds,
        "seconds": args.seconds,
        "generation_ceiling": args.generation_ceiling,
        "environment": environment_details(),
        "protected_before": protected_file_state(),
        "log_dir": args.log_dir,
    }
    if not args.skip_neutrality_check:
        print(f"Neutrality check: old solver, seed {args.seeds[0]}, {NEUTRALITY_GENERATIONS} generations "
              "with and without observation...", flush=True)
        context["neutrality"] = neutrality_check(args.seeds[0], args.log_dir)
        print(f"  identical final populations: {context['neutrality']['identical']}", flush=True)

    results = []
    for seed in args.seeds:
        for solver in ("new", "old"):
            print(f"Running {solver} solver, seed {seed}, {args.seconds:g}s budget...", flush=True)
            result = run_worker_subprocess(solver, seed, args.seconds, args.generation_ceiling, args.log_dir)
            replay = result.get("replay") or {}
            print(f"  {result['status']} ({result.get('stop_reason')}): best H={replay.get('cost')} "
                  f"moves={(result.get('best') or {}).get('length')} generations="
                  f"{result.get('generations_completed')} wall={result.get('process_wall', 0):.1f}s", flush=True)
            results.append(result)

    context["protected_after"] = protected_file_state()
    context["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = free_path(args.output)
    with open(path, "w") as handle:
        handle.write(render_report(results, context))
    print(f"Report written to {path}")
    unchanged = all(context["protected_before"][name] == context["protected_after"][name] for name in PROTECTED_FILES)
    return 0 if unchanged and all(run["status"] != "ERROR" for run in results) else 3


if __name__ == "__main__":
    sys.exit(main())
