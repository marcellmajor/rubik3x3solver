# Rubik's 3×3 Solver

Experimental Python solvers for a scrambled 3×3 Rubik's cube. Three search
front-ends — a hand-rolled solver, a [DEAP](https://deap.readthedocs.io/)
genetic algorithm, and a DEAP cooperative coevolution of solutions and move
blocks — share one cube model, one move encoding, and heuristic scoring. This
is research code, not a general-purpose solver: inputs are hardcoded, only the
coevolution solver has a CLI, and no search guarantees an optimal (or any)
solution.

## Layout

| File | Role |
|---|---|
| [cube.py](cube.py) | `Cube` state, the six move operations, heuristic counters, terminal rendering, validation. |
| [solution.py](solution.py) | `Solution`: move-sequence ("chromosome") handling and fitness evaluation used by [solver.py](solver.py). |
| [solver.py](solver.py) | Hand-rolled solver: exhaustive combination search (single/multi-threaded, multiprocess, meet-in-the-middle), then a mutation/crossover hill-climb. |
| [solver_deap.py](solver_deap.py) | DEAP genetic algorithm with elitism, algorithm-aware mutation, and a fitness plot. Current, most-tuned entry point. |
| [solver_deap_coevolution.py](solver_deap_coevolution.py) | DEAP cooperative coevolution: solution candidates plus evolving move blocks, with a CLI. |
| [test_solver_deap_coevolution.py](test_solver_deap_coevolution.py) | Deterministic `unittest` suite for the coevolution solver. |
| [benchmark_coevolution.py](benchmark_coevolution.py), [experiment_coevolution.txt](experiment_coevolution.txt) | Bounded comparison of the two DEAP solvers, and its recorded report. |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Project metadata, dependencies, and the lockfile used by [uv](https://docs.astral.sh/uv/). |

## Cube model ([cube.py](cube.py))

A cube is `Cube(top_face, row_of_faces, bottom_face)`: the four side faces in
a list between top and bottom. Each face is a 9-character string in row-major
order (indices `0..8`, centre at `4`) over the colours `Y R W O G B`:

```python
Cube(
                             "BRRBGYGGG",
    ["GBRRRGRRB", "YWWWWWRWW", "OOOOOOOOO", "YGWYYGYYG"],
                             "BBBRBBWYY"
)
```

Moves permute stickers by string slicing; `count` repeats a quarter-turn
`count` times. `clone()` deep-copies, `test_cube()` asserts every face has 9
valid stickers and each colour occurs exactly 9 times, `print_colored()`
draws an unfolded net via `colorama`, and `print_as_python_list()` dumps the
state in the literal form above (handy for pasting a found position back in
as a new starting cube).

Heuristic counters used for scoring: `count_solved_faces`,
`count_solved_edges`, `count_same_as_center` (stickers matching their face
centre), `count_triangles` (rows/columns through the centre matching it), and
`count_pieces_in_place` (staged beginner-method checks — white cross, second
layer, yellow cross/edges/corners — that bail out as soon as a stage fails).

## Move encoding ([solution.py](solution.py))

A move is a 2-character gene: a direction in `R L U D C B` plus a count
`1..3` (quarter-turns, so `3` is one counter-turn). A chromosome is genes
concatenated, e.g. `L3U2C2L1...`. These letters are internal directions, not
standard Singmaster notation:

| Gene | `Cube` method |
|---|---|
| `R` / `L` | `rotate_right` / `rotate_left` |
| `U` / `D` | `rotate_up` / `rotate_down` |
| `C` / `B` | `rotate_cloclkwise` *(sic)* / `rotate_backwards` |

`Solution.calculate_fitness(cube, chromosome, ...)` applies the genes to the
cube **in place** (callers pass a clone), scoring after every gene as

```
pieces_in_place + solved_faces + triangles + same_as_center − step/100
```

and returns `(best_index, fitness)` under one of three `Mode`s: `MAX` (best
prefix), `LAST` (final position), or `CUMULATED` (sum over steps). Reaching
six solved faces returns early with `solved_fitness − step`. `Solution` also
provides random gene generation and point mutation; note
`simplify_chromosome()` currently returns its input unchanged (the merging
logic below the `return` is dead code).

## Hand-rolled solver ([solver.py](solver.py))

`solver.py` first enumerates **all** move combinations up to `N = 6` genes
into `combinations.txt`, then scores them against the hardcoded scramble in
the mode selected by `Processing` (currently `MEET_IN_THE_MIDDLE`):

- `SINGLE_THREAD` — greedily appends the best-scoring combination chunk by
  chunk, re-targeting the "almost complete" colour as fitness saturates.
- `MULTI_THREAD` / `MULTI_PROCESSING` — split the combination list across 8
  threads / 6 processes and keep the best `Mode.LAST` score per worker
  (the threaded variant discards worker results, so it is a benchmark only).
- `MEET_IN_THE_MIDDLE` — applies every combination to a solved cube and
  records `move:state` lines in `combinations_solved.txt`, keeping the
  shortest move per state in a dict. It then applies the same combinations
  to the scramble; on a state match it reverses the solved-side sequence
  (gene order flipped, `1`↔`3`, `2` unchanged), concatenates it after the
  scramble-side sequence, verifies, and exits.

Afterwards a hill-climbing loop runs until near-solved: each round mutates
the incumbent chromosome or crosses it with a fresh random one at the best
index, keeping only improvements, and raises the mutation rate after
50 000 stagnant rounds.

As shipped, `JUST_READING = True`, so it expects `combinations.txt` (and
`combinations_solved.txt` for meet-in-the-middle) to already exist; neither
is in the repo. Flip it to `False` to regenerate them — the combination set
grows as 18^length, so this is slow and memory-hungry.

## DEAP solver ([solver_deap.py](solver_deap.py))

A module-level script (it runs on import; there is no `main()` guard). An
individual is 12 integers (`MAX_GENES`), each indexing the 18 moves
`R1..B3`. Fitness is minimised (`FitnessMin`, weight `-1`).

The active fitness, `evaluate_individual_manhattan_misplaced`, clones the
target cube, applies the moves, and scores **every prefix**: it keeps the
best value of `manhattan + (moves used / MAX_GENES)` (misplaced-sticker
weight is currently `0`), so shorter prefixes win ties. `manhattan_distance`
is a colour-mismatch heuristic, not a true move distance: for each sticker it
adds the grid distance between its own colour's block and its face centre
colour's block on the unfolded net (the sticker's position within the face
cancels out).
Side effects track the global best-ever prefix length (`min_moves`), which
`print_best()` uses to show the effective solution. Simpler alternatives
(misplaced-sticker count, final Manhattan distance) remain registered-out.

Genetic operators:

- **Selection**: tournament (size 5); a 1-member Hall of Fame is re-injected
  each generation by the custom `eaSimpleWithElitism` loop (elitism).
- **Crossover**: one-point, probability `0.001`.
- **Mutation** (probability `0.9`, `mutate_individual_mixed`): 45 % replace a
  uniformly random gene, 45 % replace a gene biased toward the tail of the
  chromosome, 10 % splice in a known CFOP fragment from `translated_algs`
  (beginner layer-by-layer triggers, OLL/PLL examples) translated into the
  internal encoding.

Defaults are population 5000 × 5000 generations. Every 25 generations it
prints the hall-of-fame individual and its cube; at the end it applies the
best individual, prints the resulting cube and misplaced-sticker count, and
plots min/average fitness with `matplotlib` (`plt.show()`).

The starting cube is also hardcoded — and deliberately so: `my_cube` is
reassigned several times to progressively more-solved positions copied from
earlier runs (see the commented "Best individual moves" blocks), so the
script currently continues from the last checkpoint rather than from the
original scramble.

## Cooperative coevolution solver ([solver_deap_coevolution.py](solver_deap_coevolution.py))

A separate DEAP entry point that evolves two populations together:

- **Solution candidates** are move sequences applied to the input cube. Each
  candidate is normalised to its best prefix and stored as exactly that
  reduced sequence, padded with `NOOP`, so the stored moves are the ones that
  produced its score. Candidates rank by `(unsolved, H(endpoint), length)`,
  all minimised.
- **Move blocks** are reusable combinations of up to 16 moves, scored on a
  shared batch of up to 16 solution endpoints ("contexts": the original cube,
  the best elite states, and a random sample of other states). A block is
  useful on a context when some prefix lowers `H` by at least
  `--min-block-gain`, so a block may pass through worse states first.

Each generation keeps the best 20 % of solutions unchanged, replaces the
worst 20 % with empty candidates, and breeds the middle 60 % by appending
or splicing blocks, varying single moves, or crossover. A fixed share of those
offspring always restarts from the empty candidate. The block population keeps
10 % elites, breeds 60 % with seven operators, and draws 30 % fresh blocks,
with a cap on duplicate sequences.

Moves use the same `R L U D C B` letters with counts `1..3`, plus `NOOP`. A
shared simplifier merges same-face turns and commutes opposite faces (`R`/`L`,
`U`/`D`, `C`/`B`), so reported lengths are reduced, not globally minimal.
`--fitness` picks `H`: `manhattan` (default, the heuristic above as a pure
function), `misplaced`, the weighted `manhattan-misplaced`, `cubie`,
`cubie-refined`, `grouped-cubie`, or `edges`. `cubie` is the sum over the 20 corner and edge
cubies of the moves each would need on its own to reach its home slot and
orientation; it and `cubie-refined` reject an input whose stickers do not
form a consistent set of cubies (as does `grouped-cubie`). `cubie-refined` is `64 * cubie + misplaced`:
the cubie cost stays strictly primary (misplaced counts only non-centre
stickers, so it is at most 48 and can never outweigh one cubie unit), and
the sticker count breaks the large ties between states with equal cubie
cost, where selection otherwise has no gradient except length. `edges`
counts the 20 blocks around the centres (8 corners and 12 edges) that do not
show all their correct colours, so it scores whole blocks instead of single
stickers on each side. Solved means every sticker matches its
face centre, whatever `H` is in use, and every reported result is replayed
from the original cube before it counts.

`grouped-cubie` scores `4096 * P + 64 * cubie + misplaced`, where `P` is
the sum of exact move distances for two groups of four corners and three
groups of four edges, including their orientations. Within each piece type,
groups follow the deterministic slot order derived from `cube.py`. Each group
is solved jointly while ignoring the other pieces. The scales preserve that
priority exactly; the sum is a heuristic, not a solution-length lower bound,
and does not guarantee faster solving or eliminate all local minima.
The five pattern tables are built lazily once per process, before the search
timer starts, using all 18 turns (a half turn costs one move). Their direct
indexing uses about 1.6 MiB of distance data, including unused overlap entries;
construction needs additional temporary memory. Nothing is saved to disk.
Scoring decodes each piece once: a lookup supplies its pattern-index
contribution, solo distance, and misplaced-sticker count together.
Existing selection, block scoring, and the default fitness remain unchanged.

Unlike the older scripts it has a `main()` guard and no import side effects.
The input is still hardcoded: `default_input_cube()` is a fully scrambled cube
(`H` = 198 under `manhattan`, cubie cost 32). Other cubes can go through the
library call `run_search(SearchConfig(...), cube)`. Defaults: 1000 solutions,
500 blocks, 10000 generations, solution capacity 64, block capacity 24 and 16
contexts. Main options: `--seed`, `--fitness`, `--time-limit`,
`--generations`, the population, capacity and context sizes, and `--plot`
(saves `coevolution_progress.png` without opening a window). Exit codes: `0`
verified solve, `1` budget exhausted, `2` invalid input, `3` internal or
replay error, `130` interrupted.

### Population diversity and stagnation warnings

After every 50th completed generation the solver measures the solution
population and prints a short summary
([specification](solution_population_diversity_specs.md)). It only measures and
warns; selection, mutation and every other part of the search are unchanged.

- **Reset slots excluded:** the three diversity thresholds use the non-reset
  candidates (elites, offspring and baseline-start offspring, empty ones
  included). The current generation's 20% reset slots, which are intentionally
  identical, are excluded; whole-population values are shown for context only.
- **Warnings** (any one is enough; all active ones are listed together):
  `LOW_UNIQUE_STATES` below 10% distinct endpoint states, `DOMINANT_STATE` one
  state holding over 80%, `LOW_CUBIE_SPREAD` an exact mean below 2 differing
  cubies (position or orientation, out of 20) between candidates, and
  `NO_QUALITY_PROGRESS` 100 or more generations without a better
  `(solved, H)`. A shorter sequence at the same `H` does not count as progress.
- **Low diversity is not the same as a stalled search:** a population can
  converge while approaching a solution, and progress can stall in a diverse
  population, so each warning says which it is. A recovery note appears once
  when no threshold is crossed any more.
- `--diversity-min-unique-ratio`, `--diversity-max-dominance`,
  `--diversity-min-cubie-distance` and `--stagnation-generations` change the
  thresholds (not the 50-generation schedule); `--no-diversity-checks` turns the
  feature off. Each check's record is attached to that generation's history row
  and to `RunResult.diversity_records`.
- The checks add a few milliseconds per checkpoint and count toward
  `--time-limit`, so a time-limited run may complete slightly fewer generations;
  fixed-generation runs produce exactly the same search with the checks on or off.

### Persistent move-block library

The CLI keeps a shared library of up to 100 distinct move blocks across runs,
independent of the starting cube, seed and `--fitness`
([specification](persistent_move_block_library_specs.md)). It lives at
`$XDG_DATA_HOME/rubik3x3solver/move_blocks_v1.json` (else
`~/.local/share/...`); `--block-library PATH` overrides it and
`--no-block-library` turns all library reads, writes, scoring and seeding off,
reproducing the plain search exactly. Library callers opt in with
`SearchConfig(block_library=path)`; it is off by default there.

- **Distinct** means a different 54-sticker permutation: `R1 R1` and `R2`, or
  two spellings of the same algorithm, share one slot, keeping the shortest
  known sequence. Identity blocks are dropped.
- **Best** means the ranking on a fixed, versioned panel of 48 scrambles,
  scored with `64 * cubie + misplaced` whatever the run's objective: more
  solved panel states, then more improved ones, then the larger summed relative
  gain, then fewer moves. The complete block is scored, never its best prefix.
- Candidates are submitted after the first block batch, every 50th generation
  and at the end, plus the exact blocks behind each new best solution. Reference
  scoring counts against the time budget and stops at it without saving
  partial scores.
- A new run seeds at most a quarter of its first block population from the
  library (blocks that fit `--block-capacity`), fills the rest with fresh
  blocks, and scores every block on its own contexts before selection.
- The file is plain JSON, validated in full before use. Saves merge with the
  latest file under an OS lock and replace it atomically. A corrupt,
  incompatible or locked library is reported, left untouched and never changes
  the solver's result.
  A failed load disables further library collection and scoring for that run.
  After timeout or interruption, saves can use already verified cached scores;
  if concurrent additions require fresh reference scoring, the save is deferred
  with a diagnostic and the current disk file is preserved.

## Benchmark ([benchmark_coevolution.py](benchmark_coevolution.py))

Runs both DEAP solvers on the active checkpoint from `solver_deap.py` for
seeds 42, 43 and 44, with 120 s, solution capacity 32 and population 200 per
run, serially in fresh subprocesses. The benchmark keeps its own copy of that
cube and of the spec's original block settings (100 blocks, capacity 16, 16
contexts), so it does not follow the coevolution solver's defaults.
`solver_deap.py` is read from disk and changed in memory only
(seed, `MAX_GENES`, `POPULATION_SIZE`, generation ceiling, no final plot),
plus read-only hooks that enforce the time limit and record its best prefix.
The report goes to `experiment_coevolution.txt`, numbered if that file exists.

In the recorded run ([experiment_coevolution.txt](experiment_coevolution.txt))
neither solver solved or improved the active checkpoint: `H` stayed at 24.
The coevolution solver ran about 9,500 generations per run to the GA's
1,900, but no block of up to 16 moves lowered this near-solved cube's cost.
Three seeds describe behaviour on one cube; they are not a solve rate.

## Running

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
run the solvers from this directory. `uv run` creates `.venv` on first use,
installs the locked dependencies (`colorama`, `deap`, `numpy`, `matplotlib`),
and uses the Python in [`.python-version`](.python-version) (3.12), downloading
that interpreter if it is not already installed.

```bash
uv run python solver_deap.py   # DEAP genetic algorithm
uv run python solver.py        # needs combinations*.txt, see above
uv run python solver_deap_coevolution.py --seed 42   # coevolution; --help lists options
uv run python solver_deap_coevolution.py --fitness cubie-refined --seed 42
uv run python solver_deap_coevolution.py --fitness grouped-cubie --seed 42
uv run python solver_deap_coevolution.py --seed 42 --block-library my_blocks.json
uv run python solver_deap_coevolution.py --seed 42 --no-block-library
uv run python -m unittest test_solver_deap_coevolution -v
uv run python -m unittest test_block_library -v
uv run python -m unittest test_population_diversity -v
uv run python benchmark_coevolution.py   # ~15 minutes
```

`uv sync` only installs the environment, without starting a solver.

To solve a different scramble, replace the active `Cube(...)` literal in the
script you run, keeping exactly nine stickers of each colour —
`test_cube()` will assert otherwise.

## Limitations

- `solver.py` and `solver_deap.py` use hardcoded inputs and parameters, with
  no arguments or tests. The coevolution solver has a CLI and tests but still
  starts from a hardcoded input cube. Dependencies are installed with uv.
- The coevolution solver solves short scrambles (up to about 6 moves in
  testing) but did not get past the active checkpoint's plateau in the
  benchmark. On the scrambled default, with 2000 solutions and 1000 blocks, it
  typically reached `H` 60–80 within 30 generations and then stalled; none of
  those test runs solved it.
- Fitness is a heuristic, not an admissible bound: search can stall at local
  optima and solutions are neither guaranteed nor minimal.
- Exhaustive modes materialise every combination in memory/on disk and do
  not scale beyond short sequences; the DEAP run is similarly expensive at
  the shipped population/generation sizes.
