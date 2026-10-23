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
function), `misplaced`, or the weighted `manhattan-misplaced`. Solved means
every sticker matches its face centre, whatever `H` is in use, and every
reported result is replayed from the original cube before it counts.

Unlike the older scripts it has a `main()` guard and no import side effects.
The input is still hardcoded: `default_input_cube()` is the active checkpoint
from `solver_deap.py`. Other cubes can go through the library call
`run_search(SearchConfig(...), cube)`. Main options: `--seed`,
`--time-limit`, `--generations`, the population, capacity and context sizes,
and `--plot` (saves `coevolution_progress.png` without opening a window).
Exit codes: `0` verified solve, `1` budget exhausted, `2` invalid input, `3`
internal or replay error, `130` interrupted.

## Benchmark ([benchmark_coevolution.py](benchmark_coevolution.py))

Runs both DEAP solvers on the active checkpoint for seeds 42, 43 and 44, with
120 s, solution capacity 32 and population 200 per run, serially in fresh
subprocesses. `solver_deap.py` is read from disk and changed in memory only
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
uv run python -m unittest test_solver_deap_coevolution -v
uv run python benchmark_coevolution.py   # ~15 minutes
```

`uv sync` only installs the environment, without starting a solver.

To solve a different scramble, replace the active `Cube(...)` literal in the
script you run, keeping exactly nine stickers of each colour —
`test_cube()` will assert otherwise.

## Limitations

- `solver.py` and `solver_deap.py` use hardcoded inputs and parameters, with
  no arguments or tests. The coevolution solver has a CLI and tests but still
  starts from the hardcoded checkpoint. Dependencies are installed with uv.
- The coevolution solver solves short scrambles (up to about 6 moves in
  testing) but did not get past the active checkpoint's plateau in the
  benchmark.
- Fitness is a heuristic, not an admissible bound: search can stall at local
  optima and solutions are neither guaranteed nor minimal.
- Exhaustive modes materialise every combination in memory/on disk and do
  not scale beyond short sequences; the DEAP run is similarly expensive at
  the shipped population/generation sizes.
