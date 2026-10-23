# Rubik's 3×3 Solver

Experimental Python solvers for a scrambled 3×3 Rubik's cube. Two search
front-ends — a hand-rolled solver and a [DEAP](https://deap.readthedocs.io/)
genetic algorithm — share one cube model, one move encoding, and heuristic
scoring. This is research code, not a general-purpose solver: inputs are
hardcoded, there is no CLI, and no search guarantees an optimal (or any)
solution.

## Layout

| File | Role |
|---|---|
| [cube.py](cube.py) | `Cube` state, the six move operations, heuristic counters, terminal rendering, validation. |
| [solution.py](solution.py) | `Solution`: move-sequence ("chromosome") handling and fitness evaluation used by [solver.py](solver.py). |
| [solver.py](solver.py) | Hand-rolled solver: exhaustive combination search (single/multi-threaded, multiprocess, meet-in-the-middle), then a mutation/crossover hill-climb. |
| [solver_deap.py](solver_deap.py) | DEAP genetic algorithm with elitism, algorithm-aware mutation, and a fitness plot. Current, most-tuned entry point. |

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
measures each sticker's grid distance to its goal block on the unfolded net.
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

## Running

Requires Python 3 and, for the DEAP solver, third-party packages (there is
no `requirements.txt`):

```bash
pip install colorama deap numpy matplotlib
python solver_deap.py   # DEAP genetic algorithm
python solver.py        # needs combinations*.txt, see above
```

To solve a different scramble, replace the active `Cube(...)` literal in the
script you run, keeping exactly nine stickers of each colour —
`test_cube()` will assert otherwise.

## Limitations

- Hardcoded inputs and parameters; no arguments, tests, or packaging.
- Fitness is a heuristic, not an admissible bound: search can stall at local
  optima and solutions are neither guaranteed nor minimal.
- Exhaustive modes materialise every combination in memory/on disk and do
  not scale beyond short sequences; the DEAP run is similarly expensive at
  the shipped population/generation sizes.
