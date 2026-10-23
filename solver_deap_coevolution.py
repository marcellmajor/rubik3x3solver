"""Cooperative DEAP evolution of cube solutions and reusable move blocks.

Two populations evolve together:

* solution candidates: move sequences applied to the original input cube;
* move blocks: reusable move combinations, scored by how much they improve the
  endpoint states of current solution candidates (a shared "context batch").

Conventions used throughout this module:

* Cube quality is a cost ``H(cube)`` to minimise. An improvement is a strictly
  positive reduction of that cost. Solutions rank lexicographically by
  ``(unsolved_flag, H(endpoint), active_length)``, every component minimised.
* "Minimal length" means reduced by the deterministic move-equivalent
  simplifier below (same-face merging, opposite-face commuting). It is not a
  globally shortest solution.
* The "manhattan" objective is the repository's historical colour/face-offset
  mismatch heuristic. It is not a geometric cube distance and not a lower bound
  on solution length.
* The "cubie" objective sums, over the 20 corner and edge cubies, the moves each
  would need on its own to reach its home slot and orientation.
* Solved detection is independent of the objective: every sticker must equal
  its own face centre.

Usage::

    uv run python solver_deap_coevolution.py --seed 42
    uv run python solver_deap_coevolution.py --fitness misplaced --seed 42

Importing this module runs no search, prints nothing and creates no plots.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Optional

from deap import base, creator, tools

from cube import Cube

# ---------------------------------------------------------------------------
# Configuration defaults (section 10)
# ---------------------------------------------------------------------------

DEFAULT_FITNESS = "manhattan"
DEFAULT_SOLUTION_POPULATION = 200
DEFAULT_BLOCK_POPULATION = 100
DEFAULT_SOLUTION_CAPACITY = 32
DEFAULT_BLOCK_CAPACITY = 16
DEFAULT_CONTEXT_COUNT = 16
DEFAULT_GENERATIONS = 1000
DEFAULT_MIN_BLOCK_GAIN = 1.0
DEFAULT_LOG_EVERY = 10
DEFAULT_MANHATTAN_WEIGHT = 1.0
DEFAULT_MISPLACED_WEIGHT = 1.0

FITNESS_CHOICES = ("manhattan", "misplaced", "manhattan-misplaced", "manhattan-min", "cubie")
INTEGER_VALUED_FITNESS = ("manhattan", "manhattan-min", "misplaced", "cubie")

# Ordinary solution-offspring operators and their probabilities (section 7.5).
SOLUTION_OPERATOR_PROBABILITIES = (
    ("append-block", 0.40),
    ("splice-block", 0.20),
    ("move-variation", 0.25),
    ("crossover", 0.15),
)
MOVE_VARIATIONS = ("insert-moves", "delete-gene", "substitute-gene", "cube-inverse", "span-replace")
BLOCK_OPERATORS = ("insert", "delete", "substitute", "inverse", "concatenate", "span-replace", "crossover")

TOURNAMENT_SIZE = 3
MAX_PAYLOAD_LENGTH = 3
MAX_SPAN_LENGTH = 3
BLOCK_VARIATION_ATTEMPTS = 30
DUPLICATE_RESAMPLE_ATTEMPTS = 30
GAIN_RELATIVE_TOLERANCE = 1e-12
BLOCK_CACHE_BATCHES = 4
COST_CACHE_LIMIT = 1_000_000
PLOT_FILENAME = "coevolution_progress.png"

STATUS_SOLVED = "SOLVED"
STATUS_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
STATUS_INVALID_INPUT = "INVALID_INPUT"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_ERROR = "ERROR"

EXIT_CODES = {
    STATUS_SOLVED: 0,
    STATUS_BUDGET_EXHAUSTED: 1,
    STATUS_INVALID_INPUT: 2,
    STATUS_ERROR: 3,
    STATUS_INTERRUPTED: 130,
}
EXIT_EXPLANATIONS = {
    STATUS_SOLVED: "a solution was found and verified by independent replay",
    STATUS_BUDGET_EXHAUSTED: "the generation or time budget ran out without a verified solve",
    STATUS_INVALID_INPUT: "invalid configuration or input cube",
    STATUS_ERROR: "internal error or replay consistency failure",
    STATUS_INTERRUPTED: "interrupted by SIGINT",
}

# ---------------------------------------------------------------------------
# Errors and search-control signals
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Invalid configuration; reported as INVALID_INPUT (exit 2)."""


class InvalidInputError(ValueError):
    """Invalid input cube; reported as INVALID_INPUT (exit 2)."""


class ConsistencyError(RuntimeError):
    """Stored metadata disagrees with an independent replay (exit 3)."""


class NumericalError(ArithmeticError):
    """A state cost or block gain was not finite (exit 3)."""


class SearchStopped(Exception):
    """Base class for budget-driven stops raised from inside evaluations."""

    status = STATUS_BUDGET_EXHAUSTED


class BudgetExpired(SearchStopped):
    status = STATUS_BUDGET_EXHAUSTED


class SearchInterrupted(SearchStopped):
    status = STATUS_INTERRUPTED


class SolutionFound(Exception):
    """Raised once a feasible solve has been independently verified."""

    def __init__(self, entry):
        super().__init__("verified solve")
        self.entry = entry


# ---------------------------------------------------------------------------
# Move alphabet (section 3.2)
# ---------------------------------------------------------------------------

NOOP = 0
FACE_LETTERS = "RLUDCB"
MOVE_NAMES = ("NOOP",) + tuple(f"{face}{turns}" for face in FACE_LETTERS for turns in (1, 2, 3))
REAL_MOVES = tuple(range(1, len(MOVE_NAMES)))
ALLELE_BY_NAME = {name: allele for allele, name in enumerate(MOVE_NAMES)}

# Repository move letter -> standard face actually turned. Storage order in the
# standard frame is top=U, side0=B, side1=L, side2=F, side3=R, bottom=D.
STANDARD_FACE_FOR_LETTER = {"R": "D", "L": "U", "U": "R", "D": "L", "C": "F", "B": "B"}

# Face index f = 2 * axis + side. Opposite pairs share an axis: (R,L), (U,D), (C,B),
# and the first letter of each pair sorts first inside a canonical same-axis run.
_ALLELE_FACE = (-1,) + tuple(face for face in range(6) for _ in range(3))
_ALLELE_TURNS = (0,) + (1, 2, 3) * 6
_FACE_ROTATIONS = (
    Cube.rotate_right,
    Cube.rotate_left,
    Cube.rotate_up,
    Cube.rotate_down,
    Cube.rotate_cloclkwise,
    Cube.rotate_backwards,
)


def move_allele(face, turns):
    return 3 * face + turns


def validate_alleles(sequence):
    """Return ``sequence`` as a tuple, rejecting anything but integer alleles 0..18."""
    result = tuple(sequence)
    for allele in result:
        if type(allele) is not int or not 0 <= allele < len(MOVE_NAMES):
            raise ValueError(f"invalid move allele {allele!r}")
    return result


def active_moves(sequence):
    """Validated sequence with every NOOP removed."""
    return tuple(allele for allele in validate_alleles(sequence) if allele != NOOP)


def parse_moves(text):
    """Parse space-separated tokens such as ``"R1 L2 C3"``."""
    try:
        return tuple(ALLELE_BY_NAME[token] for token in text.split())
    except KeyError as exc:
        raise ValueError(f"unknown move token {exc.args[0]!r}") from None


def format_moves(sequence):
    moves = active_moves(sequence)
    if not moves:
        return "(empty move list)"
    return " ".join(MOVE_NAMES[allele] for allele in moves)


def inverse_allele(allele):
    turns = _ALLELE_TURNS[allele]
    return move_allele(_ALLELE_FACE[allele], 4 - turns)


def cube_inverse(sequence):
    """Cube inverse: reverse the order and swap 1/3 turn counts (not a plain reversal)."""
    return tuple(inverse_allele(allele) for allele in reversed(active_moves(sequence)))


def pad_chromosome(active, capacity):
    if len(active) > capacity:
        raise ConsistencyError(f"active length {len(active)} exceeds capacity {capacity}")
    return list(active) + [NOOP] * (capacity - len(active))


# ---------------------------------------------------------------------------
# Cube state helpers
# ---------------------------------------------------------------------------

STICKER_COUNT = 54
FACE_STARTS = (0, 9, 18, 27, 36, 45)
CUBE_COLORS = ("Y", "R", "W", "O", "G", "B")


def state_key(cube):
    """Immutable 54-sticker key: top, the four side faces, bottom."""
    return cube.top_face + "".join(cube.row_of_faces) + cube.bottom_face


def cube_from_key(key):
    return Cube(key[0:9], [key[9:18], key[18:27], key[27:36], key[36:45]], key[45:54])


def is_solved_key(key):
    """Every sticker equals its own face centre (independent of any objective)."""
    for start in FACE_STARTS:
        if key[start:start + 9] != key[start + 4] * 9:
            return False
    return True


def copy_cube(cube):
    """Independent copy without ``deepcopy``.

    cube.py moves only rebind face strings and list items (strings are
    immutable), so copying the face list is enough.
    """
    return Cube(cube.top_face, list(cube.row_of_faces), cube.bottom_face)


def apply_move(cube, allele, counters=None):
    if allele == NOOP:
        return
    turns = _ALLELE_TURNS[allele]
    _FACE_ROTATIONS[_ALLELE_FACE[allele]](cube, turns)
    if counters is not None:
        counters.quarter_turns += turns
        counters.move_applications += 1


def replay(start_cube, sequence, counters=None):
    """Apply ``sequence`` to an independent copy of ``start_cube`` with cube.py's rotations."""
    cube = copy_cube(start_cube)
    if counters is not None:
        counters.cube_clones += 1
    for allele in active_moves(sequence):
        apply_move(cube, allele, counters)
    return cube


def _build_move_tables():
    """Sticker permutation of every allele, taken from cube.py's own rotations.

    Applying a move to a uniquely labelled cube shows where each sticker comes
    from; ``table(key)`` then returns the moved key's stickers in order.
    """
    labels = "".join(chr(ord("0") + index) for index in range(STICKER_COUNT))
    tables = [None]
    for allele in REAL_MOVES:
        cube = cube_from_key(labels)
        apply_move(cube, allele)
        tables.append(itemgetter(*(labels.index(sticker) for sticker in state_key(cube))))
    return tuple(tables)


_MOVE_TABLES = _build_move_tables()


def apply_move_to_key(key, allele, counters=None):
    """Return the state key after one move; same result as :func:`apply_move` on a cube."""
    if allele == NOOP:
        return key
    if counters is not None:
        counters.quarter_turns += _ALLELE_TURNS[allele]
        counters.move_applications += 1
    return "".join(_MOVE_TABLES[allele](key))


def replay_key(start_key, sequence, counters=None):
    """State key after applying ``sequence`` to ``start_key`` through the move tables."""
    key = start_key
    for allele in active_moves(sequence):
        key = apply_move_to_key(key, allele, counters)
    return key


def default_input_cube():
    """The active checkpoint from ``solver_deap.py`` (its last ``my_cube`` assignment)."""
    return Cube(
                                   "GGGWGGGOG",
        ["RWRRRGRRR", "WRWRWWWWW", "OGOOOOOOO", "YYYYYYYYY"],
                                   "BBBBBBBBB",
    )


def validate_input_cube(cube):
    """Check face shapes, colours, counts and distinct centres; return the state key.

    Passing these checks does not prove the cube is physically solvable.
    """
    rows = getattr(cube, "row_of_faces", None)
    if not isinstance(rows, (list, tuple)) or len(rows) != 4:
        raise InvalidInputError("the cube must have exactly four side faces")
    faces = [cube.top_face, *rows, cube.bottom_face]
    for index, face in enumerate(faces):
        if not isinstance(face, str) or len(face) != 9:
            raise InvalidInputError(f"face {index} must be a string of 9 stickers")
        for sticker in face:
            if sticker not in CUBE_COLORS:
                raise InvalidInputError(f"face {index} contains invalid colour {sticker!r}")
    key = "".join(faces)
    counts = Counter(key)
    for color in CUBE_COLORS:
        if counts[color] != 9:
            raise InvalidInputError(f"colour {color} appears {counts[color]} times, expected 9")
    centers = [key[start + 4] for start in FACE_STARTS]
    if len(set(centers)) != 6:
        raise InvalidInputError(f"face centres must be six distinct colours, got {''.join(centers)}")
    return key


# ---------------------------------------------------------------------------
# State objectives (section 5.1)
# ---------------------------------------------------------------------------

COLOR_OFFSETS = {"G": (0, 0), "R": (0, 3), "W": (0, 6), "O": (3, 0), "Y": (3, 3), "B": (3, 6)}
_COLOR_DISTANCE = {
    center: {
        sticker: abs(center_row - sticker_row) + abs(center_col - sticker_col)
        for sticker, (sticker_row, sticker_col) in COLOR_OFFSETS.items()
    }
    for center, (center_row, center_col) in COLOR_OFFSETS.items()
}
MAX_MANHATTAN_COST = STICKER_COUNT * max(max(row.values()) for row in _COLOR_DISTANCE.values())
MAX_MISPLACED_COST = STICKER_COUNT


def misplaced_cost(key):
    """Number of stickers that differ from their own face centre."""
    total = 0
    for start in FACE_STARTS:
        face = key[start:start + 9]
        total += 9 - face.count(face[4])
    return total


def manhattan_cost(key):
    """Historical compatibility heuristic, as a pure state function.

    For every sticker, the Manhattan distance between the colour offset of its
    face centre and the offset of its own colour (local sticker indices cancel).
    """
    total = 0
    for start in FACE_STARTS:
        face = key[start:start + 9]
        total += sum(map(_COLOR_DISTANCE[face[4]].__getitem__, face))
    return total


def _build_cubie_tables():
    """Per-slot lookup from the face indices on a slot's stickers to that cubie's distance home.

    Cubie slots are the groups of sticker positions moved by exactly the same
    turns (8 corners, 12 edges). Each cubie's distance is a breadth-first search
    over the 18 moves of where its stickers can be, so it covers orientation.
    """
    positions = tuple(range(STICKER_COUNT))
    destination = {}
    for allele in REAL_MOVES:
        target = [0] * STICKER_COUNT
        for new_position, old_position in enumerate(_MOVE_TABLES[allele](positions)):
            target[old_position] = new_position
        destination[allele] = target
    groups = defaultdict(list)
    for position in positions:
        moved_by = frozenset(allele for allele in REAL_MOVES if destination[allele][position] != position)
        if moved_by:
            groups[moved_by].append(position)
    slots = [tuple(group) for group in groups.values()]
    if sorted(len(slot) for slot in slots) != [2] * 12 + [3] * 8:
        raise ConsistencyError("move tables do not split into 8 corners and 12 edges")
    slot_index = {frozenset(slot): index for index, slot in enumerate(slots)}
    tables = [{} for _ in slots]
    for home in slots:
        labels = [str(position // 9) for position in home]
        distance = {home: 0}
        queue = deque([home])
        while queue:
            placement = queue.popleft()
            for allele in REAL_MOVES:
                moved = tuple(destination[allele][position] for position in placement)
                if moved not in distance:
                    distance[moved] = distance[placement] + 1
                    queue.append(moved)
        for placement, steps in distance.items():
            index = slot_index[frozenset(placement)]
            label_at = dict(zip(placement, labels))
            tables[index]["".join(label_at[position] for position in slots[index])] = steps
    return tuple((itemgetter(*slot), table) for slot, table in zip(slots, tables))


_CUBIE_TABLES = _build_cubie_tables()
_FACE_INDEX_TRANSLATIONS = {}


def _face_indices(key):
    """The key with every sticker replaced by the index (0-5) of the face whose centre it matches."""
    centres = key[4] + key[13] + key[22] + key[31] + key[40] + key[49]
    translation = _FACE_INDEX_TRANSLATIONS.get(centres)
    if translation is None:
        translation = _FACE_INDEX_TRANSLATIONS[centres] = str.maketrans(centres, "012345")
    return key.translate(translation)


def cubie_cost(key):
    """Sum over the 20 cubies of the moves each would need, on its own, to reach its home slot and orientation."""
    faces = _face_indices(key)
    total = 0
    try:
        for getter, distances in _CUBIE_TABLES:
            total += distances["".join(getter(faces))]
    except KeyError:
        raise ValueError("the stickers do not form a consistent set of cubies") from None
    return total


def validate_cubie_input(key):
    """Require every corner and edge slot to hold a distinct real cubie (needed by the cubie objective)."""
    try:
        cubie_cost(key)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from None
    faces = _face_indices(key)
    cubies = {"".join(sorted(getter(faces))) for getter, _ in _CUBIE_TABLES}
    if len(cubies) != len(_CUBIE_TABLES):
        raise InvalidInputError("the stickers repeat a cubie, so they are not a consistent set of cubies")


@dataclass(frozen=True)
class Objective:
    name: str
    state_function: str
    manhattan_weight: Optional[float] = None
    misplaced_weight: Optional[float] = None

    @property
    def integer_valued(self):
        return self.name in INTEGER_VALUED_FITNESS

    @property
    def identity(self):
        return (self.state_function, self.manhattan_weight, self.misplaced_weight)

    def cost(self, key):
        if self.state_function == "manhattan":
            return manhattan_cost(key)
        if self.state_function == "misplaced":
            return misplaced_cost(key)
        if self.state_function == "cubie":
            return cubie_cost(key)
        value = self.manhattan_weight * manhattan_cost(key) + self.misplaced_weight * misplaced_cost(key)
        if not math.isfinite(value):
            raise NumericalError(f"non-finite state cost {value!r}")
        return value

    def describe(self):
        if self.state_function == "manhattan-misplaced":
            return (f"{self.name}: {self.manhattan_weight!r} * manhattan + "
                    f"{self.misplaced_weight!r} * misplaced")
        if self.state_function == "misplaced":
            return f"{self.name}: stickers differing from their face centre"
        if self.state_function == "cubie":
            return f"{self.name}: sum over the 20 cubies of each one's move distance to its home slot and orientation"
        alias =" (compatibility alias of manhattan)" if self.name == "manhattan-min" else ""
        return (f"{self.name}{alias}: colour/face-offset mismatch heuristic "
                "(not a geometric distance, not a lower bound)")


def make_objective(name, manhattan_weight=None, misplaced_weight=None):
    """Resolve an objective name; explicit weights are allowed only for the mixed objective."""
    if name not in FITNESS_CHOICES:
        raise ConfigError(f"unknown fitness {name!r}; choose from {', '.join(FITNESS_CHOICES)}")
    if name != "manhattan-misplaced":
        if manhattan_weight is not None or misplaced_weight is not None:
            raise ConfigError("--manhattan-weight/--misplaced-weight are valid only with "
                              "--fitness manhattan-misplaced")
        return Objective(name, name if name in ("misplaced", "cubie") else "manhattan")
    weights = []
    for label, value, default in (
        ("manhattan", manhattan_weight, DEFAULT_MANHATTAN_WEIGHT),
        ("misplaced", misplaced_weight, DEFAULT_MISPLACED_WEIGHT),
    ):
        try:
            weight = float(default if value is None else value)
        except (TypeError, ValueError, OverflowError):
            raise ConfigError(f"{label} weight must be a number") from None
        if not math.isfinite(weight) or weight < 0:
            raise ConfigError(f"{label} weight must be finite and nonnegative, got {value!r}")
        weights.append(weight)
    if not any(weight > 0 for weight in weights):
        raise ConfigError("at least one objective weight must be strictly positive")
    bound = weights[0] * MAX_MANHATTAN_COST + weights[1] * MAX_MISPLACED_COST
    if not math.isfinite(bound):
        raise ConfigError("objective weights overflow the cost calculation")
    return Objective(name, "manhattan-misplaced", weights[0], weights[1])


def gain_qualifies(delta, min_block_gain, integer_valued):
    """Usefulness threshold for a non-solving block prefix (section 8.3)."""
    if not math.isfinite(delta):
        raise NumericalError(f"non-finite block gain {delta!r}")
    if integer_valued:
        return delta > 0 and delta >= min_block_gain
    return delta > 0 and (
        delta >= min_block_gain
        or math.isclose(delta, min_block_gain, rel_tol=GAIN_RELATIVE_TOLERANCE, abs_tol=0.0)
    )


# ---------------------------------------------------------------------------
# Move-equivalent simplification (section 4)
# ---------------------------------------------------------------------------


class MoveReducer:
    """Incremental stack-of-runs reducer shared by both populations.

    Each run is ``(axis, first_face_turns, second_face_turns)`` for one
    contiguous same-axis stretch; adjacent runs always differ in axis. Its
    expansion is the canonical reduced sequence (R before L, U before D,
    C before B). ``length`` is the reduced active length after every push.
    """

    __slots__ = ("runs", "length")

    def __init__(self, runs=(), length=None):
        self.runs = list(runs)
        if length is None:
            length = sum((first != 0) + (second != 0) for _, first, second in self.runs)
        self.length = length

    @classmethod
    def from_moves(cls, sequence):
        reducer = cls()
        for allele in sequence:
            reducer.push(allele)
        return reducer

    def copy(self):
        return MoveReducer(self.runs, self.length)

    def push(self, allele):
        if allele == NOOP:
            return
        face = _ALLELE_FACE[allele]
        turns = _ALLELE_TURNS[allele]
        axis = face >> 1
        runs = self.runs
        if runs and runs[-1][0] == axis:
            _, first, second = runs[-1]
            before = (first != 0) + (second != 0)
            if face & 1:
                second = (second + turns) & 3
            else:
                first = (first + turns) & 3
            after = (first != 0) + (second != 0)
            if after:
                runs[-1] = (axis, first, second)
            else:
                runs.pop()
            self.length += after - before
        else:
            runs.append((axis, 0, turns) if face & 1 else (axis, turns, 0))
            self.length += 1

    def extends(self, allele):
        """True when pushing ``allele`` lengthens the reduced sequence by exactly one."""
        face = _ALLELE_FACE[allele]
        runs = self.runs
        if not runs or runs[-1][0] != face >> 1:
            return True
        return runs[-1][1 + (face & 1)] == 0

    def moves(self):
        result = []
        for axis, first, second in self.runs:
            if first:
                result.append(move_allele(2 * axis, first))
            if second:
                result.append(move_allele(2 * axis + 1, second))
        return tuple(result)


def simplify(sequence):
    """Deterministic, idempotent, permutation-preserving reduction without padding."""
    reducer = MoveReducer()
    for allele in validate_alleles(sequence):
        reducer.push(allele)
    return reducer.moves()


def _canonical_extension(reducer, allele):
    """True when appending ``allele`` to a reduced sequence keeps it reduced as written."""
    face = _ALLELE_FACE[allele]
    runs = reducer.runs
    if not runs or runs[-1][0] != face >> 1:
        return True
    return face & 1 == 1 and runs[-1][2] == 0


def iter_canonical_sequences(length, check=None):
    """Yield every reduced sequence of exactly ``length`` moves in allele-lexicographic order."""
    prefix = []

    def extend(reducer):
        if check is not None:
            check()
        if len(prefix) == length:
            yield tuple(prefix)
            return
        for allele in REAL_MOVES:
            if _canonical_extension(reducer, allele):
                child = reducer.copy()
                child.push(allele)
                prefix.append(allele)
                yield from extend(child)
                prefix.pop()

    yield from extend(MoveReducer())


def reduced_length_schedule(slots, capacity, rng):
    """Shuffled cycles of target reduced lengths 1..capacity, truncated to ``slots``."""
    schedule = []
    while len(schedule) < slots:
        cycle = list(range(1, capacity + 1))
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:slots]


def generate_reduced_block(target_length, rng):
    """Random reduced block whose simplified length is exactly ``target_length``."""
    reducer = MoveReducer()
    for _ in range(target_length):
        allowed = [allele for allele in REAL_MOVES if reducer.extends(allele)]
        reducer.push(allowed[rng.randrange(len(allowed))])
    moves = reducer.moves()
    if len(moves) != target_length:
        raise ConsistencyError("fresh block generator missed its target length")
    return moves


# ---------------------------------------------------------------------------
# Variation primitives (sections 7.5 and 8.5); all randomness comes from ``rng``
# ---------------------------------------------------------------------------


def tournament_select(candidates, key, rng, size=TOURNAMENT_SIZE):
    """Size-``size`` tournament with replacement, minimising ``key``.

    Ties among the sampled occurrences are broken uniformly with ``rng``;
    neither position nor ID acts as a hidden tie-break.
    """
    draws = [candidates[rng.randrange(len(candidates))] for _ in range(size)]
    keys = [key(candidate) for candidate in draws]
    best = min(keys)
    tied = [candidate for candidate, value in zip(draws, keys) if value == best]
    if len(tied) == 1:
        return tied[0]
    return tied[rng.randrange(len(tied))]


def random_payload(rng, length=None):
    if length is None:
        length = rng.randint(1, MAX_PAYLOAD_LENGTH)
    return tuple(REAL_MOVES[rng.randrange(len(REAL_MOVES))] for _ in range(length))


def choose_span(length, rng):
    """Span length uniform in 1..min(3, length), then start uniform among legal starts."""
    span = rng.randint(1, min(MAX_SPAN_LENGTH, length))
    start = rng.randint(0, length - span)
    return start, span


def op_insert_moves(sequence, rng):
    payload = random_payload(rng)
    position = rng.randint(0, len(sequence))
    return sequence[:position] + payload + sequence[position:]


def op_delete_gene(sequence, rng):
    index = rng.randrange(len(sequence))
    return sequence[:index] + sequence[index + 1:]


def op_delete_span(sequence, rng):
    start, span = choose_span(len(sequence), rng)
    return sequence[:start] + sequence[start + span:]


def op_substitute(sequence, rng):
    """Replace one gene with a uniformly chosen *different* allele from 0..18."""
    index = rng.randrange(len(sequence))
    replacement = rng.randrange(len(MOVE_NAMES) - 1)
    if replacement >= sequence[index]:
        replacement += 1
    return sequence[:index] + (replacement,) + sequence[index + 1:]


def op_span_replace(sequence, rng):
    start, span = choose_span(len(sequence), rng)
    return sequence[:start] + random_payload(rng) + sequence[start + span:]


def op_crossover(first, second, rng):
    left = rng.randint(0, len(first))
    right = rng.randint(0, len(second))
    return first[:left] + second[right:]


def op_insert_block(sequence, block, rng):
    position = rng.randint(0, len(sequence))
    return sequence[:position] + block + sequence[position:]


def op_replace_span_with_block(sequence, block, rng):
    start, span = choose_span(len(sequence), rng)
    return sequence[:start] + block + sequence[start + span:]


def baseline_start_quota(offspring_count):
    """20% of the offspring slots, rounded to nearest, at least one (section 7.2)."""
    return max(1, (offspring_count + 2) // 5)


def validate_operator_probabilities(probabilities=SOLUTION_OPERATOR_PROBABILITIES):
    total = 0.0
    for name, probability in probabilities:
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or probability < 0:
            raise ConfigError(f"operator probability for {name} must be finite and nonnegative")
        total += probability
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ConfigError(f"operator probabilities must sum to 1, got {total!r}")


# ---------------------------------------------------------------------------
# Budget, counters and cached costs
# ---------------------------------------------------------------------------


class Budget:
    """Monotonic time limit plus a SIGINT flag, checked inside evaluations."""

    def __init__(self, time_limit=None, clock=time.monotonic):
        self.time_limit = time_limit
        self.clock = clock
        self.started_at = None
        self.deadline = None
        self.interrupt_requested = False

    def start(self):
        self.started_at = self.clock()
        if self.time_limit is not None:
            self.deadline = self.started_at + self.time_limit

    def elapsed(self):
        return 0.0 if self.started_at is None else self.clock() - self.started_at

    def request_interrupt(self):
        self.interrupt_requested = True

    def check(self):
        if self.interrupt_requested:
            raise SearchInterrupted("SIGINT received")
        if self.deadline is not None and self.clock() >= self.deadline:
            raise BudgetExpired("time limit reached")


@dataclass
class SearchCounters:
    quarter_turns: int = 0
    move_applications: int = 0
    cube_clones: int = 0
    cost_computations: int = 0
    cost_cache_hits: int = 0
    normalizations: int = 0
    independent_replays: int = 0
    block_sequence_evaluations: int = 0
    block_evaluation_cache_hits: int = 0
    solution_capacity_rejections: int = 0
    block_oversized_rejections: int = 0
    block_prefix_capacity_exclusions: int = 0
    no_change_offspring: int = 0
    unchanged_block_proposals: int = 0
    duplicate_block_retries: int = 0
    target_length_fallbacks: int = 0
    baseline_start_attempts: int = 0
    baseline_start_improvements: int = 0
    targeted_appends: int = 0
    exploratory_appends: int = 0
    archive_updates: int = 0
    block_discoveries_archived: int = 0
    block_fallbacks: Counter = field(default_factory=Counter)
    attributed_improvements: Counter = field(default_factory=Counter)


class CostCache:
    """Memoised pure state cost; caching never changes a result."""

    def __init__(self, objective, counters, max_entries=COST_CACHE_LIMIT):
        self.objective = objective
        self.counters = counters
        self.max_entries = max_entries
        self._values = {}

    def __call__(self, key):
        value = self._values.get(key)
        if value is not None:
            self.counters.cost_cache_hits += 1
            return value
        value = self.objective.cost(key)
        if not math.isfinite(value):
            raise NumericalError(f"non-finite state cost {value!r}")
        self.counters.cost_computations += 1
        if len(self._values) >= self.max_entries:
            self._values.clear()
        self._values[key] = value
        return value


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

SOLUTION_FITNESS_NAME = "CoevolutionSolutionFitness"
SOLUTION_INDIVIDUAL_NAME = "CoevolutionSolution"
BLOCK_FITNESS_NAME = "CoevolutionBlockFitness"
BLOCK_INDIVIDUAL_NAME = "CoevolutionBlock"


def ensure_deap_types():
    """Create the DEAP fitness/individual classes once; safe to call repeatedly."""
    if not hasattr(creator, SOLUTION_FITNESS_NAME):
        # (unsolved_flag, H(endpoint), active_length), all minimised.
        creator.create(SOLUTION_FITNESS_NAME, base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, SOLUTION_INDIVIDUAL_NAME):
        creator.create(SOLUTION_INDIVIDUAL_NAME, list, fitness=getattr(creator, SOLUTION_FITNESS_NAME))
    if not hasattr(creator, BLOCK_FITNESS_NAME):
        # (solved contexts, useful contexts, mean rel. gain, max rel. gain, -length), all maximised.
        creator.create(BLOCK_FITNESS_NAME, base.Fitness, weights=(1.0, 1.0, 1.0, 1.0, 1.0))
    if not hasattr(creator, BLOCK_INDIVIDUAL_NAME):
        creator.create(BLOCK_INDIVIDUAL_NAME, list, fitness=getattr(creator, BLOCK_FITNESS_NAME))
    return getattr(creator, SOLUTION_INDIVIDUAL_NAME), getattr(creator, BLOCK_INDIVIDUAL_NAME)


@dataclass(frozen=True)
class NormalizedCandidate:
    active: tuple
    endpoint: str
    cost: float
    solved: bool
    capacity_rejected: bool = False

    @property
    def rank(self):
        return (0 if self.solved else 1, self.cost, len(self.active))


@dataclass(frozen=True)
class ArchiveEntry:
    active: tuple
    endpoint: str
    cost: float
    solved: bool
    uid: Optional[int]
    generation: int
    source: str
    block_id: Optional[int] = None

    @property
    def rank(self):
        return (0 if self.solved else 1, self.cost, len(self.active))


@dataclass(frozen=True)
class Context:
    context_id: int
    key: str
    source: tuple
    source_uid: Optional[int]
    group: str
    cost: float
    solved: bool
    source_runs: tuple = field(repr=False)
    source_length: int = 0


@dataclass(frozen=True)
class ContextBatch:
    serial: int
    identity: tuple
    contexts: tuple


@dataclass(frozen=True)
class PrefixOutcome:
    k: int
    gain: float
    relative_gain: float
    solved: bool
    endpoint: str
    endpoint_cost: float
    representative_length: int
    representative_feasible: bool


@dataclass(frozen=True)
class ContextOutcome:
    context_id: int
    best: Optional[PrefixOutcome]
    qualifying: tuple


@dataclass(frozen=True)
class BlockEvaluation:
    batch_serial: int
    outcomes: tuple
    solved_contexts: int
    useful_contexts: int
    mean_relative_gain: float
    max_relative_gain: float
    active_length: int

    @property
    def score(self):
        return (self.solved_contexts, self.useful_contexts, self.mean_relative_gain,
                self.max_relative_gain, -self.active_length)

    @property
    def useful(self):
        return self.useful_contexts > 0


def _prefix_preference(solved, gain, complete_length, k):
    """Smaller is better: solved first, greater gain, shorter complete candidate, shorter prefix."""
    return (0 if solved else 1, -gain, complete_length, k)


def _solution_sort_key(individual):
    return (individual.rank, individual.uid)


def _block_sort_key(block):
    return (tuple(-value for value in block.evaluation.score), block.uid)


def _block_tournament_key(block):
    return tuple(-value for value in block.evaluation.score)


_SOLUTION_FIELDS = ("uid", "active", "endpoint", "cost", "solved", "rank", "origin", "operator",
                    "parents", "block_id", "objective_identity", "input_key", "created_generation")
_BLOCK_FIELDS = ("uid", "active", "origin", "operator", "parents", "fallback_reason",
                 "target_length", "created_generation")


def select_context_states(ranked_population, context_count, rng, original_key):
    """Choose the frozen context states for one generation (section 8.2).

    ``ranked_population`` is sorted best-first by (fitness, ID). Returns a list of
    ``(state_key, source_sequence, source_uid, group)``; the original state is first.
    """
    count = len(ranked_population)
    fifth = count // 5
    pool = ranked_population[:count - fifth]
    elites = ranked_population[:fifth]

    sources = {}
    for individual in pool:
        if individual.endpoint == original_key:
            continue
        candidate = (len(individual.active), tuple(individual.active), individual.uid)
        current = sources.get(individual.endpoint)
        if current is None or candidate < current:
            sources[individual.endpoint] = candidate

    elite_order = {}
    for individual in elites:
        if individual.endpoint == original_key:
            continue
        candidate = (individual.rank, individual.uid)
        current = elite_order.get(individual.endpoint)
        if current is None or candidate < current:
            elite_order[individual.endpoint] = candidate

    elite_keys = sorted(elite_order, key=elite_order.__getitem__)
    non_elite_keys = sorted(key for key in sources if key not in elite_order)

    remaining = context_count - 1
    chosen_elite = elite_keys[:(remaining + 1) // 2]
    non_elite_quota = remaining // 2
    chosen_non_elite = rng.sample(non_elite_keys, min(non_elite_quota, len(non_elite_keys)))
    shortfall = remaining - len(chosen_elite) - len(chosen_non_elite)
    chosen_fill = []
    if shortfall > 0:
        taken = set(chosen_elite) | set(chosen_non_elite)
        leftovers = sorted(key for key in sources if key not in taken)
        chosen_fill = rng.sample(leftovers, min(shortfall, len(leftovers)))

    states = [(original_key, (), None, "original")]
    for group, keys in (("elite", chosen_elite), ("non-elite", chosen_non_elite), ("fill", chosen_fill)):
        for key in keys:
            _, source, uid = sources[key]
            states.append((key, source, uid, group))
    return states


# ---------------------------------------------------------------------------
# Search configuration and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    fitness: str = DEFAULT_FITNESS
    manhattan_weight: Optional[float] = None
    misplaced_weight: Optional[float] = None
    solution_population: int = DEFAULT_SOLUTION_POPULATION
    block_population: int = DEFAULT_BLOCK_POPULATION
    solution_capacity: int = DEFAULT_SOLUTION_CAPACITY
    block_capacity: int = DEFAULT_BLOCK_CAPACITY
    contexts: int = DEFAULT_CONTEXT_COUNT
    generations: int = DEFAULT_GENERATIONS
    min_block_gain: float = DEFAULT_MIN_BLOCK_GAIN
    seed: Optional[int] = None
    time_limit: Optional[float] = None
    log_every: int = DEFAULT_LOG_EVERY
    plot: bool = False


def _require_int(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")


def validate_config(config):
    """Validate every option before a search starts; return the resolved objective."""
    _require_int("solution population", config.solution_population, 5)
    if config.solution_population % 5:
        raise ConfigError("solution population must be divisible by 5")
    _require_int("block population", config.block_population, 10)
    if config.block_population % 10:
        raise ConfigError("block population must be divisible by 10")
    _require_int("solution capacity", config.solution_capacity, 1)
    _require_int("block capacity", config.block_capacity, 1)
    if config.block_capacity > config.solution_capacity:
        raise ConfigError("block capacity must not exceed solution capacity")
    _require_int("contexts", config.contexts, 1)
    _require_int("log interval", config.log_every, 1)
    _require_int("generations", config.generations, 0)
    if config.seed is not None and type(config.seed) is not int:
        raise ConfigError("seed must be an integer")
    gain = config.min_block_gain
    if not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
        raise ConfigError(f"minimum block gain must be finite and strictly positive, got {gain!r}")
    limit = config.time_limit
    if limit is not None and (not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0):
        raise ConfigError(f"time limit must be finite and strictly positive, got {limit!r}")
    validate_operator_probabilities()
    return make_objective(config.fitness, config.manhattan_weight, config.misplaced_weight)


def generate_seed():
    return random.SystemRandom().randrange(2 ** 32)


@dataclass
class RunResult:
    status: str
    reason: str
    archive: ArchiveEntry
    seed: int
    input_key: str
    objective: Objective
    solution_capacity: int
    completed_generations: int
    partial_generation: Optional[int]
    partial_phase: Optional[str]
    search_seconds: float
    counters: SearchCounters
    history: list
    block_population_created: bool


@dataclass
class FinalReport:
    status: str
    moves: tuple
    cube: Optional[Cube]
    cost: Optional[float]
    solved: Optional[bool]
    problems: list


# ---------------------------------------------------------------------------
# The cooperative search
# ---------------------------------------------------------------------------


class CoevolutionRun:
    """One reproducible run: owns its RNG, counters, archive and both populations."""

    def __init__(self, config, input_key, objective, seed, *, budget=None, log=None):
        self.config = config
        self.objective = objective
        self.seed = seed
        self.rng = random.Random(seed)
        self.budget = budget if budget is not None else Budget(config.time_limit)
        self.log = log if log is not None else _print_flush
        self.counters = SearchCounters()
        self.cost = CostCache(objective, self.counters)
        self.input_key = input_key
        self._original = cube_from_key(input_key)
        self.solution_capacity = config.solution_capacity
        self.block_capacity = config.block_capacity
        self.max_block_copies = max(2, math.ceil(config.block_population / 18))
        self.SolutionClass, self.BlockClass = ensure_deap_types()

        self._next_uid = 0
        self._batch_serial = 0
        self._block_cache = {}
        self._block_index = {}
        self._block_representatives = []
        self._context_sources = {}
        self._block_info = {}
        self.batch = None
        self.population = None
        self.blocks = None
        self.generation_in_progress = 0
        self.completed_generation = 0
        self.partial_generation = None
        self.partial_phase = None
        self.last_improvement_generation = 0
        self.last_block_summary = None
        self.history = tools.Logbook()

        self._last_report_marker = None
        if objective.state_function == "cubie":
            validate_cubie_input(input_key)
        baseline = self.normalize_solution((), check_budget=False)
        self.baseline = baseline
        self.baseline_rank = baseline.rank
        self.archive = ArchiveEntry(baseline.active, baseline.endpoint, baseline.cost,
                                    baseline.solved, None, 0, "empty baseline")

    # -- identities and records -------------------------------------------------

    def new_uid(self):
        self._next_uid += 1
        return self._next_uid

    def make_solution(self, candidate, *, origin, operator=None, parents=(), block_id=None, uid=None):
        individual = self.SolutionClass(pad_chromosome(candidate.active, self.solution_capacity))
        individual.uid = self.new_uid() if uid is None else uid
        individual.active = candidate.active
        individual.endpoint = candidate.endpoint
        individual.cost = candidate.cost
        individual.solved = candidate.solved
        individual.rank = candidate.rank
        individual.origin = origin
        individual.operator = operator
        individual.parents = tuple(parents)
        individual.block_id = block_id
        individual.objective_identity = self.objective.identity
        individual.input_key = self.input_key
        individual.created_generation = self.generation_in_progress
        individual.slot = origin
        individual.fitness.values = individual.rank
        return individual

    def copy_solution(self, individual, slot="elite"):
        clone = self.SolutionClass(individual)
        for name in _SOLUTION_FIELDS:
            setattr(clone, name, getattr(individual, name))
        clone.slot = slot
        clone.fitness.values = individual.fitness.values
        return clone

    def baseline_record(self, origin):
        return self.make_solution(self.baseline, origin=origin, operator="all-noop")

    def make_block(self, active, *, origin, operator=None, parents=(), fallback_reason=None,
                   target_length=None):
        block = self.BlockClass(pad_chromosome(active, self.block_capacity))
        block.uid = self.new_uid()
        block.active = active
        block.origin = origin
        block.operator = operator
        block.parents = tuple(parents)
        block.fallback_reason = fallback_reason
        block.target_length = target_length
        block.created_generation = self.generation_in_progress
        block.slot = origin
        block.evaluation = None
        block.batch_serial = None
        self._block_info[block.uid] = (origin, operator, len(active))
        return block

    def copy_block(self, block):
        """Elite survivor copy: same sequence and ID, score invalidated for the next batch."""
        clone = self.BlockClass(block)
        for name in _BLOCK_FIELDS:
            setattr(clone, name, getattr(block, name))
        clone.slot = "elite"
        clone.evaluation = None
        clone.batch_serial = None
        return clone

    # -- solution normalisation (section 6) ----------------------------------------

    def _best_prefix(self, moves, check_budget):
        reducer = MoveReducer()
        key = self.input_key
        cost = self.cost(key)
        solved = is_solved_key(key)
        best = (0, key, cost, solved)
        best_rank = (0 if solved else 1, cost, 0)
        best_infeasible = None
        capacity = self.solution_capacity
        counters = self.counters
        for index, allele in enumerate(moves, 1):
            if check_budget:
                self.budget.check()
            key = apply_move_to_key(key, allele, counters)
            reducer.push(allele)
            cost = self.cost(key)
            solved = is_solved_key(key)
            rank = (0 if solved else 1, cost, reducer.length)
            if reducer.length <= capacity:
                if rank < best_rank:
                    best_rank = rank
                    best = (index, key, cost, solved)
            elif best_infeasible is None or rank < best_infeasible:
                best_infeasible = rank
        return best, best_rank, best_infeasible

    def normalize_solution(self, proposal, *, check_budget=True):
        """Best simplified prefix of a complete proposal, verified by independent replay."""
        moves = active_moves(proposal)
        self.counters.normalizations += 1
        (k, key, cost, solved), rank, infeasible = self._best_prefix(moves, check_budget)
        capacity_rejected = infeasible is not None and infeasible < rank
        sequence = moves
        for _ in range(len(moves) + 2):
            chosen = simplify(sequence[:k])
            if chosen == sequence:
                break
            # Reordering or merging can expose a better intermediate prefix; rescan.
            sequence = chosen
            (k, key, cost, solved), rank, _ = self._best_prefix(sequence, check_budget)
        else:
            raise ConsistencyError("solution normalisation did not stabilise")
        candidate = NormalizedCandidate(chosen, key, cost, solved, capacity_rejected)
        self._verify_candidate(candidate)
        if capacity_rejected:
            self.counters.solution_capacity_rejections += 1
        return candidate

    def _verify_candidate(self, candidate):
        """Independent replay through cube.py's rotations (not the move tables); raise on disagreement."""
        self.counters.independent_replays += 1
        if len(candidate.active) > self.solution_capacity:
            raise ConsistencyError("candidate exceeds the solution capacity")
        if NOOP in candidate.active or simplify(candidate.active) != candidate.active:
            raise ConsistencyError("candidate active moves are not reduced")
        cube = replay(self._original, candidate.active, self.counters)
        key = state_key(cube)
        if key != candidate.endpoint:
            raise ConsistencyError("replayed endpoint differs from the stored endpoint")
        if self.objective.cost(key) != candidate.cost:
            raise ConsistencyError("replayed cost differs from the stored cost")
        if is_solved_key(key) != candidate.solved:
            raise ConsistencyError("replayed solved flag differs from the stored flag")

    def offer_to_archive(self, candidate, *, uid, source, block_id=None):
        """Replace the best-ever entry only on a strictly better, verified rank."""
        if not candidate.rank < self.archive.rank:
            return False
        self._verify_candidate(candidate)
        self.archive = ArchiveEntry(candidate.active, candidate.endpoint, candidate.cost,
                                    candidate.solved, uid, self.generation_in_progress, source, block_id)
        self.counters.archive_updates += 1
        self.last_improvement_generation = self.generation_in_progress
        if candidate.solved:
            raise SolutionFound(self.archive)
        return True

    # -- context batches and block scoring (section 8.2, 8.3) ---------------------------

    def build_context_batch(self, ranked_population):
        states = select_context_states(ranked_population, self.config.contexts, self.rng, self.input_key)
        contexts = []
        for context_id, (key, source, source_uid, group) in enumerate(states):
            if state_key(replay(self._original, source, self.counters)) != key:
                raise ConsistencyError("context source does not replay to its state")
            reducer = MoveReducer.from_moves(source)
            contexts.append(Context(context_id, key, tuple(source), source_uid, group, self.cost(key),
                                    is_solved_key(key), tuple(reducer.runs), reducer.length))
        identity = (self.objective.identity, self.solution_capacity,
                    tuple((context.key, context.source) for context in contexts))
        self._batch_serial += 1
        return ContextBatch(self._batch_serial, identity, tuple(contexts))

    def evaluate_blocks(self, blocks, batch):
        """Score every block against the same frozen batch; identical sequences share a score."""
        cache = self._block_cache.get(batch.identity)
        if cache is None:
            if len(self._block_cache) >= BLOCK_CACHE_BATCHES:
                del self._block_cache[next(iter(self._block_cache))]
            cache = self._block_cache[batch.identity] = {}
        for block in blocks:
            self.budget.check()
            evaluation = cache.get(block.active)
            if evaluation is None:
                evaluation = self._evaluate_block_sequence(block.active, batch, block.uid)
                cache[block.active] = evaluation
            else:
                self.counters.block_evaluation_cache_hits += 1
            block.evaluation = evaluation
            block.batch_serial = batch.serial
            block.fitness.values = evaluation.score

    def _evaluate_block_sequence(self, sequence, batch, block_uid):
        self.counters.block_sequence_evaluations += 1
        outcomes = []
        for context in batch.contexts:
            self.budget.check()
            outcomes.append(self._evaluate_block_on_context(sequence, context, block_uid))
        relative = [outcome.best.relative_gain if outcome.best else 0.0 for outcome in outcomes]
        mean_gain = math.fsum(relative) / len(relative)
        max_gain = max(relative)
        if not (math.isfinite(mean_gain) and math.isfinite(max_gain)):
            raise NumericalError("non-finite block score")
        return BlockEvaluation(
            batch.serial,
            tuple(outcomes),
            sum(1 for outcome in outcomes if outcome.best and outcome.best.solved),
            sum(1 for outcome in outcomes if outcome.best),
            mean_gain,
            max_gain,
            len(sequence),
        )

    def _evaluate_block_on_context(self, sequence, context, block_uid):
        """Replay the whole block from the context's state key and inspect every prefix."""
        key = context.key
        reducer = MoveReducer(context.source_runs, context.source_length)
        scale = max(context.cost, 1.0)
        capacity = self.solution_capacity
        threshold = self.config.min_block_gain
        integer_valued = self.objective.integer_valued
        qualifying = []
        best = None
        best_preference = None
        for k, allele in enumerate(sequence, 1):
            self.budget.check()
            key = apply_move_to_key(key, allele, self.counters)
            reducer.push(allele)
            cost = self.cost(key)
            delta = context.cost - cost
            solved = not context.solved and is_solved_key(key)
            if not solved and not gain_qualifies(delta, threshold, integer_valued):
                continue
            length = reducer.length
            relative = delta / scale
            if not math.isfinite(relative):
                raise NumericalError("non-finite relative block gain")
            outcome = PrefixOutcome(k, delta, relative, solved, key, cost, length, length <= capacity)
            qualifying.append(outcome)
            if not outcome.representative_feasible:
                self.counters.block_prefix_capacity_exclusions += 1
                continue
            preference = _prefix_preference(solved, delta, length, k)
            if best is None or preference < best_preference:
                best, best_preference = outcome, preference
            if solved:
                self._archive_block_discovery(context, sequence, outcome, block_uid)
        if best is not None:
            self._archive_block_discovery(context, sequence, best, block_uid)
        return ContextOutcome(context.context_id, best, tuple(qualifying))

    def _archive_block_discovery(self, context, sequence, outcome, block_uid):
        """Rebuild source + prefix from the original input and offer it to the archive."""
        claimed = (0 if outcome.solved else 1, outcome.endpoint_cost, outcome.representative_length)
        if not claimed < self.archive.rank:
            return
        # Bounded verification: not interrupted by the budget, so a found solve is never lost.
        candidate = self.normalize_solution(context.source + tuple(sequence[:outcome.k]), check_budget=False)
        if outcome.solved and not candidate.solved:
            raise ConsistencyError("a solving block prefix did not reproduce from the original input")
        source = f"block B{block_uid} prefix {outcome.k} on context {context.context_id}"
        if self.offer_to_archive(candidate, uid=None, source=source, block_id=block_uid):
            self.counters.block_discoveries_archived += 1

    def _index_blocks(self, blocks, batch):
        """Distinct canonical sequences (smallest-ID representative) and per-endpoint evidence."""
        representatives = {}
        for block in blocks:
            current = representatives.get(block.active)
            if current is None or block.uid < current.uid:
                representatives[block.active] = block
        ordered = sorted(representatives.values(), key=lambda block: block.uid)
        index = {}
        for block in ordered:
            for context, outcome in zip(batch.contexts, block.evaluation.outcomes):
                if outcome.qualifying:
                    index.setdefault(context.key, []).append((block.active, block.uid, outcome.qualifying))
        self._block_representatives = ordered
        self._block_index = index
        self._context_sources = {context.key: context.source for context in batch.contexts}

    def _exploratory_block(self):
        block = self._block_representatives[self.rng.randrange(len(self._block_representatives))]
        return block.active, block.uid

    def _score_append_entry(self, entry, base_reducer):
        """Best ``(preference, k)`` of one indexed block for a parent, or None if no prefix fits.

        ``base_reducer`` None means the parent is the context's own source sequence,
        so the stored representative lengths and feasibility are exact for it.
        """
        sequence, _, qualifying = entry
        best = None
        if base_reducer is None:
            for outcome in qualifying:
                if not outcome.representative_feasible:
                    continue
                preference = _prefix_preference(outcome.solved, outcome.gain, outcome.representative_length,
                                                outcome.k)
                if best is None or preference < best[0]:
                    best = (preference, outcome.k)
            return best
        reducer = base_reducer.copy()
        pushed = 0
        for outcome in qualifying:
            while pushed < outcome.k:
                reducer.push(sequence[pushed])
                pushed += 1
            # Feasibility is rechecked against this actual parent, not the representative.
            if reducer.length > self.solution_capacity:
                continue
            preference = _prefix_preference(outcome.solved, outcome.gain, reducer.length, outcome.k)
            if best is None or preference < best[0]:
                best = (preference, outcome.k)
        return best

    def choose_append_payload(self, parent_active, parent_endpoint):
        """Block prefix to append to a parent (section 7.4); returns (payload, block_id, targeted).

        Tournament entrants are drawn uniformly from the blocks with evidence for this
        endpoint and only those are scored. A drawn block with no prefix that fits the
        parent is rejected and redrawn, which gives the same tournament as scoring every
        block first. Each block is scored at most once per call; once every block has
        been rejected the append falls back to an exploratory block.
        """
        entries = self._block_index.get(parent_endpoint, ())
        if entries:
            if tuple(parent_active) == self._context_sources.get(parent_endpoint):
                base_reducer = None
            else:
                base_reducer = MoveReducer.from_moves(parent_active)
            scores = {}
            draws = []
            rejected = 0
            while len(draws) < TOURNAMENT_SIZE and rejected < len(entries):
                index = self.rng.randrange(len(entries))
                if index not in scores:
                    scores[index] = self._score_append_entry(entries[index], base_reducer)
                    if scores[index] is None:
                        rejected += 1
                if scores[index] is not None:
                    draws.append((scores[index][0], index))
            if draws:
                best = min(preference for preference, _ in draws)
                tied = [index for preference, index in draws if preference == best]
                index = tied[0] if len(tied) == 1 else tied[self.rng.randrange(len(tied))]
                sequence, uid, _ = entries[index]
                self.counters.targeted_appends += 1
                return tuple(sequence[:scores[index][1]]), uid, True
        sequence, uid = self._exploratory_block()
        self.counters.exploratory_appends += 1
        return sequence, uid, False

    # -- solution population (section 7) -----------------------------------------------

    def initial_solution_population(self):
        return [self.baseline_record("initial") for _ in range(self.config.solution_population)]

    def next_solution_population(self, population):
        ranked = sorted(population, key=_solution_sort_key)
        count = len(ranked)
        fifth = count // 5
        next_population = [self.copy_solution(individual) for individual in ranked[:fifth]]
        pool = ranked[:count - fifth]
        offspring_count = 3 * fifth
        fresh_starts = baseline_start_quota(offspring_count)
        slots = ["baseline-start"] * fresh_starts + ["ordinary"] * (offspring_count - fresh_starts)
        self.rng.shuffle(slots)
        for slot in slots:
            self.budget.check()
            if slot == "baseline-start":
                next_population.append(self._baseline_start_offspring())
            else:
                next_population.append(self._ordinary_offspring(pool))
        next_population.extend(self.baseline_record("reset") for _ in range(fifth))
        return next_population

    def _choose_solution_operator(self):
        draw = self.rng.random()
        cumulative = 0.0
        for name, probability in SOLUTION_OPERATOR_PROBABILITIES:
            cumulative += probability
            if draw < cumulative:
                return name
        return SOLUTION_OPERATOR_PROBABILITIES[-1][0]

    def _baseline_start_offspring(self):
        payload, block_id, targeted = self.choose_append_payload((), self.input_key)
        candidate = self.normalize_solution(payload)
        self.counters.baseline_start_attempts += 1
        if candidate.rank < self.baseline_rank:
            self.counters.baseline_start_improvements += 1
            self.counters.attributed_improvements[block_id] += 1
        operator = "baseline-append-targeted" if targeted else "baseline-append-exploratory"
        child = self.make_solution(candidate, origin="baseline-start", operator=operator, block_id=block_id)
        self.offer_to_archive(candidate, uid=child.uid, source=operator, block_id=block_id)
        return child

    def _ordinary_offspring(self, pool):
        rng = self.rng
        parent = tournament_select(pool, lambda individual: individual.rank, rng)
        parent_active = parent.active
        parents = (parent.uid,)
        block_id = None
        operator = self._choose_solution_operator()
        if operator == "append-block":
            payload, block_id, targeted = self.choose_append_payload(parent_active, parent.endpoint)
            operator = "append-targeted" if targeted else "append-exploratory"
            proposal = parent_active + payload
            bound = self.solution_capacity + self.block_capacity
        elif operator == "splice-block":
            block, block_id = self._exploratory_block()
            insert = rng.random() < 0.5
            if insert or not parent_active:
                operator = "splice-insert"
                proposal = op_insert_block(parent_active, block, rng)
            else:
                operator = "splice-replace"
                proposal = op_replace_span_with_block(parent_active, block, rng)
            bound = self.solution_capacity + self.block_capacity
        elif operator == "move-variation":
            variation = MOVE_VARIATIONS[rng.randrange(len(MOVE_VARIATIONS))]
            if variation == "insert-moves":
                proposal = op_insert_moves(parent_active, rng)
            elif not parent_active:
                variation += "-empty-fallback"
                proposal = random_payload(rng, 1)
            elif variation == "delete-gene":
                proposal = op_delete_gene(parent_active, rng)
            elif variation == "substitute-gene":
                proposal = op_substitute(parent_active, rng)
            elif variation == "cube-inverse":
                proposal = cube_inverse(parent_active)
            else:
                proposal = op_span_replace(parent_active, rng)
            operator = variation
            bound = self.solution_capacity + MAX_PAYLOAD_LENGTH
        else:
            other = tournament_select(pool, lambda individual: individual.rank, rng)
            parents = (parent.uid, other.uid)
            if not parent_active and not other.active:
                operator = "crossover-empty-fallback"
                proposal, block_id = self._exploratory_block()
                bound = self.block_capacity
            else:
                proposal = op_crossover(parent_active, other.active, rng)
                bound = 2 * self.solution_capacity
        if len(proposal) > bound:
            raise ConsistencyError(f"operator {operator} exceeded its proposal bound")
        candidate = self.normalize_solution(proposal)
        if candidate.active == parent_active:
            self.counters.no_change_offspring += 1
        if block_id is not None and candidate.rank < parent.rank:
            self.counters.attributed_improvements[block_id] += 1
        child = self.make_solution(candidate, origin="offspring", operator=operator,
                                   parents=parents, block_id=block_id)
        self.offer_to_archive(candidate, uid=child.uid, source=operator, block_id=block_id)
        return child

    # -- block population (section 8) ---------------------------------------------------

    def initial_block_population(self):
        counts = {}
        schedule = reduced_length_schedule(self.config.block_population, self.block_capacity, self.rng)
        return [self.fresh_block(target, counts, origin="initial") for target in schedule]

    def fresh_block(self, target_length, counts, *, origin, reason=None, operator=None, parents=()):
        """Fresh block at a scheduled reduced length, respecting the duplicate cap."""
        sequence = None
        for _ in range(DUPLICATE_RESAMPLE_ATTEMPTS):
            self.budget.check()
            proposal = generate_reduced_block(target_length, self.rng)
            if counts.get(proposal, 0) < self.max_block_copies:
                sequence = proposal
                break
            self.counters.duplicate_block_retries += 1
        if sequence is None:
            sequence = self._enumerate_available_block(target_length, counts)
            if len(sequence) != target_length:
                self.counters.target_length_fallbacks += 1
        if reason is not None:
            self.counters.block_fallbacks[reason] += 1
        counts[sequence] = counts.get(sequence, 0) + 1
        return self.make_block(sequence, origin=origin, operator=operator or "fresh", parents=parents,
                               fallback_reason=reason, target_length=target_length)

    def _enumerate_available_block(self, target_length, counts):
        lengths = [target_length] + [length for length in range(1, self.block_capacity + 1)
                                     if length != target_length]
        for length in lengths:
            for sequence in iter_canonical_sequences(length, check=self.budget.check):
                if counts.get(sequence, 0) < self.max_block_copies:
                    return sequence
        raise ConsistencyError("no canonical block sequence has room under the duplicate cap")

    def _select_block_parent(self, representatives, useful_regime):
        if useful_regime:
            return tournament_select(representatives, _block_tournament_key, self.rng)
        # Neutral regime: uniform over lengths present, then over sequences of that length.
        lengths = sorted({len(block.active) for block in representatives})
        length = lengths[self.rng.randrange(len(lengths))]
        same_length = [block for block in representatives if len(block.active) == length]
        return same_length[self.rng.randrange(len(same_length))]

    def vary_block(self, operator, parents):
        """Raw (unsimplified) proposal of one block operator; returns (proposal, raw bound)."""
        rng = self.rng
        capacity = self.block_capacity
        first = parents[0].active
        if operator == "insert":
            return op_insert_moves(first, rng), capacity + MAX_PAYLOAD_LENGTH
        if operator == "delete":
            return op_delete_span(first, rng), capacity
        if operator == "substitute":
            return op_substitute(first, rng), capacity
        if operator == "inverse":
            return cube_inverse(first), capacity
        if operator == "span-replace":
            return op_span_replace(first, rng), capacity + MAX_PAYLOAD_LENGTH
        second = parents[1].active
        if operator == "concatenate":
            return first + second, 2 * capacity
        if operator == "crossover":
            return op_crossover(first, second, rng), 2 * capacity
        raise ValueError(f"unknown block operator {operator!r}")

    def block_offspring(self, representatives, useful_regime, counts):
        rng = self.rng
        for _ in range(BLOCK_VARIATION_ATTEMPTS):
            self.budget.check()
            operator = BLOCK_OPERATORS[rng.randrange(len(BLOCK_OPERATORS))]
            parents = [self._select_block_parent(representatives, useful_regime)]
            if operator in ("concatenate", "crossover"):
                parents.append(self._select_block_parent(representatives, useful_regime))
            proposal, bound = self.vary_block(operator, parents)
            if len(proposal) > bound:
                raise ConsistencyError(f"block operator {operator} exceeded its proposal bound")
            parent_ids = tuple(parent.uid for parent in parents)
            reduced = simplify(proposal)
            if not reduced:
                return self.fresh_block(rng.randint(1, self.block_capacity), counts, origin="offspring",
                                        reason="empty-result", operator=f"{operator}->fresh",
                                        parents=parent_ids)
            if len(reduced) > self.block_capacity:
                self.counters.block_oversized_rejections += 1
                continue
            if counts.get(reduced, 0) >= self.max_block_copies:
                self.counters.duplicate_block_retries += 1
                continue
            if any(reduced == parent.active for parent in parents):
                self.counters.unchanged_block_proposals += 1
            counts[reduced] = counts.get(reduced, 0) + 1
            return self.make_block(reduced, origin="offspring", operator=operator, parents=parent_ids)
        return self.fresh_block(rng.randint(1, self.block_capacity), counts, origin="offspring",
                                reason="variation-attempts-exhausted", operator="fresh")

    def next_block_population(self, blocks):
        count = len(blocks)
        elite_count, offspring_count, fresh_count = count // 10, 6 * count // 10, 3 * count // 10
        ranked = sorted(blocks, key=_block_sort_key)
        counts = {}
        next_blocks = []
        for block in ranked[:elite_count]:
            counts[block.active] = counts.get(block.active, 0) + 1
            next_blocks.append(self.copy_block(block))
        representatives = self._block_representatives
        useful_regime = any(block.evaluation.useful for block in representatives)
        for _ in range(offspring_count):
            next_blocks.append(self.block_offspring(representatives, useful_regime, counts))
        for target in reduced_length_schedule(fresh_count, self.block_capacity, self.rng):
            next_blocks.append(self.fresh_block(target, counts, origin="fresh"))
        return next_blocks

    # -- generation schedule (section 9) ------------------------------------------------

    def run_generation(self, generation, build_next_blocks=True):
        """One synchronous generation; publishes populations only when complete."""
        self.generation_in_progress = generation
        self.partial_generation = generation
        self.partial_phase = "context batch"
        ranked = sorted(self.population, key=_solution_sort_key)
        batch = self.build_context_batch(ranked)
        self.batch = batch
        self.partial_phase = "block evaluation"
        self.evaluate_blocks(self.blocks, batch)
        self._index_blocks(self.blocks, batch)
        summary = self._block_summary(self.blocks, batch)
        self.partial_phase = "solution variation"
        next_population = self.next_solution_population(self.population)
        next_blocks = None
        if build_next_blocks:
            self.partial_phase = "block replacement"
            next_blocks = self.next_block_population(self.blocks)
        self.population = next_population
        if next_blocks is not None:
            self.blocks = next_blocks
        self.last_block_summary = summary
        self.completed_generation = generation
        self.partial_generation = None
        self.partial_phase = None
        return summary

    def run(self):
        if self.baseline.solved:
            return self._result(STATUS_SOLVED, "the input cube is already solved")
        self.budget.start()
        try:
            self.budget.check()
            self.population = self.initial_solution_population()
            self._record_history(0)
            if self.config.generations == 0:
                return self._result(STATUS_BUDGET_EXHAUSTED,
                                    "zero generations requested; block population not created")
            self.partial_generation = 0
            self.partial_phase = "initial block population"
            self.blocks = self.initial_block_population()
            self.partial_generation = None
            self.partial_phase = None
            generations = self.config.generations
            for generation in range(1, generations + 1):
                self.run_generation(generation, build_next_blocks=generation < generations)
                self._record_history(generation)
                if generation % self.config.log_every == 0:
                    self.report_progress()
            return self._result(STATUS_BUDGET_EXHAUSTED, "generation budget exhausted")
        except SolutionFound:
            return self._result(STATUS_SOLVED, "verified solve found")
        except SearchStopped as stop:
            return self._result(stop.status, str(stop) or stop.status)

    def interrupted_result(self):
        """Result for a hard interrupt (second SIGINT) that escaped the budget checks."""
        return self._result(STATUS_INTERRUPTED, "SIGINT received")

    def _result(self, status, reason):
        if self.archive.solved:
            status = STATUS_SOLVED
        last = self.history[-1] if len(self.history) else None
        if last is None or last["best_cost"] != self.archive.cost or last["best_length"] != len(self.archive.active):
            self._record_history(self.partial_generation if self.partial_generation is not None
                                 else self.completed_generation, partial=self.partial_generation is not None)
        return RunResult(
            status=status,
            reason=reason,
            archive=self.archive,
            seed=self.seed,
            input_key=self.input_key,
            objective=self.objective,
            solution_capacity=self.solution_capacity,
            completed_generations=self.completed_generation,
            partial_generation=self.partial_generation,
            partial_phase=self.partial_phase,
            search_seconds=self.budget.elapsed(),
            counters=self.counters,
            history=list(self.history),
            block_population_created=self.blocks is not None,
        )

    # -- reporting (section 11) ----------------------------------------------------------

    def _block_summary(self, blocks, batch):
        evaluations = [block.evaluation for block in blocks]
        best_gain = max((outcome.best.gain for evaluation in evaluations
                         for outcome in evaluation.outcomes if outcome.best), default=0)
        return {
            "batch": batch.serial,
            "contexts": len(batch.contexts),
            "useful_blocks": sum(1 for evaluation in evaluations if evaluation.useful),
            "best_mean_gain": max(evaluation.mean_relative_gain for evaluation in evaluations),
            "best_max_gain": max(evaluation.max_relative_gain for evaluation in evaluations),
            "best_abs_gain": best_gain,
            "lengths": Counter(len(block.active) for block in blocks),
        }

    def _record_history(self, generation, partial=False):
        summary = self.last_block_summary
        self.history.record(
            gen=generation,
            partial=partial,
            best_cost=self.archive.cost,
            best_length=len(self.archive.active),
            solved=self.archive.solved,
            block_mean_gain=None if summary is None or partial else summary["best_mean_gain"],
            batch=None if summary is None or partial else summary["batch"],
        )

    def _report_marker(self):
        return (self.completed_generation, self.partial_generation, self.archive.rank)

    def report_progress_if_stale(self):
        """Final progress lines, unless the same state was just reported."""
        if self.population is not None and self._last_report_marker != self._report_marker():
            self.report_progress()

    def report_progress(self):
        self._last_report_marker = self._report_marker()
        counters = self.counters
        archive = self.archive
        population = self.population or []
        endpoints = Counter(individual.endpoint for individual in population)
        empty = sum(1 for individual in population if not individual.active)
        lines = [
            f"[gen {self.completed_generation} | {self.budget.elapsed():.1f}s] best-ever cost={archive.cost} "
            f"moves={len(archive.active)} solved={'yes' if archive.solved else 'no'} "
            f"generations since improvement={self.completed_generation - self.last_improvement_generation}",
            f"  solutions: distinct endpoints={len(endpoints)}/{len(population)} "
            f"(duplicate endpoints={len(population) - len(endpoints)}) empty={empty} | baseline-start "
            f"attempts={counters.baseline_start_attempts} improvements={counters.baseline_start_improvements}",
        ]
        summary = self.last_block_summary
        if summary is not None:
            histogram = " ".join(f"{length}:{summary['lengths'][length]}" for length in sorted(summary["lengths"]))
            lines.append(
                f"  blocks (batch #{summary['batch']}, {summary['contexts']} distinct contexts): "
                f"useful={summary['useful_blocks']} best mean rel gain={summary['best_mean_gain']:.4f} "
                f"best max rel gain={summary['best_max_gain']:.4f} best gain={summary['best_abs_gain']}"
            )
            lines.append(f"  block reduced lengths: {histogram}")
        lines.append(
            f"  work: H computations={counters.cost_computations} cache hits={counters.cost_cache_hits} "
            f"quarter turns={counters.quarter_turns} normalisations={counters.normalizations} "
            f"block sequence evaluations={counters.block_sequence_evaluations}"
        )
        fallbacks = dict(counters.block_fallbacks) or "none"
        lines.append(
            f"  rejections: solution capacity={counters.solution_capacity_rejections} "
            f"block oversized={counters.block_oversized_rejections} "
            f"block-prefix capacity={counters.block_prefix_capacity_exclusions} "
            f"no-change offspring={counters.no_change_offspring} "
            f"duplicate block retries={counters.duplicate_block_retries} block fallbacks={fallbacks} "
            f"target-length fallbacks={counters.target_length_fallbacks}"
        )
        lines.append(f"  attributed improvements: {self.describe_attributions()}")
        if archive.rank == self.baseline_rank:
            lines.append("  STAGNATION: every solution still has the initial score; "
                         "block exploration has not yet crossed the baseline plateau")
        for line in lines:
            self.log(line)

    def describe_attributions(self, limit=5):
        attributions = self.counters.attributed_improvements
        if not attributions:
            return "none"
        parts = []
        for block_id, count in sorted(attributions.items(), key=lambda item: (-item[1], item[0]))[:limit]:
            origin, operator, length = self._block_info.get(block_id, ("?", "?", "?"))
            parts.append(f"B{block_id}({origin}/{operator}, len {length})={count}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Library entry points, finalisation and plotting
# ---------------------------------------------------------------------------


def _print_flush(message=""):
    print(message, flush=True)


def run_search(config, input_cube=None, *, budget=None, log=None):
    """Validate, then run one search; ``input_cube`` defaults to :func:`default_input_cube`."""
    objective = validate_config(config)
    cube = default_input_cube() if input_cube is None else input_cube
    input_key = validate_input_cube(cube)
    seed = config.seed if config.seed is not None else generate_seed()
    run = CoevolutionRun(config, input_key, objective, seed, budget=budget, log=log)
    return run.run()


def finalize_result(result):
    """Replay only the archived active moves from a fresh original and compare metadata."""
    entry = result.archive
    problems = []
    try:
        moves = validate_alleles(entry.active)
    except ValueError as exc:
        return FinalReport(STATUS_ERROR, (), None, None, None, [str(exc)])
    if NOOP in moves:
        problems.append("archived active moves contain NOOP padding")
    if simplify(moves) != moves:
        problems.append("archived active moves are not reduced")
    if len(moves) > result.solution_capacity:
        problems.append("archived active moves exceed the solution capacity")
    cube = replay(cube_from_key(result.input_key), moves)
    key = state_key(cube)
    cost = result.objective.cost(key)
    solved = is_solved_key(key)
    if key != entry.endpoint:
        problems.append("replayed endpoint differs from the archived endpoint")
    if cost != entry.cost:
        problems.append(f"replayed cost {cost} differs from the archived cost {entry.cost}")
    if solved != entry.solved:
        problems.append("replayed solved flag differs from the archived flag")
    status = result.status
    if problems or (status == STATUS_SOLVED and not solved):
        status = STATUS_ERROR
        if not problems:
            problems.append("SOLVED reported but the replayed cube is not solved")
    return FinalReport(status, moves, cube, cost, solved, problems)


def _free_path(directory, filename):
    stem, extension = os.path.splitext(filename)
    path = os.path.join(directory, filename)
    number = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{number}{extension}")
        number += 1
    return path


def save_progress_plot(history, directory="."):
    """Save the progress chart with the Agg canvas (never a GUI window); return its path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 5))
    FigureCanvasAgg(figure)
    cost_axis = figure.add_subplot()
    generations = [record["gen"] for record in history]
    costs = [record["best_cost"] for record in history]
    cost_axis.step(generations, costs, where="post", marker="o", markersize=3, color="tab:red",
                   label="best-ever solution cost H")
    cost_axis.set_xlabel("generation")
    cost_axis.set_ylabel("best-ever cost (lower is better)")
    gain_axis = cost_axis.twinx()
    gain_points = [(record["gen"], record["block_mean_gain"]) for record in history
                   if record["block_mean_gain"] is not None]
    if gain_points:
        gain_axis.plot([point[0] for point in gain_points], [point[1] for point in gain_points],
                       color="tab:blue", linewidth=1,
                       label="best block mean relative gain (current batch; contexts change)")
    gain_axis.set_ylabel("block mean relative gain (changing contexts)")
    handles = cost_axis.get_legend_handles_labels()
    gain_handles = gain_axis.get_legend_handles_labels()
    cost_axis.legend(handles[0] + gain_handles[0], handles[1] + gain_handles[1], loc="upper right", fontsize=8)
    cost_axis.set_title("Cooperative coevolution progress")
    path = _free_path(directory, PLOT_FILENAME)
    figure.savefig(path)
    return path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Cooperative DEAP evolution of cube solutions and reusable move blocks. "
                    "Reported lengths are reduced by move-equivalent simplification, not globally minimal.",
    )
    parser.add_argument("--fitness", choices=FITNESS_CHOICES, default=DEFAULT_FITNESS,
                        help=f"state cost to minimise (default: {DEFAULT_FITNESS})")
    parser.add_argument("--manhattan-weight", type=float, default=None,
                        help=f"manhattan weight for manhattan-misplaced (default {DEFAULT_MANHATTAN_WEIGHT})")
    parser.add_argument("--misplaced-weight", type=float, default=None,
                        help=f"misplaced weight for manhattan-misplaced (default {DEFAULT_MISPLACED_WEIGHT})")
    parser.add_argument("--solution-population", type=int, default=DEFAULT_SOLUTION_POPULATION)
    parser.add_argument("--block-population", type=int, default=DEFAULT_BLOCK_POPULATION)
    parser.add_argument("--solution-capacity", type=int, default=DEFAULT_SOLUTION_CAPACITY)
    parser.add_argument("--block-capacity", type=int, default=DEFAULT_BLOCK_CAPACITY)
    parser.add_argument("--contexts", type=int, default=DEFAULT_CONTEXT_COUNT)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--min-block-gain", type=float, default=DEFAULT_MIN_BLOCK_GAIN)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="optional wall-clock search budget in seconds")
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--plot", action="store_true", help=f"save {PLOT_FILENAME} (off by default)")
    return parser


def config_from_args(args):
    return SearchConfig(
        fitness=args.fitness,
        manhattan_weight=args.manhattan_weight,
        misplaced_weight=args.misplaced_weight,
        solution_population=args.solution_population,
        block_population=args.block_population,
        solution_capacity=args.solution_capacity,
        block_capacity=args.block_capacity,
        contexts=args.contexts,
        generations=args.generations,
        min_block_gain=args.min_block_gain,
        seed=args.seed,
        time_limit=args.time_limit,
        log_every=args.log_every,
        plot=args.plot,
    )


def _print_status(status, detail=None):
    if detail:
        print(detail)
    print(f"Status: {status}")
    print(f"Exit code: {EXIT_CODES[status]} ({EXIT_EXPLANATIONS[status]})")


def _print_startup(config, objective, seed, seed_generated, input_cube):
    solution_fifth = config.solution_population // 5
    offspring = 3 * solution_fifth
    print("Cooperative coevolution solver: solution candidates + move blocks (DEAP)")
    print("Original cube:")
    input_cube.print_colored()
    input_cube.print_as_python_list()
    print(f"Objective: {objective.describe()}")
    if objective.state_function == "manhattan-misplaced":
        print(f"Effective weights: manhattan={objective.manhattan_weight!r} "
              f"misplaced={objective.misplaced_weight!r}")
    print("Ranking: (unsolved_flag, H(endpoint), reduced active length), all minimised; "
          "'reduced' means move-equivalent simplification, not a globally shortest sequence")
    print(f"Seed: {seed} ({'generated' if seed_generated else 'supplied'})")
    print(f"Capacities: solution={config.solution_capacity} block={config.block_capacity} | "
          f"contexts={config.contexts} | generations={config.generations} | "
          f"time limit={'none' if config.time_limit is None else f'{config.time_limit}s'} | "
          f"min block gain={config.min_block_gain!r} | log every={config.log_every}")
    print(f"Solution population {config.solution_population}: {solution_fifth} elite copies, "
          f"{offspring} offspring ({baseline_start_quota(offspring)} baseline-start), "
          f"{solution_fifth} all-NOOP resets")
    print(f"Block population {config.block_population}: {config.block_population // 10} elite, "
          f"{6 * config.block_population // 10} offspring, {3 * config.block_population // 10} fresh; "
          f"duplicate cap {max(2, math.ceil(config.block_population / 18))}")
    print("Solution operators: " + ", ".join(f"{name}={probability:.2f}"
                                               for name, probability in SOLUTION_OPERATOR_PROBABILITIES))
    print("Block operators: " + ", ".join(BLOCK_OPERATORS) + " (uniform)")


def _print_final(result, final, finalization_seconds):
    entry = result.archive
    print()
    print("=" * 72)
    print(f"Search stopped: {result.reason}")
    if result.partial_generation is not None:
        print(f"Completed generations: {result.completed_generations}; generation "
              f"{result.partial_generation} was partially evaluated ({result.partial_phase}) and not published")
    else:
        print(f"Completed generations: {result.completed_generations}")
    if not result.block_population_created:
        print("Block population: not created")
    print(f"Search time: {result.search_seconds:.3f}s; finalisation time: {finalization_seconds:.3f}s")
    counters = result.counters
    print(f"Work: H computations={counters.cost_computations} cache hits={counters.cost_cache_hits} "
          f"quarter turns={counters.quarter_turns} clones={counters.cube_clones} "
          f"normalisations={counters.normalizations} replays={counters.independent_replays}")
    print(f"Best verified attempt found in generation {entry.generation} via {entry.source}")
    if final.problems:
        for problem in final.problems:
            print(f"CONSISTENCY ERROR: {problem}")
    print(f"Moves ({len(final.moves)}, reduced): {format_moves(final.moves)}")
    if final.cube is not None:
        print(f"Replayed cost H={final.cost} solved={'yes' if final.solved else 'no'}")
        print("Final cube:")
        final.cube.print_colored()
        final.cube.print_as_python_list()


def _install_sigint(budget):
    def handler(signum, frame):
        if budget.interrupt_requested:
            raise KeyboardInterrupt
        budget.request_interrupt()

    try:
        return signal.signal(signal.SIGINT, handler)
    except ValueError:  # not in the main thread
        return None


def main(argv=None, *, clock=None):
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 2
    config = config_from_args(args)
    try:
        objective = validate_config(config)
        input_cube = default_input_cube()
        input_key = validate_input_cube(input_cube)
        if objective.state_function == "cubie":
            validate_cubie_input(input_key)
    except (ConfigError, InvalidInputError) as exc:
        _print_status(STATUS_INVALID_INPUT, f"Invalid configuration or input: {exc}")
        return EXIT_CODES[STATUS_INVALID_INPUT]

    seed_generated = config.seed is None
    seed = generate_seed() if seed_generated else config.seed
    _print_startup(config, objective, seed, seed_generated, input_cube)

    budget = Budget(config.time_limit, clock or time.monotonic)
    previous_handler = _install_sigint(budget)
    run = None
    try:
        run = CoevolutionRun(config, input_key, objective, seed, budget=budget)
        print(f"Baseline: H={run.baseline.cost} solved={'yes' if run.baseline.solved else 'no'} (0 moves)")
        result = run.run()
        run.report_progress_if_stale()
    except KeyboardInterrupt:
        if run is None:
            _print_status(STATUS_INTERRUPTED)
            return EXIT_CODES[STATUS_INTERRUPTED]
        result = run.interrupted_result()
    except (ConsistencyError, NumericalError) as exc:
        _print_status(STATUS_ERROR, f"Internal error: {exc}")
        return EXIT_CODES[STATUS_ERROR]
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)

    finalization_started = time.monotonic()
    final = finalize_result(result)
    finalization_seconds = time.monotonic() - finalization_started
    _print_final(result, final, finalization_seconds)
    if config.plot:
        print(f"Progress plot saved to {save_progress_plot(result.history)}")
    _print_status(final.status)
    return EXIT_CODES[final.status]


if __name__ == "__main__":
    sys.exit(main())
