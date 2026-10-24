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
* The "edges" objective counts the 20 blocks around the centres (8 corners and
  12 edges) that do not show all their correct colours.
* "grouped-cubie" uses exact joint distances for five four-piece groups, with
  solo cubie distance and misplaced stickers as lower-priority tie-breakers.
* Solved detection is independent of the objective: every sticker must equal
  its own face centre.

Usage::

    uv run python solver_deap_coevolution.py --seed 42
    uv run python solver_deap_coevolution.py --fitness misplaced --seed 42

The CLI also keeps a persistent library of up to 100 distinct move blocks
(``--block-library PATH``, ``--no-block-library``); see
persistent_move_block_library_specs.md.

Importing this module runs no search, prints nothing, creates no plots and
touches no library file.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from operator import itemgetter
from typing import Optional

import colorama
from deap import base, creator, tools

from cube import Cube

# ---------------------------------------------------------------------------
# Configuration defaults (section 10)
# ---------------------------------------------------------------------------

DEFAULT_FITNESS = "manhattan"
DEFAULT_SOLUTION_POPULATION = 100000
DEFAULT_BLOCK_POPULATION = 30000
DEFAULT_SOLUTION_CAPACITY = 72
DEFAULT_BLOCK_CAPACITY = 64
DEFAULT_CONTEXT_COUNT = 32
DEFAULT_GENERATIONS = 10000
DEFAULT_MIN_BLOCK_GAIN = 1.0
DEFAULT_LOG_EVERY = 10
DEFAULT_MANHATTAN_WEIGHT = 1.0
DEFAULT_MISPLACED_WEIGHT = 1.0

# Solution-population diversity and stagnation checks (solution_population_diversity_specs.md).
DIVERSITY_CHECK_INTERVAL = 50
DEFAULT_DIVERSITY_MIN_UNIQUE_RATIO = 0.10
DEFAULT_DIVERSITY_MAX_DOMINANCE = 0.80
DEFAULT_DIVERSITY_MIN_CUBIE_DISTANCE = 2.0
DEFAULT_STAGNATION_GENERATIONS = 100

FITNESS_CHOICES = ("manhattan", "misplaced", "manhattan-misplaced", "manhattan-min", "cubie", "edges",
                   "cubie-refined", "grouped-cubie", "corners-first")
INTEGER_VALUED_FITNESS = ("manhattan", "manhattan-min", "misplaced", "cubie", "edges", "cubie-refined",
                          "corners-first", "grouped-cubie")

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
    """A fully scrambled default input (manhattan H=198, cubie cost 32).

    The spec's original checkpoint from ``solver_deap.py`` is kept in
    ``benchmark_coevolution.CHECKPOINT_KEY``.
    """
    #return Cube(
    #                               "WYRYGRGRY",
    #    ["WOGBRGYOR", "RGOWWOYYG", "YWOBOBOOO", "BGBWYRBYB"],
    #                               "WGWRBBGWR",
    #)
    return Cube(
                                    "GGGYGWGGG",
        [ "RORRRRRRR",  "WGWWWWWWW",  "OROOOOOOO",  "YGYYYYYYY", ],
                                    "BBBBBBBBB"
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
    return tuple(slots), tuple((itemgetter(*slot), table) for slot, table in zip(slots, tables))


_CUBIE_SLOTS, _CUBIE_TABLES = _build_cubie_tables()
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


def _cubie_distances(key):
    """(corner, edge) sums of the per-cubie distances that make up ``cubie_cost``."""
    faces = _face_indices(key)
    corner_total = edge_total = 0
    try:
        for slot, (getter, distances) in zip(_CUBIE_SLOTS, _CUBIE_TABLES):
            if len(slot) == 3:
                corner_total += distances["".join(getter(faces))]
            else:
                edge_total += distances["".join(getter(faces))]
    except KeyError:
        raise ValueError("the stickers do not form a consistent set of cubies") from None
    return corner_total, edge_total


def corner_cubie_distance(key):
    """Sum over the 8 corners of each one's move distance to its home slot and orientation."""
    return _cubie_distances(key)[0]


def edge_cubie_distance(key):
    """Sum over the 12 edges of each one's move distance to its home slot and orientation."""
    return _cubie_distances(key)[1]


# One corner unit must outweigh every possible edge total, so corners stay
# strictly primary: the largest achievable edge distance sum, plus one.
_MAX_EDGE_CUBIE_DISTANCE = sum(max(distances.values())
                               for slot, (_, distances) in zip(_CUBIE_SLOTS, _CUBIE_TABLES)
                               if len(slot) == 2)
CORNERS_FIRST_EDGE_SCALE = _MAX_EDGE_CUBIE_DISTANCE + 1


def corners_first_cost(key):
    """Staged cost: corners home first (slot and orientation), then edges.

    The edge distance only breaks through once the corner distance is zero,
    and any corner regression outweighs every possible edge gain, so the
    cost reverts to rewarding corners alone. Stateless and pure, like every
    other objective: ``SCALE * corner_distance + edge_distance``.
    """
    corner_total, edge_total = _cubie_distances(key)
    return CORNERS_FIRST_EDGE_SCALE * corner_total + edge_total


# The 20 blocks around the centres (8 corners, 12 edges): each block's stickers paired with their face centres.
_BLOCK_STICKER_CENTRES = tuple(tuple((position, (position // 9) * 9 + 4) for position in slot)
                               for slot in _CUBIE_SLOTS)


def edge_cost(key):
    """Blocks around the centres (8 corners, 12 edges) not showing all their correct colours (0-20)."""
    total = 0
    for block in _BLOCK_STICKER_CENTRES:
        for position, centre in block:
            if key[position] != key[centre]:
                total += 1
                break
    return total


# Lexicographic scale for cubie_refined_cost: the misplaced tier must never
# outweigh one unit of cubie distance. Misplaced counts only non-centre
# stickers (centres always match their own face), so it is at most 48 < 64.
_CUBIE_REFINED_CUBIE_SCALE = 64


def cubie_refined_cost(key):
    """Cubie distance, refined within its ties by misplaced stickers.

    ``cubie_cost`` alone leaves large plateaus of states with equal cost
    between which selection has no gradient except active length. This cost
    keeps the cubie distance strictly primary, then prefers more stickers
    showing their own face's colour: ``64 * cubie + misplaced``.
    """
    return _CUBIE_REFINED_CUBIE_SCALE * cubie_cost(key) + misplaced_cost(key)


@lru_cache(maxsize=1)
def _grouped_cubie_tables():
    """Build once, only when this fitness is used; no files or external solvers."""
    from grouped_cubie_fitness import GroupedCubieTables

    # Bound the lower tiers using the actual solo-distance tables.
    if 64 * sum(max(table.values()) for _, table in _CUBIE_TABLES) + 48 >= 4096:
        raise ConsistencyError("grouped-cubie scale no longer dominates its tie-breakers")
    positions = tuple(range(STICKER_COUNT))
    return GroupedCubieTables(_CUBIE_SLOTS, tuple(_MOVE_TABLES[a](positions) for a in REAL_MOVES))


def grouped_cubie_cost(key):
    """Joint four-piece distances, then solo distances, then misplaced stickers.

    P sums two four-corner and three four-edge exact pattern distances under
    all 18 moves. The sum is a heuristic, not a solution-length lower bound.
    4096 dominates 64*C+M (C<=52, M<=48); 64 dominates M. Search length and
    solved detection remain separate from this pure state cost.
    """
    return _grouped_cubie_tables().cost(_face_indices(key))


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
        if self.state_function == "cubie-refined":
            return cubie_refined_cost(key)
        if self.state_function == "grouped-cubie":
            return grouped_cubie_cost(key)
        if self.state_function == "corners-first":
            return corners_first_cost(key)
        if self.state_function == "edges":
            return edge_cost(key)
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
        if self.state_function == "cubie-refined":
            return (f"{self.name}: cubie distance first, then misplaced stickers "
                    "(64 * cubie + misplaced)")
        if self.state_function == "grouped-cubie":
            return (f"{self.name}: joint four-piece distances, then cubie, then misplaced "
                    "(4096 * pattern sum + 64 * cubie + misplaced; not a length lower bound)")
        if self.state_function == "corners-first":
            return (f"{self.name}: corner cubie distance first (slot and orientation), "
                    f"then edge cubie distance ({CORNERS_FIRST_EDGE_SCALE} * corners + edges; "
                    "edges count only once corners are home, and any corner regression dominates again)")
        if self.state_function == "edges":
            return (f"{self.name}: blocks around the centres (8 corners, 12 edges) "
                    "not showing all their correct colours")
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
        return Objective(name, name if name in ("misplaced", "cubie", "edges", "cubie-refined",
                                               "grouped-cubie", "corners-first") else "manhattan")
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
    # Persistent block library; all stay zero when the library is disabled.
    library_seeded_blocks: int = 0
    library_candidates_submitted: int = 0
    library_reference_evaluations: int = 0
    library_reference_cache_hits: int = 0
    library_additions: int = 0
    library_equivalent_merges: int = 0
    library_representative_updates: int = 0
    library_evictions: int = 0
    library_rejected_candidates: int = 0
    library_too_long_candidates: int = 0
    library_rejected_records: int = 0
    library_pending_payloads: int = 0
    library_checkpoints: int = 0
    library_saves: int = 0
    library_load_seconds: float = 0.0
    library_evaluation_seconds: float = 0.0
    library_write_seconds: float = 0.0


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
                 "target_length", "created_generation", "library_id")


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
    # Absolute path of the persistent block library, or None to disable it. The CLI
    # enables it by default; library callers opt in so tests and benchmarks stay isolated.
    block_library: Optional[str] = None
    # Diversity/stagnation warnings every 50 completed generations (measurement only).
    diversity_checks: bool = True
    diversity_min_unique_ratio: float = DEFAULT_DIVERSITY_MIN_UNIQUE_RATIO
    diversity_max_dominance: float = DEFAULT_DIVERSITY_MAX_DOMINANCE
    diversity_min_cubie_distance: float = DEFAULT_DIVERSITY_MIN_CUBIE_DISTANCE
    stagnation_generations: int = DEFAULT_STAGNATION_GENERATIONS


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
    library = config.block_library
    if library is not None and (not isinstance(library, str) or not library):
        raise ConfigError("block library must be a non-empty path or None")
    if type(config.diversity_checks) is not bool:
        raise ConfigError("diversity checks must be True or False")
    for name, value, upper in (("diversity minimum unique ratio", config.diversity_min_unique_ratio, 1),
                               ("diversity maximum dominance", config.diversity_max_dominance, 1),
                               ("diversity minimum cubie distance", config.diversity_min_cubie_distance, 20)):
        if (type(value) not in (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ConfigError(f"{name} must be a finite number in [0, {upper}], got {value!r}")
    _require_int("stagnation generations", config.stagnation_generations, 1)
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
    library_report: Optional[dict] = None
    diversity_records: list = field(default_factory=list)
    diversity_seconds: float = 0.0


@dataclass
class FinalReport:
    status: str
    moves: tuple
    cube: Optional[Cube]
    cost: Optional[float]
    solved: Optional[bool]
    problems: list


# ---------------------------------------------------------------------------
# Persistent move-block library (persistent_move_block_library_specs.md)
# ---------------------------------------------------------------------------

BLOCK_LIBRARY_SCHEMA_VERSION = 1
BLOCK_LIBRARY_EVALUATOR_VERSION = "cubie-refined-v1: 64 * solo cubie distance + misplaced stickers"
BLOCK_LIBRARY_CAPACITY = 100
BLOCK_LIBRARY_MAX_MOVES = 64
BLOCK_LIBRARY_MAX_BYTES = 1 << 20
BLOCK_LIBRARY_CHECKPOINT_INTERVAL = 50
BLOCK_LIBRARY_PENDING_LIMIT = 100
BLOCK_LIBRARY_CACHE_LIMIT = 4096
BLOCK_LIBRARY_LOCK_TIMEOUT = 2.0
BLOCK_LIBRARY_FILENAME = "move_blocks_v1.json"
REFERENCE_PANEL_LENGTHS = (2, 4, 8, 12, 20, 30)
REFERENCE_PANEL_SAMPLES = 8
REFERENCE_SOLVED_KEY = "".join(colour * 9 for colour in "GRWOYB")
_IDENTITY_PERMUTATION = tuple(range(STICKER_COUNT))


class BlockLibraryError(Exception):
    """A library file problem. The solver reports it once, preserves the file and continues."""

    def __init__(self, message, kind="corrupt"):
        super().__init__(message)
        self.kind = kind


def default_block_library_path():
    """``$XDG_DATA_HOME/rubik3x3solver/move_blocks_v1.json``, else under ``~/.local/share``."""
    base = os.environ.get("XDG_DATA_HOME")
    if not base or not os.path.isabs(base):
        base = os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "rubik3x3solver", BLOCK_LIBRARY_FILENAME)


def block_permutation(moves):
    """Permutation of a move sequence, in the ``_MOVE_TABLES`` convention.

    Applied to the labelled positions 0..53: the sticker at new position ``i``
    comes from old position ``permutation[i]``.
    """
    permutation = _IDENTITY_PERMUTATION
    for allele in moves:
        permutation = _MOVE_TABLES[allele](permutation)
    return permutation


def transformation_id(permutation):
    """SHA-256 of the canonical permutation encoding (one byte per position)."""
    return hashlib.sha256(bytes(permutation)).hexdigest()


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


MOVE_CONVENTION_ID = _canonical_digest([[MOVE_NAMES[allele], list(block_permutation((allele,)))]
                                        for allele in REAL_MOVES])


def _panel_allele(length, sample, attempt, position):
    text = f"rubik-block-library-v1|{length}|{sample}|{attempt}|{position}"
    return 1 + int.from_bytes(hashlib.sha256(text.encode("ascii")).digest(), "big") % 18


@lru_cache(maxsize=1)
def reference_panel():
    """The 48 versioned calibration states (8 scrambles at each length), solved and duplicates regenerated."""
    states, seen = [], set()
    for length in REFERENCE_PANEL_LENGTHS:
        for sample in range(REFERENCE_PANEL_SAMPLES):
            attempt = 0
            while True:
                key = REFERENCE_SOLVED_KEY
                for position in range(length):
                    key = apply_move_to_key(key, _panel_allele(length, sample, attempt, position))
                if not is_solved_key(key) and key not in seen:
                    break
                attempt += 1
            seen.add(key)
            states.append(key)
    return tuple(states)


@lru_cache(maxsize=1)
def reference_panel_digest():
    return _canonical_digest(list(reference_panel()))


def library_reference_cost(key):
    """The library's fixed, versioned evaluator, whatever the run's CLI objective."""
    return 64 * cubie_cost(key) + misplaced_cost(key)


@lru_cache(maxsize=1)
def _reference_baselines():
    return tuple((key, library_reference_cost(key)) for key in reference_panel())


def reference_score(permutation, check=None):
    """``(solved_count, useful_count, gain_ppm_sum)`` of the complete transformation on the panel.

    ``check`` runs before each panel state, so a budget stop leaves no partial score.
    """
    apply = itemgetter(*permutation)
    solved = useful = gain_ppm = 0
    for before, before_cost in _reference_baselines():
        if check is not None:
            check()
        after = "".join(apply(before))
        gain = before_cost - library_reference_cost(after)
        solved += is_solved_key(after)
        useful += gain >= 1
        gain_ppm += (1_000_000 * max(gain, 0)) // before_cost
    return (solved, useful, gain_ppm)


@dataclass(frozen=True)
class LibraryEntry:
    permutation: tuple
    moves: tuple
    score: tuple
    transformation_id: str

    def rank_key(self):
        solved, useful, gain_ppm = self.score
        return (-solved, -useful, -gain_ppm, len(self.moves), self.moves, self.transformation_id)

    def record(self):
        solved, useful, gain_ppm = self.score
        return {
            "transformation_id": self.transformation_id,
            "permutation": list(self.permutation),
            "moves": [MOVE_NAMES[allele] for allele in self.moves],
            "reference_score": {"solved_count": solved, "useful_count": useful, "gain_ppm_sum": gain_ppm},
        }


def _library_document(entries):
    records = [entry.record() for entry in entries]
    return {
        "schema_version": BLOCK_LIBRARY_SCHEMA_VERSION,
        "move_convention_id": MOVE_CONVENTION_ID,
        "evaluator_version": BLOCK_LIBRARY_EVALUATOR_VERSION,
        "reference_panel_digest": reference_panel_digest(),
        "revision": _canonical_digest(records),
        "entries": records,
    }


def _serialise_library(document):
    """JSON with one compact entry per line; still a single valid JSON object."""
    head = json.dumps({key: value for key, value in document.items() if key != "entries"}, indent=1)
    lines = ",\n".join("  " + json.dumps(record, separators=(",", ":")) for record in document["entries"])
    return f'{head[:-2]},\n "entries": [\n{lines}\n ]\n}}\n'


def _reject_json_constant(name):
    raise BlockLibraryError(f"library contains the non-finite number {name}")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BlockLibraryError(f"library repeats the key {key!r}")
        result[key] = value
    return result


def _read_library_file(path):
    """Parsed JSON document, or None for a missing file. Never executes stored data."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(BLOCK_LIBRARY_MAX_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BlockLibraryError(f"cannot read the library: {exc.strerror or exc}", "unreadable") from None
    if len(data) > BLOCK_LIBRARY_MAX_BYTES:
        raise BlockLibraryError("the library file exceeds 1 MiB")
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant,
                          object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise BlockLibraryError(f"the library is not valid UTF-8 JSON ({exc})") from None


_LIBRARY_KEYS = {"schema_version", "move_convention_id", "evaluator_version", "reference_panel_digest",
                 "revision", "entries"}
_LIBRARY_ENTRY_KEYS = {"transformation_id", "permutation", "moves", "reference_score"}
_LIBRARY_SCORE_KEYS = ("solved_count", "useful_count", "gain_ppm_sum")


def _parse_library_document(document, score):
    """Validated, rescored entries of a parsed library; raises BlockLibraryError on any defect."""
    if not isinstance(document, dict) or set(document) != _LIBRARY_KEYS:
        raise BlockLibraryError("the library has unexpected top-level fields")
    version = document["schema_version"]
    if type(version) is not int or version != BLOCK_LIBRARY_SCHEMA_VERSION:
        raise BlockLibraryError(f"unsupported library schema version {version!r}", "incompatible")
    if document["move_convention_id"] != MOVE_CONVENTION_ID:
        raise BlockLibraryError("the library uses a different move convention", "incompatible")
    if (document["evaluator_version"] != BLOCK_LIBRARY_EVALUATOR_VERSION
            or document["reference_panel_digest"] != reference_panel_digest()):
        # Version 1 is the first evaluator/panel, so no older version can be migrated yet.
        raise BlockLibraryError("the library uses an unknown evaluator or reference panel", "incompatible")
    records = document["entries"]
    if not isinstance(records, list) or len(records) > BLOCK_LIBRARY_CAPACITY:
        raise BlockLibraryError("the library entry list is missing or longer than 100")
    if document["revision"] != _canonical_digest(records):
        raise BlockLibraryError("the library revision does not match its entries")
    entries, seen = [], set()
    for index, record in enumerate(records):
        problem = _library_record_problem(record)
        if problem is None:
            alleles = tuple(ALLELE_BY_NAME[token] for token in record["moves"])
            permutation = tuple(record["permutation"])
            if simplify(alleles) != alleles:
                problem = "moves are not reduced"
            elif block_permutation(alleles) != permutation:
                problem = "permutation does not match a replay of its moves"
            elif permutation == _IDENTITY_PERMUTATION:
                problem = "identity transformation"
            elif transformation_id(permutation) != record["transformation_id"]:
                problem = "transformation_id does not match its permutation"
            elif permutation in seen:
                problem = "duplicate transformation"
            else:
                stored = tuple(record["reference_score"][name] for name in _LIBRARY_SCORE_KEYS)
                recomputed = score(permutation)
                if stored != recomputed:
                    problem = "stored reference score differs from the recomputed one"
                elif recomputed[1] < 1:
                    problem = "entry is useful on no reference state"
        if problem is not None:
            raise BlockLibraryError(f"library entry {index} is invalid: {problem}")
        seen.add(permutation)
        entries.append(LibraryEntry(permutation, alleles, recomputed, record["transformation_id"]))
    return entries


def _library_record_problem(record):
    if not isinstance(record, dict) or set(record) != _LIBRARY_ENTRY_KEYS:
        return "unexpected fields"
    identifier = record["transformation_id"]
    if not isinstance(identifier, str) or len(identifier) != 64 or set(identifier) - set("0123456789abcdef"):
        return "transformation_id is not a SHA-256 hex digest"
    permutation = record["permutation"]
    if (not isinstance(permutation, list) or len(permutation) != STICKER_COUNT
            or any(type(value) is not int for value in permutation)
            or sorted(permutation) != list(_IDENTITY_PERMUTATION)):
        return "permutation is not a bijection of 0..53"
    moves = record["moves"]
    if (not isinstance(moves, list) or not 1 <= len(moves) <= BLOCK_LIBRARY_MAX_MOVES
            or any(not isinstance(token, str) or token == "NOOP" or token not in ALLELE_BY_NAME for token in moves)):
        return "moves are not 1-64 real move tokens"
    score = record["reference_score"]
    if not isinstance(score, dict) or set(score) != set(_LIBRARY_SCORE_KEYS):
        return "reference_score has unexpected fields"
    values = [score[name] for name in _LIBRARY_SCORE_KEYS]
    if any(type(value) is not int or value < 0 for value in values):
        return "reference_score values are not nonnegative integers"
    panel_size = len(REFERENCE_PANEL_LENGTHS) * REFERENCE_PANEL_SAMPLES
    if values[0] > values[1] or values[1] > panel_size or values[2] > panel_size * 1_000_000:
        return "reference_score values are out of range"
    return None


@contextlib.contextmanager
def _library_lock(lock_path, timeout):
    """Exclusive OS advisory lock on a companion file, released when the process exits."""
    try:
        import fcntl
    except ImportError:
        raise BlockLibraryError("advisory file locking is unavailable on this platform", "lock") from None
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise BlockLibraryError(f"timed out after {timeout:g}s waiting for {lock_path}", "lock") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _atomic_write(path, text):
    """Write to a temporary file in the destination directory, fsync, then replace atomically."""
    data = text.encode("utf-8")
    if len(data) > BLOCK_LIBRARY_MAX_BYTES:
        raise BlockLibraryError("the library would exceed 1 MiB", "unwritable")
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".move_blocks_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    with contextlib.suppress(OSError):
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


class BlockLibrary:
    """Bounded archive of the best distinct transformations, synchronised with one JSON file.

    Membership and order come only from the fixed reference panel; runtime block
    fitness never enters the archive and archive scores never enter DEAP fitness.
    """

    def __init__(self, path, *, counters=None, report=None, lock_timeout=BLOCK_LIBRARY_LOCK_TIMEOUT):
        self.path = os.path.abspath(path)
        self.counters = counters if counters is not None else SearchCounters()
        self.report = report
        self.lock_timeout = lock_timeout
        self.entries = {}
        self.usable = True
        self.writable = True
        self.diagnostics = []
        self.snapshot_digest = None
        self.loaded_count = 0
        self._scores = OrderedDict()
        self._fingerprints = OrderedDict()

    def _diagnose(self, message):
        if message not in self.diagnostics:
            self.diagnostics.append(message)
            if self.report is not None:
                self.report(f"Block library: {message}")

    def _score(self, permutation, check=None):
        key = (BLOCK_LIBRARY_EVALUATOR_VERSION, reference_panel_digest(), MOVE_CONVENTION_ID, permutation)
        score = self._scores.get(key)
        if score is not None:
            self._scores.move_to_end(key)
            self.counters.library_reference_cache_hits += 1
            return score
        if check is not None:
            check()  # Do not even start an uncached panel evaluation after a stop.
        score = reference_score(permutation, check)
        self.counters.library_reference_evaluations += 1
        self._scores[key] = score
        if len(self._scores) > BLOCK_LIBRARY_CACHE_LIMIT:
            self._scores.popitem(last=False)
        return score

    def fingerprint(self, moves):
        """``(reduced moves, permutation)`` for a candidate, cached per distinct reduced sequence."""
        reduced = simplify(moves)
        permutation = self._fingerprints.get(reduced)
        if permutation is None:
            permutation = block_permutation(reduced)
            self._fingerprints[reduced] = permutation
            if len(self._fingerprints) > BLOCK_LIBRARY_CACHE_LIMIT:
                self._fingerprints.popitem(last=False)
        else:
            self._fingerprints.move_to_end(reduced)
        return reduced, permutation

    def ranked(self):
        return sorted(self.entries.values(), key=LibraryEntry.rank_key)

    def _merge(self, entry, *, count=True):
        current = self.entries.get(entry.permutation)
        if current is None:
            self.entries[entry.permutation] = entry
            return
        if count:
            self.counters.library_equivalent_merges += 1
        if (len(entry.moves), entry.moves) < (len(current.moves), current.moves):
            self.entries[entry.permutation] = entry
            if count:
                self.counters.library_representative_updates += 1

    def _trim(self):
        if len(self.entries) > BLOCK_LIBRARY_CAPACITY:
            self.entries = {entry.permutation: entry for entry in self.ranked()[:BLOCK_LIBRARY_CAPACITY]}

    def submit(self, candidates, check=None):
        """Reference-evaluate candidate move sequences and keep the eligible best 100.

        Candidates are deduplicated by transformation first. A budget stop raised
        by ``check`` leaves only fully evaluated candidates in the archive.
        """
        if not self.usable:
            return
        started = time.monotonic()
        try:
            distinct = {}
            for moves in candidates:
                self.counters.library_candidates_submitted += 1
                reduced, permutation = self.fingerprint(moves)
                if permutation == _IDENTITY_PERMUTATION:
                    self.counters.library_rejected_candidates += 1
                    continue
                if len(reduced) > BLOCK_LIBRARY_MAX_MOVES:
                    self.counters.library_too_long_candidates += 1
                    continue
                current = distinct.get(permutation)
                if current is None or (len(reduced), reduced) < (len(current), current):
                    distinct[permutation] = reduced
            before = set(self.entries)
            for permutation, reduced in distinct.items():
                score = self._score(permutation, check)
                if score[1] < 1:
                    self.counters.library_rejected_candidates += 1
                    continue
                self._merge(LibraryEntry(permutation, reduced, score, transformation_id(permutation)))
                self._trim()
            after = set(self.entries)
            self.counters.library_additions += len(after - before)
            self.counters.library_evictions += len(before - after)
        finally:
            self.counters.library_evaluation_seconds += time.monotonic() - started

    def load(self):
        """Read and validate one snapshot; on failure the library is unused for this run and the file kept."""
        started = time.monotonic()
        try:
            document = _read_library_file(self.path)
            entries = [] if document is None else _parse_library_document(document, self._score)
        except BlockLibraryError as exc:
            self.usable = self.writable = False
            self.counters.library_rejected_records += 1
            self._diagnose(f"{exc} ({self.path}); the file is preserved and the library is unused for this run")
            return False
        finally:
            self.counters.library_load_seconds += time.monotonic() - started
        self.entries = {entry.permutation: entry for entry in entries}
        self.loaded_count = len(entries)
        self.snapshot_digest = _canonical_digest([entry.record() for entry in self.ranked()])
        return True

    def save(self, check=None):
        """Merge and atomically replace; defer if uncached validation exceeds the budget.

        Cached, previously verified scores can still be merged after a stop.
        New entries from another writer must be validated before any replacement;
        if that needs unavailable search time, leave the entire file untouched.
        """
        if not (self.usable and self.writable):
            return False
        started = time.monotonic()
        try:
            if not self.entries and not os.path.exists(self.path):
                return False
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with _library_lock(self.path + ".lock", self.lock_timeout):
                document = _read_library_file(self.path)
                if document is not None:
                    for entry in _parse_library_document(document, lambda permutation: self._score(permutation, check)):
                        self._merge(entry, count=False)
                    self._trim()
                updated = _library_document(self.ranked())
                if document is not None and document["entries"] == updated["entries"]:
                    return False
                if not updated["entries"]:
                    return False
                _atomic_write(self.path, _serialise_library(updated))
                self.counters.library_saves += 1
                return True
        except SearchStopped as exc:
            self._diagnose(f"save deferred: reference validation stopped ({exc}); "
                           f"the file is preserved ({self.path})")
            return False
        except BlockLibraryError as exc:
            if exc.kind in ("corrupt", "incompatible"):
                self.writable = False
            self._diagnose(f"{exc} ({self.path}); the file is preserved")
            return False
        except OSError as exc:
            self._diagnose(f"cannot write the library: {exc.strerror or exc} ({self.path}); the file is preserved")
            return False
        finally:
            self.counters.library_write_seconds += time.monotonic() - started


# ---------------------------------------------------------------------------
# Solution-population diversity and stagnation (solution_population_diversity_specs.md)
# ---------------------------------------------------------------------------

DIVERSITY_CONDITIONS = ("LOW_UNIQUE_STATES", "DOMINANT_STATE", "LOW_CUBIE_SPREAD", "NO_QUALITY_PROGRESS")


def decode_cubie_poses(key):
    """Position and orientation of each of the 20 cubies, relative to the face centres.

    Returns ``(cubie, pose)`` pairs: the cubie is its sorted face-index colours and the
    pose is ``(slot, colours in slot order)``. Raises ValueError when the stickers do not
    form a distinct, consistent set of cubies. This is not a proof of solvability.
    """
    faces = _face_indices(key)
    poses, seen = [], set()
    for slot, (getter, distances) in enumerate(_CUBIE_TABLES):
        label = "".join(getter(faces))
        if label not in distances:
            raise ValueError("the stickers do not form a consistent set of cubies")
        cubie = "".join(sorted(label))
        if cubie in seen:
            raise ValueError("the stickers repeat a cubie")
        seen.add(cubie)
        poses.append((cubie, (slot, label)))
    return tuple(poses)


def mean_cubie_difference(multiplicities, poses):
    """Exact mean over unordered candidate pairs of cubies differing in position or orientation.

    ``multiplicities`` maps endpoint -> candidate count and ``poses`` maps endpoint -> decoded
    cubies. Counts agreeing pairs per cubie and pose, so the work is O(20 * unique endpoints).
    Returns ``(mean, distance_sum, pair_count)``; the mean is unavailable (None) below two candidates.
    """
    count = sum(multiplicities.values())
    pair_count = count * (count - 1) // 2
    if pair_count == 0:
        return None, 0, 0
    frequencies = defaultdict(int)
    for endpoint, multiplicity in multiplicities.items():
        for cubie, pose in poses[endpoint]:
            frequencies[(cubie, pose)] += multiplicity
    agreeing = defaultdict(int)
    for (cubie, _), frequency in frequencies.items():
        agreeing[cubie] += frequency * (frequency - 1) // 2
    distance_sum = sum(pair_count - pairs for pairs in agreeing.values())
    return distance_sum / pair_count, distance_sum, pair_count


def _subset_diversity(multiplicities, poses, decode_error):
    count = sum(multiplicities.values())
    metrics = {"count": count, "unique": len(multiplicities), "unique_ratio": None,
               "dominant": max(multiplicities.values(), default=0), "dominant_ratio": None,
               "cubie_mean": None, "cubie_reason": None}
    if count:
        metrics["unique_ratio"] = metrics["unique"] / count
        metrics["dominant_ratio"] = metrics["dominant"] / count
    if decode_error is not None:
        metrics["cubie_reason"] = decode_error
    elif count < 2:
        metrics["cubie_reason"] = "fewer than two candidates"
    else:
        metrics["cubie_mean"] = mean_cubie_difference(multiplicities, poses)[0]
    return metrics


def measure_population_diversity(population, expected_size):
    """Diversity metrics of the published population and of its non-reset part.

    Reads only each candidate's verified endpoint key and current slot; never mutates
    candidates, replays chromosomes, evaluates costs or uses randomness. Decoded cubies
    live only for this call.
    """
    record = {"available": True, "reason": None, "sizes": None, "whole": None, "non_reset": None}
    try:
        endpoints = [(individual.endpoint, individual.slot) for individual in population]
    except (AttributeError, TypeError):
        record.update(available=False, reason="population is missing or lacks endpoint/slot metadata")
        return record
    if any(not isinstance(endpoint, str) or len(endpoint) != STICKER_COUNT for endpoint, _ in endpoints):
        record.update(available=False, reason="a candidate has no valid 54-sticker endpoint")
        return record
    reset = sum(1 for _, slot in endpoints if slot == "reset")
    record["sizes"] = {"whole": len(endpoints), "non_reset": len(endpoints) - reset, "reset": reset}
    if len(endpoints) != expected_size or reset != expected_size // 5:
        record.update(available=False,
                      reason=f"population roles are inconsistent: {len(endpoints)} candidates with {reset} "
                             f"reset slots, expected {expected_size} with {expected_size // 5}")
        return record
    whole = Counter(endpoint for endpoint, _ in endpoints)
    non_reset = Counter(endpoint for endpoint, slot in endpoints if slot != "reset")
    poses, decode_error = {}, None
    for endpoint in whole:
        try:
            poses[endpoint] = decode_cubie_poses(endpoint)
        except ValueError as exc:
            decode_error = f"{exc}, so the cubie spread is unavailable"
            break
    record["whole"] = _subset_diversity(whole, poses, decode_error)
    record["non_reset"] = _subset_diversity(non_reset, poses, decode_error)
    return record


def diversity_conditions(record, generations_without_quality, thresholds):
    """Active condition IDs with their messages; diversity thresholds use the non-reset subset."""
    active = []
    subset = record["non_reset"] if record["available"] else None
    if subset is not None and subset["count"]:
        if subset["unique_ratio"] < thresholds["min_unique_ratio"]:
            active.append(("LOW_UNIQUE_STATES",
                           f"{subset['unique']}/{subset['count']} non-reset endpoints = "
                           f"{100 * subset['unique_ratio']:.1f}% (threshold <{100 * thresholds['min_unique_ratio']:.1f}%)"))
        if subset["dominant_ratio"] > thresholds["max_dominance"]:
            active.append(("DOMINANT_STATE",
                           f"{subset['dominant']}/{subset['count']} share one endpoint = "
                           f"{100 * subset['dominant_ratio']:.2f}% (threshold >{100 * thresholds['max_dominance']:.1f}%)"))
        if subset["cubie_mean"] is not None and subset["cubie_mean"] < thresholds["min_cubie_distance"]:
            active.append(("LOW_CUBIE_SPREAD",
                           f"mean differing cubies = {subset['cubie_mean']:.2f}/20 "
                           f"(threshold <{thresholds['min_cubie_distance']:g})"))
    if generations_without_quality >= thresholds["stagnation_generations"]:
        active.append(("NO_QUALITY_PROGRESS",
                       f"{generations_without_quality} generations without better cube quality "
                       f"(threshold >={thresholds['stagnation_generations']})"))
    return active


def _subset_text(metrics, name):
    unique = f"{metrics['unique']}/{metrics['count']} ({100 * metrics['unique_ratio']:.1f}%) {name}"
    dominant = f"{metrics['dominant']}/{metrics['count']} ({100 * metrics['dominant_ratio']:.2f}%) {name}"
    spread = (f"{metrics['cubie_mean']:.2f}/20 {name}" if metrics["cubie_mean"] is not None
              else f"unavailable {name} ({metrics['cubie_reason']})")
    return unique, dominant, spread


def diversity_report_lines(record, previous_flagged):
    """Summary, then one grouped warning or a single recovery note. Pure: returns text only."""
    generation = record["generation"]
    lines = []
    sizes = record["sizes"]
    if sizes is not None:
        lines.append(f"[gen {generation}] solution-population diversity ({sizes['whole']} candidates: "
                     f"{sizes['non_reset']} non-reset, {sizes['reset']} reset)")
    else:
        lines.append(f"[gen {generation}] solution-population diversity")
    if record["available"] and record["non_reset"]["count"]:
        unique_n, dominant_n, spread_n = _subset_text(record["non_reset"], "non-reset")
        unique_w, dominant_w, spread_w = _subset_text(record["whole"], "whole")
        lines.append(f"  unique endpoints {unique_n}, {unique_w}; dominant endpoint {dominant_n}, {dominant_w}")
        lines.append(f"  mean differing cubies {spread_n}, {spread_w}")
    else:
        lines.append(f"  diversity metrics unavailable: {record['reason'] or 'no non-reset candidates'}")
    flags = record["flags"]
    lines.append(f"  {record['generations_without_quality_improvement']} generations since better cube quality; "
                 f"flags: {', '.join(flags) if flags else 'none'}")
    if flags:
        diversity_flags = [flag for flag in flags if flag != "NO_QUALITY_PROGRESS"]
        if diversity_flags and "NO_QUALITY_PROGRESS" in flags:
            kind = "low diversity with stalled cube-quality progress"
        elif diversity_flags:
            kind = "low solution-population diversity"
        else:
            kind = "no cube-quality progress (a progress warning; the population is not reported as homogeneous)"
        lines.append(f"[gen {generation}] WARNING: {kind}")
        lines.extend(f"  {flag}: {message}" for flag, message in record["messages"])
        if diversity_flags:
            lines.append(f"  Intentional reset slots excluded from thresholds: {sizes['reset']}/{sizes['whole']}.")
    elif previous_flagged:
        lines.append(f"[gen {generation}] Diversity/progress recovered: none of the warning thresholds is "
                     "currently crossed (this does not guarantee convergence).")
    return lines


# ---------------------------------------------------------------------------
# The cooperative search
# ---------------------------------------------------------------------------


class CoevolutionRun:
    """One reproducible run: owns its RNG, counters, archive and both populations."""

    def __init__(self, config, input_key, objective, seed, *, budget=None, log=None, on_improvement=None):
        self.config = config
        self.objective = objective
        self.seed = seed
        self.rng = random.Random(seed)
        self.budget = budget if budget is not None else Budget(config.time_limit)
        self.log = log if log is not None else _print_flush
        # Called as on_improvement(run, entry) after every verified new best-ever entry.
        self.on_improvement = on_improvement
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
        self.last_quality_improvement_generation = 0
        self.diversity_records = []
        self.diversity_seconds = 0.0
        self._diversity_previous_flagged = False
        self._diversity_consecutive_flagged = 0
        self.last_block_summary = None
        self.history = tools.Logbook()

        self._last_report_marker = None
        if objective.state_function in ("cubie", "cubie-refined", "grouped-cubie", "corners-first"):
            validate_cubie_input(input_key)
        baseline = self.normalize_solution((), check_budget=False)
        self.baseline = baseline
        self.baseline_rank = baseline.rank
        self.archive = ArchiveEntry(baseline.active, baseline.endpoint, baseline.cost,
                                    baseline.solved, None, 0, "empty baseline")

        # Persistent block library: one validated snapshot, read before the search timer
        # starts and never again during this run. Skipped when no block search is needed.
        self.library = None
        self.library_status = "disabled" if config.block_library is None else "not needed"
        self._library_seeds = []
        self._library_capacity_compatible = 0
        self._library_snapshot = None
        self._library_pending = {}
        self._library_flush_requested = False
        if config.block_library is not None and not baseline.solved and config.generations > 0:
            self.library = BlockLibrary(config.block_library, counters=self.counters, report=self.log)
            if self.library.load():
                self.library_status = "enabled"
                eligible = [entry for entry in self.library.ranked() if len(entry.moves) <= self.block_capacity]
                self._library_capacity_compatible = len(eligible)
                self._library_seeds = eligible[:config.block_population // 4]
            else:
                self.library_status = "unusable"

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
        block.library_id = None
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

    def _remove_state_loops(self, sequence):
        """Delete moves between two visits to the same state, then re-reduce.

        Cutting a loop between equal states (all candidates start from the
        original cube) preserves the final state, and simplification after a
        cut preserves the permutation, so the endpoint, cost and solved flag
        are unchanged; ``_verify_candidate`` still replays the result.
        """
        sequence = tuple(sequence)
        for _ in range(len(sequence) + 1):
            seen = {self.input_key: 0}
            states = [self.input_key]
            kept = []
            key = self.input_key
            for allele in sequence:
                key = apply_move_to_key(key, allele, self.counters)
                if key in seen:
                    start = seen[key]
                    kept = kept[:start]
                    states = states[:start + 1]
                    seen = {state: index for index, state in enumerate(states)}
                else:
                    kept.append(allele)
                    states.append(key)
                    seen[key] = len(kept)
            shortened = simplify(tuple(kept))
            if shortened == sequence:
                return sequence
            sequence = shortened
        raise ConsistencyError("state-loop removal did not stabilise")

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
        chosen = self._remove_state_loops(chosen)
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

    def offer_to_archive(self, candidate, *, uid, source, block_id=None, block_payload=None):
        """Replace the best-ever entry only on a strictly better, verified rank.

        ``block_payload`` is the exact block sequence applied to produce the candidate;
        it becomes a pending library candidate when the archive improves.
        """
        if not candidate.rank < self.archive.rank:
            return False
        self._verify_candidate(candidate)
        # Cube quality is (unsolved, H) only; a shorter sequence at equal H does not count.
        quality_improved = ((0 if candidate.solved else 1, candidate.cost)
                            < (0 if self.archive.solved else 1, self.archive.cost))
        self.archive = ArchiveEntry(candidate.active, candidate.endpoint, candidate.cost,
                                    candidate.solved, uid, self.generation_in_progress, source, block_id)
        self.counters.archive_updates += 1
        self.last_improvement_generation = self.generation_in_progress
        if quality_improved:
            self.last_quality_improvement_generation = self.generation_in_progress
        if block_payload and self.library is not None and self.library.usable:
            self._library_note_payload(tuple(block_payload), self.archive.rank)
        if self.on_improvement is not None:
            self.on_improvement(self, self.archive)
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
        if self.offer_to_archive(candidate, uid=None, source=source, block_id=block_uid,
                                 block_payload=sequence[:outcome.k]):
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
        self.offer_to_archive(candidate, uid=child.uid, source=operator, block_id=block_id, block_payload=payload)
        return child

    def _ordinary_offspring(self, pool):
        rng = self.rng
        parent = tournament_select(pool, lambda individual: individual.rank, rng)
        parent_active = parent.active
        parents = (parent.uid,)
        block_id = None
        applied_block = None
        operator = self._choose_solution_operator()
        if operator == "append-block":
            payload, block_id, targeted = self.choose_append_payload(parent_active, parent.endpoint)
            operator = "append-targeted" if targeted else "append-exploratory"
            applied_block = payload
            proposal = parent_active + payload
            bound = self.solution_capacity + self.block_capacity
        elif operator == "splice-block":
            block, block_id = self._exploratory_block()
            applied_block = block
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
                applied_block = proposal
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
        self.offer_to_archive(candidate, uid=child.uid, source=operator, block_id=block_id,
                              block_payload=applied_block)
        return child

    # -- block population (section 8) ---------------------------------------------------

    def initial_block_population(self):
        """Library seeds first (rank order, no RNG), then fresh blocks on the usual schedule."""
        counts = {}
        blocks = []
        for entry in self._library_seeds:
            counts[entry.moves] = counts.get(entry.moves, 0) + 1
            block = self.make_block(entry.moves, origin="library", operator="library")
            block.library_id = entry.transformation_id
            blocks.append(block)
        self.counters.library_seeded_blocks = len(blocks)
        schedule = reduced_length_schedule(self.config.block_population - len(blocks), self.block_capacity, self.rng)
        blocks.extend(self.fresh_block(target, counts, origin="initial") for target in schedule)
        return blocks

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
        if self.library is not None and self.library.usable:
            # The last complete block evaluation, taken before survivors are replaced.
            self._library_snapshot = tuple((block.active, block.evaluation) for block in self._block_representatives)
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
                # Measured on the just-published population, before any library maintenance.
                diversity = None
                if self.config.diversity_checks and generation % DIVERSITY_CHECK_INTERVAL == 0:
                    diversity = self.diversity_checkpoint(generation)
                self._record_history(generation, diversity=diversity)
                if self.library is not None and self.library.usable:
                    scheduled = generation == 1 or generation % BLOCK_LIBRARY_CHECKPOINT_INTERVAL == 0
                    if scheduled or self._library_flush_requested:
                        self._library_checkpoint(include_snapshot=scheduled)
                if generation % self.config.log_every == 0:
                    self.report_progress()
            self._library_final_candidates()
            return self._result(STATUS_BUDGET_EXHAUSTED, "generation budget exhausted")
        except SolutionFound:
            self._library_final_candidates()
            return self._result(STATUS_SOLVED, "verified solve found")
        except SearchStopped as stop:
            # No new reference evaluations after a time limit or SIGINT; completed ones are kept.
            return self._result(stop.status, str(stop) or stop.status)

    # -- persistent block library (persistent_move_block_library_specs.md) ---------------

    def _library_note_payload(self, payload, solution_rank):
        """Keep at most 100 pending payloads: best resulting solution rank, then shorter, then allele order."""
        if self.library is None or not self.library.usable:
            return
        order = (solution_rank, len(payload), payload)
        current = self._library_pending.get(payload)
        if current is None or order < current:
            self._library_pending[payload] = order
        if len(self._library_pending) > BLOCK_LIBRARY_PENDING_LIMIT:
            worst = max(self._library_pending, key=self._library_pending.__getitem__)
            del self._library_pending[worst]
        self.counters.library_pending_payloads += 1
        if len(self._library_pending) >= BLOCK_LIBRARY_PENDING_LIMIT:
            self._library_flush_requested = True

    def _library_snapshot_candidates(self):
        """Per distinct block: the full sequence if useful on a context, plus its best feasible useful prefix."""
        for sequence, evaluation in self._library_snapshot or ():
            if any(outcome.qualifying and outcome.qualifying[-1].k == len(sequence)
                   for outcome in evaluation.outcomes):
                yield sequence
            best = None
            for outcome in evaluation.outcomes:
                prefix = outcome.best
                if prefix is None:
                    continue
                order = (_prefix_preference(prefix.solved, prefix.gain, prefix.representative_length, prefix.k),
                         outcome.context_id)
                if best is None or order < best[0]:
                    best = (order, prefix.k)
            if best is not None:
                yield sequence[:best[1]]

    def _drain_library_pending(self):
        payloads = sorted(self._library_pending, key=self._library_pending.__getitem__)
        self._library_pending = {}
        self._library_flush_requested = False
        return payloads

    def _library_checkpoint(self, include_snapshot):
        """Reference-evaluate this checkpoint's candidates (inside the search budget), then merge-save."""
        if self.library is None or not self.library.usable:
            return
        candidates = list(self._library_snapshot_candidates()) if include_snapshot else []
        candidates.extend(self._drain_library_pending())
        self.library.submit(candidates, check=self.budget.check)
        self.library.save(check=self.budget.check)
        self.counters.library_checkpoints += 1
        self.log(f"Block library checkpoint (generation {self.completed_generation}): "
                 f"{len(self.library.entries)} entries, {self.counters.library_additions} added, "
                 f"{self.counters.library_equivalent_merges} equivalent merges, "
                 f"{self.counters.library_evictions} evicted")

    def _library_final_candidates(self):
        """Bounded final batch after a normal finish or a solve; stops at the budget without partial scores."""
        if self.library is None or not self.library.usable:
            return
        candidates = list(self._library_snapshot_candidates()) + self._drain_library_pending()
        try:
            self.library.submit(candidates, check=self.budget.check)
        except SearchStopped:
            pass

    def finish_block_library(self, allow_save=True):
        """Merge-save the completed archive (timed separately from the search) and return a report."""
        if self.library is None:
            return {"status": self.library_status, "path": self.config.block_library}
        if allow_save:
            self.library.save(check=self.budget.check)
        return self.library_report()

    def library_report(self):
        counters = self.counters
        library = self.library
        return {
            "status": self.library_status,
            "path": self.config.block_library,
            "entries": len(library.entries) if library is not None else 0,
            "loaded": library.loaded_count if library is not None else 0,
            "capacity_compatible": self._library_capacity_compatible,
            "seeded": [entry.transformation_id for entry in self._library_seeds],
            "snapshot": library.snapshot_digest if library is not None else None,
            "diagnostics": list(library.diagnostics) if library is not None else [],
            "additions": counters.library_additions,
            "equivalent_merges": counters.library_equivalent_merges,
            "representative_updates": counters.library_representative_updates,
            "evictions": counters.library_evictions,
            "rejected_candidates": counters.library_rejected_candidates + counters.library_too_long_candidates,
            "rejected_records": counters.library_rejected_records,
            "reference_evaluations": counters.library_reference_evaluations,
            "reference_cache_hits": counters.library_reference_cache_hits,
            "load_seconds": counters.library_load_seconds,
            "evaluation_seconds": counters.library_evaluation_seconds,
            "write_seconds": counters.library_write_seconds,
        }

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
            diversity_records=list(self.diversity_records),
            diversity_seconds=self.diversity_seconds,
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

    def _record_history(self, generation, partial=False, diversity=None):
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
        if diversity is not None:
            # Set after recording: Logbook.record() would file a dict argument away as a chapter.
            # Only checkpoint rows carry a record; other rows never repeat an old one.
            self.history[-1]["diversity"] = diversity

    def diversity_checkpoint(self, generation):
        """Measure the published solution population, log the summary/warnings and return the record.

        Measurement only: no RNG use, no cost evaluation, no change to any candidate. The work is
        bounded and counts toward the search budget, so it completes once started.
        """
        started = time.monotonic()
        config = self.config
        record = measure_population_diversity(self.population, config.solution_population)
        record["generation"] = generation
        record["generations_without_quality_improvement"] = generation - self.last_quality_improvement_generation
        record["thresholds"] = {
            "min_unique_ratio": config.diversity_min_unique_ratio,
            "max_dominance": config.diversity_max_dominance,
            "min_cubie_distance": config.diversity_min_cubie_distance,
            "stagnation_generations": config.stagnation_generations,
        }
        active = diversity_conditions(record, record["generations_without_quality_improvement"],
                                      record["thresholds"])
        record["flags"] = [flag for flag, _ in active]
        record["messages"] = active
        self._diversity_consecutive_flagged = self._diversity_consecutive_flagged + 1 if active else 0
        record["consecutive_flagged"] = self._diversity_consecutive_flagged
        record["seconds"] = time.monotonic() - started
        self.diversity_seconds += record["seconds"]
        self.diversity_records.append(record)
        for line in diversity_report_lines(record, self._diversity_previous_flagged):
            self.log(line)
        self._diversity_previous_flagged = bool(active)
        return record

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

    def top_blocks(self, count=2):
        """Best distinct block sequences already scored on the current batch, and how many were scored."""
        if self.batch is None or not self.blocks:
            return [], 0
        scored = [block for block in self.blocks
                  if block.evaluation is not None and block.batch_serial == self.batch.serial]
        top, seen = [], set()
        for block in sorted(scored, key=_block_sort_key):
            if block.active not in seen:
                seen.add(block.active)
                top.append(block)
                if len(top) == count:
                    break
        return top, len(scored)

    def find_block(self, uid):
        """The block with this ID in the current block population, or None."""
        for block in self.blocks or ():
            if block.uid == uid:
                return block
        return None


# ---------------------------------------------------------------------------
# Library entry points, finalisation and plotting
# ---------------------------------------------------------------------------


def _print_flush(message=""):
    print(message, flush=True)


def _print_cube(cube):
    """Coloured net via cube.py, then undo the stream wrapping its colorama.init() adds on every call."""
    cube.print_colored()
    colorama.deinit()


def run_search(config, input_cube=None, *, budget=None, log=None):
    """Validate, then run one search; ``input_cube`` defaults to :func:`default_input_cube`."""
    objective = validate_config(config)
    cube = default_input_cube() if input_cube is None else input_cube
    input_key = validate_input_cube(cube)
    seed = config.seed if config.seed is not None else generate_seed()
    run = CoevolutionRun(config, input_key, objective, seed, budget=budget, log=log)
    result = run.run()
    if config.block_library is not None:
        # Persist only after the archived result survives independent replay.
        result.library_report = run.finish_block_library(allow_save=finalize_result(result).status != STATUS_ERROR)
    return result


_SHORTEN_INDEX = None


def _shorten_index():
    """Permutation -> shortest canonical sequence of at most four moves, built lazily once."""
    global _SHORTEN_INDEX
    if _SHORTEN_INDEX is None:
        index = {}
        for length in range(5):
            for candidate in iter_canonical_sequences(length):
                index.setdefault(bytes(block_permutation(candidate)), candidate)
        _SHORTEN_INDEX = index
    return _SHORTEN_INDEX


def shorten_sequence(sequence):
    """Shorten by window replacement: same permutation, never a longer result.

    Any contiguous window whose permutation equals a canonical sequence of
    at most four moves is replaced by that sequence, re-reducing after each
    splice, until no window improves. Deterministic and purely a function
    of the sequence's permutation, so the end state from any start is kept.
    This is a reporting-time polish, deliberately not used per candidate.
    """
    moves = simplify(sequence)
    if len(moves) <= 4:
        return moves
    index = _shorten_index()
    while True:
        hit = None
        for width in range(len(moves), 4, -1):
            for start in range(len(moves) - width + 1):
                replacement = index.get(bytes(block_permutation(moves[start:start + width])))
                if replacement is not None and len(replacement) < width:
                    hit = (start, width, replacement)
                    break
            if hit is not None:
                break
        if hit is None:
            return moves
        start, width, replacement = hit
        moves = simplify(moves[:start] + replacement + moves[start + width:])


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
    # Report a window-shortened equivalent; the replay below still compares
    # its endpoint, cost and solved flag against the archived metadata.
    moves = shorten_sequence(moves)
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
    library = parser.add_mutually_exclusive_group()
    library.add_argument("--block-library", metavar="PATH", default=None,
                         help="override the shared move-block library file (default: "
                              "$XDG_DATA_HOME/rubik3x3solver/move_blocks_v1.json, else ~/.local/share/...)")
    library.add_argument("--no-block-library", action="store_true",
                         help="disable all library reads, writes, reference scoring and population seeding")
    diversity = parser.add_argument_group(
        "solution-population diversity checks",
        f"Measured every {DIVERSITY_CHECK_INTERVAL} completed generations; warnings only, the search is unchanged. "
        "The three diversity thresholds use the non-reset candidates (elites and offspring); the current 20% "
        "reset slots are excluded. Low diversity is reported separately from stalled cube-quality progress.")
    diversity.add_argument("--diversity-min-unique-ratio", type=float, default=DEFAULT_DIVERSITY_MIN_UNIQUE_RATIO,
                           help="warn when unique endpoints / candidates is below this (default 0.10)")
    diversity.add_argument("--diversity-max-dominance", type=float, default=DEFAULT_DIVERSITY_MAX_DOMINANCE,
                           help="warn when the most common endpoint's share is above this (default 0.80)")
    diversity.add_argument("--diversity-min-cubie-distance", type=float,
                           default=DEFAULT_DIVERSITY_MIN_CUBIE_DISTANCE,
                           help="warn when the mean number of differing cubies between candidates (0-20) "
                                "is below this (default 2.0)")
    diversity.add_argument("--stagnation-generations", type=int, default=DEFAULT_STAGNATION_GENERATIONS,
                           help="warn after this many generations without better (unsolved, H) (default 100)")
    diversity.add_argument("--no-diversity-checks", action="store_true",
                           help="disable diversity measurements and warnings")
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
        block_library=_resolve_block_library(args),
        diversity_checks=not args.no_diversity_checks,
        diversity_min_unique_ratio=args.diversity_min_unique_ratio,
        diversity_max_dominance=args.diversity_max_dominance,
        diversity_min_cubie_distance=args.diversity_min_cubie_distance,
        stagnation_generations=args.stagnation_generations,
    )


def _resolve_block_library(args):
    """None when disabled; a user path relative to the working directory; else the shared default."""
    if args.no_block_library:
        return None
    if args.block_library is not None:
        return os.path.abspath(os.path.expanduser(args.block_library))
    return default_block_library_path()


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
    _print_cube(input_cube)
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
    if config.diversity_checks:
        print(f"Diversity checks: every {DIVERSITY_CHECK_INTERVAL} completed generations on non-reset candidates; "
              f"warn if unique endpoints <{100 * config.diversity_min_unique_ratio:g}%, one endpoint "
              f">{100 * config.diversity_max_dominance:g}%, mean differing cubies <{config.diversity_min_cubie_distance:g}, "
              f"or >={config.stagnation_generations} generations without better cube quality")
    else:
        print("Diversity checks: disabled")


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
    for line in _library_summary_lines(result.library_report):
        print(line)
    if result.diversity_records:  # identifies the last check without replaying its warning
        flagged = sum(1 for record in result.diversity_records if record["flags"])
        print(f"Diversity checks: {len(result.diversity_records)} (last at generation "
              f"{result.diversity_records[-1]['generation']}; {flagged} flagged); "
              f"diagnostic time {result.diversity_seconds:.3f}s")
    if final.problems:
        for problem in final.problems:
            print(f"CONSISTENCY ERROR: {problem}")
    print(f"Moves ({len(final.moves)}, reduced): {format_moves(final.moves)}")
    if final.cube is not None:
        print(f"Replayed cost H={final.cost} solved={'yes' if final.solved else 'no'}")
        print("Final cube:")
        _print_cube(final.cube)
        final.cube.print_as_python_list()


def _library_startup_lines(run):
    report = run.library_report() if run.library is not None else None
    if run.library_status == "disabled":
        return ["Block library: disabled"]
    if run.library_status == "not needed":
        return [f"Block library: {run.config.block_library} (no block search was needed; not read or written)"]
    if run.library_status == "unusable":
        return [f"Block library: {run.config.block_library} (unusable this run; see the message above)"]
    seeded = ", ".join(identifier[:12] for identifier in report["seeded"]) or "none"
    return [f"Block library: {report['path']} (enabled)",
            f"    valid entries {report['loaded']}, fit block capacity {report['capacity_compatible']}, "
            f"seeded {len(report['seeded'])}, snapshot {report['snapshot'][:16]}, "
            f"load and rescore {report['load_seconds']:.2f}s",
            f"    seeded transformations: {seeded}"]


def _library_summary_lines(report):
    if not report or report["status"] in ("disabled", "not needed"):
        return []
    lines = [f"Block library: {report['path']} ({report['status']}): {report['entries']} entries; "
             f"{report['additions']} added, {report['equivalent_merges']} equivalent merges, "
             f"{report['representative_updates']} shorter representatives, {report['evictions']} evicted, "
             f"{report['rejected_candidates']} rejected candidates, {report['rejected_records']} rejected records",
             f"    reference evaluations {report['reference_evaluations']} (cache hits "
             f"{report['reference_cache_hits']}); load {report['load_seconds']:.2f}s, evaluation "
             f"{report['evaluation_seconds']:.2f}s, write {report['write_seconds']:.3f}s"]
    lines.extend(f"    {message}" for message in report["diagnostics"])
    return lines


def _print_improvement(run, entry):
    """New best-ever state as a coloured cube, plus the two best-ranked move blocks right now."""
    print()
    print(f">>> New best state in generation {entry.generation} ({run.budget.elapsed():.1f}s): H={entry.cost}, "
          f"{len(entry.active)} moves, found via {entry.source}")
    print(f"    moves: {format_moves(entry.active)}")
    source_block = run.find_block(entry.block_id) if entry.block_id is not None else None
    if source_block is not None:
        print(f"    block B{source_block.uid}: {format_moves(source_block.active)}")
    if entry.solved:
        print("    solved; the cube is printed in the final report below")
    else:
        _print_cube(cube_from_key(entry.endpoint))
    top, scored = run.top_blocks(2)
    if not top:
        print("    top move blocks: none scored on this generation's contexts yet")
    else:
        progress = "all" if scored == len(run.blocks) else f"{scored} of {len(run.blocks)}"
        print(f"    top move blocks on batch #{run.batch.serial} ({progress} blocks scored, "
              f"{len(run.batch.contexts)} contexts):")
        for rank, block in enumerate(top, 1):
            evaluation = block.evaluation
            best_gain = max((outcome.best.gain for outcome in evaluation.outcomes if outcome.best), default=0)
            print(f"    #{rank} B{block.uid} ({block.origin}/{block.operator}, {len(block.active)} moves): "
                  f"{format_moves(block.active)}")
            print(f"        lowers H by up to {best_gain}, useful on {evaluation.useful_contexts}/"
                  f"{len(evaluation.outcomes)} contexts, mean relative gain {evaluation.mean_relative_gain:.4f}, "
                  f"credited improvements {run.counters.attributed_improvements.get(block.uid, 0)}")
    sys.stdout.flush()


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
        if objective.state_function in ("cubie", "cubie-refined", "grouped-cubie", "corners-first"):
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
    forced_stop = False
    try:
        run = CoevolutionRun(config, input_key, objective, seed, budget=budget, on_improvement=_print_improvement)
        for line in _library_startup_lines(run):
            print(line)
        print(f"Baseline: H={run.baseline.cost} solved={'yes' if run.baseline.solved else 'no'} (0 moves)")
        result = run.run()
        run.report_progress_if_stale()
    except KeyboardInterrupt:
        if run is None:
            _print_status(STATUS_INTERRUPTED)
            return EXIT_CODES[STATUS_INTERRUPTED]
        forced_stop = True
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
    # A forced stop keeps only the last checkpoint; a failed replay persists nothing new.
    result.library_report = run.finish_block_library(allow_save=not forced_stop and final.status != STATUS_ERROR)
    _print_final(result, final, finalization_seconds)
    if config.plot:
        print(f"Progress plot saved to {save_progress_plot(result.history)}")
    _print_status(final.status)
    return EXIT_CODES[final.status]


if __name__ == "__main__":
    sys.exit(main())
