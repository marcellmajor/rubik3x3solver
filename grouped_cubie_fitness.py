"""Exact four-piece pattern distances, using permutations supplied by cube.py.

No tables are constructed at import. The caller owns the lazy instance. Each
piece has 24 reachable placements (position AND orientation); a four-piece
pattern is indexed directly in base 24. Impossible overlapping placements stay
unvisited. This uses 1.58 MiB for five byte tables, trading some unused entries
for inexpensive indexing in both construction and fitness evaluation.
"""

from array import array
from dataclasses import dataclass
from operator import itemgetter


POSE_COUNT = 24
PAIR_COUNT = POSE_COUNT ** 2
PATTERN_SIZE = POSE_COUNT ** 4
UNREACHED = 255


def _pattern_index(poses):
    a, b, c, d = poses
    return ((a * POSE_COUNT + b) * POSE_COUNT + c) * POSE_COUNT + d


@dataclass(frozen=True)
class Pattern:
    members: tuple
    distances: bytes
    reachable_count: int


def _build_pattern(members, home_poses, transitions, expected_count):
    """BFS in the projected puzzle, with each of the 18 face turns costing one."""
    # Pair transitions avoid four pose lookups and base-24 decoding per edge.
    pairs = []
    for first, second in (members[:2], members[2:]):
        pairs.append(tuple(
            array("H", (transitions[first][move][a] * POSE_COUNT
                        + transitions[second][move][b]
                        for a in range(POSE_COUNT) for b in range(POSE_COUNT)))
            for move in range(18)
        ))
    move_pairs = tuple(zip(*pairs))
    home = _pattern_index(tuple(home_poses[member] for member in members))
    distances = bytearray([UNREACHED]) * PATTERN_SIZE
    distances[home] = 0
    queue = array("I", [home])
    # array iteration includes appended entries, so this is a compact FIFO.
    for state in queue:
        first, second = divmod(state, PAIR_COUNT)
        next_distance = distances[state] + 1
        for left, right in move_pairs:
            neighbor = left[first] * PAIR_COUNT + right[second]
            if distances[neighbor] == UNREACHED:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    if len(queue) != expected_count:
        raise ValueError(f"pattern has {len(queue)} reachable states, expected {expected_count}")
    return Pattern(tuple(members), bytes(distances), len(queue))


class GroupedCubieTables:
    """Two four-corner patterns and three four-edge patterns, covering all pieces.

    Slots retain their deterministic sticker-position order. Within each piece
    type, consecutive groups of four define the partition. Piece orientations
    are derived from actual moves, avoiding assumptions about face notation.
    """

    def __init__(self, slots, move_permutations):
        if len(move_permutations) != 18:
            raise ValueError("grouped cubie tables require all 18 face turns")
        destinations = []
        for permutation in move_permutations:
            inverse = [0] * 54
            for new_position, old_position in enumerate(permutation):
                inverse[old_position] = new_position
            destinations.append(inverse)
        slot_index = {frozenset(slot): i for i, slot in enumerate(slots)}
        decoders = [{} for _ in slots]
        transitions, home_poses = [], []
        for member, home in enumerate(slots):
            placements = {home: 0}
            pending = [home]
            for placement in pending:
                for destination in destinations:
                    moved = tuple(destination[p] for p in placement)
                    if moved not in placements:
                        placements[moved] = placements[placement] + 1
                        pending.append(moved)
            poses = sorted(placements)
            if len(poses) != POSE_COUNT:
                raise ValueError("a cubie must have exactly 24 reachable placements")
            pose_index = {pose: i for i, pose in enumerate(poses)}
            home_poses.append(pose_index[home])
            transitions.append(tuple(
                tuple(pose_index[tuple(destination[p] for p in pose)] for pose in poses)
                for destination in destinations
            ))
            labels = tuple(str(p // 9) for p in home)
            for pose_id, placement in enumerate(poses):
                slot = slot_index[frozenset(placement)]
                label_at = dict(zip(placement, labels))
                label = tuple(label_at[p] for p in slots[slot])
                misplaced = sum(color != str(p // 9) for color, p in zip(label, slots[slot]))
                decoders[slot][label] = (member, pose_id, 64 * placements[placement] + misplaced)
        patterns = []
        for size, expected_count in ((3, 136080), (2, 190080)):
            members = [i for i, slot in enumerate(slots) if len(slot) == size]
            for start in range(0, len(members), 4):
                patterns.append(_build_pattern(members[start:start + 4], home_poses,
                                               transitions, expected_count))
        self.patterns = tuple(patterns)
        # A slot observation already determines its piece, pose, solo distance
        # and misplaced stickers. Precompute its contribution to the group's
        # base-24 index and the two lower score tiers. No second cubie/sticker
        # scan or per-piece string construction is needed during evaluation.
        membership = {member: (group, POSE_COUNT ** (3 - digit))
                      for group, pattern in enumerate(patterns)
                      for digit, member in enumerate(pattern.members)}
        self._decoders = tuple(
            (itemgetter(*slot), {
                label: (membership[member][0], pose * membership[member][1], lower, 1 << member)
                for label, (member, pose, lower) in decoder.items()
            }) for slot, decoder in zip(slots, decoders)
        )
        self._all_members = (1 << len(slots)) - 1
        self._distance_tables = tuple(pattern.distances for pattern in patterns)

    def _decode(self, face_indices):
        """Decode all five indices and 64*cubie+misplaced in one pass."""
        indices = [0] * 5
        lower = seen = 0
        try:
            for getter, decoder in self._decoders:
                group, contribution, tie_break, member_bit = decoder[getter(face_indices)]
                indices[group] += contribution
                lower += tie_break
                seen |= member_bit
        except KeyError:
            raise ValueError("the stickers do not form a consistent set of cubies") from None
        if seen != self._all_members:
            raise ValueError("the stickers repeat a cubie")
        return indices, lower

    def _lookup(self, indices):
        distances = tuple(table[index] for table, index in zip(self._distance_tables, indices))
        if UNREACHED in distances:
            raise ValueError("the stickers form an unreachable cubie pattern")
        return distances

    def distances(self, face_indices):
        """Return five exact distances for a key labelled by centre indices 0..5."""
        indices, _ = self._decode(face_indices)
        return self._lookup(indices)

    def cost(self, face_indices):
        """Return 4096*pattern_sum + 64*cubie + misplaced with one decoding."""
        indices, lower = self._decode(face_indices)
        return 4096 * sum(self._lookup(indices)) + lower

    def corner_weighted_cost(self, face_indices, corner_weight):
        """Like :meth:`cost`, but the two corner patterns count ``corner_weight`` times.

        Patterns are ordered corner groups first (indices 0-1), then the
        three edge groups (indices 2-4); the lower tiers are unchanged.
        """
        indices, lower = self._decode(face_indices)
        distances = self._lookup(indices)
        weighted = corner_weight * (distances[0] + distances[1]) + sum(distances[2:])
        return 4096 * weighted + lower
