# Persistent library of 100 distinct move blocks

Status: implementation specification, 2026-10-23. This document specifies a feature; it does not implement it.

## 1. Purpose and scope

Add a shared, bounded move-block library to `solver_deap_coevolution.py`. It retains the best **up to 100 distinct cube transformations** discovered across runs, independently of the starting cube. A later run can seed part of its block population from that library.

“Best” means highest ranked under the fixed reference evaluation in section 4, among eligible candidates submitted at the checkpoints in section 5. It does not mean the globally best possible algorithms, the best of every block ever generated, or the best on every cube. Fewer than 100 entries is correct when fewer than 100 distinct eligible transformations have been found; do not pad the library with duplicates or untested blocks.

The feature must work with every current CLI fitness, including `grouped-cubie`. It must not change `cube.py`, fitness definitions, solution normalization, the solution population's 20/60/20 policy, or the block population's subsequent 10/60/30 replacement policy. All initial solution chromosomes still contain only NOOPs.

This specification supersedes the earlier specification's prohibition on saved block seeds only for this feature. It does not authorize hand-written algorithm seeds, a full search-checkpoint/resume mechanism, or changes to piece grouping.

## 2. Interface and location

Persistence is enabled by default. Add:

```text
--block-library PATH    Override the shared library file.
--no-block-library     Disable all library reads, writes, reference scoring,
                       and population seeding for this run.
```

Reject use of both options together. The library API must expose equivalent configuration. The default path is:

```text
${XDG_DATA_HOME}/rubik3x3solver/move_blocks_v1.json
```

If `XDG_DATA_HOME` is unset, use `~/.local/share/rubik3x3solver/move_blocks_v1.json`. Resolve a user-supplied relative path against the invocation's working directory; the default must be independent of that directory. Create directories only on the first actual save, never on module import or `--help`.

The file is shared across starting cubes, seeds, and CLI objectives. Do not incorporate any of those into its path or partition entries by them. Independent experiments can use separate paths.

`--no-block-library` must reproduce the existing seeded search, including RNG consumption, results, and counters other than newly added zero-valued library counters.

## 3. Exact distinctness and representative sequences

### 3.1 Transformation identity

1. Validate move tokens against the existing 18 real moves. Persist no NOOP padding.
2. Apply the existing move-equivalent simplifier.
3. Execute the sequence on a cube whose 54 sticker positions have distinct labels, deriving the resulting permutation through the existing move engine/tables.
4. Use the complete 54-position permutation as the authoritative identity. A SHA-256 digest may be used as an index, but compare the complete permutation before merging entries.

Use the same permutation convention as `_MOVE_TABLES`: the sticker at new position `i` comes from the recorded old position. Persist the convention's version. Validate that a recorded permutation is a bijection of `0..53` and matches replay of the recorded sequence.

Two sequences producing the same permutation consume one library slot, even if ordinary simplification does not make their strings equal. Examples: `R1 R1` and `R2`; `R1 L1` and `L1 R1`. Discard the identity transformation, including nonempty algorithms that return the entire cube to its original state.

Inverses and cube-rotated/conjugated algorithms remain separate when their actual permutations differ. Do not collapse them based on shape, similar spelling, or a Hamming-distance threshold. Exact transformation uniqueness is the meaning of “distinct” in version 1; near-duplicates are not a second exclusion rule.

### 3.2 Representative selection

For an existing transformation, retain the shortest known simplified sequence. Break equal-length ties by lexicographic order of integer move alleles. This is shortest **known**, not globally shortest.

Limit persisted representatives to 64 active moves. Longer candidates are skipped with a counter. A saved block longer than the current run's block capacity remains in the file but is ineligible for seeding that run. Never truncate a saved block to make it fit: that changes its transformation.

### 3.3 Endpoint identity versus prefix behavior

Equivalent transformations can have different intermediate prefixes. Therefore library ranking must evaluate the **complete saved sequence**, not its best prefix. Replacing a sequence with a shorter equivalent sequence preserves the reference endpoint results, but need not preserve runtime prefix behavior.

A useful prefix may be submitted as a separate candidate with its own permutation and reference evaluation. Do not attach a prefix's improvement to the full block's library score. During search, loaded blocks retain the existing best-prefix evaluation behavior.

## 4. A stable definition of “best” across cubes

Historical block fitness is contextual. Do not compare or sum raw gains from different starting cubes, changing context batches, or different CLI fitness functions to decide library membership.

### 4.1 Fixed reference panel

Define a versioned panel of 48 legal, distinct, unsolved cube states. Generate it from the repository's canonical solved cube (face order/colors `G,R,W,O,Y,B`), using eight scrambles at each raw length in `(2, 4, 8, 12, 20, 30)`.

For reproducibility independent of the search RNG or Python's random implementation, specify each move as follows:

```text
sample = 0..7; attempt starts at 0; position = 0..length-1
text = "rubik-block-library-v1|{length}|{sample}|{attempt}|{position}"
allele = 1 + int.from_bytes(SHA256(text.encode("ascii")), "big") % 18
```

Apply the resulting moves. If the state is solved or duplicates an earlier panel state, increment `attempt` and regenerate that sample. Process lengths and sample numbers in the order above. Freeze the resulting ordered 48 state keys and their checksum as a test fixture in the implementation. This is a calibration panel, not an unbiased solve benchmark.

### 4.2 Reference objective and score

Use the existing `cubie-refined` semantics, explicitly versioned as the library evaluator, regardless of the run's CLI objective:

```text
H_ref = 64 * solo_cubie_distance + misplaced_stickers
```

For each panel state, apply the complete candidate and measure the endpoint:

```text
gain = H_ref(before) - H_ref(after)
useful = gain >= 1
solved = independent solved-state predicate(after)
gain_ppm = floor(1_000_000 * max(gain, 0) / H_ref(before))
```

All panel baselines are positive. Compute `gain_ppm` with integer arithmetic. Rank candidates lexicographically by:

1. More solved panel states.
2. More useful panel states.
3. Larger sum of `gain_ppm` across all 48 states.
4. Fewer active moves.
5. Lexicographically smaller move-allele tuple, then permutation digest, for deterministic ties.

Require at least one useful panel state for membership. Maintain the first 100 distinct transformations under this ordering. Existing entries and new candidates receive exactly the same treatment; there is no incumbent bonus.

This deliberately optimizes reuse under one fixed reference objective. It does not establish that a block will improve `grouped-cubie`, Manhattan, or the next input cube. Runtime re-evaluation remains mandatory. A change to the panel, reference objective, move conventions, or ranking requires a corresponding version change and reference re-evaluation before scores are compared.

Do not use discovery counts, repeated evaluations of one state, run timestamps, or mutable generation numbers in ranking. Repeatedly saving the same block cannot improve its score.

## 5. Candidate collection and bounded work

Submit candidates after the first fully evaluated block batch, at every 50th **fully completed** generation, and at normal finalization. At a checkpoint, take the most recent fully evaluated block population and submit, for each distinct sequence:

- Its complete sequence, if useful on at least one current context.
- At most one best feasible useful prefix, selected across that block's context outcomes using the existing `_prefix_preference`, with context ID as the final tie-breaker.

Deduplicate these candidates by transformation before reference evaluation. This bounds a checkpoint's population-derived submissions by twice the block population size. Snapshot the last complete block evaluation before survivors are replaced; never treat unevaluated children or scores from a partially evaluated new batch as a complete snapshot.

Also capture exact block payloads responsible for a verified best-ever solution improvement, including discoveries that terminate the run before a batch finishes. Store the payload actually applied, not a subsequently mutated block or a guessed sequence reconstructed from its ID. Collect at most 100 such pending payloads; when full, request a flush at the next safe checkpoint boundary. If more arrive before that boundary, retain those associated with the best resulting solution ranks, then shorter payloads and deterministic allele-order ties. Prefixes are simplified and evaluated independently under section 4.

Only completely evaluated reference candidates may enter the library. Check the search budget between reference states/moves. Discard an incomplete reference evaluation; never persist a partial score as if all 48 states had been tested.

During search, reference work counts against the normal wall-clock budget. After budget exhaustion or SIGINT, do not start additional reference evaluations: save only the already completed in-memory archive. A solve reached with budget remaining may finish its bounded final candidate batch, subject to the same budget. Final disk merge/write time must be measured separately.

Keep a bounded reference-result cache, initially 4096 transformations, keyed by evaluator/panel/move version and permutation. Scores for complete equivalent transformations can be reused; the representative's length tie-breaker must still be updated. Cache eviction affects speed, not score correctness. Do not accumulate every historical block ID, context, or rejected candidate indefinitely.

## 6. Loading into a new run

If the original input is already solved or zero generations were requested, skip library loading, evaluation, seeding, and saving; report that no block search was needed. Invalid input/configuration must also leave the library untouched.

1. Validate the input/configuration, then read one library snapshot. Missing file means an empty library.
2. Validate records and recompute their reference scores before trusting membership/order. This startup work, like existing pattern-table initialization, precedes the search timer and must be reported separately.
3. Filter seeding candidates by current block capacity without deleting longer saved entries.
4. Seed at most `min(eligible_count, floor(block_population / 4))` initial slots, selecting in library rank order and using each transformation at most once.
5. Fill all remaining slots with fresh blocks using the existing reduced-length schedule for that fresh allocation. Include seeded sequences in duplicate-cap accounting.
6. Give loaded individuals fresh run-local IDs, `origin="library"`, and invalid runtime fitness. Record the library transformation ID separately for diagnostics.
7. Evaluate all blocks against the same current context batch before tournaments or targeted insertion. Never copy persisted reference scores into DEAP fitness, the block evaluation cache, or context-prefix records.

Do not load again during an active search. Concurrent changes to the file must not alter that run's evolution. Later 10/60/30 generations remain unchanged; library blocks survive or disappear through ordinary selection. All solution candidates still start from NOOP.

Reproducibility is defined by input, configuration, seed, and validated library snapshot. Log the snapshot digest and seeded transformation IDs. Library loading, reference evaluation, and serialization must not consume the search RNG; fresh-block generation retains its ordinary seeded behavior.

## 7. File format and compatibility

Use UTF-8 JSON, never pickle or executable Python. The top-level object contains:

```text
schema_version
move_convention_id        # digest of ordered move names and their permutations
evaluator_version
reference_panel_digest
revision                 # snapshot identity for merge/diagnostics
entries                  # rank-ordered array, length <= 100
```

Each entry contains:

```text
transformation_id         # SHA-256 of the canonical permutation encoding
permutation               # 54 integers, authoritative equality check
moves                     # ordinary tokens, e.g. ["R1", "U2", "R3"]
reference_score           # solved_count, useful_count, gain_ppm_sum
```

Move length is derived from `moves`. No full cube histories, cached DEAP objects, source chromosomes, or raw per-generation observations belong in the file. The starting cube is not needed to replay a block's transformation and must not control eligibility.

Bound input size to 1 MiB before parsing. Validate schema, entry counts, token lengths, integer types/ranges, permutations, replay agreement, and score fields. Reject boolean values where integers are required. No NaN or Infinity is permitted. Do not silently truncate an oversized or malformed file and overwrite it.

Schema or move-convention incompatibility disables library use for that run with a concise warning and preserves the file. A recognized older evaluator/panel version may be migrated only by replaying and rescoring its valid sequences under the current version. Write the current format only after successful migration; otherwise preserve the original.

## 8. Save, merge, concurrency, and errors

At each checkpoint and finalization, merge the in-memory archive with the latest on-disk library, deduplicate transformations, select shortest known representatives, and retain the best 100 using current reference scores. Saving the same snapshot twice must be idempotent.

Use an OS-backed advisory lock on a companion lock file, with a bounded two-second acquisition timeout. Hold the lock across the latest-file read, merge, and atomic replacement, so two concurrent runs cannot blindly overwrite each other's submissions. Revalidate/re-evaluate newly encountered records as needed; previously verified entries can use the versioned reference cache. Never force replacement of an incompatible/corrupt file.

Write a temporary file in the destination directory, flush and fsync it, then atomically replace the destination. Clean up temporary files on failure and sync the directory where supported. A reader must see either the old complete file or the new complete file. A crash must not leave a partial JSON library, and OS locks must release when their process exits.

Missing library: continue normally and create it only when there are eligible entries to save. Corrupt, incompatible, unreadable, or unwritable library, disk-full error, or lock timeout: report one concise diagnostic, preserve the existing file, and continue/finish the solver. A library error must not change an independently verified SOLVED result to ERROR. After a corruption/compatibility load failure, disable writes to that path for the remainder of the run.

Do not persist new candidates following a solver consistency error or failed independent replay. On clean SIGINT or budget exhaustion, already completed and verified reference results may be saved. Forced termination need only preserve the last successful checkpoint.

## 9. Reporting and performance

Report the resolved path, enabled/disabled status, valid entries, capacity-compatible entries, seeded count, and snapshot digest at startup. Report additions, equivalent-transform merges, replacements, rejected records, current entry count, and library load/evaluation/write time at checkpoints/finalization. Keep these separate from search fitness and current-context block scores.

The ordinary per-state fitness hot path must not perform filesystem access, locking, reference-panel scoring, or transformation fingerprinting. Compute a candidate fingerprint once per distinct submitted sequence and cache bounded results. Do not copy a 54-sticker permutation into every gene or candidate.

Measure overhead on the existing solver's fixed-seed runs. Aim for under 10% additional end-to-end time on a long run with checkpoint interval 50; report startup and finalization separately. If the target is missed, optimize batching/cache reuse or revise the checkpoint interval explicitly. Do not hide reference work by pausing the search timer.

## 10. Acceptance tests

The feature is complete only when tests establish:

1. **Cross-cube reuse:** a block saved while solving cube A is available on cube B, including with another CLI objective, and receives fresh runtime fitness.
2. **Distinctness:** equivalent strings, commuting sequences, and equivalent longer algorithms merge; distinct permutations remain separate even when scores match. Identity blocks are excluded.
3. **Prefix correctness:** a full block never inherits a useful prefix's library score; that prefix can enter separately. Changing to a shorter equivalent representative preserves endpoint reference results.
4. **Top-100 rule:** after at least 120 eligible distinct candidates are submitted, the file contains exactly the reference top 100. With fewer eligible candidates it contains fewer, with no duplicates or filler. Repeated submissions give no bonus.
5. **Stable ranking:** panel regeneration matches the frozen fixture; score comparisons and tie-breaking are deterministic across runs and independent of search RNG/CLI objective.
6. **Population rules:** no more than 25% initial blocks are loaded, the rest are fresh, duplicate caps hold, incompatible lengths are skipped without truncation/deletion, and all initial solutions remain NOOP.
7. **Disabled behavior:** fixed-seed runs with `--no-block-library` match the pre-feature search and create no files.
8. **Safety of stored data:** malformed JSON, oversized files, incompatible versions, invalid tokens, mismatched fingerprints, and non-bijective permutations never execute data or destroy a previous file.
9. **Persistence lifecycle:** missing/empty file, already-solved input, zero generations, budget expiry mid-reference evaluation, clean interruption, and consistency failure follow the rules above.
10. **Concurrency and atomicity:** concurrent writers produce the top 100 of their merged submitted archives; interrupted writes preserve a valid old or new file. Lock failures preserve solver results.
11. **Bounded resources:** caches/queues remain bounded in long plateau runs. Per-state fitness does not access disk or evaluate the reference panel.
12. **Replay verification:** saved sequences and all reported solves remain independently reproducible through unchanged `cube.py`.

Benchmark usefulness separately from correctness. Train one library, freeze it, then compare library-seeded and disabled runs on held-out starting cubes with matching seeds, populations, and wall-clock budgets. Report solve rate, time to verified solution, and total library overhead. An earlier decrease in H or a higher calibration score alone is not evidence of faster solving.
