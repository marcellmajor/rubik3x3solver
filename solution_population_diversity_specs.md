# Solution-population diversity and stagnation warnings

Status: implementation specification, 2026-10-23. This document does not implement the feature.

## 1. Goal

Measure whether the solution-candidate population is becoming homogeneous and whether cube-quality progress has stopped. Perform the four checks below after every **50th fully completed generation** and warn the user when **any one or more** conditions hold.

This is measurement and reporting only. Do not change selection, fitness, mutation, elites, reset proportions, solution normalization, block-library behavior, or the starting cube. Do not automatically restart, increase mutation, inject solutions, or discard individuals. Keep `cube.py` unchanged.

The checks apply to every current fitness option. Low diversity alone is not proof of failure: it may occur while approaching a solution. Stagnation is reported separately and may occur even in a diverse population.

## 2. Schedule and integration

- Run at completed generations `50, 100, 150, ...`, independently of `--log-every` and block-library checkpoint settings.
- Measure the newly published solution population after `run_generation()` has completed successfully, including its explicit reset slots.
- Do not inspect a partially built next population, use the previous generation's population with the new generation number, or emit a checkpoint for generation zero.
- Run the check immediately after population publication and before optional block-library maintenance. A timeout in later library work must not suppress a diversity check for an already completed generation.
- Run each scheduled check once. Final progress reporting must not print the same warning again.
- A run ending at generation 123 has checks at 50 and 100, not an additional check at 123. A partial generation 150 does not count as a completed checkpoint.
- If a solve terminates a generation before publication, report the solve normally; do not fabricate a completed diversity check. Already-solved input and zero-generation runs perform no checks.

Emit a warning on the first scheduled check whose conditions hold. **There is no five-consecutive-check gate**: that would delay warning for 250 generations. Record consecutive flagged checkpoint counts as optional diagnostic context, not a prerequisite.

## 3. Population subsets

Measure both:

1. **Whole population:** every current solution candidate.
2. **Non-reset population:** the 20% elite copies plus the 60% offspring, excluding only the current generation's explicitly designated 20% reset slots.

Use the current slot marker (`individual.slot == "reset"` in the present implementation), not historical `origin`, active length, state equality, or a fresh fitness sort. An elite copied from an earlier reset is now an elite and belongs in the non-reset population. Baseline-start offspring are also included.

In particular, retain empty/NOOP offspring and elites in the non-reset statistics. They represent real collapse or unsuccessful exploration, whereas the explicitly allocated resets are intentional.

For a valid population of size `N`, expect `N/5` current reset slots and `4*N/5` non-reset slots. Validate this invariant. A missing/malformed population must produce a diagnostic marked unavailable, not invented diversity percentages or a false homogeneity warning. Do not change the population to repair reporting metadata.

Threshold warnings for the three diversity metrics use the **non-reset** population. Whole-population values are descriptive context only; mandated resets must not trigger the warnings by themselves.

## 4. Metrics and default warning conditions

| ID | Metric | Default condition, non-reset population unless stated otherwise |
|---|---|---|
| `LOW_UNIQUE_STATES` | Unique endpoint count / candidate count | Strictly less than `0.10` |
| `DOMINANT_STATE` | Largest group with identical endpoints / candidate count | Strictly greater than `0.80` |
| `LOW_CUBIE_SPREAD` | Mean pairwise differing cubies, range `0..20` | Strictly less than `2.0` |
| `NO_QUALITY_PROGRESS` | Completed generations since best cube-quality improvement | Greater than or equal to `100` |

Evaluate all four conditions independently. One failing condition is enough for a warning; list every condition that holds. Boundary behavior is intentional: 10% unique, 80% dominance, and mean distance exactly 2 do not trigger their respective warnings; 100 generations without quality improvement does trigger stagnation.

These defaults are starting engineering choices, not established convergence guarantees. Make them named source constants and expose optional CLI overrides:

```text
--diversity-min-unique-ratio FLOAT       default 0.10, finite, in [0, 1]
--diversity-max-dominance FLOAT         default 0.80, finite, in [0, 1]
--diversity-min-cubie-distance FLOAT    default 2.0, finite, in [0, 20]
--stagnation-generations INT           default 100, integer >= 1
--no-diversity-checks                  disable measurements and warnings
```

The checkpoint interval remains fixed at 50 generations in version 1. Threshold overrides do not change that schedule. Invalid values must fail configuration validation before search. Library/API users must be able to configure the same behavior.

### 4.1 Unique endpoints

Use each candidate's stored, independently verified 54-sticker endpoint key. Different chromosomes producing the same endpoint count as one state. Different states with equal H count separately.

For subset size `n`, let endpoint multiplicities be `c_1, ..., c_k`:

```text
unique_count = k
unique_ratio = k / n
```

Report the count, denominator, and percentage. Do not compare chromosomes, NOOP padding, costs, or block IDs as substitutes for endpoint identity.

### 4.2 Dominant endpoint

```text
dominant_count = max(c_1, ..., c_k)
dominant_ratio = dominant_count / n
```

Report the multiplicity and percentage. The warning does not require the dominant endpoint to be the best-scoring state or the original input.

### 4.3 Mean pairwise cubie difference

For two candidate states, count how many of the 20 physical pieces differ in **position or orientation**. A piece counts once even if both differ. Each pair distance is an integer in `0..20`; identical endpoints have distance zero. Use unordered pairs of distinct candidate instances, so duplicated endpoints contribute zero-distance pairs in proportion to their actual population frequency.

Compute the exact mean without enumerating pairs. Decode each unique endpoint into the position/orientation of each of the 20 cubies, relative to the fixed face-center frame. For each cubie `j`, let `f[j,p]` be the number of candidates placing that cubie in pose `p`, weighted by endpoint multiplicity. Then:

```text
pair_count = n * (n - 1) // 2
disagreeing_pairs[j] = pair_count - sum(f[j,p] * (f[j,p] - 1) // 2 for p)
distance_sum = sum(disagreeing_pairs[j] for j in the 20 cubies)
mean_cubie_difference = distance_sum / pair_count
```

This is exactly the pairwise mean, not an estimate. It requires no sampling or additional RNG. Use integer counts throughout; round only for display. Evaluate thresholds using unrounded values.

Reuse the existing small cubie placement/move information, or extract a lightweight decoder shared with it. Do not initialize the grouped-cubie pattern databases merely to measure diversity under another fitness. Cache decoded endpoints within the checkpoint and share that work between the whole and non-reset subsets; do not retain an unbounded cross-generation cache.

If `n < 2`, the pairwise mean is unavailable, not zero. If a state cannot be decoded into a distinct consistent set of cubies, mark the spread metric unavailable and print an explanatory diagnostic. The current sticker-based objectives may admit such inputs; this reporting feature must not silently alter their input-validation policy or turn an unknown value into a homogeneity warning. Other valid metrics still run. Cubie decoding is not a proof of physical solvability.

### 4.4 Cube-quality stagnation

Track a separate `last_quality_improvement_generation`, initialized to zero with the original baseline. Update it whenever a verified archive update strictly improves:

```text
(unsolved_flag, H)
```

Use the same exact objective ordering as the solver. Do not add a new epsilon or reinterpret weighted fitness.

A shorter chromosome at unchanged H is a useful length improvement, but **does not reset this quality-stagnation clock**. The present `last_improvement_generation` also counts shorter equal-H solutions, so it must not be reused without this distinction. Leave existing archive/rank behavior unchanged.

At a completed checkpoint `g`:

```text
generations_without_quality_improvement = g - last_quality_improvement_generation
```

Count verified improvements from either population, including block-discovered candidates and improvements between checkpoints. Do not infer stagnation from the average population cost, or only from the last two checkpoint values.

## 5. User-visible output

Always emit one concise diversity summary at each scheduled checkpoint, even if `--log-every` is larger than 50. It must state:

- Generation and whole/non-reset/reset counts.
- Unique endpoint counts and percentages for both measured subsets.
- Dominant endpoint counts and percentages for both subsets.
- Exact mean cubie difference for both subsets, or an explicit unavailable reason.
- Generations since cube-quality improvement.
- Active warning condition IDs, or `none`.

If any condition holds, emit one grouped warning containing all failing metrics, their observed values, and thresholds. For example:

```text
[gen 150] WARNING: solution-population diversity/stagnation
  LOW_UNIQUE_STATES: 48/800 non-reset endpoints = 6.0% (threshold <10.0%)
  DOMINANT_STATE: 690/800 share one endpoint = 86.25% (threshold >80.0%)
  LOW_CUBIE_SPREAD: mean differing cubies = 1.18/20 (threshold <2.0)
  NO_QUALITY_PROGRESS: 117 generations without better cube quality (threshold >=100)
  Intentional reset slots excluded from thresholds: 200/1000.
```

When only stagnation holds, identify it as a progress warning and do not claim the population is homogeneous. When only diversity conditions hold, do not claim the search has stalled. A combined condition may say “low diversity with stalled cube-quality progress.”

Repeat the grouped warning at subsequent 50-generation checkpoints while any conditions hold, with current values. No per-generation repetitions, dialogs, approval prompts, or external notifications. If the previous checkpoint was flagged and the current one has no active flags, emit one recovery message saying that none of the warning thresholds is currently crossed; do not claim convergence is guaranteed.

Route output through the run's existing logger/callback so library callers can capture or suppress it. Do not print directly from metric helpers. Disabling ordinary progress output must not inadvertently disable these checks; explicit suppression of the logger still suppresses output normally.

## 6. Structured data and performance

Produce a structured checkpoint record with generation, subset sizes, reset count, metrics, availability/reasons, threshold values, quality-stagnation age, and a list of active condition IDs. Attach it to the run's result/history without creating a separate file by default.

Keep all measurements for one checkpoint tied to the same published population. Non-checkpoint history rows must have no fresh diversity record; never repeat an old measurement as if it describes the current generation. Final output may identify the last checked generation but must not replay its warning.

Expected work is `O(N + 20*U)` per checkpoint, where U is the number of distinct endpoints, with checkpoint-local memory of the same order. Do not do `O(N^2)` comparisons, replay every chromosome, recompute H, or construct pattern databases. Never mutate candidate metadata, populations, archive ranking, or RNG state while measuring.

The diagnostic runs synchronously and its wall time counts toward the existing search budget. Because the work is bounded, complete an initiated checkpoint on its published population rather than publishing partial metrics; the normal search budget check then resumes. Track diagnostic elapsed time separately for overhead measurement.

For fixed-generation runs without a time limit, enabling/disabling diagnostics must yield identical solutions, scores, operator choices, and RNG evolution. Time-limited runs may explore slightly fewer generations because reporting consumes time; document that ordinary consequence rather than pausing the budget.

Target less than 1% total runtime overhead on a representative run of at least 500 generations with the current default solution population. Report actual overhead and checkpoint latency before claiming the target is met.

## 7. Acceptance tests

1. **Schedule:** checks occur exactly at completed 50-generation boundaries, regardless of log interval or library settings; zero, partial, and final non-boundary generations do not create checks.
2. **Any/all conditions:** each of the four conditions independently produces a warning, and combinations produce one grouped warning listing every active condition.
3. **Boundaries:** test values below, exactly at, and above all four thresholds, including unrounded values that would display identically.
4. **Reset handling:** exactly the explicitly tagged current resets are excluded. Empty offspring, baseline-start offspring, and elites descended from old resets remain included. Whole-population reset duplication alone does not trigger a warning.
5. **Endpoint identity:** distinct chromosomes with identical endpoints count as duplicates; different endpoints with equal H remain distinct.
6. **Exact cubie mean:** the frequency-count formula matches brute-force pair comparisons on small fixture populations, including duplicates, orientation-only differences, and moved pieces. All-identical states yield zero; valid means remain in `[0,20]`.
7. **Stagnation:** strict H improvement resets the clock; length-only improvements do not. Improvements from blocks and between checkpoints are counted. Stagnation alone must not be described as homogeneity.
8. **Unavailable data:** invalid cubie decoding and populations with fewer than two members do not produce a false zero-distance warning; valid metrics remain available. Population-role inconsistencies are reported explicitly.
9. **Warning lifecycle:** immediate warning at the first flagged checkpoint, repeated checkpoint warnings while active, one recovery message when cleared, and no duplicate final warning.
10. **No algorithm changes:** enabled/disabled fixed-seed, fixed-generation runs have identical final archives and search counters. Search RNG state is unchanged by a standalone diagnostic invocation.
11. **No expensive side effects:** Manhattan diagnostics do not initialize grouped-cubie tables, replay chromosomes, write files, or call cost evaluation. Temporary decoded-state storage is released after each checkpoint.
12. **Configuration and reporting:** invalid thresholds are rejected, logger capture works, records contain the correct generation/subsets/flags, and `--no-diversity-checks` performs no diagnostic work.

Implementation should update the README and CLI help with the schedule, thresholds, reset exclusion, and the distinction between low diversity and stalled progress. Automatic recovery policies are outside this feature.
