# HW6 submission

## AI-use declaration

ChatGPT was used to assist with Python code implementation, debugging, and result validation. All experiments were executed and verified locally.

## Setup and reproduction

The project was developed using Python 3.9.6 with Gymnasium 1.1.1, NumPy 2.0.2, and Pandas 2.3.3. Dependencies were installed in a virtual environment using pip. Part B was executed using `part_b.py`, `test_fixed_policy.py`, `run_experiments.py`, and `save_traces.py`. All experimental results were saved in the `outputs/` directory.

## Part A — Paper calculations

Seven handwritten notebook pages in reading order. Photographs are kept at their original resolution.

**Note:** The image files need to be uploaded to the `part_a/` folder. The links below will work once that upload is completed.

1. [Page 01 — Original map, rewards, states and actions](part_a/page_01.jpg)
2. [Page 02 — MDP definitions, V0 map, start of V1](part_a/page_02.jpg)
3. [Page 03 — V1 calculations, including ice probabilities](part_a/page_03.jpg)
4. [Page 04 — Remaining V1, completed V1 map, start of V2](part_a/page_04.jpg)
5. [Page 05 — V2 for mud and ordinary floor, ice setup](part_a/page_05.jpg)
6. [Page 06 — Ice V2 calculations, arrows and tied actions](part_a/page_06.jpg)
7. [Page 07 — Explanation and conclusion](part_a/page_07.jpg)

[Part A folder](part_a/)

**Recorded results:** `V2(1,2)=8` (Right), `V2(1,3)=10` (Right), `V2(2,3)=5.21` (Up). No tied best actions at the three requested states.

## Part B — FrozenLake

### Fixed policy, written before the experiments

Environment: `FrozenLake-v1`, `4x4` map, with default rewards and transition rules.

The fixed policy was defined before running the experiments and remained unchanged across all four conditions. It follows a safe route toward the goal, avoiding holes when movements are deterministic.

**Fixed policy:**

| State | Action |
| --- | --- |
| 0 | Down |
| 1 | Left |
| 2 | Down |
| 3 | Left |
| 4 | Down |
| 6 | Down |
| 8 | Right |
| 9 | Right |
| 10 | Down |
| 13 | Right |
| 14 | Right |

The main intended path is:

`0 → 4 → 8 → 9 → 10 → 14 → 15 (Goal)`

The policy was tested before the experiments and successfully reached the goal in 6 steps with `is_slippery=False`.

The random policy selects actions uniformly from the four possible directions. Neither policy changes during the experiments.

### Experimental Setup

- Python: 3.9.6
- Gymnasium: 1.1.1
- NumPy: 2.0.2
- Pandas: 2.3.3
- Maximum episode length: 100 steps
- Batch seeds: 11, 22, 33
- Episodes per batch: 50
- Episode seed: `1000 * batch_seed + i`, where `i = 0...49`
- Total: 600 episodes across 4 conditions and 12 batches

Both policies were evaluated with `is_slippery=False` and `is_slippery=True`, using the same episode seeds. The random action generator was seeded for reproducibility.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | Random | 11 | 0 | 50 | 0 | 0% | 8.14 |
| False | Random | 22 | 1 | 49 | 0 | 2% | 7.56 |
| False | Random | 33 | 0 | 50 | 0 | 0% | 6.88 |
| False | Fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| False | Fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| False | Fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| True | Random | 11 | 0 | 50 | 0 | 0% | 8.06 |
| True | Random | 22 | 0 | 50 | 0 | 0% | 7.60 |
| True | Random | 33 | 0 | 50 | 0 | 0% | 7.98 |
| True | Fixed | 11 | 3 | 47 | 0 | 6% | 5.52 |
| True | Fixed | 22 | 2 | 48 | 0 | 4% | 5.20 |
| True | Fixed | 33 | 0 | 50 | 0 | 0% | 4.76 |

**Overall success rates:**

- Non-slippery Random: 1/150 (0.67%)
- Non-slippery Fixed: 150/150 (100%)
- Slippery Random: 0/150 (0%)
- Slippery Fixed: 5/150 (3.33%)

No episode reached the 100-step time limit.

### Saved Outputs

- [All 600 episode records](outputs/part_b_episodes.csv)
- [Batch summary results](outputs/part_b_summary.csv)
- [Initial successful episode](outputs/part_b_first_episode.txt)
- [Successful slippery episode](outputs/part_b_success_trace.txt)
- [Failed slippery episode](outputs/part_b_failure_trace.txt)

**Example traces:**

Successful episode: Fixed policy, slippery environment, seed 11012. The agent reached the goal in 21 steps with a total reward of 1.0.

Failed episode: Fixed policy, slippery environment, seed 11000. The agent fell into a hole after 8 steps with a total reward of 0.0.

### Analysis

**1. What did slipping change?**

Without slipping, the fixed policy achieved 100% success because the agent always followed its intended route. With slipping, its success rate dropped to 3.33% because the agent could move in unintended directions and fall into holes. Slipping made movement less predictable.

**2. How consistent were the batches?**

The non-slippery fixed policy was completely consistent, achieving 50/50 goals in every batch. The slippery fixed policy varied between 0 and 3 goals per batch. Random policies performed poorly in all batches, with only one successful episode overall.

**3. Why is one successful episode insufficient evidence?**

A single successful episode does not demonstrate that a policy is reliable. Random transitions can produce occasional successes or failures. Repeated experiments provide a better estimate of success rates and reveal variation between batches.

**4. Why are these simulations not training?**

The experiments only evaluate predefined policies. No policy parameters or actions are updated based on rewards or previous results. Therefore, the agent is not learning; it is executing and testing fixed decision rules.

### Conclusion

The fixed policy performed perfectly in the deterministic environment but was much less successful when slipping was enabled. The experiments demonstrate how stochastic transitions affect the reliability of a policy and why repeated simulations are necessary for evaluating performance.
## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

