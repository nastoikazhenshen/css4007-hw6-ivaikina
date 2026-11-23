# HW6 submission

## AI-use declaration

ChatGPT was used to explain Markov decision processes and value iteration, clarify the ice transition probabilities, and help check the Part A hand calculations. The handwritten solution was written in my notebook.

## Setup and reproduction

Record dependency versions, installation commands and exact run commands.

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

State the action for each nonterminal state and your reasoning.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |

Link episode rows and example traces in outputs/. Explain slipping, variation across batches and why these runs do not train a policy.

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

