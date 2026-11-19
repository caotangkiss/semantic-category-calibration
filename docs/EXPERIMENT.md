# Semantic Category Calibration: can category probabilities improve next-token selection?

This article explores **Semantic Category Calibration**, an inference-time decoding postprocessing method. It combines semantic categories, input-token attention, and candidate probabilities to adjust next-token selection while preserving model weights and the original predicted probabilities.

## Motivation and hypothesis

For the prefix `I put ketchup on my`, my local GPT-2 model selects ` face` under greedy decoding. In an ordinary eating context, I would prefer a food-related continuation such as ` burger` or ` salad`.

The distribution suggests a possible opportunity: `face` has the highest individual probability, but several food candidates collectively have greater probability mass, and the strongest food candidate is only slightly behind `face`. The hypothesis is that support associated with a higher-attention input may be spread across many related candidates, allowing a more concentrated individual candidate to win. Could a postprocessor combine attention ordering, semantic-group probabilities, and the best candidate's probability gap to make a more appropriate choice?

Based on this hypothesis, I collaborated with Codex and Doubao on a small experiment to preliminarily test the idea. The procedure and observations follow.

My initial guess was that `ketchup` would receive the most attention, or at least more than `my`. The experiment corrected the first part of that guess. The implementation now estimates input-to-candidate support by embedding ablation instead of configuring associations such as `ketchup → food` or `my → body parts`. This estimates influence under specified interventions, not a unique causal origin. Also, `face` is an output candidate, not an input token with a directly comparable attention score.

## Experimental setup

GPT-2 Small has 12 layers, 12 heads per layer, a hidden size of 768, and a vocabulary of 50,257 tokens. The reference environment used Python 3.9, PyTorch 2.2.2, and Transformers 4.46.3. Inference ran on CPU, in evaluation mode, with eager attention and no sampling or temperature adjustment.

See [README.md's Quick start](../README.md#quick-start) for the reference checkpoint revision and installation commands, then run `.venv/bin/python semantic_category_calibration.py "I put ketchup on my" "I put ketchup on a" --output-dir outputs`. Prompts must be supplied explicitly and can be replaced with your own text; the script has no default examples or expected answers.

If GPT-2 Small is already downloaded, install the dependencies, skip the model download, and run `.venv/bin/python semantic_category_calibration.py "your prompt" --model-dir /path/to/gpt2` with a directory containing compatible configuration, tokenizer files, and safetensors weights. The project can also run on a remote machine. A remote HTTP API adapter is not yet implemented and would require full vocabulary probabilities, attention, and repeated inference with controlled input embeddings; see [README.md's model setup instructions](../README.md#existing-models-and-remote-execution).

Semantic categories and candidate members are manually defined in [selection-policy.json](../selection-policy.json). The food group contains 12 candidates:

`salad, burger, sandwich, fries, chips, pizza, eggs, steak, rice, soup, chicken, toast`

Calculations use their single-token forms with a leading space. At each examined input, sum only the configured members with consistent positive attribution to that input. For example, `potato` contributes nothing unless it is listed in the policy. This is a finite candidate list, not the complete probability of all possible food expressions. Duplicate token IDs are counted once per group; multi-token members are skipped and recorded explicitly.

Version 3 removes input anchors and context-pattern rules. Associations are computed from model interventions; the semantic labels and membership lists remain user supplied.

## Complete selection procedure

1. **Process the full prompt.** Pass the context, style requirements, and unfinished text together to GPT-2 to obtain attention and next-token probabilities for that complete input.
2. **Preserve original probabilities.** Let `t` denote any next-token candidate and `p(t)` its original model probability. `t*` is the candidate with the highest probability over the complete vocabulary (the original winner); the star `*` marks this winner, and `p* = p(t*)` is its probability. Neither weights nor probabilities change.
3. **Estimate the original winner's source.** Replace one input token embedding at a time using zero, the mean vocabulary embedding, and the space-token embedding. Keep sequence length, positions, other embeddings, and output position unchanged. For each baseline, calculate `log(original probability) - log(perturbed probability)`. A positive drop indicates support relative to that replacement. Accept a stopping source only when all three baselines identify the same unique input with the greatest positive drop for `t*`; otherwise retain `t*`. A `1e-6` tolerance handles near-zero values and source ties; it is not a confidence threshold. If the sources disagree, this version does not handle multiple possible stopping points and returns before category evaluation. This conservative fallback can miss a useful replacement; it does not establish that the original winner is preferable or that alternatives failed the probability gates.
4. **Inspect only higher-attention inputs.** Use the last input position as query, average equally across all 12 layers and 144 heads, and inspect inputs in descending attention strictly above the stopping source. Never evaluate categories or gates at that source or at attention ties or lower-attention inputs.
5. **Choose a primary category.** Reuse the ablation measurements for every configured member. At each input, retain only members with positive drops under all three baselines, then choose the category with the greatest summed original probability. The current input need not be a member's strongest source: weak positive support above the numerical tolerance also qualifies, and other inputs may support the same candidate. Break mass ties by the best member probability, then random choice if still tied. Skip inputs with no consistently supported members. Positive support measures an effect on probability; it does not imply shared semantic meaning.
6. **Check replacement gates.** For group `G`, let `S_G = sum(p(t) for t in G)` over its supported members and let `c_G` be its highest-probability candidate. If `c_G` is already `t*`, continue to the next eligible input without creating an earlier stopping point. For alternative candidates, both conditions must hold:

   - **`R`: the minimum category-mass ratio.** Require `S_G / p* >= R`: the category's total probability must be at least `R` times the original winner's probability. With `R = 2` and `p* = 4%`, the category must have `S_G >= 8%`.
   - **`delta`: the maximum relative probability deficit of the group's best candidate.** Require `(p* - p(c_G)) / p* <= delta`. A value of `delta = 0.10` allows the candidate to be at most 10% below the original winner, relative to that winner's probability. With `p* = 4%`, this requires `p(c_G) >= 3.6%`.

7. **Replace or stop.** Select the first passing alternative and stop. On failure, move to the next higher-attention input, not a secondary category of the same input. If none passes before the stopping source, keep `t*`.

All examples use `R = 2` and `delta = 0.10`. They are experimental defaults, not independently calibrated settings. The interventions follow the [feature ablation approach](https://captum.ai/docs/attribution_algorithms); attribution depends on the replacement baseline.

## Test 1: `I put ketchup on my`

The original top eight probabilities (leading spaces omitted in these labels) are:

- `face`: 4.045409%
- `salad`: 3.703902%
- `burger`: 3.204580%
- `sandwich`: 2.200105%
- `head`: 1.764385%
- `plate`: 1.325593%
- `fries`: 1.071093%
- `car`: 0.989428%

Using the last `my` position as the query, the aggregated input-token ranking is:

`I` 60.28% → `on` 12.31% → `etchup` 10.01% → `my` 8.07% → `put` 5.69% → `k` 3.66%.

`I` has the highest attention. This is an attention summary, not a causal contribution ranking for the output `face`.

The actual decision trace is:

1. All three ablation baselines identify `my` as the strongest positive source for `face`.
2. Eligible inputs are `I`, `on`, and `etchup`, in that order. `my` itself is excluded.
3. At `I`, the supported food members are `salad, fries, steak, pizza, chicken, soup, rice, eggs, chips`, with original mass **8.2084%**, or **2.029×** the probability of `face`.
4. `salad` has a relative deficit of **8.442%**, so both gates pass at `I`. The procedure does not visit `on` or `etchup`.

Selection happens at `I`, rather than at `etchup` as I initially expected. My interpretation is that the full context, including ketchup, constrains the likely continuations; summing related candidates exposes support spread across them even while inspecting `I`. This is an interpretation of the complete-context result, not a measured conditional probability of `etchup` given `I`, nor evidence that `I` itself has food semantics. The sum is the original probability of the supported candidate set, not probability contributed exclusively by `I`.

The postprocessor selects **`salad`**. Its model probability remains **3.703902%**. It does not inherit the group's probability mass. The absolute gap from `face` is **0.341507 percentage points**, while the relative gap is **8.442%**.

The command above writes the complete report to `outputs/prompt-1.json`.

English output for the replacement example:

```text

Prompt: "I put ketchup on my"
  Original: " face" 4.045409%
  Selected: " salad" 3.703902%
  Decision: Selected the best candidate from the first category that passed all gates.

Top 8 next-token candidates (original probabilities):
Rank  Token ID    Probability  Token
   1      1986      4.045409%  " face"
   2     20698      3.703902%  " salad"
   3     26593      3.204580%  " burger"
   4     20433      2.200105%  " sandwich"
   5      1182      1.764385%  " head"
   6      7480      1.325593%  " plate"
   7     31757      1.071093%  " fries"
   8      1097      0.989428%  " car"
  Listed probability mass: 18.304497%

Input token attention (last input position, averaged over all layers and heads):
Rank  Input index  Token ID   Attention  Token
   1            0        40  60.277733%  "I"
   2            4       319  12.307302%  " on"
   3            3     47132  10.005898%  "etchup"
   4            5       616   8.067307%  " my"
   5            1      1234   5.685681%  " put"
   6            2       479   3.656077%  " k"

Attribution for the original winner (log-probability drops; columns: zero, mean, space):
  input[0] "I": -1.152272, -0.956069, -0.522831
  input[1] " put": -0.317770, -0.104044, +0.479395
  input[2] " k": -0.553766, +0.230486, +0.415522
  input[3] "etchup": +0.921716, +1.162362, +1.122655
  input[4] " on": +1.991290, +3.269972, +2.222056
  input[5] " my": +3.412977, +6.159705, +14.261715
  Estimated source / stopping point: input[5] " my"; attention=8.067307%. Only strictly higher-attention inputs are evaluated.

Selection trace:
  1. input[0] "I"; attention=60.277733%; primary group=food
     Group mass / original: 2.029x (required >= 2x)
     Best candidate relative deficit: 8.442% (allowed <= 10.000%)
     Selected the best candidate from the first category that passed all gates.

Semantic groups:
  Group 'personal_pronouns', input[0] "I": total probability=0.017239%
    Supported candidates: [" I"]
    Mass / original=0.004x; best relative deficit=99.574%
    Best candidate: " I" 0.017239%
  Group 'food', input[0] "I": total probability=8.208383%
    Supported candidates: [" salad", " fries", " steak", " pizza", " chicken", " soup", " rice", " eggs", " chips"]
    Mass / original=2.029x; best relative deficit=8.442%
    Best candidate: " salad" 3.703902%
  Group 'body_parts', input[0] "I": total probability=1.610022%
    Supported candidates: [" fingers", " hair", " legs", " arms"]
    Mass / original=0.398x; best relative deficit=77.032%
    Best candidate: " fingers" 0.929156%
  Group 'personal_objects', input[0] "I": total probability=3.360051%
    Supported candidates: [" car", " shirt", " bed", " pants", " jacket", " coat", " clothes", " phone", " shoes", " bag", " dress", " hat"]
    Mass / original=0.831x; best relative deficit=75.542%
    Best candidate: " car" 0.989428%
```

## Test 2: replace ketchup with mustard — `I put mustard on my`

Run `.venv/bin/python semantic_category_calibration.py "I put mustard on my"`.

Both the original and selected token are **`face`**, with probability **4.597261%**. All three baselines identify `my` as its source, so this example reaches category evaluation and the probability gates.

1. At `I`, the primary category is `personal_objects`. Its mass is only **0.820×** the probability of `face`, and its best candidate trails by **84.669%**. Both gates fail.
2. At `on` and then `mustard`, the primary category is `body_parts`, whose best candidate is already `face`. Neither proposes a replacement.
3. The procedure reaches the stopping point `my` and retains `face` without evaluating that input.

I found the `body_parts` result at `mustard` surprising and suspect that the training data may play a role. This experiment does not identify the training examples responsible. Unlike the source-disagreement example below, this case exercises retention after the actual category checks.

English output through the selection trace:

```text

Prompt: "I put mustard on my"
  Original: " face" 4.597261%
  Selected: " face" 4.597261%
  Decision: Reached the original winner's attributed source; keeping the original choice without evaluating that source or lower-attention tokens.

Top 8 next-token candidates (original probabilities):
Rank  Token ID    Probability  Token
   1      1986      4.597261%  " face"
   2     20698      2.846544%  " salad"
   3      1182      2.835230%  " head"
   4      2832      1.053492%  " hands"
   5     11914      1.047497%  " lips"
   6      7480      1.033825%  " plate"
   7      7660      1.031808%  " finger"
   8     26593      0.979568%  " burger"
  Listed probability mass: 15.425225%

Input token attention (last input position, averaged over all layers and heads):
Rank  Input index  Token ID   Attention  Token
   1            0        40  63.549868%  "I"
   2            3       319  12.951907%  " on"
   3            2     33694   9.117294%  " mustard"
   4            4       616   8.255009%  " my"
   5            1      1234   6.125921%  " put"

Attribution for the original winner (log-probability drops; columns: zero, mean, space):
  input[0] "I": -0.791062, -0.685913, -0.259855
  input[1] " put": -0.439240, -0.248299, +0.215239
  input[2] " mustard": +0.762165, +2.127613, +1.343091
  input[3] " on": +2.831492, +4.074809, +2.355657
  input[4] " my": +5.047734, +6.524763, +13.982595
  Estimated source / stopping point: input[4] " my"; attention=8.255009%. Only strictly higher-attention inputs are evaluated.

Selection trace:
  1. input[0] "I"; attention=63.549868%; primary group=personal_objects
     Group mass / original: 0.820x (required >= 2x)
     Best candidate relative deficit: 84.669% (allowed <= 10.000%)
     The category's total probability is below the required ratio.
  2. input[3] " on"; attention=12.951907%; primary group=body_parts
     This category's best candidate is the original winner; continue to the next higher-attention input token.
  3. input[2] " mustard"; attention=9.117294%; primary group=body_parts
     This category's best candidate is the original winner; continue to the next higher-attention input token.
```

## Test 3: replace my with a — `I put ketchup on a`

Selected original probabilities:

- `piece`: 2.870024%
- `hot`: 2.382185%
- `sandwich`: 2.357416%
- `plate`: 2.125851%
- `burger`: 1.917004%
- `salad`: 1.840515%

The zero-embedding baseline identifies `on` as the strongest source for `piece`; the mean and space baselines identify `a`. Because the sources disagree, the procedure retains **`piece`** without evaluating categories or probability gates.

The command above writes the complete report to `outputs/prompt-2.json`.

## Test 4: replace ketchup with cream — `I put cream on my`

Run `.venv/bin/python semantic_category_calibration.py "I put cream on my"`.

- Original: **`face`**, **19.713334%**.
- Selected: **`face`**, **19.713334%**.

The estimated stopping source is `my`. The alternative at `I` fails both gates; at `on` and `cream`, the primary category already favors `face`. Retention agrees with the selection rule and the ordinary interpretation of putting cream on one's face.

## Test 5: replace my with her — `I put ketchup on her`

Run `.venv/bin/python semantic_category_calibration.py "I put ketchup on her"`.

- Original: **`face`**, **8.639087%**.
- Selected: **`face`**, **8.639087%**.

The estimated stopping source is `her`. The alternative at `I` fails both gates; at `on`, the primary category already favors `face`. The result is consistent with the rule without establishing that it is the best continuation.

The single-prompt commands reuse `outputs/prompt-1.json`; use different `--output-dir` values to preserve each run.

## Analysis

The five examples show three distinct paths: replacement when both probability gates pass, retention after category checks find no eligible replacement, and early retention when source estimates disagree. Only the first two paths exercise the replacement checks. The outcomes are consistent with the implemented rule, but they do not establish better general accuracy.

Category mass now depends on the examined input's supported members. At `I` in Test 1, the food mass is **8.2084%**; using the full configured food-list mass for every input would bypass attribution.
Other limitations:

- Larger or broader groups can accumulate more probability. Group coverage, overlap, and membership affect the result.
- Zero and mean embeddings can be outside ordinary language inputs; space replacement also changes grammar. Baseline agreement does not prove a unique causal source. Effects on log probabilities can reflect competition among candidates and input interactions.
- Configured categories do not cover every vocabulary token; automatically estimating support does not automatically discover semantic classes.
- The examples informed the design and are not independent generalization tests.
- `face` is not universally incorrect; it could fit a prank or makeup context. The target is a conventional-context preference.
- Category aggregation covers only one possible source of undesirable continuation. It does not solve general reasoning or factual accuracy.

A next evaluation should compare raw greedy decoding, grouping with probability gates but without attention, and the same method with attention ordering. Tune thresholds separately from evaluation data. Measure desired corrections, unwanted replacements, and computational overhead across diverse contexts, including explicit unconventional requests. Explore combining Semantic Category Calibration with Supervised Fine-Tuning and Preference Learning, with Semantic Category Calibration taking lower priority than both.

## Preliminary conclusion

The prototype demonstrates that an unchanged model probability distribution can support an explicit alternative decision policy based on input attribution, semantic groups, attention ordering, and candidate-level margins. The five examples demonstrate both replacement and retention, including retention after the category and probability checks. It has not established general improvement or applicability to larger models.

This is an inference-time decision-rule hypothesis, not a change to learned model parameters. It introduces an additional preference for contextually appropriate categories.

## Potential benefit and cost

Related candidates can contribute collectively to a decision even when none individually wins. The margin gate limits how far the replacement can depart from the original winner.

Costs include obtaining attention, estimating support, and evaluating groups. For `n` input tokens, attribution adds `3n` complete forward passes, and the cost of each pass also grows with sequence length; total cost is not simply linear. Raw mode skips attribution. A general version might require an additional classifier or semantic model. Latency, memory use, and overall cost–benefit have not been measured.

## Open questions

1. How robust are ablation-based associations across replacement baselines and prompts, and how should semantic categories and members be discovered automatically? Polysemy, subwords, and multi-token expressions remain open issues.
2. How should unequal category sizes, incomplete coverage, overlap, and nearly tied competing categories be handled?
3. Does attention provide value beyond semantic grouping? Is averaging all layers/heads appropriate, especially when words such as `I` or `on` rank highly?
4. How should thresholds be calibrated? Token-probability standard deviations describe dispersion, not prediction error, confidence intervals, or statistical significance. In this implementation, standard deviations are diagnostic only.
5. What decision objective should replace single-token probability maximization? How should conventionality, user intent, useful corrections, and unwanted overrides be evaluated?

Feedback on the probabilistic interpretation, decision rule, and experimental design is welcome.
