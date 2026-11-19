# Semantic Category Calibration: can category probabilities improve next-token selection?

This article explores **Semantic Category Calibration**, an inference-time decoding postprocessing method. It combines semantic categories, input attention, and candidate probabilities to adjust next-token selection while preserving model weights and the original predicted probabilities.

## Hypothesis

When GPT-2 uses greedy decoding for `I put ketchup on my`, it selects `face`. In an ordinary eating context, however, we would expect a food-related continuation such as `burger` or `salad`.

Looking at the original distribution, `face` has the highest individual probability, but its lead is small; several food candidates have a relatively high combined probability, and the best of them is very close to `face`.

The guess is that `ketchup` receives the most attention, or at least more than `my`, and that ketchup is associated with food while my is associated with body parts. Could we therefore combine input-word attention rankings, total category probabilities, and probability gaps to select a candidate closer to our expectations during postprocessing?

Based on this hypothesis, I worked with Codex and Doubao on a simple experiment that roughly verified the idea. The procedure follows:

## Experimental setup

Use GPT-2 Small.

Run `.venv/bin/python semantic_category_calibration.py "I put ketchup on my" "I put ketchup on a"`

Semantic categories and their members are explicitly configured in [selection-policy.json](../selection-policy.json), while associations between input tokens and candidates are calculated through model attribution. The food group has 12 members:

`salad, burger, sandwich, fries, chips, pizza, eggs, steak, rice, soup, chicken, toast`

The actual forms used are single tokens with a leading space. The "total food-group probability" refers only to the sum of the original probabilities of explicitly listed candidates that receive positive support from the current input token (for example, the food token "potato" does not enter the sum if it is not configured in the JSON file). It does not represent the complete probability of all food expressions. Duplicate tokens within a group are counted once; multi-token members are skipped.

Associations such as `my → body parts` and `ketchup → food` are no longer predefined. For each input token, replace its input vector in turn and measure the change in each candidate's probability; semantic categories are still configured by the user.

## Complete procedure

Notation:
- `t` represents any next-token candidate.
- `p(t)` represents the original probability predicted by the model for that candidate.
- `t`* specifically denotes the candidate with the highest probability across the entire vocabulary (the original winner). The star `*` marks the winner; `p* = p(t*)` is its probability. Original probabilities are used throughout the subsequent steps.

1. **Estimate the original winner's main source as the stopping point.**
Perform attribution experiments:
- Replace only one input token's embedding at a time, using a zero vector, the mean embedding of the entire vocabulary, and the space-token embedding as separate baselines.
- Keep the input length, positions, other tokens, and output position unchanged.
- The attribution score is `log(original probability) - log(probability after replacement)`. A positive value means that this input supports the candidate relative to that baseline.
- Use an input token as the stopping point only when all three baselines identify it as the unique source with the strongest positive support for `t*`; otherwise, retain the original winner. A `1e-6` tolerance handles values close to zero or ties; it is not a confidence threshold.
- If the sources selected by the three baselines disagree (for example, they point to `on` and `a`), the current version does not handle multiple possible stopping points, so it directly retains the original winner without proceeding to category selection. This is a conservative choice: it does not mean that the original candidate is necessarily more appropriate, nor that other categories failed the probability gates; it may miss a possible replacement.

2. **Inspect inputs with higher attention than the stopping point and identify the category members they support.**
- Use the last input position as the query, average equally over 12 layers and 144 heads, and inspect inputs in descending attention order.
- The stopping point itself does not participate in category selection or gate comparisons; inputs with equal or lower attention are also excluded. For example, if the estimated source of `face` is `my`, consider only inputs with higher attention than `my`.

3. **Sum the original probabilities of these members.**
- Reuse the attribution experiments to check whether members of each **manually configured** category receive positive support from the current input token under all three baselines.
- For each group, sum only the original probabilities of these supported members to obtain the group probability.

4. **Check the replacement gates. (Select the group's best candidate when the total category probability is high enough and that candidate is not far behind.)**
- Select the category with the highest group probability.
- The group probability is `S = sum(p(t))`; its highest-probability candidate is `c`, whose probability is `p(c)`.
- If `c` is `t*`, make no replacement and continue to the next input with higher attention than the stopping point.
- When total probabilities are tied, compare each group's highest candidate probability and select the group closer to the original winner. Choose randomly only when both measures are tied.
- Both of the following conditions must hold to replace the original next token:
   - **`R`: the minimum multiple for total category probability.** Require `S / p* >= R`, meaning that the total category probability must be at least `R` times the original winner's probability. `R=2` means at least 2 times; for example, when `p*=4%`, require `S>=8%`.
   - **`delta`: the maximum allowed gap between the group's best candidate and the original winner, relative to the original winner.** Require `(p* - p(c)) / p* <= delta`. `delta=0.10` means a relative deficit of at most 10%; for example, when `p*=4%`, require `p(c)>=3.6%`.

5. **Replace or stop.**
- Select the best candidate from the first category that passes the gates, then stop.
- If the gates fail, inspect the next input with higher attention than the stopping point.
- If no replacement candidate has been found by the stopping point, retain the original winner.

The tests all use **R=2, delta=0.10**, merely an uncalibrated starting point for the experiment. Attribution follows the idea of [feature ablation](https://captum.ai/docs/attribution_algorithms); it estimates contribution relative to a baseline, rather than proving a unique causal source.

## Expected replacement of the original next token: I put ketchup on my — result: face replaced with salad

The top eight original probabilities (leading spaces omitted in the display):

- `face`: 4.045409%
- `salad`: 3.703902%
- `burger`: 3.204580%
- `sandwich`: 2.200105%
- `head`: 1.764385%
- `plate`: 1.325593%
- `fries`: 1.071093%
- `car`: 0.989428%

Attention from the final `my` to each input token, averaged over all layers and heads:

`I` 60.28% → `on` 12.31% → `etchup` 10.01% → `my` 8.07% → `put` 5.69% → `k` 3.66%.

`I` has the highest attention.

The actual procedure:
- All three attribution baselines identify `my` as the input with the strongest support for `face`, so `my` becomes the stopping point.
- The inputs with higher attention than `my` are `I`, `on`, and `etchup`, in that order; `my` itself is not inspected.
- When inspecting `I`, the food-group members with consistent positive support are `salad, fries, steak, pizza, chicken, soup, rice, eggs, chips`. Their total original probability is **8.2084%**, or **2.029 times** that of `face`. (It is `I`, rather than `etchup`, which went beyond my initial expectation that replacement would be triggered while inspecting etchup. My hypothesis is that the candidates supported by `I` and etchup overlap, including food candidates; the full context may give these food candidates a relatively high combined probability, leading to the selection of the food group while inspecting I.)
- The group's best candidate, `salad`, trails by **8.442%** in relative terms. Both conditions pass, so **salad** is selected, and `on` and `etchup` are not inspected further.

After running the command above, the complete JSON result is at `outputs/prompt-1.json`.

Test output:
```shell
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

## Replace ketchup with mustard: I put mustard on my — result: face is not replaced
Test results:
Original: " face" 4.597261%
Selected: " face" 4.597261%

Analysis: This is somewhat counterintuitive; I think it is an issue with the training data. The primary category for `mustard` is body_parts—what on earth was this model fed, haha?

Test output:
```shell
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

## Replace my with a: I put ketchup on a — result: piece is not replaced

The original candidates include `piece` **2.870024%**, `hot` **2.382185%**, `sandwich` **2.357416%**, `plate` **2.125851%**, `burger` **1.917004%**, and `salad` **1.840515%**.

For the original winner `piece`, the zero-vector baseline selects `on` as its source, while the mean-embedding and space-embedding baselines select `a`. Since the source estimates disagree, **piece** is retained directly, without traversing attention rankings or comparing category gates.

After running the command above, the complete JSON result is at `outputs/prompt-2.json`.

## Replace ketchup with cream: "I put cream on my" — result: face is not replaced
[Test results]
  Original: " face" 19.713334%
  Selected: " face" 19.713334%

[Analysis] Consistent with mathematical expectations and common sense.

## Replace my with her, which is uncommon here: "I put ketchup on her" — result: face is not replaced
[Test results]
Original: " face" 8.639087%
Selected: " face" 8.639087%

[Analysis] Consistent with mathematical expectations.

## Full test output examples

- [Replacement: I put ketchup on my (face → salad)](I_put_ketchup_on_my.md)
- [Original retained: I put mustard on my (face unchanged)](I_put_mustard_on_my.md)

## Analysis

The tests above are consistent with mathematical expectations.

## Conclusion

The experiment provides preliminary verification of the feasibility of this selection approach: while preserving the model's original probabilities, combining model attribution, semantic grouping, attention traversal order, and candidate-gap gates can produce an interpretable and adjustable rule for next-token selection.
However, there are too few test cases to establish an overall quality improvement or applicability to larger models.

## Reflections

Limitations:
- Manually configured categories do not cover the entire vocabulary.
- Attribution is also only an estimate of the effect on probability;
- Categories are just one angle on outputs that do not meet expectations; they do not solve all common-sense, factual, and reasoning problems.

Further verification and exploration:

- More tests: to verify the hypothesis and adjust the thresholds.
- Distinguish the dataset from the test set.
- Introduce other optimization mechanisms to work together, such as Supervised Fine-Tuning and Preference Learning; I believe Semantic Category Calibration should take lower priority than both.

## Benefits and costs

- The potential benefit is to let information whose probability is spread across several related candidates contribute jointly to the decision, while using the candidate gap to limit replacements. Costs include obtaining attention, establishing associations, and category calculations;
- An automated version may also need an additional classifier. Latency, memory use, and the overall cost–benefit have not been measured in this experiment.
- For a prompt containing `n` input tokens, attribution adds `3n` complete forward passes, and the cost of each inference pass also grows with input length.

# Discussion welcome

Feedback on the interpretation of probabilities, the rules, and the experimental design is welcome, as are recommendations of related research.
