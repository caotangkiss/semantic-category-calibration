```bash
% .venv/bin/python semantic_category_calibration.py "I put ketchup on my"

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