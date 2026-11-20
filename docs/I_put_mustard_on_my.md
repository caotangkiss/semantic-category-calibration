```bash
% .venv/bin/python semantic_category_selection.py "I put mustard on my"

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

Semantic groups:
  Group 'personal_pronouns', input[0] "I": total probability=0.026186%
    Supported candidates: [" I", " me"]
    Mass / original=0.006x; best relative deficit=99.563%
    Best candidate: " I" 0.020104%
  Group 'food', input[0] "I": total probability=1.735658%
    Supported candidates: [" chicken", " pizza", " rice", " fries", " eggs"]
    Mass / original=0.378x; best relative deficit=86.145%
    Best candidate: " chicken" 0.636965%
  Group 'body_parts', input[0] "I": total probability=1.957486%
    Supported candidates: [" fingers", " hair", " legs", " arms"]
    Mass / original=0.426x; best relative deficit=80.633%
    Best candidate: " fingers" 0.890329%
  Group 'personal_objects', input[0] "I": total probability=3.771675%
    Supported candidates: [" car", " shirt", " bed", " pants", " clothes", " shoes", " dress", " coat", " bag", " hat", " jacket", " phone"]
    Mass / original=0.820x; best relative deficit=84.669%
    Best candidate: " car" 0.704792%
  Group 'personal_pronouns', input[3] " on": total probability=0.001533%
    Supported candidates: [" him"]
    Mass / original=0.000x; best relative deficit=99.967%
    Best candidate: " him" 0.001533%
  Group 'food', input[3] " on": total probability=7.855298%
    Supported candidates: [" salad", " burger", " toast", " sandwich", " steak", " chicken", " pizza", " rice", " fries", " soup", " chips"]
    Mass / original=1.709x; best relative deficit=38.082%
    Best candidate: " salad" 2.846544%
  Group 'body_parts', input[3] " on": total probability=17.318421%
    Supported candidates: [" face", " head", " hands", " lips", " finger", " fingers", " skin", " back", " hand", " nose", " feet", " hair", " chest", " legs", " leg", " arm", " neck", " arms"]
    Mass / original=3.767x; best relative deficit=0.000%
    Best candidate: " face" 4.597261%
  Group 'personal_objects', input[3] " on": total probability=5.481284%
    Supported candidates: [" plate", " car", " table", " shirt", " bed", " pants", " clothes", " shoes", " dress", " coat", " bag", " hat", " jacket", " phone", " desk"]
    Mass / original=1.192x; best relative deficit=77.512%
    Best candidate: " plate" 1.033825%
  Group 'personal_pronouns', input[2] " mustard": total probability=0.018383%
    Supported candidates: [" her", " she"]
    Mass / original=0.004x; best relative deficit=99.783%
    Best candidate: " her" 0.009969%
  Group 'food', input[2] " mustard": total probability=7.959744%
    Supported candidates: [" salad", " burger", " toast", " sandwich", " steak", " chicken", " pizza", " rice", " fries", " soup", " eggs", " chips"]
    Mass / original=1.731x; best relative deficit=38.082%
    Best candidate: " salad" 2.846544%
  Group 'body_parts', input[2] " mustard": total probability=12.890793%
    Supported candidates: [" face", " hands", " lips", " finger", " fingers", " skin", " nose", " mouth", " feet", " hair", " legs", " leg", " neck", " arms"]
    Mass / original=2.804x; best relative deficit=0.000%
    Best candidate: " face" 4.597261%
  Group 'personal_objects', input[2] " mustard": total probability=2.993536%
    Supported candidates: [" plate", " car", " table", " pants", " clothes"]
    Mass / original=0.651x; best relative deficit=77.512%
    Best candidate: " plate" 1.033825%
```