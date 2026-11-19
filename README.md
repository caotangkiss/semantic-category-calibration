# Semantic Category Calibration

[Experiment](docs/EXPERIMENT.md)

Semantic Category Calibration is a proposed method for improving model output, discussed alongside Supervised Fine-Tuning and Preference Learning. When combined, it takes lower priority than both. This implementation uses semantic categories, input-token attention, and probability thresholds to select the next token without updating GPT-2 weights or changing its probabilities.

## Files

- `semantic_category_calibration.py`: model loading, calibration logic, and English reports.
- `test_calibration.py`: tests for the postprocessor and command-line behavior.
- `selection-policy.json`: editable semantic categories, candidate members, and thresholds.
- `next-token-settings.json`: default model path, output path, and candidate count.
- `requirements.txt`: pinned dependencies.

The `docs/` directory explains the method and experiment results.

## Quick start

Use Python 3.9–3.11. From the project directory:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use your installed Python 3.9 or 3.10 executable instead of `python3.11` if needed. Each user creates their own environment and installs `requirements.txt` after cloning. An existing environment can be reused. The repository includes its own `next-token-settings.json` and `selection-policy.json`; `.venv`, model weights, caches, and generated reports are excluded from Git.

If you already have GPT-2 Small, skip downloading and specify its directory with `--model-dir`. Otherwise, download the reference checkpoint using the installed Hugging Face library:

```bash
.venv/bin/python - <<'PYTHON'
from huggingface_hub import snapshot_download
from semantic_category_calibration import DEFAULT_MODEL_DIR

snapshot_download(
    repo_id="openai-community/gpt2",
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    local_dir=str(DEFAULT_MODEL_DIR),
    allow_patterns=["config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
)
PYTHON
```

Run with your own prompt:

```bash
.venv/bin/python semantic_category_calibration.py "The capital of France is"
```

The prompt is required. The English report shows the original and selected tokens, the top 8 candidates by default, input-token attention, attribution of the original winner, its stopping source, selection steps, and supported category members. JSON reports are written automatically.

## Commands

```bash
# Several prompts, loading the model once.
.venv/bin/python semantic_category_calibration.py "I put ketchup on my" "I put ketchup on a" --output-dir outputs/my-run

# Name the JSON report for a single prompt.
.venv/bin/python semantic_category_calibration.py "I put ketchup on my" --json result.json

# Adjust the category-mass ratio, candidate margin, and displayed count.
.venv/bin/python semantic_category_calibration.py "I put ketchup on my" --min-group-ratio 3 --max-relative-gap 0.10 --top-k 10

# Compare against raw greedy selection.
.venv/bin/python semantic_category_calibration.py "I put ketchup on my" --mode raw

# Include style requirements in the complete model prompt.
.venv/bin/python semantic_category_calibration.py "Write a surreal sentence: I put ketchup on my"
```

The full prompt is passed to GPT-2 so context and style requirements can affect attention and next-token probabilities. Use `--policy path/to/policy.json` for different categories and candidate members. Version 3 removes `anchors`, `context_patterns`, and `association_basis`; associations are estimated from the model. `--std-top-k` controls diagnostics only. See `--help` for all options.

Numbered reports use `prompt-1.json`, `prompt-2.json`, etc.; another run in the same directory overwrites them. `--json` accepts one prompt and cannot be combined with `--output-dir`. A bare JSON filename uses the configured output directory; an absolute path or a path containing a subdirectory uses that location. The specified parent directory must already exist.

## Existing models and remote execution

```bash
.venv/bin/python semantic_category_calibration.py "your prompt" --model-dir /path/to/gpt2
```

The model directory needs compatible configuration, tokenizer files, and safetensors weights. Inference uses local files. You can also run the project on a remote machine over SSH. A remote HTTP API adapter is not implemented; the algorithm needs full vocabulary probabilities, input-token attention, matching tokenization, and repeated inference with controlled input embeddings. A text-generation-only endpoint is insufficient.

Default paths come only from this project's `next-token-settings.json`, relative to its directory. Command-line arguments override defaults. This project does not import code or settings from the prediction project.

## Python API

```python
from semantic_category_calibration import DEFAULT_MODEL_DIR, calibrate_next_token, load_model

tokenizer, model = load_model(DEFAULT_MODEL_DIR)
report = calibrate_next_token("I put ketchup on my", tokenizer, model)
print(report["decision"]["selected"])
```

You can reuse an already loaded tokenizer and model. The model must be in evaluation mode and provide eager attention; the current loader runs GPT-2 on CPU. For integration with an existing inference step, call `attribute_candidates(inputs, logits, attention, tokenizer, model, policy)`, then pass its result as `attribution=` to `choose_token`. Probabilities and attention alone cannot identify the stopping source.

With the reference model and policy, `I put ketchup on my` selects `salad` instead of `face` while inspecting `I`. The mustard, cream, and her variants retain `face` after category evaluation finds no eligible replacement. Separately, `I put ketchup on a` retains `piece` before category evaluation because its estimated source differs across baselines. Categories and members remain explicit; input-to-candidate associations are estimated by model interventions. These five examples illustrate the rule; broader improvement needs separate evaluation. See the [experiment document](docs/EXPERIMENT.md) for commands, measurements, and English output.

## Attribution and stopping rule

1. Run the original prompt once and retain its full probabilities and input-token attention.
2. For each input token, replace only its token embedding with zero, the mean vocabulary embedding, and the space-token embedding in separate runs. Keep the input length, positions, other embeddings, and output position unchanged. Compute the original minus perturbed log probability for the original winner and every configured candidate.
3. The original winner has an estimated source only when all three baselines identify the same unique input token with the largest positive drop. Otherwise, retain the original winner.
4. Examine input tokens in descending attention, strictly above that source's attention. The source itself and attention ties are excluded. At each examined input, retain only category members with positive drops under all three baselines, then select the largest category by summed original probability. Break mass ties by the best member probability, then random choice if still tied.
5. For an alternative candidate, require the existing mass ratio `R` and relative deficit `delta` gates (defaults: 2 and 0.10). If a category's best candidate is already the original winner, continue without treating that input as a new stopping source. Retain the original winner if no eligible replacement passes.

This is baseline-dependent attribution, not proof of a unique causal origin or automatic semantic classification. A `1e-6` tolerance handles near-zero scores and near-tied source estimates; it is not a calibrated confidence threshold. JSON includes per-candidate drops for each input and all three baselines. The [feature ablation method](https://captum.ai/docs/attribution_algorithms) motivates these interventions; no new dependency is required.

For `n` input tokens, auto mode adds `3n` complete forward passes. Long prompts can be expensive; raw mode skips attribution. Model weights and the original probability distribution remain unchanged.

## Tests

After installing dependencies and providing the local reference checkpoint:

```bash
.venv/bin/python -B -m unittest -v test_calibration.py
```
