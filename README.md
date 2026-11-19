# Semantic Category Calibration

Semantic Category Calibration is an inference-time decoding postprocessing method. It combines semantic categories, input-token attention, and candidate probabilities to adjust next-token selection while preserving model weights and the original predicted probabilities.

<a id="existing-models-and-remote-execution"></a>

## Quick start

You need Python 3.9–3.11 and a local GPT-2 Small model directory containing configuration, tokenizer files, and safetensors weights. Run these commands from the project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python semantic_category_calibration.py "I put ketchup on my" --model-dir /path/to/gpt2
```

Replace `/path/to/gpt2` with your local model directory and the prompt with your own input.

## Output

The terminal shows the original and calibrated choices, the top 8 candidates by default, input attention, and attribution results.

The full JSON report is saved to `outputs/prompt-1.json` by default. Another run overwrites the same file; use `--output-dir` to specify a different output directory.

See [EXPERIMENT.md](docs/EXPERIMENT.md) for the method and experiments. For more arguments, run `.venv/bin/python semantic_category_calibration.py --help`.
