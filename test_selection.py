"""Offline behavioral tests using the installed local GPT-2 checkpoint."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from semantic_category_selection import DEFAULT_MODEL_DIR, select_next_token, encode_prompt, load_model
import semantic_category_selection as selection
from semantic_category_selection import choose_token, load_policy, validate_policy

PROMPT = "I put ketchup on my"


class SelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.model = load_model(DEFAULT_MODEL_DIR)
        cls.policy = load_policy()
        cls.report = select_next_token(PROMPT, cls.tokenizer, cls.model)
        inputs, _ = encode_prompt(PROMPT, cls.tokenizer, cls.model)
        with torch.inference_mode():
            cls.probs = cls.model(**inputs, use_cache=False).logits[0, -1].double().softmax(-1)

    def synthetic_probs(self, values):
        by_id = {self.tokenizer.encode(" " + word, add_special_tokens=False)[0]: p for word, p in values.items()}
        result = torch.full((self.tokenizer.vocab_size,), (1 - sum(by_id.values())) / (self.tokenizer.vocab_size - len(by_id)), dtype=torch.float64)
        for token_id, value in by_id.items():
            result[token_id] = value
        return result

    def food_group(self, policy=None):
        return next(group for group in (policy or self.policy)["groups"] if group["id"] == "food")

    def source_group(self, group_id, members):
        return {"id": group_id, "label": group_id, "members": members}

    def evidence(self, support, stop="my", attention=None):
        tokens = (attention or self.report["attention"])["tokens"]
        source = next((token for token in tokens if token["text"].strip() == stop), None)
        return {"original_source": source, "baselines": list(selection.ATTRIBUTION_BASELINES),
                "tokens": [{"source": token, "original_drops": [1.0] * len(selection.ATTRIBUTION_BASELINES),
                            "supported_token_ids": [self.tokenizer.encode(" " + word, add_special_tokens=False)[0]
                                                    for word in support.get(token["text"].strip(), [])]}
                           for token in tokens]}

    def decide(self, policy=None, probs=None, attribution=None, attention=None, **kwargs):
        return choose_token(PROMPT, self.probs if probs is None else probs, self.tokenizer,
                            attention or self.report["attention"], policy or self.policy,
                            attribution=self.report["decision"]["attribution"] if attribution is None else attribution, **kwargs)

    def test_model_attribution_finds_my_without_manual_associations(self):
        decision = self.report["decision"]
        attribution = decision["attribution"]
        self.assertEqual(decision["original"]["text"], " face")
        self.assertEqual(attribution["original_source"]["text"], " my")
        self.assertEqual(attribution["baseline_source_indices"], [5, 5, 5])
        self.assertEqual(attribution["additional_forward_passes"], 18)
        self.assertTrue(all("anchors" not in group and "context_patterns" not in group for group in self.policy["groups"]))
        self.assertEqual(decision["selected"]["text"], " salad")
        self.assertEqual(decision["selected_source"]["text"], "I")
        self.assertAlmostEqual(decision["original"]["probability"], 0.0404540935, places=7)
        self.assertAlmostEqual(decision["selected"]["probability"], self.probs[decision["selected"]["token_id"]].item(), places=12)
        first = next(row for row in attribution["tokens"] if row["source"]["text"] == "I")
        self.assertNotIn(decision["original"]["token_id"], first["supported_token_ids"])
        self.assertIn(decision["selected"]["token_id"], first["supported_token_ids"])

    def test_attribution_preserves_positions_and_only_changes_one_embedding(self):
        inputs, spans = encode_prompt(PROMPT, self.tokenizer, self.model, offsets=True)
        original_inputs = {key: value.clone() for key, value in inputs.items()}
        with torch.inference_mode():
            embeddings = self.model.get_input_embeddings()(inputs["input_ids"]).clone()
            logits = self.model(**inputs, use_cache=False).logits[0, -1]
        forward = self.model.forward
        def checked_forward(*args, **kwargs):
            changed = (kwargs["inputs_embeds"] != embeddings).any(-1)
            self.assertEqual(int(changed.sum()), 1)
            self.assertEqual(tuple(kwargs["inputs_embeds"].shape), tuple(embeddings.shape))
            self.assertTrue(torch.equal(kwargs["attention_mask"], inputs["attention_mask"]))
            return forward(*args, **kwargs)
        with patch.object(self.model, "forward", side_effect=checked_forward):
            evidence = selection.attribute_candidates(inputs, logits, self.report["attention"], self.tokenizer, self.model, self.policy)
        self.assertEqual(evidence, self.report["decision"]["attribution"])
        self.assertTrue(all(torch.equal(inputs[key], value) for key, value in original_inputs.items()))
        self.assertAlmostEqual(evidence["tokens"][5]["original_drops"][0], 3.41297689, places=4)

    def test_inconsistent_source_keeps_original_without_category_evaluation(self):
        report = select_next_token("I put ketchup on a", self.tokenizer, self.model)
        decision = report["decision"]
        self.assertEqual(decision["attribution"]["baseline_source_indices"], [4, 5, 5])
        self.assertIsNone(decision["attribution"]["original_source"])
        self.assertEqual(decision["status"], "uncertain_source")
        self.assertEqual(decision["selected"], decision["original"])
        self.assertEqual(decision["groups"], [])
        self.assertEqual(decision["trace"], [])

    def test_source_requires_positive_unique_agreement(self):
        for scores in [([1, 1, 1], [1, 1, 1]), ([-1, -1, -1], [-2, -2, -2]),
                       ([2, 1, 2], [1, 2, 1]), ([0, 0, 0], [-1, -1, -1])]:
            with self.subTest(scores=scores):
                rows = [{"source": {"index": index}, "original_drops": values} for index, values in enumerate(scores)]
                self.assertIsNone(selection._original_source(rows)[0])

    def test_stop_source_and_later_sources_are_never_evaluated(self):
        policy = copy.deepcopy(self.policy)
        policy["groups"] = [self.source_group("food", ["salad", "burger", "fries"])]
        probs = self.synthetic_probs({"face": 0.04, "salad": 0.038, "burger": 0.035, "fries": 0.01})
        evidence = self.evidence({"my": ["salad", "burger", "fries"], "put": ["salad", "burger", "fries"]})
        decision = self.decide(policy=policy, probs=probs, attribution=evidence)
        self.assertEqual(decision["status"], "accepted_original")
        self.assertEqual(decision["selected"], decision["original"])
        self.assertEqual([step["source"]["text"].strip() for step in decision["trace"]], ["I", "on", "etchup"])
        self.assertTrue(all(group["source"]["text"].strip() not in {"my", "put", "k"} for group in decision["groups"]))
        # Moving the same supported candidates above the stopping point permits replacement.
        evidence = self.evidence({"on": ["salad", "burger", "fries"]})
        self.assertEqual(self.decide(policy=policy, probs=probs, attribution=evidence)["selected"]["text"], " salad")

    def test_equal_attention_is_not_eligible(self):
        attention = copy.deepcopy(self.report["attention"])
        for token in attention["tokens"]:
            token["attention"] = 1 / len(attention["tokens"])
        evidence = self.evidence({"I": ["salad", "burger", "fries"]}, attention=attention)
        decision = self.decide(attribution=evidence, attention=attention)
        self.assertEqual(decision["status"], "accepted_original")
        self.assertEqual(decision["groups"], [])
        self.assertEqual(decision["trace"], [])

    def test_original_candidate_does_not_create_an_earlier_stopping_point(self):
        policy = copy.deepcopy(self.policy)
        policy["groups"] = [self.source_group("body", ["face"]), self.source_group("food", ["salad", "burger", "fries"])]
        evidence = self.evidence({"I": ["face"], "on": ["salad", "burger", "fries"]})
        probs = self.synthetic_probs({"face": 0.04, "salad": 0.038, "burger": 0.035, "fries": 0.01})
        decision = self.decide(policy=policy, probs=probs, attribution=evidence)
        self.assertEqual([step["status"] for step in decision["trace"]], ["original_candidate", "changed"])
        self.assertEqual(decision["trace"][0]["checks"], {})
        self.assertEqual(decision["selected"]["text"], " salad")

    def test_mass_counts_only_supported_members_without_duplicates(self):
        policy = copy.deepcopy(self.policy)
        policy["groups"] = [self.source_group("food", ["salad", "salad", "burger", "fries", "chicken"])]
        probs = self.synthetic_probs({"face": 0.04, "salad": 0.038, "burger": 0.035, "fries": 0.01, "chicken": 0.02})
        evidence = self.evidence({"I": ["salad", "burger", "fries"]})
        decision = self.decide(policy=policy, probs=probs, attribution=evidence)
        self.assertAlmostEqual(decision["groups"][0]["mass"], 0.083)
        self.assertEqual(len(decision["groups"][0]["members"]), 3)
        self.assertEqual(decision["selected"]["text"], " salad")

    def test_multitoken_members_do_not_count_the_first_fragment(self):
        word = "electroencephalography"
        ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
        self.assertGreater(len(ids), 1)
        group = self.source_group("example", [word, "salad"])
        members, skipped = selection.category_members(group, self.probs, self.tokenizer)
        self.assertEqual(skipped, [word])
        self.assertEqual([member["text"] for member in members], [" salad"])

    def test_probability_gates_both_apply_to_replacements(self):
        policy = copy.deepcopy(self.policy)
        policy["groups"] = [self.source_group("food", ["salad", "burger", "fries"])]
        evidence = self.evidence({"I": ["salad", "burger", "fries"]})
        for salad, burger, fries, expected in [(0.038, 0.035, 0.01, "changed"),
                                              (0.038, 0.001, 0.001, "small_group_mass"),
                                              (0.035, 0.034, 0.03, "wide_gap"),
                                              (0.036, 0.034, 0.01, "changed")]:
            with self.subTest(salad=salad, burger=burger):
                probs = self.synthetic_probs({"face": 0.04, "salad": salad, "burger": burger, "fries": fries})
                decision = self.decide(policy=policy, probs=probs, attribution=evidence)
                self.assertEqual(decision["trace"][0]["status"], expected)
                self.assertEqual(decision["changed"], expected == "changed")

    def test_category_ties_use_best_candidate_then_random(self):
        policy = copy.deepcopy(self.policy)
        policy["groups"] = [self.source_group("food", ["salad", "burger", "fries"]),
                            self.source_group("body", ["head", "foot", "arm"])]
        evidence = self.evidence({"I": ["salad", "burger", "fries", "head", "foot", "arm"]})
        for head, foot, expected in [(0.039, 0.034, "body"), (0.038, 0.035, "food")]:
            probs = self.synthetic_probs({"face": 0.04, "salad": 0.038, "burger": 0.035, "fries": 0.01,
                                          "head": head, "foot": foot, "arm": 0.01})
            with patch.object(selection.random, "choice", side_effect=lambda groups: next(g for g in groups if g["id"] == "food")) as draw:
                decision = self.decide(policy=policy, probs=probs, attribution=evidence)
            self.assertEqual(decision["selected_group"], expected)
            self.assertEqual(draw.call_count, int(expected == "food"))

    def test_only_primary_category_is_checked_and_failures_continue(self):
        policy = copy.deepcopy(self.policy)
        bad = ["head", "foot", "arm", "leg"]
        policy["groups"] = [self.source_group("body", bad), self.source_group("food", ["salad", "burger", "fries"])]
        probs = self.synthetic_probs(dict(face=0.04, salad=0.038, burger=0.035, fries=0.01, **{w:0.03 for w in bad}))
        evidence = self.evidence({"I": bad + ["salad", "burger", "fries"], "on": ["salad", "burger", "fries"]})
        decision = self.decide(policy=policy, probs=probs, attribution=evidence)
        self.assertEqual([step["status"] for step in decision["trace"]], ["wide_gap", "changed"])
        self.assertEqual(decision["selected_source"]["text"], " on")

    def test_raw_and_unsupported_inputs_do_not_run_ablation(self):
        with patch.object(selection, "attribute_candidates", side_effect=AssertionError("unexpected ablation")):
            raw = select_next_token(PROMPT, self.tokenizer, self.model, mode="raw")
            boundary = select_next_token(PROMPT + " ", self.tokenizer, self.model)
        self.assertEqual(raw["decision"]["status"], "raw")
        self.assertEqual(boundary["decision"]["status"], "unsupported_boundary")
        self.assertIsNone(raw["decision"]["attribution"])
        self.assertEqual(boundary["prompt"], PROMPT + " ")

    def test_missing_attribution_and_legacy_policy_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires attribution"):
            choose_token(PROMPT, self.probs, self.tokenizer, self.report["attention"], self.policy)
        policy = copy.deepcopy(self.policy)
        policy["groups"][0]["anchors"] = ["I"]
        with self.assertRaisesRegex(ValueError, "manual input associations"):
            validate_policy(policy)

    def test_attention_keeps_subwords_separate(self):
        tokens = self.report["attention"]["tokens"]
        self.assertAlmostEqual(sum(token["attention"] for token in tokens), 1, places=6)
        self.assertEqual([token["token_id"] for token in sorted(tokens, key=lambda token: token["index"])],
                         self.tokenizer.encode(PROMPT, add_special_tokens=False))
        self.assertIn("etchup", [token["text"] for token in tokens])
        self.assertIn(" k", [token["text"] for token in tokens])

    def test_selection_requires_user_prompts(self):
        with patch.object(selection, "load_model") as load, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                selection.main([])
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()

    def test_selection_named_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "named.json"
            with patch.object(selection, "load_model", return_value=(self.tokenizer, self.model)), redirect_stdout(io.StringIO()):
                selection.main([PROMPT, "--json", str(path)])
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["prompt"], PROMPT)
            self.assertEqual(report["decision"], self.report["decision"])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_selection_named_report_rejects_multiple_prompts(self):
        with patch.object(selection, "load_model") as load, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                selection.main([PROMPT, "another prompt", "--json", "report.json"])
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()

    def test_selection_runs_only_supplied_prompts_and_options(self):
        policy = copy.deepcopy(self.policy)
        policy["groups"] = [copy.deepcopy(self.food_group())]
        policy["groups"][0]["id"] = "custom-food"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            policy_path = output / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            args = [PROMPT, "", "--policy", str(policy_path), "--output-dir", str(output),
                    "--model-dir", "existing-checkpoint", "--min-group-ratio", "99",
                    "--top-k", "2"]
            output_text = io.StringIO()
            with patch.object(selection, "load_model", return_value=(self.tokenizer, self.model)) as load, redirect_stdout(output_text):
                selection.main(args)
            load.assert_called_once_with(Path("existing-checkpoint"))
            paths = sorted(output.glob("prompt-*.json"))
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            self.assertEqual([r["prompt"] for r in reports], [PROMPT, ""])
            self.assertEqual({g["id"] for g in reports[0]["decision"]["groups"]}, {"custom-food"})
            self.assertFalse(reports[0]["decision"]["changed"])
            self.assertEqual(reports[0]["decision"]["attribution"]["original_source"]["text"], " my")
            self.assertEqual(reports[0]["reproduction"]["policy"]["thresholds"]["min_group_mass_ratio"], 99)
            self.assertTrue(all(len(r["baseline"]["top"]) == 2 for r in reports))
            self.assertEqual(reports[1]["decision"]["status"], "unsupported_boundary")
            self.assertEqual(output_text.getvalue().count("Top 2 next-token candidates"), 2)
            self.assertNotIn("Top 8 next-token candidates", output_text.getvalue())

    def test_selection_raw_mode_accepts_user_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            output_text = io.StringIO()
            with patch.object(selection, "load_model", return_value=(self.tokenizer, self.model)), redirect_stdout(output_text):
                selection.main([PROMPT, "--mode", "raw", "--output-dir", directory])
            report = json.loads((Path(directory) / "prompt-1.json").read_text(encoding="utf-8"))
            self.assertEqual(report["decision"]["status"], "raw")
            self.assertEqual(report["decision"]["selected"], report["decision"]["original"])
            self.assertIn("next-token candidates", output_text.getvalue())
            self.assertIn("No selection steps.", output_text.getvalue())
            self.assertIn("No groups evaluated.", output_text.getvalue())

    def test_selection_prints_candidates_and_decision_details(self):
        output_text = io.StringIO()
        with redirect_stdout(output_text):
            selection.print_report(self.report)
        rendered = output_text.getvalue()
        candidate_table = rendered.split("next-token candidates (original probabilities):\n", 1)[1].split("\n  Listed probability mass:", 1)[0]
        rows = candidate_table.splitlines()[1:]
        expected = self.report["baseline"]["top"]
        self.assertEqual(len(rows), len(expected))
        for rank, (line, token) in enumerate(zip(rows, expected), 1):
            actual_rank, token_id, percent, text = line.split(maxsplit=3)
            self.assertEqual(int(actual_rank), rank)
            self.assertEqual(int(token_id), token["token_id"])
            self.assertEqual(json.loads(text), token["text"])
            self.assertAlmostEqual(float(percent.rstrip("%")) / 100, token["probability"], places=8)
        self.assertIn("Input token attention", rendered)
        self.assertIn("Selection trace:", rendered)
        self.assertIn("Estimated source / stopping point: input[5]", rendered)
        self.assertIn("Best candidate relative deficit:", rendered)
        self.assertIn("Semantic groups:", rendered)
        self.assertIn("Supported candidates:", rendered)
        self.assertTrue(rendered.isascii())


    def test_english_help(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as error:
            selection.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("One or more prompts", output.getvalue())
        self.assertNotIn("--intent", output.getvalue())
        self.assertIn("10%", output.getvalue())

    def test_top_k_display_does_not_limit_group_probability(self):
        report = select_next_token(PROMPT, self.tokenizer, self.model, top_k=1)
        self.assertEqual(len(report["baseline"]["top"]), 1)
        self.assertEqual(report["decision"]["selected"], self.report["decision"]["selected"])

    def test_invalid_thresholds_and_context_length(self):
        with self.assertRaises(ValueError):
            encode_prompt(" hello" * (self.model.config.max_position_embeddings + 1), self.tokenizer, self.model)
        policy = copy.deepcopy(self.policy)
        policy["thresholds"]["max_relative_gap"] = float("nan")
        with self.assertRaises(ValueError):
            validate_policy(policy)


class PortabilityTest(unittest.TestCase):
    def test_project_settings_are_independent_of_parent(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            checkout = parent / "repo"
            checkout.mkdir()
            for name in ("semantic_category_selection.py", "next-token-settings.json"):
                shutil.copyfile(source / name, checkout / name)
            command = [sys.executable, "-B", "-c",
                       "import json; import semantic_category_selection as selection; print(json.dumps([str(selection.DEFAULT_MODEL_DIR),str(selection.OUTPUT_DIR),selection.DEFAULT_TOP_K]))"]
            actual = json.loads(subprocess.check_output(command, cwd=checkout, text=True))
            self.assertEqual(actual, [str(checkout / "models/gpt2"), str(checkout / "outputs"), 8])
            (parent / "next-token-settings.json").write_text(json.dumps(
                {"model_dir": "existing-model", "output_dir": "old-output", "top_k": 3}))
            actual = json.loads(subprocess.check_output(command, cwd=checkout, text=True))
            self.assertEqual(actual, [str(checkout / "models/gpt2"), str(checkout / "outputs"), 8])


    def test_source_only_checkout_runs_with_an_existing_model(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkout = parent / "standalone"
            checkout.mkdir()
            for name in ("semantic_category_selection.py", "next-token-settings.json", "selection-policy.json", "requirements.txt"):
                shutil.copyfile(source / name, checkout / name)
            # An unrelated parent config must not affect this checkout.
            (parent / "next-token-settings.json").write_text("{}", encoding="utf-8")
            report_path = parent / "report.json"
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            result = subprocess.run(
                [sys.executable, "-B", str(checkout / "semantic_category_selection.py"), PROMPT,
                 "--model-dir", str(DEFAULT_MODEL_DIR), "--json", str(report_path)],
                cwd=parent, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prompt"], PROMPT)
            self.assertEqual(report["decision"]["original"]["text"], " face")
            self.assertEqual(report["decision"]["selected"]["text"], " salad")
            self.assertEqual(len(report["baseline"]["top"]), 8)
            self.assertIn("Top 8 next-token candidates", result.stdout)
            self.assertFalse((checkout / ".venv").exists())
            self.assertFalse((checkout / "models").exists())


if __name__ == "__main__":
    unittest.main()
