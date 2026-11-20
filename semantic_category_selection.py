#!/usr/bin/env python3
"""Inference-time next-token selection for user-supplied prompts."""
import argparse
import json
import math
import os
import platform
import random
import re
from importlib.metadata import version
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / "next-token-settings.json"
SETTINGS = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
DEFAULT_MODEL_DIR = (SETTINGS_PATH.parent / SETTINGS["model_dir"]).resolve()
DEFAULT_TOP_K = SETTINGS["top_k"]
OUTPUT_DIR = SETTINGS_PATH.parent / SETTINGS["output_dir"]


ATTRIBUTION_BASELINES = ("zero", "mean", "space")
ATTRIBUTION_TOLERANCE = 1e-6


DEFAULT_POLICY_PATH = Path(__file__).with_name("selection-policy.json")
REASONS = {
    "raw": "Raw decoding requested; postprocessing bypassed.",
    "unsupported_boundary": "Input is outside the supported continuation boundary; keeping the original choice.",
    "no_association": "No category members have consistent positive attribution to this input token.",
    "accepted_original": "Reached the original winner's attributed source; keeping the original choice without evaluating that source or lower-attention tokens.",
    "uncertain_source": "The original winner has no consistent, unique positive source across ablation baselines; keeping it.",
    "original_candidate": "This category's best candidate is the original winner; continue to the next higher-attention input token.",
    "wide_gap": "The category's best candidate is too far behind the original choice.",
    "small_group_mass": "The category's total probability is below the required ratio.",
    "changed": "Selected the best candidate from the first category that passed all gates.",
}


def token_record(token_id, probs, tokenizer, logits=None):
    token_id = int(token_id)
    row = {"token_id": token_id, "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
           "probability": probs[token_id].item()}
    if logits is not None:
        row["logit"] = logits[token_id].item()
    return row


def load_policy(path=DEFAULT_POLICY_PATH):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_policy(policy)


def validate_policy(policy):
    if policy.get("version") != 3:
        raise ValueError("Use selection-policy version 3: categories and members only; input associations are computed by attribution.")
    limits = policy["thresholds"]
    for key in ["max_relative_gap"]:
        value = limits[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    ratio = limits["min_group_mass_ratio"]
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("min_group_mass_ratio must be greater than 1.")
    count = policy["diagnostics"]["std_top_k"]
    if type(count) is not int or count < 2:
        raise ValueError("std_top_k must be an integer of at least 2.")
    groups = policy["groups"]
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Group IDs must be unique.")
    for group in groups:
        if any(field in group for field in ("anchors", "context_patterns", "association_basis")):
            raise ValueError("Remove manual input associations; version 3 computes them from the model.")
        if not group["members"] or any(not isinstance(word, str) or not word.strip() for word in group["members"]):
            raise ValueError(f"{group['id']}.members must contain nonempty strings.")
        if any(len(word.split()) != 1 for word in group["members"]):
            raise ValueError("Members of this single-token policy must be individual words, not phrases.")
    return policy


def attention_summary(inputs, spans, attentions, tokenizer):
    import torch
    weights = torch.stack([a[0, :, -1, :] for a in attentions]).double().mean((0, 1)).tolist()
    ids = inputs["input_ids"][0].tolist()
    tokens = [{"index": i, "token_id": tid, "raw_token": tokenizer.convert_ids_to_tokens(tid),
               "text": tokenizer.decode([tid]), "span": span, "attention": weight}
              for i, (tid, span, weight) in enumerate(zip(ids, spans, weights))]
    return {"aggregation": "Use the final input position as query and average equally over all layers and heads; preserve individual tokens. These scores are not causal contributions.",
            "tokens": sorted(tokens, key=lambda x: (-x["attention"], x["index"]))}


def category_members(group, probs, tokenizer):
    members, skipped, seen = [], [], set()
    for word in group["members"]:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1 or ids[0] in tokenizer.all_special_ids:
            skipped.append(word)
        elif ids[0] not in seen:
            seen.add(ids[0])
            members.append(token_record(ids[0], probs, tokenizer))
    return sorted(members, key=lambda row: (-row["probability"], row["token_id"])), skipped


def _original_source(rows):
    winners = []
    for baseline_index in range(len(ATTRIBUTION_BASELINES)):
        largest = max(row["original_drops"][baseline_index] for row in rows)
        tied = [row["source"]["index"] for row in rows
                if abs(row["original_drops"][baseline_index] - largest) <= ATTRIBUTION_TOLERANCE]
        winners.append(tied[0] if largest > ATTRIBUTION_TOLERANCE and len(tied) == 1 else None)
    if winners[0] is not None and len(set(winners)) == 1:
        return next(row["source"] for row in rows if row["source"]["index"] == winners[0]), winners
    return None, winners


def attribute_candidates(inputs, logits, attention, tokenizer, model, policy):
    """Estimate input support with three embedding ablations; no unique causal claim."""
    import torch
    if model.training:
        raise ValueError("Call model.eval() before attribution.")
    validate_policy(policy)
    original_id = int(logits.argmax())
    probs = probabilities(logits)
    target_ids = {original_id}
    for group in policy["groups"]:
        members, _ = category_members(group, probs, tokenizer)
        target_ids.update(row["token_id"] for row in members)
    target_ids = sorted(target_ids)
    original_column = target_ids.index(original_id)
    with torch.inference_mode():
        layer = model.get_input_embeddings()
        embeddings = layer(inputs["input_ids"])
        references = {"zero": torch.zeros_like(embeddings[0, 0]), "mean": layer.weight.mean(0),
                      "space": layer.weight[tokenizer.encode(" ", add_special_tokens=False)[0]]}
        original_logp = logits.double().log_softmax(-1)[target_ids]
        rows = []
        for source in sorted(attention["tokens"], key=lambda row: row["index"]):
            drops = []
            for baseline in ATTRIBUTION_BASELINES:
                perturbed = embeddings.clone()
                perturbed[0, source["index"]] = references[baseline]
                changed = model(inputs_embeds=perturbed, attention_mask=inputs["attention_mask"],
                                use_cache=False).logits[0, -1].double().log_softmax(-1)
                drops.append(original_logp - changed[target_ids])
            scores = torch.stack(drops, dim=1)
            rows.append({"source": source, "original_drops": scores[original_column].tolist(),
                         "candidate_drops": [{"token_id": token_id, "log_probability_drops": values.tolist()}
                                             for token_id, values in zip(target_ids, scores)],
                         "supported_token_ids": [token_id for token_id, values in zip(target_ids, scores)
                                                 if bool((values > ATTRIBUTION_TOLERANCE).all())]})
    original_source, winners = _original_source(rows)
    return {"method": "single_token_embedding_ablation", "baselines": list(ATTRIBUTION_BASELINES),
            "interpretation": "Positive log-probability drops indicate support relative to the chosen replacement. This estimates input influence, not a unique causal origin or a semantic category.",
            "original_source": original_source, "baseline_source_indices": winners,
            "additional_forward_passes": len(rows) * len(ATTRIBUTION_BASELINES), "tokens": rows}


def _group_evidence(group, members, skipped, source, original, std):
    best = members[0] if members else None
    mass = sum(row["probability"] for row in members)
    gap = original["probability"] - best["probability"] if best else original["probability"]
    return {"id": group["id"], "label": group["label"], "source": source, "members": members,
            "skipped_multitoken_members": skipped, "mass": mass, "mass_ratio": mass / original["probability"],
            "mass_scope": "Only distinct configured single-token members with positive support from this input token under every baseline are counted.",
            "best": best, "relative_gap": gap / original["probability"], "absolute_gap": gap,
            "log_probability_gap": math.log(original["probability"] / best["probability"]) if best and best["probability"] > 0 else None,
            "gap_over_top_k_std": gap / std if std > 0 else None}


def supported_boundary(text):
    return bool(text and re.search(r"[A-Za-z]\Z", text))


def _at_least(value, threshold):
    return value >= threshold or math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def choose_token(text, probs, tokenizer, attention, policy, mode="auto", attribution=None):
    validate_policy(policy)
    if mode not in {"auto", "raw"}:
        raise ValueError("mode must be auto or raw.")
    original_id = int(probs.argmax())
    original = token_record(original_id, probs, tokenizer)
    result = {"original": original, "selected": original, "changed": False,
              "model_probabilities_unchanged": True, "groups": [], "trace": [],
              "thresholds": policy["thresholds"], "mode": mode,
              "attribution": attribution,
              "semantic_source": "Configured semantic categories and candidate members; input-to-candidate support is estimated by model ablation.",
              "algorithm": "Attribute the original winner to a consistent positive source; visit only inputs with strictly greater attention; retain positively supported category members, choose the largest category with candidate-probability and random tie breakers, and accept the first alternative passing the probability gates. Never evaluate the stopping source."}
    def finish(status):
        result.update(status=status, reason=REASONS[status])
        return result
    if mode == "raw":
        return finish("raw")
    if not supported_boundary(text):
        return finish("unsupported_boundary")
    if attribution is None:
        raise ValueError("Automatic selection requires attribution; call select_next_token or supply attribute_candidates output.")
    stop_source = attribution["original_source"]
    thresholds = policy["thresholds"]
    k = min(policy["diagnostics"]["std_top_k"], probs.numel())
    std = probs.topk(k).values.double().std(unbiased=False).item()
    full_std = probs.double().std(unbiased=False).item()
    result["dispersion_diagnostics"] = {"top_k": k, "top_k_probability_std": std,
        "vocabulary_probability_std": full_std, "used_for_decision": False,
        "meaning": "Population standard deviation across token probabilities; not prediction error, a confidence interval, or statistical significance."}
    if stop_source is None:
        return finish("uncertain_source")
    categories = [(group, *category_members(group, probs, tokenizer)) for group in policy["groups"]]
    supported = {row["source"]["index"]: set(row["supported_token_ids"]) for row in attribution["tokens"]}
    sources = sorted((token for token in attention["tokens"] if token["attention"] > stop_source["attention"]),
                     key=lambda token: (-token["attention"], token["index"]))
    for rank, source in enumerate(sources, 1):
        groups = [_group_evidence(group, [m for m in members if m["token_id"] in supported[source["index"]]],
                                  skipped, source, original, std) for group, members, skipped in categories]
        result["groups"].extend(groups)
        active = sorted((group for group in groups if group["members"]), key=lambda group: (-group["mass"], group["id"]))
        step = {"rank": rank, "source": source, "associated_groups": [group["id"] for group in active],
                "primary_group": None, "checks": {}}
        if not active:
            status = "no_association"
        else:
            largest = [g for g in active if math.isclose(g["mass"], active[0]["mass"], rel_tol=1e-12, abs_tol=1e-15)]
            if len(largest) > 1:
                best_probability = max(g["best"]["probability"] if g["best"] else -1 for g in largest)
                closest = [g for g in largest if math.isclose(g["best"]["probability"] if g["best"] else -1,
                           best_probability, rel_tol=1e-12, abs_tol=1e-15)]
                group = random.choice(closest) if len(closest) > 1 else closest[0]
                step["tied_groups"] = [g["id"] for g in largest]
                step["tie_breaker"] = "random" if len(closest) > 1 else "best_candidate"
            else:
                group = largest[0]
            step["primary_group"] = group["id"]
            step["group_mass_ratio"] = group["mass_ratio"]
            step["relative_gap"] = group["relative_gap"]
            if group["best"] is not None and group["best"]["token_id"] == original_id:
                step.update(status="original_candidate", reason=REASONS["original_candidate"])
                result["trace"].append(step)
                continue
            step["checks"] = {
                "group_mass": _at_least(group["mass_ratio"], thresholds["min_group_mass_ratio"]),
                "candidate_close": group["best"] is not None and _at_least(thresholds["max_relative_gap"], group["relative_gap"]),
            }
            if not step["checks"]["group_mass"]:
                status = "small_group_mass"
            elif not step["checks"]["candidate_close"]:
                status = "wide_gap"
            else:
                result.update(selected=group["best"], selected_group=group["id"], selected_source=source,
                              changed=True)
                status = "changed"
        step.update(status=status, reason=REASONS[status])
        result["trace"].append(step)
        if status == "changed":
            return finish(status)
    return finish("accepted_original")


def load_model(model_dir):
    # Keep cache metadata in this project unless HF_HOME is already configured.
    os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".cache" / "huggingface"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Run the small model on CPU with a limited number of threads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def encode_prompt(text, tokenizer, model, offsets=False):
    import torch
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                        return_offsets_mapping=offsets, verbose=False)
    spans = encoded.pop("offset_mapping", None)
    if encoded["input_ids"].shape[1] == 0:
        encoded = {"input_ids": torch.tensor([[tokenizer.bos_token_id]]),
                   "attention_mask": torch.ones((1, 1), dtype=torch.long)}
        spans = torch.tensor([[[0, 0]]]) if offsets else None
    limit = model.config.max_position_embeddings
    if encoded["input_ids"].shape[1] > limit:
        raise ValueError(f"Input exceeds the model limit of {limit} tokens; it will not be silently truncated.")
    return encoded, spans[0].tolist() if spans is not None else None


def probabilities(logits):
    import torch
    return torch.softmax(logits.double(), dim=-1)


def distribution_report(logits, tokenizer, watch_ids=(), top_k=DEFAULT_TOP_K):
    probs = probabilities(logits)
    chosen = int(probs.argmax())
    top = probs.topk(min(top_k, probs.numel())).indices.tolist()
    return {"chosen": token_record(chosen, probs, tokenizer, logits),
            "top": [token_record(i, probs, tokenizer, logits) for i in top],
            "watched": [dict(token_record(i, probs, tokenizer, logits),
                             rank=int((probs > probs[i]).sum()) + 1) for i in watch_ids]}


def finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Values must be finite, not NaN or infinity.")
    return value


def write_json(path, report):
    path = Path(path)
    if path.parent == Path("."):
        path = OUTPUT_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path.resolve()


def describe_token(record):
    return f'{json.dumps(record["text"], ensure_ascii=False)} {record["probability"]:.6%}'


def select_next_token(text, tokenizer, model, policy=None, mode="auto", top_k=DEFAULT_TOP_K):
    import torch
    if model.training:
        raise ValueError("Call model.eval() before next-token selection.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    policy = load_policy() if policy is None else validate_policy(policy)
    if mode not in {"auto", "raw"}:
        raise ValueError("mode must be auto or raw.")
    inputs, spans = encode_prompt(text, tokenizer, model, offsets=True)
    with torch.inference_mode():
        out = model(**inputs, output_attentions=True, use_cache=False)
        logits = out.logits[0, -1]
        attention = attention_summary(inputs, spans, out.attentions, tokenizer)
        attribution = attribute_candidates(inputs, logits, attention, tokenizer, model, policy) if mode == "auto" and supported_boundary(text) else None
        decision = choose_token(text, probabilities(logits), tokenizer, attention, policy, mode=mode, attribution=attribution)
        baseline = distribution_report(logits, tokenizer, top_k=top_k)
    return {"prompt": text, "decision": decision, "attention": attention, "baseline": baseline,
            "policy_version": policy["version"], "model_type": model.config.model_type}


STATUS_DESCRIPTIONS = REASONS


ENGLISH_TEXT = {'description': 'Semantic category-guided next-token selection for user-supplied prompts.',
 'prompt_help': 'One or more prompts; quote each prompt separately',
 'ratio_help': 'Minimum category mass divided by the original top-token probability',
 'gap_help': 'Maximum relative candidate deficit, e.g. 0.10 for 10%%',
 'std_help': 'Top K probabilities used for standard-deviation diagnostics; does not affect selection',
 'json_help': 'Save one prompt to a named JSON report instead of numbered reports',
 'top_k_error': '--top-k must be greater than zero',
 'json_error': '--json requires exactly one prompt; use --output-dir for multiple prompts',
 'report_path': '\nReport: {path}',
 'error': 'Error: {error}\n',
 'prompt': '\nPrompt: {prompt}',
 'original': '  Original: {token}',
 'selected': '  Selected: {token}',
 'decision': '  Decision: {reason}',
 'top': '\nTop {count} next-token candidates (original probabilities):',
 'candidate_header': 'Rank  Token ID    Probability  Token',
 'listed_mass': '  Listed probability mass: {mass:.6%}',
 'attention_title': '\nInput token attention (last input position, averaged over all layers and heads):',
 'attention_header': 'Rank  Input index  Token ID   Attention  Token',
 'attribution_title': '\nAttribution for the original winner (log-probability drops; columns: {baselines}):',
 'attribution_row': '  input[{index}] {token}: {drops}',
 'attribution_source': '  Estimated source / stopping point: input[{index}] {token}; attention={attention:.6%}. Only strictly higher-attention inputs are evaluated.',
 'attribution_uncertain': '  No consistent unique source across baselines; keeping the original winner.',
 'trace_title': '\nSelection trace:',
 'no_steps': '  No selection steps.',
 'step': '  {rank}. input[{index}] {token}; attention={attention:.6%}; primary group={group}',
 'tie_best_candidate': '     Categories tied for maximum total probability: {groups}; selected the one with the highest-probability candidate.',
 'tie_random': '     Categories tied for maximum total probability: {groups}; compared their best candidates, then broke the remaining tie at random.',
 'none': 'none',
 'group_mass': '     Group mass / original: {ratio:.3f}x (required >= {threshold:g}x)',
 'gap': '     Best candidate relative deficit: {gap:.3%} (allowed <= {threshold:.3%})',
 'groups_title': '\nSemantic groups:',
 'no_groups': '  No groups evaluated.',
 'group': '  Group {id!r}, input[{index}] {token}: total probability={mass:.6%}',
 'group_members': '    Supported candidates: {members}',
 'group_stats': '    Mass / original={ratio:.3f}x; best relative deficit={gap:.3%}',
 'best': '    Best candidate: {token}',
 'skipped': '    Skipped multi-token members: {members}'}


def add_selection_arguments(parser, text=ENGLISH_TEXT):
    """Options shared by the English and Chinese entry points."""
    parser.add_argument("--mode", choices=["auto", "raw"], default="auto")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--min-group-ratio", type=finite_float, help=text["ratio_help"])
    parser.add_argument("--max-relative-gap", type=finite_float, help=text["gap_help"])
    parser.add_argument("--std-top-k", type=int, help=text["std_help"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)


def load_policy_from_args(args):
    policy = load_policy(args.policy)
    for key, value in [("min_group_mass_ratio", args.min_group_ratio),
                       ("max_relative_gap", args.max_relative_gap)]:
        if value is not None:
            policy["thresholds"][key] = value
    if args.std_top_k is not None:
        policy["diagnostics"]["std_top_k"] = args.std_top_k
    return validate_policy(policy)


def print_report(report, *, text=ENGLISH_TEXT, statuses=STATUS_DESCRIPTIONS):
    """Render the same report with the selected entry point's display text."""
    def emit(key, **values):
        print(text[key].format(**values))

    decision = report["decision"]
    emit("prompt", prompt=json.dumps(report["prompt"], ensure_ascii=False))
    emit("original", token=describe_token(decision["original"]))
    emit("selected", token=describe_token(decision["selected"]))
    emit("decision", reason=statuses.get(decision["status"], decision["status"]))

    candidates = report["baseline"]["top"]
    emit("top", count=len(candidates))
    emit("candidate_header")
    for rank, row in enumerate(candidates, 1):
        print(f'{rank:>4}  {row["token_id"]:>8}  {row["probability"]:>13.6%}  {json.dumps(row["text"], ensure_ascii=False)}')
    emit("listed_mass", mass=sum(row["probability"] for row in candidates))

    emit("attention_title")
    emit("attention_header")
    for rank, token in enumerate(report["attention"]["tokens"], 1):
        print(f'{rank:>4}  {token["index"]:>11}  {token["token_id"]:>8}  {token["attention"]:>10.6%}  {json.dumps(token["text"], ensure_ascii=False)}')

    attribution = decision["attribution"]
    if attribution is not None:
        emit("attribution_title", baselines=", ".join(attribution["baselines"]))
        for row in attribution["tokens"]:
            emit("attribution_row", index=row["source"]["index"], token=json.dumps(row["source"]["text"], ensure_ascii=False),
                 drops=", ".join(f"{value:+.6f}" for value in row["original_drops"]))
        stop = attribution["original_source"]
        if stop is None:
            emit("attribution_uncertain")
        else:
            emit("attribution_source", index=stop["index"], token=json.dumps(stop["text"], ensure_ascii=False), attention=stop["attention"])

    thresholds = decision["thresholds"]
    emit("trace_title")
    if not decision["trace"]:
        emit("no_steps")
    for step in decision["trace"]:
        source = step["source"]
        emit("step", rank=step["rank"], index=source["index"], token=json.dumps(source["text"], ensure_ascii=False),
             attention=source["attention"], group=step["primary_group"] or text["none"])
        if step.get("tied_groups"):
            emit("tie_" + step["tie_breaker"], groups=", ".join(step["tied_groups"]))
        if step["checks"]:
            emit("group_mass", ratio=step["group_mass_ratio"], threshold=thresholds["min_group_mass_ratio"])
            emit("gap", gap=step["relative_gap"], threshold=thresholds["max_relative_gap"])
        print("    ", statuses.get(step["status"], step["status"]))

    emit("groups_title")
    if not decision["groups"]:
        emit("no_groups")
    for group in decision["groups"]:
        emit("group", id=group["id"], index=group["source"]["index"], token=json.dumps(group["source"]["text"], ensure_ascii=False), mass=group["mass"])
        emit("group_members", members=json.dumps([row["text"] for row in group["members"]], ensure_ascii=False))
        emit("group_stats", ratio=group["mass_ratio"], gap=group["relative_gap"])
        if group["best"] is not None:
            emit("best", token=describe_token(group["best"]))
        if group["skipped_multitoken_members"]:
            emit("skipped", members=json.dumps(group["skipped_multitoken_members"], ensure_ascii=False))


def main(argv=None, *, text=ENGLISH_TEXT, statuses=STATUS_DESCRIPTIONS):
    parser = argparse.ArgumentParser(description=text["description"])
    parser.add_argument("prompts", nargs="+", metavar="PROMPT", help=text["prompt_help"])
    add_selection_arguments(parser, text)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    output.add_argument("--json", type=Path, help=text["json_help"])
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error(text["top_k_error"])
    if args.json is not None and len(args.prompts) != 1:
        parser.error(text["json_error"])
    try:
        policy = load_policy_from_args(args)
        tokenizer, model = load_model(args.model_dir)
        if args.json is None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"python": platform.python_version(), "torch": version("torch"),
                    "transformers": version("transformers"), "device": "cpu",
                    "attention_implementation": "eager", "policy": policy}
        for index, prompt in enumerate(args.prompts, 1):
            report = select_next_token(prompt, tokenizer, model, policy=policy,
                                       mode=args.mode, top_k=args.top_k)
            report["reproduction"] = metadata
            if args.json is not None:
                path = write_json(args.json, report)
            else:
                path = args.output_dir / f"prompt-{index}.json"
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print_report(report, text=text, statuses=statuses)
            print(text["report_path"].format(path=path.resolve()))
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, text["error"].format(error=exc))


if __name__ == "__main__":
    main()
