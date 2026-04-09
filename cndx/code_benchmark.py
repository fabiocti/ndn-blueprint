"""CNDX Code-Native Benchmark Suite — information retrieval through the latent
bottleneck evaluated on Formal Technical Artifacts (Python source code).

The prose-memory probes (benchmark.py) test natural-language knowledge.
This module tests the same capabilities on code:

  1. Code-NIAH   — recover function/class/variable/import needles among distractor code
  2. Code-FACT1  — single-hop binding: function→args, class→method, variable→value
  3. Code-FACT2  — two-hop code chain: A calls B, B uses C
  4. Code-TRACK  — state tracking: variable reassignment / field mutation final state
  5. Code-EXACT  — exact recovery of function signature / return expression
  6. Code-POS    — positional sensitivity across the code window

All tests:
  - Use synthetic Python code as input
  - Compress through encoder → decode from latents → score recovery
  - Include no-compression control
  - Designed for seq_len=128 latent-native encoder-decoder

Usage:
    python -m cndx.code_benchmark --checkpoint_dir <path> [--tests niah,fact1,fact2,track,exact,pos]
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cndx.native_model import CNDXNativeModel, NativeConfig, build_native_model


# ---------------------------------------------------------------------------
# Code templates — synthetic Python building blocks
# ---------------------------------------------------------------------------

FUNC_NAMES = [
    "parse_config", "build_index", "validate_input", "merge_records",
    "compute_hash", "send_request", "load_weights", "update_cache",
    "format_output", "check_status", "decode_payload", "init_session",
    "transform_data", "run_pipeline", "export_results", "handle_error",
    "process_batch", "fetch_metadata", "apply_filter", "register_hook",
]

CLASS_NAMES = [
    "DataLoader", "ConfigParser", "TokenBuffer", "SessionManager",
    "CacheStore", "EventDispatcher", "ModelRegistry", "TaskScheduler",
    "StreamProcessor", "MetricsCollector", "RequestHandler", "StateManager",
]

VAR_NAMES = [
    "max_retries", "batch_size", "learning_rate", "timeout_ms",
    "num_workers", "buffer_capacity", "threshold", "decay_factor",
    "port_number", "queue_depth", "chunk_size", "epsilon",
]

IMPORT_MODULES = [
    ("os", "path"), ("sys", "argv"), ("json", "loads"),
    ("hashlib", "sha256"), ("collections", "defaultdict"),
    ("itertools", "chain"), ("functools", "lru_cache"),
    ("pathlib", "Path"), ("typing", "Optional"),
    ("logging", "getLogger"), ("datetime", "datetime"),
    ("re", "compile"),
]

ARG_TYPES = ["str", "int", "float", "bool", "list", "dict", "Optional[str]", "bytes"]

RETURN_EXPRESSIONS = [
    "result", "data.copy()", "len(items)", "sum(values) / count",
    "output[:max_len]", "{k: v for k, v in pairs}", "[x for x in filtered]",
    "None if failed else response", "hash_value.hexdigest()",
    "sorted(records, key=lambda r: r.score)", "base64.b64encode(payload)",
    "config.get(key, default)", "max(scores)", "tuple(buffer)",
]

FILLER_FUNCTIONS = [
    'def _noop():\n    pass',
    'def _identity(x):\n    return x',
    'def _log(msg):\n    print(f"[LOG] {msg}")',
    'def _clamp(val, lo, hi):\n    return max(lo, min(val, hi))',
    'def _is_valid(x):\n    return x is not None and len(x) > 0',
    'def _flatten(nested):\n    return [item for sub in nested for item in sub]',
    'def _retry(fn, n=3):\n    for _ in range(n):\n        try:\n            return fn()\n        except Exception:\n            continue',
    'def _safe_div(a, b):\n    return a / b if b != 0 else 0.0',
    'def _unique(seq):\n    return list(dict.fromkeys(seq))',
    'def _chunk(lst, n):\n    for i in range(0, len(lst), n):\n        yield lst[i:i+n]',
    'def _merge(a, b):\n    out = a.copy()\n    out.update(b)\n    return out',
    'def _timestamp():\n    import time\n    return int(time.time())',
]

FILLER_LINES = [
    "x = 0",
    "result = []",
    "flag = True",
    "count = len(data)",
    "temp = None",
    "idx = 0",
    "buf = bytearray()",
    "seen = set()",
    "stack = []",
    "total = 0.0",
    "prefix = 'tmp_'",
    "mapping = {}",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_tokens_in_text(tokens: list, text: str) -> float:
    """Fraction of tokens found in text (case-insensitive)."""
    if not tokens:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for t in tokens if t.lower() in text_lower)
    return found / len(tokens)


def _score_exact_substring(needle: str, haystack: str) -> float:
    """Score how much of needle appears as exact substring in haystack."""
    if not needle:
        return 1.0
    needle = needle.strip()
    haystack = haystack.strip()
    if needle in haystack:
        return 1.0
    words = needle.split()
    found = sum(1 for w in words if w in haystack)
    return found / len(words)


def _encode_decode(model, tokenizer, text, device, amp_ctx, seq_len):
    """Encode text through latent bottleneck and decode."""
    ids = tokenizer(text, truncation=True, max_length=seq_len,
                    padding="max_length", return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attn_mask = ids["attention_mask"].to(device)
    with amp_ctx:
        gen = model.generate(input_ids, attn_mask, max_new_tokens=seq_len)
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def _tokenizer_roundtrip(tokenizer, text, seq_len):
    """No-compression control: just tokenize and detokenize."""
    ids = tokenizer(text, truncation=True, max_length=seq_len,
                    padding="max_length", return_tensors="pt")
    return tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)


def _build_code_passage(needle_code: str, fillers: list, position: str) -> str:
    """Insert needle code among filler code at specified position."""
    n_filler = min(len(fillers), 4)
    chosen = random.sample(fillers, n_filler)

    if position == "start":
        parts = [needle_code] + chosen
    elif position == "end":
        parts = chosen + [needle_code]
    else:
        mid = len(chosen) // 2
        parts = chosen[:mid] + [needle_code] + chosen[mid:]

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Test 1: Code-NIAH — recover code needles among distractor code
# ---------------------------------------------------------------------------

def test_code_niah(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    """Needle-in-a-haystack with code: can the model recover a specific
    function definition, class, variable assignment, or import from among
    distractor code?"""
    print("\n══════════════════════════════════════════")
    print("  Code-NIAH: Code Needle Recovery")
    print("══════════════════════════════════════════")
    model.eval()

    needle_types = {
        "function": lambda: _make_func_needle(),
        "variable": lambda: _make_var_needle(),
        "import":   lambda: _make_import_needle(),
        "class":    lambda: _make_class_needle(),
    }

    results_by_type = {}
    for ntype, make_fn in needle_types.items():
        type_results = []
        for position in ["start", "middle", "end"]:
            for trial in range(n_trials):
                needle_code, needle_key, needle_value = make_fn()
                passage = _build_code_passage(needle_code, FILLER_FUNCTIONS, position)

                decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

                key_recall = _score_tokens_in_text(needle_key.split(), decoded)
                value_recall = _score_tokens_in_text(needle_value.split(), decoded)
                exact_hit = 1.0 if needle_key in decoded else 0.0

                trial_result = {
                    "position": position,
                    "key_recall": round(key_recall, 3),
                    "value_recall": round(value_recall, 3),
                    "exact_key_hit": exact_hit,
                }
                if trial < 2:
                    trial_result["needle_code"] = needle_code[:150]
                    trial_result["decoded_preview"] = decoded[:150]
                type_results.append(trial_result)

        avg_key = sum(r["key_recall"] for r in type_results) / len(type_results)
        avg_val = sum(r["value_recall"] for r in type_results) / len(type_results)
        avg_exact = sum(r["exact_key_hit"] for r in type_results) / len(type_results)
        results_by_type[ntype] = {
            "n_trials": len(type_results),
            "avg_key_recall": round(avg_key, 3),
            "avg_value_recall": round(avg_val, 3),
            "exact_key_hit_rate": round(avg_exact, 3),
            "samples": [r for r in type_results if "decoded_preview" in r][:6],
        }
        print(f"  {ntype}: key={avg_key:.1%}  value={avg_val:.1%}  exact_key={avg_exact:.1%}")

    overall_key = sum(r["avg_key_recall"] for r in results_by_type.values()) / len(results_by_type)
    overall_val = sum(r["avg_value_recall"] for r in results_by_type.values()) / len(results_by_type)
    print(f"  OVERALL: key={overall_key:.1%}  value={overall_val:.1%}")

    return {
        "test": "Code-NIAH",
        "results_by_type": results_by_type,
        "overall_key_recall": round(overall_key, 3),
        "overall_value_recall": round(overall_val, 3),
    }


def _make_func_needle():
    name = random.choice(FUNC_NAMES)
    arg1 = random.choice(VAR_NAMES)
    arg2 = random.choice(VAR_NAMES)
    while arg2 == arg1:
        arg2 = random.choice(VAR_NAMES)
    ret = random.choice(RETURN_EXPRESSIONS)
    code = f"def {name}({arg1}, {arg2}):\n    return {ret}"
    return code, name, f"{arg1}, {arg2}"


def _make_var_needle():
    name = random.choice(VAR_NAMES)
    value = str(random.choice([42, 128, 256, 512, 1024, 0.001, 0.01, 0.1, 3.14, 2.718]))
    code = f"{name} = {value}"
    return code, name, value


def _make_import_needle():
    module, symbol = random.choice(IMPORT_MODULES)
    code = f"from {module} import {symbol}"
    return code, symbol, module


def _make_class_needle():
    name = random.choice(CLASS_NAMES)
    method = random.choice(FUNC_NAMES)
    code = f"class {name}:\n    def {method}(self):\n        pass"
    return code, name, method


# ---------------------------------------------------------------------------
# Test 2: Code-FACT1 — single-hop binding
# ---------------------------------------------------------------------------

def test_code_fact1(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Single-hop code binding: does the model preserve which function has
    which arguments, which class has which method, which variable has which value?"""
    print("\n══════════════════════════════════════════")
    print("  Code-FACT1: Single-Hop Code Binding")
    print("══════════════════════════════════════════")
    model.eval()

    binding_types = {
        "func_args": _make_func_binding,
        "class_method": _make_class_binding,
        "var_value": _make_var_binding,
        "import_usage": _make_import_binding,
    }

    results_by_type = {}
    for btype, make_fn in binding_types.items():
        type_results = []
        for trial in range(n_trials):
            passage, key_name, bound_value, distractor_value = make_fn()

            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

            key_present = key_name.lower() in decoded.lower()
            value_present = bound_value.lower() in decoded.lower()
            distractor_present = distractor_value.lower() in decoded.lower()
            binding_correct = key_present and value_present

            trial_result = {
                "key_present": key_present,
                "value_present": value_present,
                "distractor_present": distractor_present,
                "binding_correct": binding_correct,
            }
            if trial < 3:
                trial_result["passage_preview"] = passage[:200]
                trial_result["decoded_preview"] = decoded[:200]
                trial_result["key"] = key_name
                trial_result["value"] = bound_value
                trial_result["distractor"] = distractor_value
            type_results.append(trial_result)

        key_rate = sum(1 for r in type_results if r["key_present"]) / len(type_results)
        value_rate = sum(1 for r in type_results if r["value_present"]) / len(type_results)
        binding_rate = sum(1 for r in type_results if r["binding_correct"]) / len(type_results)
        distractor_rate = sum(1 for r in type_results if r["distractor_present"]) / len(type_results)

        results_by_type[btype] = {
            "n_trials": n_trials,
            "key_rate": round(key_rate, 3),
            "value_rate": round(value_rate, 3),
            "binding_rate": round(binding_rate, 3),
            "distractor_intrusion": round(distractor_rate, 3),
            "samples": [r for r in type_results if "decoded_preview" in r],
        }
        print(f"  {btype}: key={key_rate:.1%}  value={value_rate:.1%}  binding={binding_rate:.1%}  distractor={distractor_rate:.1%}")

    overall_binding = sum(r["binding_rate"] for r in results_by_type.values()) / len(results_by_type)
    print(f"  OVERALL binding: {overall_binding:.1%}")

    return {
        "test": "Code-FACT1",
        "results_by_type": results_by_type,
        "overall_binding_rate": round(overall_binding, 3),
    }


def _make_func_binding():
    f1, f2 = random.sample(FUNC_NAMES, 2)
    a1, a2 = random.sample(VAR_NAMES, 2)
    a3 = random.choice([v for v in VAR_NAMES if v not in (a1, a2)])
    target = f"def {f1}({a1}, {a2}):\n    return {a1} + {a2}"
    distractor = f"def {f2}({a3}):\n    return {a3}"
    fillers = random.sample(FILLER_FUNCTIONS, 2)
    passage = "\n\n".join([distractor] + fillers + [target])
    return passage, f1, a1, a3


def _make_class_binding():
    c1, c2 = random.sample(CLASS_NAMES, 2)
    m1, m2 = random.sample(FUNC_NAMES, 2)
    target = f"class {c1}:\n    def {m1}(self):\n        pass"
    distractor = f"class {c2}:\n    def {m2}(self):\n        pass"
    fillers = random.sample(FILLER_FUNCTIONS, 2)
    passage = "\n\n".join([distractor] + fillers + [target])
    return passage, c1, m1, m2


def _make_var_binding():
    v1, v2 = random.sample(VAR_NAMES, 2)
    val1 = str(random.choice([42, 128, 256, 0.001, 0.01, 3.14]))
    val2 = str(random.choice([99, 512, 1024, 0.05, 0.1, 2.718]))
    while val1 == val2:
        val2 = str(random.choice([99, 512, 1024, 0.05, 0.1, 2.718]))
    fillers = random.sample(FILLER_LINES, 3)
    lines = [f"{v2} = {val2}"] + fillers + [f"{v1} = {val1}"]
    passage = "\n".join(lines)
    return passage, v1, val1, val2


def _make_import_binding():
    (m1, s1), (m2, s2) = random.sample(IMPORT_MODULES, 2)
    target = f"from {m1} import {s1}"
    distractor = f"from {m2} import {s2}"
    fillers = random.sample(FILLER_LINES, 3)
    passage = "\n".join([distractor] + fillers + [target, f"result = {s1}()"])
    return passage, s1, m1, m2


# ---------------------------------------------------------------------------
# Test 3: Code-FACT2 — two-hop code chain
# ---------------------------------------------------------------------------

def test_code_fact2(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Two-hop code chain: A calls B, B uses C. Does the full chain survive?"""
    print("\n══════════════════════════════════════════")
    print("  Code-FACT2: Two-Hop Code Chain")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for trial in range(n_trials):
        func_a, func_b, func_c = random.sample(FUNC_NAMES, 3)
        arg = random.choice(VAR_NAMES)

        code_c = f"def {func_c}(x):\n    return x * 2"
        code_b = f"def {func_b}(val):\n    return {func_c}(val)"
        code_a = f"def {func_a}({arg}):\n    return {func_b}({arg})"

        fillers = random.sample(FILLER_FUNCTIONS, 2)
        passage = "\n\n".join(fillers[:1] + [code_c, code_b] + fillers[1:] + [code_a])

        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        a_present = func_a in decoded
        b_present = func_b in decoded
        c_present = func_c in decoded
        a_calls_b = f"{func_b}(" in decoded and func_a in decoded
        b_calls_c = f"{func_c}(" in decoded and func_b in decoded
        chain_intact = a_calls_b and b_calls_c

        trial_result = {
            "a_present": a_present,
            "b_present": b_present,
            "c_present": c_present,
            "a_calls_b": a_calls_b,
            "b_calls_c": b_calls_c,
            "chain_intact": chain_intact,
        }
        if trial < 3:
            trial_result["passage_preview"] = passage[:250]
            trial_result["decoded_preview"] = decoded[:250]
            trial_result["chain"] = f"{func_a} -> {func_b} -> {func_c}"
        results.append(trial_result)

    a_rate = sum(1 for r in results if r["a_present"]) / len(results)
    b_rate = sum(1 for r in results if r["b_present"]) / len(results)
    c_rate = sum(1 for r in results if r["c_present"]) / len(results)
    ab_rate = sum(1 for r in results if r["a_calls_b"]) / len(results)
    bc_rate = sum(1 for r in results if r["b_calls_c"]) / len(results)
    chain_rate = sum(1 for r in results if r["chain_intact"]) / len(results)

    print(f"  A present: {a_rate:.1%}  B present: {b_rate:.1%}  C present: {c_rate:.1%}")
    print(f"  A->B link: {ab_rate:.1%}  B->C link: {bc_rate:.1%}")
    print(f"  Full chain intact: {chain_rate:.1%}")

    return {
        "test": "Code-FACT2",
        "n_trials": n_trials,
        "a_present": round(a_rate, 3),
        "b_present": round(b_rate, 3),
        "c_present": round(c_rate, 3),
        "a_calls_b": round(ab_rate, 3),
        "b_calls_c": round(bc_rate, 3),
        "chain_intact": round(chain_rate, 3),
        "samples": [r for r in results if "decoded_preview" in r],
    }


# ---------------------------------------------------------------------------
# Test 4: Code-TRACK — state tracking across updates
# ---------------------------------------------------------------------------

def test_code_track(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Track variable state through multiple reassignments. Does the model
    preserve the final value, not an earlier one?"""
    print("\n══════════════════════════════════════════")
    print("  Code-TRACK: Variable State Tracking")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for trial in range(n_trials):
        var_name = random.choice(VAR_NAMES)
        n_updates = random.choice([3, 4, 5])
        values = random.sample(range(1, 500), n_updates)
        final_value = str(values[-1])

        lines = [f"# initialize {var_name}"]
        for i, v in enumerate(values):
            lines.append(f"{var_name} = {v}")
            if i < n_updates - 1:
                filler = random.choice(FILLER_LINES)
                lines.append(filler)
        lines.append(f"print({var_name})")

        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        var_present = var_name in decoded
        final_present = final_value in decoded
        earlier_values = [str(v) for v in values[:-1]]
        earlier_present = any(str(v) in decoded for v in earlier_values)

        trial_result = {
            "var_present": var_present,
            "final_value_present": final_present,
            "earlier_value_present": earlier_present,
            "n_updates": n_updates,
        }
        if trial < 3:
            trial_result["passage_preview"] = passage[:200]
            trial_result["decoded_preview"] = decoded[:200]
            trial_result["var_name"] = var_name
            trial_result["final_value"] = final_value
            trial_result["all_values"] = [str(v) for v in values]
        results.append(trial_result)

    var_rate = sum(1 for r in results if r["var_present"]) / len(results)
    final_rate = sum(1 for r in results if r["final_value_present"]) / len(results)
    earlier_rate = sum(1 for r in results if r["earlier_value_present"]) / len(results)

    print(f"  Variable present: {var_rate:.1%}")
    print(f"  Final value recovered: {final_rate:.1%}")
    print(f"  Earlier value also present: {earlier_rate:.1%}")

    return {
        "test": "Code-TRACK",
        "n_trials": n_trials,
        "n_updates_range": "3-5",
        "var_present_rate": round(var_rate, 3),
        "final_value_rate": round(final_rate, 3),
        "earlier_value_rate": round(earlier_rate, 3),
        "samples": [r for r in results if "decoded_preview" in r],
    }


# ---------------------------------------------------------------------------
# Test 5: Code-EXACT — exact recovery of structured code elements
# ---------------------------------------------------------------------------

def test_code_exact(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Exact recovery: can the model reproduce a function signature, a return
    expression, or a critical line character-for-character?"""
    print("\n══════════════════════════════════════════")
    print("  Code-EXACT: Exact Code Recovery")
    print("══════════════════════════════════════════")
    model.eval()

    element_types = {
        "signature": _make_signature_trial,
        "return_expr": _make_return_trial,
        "assignment": _make_assignment_trial,
    }

    results_by_type = {}
    for etype, make_fn in element_types.items():
        type_results = []
        for trial in range(n_trials):
            passage, exact_target = make_fn()

            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

            exact_match = exact_target in decoded
            token_recall = _score_exact_substring(exact_target, decoded)

            trial_result = {
                "exact_match": exact_match,
                "token_recall": round(token_recall, 3),
            }
            if trial < 3:
                trial_result["target"] = exact_target
                trial_result["passage_preview"] = passage[:200]
                trial_result["decoded_preview"] = decoded[:200]
            type_results.append(trial_result)

        exact_rate = sum(1 for r in type_results if r["exact_match"]) / len(type_results)
        avg_recall = sum(r["token_recall"] for r in type_results) / len(type_results)
        results_by_type[etype] = {
            "n_trials": n_trials,
            "exact_match_rate": round(exact_rate, 3),
            "avg_token_recall": round(avg_recall, 3),
            "samples": [r for r in type_results if "decoded_preview" in r],
        }
        print(f"  {etype}: exact={exact_rate:.1%}  token_recall={avg_recall:.1%}")

    overall_exact = sum(r["exact_match_rate"] for r in results_by_type.values()) / len(results_by_type)
    overall_recall = sum(r["avg_token_recall"] for r in results_by_type.values()) / len(results_by_type)
    print(f"  OVERALL: exact={overall_exact:.1%}  recall={overall_recall:.1%}")

    return {
        "test": "Code-EXACT",
        "results_by_type": results_by_type,
        "overall_exact_rate": round(overall_exact, 3),
        "overall_token_recall": round(overall_recall, 3),
    }


def _make_signature_trial():
    name = random.choice(FUNC_NAMES)
    args = random.sample(VAR_NAMES, random.randint(2, 4))
    sig = f"def {name}({', '.join(args)}):"
    body = f"    return {random.choice(RETURN_EXPRESSIONS)}"
    fillers = random.sample(FILLER_FUNCTIONS, 3)
    passage = "\n\n".join(fillers[:2] + [f"{sig}\n{body}"] + fillers[2:])
    return passage, sig


def _make_return_trial():
    name = random.choice(FUNC_NAMES)
    ret = random.choice(RETURN_EXPRESSIONS)
    target = f"return {ret}"
    func = f"def {name}(data):\n    {target}"
    fillers = random.sample(FILLER_FUNCTIONS, 3)
    passage = "\n\n".join(fillers[:1] + [func] + fillers[1:])
    return passage, target


def _make_assignment_trial():
    var = random.choice(VAR_NAMES)
    val = random.choice([42, 128, 256, 512, 1024, 0.001, 0.01, 0.1, 3.14])
    target = f"{var} = {val}"
    fillers = random.sample(FILLER_LINES, 4) + [random.choice(FILLER_FUNCTIONS)]
    passage = "\n".join(fillers[:2] + [target] + fillers[2:])
    return passage, target


# ---------------------------------------------------------------------------
# Test 6: Code-POS — positional sensitivity in code window
# ---------------------------------------------------------------------------

def test_code_pos(model, tokenizer, device, amp_ctx, cfg, n_trials=20):
    """Does the model preserve code facts equally at start, middle, and end?"""
    print("\n══════════════════════════════════════════")
    print("  Code-POS: Positional Sensitivity")
    print("══════════════════════════════════════════")
    model.eval()

    positions = ["start", "middle", "end"]
    results_by_pos = {}

    for pos in positions:
        pos_results = []
        for trial in range(n_trials):
            needle_code, needle_key, needle_value = _make_func_needle()
            passage = _build_code_passage(needle_code, FILLER_FUNCTIONS, pos)

            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

            key_recall = _score_tokens_in_text(needle_key.split(), decoded)
            value_recall = _score_tokens_in_text(needle_value.split(), decoded)
            exact_hit = 1.0 if needle_key in decoded else 0.0

            pos_results.append({
                "key_recall": round(key_recall, 3),
                "value_recall": round(value_recall, 3),
                "exact_key_hit": exact_hit,
            })

        avg_key = sum(r["key_recall"] for r in pos_results) / len(pos_results)
        avg_val = sum(r["value_recall"] for r in pos_results) / len(pos_results)
        avg_exact = sum(r["exact_key_hit"] for r in pos_results) / len(pos_results)
        results_by_pos[pos] = {
            "n_trials": n_trials,
            "avg_key_recall": round(avg_key, 3),
            "avg_value_recall": round(avg_val, 3),
            "exact_key_hit_rate": round(avg_exact, 3),
        }
        print(f"  {pos}: key={avg_key:.1%}  value={avg_val:.1%}  exact_key={avg_exact:.1%}")

    return {
        "test": "Code-POS",
        "results_by_position": results_by_pos,
    }


# ---------------------------------------------------------------------------
# Control: no-compression baseline
# ---------------------------------------------------------------------------

def test_code_control(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    """No-compression control for code: tokenizer round-trip vs latent bottleneck."""
    print("\n══════════════════════════════════════════")
    print("  Code-CONTROL: No-Compression Baseline")
    print("══════════════════════════════════════════")
    model.eval()

    raw_scores = []
    compressed_scores = []

    for trial in range(n_trials):
        needle_code, needle_key, needle_value = _make_func_needle()
        passage = _build_code_passage(needle_code, FILLER_FUNCTIONS, "middle")

        raw_text = _tokenizer_roundtrip(tokenizer, passage, cfg.seq_len)
        raw_key = _score_tokens_in_text(needle_key.split(), raw_text)
        raw_val = _score_tokens_in_text(needle_value.split(), raw_text)
        raw_scores.append({"key": raw_key, "value": raw_val})

        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        comp_key = _score_tokens_in_text(needle_key.split(), decoded)
        comp_val = _score_tokens_in_text(needle_value.split(), decoded)
        compressed_scores.append({"key": comp_key, "value": comp_val})

    raw_key_avg = sum(s["key"] for s in raw_scores) / n_trials
    raw_val_avg = sum(s["value"] for s in raw_scores) / n_trials
    comp_key_avg = sum(s["key"] for s in compressed_scores) / n_trials
    comp_val_avg = sum(s["value"] for s in compressed_scores) / n_trials

    print(f"  Raw (tokenizer):     key={raw_key_avg:.1%}  value={raw_val_avg:.1%}")
    print(f"  Compressed (latent): key={comp_key_avg:.1%}  value={comp_val_avg:.1%}")
    print(f"  Compression cost:    key={raw_key_avg - comp_key_avg:+.1%}  value={raw_val_avg - comp_val_avg:+.1%}")

    return {
        "test": "Code-CONTROL",
        "n_trials": n_trials,
        "raw_key_recall": round(raw_key_avg, 3),
        "raw_value_recall": round(raw_val_avg, 3),
        "compressed_key_recall": round(comp_key_avg, 3),
        "compressed_value_recall": round(comp_val_avg, 3),
        "compression_cost_key": round(raw_key_avg - comp_key_avg, 3),
        "compression_cost_value": round(raw_val_avg - comp_val_avg, 3),
    }


# ---------------------------------------------------------------------------
# Checkpoint loader (shared with benchmark.py)
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_dir, device="cuda"):
    """Load model + config from a results directory."""
    results_path = Path(ckpt_dir) / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        cfg_dict = results.get("config", {})
    else:
        cfg_dict = {}

    cfg = NativeConfig(**{k: v for k, v in cfg_dict.items()
                          if k in NativeConfig.__dataclass_fields__})

    model, tokenizer = build_native_model(cfg, device=device)

    ckpt_path = Path(ckpt_dir) / "model.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No model.pt in {ckpt_dir}")

    model.eval()
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_code_benchmarks(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    amp_ctx = (torch.autocast("cuda", dtype=dtype) if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))

    print(f"Device: {device}, AMP dtype: {dtype}")

    model, tokenizer, cfg = load_checkpoint(args.checkpoint_dir, device=device)
    print(f"Model: {model.num_params()/1e6:.1f}M params, K={cfg.num_latents}")

    ALL_TESTS = ["control", "niah", "fact1", "fact2", "track", "exact", "pos"]
    tests = args.tests.split(",") if args.tests else ALL_TESTS
    all_results = {
        "checkpoint": args.checkpoint_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_summary": {
            "num_latents": cfg.num_latents,
            "latent_groups": cfg.latent_groups,
            "seq_len": cfg.seq_len,
            "warmup_ratio": cfg.warmup_ratio,
        },
    }

    if "control" in tests:
        all_results["control"] = test_code_control(model, tokenizer, device, amp_ctx, cfg)
    if "niah" in tests:
        all_results["niah"] = test_code_niah(model, tokenizer, device, amp_ctx, cfg)
    if "fact1" in tests:
        all_results["fact1"] = test_code_fact1(model, tokenizer, device, amp_ctx, cfg)
    if "fact2" in tests:
        all_results["fact2"] = test_code_fact2(model, tokenizer, device, amp_ctx, cfg)
    if "track" in tests:
        all_results["track"] = test_code_track(model, tokenizer, device, amp_ctx, cfg)
    if "exact" in tests:
        all_results["exact"] = test_code_exact(model, tokenizer, device, amp_ctx, cfg)
    if "pos" in tests:
        all_results["pos"] = test_code_pos(model, tokenizer, device, amp_ctx, cfg)

    out_path = Path(args.checkpoint_dir) / "code_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n══════════════════════════════════════════")
    print(f"  Code benchmark results saved to {out_path}")
    print(f"══════════════════════════════════════════")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNDX Code-Native Benchmark Suite — "
                    "Formal Technical Artifacts (Python)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model.pt and results.json")
    parser.add_argument("--tests", type=str, default=None,
                        help="Comma-separated: control,niah,fact1,fact2,track,exact,pos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    run_code_benchmarks(args)
