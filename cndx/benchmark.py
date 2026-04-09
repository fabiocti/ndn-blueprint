"""CNDX Benchmark Suite — adapted memory evaluation for latent-native models.

NOT stock RULER/BABILong. These are purpose-built probes that test the same
underlying capability — information retrieval and reasoning — but evaluated
through latent compression rather than autoregressive long-context prompting.

Latent-RULER probes (information retrieval through the bottleneck):
  1. CNDX-NIAH   — single fact at controlled positions
  2. CNDX-MNIH   — multiple facts scattered across passage
  3. CNDX-FTYPE  — fact-type sweep: names, numbers, dates, technical terms
  4. CNDX-POS    — fine-grained positional sensitivity

Latent-BABILong-lite probes (reasoning through the bottleneck):
  5. CNDX-FACT1  — single supporting fact retrieval (entity-property)
  6. CNDX-FACT2  — two-hop fact chaining (entity-action-location)
  7. CNDX-TRACK  — entity state tracking (multiple updates, last state matters)

All tests:
  - Produce scores in [0,1] per trial
  - Include a no-compression control (score the raw input as ceiling)
  - Designed for seq_len=128 latent-native encoder-decoder

Usage:
    python -m cndx.benchmark --checkpoint_dir <path> [--tests niah,multi,ftype,pos,fact1,fact2,track]
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cndx.native_model import CNDXNativeModel, NativeConfig, build_native_model


# ---------------------------------------------------------------------------
# Needle templates
# ---------------------------------------------------------------------------

PERSON_NEEDLES = [
    ("Dr. Elara Montclair", "pioneered the first synthetic photosynthesis reactor in 2031"),
    ("Marcus Jennings", "won the 2019 World Crossword Championship in Tokyo"),
    ("Priya Lakshmi Rao", "discovered the high-conductivity alloy known as ferrocene blue"),
    ("Captain Yusuf al-Mansur", "navigated the first autonomous cargo vessel across the Atlantic"),
    ("Helena Ostrowski", "authored the definitive monograph on Baltic amber inclusions"),
    ("Tomás Esperanza", "designed the vaulted glass canopy of the Santiago transit hub"),
    ("Anika Björkqvist", "identified the high-latitude lichen species Cladonia borealis variant F"),
    ("Li Wei-Chen", "developed the quad-band RF filter used in modern satellite uplinks"),
]

NUMBER_NEEDLES = [
    ("population of Greenvale", "was recorded at 47,832 in the 2024 census"),
    ("depth of Lake Mirova", "reaches a maximum of 1,247 metres near the eastern basin"),
    ("wingspan of the ivory-billed kite", "measures between 162 and 178 centimetres"),
    ("mass of the prototype reactor core", "weighed exactly 3,891 kilograms at commissioning"),
    ("annual rainfall in the Kaza Valley", "averages 2,103 millimetres per year"),
    ("distance from Outpost Seven to the relay", "is precisely 482 kilometres by the northern route"),
]

DATE_NEEDLES = [
    ("the Treaty of Valence", "was ratified on 14 March 1793"),
    ("the Kamino Observatory fire", "occurred on the night of 7 November 1962"),
    ("the first manned descent into the Yulara Trench", "took place on 22 August 2018"),
    ("the founding of the Meridian Postal Service", "dates back to 3 June 1841"),
]

TECHNICAL_NEEDLES = [
    ("ferrocene blue", "exhibits a thermal conductivity of 412 W/mK at room temperature"),
    ("the Zhukovsky-Kline integral", "converges for all alpha in the open interval (0, pi)"),
    ("protocol RFC 9147", "mandates a 256-bit session key derived via HKDF-SHA384"),
    ("the CRISPR-Cas14a variant", "achieves single-nucleotide specificity on ssDNA targets"),
]

FILLER_SENTENCES = [
    "The region is known for its temperate climate and diverse ecosystems.",
    "Local industries include agriculture, forestry, and small-scale manufacturing.",
    "The university was established in the early nineteenth century.",
    "Several rivers converge near the town center, forming a natural floodplain.",
    "Historical records suggest the area was settled during the medieval period.",
    "The municipality maintains a network of public libraries and cultural centers.",
    "Annual festivals attract visitors from neighboring provinces and beyond.",
    "Recent infrastructure investments have improved road and rail connections.",
    "The surrounding landscape features rolling hills and scattered woodlands.",
    "Economic growth has been steady, driven by technology and service sectors.",
    "Archaeological excavations have uncovered artifacts dating to the Bronze Age.",
    "The coastal waters support a productive fishing industry.",
    "Conservation efforts have helped restore native wildlife populations.",
    "A series of reforms modernized the local governance structure in the 1990s.",
    "The observatory sits atop a granite ridge, offering unobstructed sky coverage.",
    "Temperatures rarely exceed thirty-five degrees in summer.",
    "The alloy performs well under cyclic loading conditions.",
    "Field measurements confirm the theoretical predictions within five percent.",
    "The canal system was originally built for grain transport.",
    "Volcanic soil in the lowlands supports intensive rice cultivation.",
]


# ---------------------------------------------------------------------------
# BABILong-lite templates — entity/fact/reasoning patterns
# ---------------------------------------------------------------------------

ENTITIES = ["John", "Mary", "Sandra", "Daniel", "Emily", "Robert",
            "Sarah", "James", "Anna", "Thomas"]

LOCATIONS = ["kitchen", "garden", "office", "bedroom", "hallway",
             "bathroom", "library", "basement", "balcony", "garage"]

OBJECTS = ["football", "apple", "milk", "keys", "notebook",
           "phone", "umbrella", "bottle", "laptop", "glasses"]

BABI_FILLER = [
    "The windows were cleaned last Tuesday.",
    "A large painting hung above the fireplace.",
    "The temperature outside had dropped below freezing.",
    "Several books were arranged neatly on the shelf.",
    "The clock on the wall showed a quarter past three.",
    "A pot of coffee was brewing in the other room.",
    "The carpet had been recently replaced.",
    "An old radio played softly in the background.",
    "The curtains were drawn to block the afternoon sun.",
    "A stack of letters sat unopened on the table.",
    "The dog was sleeping near the front door.",
    "Someone had left the porch light on overnight.",
]


def _build_passage(needle_text: str, filler: list[str], position: str,
                   target_sentences: int = 8) -> str:
    """Insert needle_text into a passage of filler at the given position.

    position: 'start', 'middle', 'end', or a float in [0,1].
    """
    n_filler = target_sentences - 1
    chosen_filler = random.sample(filler, min(n_filler, len(filler)))

    if position == "start":
        idx = 0
    elif position == "end":
        idx = len(chosen_filler)
    elif position == "middle":
        idx = len(chosen_filler) // 2
    else:
        idx = int(float(position) * len(chosen_filler))
        idx = max(0, min(idx, len(chosen_filler)))

    chosen_filler.insert(idx, needle_text)
    return " ".join(chosen_filler)


def _score_tokens_in_text(tokens: list[str], text: str) -> float:
    """Fraction of tokens found in text (case-insensitive)."""
    text_lower = text.lower()
    hits = sum(1 for t in tokens if t.lower() in text_lower)
    return hits / max(len(tokens), 1)


def _check_needle_in_output(needle_entity: str, needle_fact: str,
                            decoded_text: str) -> dict:
    """Score whether needle information survived reconstruction."""
    entity_tokens = needle_entity.lower().split()
    fact_tokens = needle_fact.lower().split()
    decoded_lower = decoded_text.lower()

    entity_recall = _score_tokens_in_text(entity_tokens, decoded_text)
    fact_recall = _score_tokens_in_text(fact_tokens, decoded_text)

    numbers_in_fact = re.findall(r'\d[\d,\.]*', needle_fact)
    numbers_recovered = sum(1 for n in numbers_in_fact
                           if n.replace(",", "") in decoded_text.replace(",", ""))
    number_recall = (numbers_recovered / max(len(numbers_in_fact), 1)
                     if numbers_in_fact else None)

    return {
        "entity_recall": round(entity_recall, 3),
        "fact_recall": round(fact_recall, 3),
        "number_recall": round(number_recall, 3) if number_recall is not None else None,
        "entity_present": entity_recall > 0.5,
        "fact_present": fact_recall > 0.3,
    }


# ---------------------------------------------------------------------------
# Test 1: Single-needle NIAH
# ---------------------------------------------------------------------------

def test_niah(model, tokenizer, device, amp_ctx, cfg,
              n_trials: int = 40, positions=("start", "middle", "end")):
    """Needle-In-A-Haystack: single fact at controlled positions."""
    print("\n══════════════════════════════════════════")
    print("  CNDX-NIAH: Single Needle-In-A-Haystack")
    print("══════════════════════════════════════════")
    model.eval()

    all_needles = PERSON_NEEDLES + NUMBER_NEEDLES + DATE_NEEDLES + TECHNICAL_NEEDLES
    results_by_pos = {}

    for pos in positions:
        pos_results = []
        for trial in range(n_trials):
            entity, fact = random.choice(all_needles)
            needle_sentence = f"{entity} {fact}."
            passage = _build_passage(needle_sentence, FILLER_SENTENCES, pos)

            ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                           padding="max_length", return_tensors="pt")
            input_ids = ids["input_ids"].to(device)
            attn_mask = ids["attention_mask"].to(device)

            with amp_ctx:
                gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)

            score = _check_needle_in_output(entity, fact, decoded)
            score["position"] = pos
            score["needle_entity"] = entity
            score["needle_fact"] = fact
            if trial < 3:
                score["passage_preview"] = passage[:150]
                score["decoded_preview"] = decoded[:150]
            pos_results.append(score)

        avg_entity = sum(r["entity_recall"] for r in pos_results) / len(pos_results)
        avg_fact = sum(r["fact_recall"] for r in pos_results) / len(pos_results)
        num_scores = [r["number_recall"] for r in pos_results if r["number_recall"] is not None]
        avg_number = sum(num_scores) / max(len(num_scores), 1) if num_scores else None

        results_by_pos[pos] = {
            "n_trials": n_trials,
            "avg_entity_recall": round(avg_entity, 3),
            "avg_fact_recall": round(avg_fact, 3),
            "avg_number_recall": round(avg_number, 3) if avg_number is not None else None,
            "entity_hit_rate": round(sum(1 for r in pos_results if r["entity_present"]) / n_trials, 3),
            "fact_hit_rate": round(sum(1 for r in pos_results if r["fact_present"]) / n_trials, 3),
            "samples": pos_results[:3],
        }

        print(f"\n  Position: {pos}")
        print(f"    Entity recall:  {avg_entity:.1%}")
        print(f"    Fact recall:    {avg_fact:.1%}")
        if avg_number is not None:
            print(f"    Number recall:  {avg_number:.1%}")
        print(f"    Entity hit rate: {results_by_pos[pos]['entity_hit_rate']:.1%}")
        print(f"    Fact hit rate:   {results_by_pos[pos]['fact_hit_rate']:.1%}")

    overall_entity = sum(v["avg_entity_recall"] for v in results_by_pos.values()) / len(results_by_pos)
    overall_fact = sum(v["avg_fact_recall"] for v in results_by_pos.values()) / len(results_by_pos)
    print(f"\n  OVERALL: entity_recall={overall_entity:.1%}  fact_recall={overall_fact:.1%}")

    return {
        "test": "CNDX-NIAH",
        "results_by_position": results_by_pos,
        "overall_entity_recall": round(overall_entity, 3),
        "overall_fact_recall": round(overall_fact, 3),
    }


# ---------------------------------------------------------------------------
# Test 2: Multi-needle
# ---------------------------------------------------------------------------

def test_multi_needle(model, tokenizer, device, amp_ctx, cfg,
                      n_trials: int = 30, needle_counts=(1, 2, 3)):
    """Multi-needle: embed N facts in one passage, score how many survive."""
    print("\n══════════════════════════════════════════")
    print("  CNDX-MNIH: Multi-Needle Retrieval")
    print("══════════════════════════════════════════")
    model.eval()

    all_needles = PERSON_NEEDLES + NUMBER_NEEDLES + DATE_NEEDLES + TECHNICAL_NEEDLES
    results_by_count = {}

    for n_needles in needle_counts:
        count_results = []
        for trial in range(n_trials):
            chosen = random.sample(all_needles, min(n_needles, len(all_needles)))
            needle_sentences = [f"{e} {f}." for e, f in chosen]

            n_filler = max(8 - n_needles, 3)
            filler_chosen = random.sample(FILLER_SENTENCES, min(n_filler, len(FILLER_SENTENCES)))

            all_sentences = filler_chosen.copy()
            for i, ns in enumerate(needle_sentences):
                pos = int((i + 1) / (n_needles + 1) * len(all_sentences))
                all_sentences.insert(pos, ns)
            passage = " ".join(all_sentences)

            ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                           padding="max_length", return_tensors="pt")
            input_ids = ids["input_ids"].to(device)
            attn_mask = ids["attention_mask"].to(device)

            with amp_ctx:
                gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)

            needles_found = 0
            per_needle = []
            for entity, fact in chosen:
                score = _check_needle_in_output(entity, fact, decoded)
                per_needle.append(score)
                if score["entity_present"] and score["fact_present"]:
                    needles_found += 1

            count_results.append({
                "n_needles": n_needles,
                "needles_found": needles_found,
                "recall": needles_found / n_needles,
                "per_needle": per_needle if trial < 3 else None,
            })

        avg_recall = sum(r["recall"] for r in count_results) / len(count_results)
        results_by_count[n_needles] = {
            "n_trials": n_trials,
            "avg_recall": round(avg_recall, 3),
            "perfect_rate": round(sum(1 for r in count_results if r["recall"] == 1.0) / n_trials, 3),
            "zero_rate": round(sum(1 for r in count_results if r["recall"] == 0.0) / n_trials, 3),
        }
        print(f"\n  {n_needles} needle(s): avg_recall={avg_recall:.1%}  "
              f"perfect={results_by_count[n_needles]['perfect_rate']:.0%}  "
              f"zero={results_by_count[n_needles]['zero_rate']:.0%}")

    return {
        "test": "CNDX-MNIH",
        "results_by_count": results_by_count,
    }


# ---------------------------------------------------------------------------
# Test 3: Fact-type sweep
# ---------------------------------------------------------------------------

def test_fact_type(model, tokenizer, device, amp_ctx, cfg, n_trials: int = 20):
    """Score needle retrieval broken down by fact type."""
    print("\n══════════════════════════════════════════")
    print("  CNDX-FTYPE: Fact-Type Retrieval Sweep")
    print("══════════════════════════════════════════")
    model.eval()

    type_pools = {
        "person": PERSON_NEEDLES,
        "number": NUMBER_NEEDLES,
        "date": DATE_NEEDLES,
        "technical": TECHNICAL_NEEDLES,
    }
    results_by_type = {}

    for ftype, pool in type_pools.items():
        type_results = []
        for trial in range(n_trials):
            entity, fact = random.choice(pool)
            needle_sentence = f"{entity} {fact}."
            passage = _build_passage(needle_sentence, FILLER_SENTENCES, "middle")

            ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                           padding="max_length", return_tensors="pt")
            input_ids = ids["input_ids"].to(device)
            attn_mask = ids["attention_mask"].to(device)

            with amp_ctx:
                gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)

            score = _check_needle_in_output(entity, fact, decoded)
            type_results.append(score)

        avg_entity = sum(r["entity_recall"] for r in type_results) / len(type_results)
        avg_fact = sum(r["fact_recall"] for r in type_results) / len(type_results)
        results_by_type[ftype] = {
            "n_trials": n_trials,
            "avg_entity_recall": round(avg_entity, 3),
            "avg_fact_recall": round(avg_fact, 3),
            "entity_hit_rate": round(sum(1 for r in type_results if r["entity_present"]) / n_trials, 3),
            "fact_hit_rate": round(sum(1 for r in type_results if r["fact_present"]) / n_trials, 3),
        }
        print(f"  {ftype:>12}: entity={avg_entity:.1%}  fact={avg_fact:.1%}")

    return {
        "test": "CNDX-FTYPE",
        "results_by_type": results_by_type,
    }


# ---------------------------------------------------------------------------
# Test 4: Positional sensitivity (fine-grained)
# ---------------------------------------------------------------------------

def test_positional(model, tokenizer, device, amp_ctx, cfg, n_trials: int = 20):
    """Fine-grained positional sensitivity: needle at 0%, 25%, 50%, 75%, 100%."""
    print("\n══════════════════════════════════════════")
    print("  CNDX-POS: Positional Sensitivity")
    print("══════════════════════════════════════════")
    model.eval()

    all_needles = PERSON_NEEDLES + NUMBER_NEEDLES + DATE_NEEDLES + TECHNICAL_NEEDLES
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    results_by_pos = {}

    for pos in positions:
        pos_results = []
        for trial in range(n_trials):
            entity, fact = random.choice(all_needles)
            needle_sentence = f"{entity} {fact}."
            passage = _build_passage(needle_sentence, FILLER_SENTENCES, str(pos))

            ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                           padding="max_length", return_tensors="pt")
            input_ids = ids["input_ids"].to(device)
            attn_mask = ids["attention_mask"].to(device)

            with amp_ctx:
                gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)

            score = _check_needle_in_output(entity, fact, decoded)
            pos_results.append(score)

        avg_entity = sum(r["entity_recall"] for r in pos_results) / len(pos_results)
        avg_fact = sum(r["fact_recall"] for r in pos_results) / len(pos_results)
        results_by_pos[str(pos)] = {
            "n_trials": n_trials,
            "avg_entity_recall": round(avg_entity, 3),
            "avg_fact_recall": round(avg_fact, 3),
        }
        print(f"  pos={pos:.0%}: entity={avg_entity:.1%}  fact={avg_fact:.1%}")

    return {
        "test": "CNDX-POS",
        "results_by_position": results_by_pos,
    }


# ---------------------------------------------------------------------------
# Test 5: BABILong-lite — Single supporting fact (CNDX-FACT1)
# ---------------------------------------------------------------------------

def _make_fact1_trial():
    """Generate a single-fact retrieval passage.

    Pattern: '<Entity> went to the <location>.' embedded in filler.
    Target: does reconstruction preserve the entity-location binding?
    """
    entity = random.choice(ENTITIES)
    location = random.choice(LOCATIONS)
    fact_sentence = f"{entity} went to the {location}."
    filler = random.sample(BABI_FILLER, min(6, len(BABI_FILLER)))
    pos = random.randint(0, len(filler))
    filler.insert(pos, fact_sentence)
    passage = " ".join(filler)
    return passage, entity, location


def test_fact1(model, tokenizer, device, amp_ctx, cfg, n_trials: int = 60):
    """Single supporting fact: entity went to location. Does the binding survive?"""
    print("\n══════════════════════════════════════════")
    print("  CNDX-FACT1: Single Fact Retrieval")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    control_results = []

    for trial in range(n_trials):
        passage, entity, location = _make_fact1_trial()

        ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                       padding="max_length", return_tensors="pt")
        input_ids = ids["input_ids"].to(device)
        attn_mask = ids["attention_mask"].to(device)

        with amp_ctx:
            gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)

        entity_found = entity.lower() in decoded.lower()
        location_found = location.lower() in decoded.lower()
        binding_pattern = re.search(
            rf'{re.escape(entity)}.*{re.escape(location)}',
            decoded, re.IGNORECASE)
        binding_ok = binding_pattern is not None

        results.append({
            "entity_found": entity_found,
            "location_found": location_found,
            "binding_preserved": binding_ok,
        })

        # No-compression control: score against original passage
        ctrl_binding = re.search(
            rf'{re.escape(entity)}.*{re.escape(location)}',
            passage, re.IGNORECASE)
        control_results.append({"binding_preserved": ctrl_binding is not None})

    entity_rate = sum(r["entity_found"] for r in results) / n_trials
    location_rate = sum(r["location_found"] for r in results) / n_trials
    binding_rate = sum(r["binding_preserved"] for r in results) / n_trials
    control_binding = sum(r["binding_preserved"] for r in control_results) / n_trials

    print(f"  Entity found:      {entity_rate:.1%}")
    print(f"  Location found:    {location_rate:.1%}")
    print(f"  Binding preserved: {binding_rate:.1%}")
    print(f"  Control (no-comp): {control_binding:.1%}")
    print(f"  Compression cost:  {control_binding - binding_rate:+.1%}")

    return {
        "test": "CNDX-FACT1",
        "n_trials": n_trials,
        "entity_rate": round(entity_rate, 3),
        "location_rate": round(location_rate, 3),
        "binding_rate": round(binding_rate, 3),
        "control_binding_rate": round(control_binding, 3),
        "compression_cost": round(control_binding - binding_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 6: BABILong-lite — Two-hop fact chaining (CNDX-FACT2)
# ---------------------------------------------------------------------------

def _make_fact2_trial():
    """Generate a two-hop fact-chaining passage.

    Pattern: '<Entity> picked up the <object>. <Entity> went to the <location>.'
    Target: does reconstruction preserve all three bindings (entity+object+location)?
    """
    entity = random.choice(ENTITIES)
    obj = random.choice(OBJECTS)
    location = random.choice(LOCATIONS)
    fact1 = f"{entity} picked up the {obj}."
    fact2 = f"{entity} went to the {location}."

    filler = random.sample(BABI_FILLER, min(5, len(BABI_FILLER)))
    pos1 = random.randint(0, max(len(filler) // 2, 1))
    filler.insert(pos1, fact1)
    pos2 = random.randint(pos1 + 1, len(filler))
    filler.insert(pos2, fact2)
    passage = " ".join(filler)
    return passage, entity, obj, location


def test_fact2(model, tokenizer, device, amp_ctx, cfg, n_trials: int = 60):
    """Two-hop fact chaining: entity+object+location. All three must survive."""
    print("\n══════════════════════════════════════════")
    print("  CNDX-FACT2: Two-Hop Fact Chaining")
    print("══════════════════════════════════════════")
    model.eval()

    results = []

    for trial in range(n_trials):
        passage, entity, obj, location = _make_fact2_trial()

        ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                       padding="max_length", return_tensors="pt")
        input_ids = ids["input_ids"].to(device)
        attn_mask = ids["attention_mask"].to(device)

        with amp_ctx:
            gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        decoded_lower = decoded.lower()

        entity_found = entity.lower() in decoded_lower
        obj_found = obj.lower() in decoded_lower
        location_found = location.lower() in decoded_lower
        all_three = entity_found and obj_found and location_found

        chain_intact = bool(re.search(
            rf'{re.escape(entity)}.*{re.escape(obj)}.*{re.escape(entity)}.*{re.escape(location)}',
            decoded, re.IGNORECASE))

        results.append({
            "entity": entity_found,
            "object": obj_found,
            "location": location_found,
            "all_three": all_three,
            "chain_intact": chain_intact,
        })

    entity_rate = sum(r["entity"] for r in results) / n_trials
    obj_rate = sum(r["object"] for r in results) / n_trials
    loc_rate = sum(r["location"] for r in results) / n_trials
    all_rate = sum(r["all_three"] for r in results) / n_trials
    chain_rate = sum(r["chain_intact"] for r in results) / n_trials

    print(f"  Entity found:     {entity_rate:.1%}")
    print(f"  Object found:     {obj_rate:.1%}")
    print(f"  Location found:   {loc_rate:.1%}")
    print(f"  All three found:  {all_rate:.1%}")
    print(f"  Full chain intact:{chain_rate:.1%}")

    return {
        "test": "CNDX-FACT2",
        "n_trials": n_trials,
        "entity_rate": round(entity_rate, 3),
        "object_rate": round(obj_rate, 3),
        "location_rate": round(loc_rate, 3),
        "all_three_rate": round(all_rate, 3),
        "chain_intact_rate": round(chain_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 7: BABILong-lite — Entity state tracking (CNDX-TRACK)
# ---------------------------------------------------------------------------

def _make_tracking_trial(n_updates: int = 3):
    """Generate an entity-tracking passage with multiple location updates.

    Pattern: '<Entity> went to the <loc1>. ... <Entity> moved to the <loc2>. ...'
    Target: does reconstruction preserve the FINAL location (not earlier ones)?
    """
    entity = random.choice(ENTITIES)
    locs = random.sample(LOCATIONS, min(n_updates, len(LOCATIONS)))

    verbs = ["went to the", "moved to the", "travelled to the",
             "walked to the", "returned to the"]

    sentences = []
    filler_pool = list(BABI_FILLER)
    random.shuffle(filler_pool)

    for i, loc in enumerate(locs):
        if i > 0 and filler_pool:
            sentences.append(filler_pool.pop())
        verb = random.choice(verbs)
        sentences.append(f"{entity} {verb} {loc}.")

    while len(sentences) < 7 and filler_pool:
        sentences.append(filler_pool.pop())

    passage = " ".join(sentences)
    final_location = locs[-1]
    earlier_locations = locs[:-1]
    return passage, entity, final_location, earlier_locations


def test_tracking(model, tokenizer, device, amp_ctx, cfg, n_trials: int = 60):
    """Entity state tracking: multiple location updates, check final state."""
    print("\n══════════════════════════════════════════")
    print("  CNDX-TRACK: Entity State Tracking")
    print("══════════════════════════════════════════")
    model.eval()

    for n_updates in [2, 3, 4]:
        results = []
        for trial in range(n_trials):
            passage, entity, final_loc, earlier_locs = _make_tracking_trial(n_updates)

            ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                           padding="max_length", return_tensors="pt")
            input_ids = ids["input_ids"].to(device)
            attn_mask = ids["attention_mask"].to(device)

            with amp_ctx:
                gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
            decoded_lower = decoded.lower()

            entity_found = entity.lower() in decoded_lower
            final_found = final_loc.lower() in decoded_lower
            any_earlier_found = any(el.lower() in decoded_lower for el in earlier_locs)

            results.append({
                "entity_found": entity_found,
                "final_location_found": final_found,
                "earlier_location_found": any_earlier_found,
            })

        entity_rate = sum(r["entity_found"] for r in results) / n_trials
        final_rate = sum(r["final_location_found"] for r in results) / n_trials
        earlier_rate = sum(r["earlier_location_found"] for r in results) / n_trials

        print(f"\n  {n_updates} updates:")
        print(f"    Entity found:          {entity_rate:.1%}")
        print(f"    Final location found:  {final_rate:.1%}")
        print(f"    Earlier loc also found:{earlier_rate:.1%}")

    # Return results for the hardest case (4 updates)
    return {
        "test": "CNDX-TRACK",
        "n_trials": n_trials,
        "n_updates": 4,
        "entity_rate": round(entity_rate, 3),
        "final_location_rate": round(final_rate, 3),
        "earlier_location_rate": round(earlier_rate, 3),
    }


# ---------------------------------------------------------------------------
# No-compression control
# ---------------------------------------------------------------------------

def test_control(model, tokenizer, device, amp_ctx, cfg, n_trials: int = 40):
    """Measure needle scores on raw (uncompressed) input text as ceiling.

    This establishes the maximum possible score: if tokenize → detokenize
    preserves the needle perfectly, the ceiling is 1.0. Any score below this
    on the compressed version is the compression cost.
    """
    print("\n══════════════════════════════════════════")
    print("  CONTROL: No-Compression Ceiling")
    print("══════════════════════════════════════════")

    all_needles = PERSON_NEEDLES + NUMBER_NEEDLES + DATE_NEEDLES + TECHNICAL_NEEDLES

    raw_scores = []
    compressed_scores = []

    model.eval()

    for trial in range(n_trials):
        entity, fact = random.choice(all_needles)
        needle_sentence = f"{entity} {fact}."
        passage = _build_passage(needle_sentence, FILLER_SENTENCES, "middle")

        # Tokenize-detokenize control (no model, just tokenizer round-trip)
        ids = tokenizer(passage, truncation=True, max_length=cfg.seq_len,
                       padding="max_length", return_tensors="pt")
        raw_roundtrip = tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)
        raw_score = _check_needle_in_output(entity, fact, raw_roundtrip)
        raw_scores.append(raw_score)

        # Compressed version (through the latent bottleneck)
        input_ids = ids["input_ids"].to(device)
        attn_mask = ids["attention_mask"].to(device)
        with amp_ctx:
            gen = model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        comp_score = _check_needle_in_output(entity, fact, decoded)
        compressed_scores.append(comp_score)

    raw_entity = sum(s["entity_recall"] for s in raw_scores) / n_trials
    raw_fact = sum(s["fact_recall"] for s in raw_scores) / n_trials
    comp_entity = sum(s["entity_recall"] for s in compressed_scores) / n_trials
    comp_fact = sum(s["fact_recall"] for s in compressed_scores) / n_trials

    print(f"  Raw (tokenizer round-trip):  entity={raw_entity:.1%}  fact={raw_fact:.1%}")
    print(f"  Compressed (latent):         entity={comp_entity:.1%}  fact={comp_fact:.1%}")
    print(f"  Compression cost (entity):   {raw_entity - comp_entity:+.1%}")
    print(f"  Compression cost (fact):     {raw_fact - comp_fact:+.1%}")

    return {
        "test": "CONTROL",
        "n_trials": n_trials,
        "raw_entity_recall": round(raw_entity, 3),
        "raw_fact_recall": round(raw_fact, 3),
        "compressed_entity_recall": round(comp_entity, 3),
        "compressed_fact_recall": round(comp_fact, 3),
        "compression_cost_entity": round(raw_entity - comp_entity, 3),
        "compression_cost_fact": round(raw_fact - comp_fact, 3),
    }


# ---------------------------------------------------------------------------
# Orchestrator
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
        # Strip torch.compile _orig_mod. prefix if present
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No model.pt in {ckpt_dir}")

    model.eval()
    return model, tokenizer, cfg


def run_benchmarks(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    amp_ctx = (torch.autocast("cuda", dtype=dtype) if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))

    print(f"Device: {device}, AMP dtype: {dtype}")

    model, tokenizer, cfg = load_checkpoint(args.checkpoint_dir, device=device)
    print(f"Model: {model.num_params()/1e6:.1f}M params, K={cfg.num_latents}")

    ALL_TESTS = ["control", "niah", "multi", "ftype", "pos", "fact1", "fact2", "track"]
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
        all_results["control"] = test_control(model, tokenizer, device, amp_ctx, cfg)

    if "niah" in tests:
        all_results["niah"] = test_niah(model, tokenizer, device, amp_ctx, cfg)

    if "multi" in tests:
        all_results["multi_needle"] = test_multi_needle(model, tokenizer, device, amp_ctx, cfg)

    if "ftype" in tests:
        all_results["fact_type"] = test_fact_type(model, tokenizer, device, amp_ctx, cfg)

    if "pos" in tests:
        all_results["positional"] = test_positional(model, tokenizer, device, amp_ctx, cfg)

    if "fact1" in tests:
        all_results["fact1"] = test_fact1(model, tokenizer, device, amp_ctx, cfg)

    if "fact2" in tests:
        all_results["fact2"] = test_fact2(model, tokenizer, device, amp_ctx, cfg)

    if "track" in tests:
        all_results["tracking"] = test_tracking(model, tokenizer, device, amp_ctx, cfg)

    out_path = Path(args.checkpoint_dir) / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n══════════════════════════════════════════")
    print(f"  All benchmark results saved to {out_path}")
    print(f"══════════════════════════════════════════")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNDX benchmark suite")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model.pt and results.json")
    parser.add_argument("--tests", type=str, default=None,
                        help="Comma-separated: control,niah,multi,ftype,pos,fact1,fact2,track")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    random.seed(args.seed)
    run_benchmarks(args)
