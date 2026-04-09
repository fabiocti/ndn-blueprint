"""Load TinyStories, Wikitext, Wikipedia, or other HF text datasets."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from cndx.config import CNDXConfig

DATASET_CONFIGS = {
    "roneneldan/TinyStories": {"text_col": "text", "subset": None},
    "wikitext": {"text_col": "text", "subset": "wikitext-103-raw-v1"},
    "wikipedia": {"text_col": "text", "subset": "20231101.en", "paragraph_chunk": True},
    "code": {"text_col": "text", "subset": None, "code_functions": True},
    "security": {"text_col": "text", "subset": None, "security_reports": True},
    "state": {"text_col": "text", "subset": None, "state_traces": True},
    "hwm": {"text_col": "text", "subset": None, "hwm_traces": True},
    "regulated": {"text_col": "text", "subset": None, "regulated_text": True},
    "conversation": {"text_col": "text", "subset": None, "conversation_mem": True},
    "aoj": {"text_col": "text", "subset": None, "aoj_traces": True},
}


def _load_wikipedia_paragraphs(cfg: CNDXConfig, max_samples: int):
    """Load full Wikipedia dump and split articles into paragraph-level examples.

    Logs full provenance: articles scanned, raw paragraphs, filtered, final sampled.
    Training source label: wikipedia_full_articles_paragraphized
    """
    print(f"[train_source = wikipedia_full_articles_paragraphized]")
    print(f"  Loading wikimedia/wikipedia 20231101.en (streaming) ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    print(f"  Streaming mode: will stop after collecting {max_samples:,} usable paragraphs")

    min_chars = cfg.min_text_chars
    paragraphs = []
    articles_scanned = 0
    raw_paragraphs = 0
    filtered_out = 0

    for article in ds:
        articles_scanned += 1
        for p in article["text"].split("\n\n"):
            p = p.strip()
            raw_paragraphs += 1
            if len(p) >= min_chars:
                paragraphs.append(" " + p.lstrip())  # normalize: strip left, then single leading space to match wikitext BPE convention
            else:
                filtered_out += 1
            if len(paragraphs) >= max_samples:
                break
        if len(paragraphs) >= max_samples:
            break

    print(f"  === Wikipedia paragraph extraction report ===")
    print(f"  Articles scanned:       {articles_scanned:>12,}")
    print(f"  Raw paragraphs found:   {raw_paragraphs:>12,}")
    print(f"  Filtered out (<{min_chars} chars): {filtered_out:>12,}")
    print(f"  Usable paragraphs:      {len(paragraphs):>12,}")
    print(f"  Final sampled (cap):    {min(len(paragraphs), max_samples):>12,}")
    print(f"  ============================================")
    return Dataset.from_dict({"text": paragraphs[:max_samples]})


def _load_code_functions(cfg: CNDXConfig, max_samples: int, split: str = "train"):
    """Load function-level Python code chunks from code_search_net.

    Training source label: code_search_net_python_functions
    """
    hf_split = "validation" if split == "validation" else "train"
    print(f"[source = code_search_net_python_functions ({hf_split})]")
    print(f"  Loading code_search_net python ({hf_split}) ...")
    ds = load_dataset("code_search_net", "python", split=hf_split)

    min_chars = cfg.min_text_chars
    functions: list[str] = []
    raw_count = 0
    filtered_out = 0

    for row in ds:
        func = row["whole_func_string"].strip()
        raw_count += 1
        if len(func) >= min_chars:
            functions.append(func)
        else:
            filtered_out += 1
        if len(functions) >= max_samples:
            break

    print(f"  === Code function extraction report ===")
    print(f"  Split:                   {hf_split:>12}")
    print(f"  Raw functions scanned:   {raw_count:>12,}")
    print(f"  Filtered out (<{min_chars} chars): {filtered_out:>12,}")
    print(f"  Usable functions:        {len(functions):>12,}")
    print(f"  Final sampled (cap):     {min(len(functions), max_samples):>12,}")
    print(f"  ========================================")
    return Dataset.from_dict({"text": functions[:max_samples]})


def _load_security_reports(cfg: CNDXConfig, max_samples: int, split: str = "train"):
    """Load HackerOne disclosed vulnerability reports, chunked into ~250-char segments.

    Each report is: metadata header + vulnerability_information body,
    split at line boundaries into chunks targeting ~250 chars (~64 tokens).

    Training source label: hackerone_disclosed_reports_chunked
    """
    hf_split = "validation" if split == "validation" else "train"
    print(f"[source = hackerone_disclosed_reports_chunked ({hf_split})]")
    print(f"  Loading Hacker0x01/hackerone_disclosed_reports ({hf_split}) ...")
    ds = load_dataset("Hacker0x01/hackerone_disclosed_reports", split=hf_split)

    TARGET_CHARS = 250
    min_chars = cfg.min_text_chars
    chunks: list[str] = []
    reports_used = 0
    empty_skipped = 0

    for row in ds:
        vi = (row.get("vulnerability_information") or "").strip()
        if len(vi) < min_chars:
            empty_skipped += 1
            continue

        reports_used += 1
        title = (row.get("title") or "").strip()
        weakness_name = ""
        w = row.get("weakness")
        if w and isinstance(w, dict):
            weakness_name = w.get("name", "")
        target = ""
        scope = row.get("structured_scope")
        if scope and isinstance(scope, dict):
            target = scope.get("asset_identifier", "")

        header = f"Title: {title}"
        if weakness_name:
            header += f"\nWeakness: {weakness_name}"
        if target:
            header += f"\nTarget: {target}"

        full_text = header + "\n\n" + vi

        lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
        current = ""
        for line in lines:
            if current and len(current) + len(line) + 1 > TARGET_CHARS:
                if len(current) >= min_chars:
                    chunks.append(current)
                if len(chunks) >= max_samples:
                    break
                current = line
            else:
                current = (current + "\n" + line).strip() if current else line

            while len(current) > TARGET_CHARS * 2:
                split_at = current.rfind(". ", 0, TARGET_CHARS)
                if split_at < min_chars:
                    split_at = TARGET_CHARS
                chunk = current[:split_at + 1].strip()
                if len(chunk) >= min_chars:
                    chunks.append(chunk)
                if len(chunks) >= max_samples:
                    break
                current = current[split_at + 1:].strip()

            if len(chunks) >= max_samples:
                break

        if current and len(current) >= min_chars and len(chunks) < max_samples:
            chunks.append(current)
        if len(chunks) >= max_samples:
            break

    print(f"  === Security report extraction report ===")
    print(f"  Split:                   {hf_split:>12}")
    print(f"  Reports scanned:         {len(ds):>12,}")
    print(f"  Reports used:            {reports_used:>12,}")
    print(f"  Empty/short skipped:     {empty_skipped:>12,}")
    print(f"  Chunks created:          {len(chunks):>12,}")
    print(f"  Final sampled (cap):     {min(len(chunks), max_samples):>12,}")
    print(f"  =========================================")
    return Dataset.from_dict({"text": chunks[:max_samples]})


def _load_regulated_text(cfg: CNDXConfig, max_samples: int, split: str = "train"):
    """Generate synthetic high-precision regulated text.

    Domain: High-Precision Regulated Text — policy clauses, operating
    procedures, compliance rules, thresholds, escalation chains, exceptions,
    role-based obligations. Exactness-sensitive: one wrong word matters.

    Training source label: synthetic_regulated_v1
    """
    import random as _rng

    seed_offset = 0 if split == "train" else 888888
    _rng.seed(42 + seed_offset)

    print(f"[source = synthetic_regulated_v1 ({split})]")
    print(f"  Generating {max_samples:,} regulated-text samples ...")

    _ROLES = [
        "Analyst", "Senior Analyst", "Compliance Officer", "Risk Manager",
        "Auditor", "Team Lead", "Director", "VP of Operations",
        "Data Protection Officer", "Incident Commander", "Account Manager",
        "Quality Assurance Lead", "Regional Supervisor", "Chief Risk Officer",
        "Operations Manager", "Security Officer",
    ]
    _DEPARTMENTS = [
        "Risk & Compliance", "Internal Audit", "Operations", "Legal",
        "Information Security", "Human Resources", "Finance",
        "Quality Assurance", "Client Services", "Technology",
    ]
    _ACTIONS = [
        "escalate to", "notify", "file a report with", "obtain approval from",
        "submit documentation to", "initiate review by", "suspend pending review by",
        "transfer ownership to", "request exception from", "archive records with",
    ]
    _DURATIONS = [
        "7 days", "14 days", "30 days", "60 days", "90 days",
        "120 days", "6 months", "1 year", "3 business days",
        "5 business days", "10 business days", "24 hours", "48 hours",
    ]
    _THRESHOLDS = [
        "$5,000", "$10,000", "$25,000", "$50,000", "$100,000",
        "$250,000", "$500,000", "$1,000,000", "500 records",
        "1,000 transactions", "10% variance", "15% deviation",
        "3 occurrences", "5 incidents", "threshold level 2",
        "threshold level 3",
    ]
    _CATEGORIES = [
        "Category A", "Category B", "Category C", "Tier-1", "Tier-2",
        "Tier-3", "Priority 1", "Priority 2", "Class I", "Class II",
        "Level Alpha", "Level Beta",
    ]
    _DOC_TYPES = [
        "incident report", "exception request", "variance analysis",
        "compliance attestation", "audit trail", "change request",
        "risk assessment", "remediation plan", "disclosure form",
        "retention schedule", "escalation log", "approval record",
    ]
    _CONDITIONS = [
        "the amount exceeds", "the count surpasses", "the variance is above",
        "the elapsed time exceeds", "the risk score is above",
        "the error rate exceeds", "the exposure is greater than",
        "the transaction volume surpasses",
    ]
    _FLAGS = [
        "confidentiality flag", "PII indicator", "restricted access marker",
        "cross-border flag", "manual override indicator", "high-risk tag",
        "expedited processing flag", "regulatory hold flag",
    ]
    _VERSIONS = ["1.0", "1.1", "2.0", "2.1", "3.0", "3.1", "4.0"]
    _DATES = [
        "1 January 2025", "15 March 2025", "1 April 2025", "30 June 2025",
        "1 September 2025", "31 December 2025", "1 February 2026",
        "15 May 2026", "1 July 2026", "30 September 2026",
    ]
    _STATUSES = [
        "Draft", "Under Review", "Approved", "Effective", "Superseded",
        "Withdrawn", "Pending Amendment",
    ]
    _FILLER_CLAUSES = [
        "All personnel must complete annual compliance training before the deadline.",
        "Records must be stored in an immutable audit-compliant repository.",
        "Quarterly reconciliation reports shall be submitted to the designated authority.",
        "Access to restricted systems requires multi-factor authentication.",
        "Any deviation from standard procedure must be documented and justified.",
        "Communication with external regulators must follow the approved template.",
        "All exceptions must be logged in the central exception register.",
        "Backup procedures must be tested at least once per fiscal quarter.",
        "Third-party vendor assessments must be completed prior to contract renewal.",
        "Data classification labels must be applied to all new documents upon creation.",
    ]

    def _gen_policy_clause(rng):
        role = rng.choice(_ROLES)
        dept = rng.choice(_DEPARTMENTS)
        action = rng.choice(_ACTIONS)
        dur = rng.choice(_DURATIONS)
        thresh = rng.choice(_THRESHOLDS)
        cond = rng.choice(_CONDITIONS)
        cat = rng.choice(_CATEGORIES)
        templates = [
            f"CLAUSE: If {cond} {thresh}, {role} must {action} {dept} within {dur}.",
            f"REQUIREMENT: {role} ({dept}) shall retain all {rng.choice(_DOC_TYPES)} records for a minimum of {dur}.",
            f"POLICY: {cat} items require {role} approval when {cond} {thresh}. Exceptions must be filed within {rng.choice(_DURATIONS)}.",
            f"RULE: Upon classification as {cat}, {action} {role} no later than {dur} after the triggering event.",
        ]
        return rng.choice(templates)

    def _gen_procedure(rng):
        role1 = rng.choice(_ROLES)
        role2 = rng.choice([r for r in _ROLES if r != role1])
        dur1 = rng.choice(_DURATIONS)
        dur2 = rng.choice(_DURATIONS)
        doc = rng.choice(_DOC_TYPES)
        templates = [
            f"PROCEDURE: Step 1: {role1} prepares the {doc}. Step 2: Submit to {role2} within {dur1}. Step 3: {role2} reviews and responds within {dur2}.",
            f"WORKFLOW: {role1} initiates the {doc}. If no response from {role2} within {dur1}, auto-escalate to {rng.choice([r for r in _ROLES if r != role1 and r != role2])}.",
            f"PROCESS: Draft {doc} ({role1}) → Review ({role2}, {dur1}) → Final approval ({rng.choice(_ROLES)}, {dur2}).",
        ]
        return rng.choice(templates)

    def _gen_threshold_rule(rng):
        thresh = rng.choice(_THRESHOLDS)
        role = rng.choice(_ROLES)
        action = rng.choice(_ACTIONS)
        cat = rng.choice(_CATEGORIES)
        cond = rng.choice(_CONDITIONS)
        flag = rng.choice(_FLAGS)
        templates = [
            f"THRESHOLD: When {cond} {thresh}, classify as {cat} and {action} {role}.",
            f"LIMIT: {cat} designation applies when {cond} {thresh}. If {flag} is set, {action} {rng.choice(_ROLES)} immediately.",
            f"ESCALATION: Below {thresh}: {rng.choice(_ROLES)} handles. At or above {thresh}: {role} required. If {flag} is active, skip to {rng.choice(_ROLES)}.",
        ]
        return rng.choice(templates)

    def _gen_exception(rng):
        role = rng.choice(_ROLES)
        cat = rng.choice(_CATEGORIES)
        flag = rng.choice(_FLAGS)
        dur = rng.choice(_DURATIONS)
        templates = [
            f"EXCEPTION: {cat} items are exempt from the standard {dur} retention period only when {flag} is set and approved by {role}.",
            f"OVERRIDE: The {dur} deadline may be extended by {role} if the {flag} is active. Extension must not exceed {rng.choice(_DURATIONS)}.",
            f"WAIVER: {role} may waive the {cat} requirement for {dur} subject to written justification filed with {rng.choice(_DEPARTMENTS)}.",
        ]
        return rng.choice(templates)

    def _gen_version_update(rng):
        v_old = rng.choice(_VERSIONS[:4])
        v_new = rng.choice(_VERSIONS[3:])
        while v_new <= v_old:
            v_new = rng.choice(_VERSIONS[3:])
        date = rng.choice(_DATES)
        doc = rng.choice(_DOC_TYPES)
        templates = [
            f"VERSION: {doc} procedure version {v_new} supersedes version {v_old} effective {date}.",
            f"UPDATE: As of {date}, version {v_old} of the {doc} process is {rng.choice(['Superseded', 'Withdrawn'])}. Version {v_new} is now {rng.choice(['Effective', 'Approved'])}.",
            f"REVISION: {doc} v{v_new} ({date}): replaces v{v_old}. Status: {rng.choice(_STATUSES)}.",
        ]
        return rng.choice(templates)

    def _gen_conditional(rng):
        cond = rng.choice(_CONDITIONS)
        thresh = rng.choice(_THRESHOLDS)
        role1 = rng.choice(_ROLES)
        role2 = rng.choice([r for r in _ROLES if r != role1])
        action1 = rng.choice(_ACTIONS)
        action2 = rng.choice(_ACTIONS)
        flag = rng.choice(_FLAGS)
        templates = [
            f"CONDITIONAL: If {cond} {thresh} AND {flag} is set, {action1} {role1}. Otherwise, {action2} {role2}.",
            f"RULE: When {cond} {thresh}: (a) {action1} {role1} within {rng.choice(_DURATIONS)}, (b) if {flag} applies, also {action2} {role2}.",
            f"BRANCH: {cond} {thresh} → route to {role1}. If additionally {flag} is active → route to {role2} instead.",
        ]
        return rng.choice(templates)

    samples = []
    for _ in range(max_samples):
        n_clauses = _rng.randint(3, 6)
        clauses = []
        for _ in range(n_clauses):
            ctype = _rng.choice([
                "policy", "procedure", "threshold", "exception",
                "version", "conditional", "filler",
            ])
            if ctype == "policy":
                clauses.append(_gen_policy_clause(_rng))
            elif ctype == "procedure":
                clauses.append(_gen_procedure(_rng))
            elif ctype == "threshold":
                clauses.append(_gen_threshold_rule(_rng))
            elif ctype == "exception":
                clauses.append(_gen_exception(_rng))
            elif ctype == "version":
                clauses.append(_gen_version_update(_rng))
            elif ctype == "conditional":
                clauses.append(_gen_conditional(_rng))
            else:
                clauses.append(_rng.choice(_FILLER_CLAUSES))
        samples.append("\n".join(clauses))

    print(f"  Samples generated:       {len(samples):>12,}")
    print(f"  Avg length (chars):      {sum(len(s) for s in samples) // len(samples):>12,}")
    print(f"  =========================================")
    return Dataset.from_dict({"text": samples[:max_samples]})


def _load_hwm_traces(cfg: CNDXConfig, max_samples: int, split: str = "train"):
    """Generate synthetic human working-memory text.

    Domain: Human Working Memory Text — fragmented notes, reminders, partial
    plans, messy task state, inconsistent wording, real-ish working context.
    Unlike all previous domains, this has no clean structure to lean on.

    Training source label: synthetic_hwm_v1
    """
    import random as _rng

    seed_offset = 0 if split == "train" else 777777
    _rng.seed(42 + seed_offset)

    print(f"[source = synthetic_hwm_v1 ({split})]")
    print(f"  Generating {max_samples:,} human working-memory samples ...")

    _PEOPLE = [
        "Sarah", "Mike", "John", "Lisa", "Dave", "Emma", "Chris", "Amy",
        "Tom", "Rachel", "Ben", "Maria", "Jake", "Nina", "Alex", "Kate",
        "Dan", "Olivia", "Ryan", "Sophie",
    ]
    _PLACES = [
        "the office", "room 204", "downtown", "the cafe on 5th",
        "building B", "the parking garage", "conference room",
        "the library", "home", "the gym", "airport terminal 3",
        "the park", "upstairs", "the lab", "client site",
    ]
    _TIMES = [
        "2pm", "3:30", "tomorrow morning", "friday", "next week",
        "before lunch", "after the meeting", "end of day", "9am sharp",
        "sometime this afternoon", "monday at 10", "asap", "tonight",
        "in 20 minutes", "by thursday",
    ]
    _TASKS = [
        "fix the login bug", "review the PR", "update the docs",
        "send the report", "call the vendor", "book the flight",
        "finish the slides", "order new monitors", "cancel the subscription",
        "reschedule the demo", "write the proposal", "clean up the repo",
        "test the deployment", "set up the new hire", "draft the email",
        "check the invoices", "merge the branch", "backup the database",
        "prepare the agenda", "file the expense report",
    ]
    _ITEMS = [
        "milk", "eggs", "bread", "coffee", "batteries", "notebooks",
        "printer paper", "tape", "stamps", "charger", "headphones",
        "umbrella", "snacks", "water bottles", "pens", "folders",
    ]
    _PROJECTS = [
        "Q3 budget", "website redesign", "API migration",
        "client onboarding", "security audit", "performance review",
        "product launch", "data cleanup", "hiring pipeline",
        "vendor evaluation", "compliance update", "user research",
    ]
    _EMOTIONS = [
        "ugh", "nice!", "worried about this", "not sure about",
        "excited re:", "annoyed that", "relieved that",
        "need to think about", "forgot about", "finally!",
    ]
    _CORRECTIONS = [
        "actually no,", "wait,", "scratch that -", "update:",
        "correction:", "nvm,", "EDIT:", "changed to:",
    ]
    _FILLERS = [
        "...", "???", "hmm", "idk", "tbd", "check later",
        "ask about this", "not sure yet", "look into it",
        "pending", "follow up needed", "low priority",
    ]

    samples = []
    for _ in range(max_samples):
        n_fragments = _rng.randint(4, 8)
        fragments = []
        for _ in range(n_fragments):
            ftype = _rng.choice([
                "todo", "reminder", "note", "plan", "status",
                "idea", "contact", "correction", "list", "thought",
            ])
            if ftype == "todo":
                fragments.append(_hwm_todo(_rng, _TASKS, _PEOPLE, _TIMES, _FILLERS))
            elif ftype == "reminder":
                fragments.append(_hwm_reminder(_rng, _PEOPLE, _PLACES, _TIMES, _TASKS))
            elif ftype == "note":
                fragments.append(_hwm_note(_rng, _PEOPLE, _PROJECTS, _PLACES))
            elif ftype == "plan":
                fragments.append(_hwm_plan(_rng, _TASKS, _PEOPLE, _TIMES))
            elif ftype == "status":
                fragments.append(_hwm_status(_rng, _TASKS, _PROJECTS, _PEOPLE))
            elif ftype == "idea":
                fragments.append(_hwm_idea(_rng, _PROJECTS, _EMOTIONS))
            elif ftype == "contact":
                fragments.append(_hwm_contact(_rng, _PEOPLE, _PLACES, _TIMES))
            elif ftype == "correction":
                fragments.append(_hwm_correction(_rng, _CORRECTIONS, _PEOPLE, _TIMES, _PLACES))
            elif ftype == "list":
                fragments.append(_hwm_list(_rng, _ITEMS, _TASKS))
            else:
                fragments.append(_hwm_thought(_rng, _EMOTIONS, _PROJECTS, _FILLERS))
        samples.append("\n".join(fragments))

    print(f"  Samples generated:       {len(samples):>12,}")
    print(f"  Avg length (chars):      {sum(len(s) for s in samples) // len(samples):>12,}")
    print(f"  =========================================")
    return Dataset.from_dict({"text": samples[:max_samples]})


def _hwm_todo(rng, tasks, people, times, fillers):
    task = rng.choice(tasks)
    prefix = rng.choice(["TODO", "todo:", "- [ ]", "TASK:", "need to:", ">>"])
    parts = [f"{prefix} {task}"]
    if rng.random() < 0.4:
        parts.append(f"- {rng.choice(people)} can help")
    if rng.random() < 0.5:
        parts.append(f"due {rng.choice(times)}")
    if rng.random() < 0.2:
        parts.append(f"({rng.choice(fillers)})")
    return " ".join(parts)


def _hwm_reminder(rng, people, places, times, tasks):
    person = rng.choice(people)
    time = rng.choice(times)
    action = rng.choice(tasks)
    templates = [
        f"reminder: {action} - {time}",
        f"REMIND: call {person} at {time}",
        f"dont forget: meet {person} at {rng.choice(places)} {time}",
        f"@ {time} - {person} re: {action}",
        f"!! {action} before {time}",
    ]
    return rng.choice(templates)


def _hwm_note(rng, people, projects, places):
    person = rng.choice(people)
    project = rng.choice(projects)
    templates = [
        f"NOTE: {person} said {project} is behind schedule",
        f"{person} mentioned the {project} thing - look into it",
        f"from meeting: {project} budget is ${rng.randint(5, 50)}k over",
        f"fyi {person} is out {rng.choice(['monday', 'next week', 'tomorrow', 'friday'])}",
        f"important: {project} deadline moved to {rng.choice(['march', 'next month', 'Q4', 'end of sprint'])}",
        f"{person} wants to discuss {project} at {rng.choice(places)}",
    ]
    return rng.choice(templates)


def _hwm_plan(rng, tasks, people, times):
    n = rng.randint(2, 4)
    items = rng.sample(tasks, min(n, len(tasks)))
    lines = [rng.choice(["plan:", "agenda:", "steps:", "today:", "this week:"])]
    for i, item in enumerate(items, 1):
        if rng.random() < 0.2:
            lines.append(f"{i}. ??? ({rng.choice(['tbd', 'ask about this', 'not sure'])})")
        else:
            lines.append(f"{i}. {item}")
    if rng.random() < 0.3:
        lines.append(f"check with {rng.choice(people)} first")
    return "\n".join(lines)


def _hwm_status(rng, tasks, projects, people):
    task = rng.choice(tasks)
    status = rng.choice([
        "done", "in progress", "blocked", "50% done",
        "started but stuck", "waiting on response", "almost there",
        "haven't started", "need more info", "deprioritized",
    ])
    templates = [
        f"status: {task} -> {status}",
        f"[{status.upper()}] {task}",
        f"{task}: {status}. {rng.choice(people)} is helping",
        f"update on {rng.choice(projects)}: {status}",
    ]
    return rng.choice(templates)


def _hwm_idea(rng, projects, emotions):
    project = rng.choice(projects)
    templates = [
        f"IDEA: what if we {rng.choice(['automate', 'simplify', 'outsource', 'parallelize', 'cache'])} the {project}?",
        f"thought: {project} could be {rng.choice(['faster', 'cheaper', 'simpler', 'better'])} if we {rng.choice(['restructure', 'split it up', 'add a buffer', 'batch it'])}",
        f"{rng.choice(emotions)} {project}",
        f"random: maybe {project} and {rng.choice(projects)} are related?",
    ]
    return rng.choice(templates)


def _hwm_contact(rng, people, places, times):
    person = rng.choice(people)
    phone = f"{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    templates = [
        f"{person}: {phone}",
        f"call {person} ({phone}) re: the thing",
        f"{person} - office at {rng.choice(places)}, avail {rng.choice(times)}",
        f"contact: {person} {phone} (best {rng.choice(['mornings', 'after 2', 'anytime', 'not fridays'])})",
    ]
    return rng.choice(templates)


def _hwm_correction(rng, corrections, people, times, places):
    correction = rng.choice(corrections)
    templates = [
        f"{correction} meeting is {rng.choice(times)} not {rng.choice(times)}",
        f"{correction} it was {rng.choice(people)} not {rng.choice(people)}",
        f"{correction} the thing is at {rng.choice(places)}",
        f"{correction} budget is ${rng.randint(10, 100)}k not ${rng.randint(10, 100)}k",
    ]
    return rng.choice(templates)


def _hwm_list(rng, items, tasks):
    n = rng.randint(3, 6)
    chosen = rng.sample(items, min(n, len(items)))
    header = rng.choice(["shopping:", "buy:", "grab:", "need:", "pick up:"])
    return f"{header} {', '.join(chosen)}"


def _hwm_thought(rng, emotions, projects, fillers):
    templates = [
        f"{rng.choice(emotions)} {rng.choice(projects)} {rng.choice(fillers)}",
        f"why is {rng.choice(projects)} so complicated {rng.choice(fillers)}",
        f"should probably {rng.choice(['prioritize', 'delegate', 'postpone', 'revisit'])} {rng.choice(projects)}",
        f"{rng.choice(fillers)}... {rng.choice(projects)}",
    ]
    return rng.choice(templates)


def _load_conversation_mem(cfg: CNDXConfig, max_samples: int, split: str = "train"):
    """Load real multi-turn conversational text from multiple public datasets.

    Domain: Conversational Memory / Multi-turn Dialogue — speaker turns,
    facts introduced casually, corrections, temporal references, preferences,
    event updates, indirect references. Uses real conversational data to
    ensure organic diversity that prevents memorization.

    Sources: DialogSum + Blended Skill Talk + OpenAssistant (oasst1)
    Training source label: real_conversation_v2
    """
    import random as _rng
    from collections import defaultdict
    _rng.seed(42 + (0 if split == "train" else 666666))

    hf_split = "validation" if split == "validation" else "train"
    print(f"[source = real_conversation_v2 ({hf_split})]")
    print(f"  Loading real conversational datasets ...")

    min_chars = cfg.min_text_chars
    samples: list[str] = []

    # Source 1: DialogSum — natural daily conversations (12k+ train)
    try:
        print(f"  Loading dialogsum ({hf_split}) ...")
        ds_split = hf_split if hf_split != "validation" else "validation"
        dd = load_dataset("knkarthick/dialogsum", split=ds_split)
        for row in dd:
            dialog = row.get("dialogue", "")
            if not dialog or len(dialog) < min_chars:
                continue
            lines = []
            for line in dialog.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#Person1#:"):
                    lines.append("user: " + line[len("#Person1#:"):].strip())
                elif line.startswith("#Person2#:"):
                    lines.append("assistant: " + line[len("#Person2#:"):].strip())
                else:
                    lines.append(line)
            text = "\n".join(lines)
            if len(text) >= min_chars:
                samples.append(text)
        print(f"    DialogSum: {len(samples)} samples so far")
    except Exception as e:
        print(f"    DialogSum failed: {e}")

    # Source 2: Blended Skill Talk — knowledge + persona + empathy (4.8k train)
    if len(samples) < max_samples:
        try:
            print(f"  Loading blended_skill_talk ({hf_split}) ...")
            bst = load_dataset("blended_skill_talk", split=hf_split)
            for row in bst:
                prev = row.get("previous_utterance", [])
                free = row.get("free_messages", [])
                guided = row.get("guided_messages", [])
                turns = list(prev) + [m for pair in zip(free, guided) for m in pair]
                if len(turns) < 2:
                    continue
                lines = []
                for i, utt in enumerate(turns):
                    utt = utt.strip()
                    if not utt:
                        continue
                    role = "user" if i % 2 == 0 else "assistant"
                    lines.append(f"{role}: {utt}")
                text = "\n".join(lines)
                if len(text) >= min_chars:
                    samples.append(text)
            print(f"    BlendedSkillTalk: {len(samples)} samples so far")
        except Exception as e:
            print(f"    BlendedSkillTalk failed: {e}")

    # Source 3: OpenAssistant (oasst1) — crowd-sourced conversations (~10k trees)
    if len(samples) < max_samples:
        try:
            print(f"  Loading OpenAssistant/oasst1 ({hf_split}) ...")
            oa = load_dataset("OpenAssistant/oasst1", split=hf_split)
            en_msgs = [r for r in oa if r.get("lang", "") == "en"]
            msg_map = {r["message_id"]: r for r in en_msgs}
            trees = defaultdict(list)
            for r in en_msgs:
                trees[r["message_tree_id"]].append(r)
            for tree_id, msgs in trees.items():
                roots = [m for m in msgs if m["parent_id"] is None]
                if not roots:
                    continue
                def _build_thread(msg_id, depth=0):
                    if depth > 10:
                        return []
                    msg = msg_map.get(msg_id)
                    if not msg:
                        return []
                    children = [m for m in msgs if m["parent_id"] == msg_id]
                    result = [(msg["role"], msg["text"])]
                    if children:
                        best = children[0]
                        result.extend(_build_thread(best["message_id"], depth + 1))
                    return result
                thread = _build_thread(roots[0]["message_id"])
                if len(thread) < 2:
                    continue
                lines = []
                for role, text in thread:
                    label = "user" if role == "prompter" else "assistant"
                    lines.append(f"{label}: {text.strip()}")
                text = "\n".join(lines)
                if len(text) >= min_chars:
                    samples.append(text)
            print(f"    OpenAssistant: {len(samples)} samples so far")
        except Exception as e:
            print(f"    OpenAssistant failed: {e}")

    # Source 4: Entity-rich synthetic conversations (~35K)
    # 12 template families with diverse named entities to prevent entity-prior projection
    _PERSON_NAMES = [
        "Alice", "Bob", "Charlie", "Dana", "Elijah", "Fiona", "Greg", "Hannah",
        "Ivan", "Julia", "Kai", "Luna", "Marcus", "Nina", "Oscar", "Priya",
        "Quinn", "Ravi", "Sofia", "Tariq", "Uma", "Victor", "Wendy", "Xander",
        "Yuki", "Zara", "Amir", "Beatrice", "Carlos", "Daphne", "Eduardo",
        "Freya", "Gustavo", "Helena", "Ibrahim", "Jasmine", "Kenji", "Liam",
        "Mei", "Nikolai", "Olga", "Pedro", "Rosa", "Sven", "Tomoko", "Ulrich",
    ]
    _HANDLES = [f"@{n.lower()}{_rng.randint(1,999)}" for n in _PERSON_NAMES[:20]]
    _URLS = [
        "github.com/project-alpha", "docs.example.io/api", "store.coolbrand.com",
        "maps.google.com/place/Tokyo", "open.spotify.com/playlist/xyz",
        "arxiv.org/abs/2401.12345", "news.ycombinator.com/item?id=99999",
        "en.wikipedia.org/wiki/Transformer", "stackoverflow.com/q/12345678",
        "reddit.com/r/MachineLearning", "medium.com/@techwriter/article",
        "twitch.tv/streamername", "youtube.com/watch?v=abcdefg",
    ]
    _PRODUCTS = [
        "iPhone 15 Pro", "Galaxy S24 Ultra", "MacBook Air M3", "ThinkPad X1",
        "AirPods Pro 2", "Kindle Paperwhite", "Steam Deck OLED", "PS5 Slim",
        "Dyson V15", "Instant Pot Duo", "Roomba j7+", "Bose QC Ultra",
        "Nintendo Switch 2", "Pixel 9", "Surface Pro 10", "iPad Mini 7",
    ]
    _PLACES = [
        "Tokyo", "Berlin", "São Paulo", "Toronto", "Mumbai", "Lagos",
        "Sydney", "Cairo", "Stockholm", "Mexico City", "Seoul", "Amsterdam",
        "Nairobi", "Singapore", "Buenos Aires", "Lisbon", "Bangkok", "Dublin",
        "Cape Town", "Vancouver", "Osaka", "Zurich", "Barcelona", "Denver",
    ]
    _BOOKS = [
        "Dune", "Project Hail Mary", "The Midnight Library", "Klara and the Sun",
        "Piranesi", "The Overstory", "Exhalation", "Children of Time",
        "Pachinko", "The Poppy War", "Circe", "Normal People",
    ]
    _SHOWS = [
        "Severance", "The Bear", "Shogun", "Fallout", "3 Body Problem",
        "Slow Horses", "Reacher", "Andor", "The Last of Us", "Beef",
        "Blue Eye Samurai", "Silo", "For All Mankind", "Dark",
    ]
    _SONGS = [
        "Bohemian Rhapsody", "Blinding Lights", "Levitating", "Heat Waves",
        "anti-hero", "Flowers", "vampire", "Cruel Summer", "Paint The Town Red",
        "Espresso", "greedy", "Stick Season", "Snooze", "Kill Bill",
    ]
    _VENUES = [
        "Madison Square Garden", "Wembley Stadium", "Red Rocks Amphitheatre",
        "Sydney Opera House", "Berghain", "Hollywood Bowl", "Royal Albert Hall",
        "Budokan", "Coachella", "Glastonbury", "Fuji Rock", "Primavera Sound",
    ]

    def _synth_entity_conv(rng):
        """Generate one entity-rich synthetic conversation from template families."""
        family = rng.randint(0, 11)
        p1, p2 = rng.sample(_PERSON_NAMES, 2)
        place = rng.choice(_PLACES)

        if family == 0:  # product recommendation
            prod = rng.choice(_PRODUCTS)
            return (f"user: Hey have you seen the {prod}? I'm thinking of getting one.\n"
                    f"assistant: Yeah {p1} got one last week. Said the battery life is amazing.\n"
                    f"user: How much was it?\n"
                    f"assistant: I think around ${rng.randint(200,1500)}. Check {rng.choice(_URLS)}")
        elif family == 1:  # travel planning
            p3 = rng.choice(_PLACES)
            return (f"user: Planning a trip to {place} next month.\n"
                    f"assistant: Nice! {p2} went there in {rng.choice(['January','March','June','October'])}. "
                    f"Said the food scene is incredible.\n"
                    f"user: Any restaurant recs?\n"
                    f"assistant: {p2} loved this place called {rng.choice(['Noma','Gaggan','Asador','Quintonil','Maido'])}. "
                    f"Book early though, {rng.randint(2,8)} week wait.")
        elif family == 2:  # book/show discussion
            item = rng.choice(_BOOKS + _SHOWS)
            return (f"user: Just finished {item}. Mind blown.\n"
                    f"assistant: Oh {p1} has been telling me to watch/read that for months.\n"
                    f"user: The ending though... no spoilers but wow.\n"
                    f"assistant: Adding it to my list. Right after {rng.choice(_BOOKS + _SHOWS)}.")
        elif family == 3:  # music/concert
            song = rng.choice(_SONGS)
            venue = rng.choice(_VENUES)
            return (f"user: Got tickets to see the concert at {venue}!\n"
                    f"assistant: No way! When?\n"
                    f"user: {rng.choice(['March','April','May','June','July'])} {rng.randint(1,28)}th. "
                    f"Section {rng.choice(['A','B','C','D'])}{rng.randint(1,30)}, row {rng.randint(1,50)}.\n"
                    f"assistant: {p2} saw them last year. Said they played {song} as the encore.")
        elif family == 4:  # work/meeting
            return (f"user: Meeting with {p1} moved to {rng.randint(1,5)}pm tomorrow.\n"
                    f"assistant: Got it. Is {p2} still joining?\n"
                    f"user: Yes, plus {rng.choice(_PERSON_NAMES)} from the {place} office.\n"
                    f"assistant: I'll update the calendar. Room {rng.choice(['A','B','C'])}{rng.randint(100,999)}?")
        elif family == 5:  # social media / links
            handle = rng.choice(_HANDLES)
            url = rng.choice(_URLS)
            return (f"user: Did you see {handle}'s post about {rng.choice(_PRODUCTS)}?\n"
                    f"assistant: The one with {rng.randint(1000,50000)} likes? Yeah it went viral.\n"
                    f"user: Here's the link: {url}\n"
                    f"assistant: {p1} shared it in our group chat already. {rng.randint(10,200)} comments.")
        elif family == 6:  # food/cooking
            return (f"user: Making dinner for {rng.randint(4,12)} people on Saturday.\n"
                    f"assistant: {p2}'s recipe for that pasta was great last time. "
                    f"Need about {rng.randint(1,5)} lbs of chicken.\n"
                    f"user: Good idea. {p1} is bringing dessert from that bakery on {rng.choice(['5th','Main','Oak','Elm'])} St.\n"
                    f"assistant: The one near {place}? Their {rng.choice(['tiramisu','cheesecake','croissants','macarons'])} are amazing.")
        elif family == 7:  # health/fitness
            return (f"user: Ran {rng.randint(3,26)}.{rng.randint(0,9)} miles today. New PR!\n"
                    f"assistant: That's faster than {p1}'s time from last month.\n"
                    f"user: Training for the {place} marathon in {rng.choice(['April','October','November'])}.\n"
                    f"assistant: {p2} is doing that too. You should train together. Meet at {rng.choice(['6','7','8'])}am?")
        elif family == 8:  # tech/coding
            return (f"user: The API at {rng.choice(_URLS)} is returning {rng.choice(['404','500','403','429'])} errors.\n"
                    f"assistant: {p1} filed a bug about that. Issue #{rng.randint(100,9999)}.\n"
                    f"user: Since when?\n"
                    f"assistant: Started around {rng.randint(1,12)}:{rng.randint(0,5)}0 UTC yesterday. "
                    f"{rng.randint(20,500)} users affected.")
        elif family == 9:  # shopping/errands
            return (f"user: Need to pick up {p2}'s birthday gift. Any ideas?\n"
                    f"assistant: They mentioned wanting a {rng.choice(_PRODUCTS)} last week.\n"
                    f"user: Budget is around ${rng.randint(50,500)}.\n"
                    f"assistant: Check {rng.choice(_URLS)}. {p1} found a deal there for ${rng.randint(30,400)}.")
        elif family == 10:  # event coordination
            return (f"user: {p1}'s party is at {rng.choice(_VENUES).split()[0]} on the {rng.randint(1,28)}th.\n"
                    f"assistant: How many people?\n"
                    f"user: About {rng.randint(15,150)}. {p2} is handling the playlist.\n"
                    f"assistant: I'll bring {rng.randint(2,6)} bottles. Should I pick up {rng.choice(_PERSON_NAMES)} "
                    f"from the {rng.choice(['airport','station','hotel'])}?")
        else:  # correction/update
            return (f"user: Actually {p1}'s number is {rng.randint(100,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}, not what I said before.\n"
                    f"assistant: Updated. And the meeting is at {rng.randint(1,12)}:{rng.choice(['00','15','30','45'])} right?\n"
                    f"user: No wait, {p2} moved it to {rng.randint(1,12)}:{rng.choice(['00','15','30','45'])}. "
                    f"Check {rng.choice(_URLS)} for the updated invite.\n"
                    f"assistant: Got it. Room changed to {rng.choice(['A','B','C','D'])}{rng.randint(100,999)} in the {place} building.")

    entity_target = min(35000, max(0, max_samples - len(samples)))
    if entity_target > 0:
        print(f"  Generating {entity_target:,} entity-rich synthetic conversations ...")
        entity_samples = [_synth_entity_conv(_rng) for _ in range(entity_target)]
        samples.extend(entity_samples)
        print(f"    Entity-rich synthetic: {len(entity_samples)} samples added")

    real_count = len(samples)
    print(f"  Total samples before windowing: {real_count}")
    if real_count < max_samples and real_count > 0:
        print(f"  Augmenting to {max_samples} via sub-conversation windowing ...")
        augmented = []
        while len(samples) + len(augmented) < max_samples:
            src = _rng.choice(samples[:real_count])
            lines = src.split("\n")
            if len(lines) <= 2:
                augmented.append(src)
                continue
            win_size = _rng.randint(2, max(2, len(lines) - 1))
            start = _rng.randint(0, len(lines) - win_size)
            window = "\n".join(lines[start:start + win_size])
            if len(window) >= min_chars:
                augmented.append(window)
        samples.extend(augmented)
        print(f"  After augmentation: {len(samples)} samples")

    _rng.shuffle(samples)
    samples = samples[:max_samples]

    print(f"  === Conversation corpus report ===")
    print(f"  Split:                   {hf_split:>12}")
    print(f"  Total samples:           {len(samples):>12,}")
    if samples:
        print(f"  Avg length (chars):      {sum(len(s) for s in samples) // len(samples):>12,}")
    print(f"  ====================================")
    return Dataset.from_dict({"text": samples})


def _load_aoj_traces(cfg: CNDXConfig, max_samples: int, split: str = "train"):
    """Generate synthetic agent operational journal entries.

    Domain: OSA / Agent Operational Journals — markdown-formatted recon
    summaries, tool outputs, host counts, findings lists, phase/status
    transitions, blockers, and operational notes. Designed to match
    real-world agent workflow journals from bug bounty, DevOps, and
    data pipeline automation contexts.

    v2: Massively expanded entity diversity — 500+ real-world domain names,
    real bug bounty program names, real OpenClaw journal mixing.
    Training source label: synthetic_aoj_v2
    """
    import random as _rng

    seed_offset = 0 if split == "train" else 555555
    _rng.seed(42 + seed_offset)

    print(f"[source = synthetic_aoj_v2 ({split})]")
    print(f"  Generating {max_samples:,} agent operational journal samples ...")

    _TLDS = [
        "com", "net", "org", "io", "co", "dev", "app", "cloud",
        "tech", "ai", "biz", "info", "xyz", "gg", "me", "pro",
        "in", "uk", "de", "fr", "jp", "br", "au", "ca", "eu",
        "ru", "cn", "kr", "nl", "se", "ch", "es", "it", "pl",
        "at", "fi", "no", "dk", "pt", "ie", "be", "sg", "hk",
        "tw", "nz", "mx", "ar", "cl", "za", "ae", "sa", "il",
    ]
    # Real-world SLDs — companies, services, products that appear on bug bounty platforms
    _SLD_PARTS = [
        # Tech giants and major platforms
        "google", "microsoft", "apple", "amazon", "meta", "netflix",
        "twitter", "github", "gitlab", "slack", "zoom", "dropbox",
        "spotify", "uber", "lyft", "airbnb", "stripe", "shopify",
        "twilio", "okta", "cloudflare", "fastly", "akamai", "vercel",
        "heroku", "digitalocean", "linode", "vultr", "hetzner",
        # Crypto / fintech
        "coinbase", "binance", "kraken", "bitfinex", "gemini",
        "robinhood", "paypal", "venmo", "revolut", "wise",
        "plaid", "brex", "mercury", "ramp", "marqeta",
        # Security / enterprise
        "crowdstrike", "paloalto", "fortinet", "zscaler", "sentinelone",
        "qualys", "tenable", "rapid7", "bitsight", "recorded-future",
        "mandiant", "fireeye", "carbonblack", "cybereason", "darktrace",
        # E-commerce / retail
        "walmart", "target", "bestbuy", "costco", "ebay",
        "etsy", "wayfair", "aliexpress", "zalando", "mercadolibre",
        # Travel / hospitality
        "booking", "expedia", "tripadvisor", "marriott", "hilton",
        "united", "delta", "southwest", "ryanair", "lufthansa",
        # Media / entertainment
        "disney", "hulu", "paramount", "warner", "sony",
        "ea", "epic", "valve", "roblox", "twitch",
        # Healthcare / pharma
        "unitedhealth", "anthem", "cigna", "pfizer", "moderna",
        "johnson", "abbott", "medtronic", "bayer", "roche",
        # Telecom
        "verizon", "att", "tmobile", "vodafone", "telefonica",
        "orange", "dtag", "comcast", "charter", "bt",
        # Auto / manufacturing
        "tesla", "ford", "toyota", "bmw", "mercedes",
        "honda", "hyundai", "rivian", "lucid", "nio",
        # SaaS / productivity
        "salesforce", "hubspot", "zendesk", "atlassian", "notion",
        "figma", "canva", "miro", "asana", "monday",
        "datadog", "splunk", "elastic", "grafana", "prometheus",
        # Bug bounty specific real targets
        "vfsglobal", "dailymotion", "expressvpn", "pinelabs", "harman",
        "jbl", "gotdrops", "bostonacoustics", "kape", "nordvpn",
        "surfshark", "protonmail", "tutanota", "signal", "telegram",
        "whatsapp", "discord", "reddit", "imgur", "flickr",
        "soundcloud", "bandcamp", "medium", "substack", "ghost",
        # Infrastructure / hosting
        "aws", "gcp", "azure", "oracle", "ibm",
        "rackspace", "ovh", "scaleway", "upcloud", "kamatera",
        # Random but realistic company names
        "acme", "nexus", "vertex", "orbit", "forge",
        "apex", "prism", "flux", "cobalt", "zenith",
        "onyx", "cipher", "titan", "nebula", "nova",
        "atlas", "cortex", "nimbus", "cedar", "ember",
        "crest", "spark", "viper", "helix", "qubit",
        "sentry", "harbor", "relay", "beacon", "meridian",
        "pylon", "tensor", "lattice", "vector", "quantum",
        # Real sub-targets from OpenClaw recon (OOV in A/B test)
        "bostonacoustics", "xvtest", "jbl", "gotdrops", "kapetech",
        "cyberghostvpn", "intego", "webselenese", "zenmate",
        # More real bug bounty targets for diversity
        "hackerone", "bugcrowd", "intigriti", "synack", "yeswehack",
        "cobalt", "immunefi", "federacy", "securityscorecard",
        # Real subdomain patterns from recon data
        "autodiscover", "cpanel", "webmail", "remote", "vpn",
        "ftp", "dns", "ns1", "ns2", "mx", "pop", "imap", "smtp",
    ]
    _SUBDOM_PREFIXES = [
        "api", "app", "admin", "auth", "cdn", "dev", "docs",
        "gw", "internal", "mail", "portal", "staging", "stg",
        "test", "www", "beta", "dashboard", "login", "sso",
        "vault", "monitor", "grafana", "k8s", "ci", "git",
        "registry", "store", "shop", "blog", "support", "help",
        "status", "ws", "rpc", "grpc", "proxy", "edge",
    ]
    _PROGRAM_SUFFIXES = [
        "bug-bounty-program", "public-bug-bounty", "responsible-disclosure",
        "security-research-program", "vdp", "web-applications",
        "mobile-applications", "api-security-program",
        "bug-bounty", "security-vulnerability-disclosure",
        "coordinated-disclosure", "web-application-security",
        "infrastructure-security", "cloud-security-program",
        "external-security-research", "product-security",
    ]
    _RECON_TOOLS = [
        ("crt.sh", "hosts"), ("subfinder", "hosts"), ("VirusTotal", "hosts"),
        ("ArgosDNS", "hosts"), ("Profundis", "hosts"), ("Cybersixgill", "hosts"),
        ("SerpAPI", "URLs"), ("Amass", "hosts"), ("Shodan", "hosts"),
        ("SecurityTrails", "hosts"), ("DNSdumpster", "hosts"),
    ]
    _SCAN_TOOLS = [
        ("httpx", "live hosts"), ("nuclei", "findings"), ("nmap", "open ports"),
        ("masscan", "responsive hosts"), ("dirsearch", "paths"),
        ("ffuf", "endpoints"), ("nikto", "issues"), ("wappalyzer", "techs"),
    ]
    _HOST_CLASSES = ["Admin", "Auth", "API", "Static", "CDN", "Login", "Dashboard", "Monitoring"]
    _PHASE_NAMES = [
        "Phase 1A: Subdomain Collection",
        "Phase 1A: Merge + Validate",
        "Phase 1B Track A: Live Host Detection",
        "Phase 1B Track B: Takeover Check",
        "Phase 1B Track B: DNS Validation",
        "Phase 2: Port Scanning",
        "Phase 3: Service Fingerprinting",
        "Phase 4: Vulnerability Scanning",
        "Phase 5: Content Discovery",
        "Phase 6: Host Classification",
        "Phase 7: Deep Recon",
        "Phase 8: Manual Analysis",
    ]
    _STATUSES = ["COMPLETED", "IN PROGRESS", "BLOCKED", "REPAIR_NEEDED", "SKIPPED", "PARTIAL"]
    _ERRORS = [
        "502 Bad Gateway", "Connection timed out", "Rate limited by API",
        "SerpAPI quota exhausted", "API key expired", "DNS resolution failed",
        "TLS handshake error", "403 Forbidden", "Service unavailable",
        "Max retries exceeded", "Socket timeout after 30s", "Empty response",
        "Invalid JSON response", "Certificate verification failed",
    ]
    _FILE_PATHS = [
        "results/all_hosts_merged.txt", "results/live_hosts.txt",
        "results/httpx_output.json", "results/nuclei_findings.txt",
        "results/takeover_hits.txt", "results/valid_dns_hosts.txt",
        "results/invalid_dns_hosts.txt", "results/nmap_scan.xml",
        "results/dirsearch_output.txt", "results/admin_hosts.txt",
        "results/auth_hosts.txt", "results/api_hosts.txt",
        "results/static_hosts.txt",
    ]
    _OPERATIONAL_NOTES = [
        "Moving to next target from target_queue.json.",
        "All phases complete for this target.",
        "Debugging merge — duplicate entries found.",
        "Re-running missing sources after API recovery.",
        "Waiting for rate limit cooldown (60s).",
        "Restarting failed scan with increased timeout.",
        "Skipping target — out of scope per program policy.",
        "Pausing for manual review of findings.",
        "Resuming after session timeout recovery.",
        "Switching to backup API key.",
    ]

    # --- DevOps/pipeline operational journal vocabulary ---
    _DEPLOY_ENVS = ["production", "staging", "canary", "dev", "qa", "perf-test", "uat"]
    _SERVICES = [
        "api-gateway", "auth-service", "user-service", "billing-service",
        "notification-service", "search-indexer", "cdn-worker", "cache-layer",
        "scheduler", "event-bus", "ingestion-pipeline", "ml-inference",
        "analytics-collector", "log-aggregator", "config-server",
    ]
    _PIPELINE_STEPS = [
        "Build", "Unit Tests", "Integration Tests", "Lint & Format",
        "Docker Build", "Push to Registry", "Deploy to Staging",
        "Smoke Tests", "Deploy to Production", "Health Check",
        "Rollback Check", "Canary Analysis", "Traffic Shift",
    ]
    _METRICS = [
        ("p99 latency", "ms"), ("error rate", "%"), ("throughput", "rps"),
        ("CPU usage", "%"), ("memory usage", "MB"), ("disk usage", "GB"),
        ("pod count", "pods"), ("queue depth", "messages"),
        ("cache hit rate", "%"), ("connection pool", "active"),
    ]
    _INCIDENT_TYPES = [
        "elevated error rates", "memory leak detected", "certificate expiring",
        "disk pressure warning", "pod crash loop", "connection pool exhaustion",
        "upstream timeout spike", "DNS resolution failures", "TLS cert mismatch",
        "config drift detected", "secret rotation needed",
    ]

    def _rand_domain(rng):
        sld = rng.choice(_SLD_PARTS)
        tld = rng.choice(_TLDS)
        r = rng.random()
        if r < 0.15:
            # Compound: prefix.company.tld (e.g. de.jbl.com, api.blts.kape.com)
            prefix = rng.choice(["de", "fr", "jp", "br", "uk", "au", "in",
                                 "api", "pro", "app", "dev", "it", "us", "eu"])
            return f"{prefix}.{sld}.{tld}"
        elif r < 0.25:
            # Two-part SLD: companyproduct.tld
            sld2 = rng.choice(_SLD_PARTS)
            return f"{sld}{sld2}.{tld}"
        elif r < 0.35:
            # Hyphenated: company-product.tld
            sld2 = rng.choice(["labs", "tech", "cloud", "data", "app",
                               "hub", "link", "pay", "id", "io", "ops",
                               "sec", "net", "web", "api", "global"])
            return f"{sld}-{sld2}.{tld}"
        return f"{sld}.{tld}"

    def _rand_program(rng, domain):
        name = domain.split(".")[0]
        suffix = rng.choice(_PROGRAM_SUFFIXES)
        return f"{name}-{suffix}"

    def _rand_ip(rng):
        return f"{rng.randint(1,254)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"

    def _rand_cve(rng):
        year = rng.randint(2018, 2025)
        num = rng.randint(1000, 49999)
        return f"CVE-{year}-{num}"

    _COMMON_PORTS = [
        21, 22, 25, 53, 80, 110, 143, 443, 445, 993, 995, 1433, 1521,
        2049, 3000, 3306, 3389, 5432, 5900, 6379, 8000, 8080, 8443,
        8888, 9090, 9200, 9300, 27017, 50000,
    ]

    _VULN_FINDINGS = [
        "XSS in search parameter", "IDOR on user profile endpoint",
        "Open redirect via next parameter", "SSRF via webhook URL",
        "SQL injection in filter query", "CSRF on password change",
        "Information disclosure via stack trace", "Broken access control on admin API",
        "JWT signature not verified", "Rate limit bypass via header manipulation",
        "Subdomain takeover — dangling CNAME", "Exposed .git directory",
        "S3 bucket misconfiguration", "GraphQL introspection enabled",
        "CORS misconfiguration allowing credential theft",
        "Host header injection", "Path traversal in file upload",
        "Insecure deserialization", "XXE in XML parser",
        "Exposed debug endpoint", "API key in client-side JavaScript",
        "Clickjacking — missing X-Frame-Options",
        "Sensitive data in URL parameters", "Weak password policy",
    ]

    def _gen_recon_journal(rng):
        """Full recon journal for one target — the primary pattern."""
        domain = _rand_domain(rng)
        program = _rand_program(rng, domain)
        n_phases = rng.randint(3, 7)
        phases = rng.sample(_PHASE_NAMES, min(n_phases, len(_PHASE_NAMES)))
        phases.sort(key=lambda p: _PHASE_NAMES.index(p))

        lines = [f"# JOURNAL.md - {program}", ""]

        for phase in phases:
            status = rng.choice(_STATUSES[:3] + ["COMPLETED"] * 4)
            tool_list = ""
            if "Subdomain" in phase:
                n_tools = rng.randint(4, 8)
                tools = rng.sample(_RECON_TOOLS, min(n_tools, len(_RECON_TOOLS)))
                tool_names = ", ".join(t[0] for t in tools)
                tool_list = f" ({tool_names})"

            lines.append(f"## {phase}{tool_list} - {status}")
            lines.append("")

            if "Subdomain" in phase:
                n_sub_domains = rng.randint(1, 3)
                sub_domains = [_rand_domain(rng) for _ in range(n_sub_domains)]
                if rng.random() < 0.4:
                    lines.append(f"- Workspace created for {program}.")
                    lines.append(f"- RECON.md and JOURNAL.md initialized.")
                for sd in sub_domains:
                    if rng.random() < 0.3:
                        lines.append(f"- Starting subdomain collection for {sd}.")
                    for tool, unit in tools:
                        count = rng.choice([0, 0, 0, rng.randint(1, 50),
                                           rng.randint(50, 500), rng.randint(500, 3000)])
                        if rng.random() < 0.08:
                            err = rng.choice(_ERRORS)
                            lines.append(f"- {tool} for {sd}: Error - {err}")
                        elif rng.random() < 0.3:
                            lines.append(f"- Ran {tool} for {sd}. Found {count} {unit}.")
                        else:
                            lines.append(f"- {tool}: {count} {unit}")

                total = rng.randint(100, 5000)
                if rng.random() < 0.5:
                    lines.append(f"- Merged all found hosts into `{program}/results/all_hosts_merged.txt`.")
                lines.append(f"- Total merged: {total} hosts")

            elif "Live Host" in phase or "httpx" in phase.lower():
                live = rng.randint(10, 500)
                lines.append(f"- Ran httpx on the merged list of hosts.")
                lines.append(f"- Live hosts (httpx): {live} hosts")
                if rng.random() < 0.5:
                    lines.append(f"- Results saved to `results/httpx_output.json` and `results/live_hosts.txt`.")

            elif "Takeover" in phase:
                hits = rng.choice([0] * 8 + [1, 2, 3])
                lines.append(f"- Ran nuclei with takeover templates on all_hosts_merged.txt.")
                lines.append(f"- Takeover hits: {hits}")
                if hits > 0:
                    for _ in range(hits):
                        sub = f"{rng.choice(_SUBDOM_PREFIXES)}.{domain}"
                        lines.append(f"  - CNAME dangling: {sub}")

            elif "DNS Validation" in phase:
                lines.append(f"- Performed DNS validation on all merged hosts.")
                lines.append(f"- Valid hosts saved to `results/valid_dns_hosts.txt`.")

            elif "Classification" in phase or "Classify" in phase:
                classes = rng.sample(_HOST_CLASSES, rng.randint(3, 6))
                counts = [rng.randint(0, 80) for _ in classes]
                class_str = ", ".join(f"{c} hosts: {n}" for c, n in zip(classes, counts))
                lines.append(f"- Classified live hosts: {class_str}.")
                lines.append(f"- Results saved to {', '.join(f'{c.lower()}_hosts.txt' for c in classes)}.")

            elif "Scanning" in phase or "Fingerprint" in phase or "Discovery" in phase:
                tool, unit = rng.choice(_SCAN_TOOLS)
                count = rng.randint(0, 200)
                lines.append(f"- Ran {tool} on live hosts. Found {count} {unit}.")
                if rng.random() < 0.5:
                    fp = rng.choice(_FILE_PATHS)
                    lines.append(f"- Results saved to `{fp}`.")
                if rng.random() < 0.35:
                    n_ports = rng.randint(1, 4)
                    ports = rng.sample(_COMMON_PORTS, min(n_ports, len(_COMMON_PORTS)))
                    ip = _rand_ip(rng)
                    lines.append(f"- Notable: {ip} open ports {', '.join(str(p) for p in ports)}")
                if rng.random() < 0.25:
                    cve = _rand_cve(rng)
                    finding = rng.choice(_VULN_FINDINGS)
                    sub = f"{rng.choice(_SUBDOM_PREFIXES)}.{domain}"
                    lines.append(f"- Finding: {finding} on {sub} ({cve})")
            else:
                lines.append(f"- {rng.choice(_OPERATIONAL_NOTES)}")

            if rng.random() < 0.12:
                err = rng.choice(_ERRORS)
                lines.append(f"- **Error**: {err}. Retrying...")
            lines.append("")

        if rng.random() < 0.4:
            lines.append("## Next Steps")
            lines.append(f"- {rng.choice(_OPERATIONAL_NOTES)}")

        return "\n".join(lines).strip()

    def _gen_compact_recon(rng):
        """Short-form recon summary (like vfsglobal style)."""
        domain = _rand_domain(rng)
        n_tools = rng.randint(5, 8)
        tools = rng.sample(_RECON_TOOLS, min(n_tools, len(_RECON_TOOLS)))

        lines = [f"## Recon for {domain}",
                 "### Phase 1A: Subdomain Collection"]
        for tool, unit in tools:
            count = rng.choice([0, 0, rng.randint(1, 50), rng.randint(50, 500),
                               rng.randint(500, 2000)])
            lines.append(f"- {tool}: {count} {unit}")
        total = rng.randint(100, 3000)
        lines.append(f"- Total merged: {total} hosts")

        lines.append("### Phase 1B: Infrastructure Analysis")
        lines.append(f"- Takeover hits: {rng.choice([0] * 6 + [1, 2])}")
        live = rng.randint(10, 500)
        lines.append(f"- Live hosts (httpx): {live} hosts")
        for cls in rng.sample(_HOST_CLASSES[:4], rng.randint(3, 4)):
            lines.append(f"- {cls} hosts: {rng.randint(0, 100)}")

        if rng.random() < 0.4:
            lines.append("### Findings")
            n_findings = rng.randint(1, 3)
            for _ in range(n_findings):
                finding = rng.choice(_VULN_FINDINGS)
                sub = f"{rng.choice(_SUBDOM_PREFIXES)}.{domain}"
                if rng.random() < 0.4:
                    cve = _rand_cve(rng)
                    lines.append(f"- {finding} on {sub} ({cve})")
                else:
                    lines.append(f"- {finding} on {sub}")

        return "\n".join(lines)

    def _gen_deploy_journal(rng):
        """Deployment / CI-CD operational journal."""
        service = rng.choice(_SERVICES)
        env = rng.choice(_DEPLOY_ENVS)
        version = f"v{rng.randint(1, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 99)}"

        lines = [f"# Deploy Journal - {service} {version} → {env}", ""]
        n_steps = rng.randint(4, 8)
        steps = _PIPELINE_STEPS[:n_steps]

        for step in steps:
            status = rng.choice(["PASSED"] * 7 + ["FAILED", "SKIPPED", "RETRIED"])
            dur = f"{rng.randint(1, 300)}s"
            lines.append(f"## {step} - {status} ({dur})")
            if status == "FAILED":
                lines.append(f"- Error: {rng.choice(_ERRORS)}")
                if rng.random() < 0.5:
                    lines.append(f"- Retrying with increased timeout...")
            elif step == "Health Check":
                n_checks = rng.randint(2, 5)
                for _ in range(n_checks):
                    metric, unit = rng.choice(_METRICS)
                    val = rng.randint(1, 999)
                    lines.append(f"- {metric}: {val} {unit}")
            elif "Deploy" in step:
                pod_count = rng.randint(2, 20)
                lines.append(f"- Replicas: {pod_count} pods rolling")
                lines.append(f"- Image: registry.internal/{service}:{version}")
            elif "Test" in step:
                passed = rng.randint(50, 500)
                failed = rng.choice([0] * 5 + [rng.randint(1, 10)])
                lines.append(f"- {passed} passed, {failed} failed")
            lines.append("")

        if rng.random() < 0.3:
            lines.append("## Post-Deploy Metrics (5 min)")
            for _ in range(rng.randint(2, 4)):
                metric, unit = rng.choice(_METRICS)
                val = rng.randint(1, 999)
                lines.append(f"- {metric}: {val} {unit}")

        return "\n".join(lines).strip()

    def _gen_incident_journal(rng):
        """Incident investigation / response journal."""
        incident_type = rng.choice(_INCIDENT_TYPES)
        service = rng.choice(_SERVICES)
        env = rng.choice(_DEPLOY_ENVS[:3])
        sev = rng.choice(["SEV-1", "SEV-2", "SEV-3", "P1", "P2"])

        lines = [f"# Incident: {incident_type} in {service}", ""]
        lines.append(f"**Severity**: {sev}")
        lines.append(f"**Environment**: {env}")
        lines.append(f"**Status**: {rng.choice(['Investigating', 'Mitigated', 'Resolved'])}")
        lines.append("")

        n_updates = rng.randint(3, 7)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        for i in range(n_updates):
            minute += rng.randint(3, 25)
            if minute >= 60:
                hour += 1
                minute -= 60
            ts = f"{hour:02d}:{minute:02d}"
            lines.append(f"### Update {i + 1} — {ts} UTC")

            actions = rng.randint(1, 3)
            for _ in range(actions):
                action_type = rng.choice(["check", "metric", "action", "finding"])
                if action_type == "metric":
                    metric, unit = rng.choice(_METRICS)
                    val = rng.randint(1, 999)
                    lines.append(f"- {metric}: {val} {unit}")
                elif action_type == "check":
                    lines.append(f"- Checked {rng.choice(_SERVICES)} — {rng.choice(['healthy', 'degraded', 'unreachable'])}")
                elif action_type == "finding":
                    lines.append(f"- Found: {rng.choice(_INCIDENT_TYPES)}")
                else:
                    lines.append(f"- {rng.choice(['Restarted', 'Scaled up', 'Drained', 'Rolled back', 'Patched'])} {rng.choice(_SERVICES)}")
            lines.append("")

        if rng.random() < 0.5:
            lines.append("## Root Cause")
            lines.append(f"- {rng.choice(_INCIDENT_TYPES)} triggered by {rng.choice(['config change', 'upstream dependency failure', 'resource exhaustion', 'bad deploy', 'DNS propagation delay'])}.")

        return "\n".join(lines).strip()

    def _gen_pipeline_journal(rng):
        """Data pipeline / ETL run journal."""
        pipeline = rng.choice(["daily-ingest", "hourly-sync", "batch-transform",
                               "export-job", "reindex-task", "cleanup-sweep",
                               "backup-snapshot", "migration-run"])
        run_id = f"run-{rng.randint(1000, 9999)}"

        lines = [f"# Pipeline: {pipeline} ({run_id})", ""]

        n_stages = rng.randint(3, 6)
        stage_names = rng.sample([
            "Extract", "Validate", "Transform", "Deduplicate", "Load",
            "Index", "Checkpoint", "Notify", "Cleanup", "Archive",
        ], min(n_stages, 10))

        for stage in stage_names:
            dur = rng.randint(1, 600)
            status = rng.choice(["OK"] * 6 + ["WARN", "ERROR", "SKIPPED"])
            lines.append(f"## {stage} - {status} ({dur}s)")
            rows = rng.randint(100, 10_000_000)
            lines.append(f"- Rows processed: {rows:,}")
            if status == "WARN":
                lines.append(f"- Warning: {rng.randint(1, 50)} rows failed validation")
            elif status == "ERROR":
                lines.append(f"- Error: {rng.choice(_ERRORS)}")
            if rng.random() < 0.3:
                lines.append(f"- Output: s3://data-lake/{pipeline}/{run_id}/{stage.lower()}.parquet")
            lines.append("")

        lines.append("## Summary")
        total_dur = rng.randint(60, 7200)
        lines.append(f"- Total duration: {total_dur}s")
        lines.append(f"- Final status: {rng.choice(['SUCCESS', 'PARTIAL', 'FAILED'])}")

        return "\n".join(lines).strip()

    def _gen_multi_target_session(rng):
        """Multi-target session journal — interleaved entries."""
        n_targets = rng.randint(2, 4)
        domains = [_rand_domain(rng) for _ in range(n_targets)]
        programs = [_rand_program(rng, d) for d in domains]

        lines = [f"# Session Journal — {n_targets} targets", ""]

        for i, (domain, program) in enumerate(zip(domains, programs)):
            lines.append(f"## Target {i + 1}: {program} ({domain})")
            n_tools = rng.randint(4, 7)
            tools = rng.sample(_RECON_TOOLS, min(n_tools, len(_RECON_TOOLS)))
            for tool, unit in tools:
                count = rng.choice([0, 0, rng.randint(1, 50),
                                   rng.randint(50, 500), rng.randint(500, 2000)])
                lines.append(f"- {tool}: {count} {unit}")
            total = rng.randint(50, 3000)
            lines.append(f"- Total merged: {total} hosts")

            if rng.random() < 0.6:
                live = rng.randint(5, 300)
                lines.append(f"- Live hosts: {live}")
                classes = rng.sample(_HOST_CLASSES[:4], rng.randint(2, 4))
                counts = [rng.randint(0, 50) for _ in classes]
                class_str = ", ".join(f"{c}: {n}" for c, n in zip(classes, counts))
                lines.append(f"- Classification: {class_str}")

            if rng.random() < 0.3:
                finding = rng.choice(_VULN_FINDINGS)
                sub = f"{rng.choice(_SUBDOM_PREFIXES)}.{domain}"
                lines.append(f"- Finding: {finding} on {sub}")
            if rng.random() < 0.2:
                ip = _rand_ip(rng)
                port = rng.choice(_COMMON_PORTS)
                lines.append(f"- Notable host: {ip}:{port}")

            status = rng.choice(_STATUSES[:3] + ["COMPLETED"] * 3)
            lines.append(f"- Status: {status}")
            if status == "BLOCKED":
                lines.append(f"- Blocker: {rng.choice(_ERRORS)}")
            lines.append("")

        lines.append("## Session Summary")
        lines.append(f"- Targets processed: {n_targets}")
        completed = rng.randint(1, n_targets)
        lines.append(f"- Completed: {completed}/{n_targets}")
        if completed < n_targets:
            lines.append(f"- {rng.choice(_OPERATIONAL_NOTES)}")

        return "\n".join(lines).strip()

    # --- Load real OpenClaw journal data for mixing ---
    real_journals = []
    _REAL_JOURNAL_PATHS = [
        "ab_data/journal_vfsglobal.md",
        "ab_data/journal_dailymotion.md",
        "ab_data/journal_expressvpn.md",
        "ab_data/journal_harman.md",
        "ab_data/journal_pinelabs.md",
    ]
    for jp in _REAL_JOURNAL_PATHS:
        p = Path(jp)
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                real_journals.append(text)
                # Also split into individual sections for chunk-level diversity
                sections = [s.strip() for s in text.split("\n## ") if s.strip()]
                for sec in sections:
                    if len(sec) >= cfg.min_text_chars:
                        real_journals.append("## " + sec if not sec.startswith("#") else sec)
    real_count = len(real_journals)
    print(f"  Real journal chunks loaded: {real_count}")

    # --- Generate synthetic samples ---
    synth_target = max_samples - int(max_samples * 0.05)  # reserve 5% for real data
    samples = []
    weights = [
        ("recon_full", 0.30),
        ("recon_compact", 0.20),
        ("multi_target", 0.15),
        ("deploy", 0.15),
        ("incident", 0.10),
        ("pipeline", 0.10),
    ]
    cum_weights = []
    total = 0
    for _, w in weights:
        total += w
        cum_weights.append(total)

    for _ in range(synth_target):
        r = _rng.random()
        if r < cum_weights[0]:
            samples.append(_gen_recon_journal(_rng))
        elif r < cum_weights[1]:
            samples.append(_gen_compact_recon(_rng))
        elif r < cum_weights[2]:
            samples.append(_gen_multi_target_session(_rng))
        elif r < cum_weights[3]:
            samples.append(_gen_deploy_journal(_rng))
        elif r < cum_weights[4]:
            samples.append(_gen_incident_journal(_rng))
        else:
            samples.append(_gen_pipeline_journal(_rng))

    # --- Mix in real data (5% of corpus, repeated/windowed to fill) ---
    if real_journals:
        real_target = max_samples - len(samples)
        real_mixed = []
        while len(real_mixed) < real_target:
            real_mixed.append(_rng.choice(real_journals))
        samples.extend(real_mixed[:real_target])
        print(f"  Real data mixed in:      {real_target:>12,} samples ({real_target/max_samples*100:.1f}%)")

    _rng.shuffle(samples)
    samples = samples[:max_samples]

    print(f"  Samples generated:       {len(samples):>12,}")
    if samples:
        print(f"  Avg length (chars):      {sum(len(s) for s in samples) // len(samples):>12,}")
    print(f"  Entity diversity:        {len(_SLD_PARTS):>12} SLDs x {len(_TLDS)} TLDs")
    print(f"  Type distribution (synth):")
    print(f"    recon_full:    ~{int(synth_target * 0.30):>8,}")
    print(f"    recon_compact: ~{int(synth_target * 0.20):>8,}")
    print(f"    multi_target:  ~{int(synth_target * 0.15):>8,}")
    print(f"    deploy:        ~{int(synth_target * 0.15):>8,}")
    print(f"    incident:      ~{int(synth_target * 0.10):>8,}")
    print(f"    pipeline:      ~{int(synth_target * 0.10):>8,}")
    print(f"    real_journals: ~{max_samples - synth_target:>8,}")
    print(f"  =========================================")
    return Dataset.from_dict({"text": samples[:max_samples]})


def _load_state_traces(cfg: CNDXConfig, max_samples: int, split: str = "train"):
    """Generate synthetic operational-state event traces.

    Domain: Operational State Artifacts — structured event/update traces.
    Unlike wiki (static factual prose) or code (static source artifacts),
    this domain tests sequential updates, state continuity, event ordering,
    value overwrites, and final-state recovery.

    Training source label: synthetic_state_traces_v1
    """
    import random as _rng

    seed_offset = 0 if split == "train" else 999999
    _rng.seed(42 + seed_offset)

    print(f"[source = synthetic_state_traces_v1 ({split})]")
    print(f"  Generating {max_samples:,} operational-state event traces ...")

    _ENTITIES = [
        "server-01", "server-02", "server-03", "server-04", "server-05",
        "node-alpha", "node-beta", "node-gamma", "node-delta",
        "worker-A", "worker-B", "worker-C", "worker-D",
        "sensor-north", "sensor-south", "sensor-east", "sensor-west",
        "pipeline-main", "pipeline-backup", "pipeline-staging",
        "container-web", "container-db", "container-cache", "container-queue",
    ]
    _STATUSES = ["idle", "active", "busy", "degraded", "offline", "standby",
                 "draining", "recovering", "pending", "running", "completed", "failed"]
    _LOCATIONS = ["rack-1", "rack-2", "rack-3", "zone-us-east", "zone-eu-west",
                  "zone-ap-south", "bay-A", "bay-B", "bay-C", "datacenter-primary",
                  "datacenter-backup", "region-north", "region-south"]
    _FIELDS = ["status", "load", "temperature", "memory_pct", "queue_depth",
               "error_count", "latency_ms", "throughput", "connections", "cpu_pct"]
    _ACTIONS = ["migrated to", "assigned to", "moved to", "relocated to",
                "transferred to", "deployed at", "switched to"]
    _FILLER_EVENTS = [
        "system health check passed",
        "routine maintenance window opened",
        "backup snapshot completed",
        "log rotation executed",
        "heartbeat acknowledged",
        "configuration reloaded",
        "certificate renewed",
        "DNS cache flushed",
        "connection pool recycled",
        "metrics aggregation cycle finished",
        "audit log rotated",
        "garbage collection completed",
    ]

    traces = []
    for _ in range(max_samples):
        trace_type = _rng.choice(["entity_state", "object_location",
                                   "counter_updates", "mixed_workflow"])

        if trace_type == "entity_state":
            traces.append(_gen_entity_state_trace(_rng, _ENTITIES, _STATUSES, _FIELDS, _FILLER_EVENTS))
        elif trace_type == "object_location":
            traces.append(_gen_location_trace(_rng, _ENTITIES, _LOCATIONS, _ACTIONS, _FILLER_EVENTS))
        elif trace_type == "counter_updates":
            traces.append(_gen_counter_trace(_rng, _ENTITIES, _FIELDS, _FILLER_EVENTS))
        else:
            traces.append(_gen_mixed_trace(_rng, _ENTITIES, _STATUSES, _LOCATIONS,
                                           _FIELDS, _ACTIONS, _FILLER_EVENTS))

    print(f"  Traces generated:        {len(traces):>12,}")
    print(f"  Avg length (chars):      {sum(len(t) for t in traces) // len(traces):>12,}")
    print(f"  =========================================")
    return Dataset.from_dict({"text": traces[:max_samples]})


def _gen_entity_state_trace(rng, entities, statuses, fields, fillers):
    """Entity goes through 4-7 state changes with distractors."""
    entity = rng.choice(entities)
    n_updates = rng.randint(4, 7)
    n_distractors = rng.randint(1, 3)
    lines = []
    t = 1
    for i in range(n_updates + n_distractors):
        if i < n_updates:
            field = rng.choice(["status"] * 3 + fields)
            if field == "status":
                val = rng.choice(statuses)
            else:
                val = str(rng.randint(1, 99))
            lines.append(f"[T={t}] {entity} {field}={val}")
        else:
            lines.append(f"[T={t}] {rng.choice(fillers)}")
        t += rng.randint(1, 3)
    rng.shuffle(lines)
    lines.sort(key=lambda x: int(x.split("]")[0].split("=")[1]))
    return "\n".join(lines)


def _gen_location_trace(rng, entities, locations, actions, fillers):
    """Object moves through 3-6 locations with distractors."""
    entity = rng.choice(entities)
    n_moves = rng.randint(3, 6)
    locs = [rng.choice(locations) for _ in range(n_moves)]
    lines = []
    t = 1
    for i, loc in enumerate(locs):
        action = rng.choice(actions)
        lines.append(f"[T={t}] {entity} {action} {loc}")
        if rng.random() < 0.4:
            other = rng.choice([e for e in entities if e != entity])
            lines.append(f"[T={t+1}] {other} {rng.choice(actions)} {rng.choice(locations)}")
            t += 1
        t += rng.randint(1, 3)
    if rng.random() < 0.5:
        lines.insert(rng.randint(0, len(lines)), f"[T={t}] {rng.choice(fillers)}")
    return "\n".join(lines)


def _gen_counter_trace(rng, entities, fields, fillers):
    """Numeric field updated 5-8 times with overwrites."""
    entity = rng.choice(entities)
    field = rng.choice([f for f in fields if f not in ("status",)])
    n_updates = rng.randint(5, 8)
    lines = []
    t = 1
    for _ in range(n_updates):
        val = rng.randint(0, 100)
        lines.append(f"[T={t}] {entity} {field}={val}")
        if rng.random() < 0.3:
            lines.append(f"[T={t}] {rng.choice(fillers)}")
        t += rng.randint(1, 2)
    return "\n".join(lines)


def _gen_mixed_trace(rng, entities, statuses, locations, fields, actions, fillers):
    """Mixed: 2-3 entities with interleaved state/location/counter updates."""
    n_entities = rng.randint(2, 3)
    chosen = rng.sample(entities, n_entities)
    lines = []
    t = 1
    for step in range(rng.randint(6, 10)):
        ent = rng.choice(chosen)
        event_type = rng.choice(["status", "location", "counter", "filler"])
        if event_type == "status":
            lines.append(f"[T={t}] {ent} status={rng.choice(statuses)}")
        elif event_type == "location":
            lines.append(f"[T={t}] {ent} {rng.choice(actions)} {rng.choice(locations)}")
        elif event_type == "counter":
            lines.append(f"[T={t}] {ent} {rng.choice(fields)}={rng.randint(0, 100)}")
        else:
            lines.append(f"[T={t}] {rng.choice(fillers)}")
        t += rng.randint(1, 3)
    return "\n".join(lines)


def _load_split(tokenizer, cfg: CNDXConfig, split: str, max_samples: int,
                dataset_override: str = None):
    ds_name = dataset_override or cfg.dataset_name
    ds_cfg = DATASET_CONFIGS.get(ds_name, {"text_col": "text", "subset": None})
    text_col = ds_cfg["text_col"]

    if ds_cfg.get("aoj_traces"):
        ds = _load_aoj_traces(cfg, max_samples, split=split)
    elif ds_cfg.get("conversation_mem"):
        ds = _load_conversation_mem(cfg, max_samples, split=split)
    elif ds_cfg.get("regulated_text"):
        ds = _load_regulated_text(cfg, max_samples, split=split)
    elif ds_cfg.get("hwm_traces"):
        ds = _load_hwm_traces(cfg, max_samples, split=split)
    elif ds_cfg.get("state_traces"):
        ds = _load_state_traces(cfg, max_samples, split=split)
    elif ds_cfg.get("security_reports"):
        ds = _load_security_reports(cfg, max_samples, split=split)
    elif ds_cfg.get("code_functions"):
        ds = _load_code_functions(cfg, max_samples, split=split)
    elif ds_cfg.get("paragraph_chunk") and split == "train":
        ds = _load_wikipedia_paragraphs(cfg, max_samples)
    else:
        subset = ds_cfg["subset"]
        if subset:
            ds = load_dataset(ds_name, subset, split=split)
        else:
            ds = load_dataset(ds_name, split=split)

        if cfg.min_text_chars > 0:
            ds = ds.filter(lambda x: len(x[text_col].strip()) >= cfg.min_text_chars)

        ds = ds.select(range(min(max_samples, len(ds))))

    def tok_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=cfg.seq_len,
            padding="max_length",
        )

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds


def build_dataloaders(tokenizer, cfg: CNDXConfig):
    train_ds = _load_split(tokenizer, cfg, "train", cfg.max_train_samples)

    eval_ds_name = cfg.eval_dataset_name if cfg.eval_dataset_name else None
    val_ds = _load_split(tokenizer, cfg, "validation", cfg.max_eval_samples,
                         dataset_override=eval_ds_name)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader
