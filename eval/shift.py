"""Deterministic distribution-shift generators for the eval harness.

Four pure functions move a state sideways without changing what it means:

* ``shuffle_keys``    -- same keys, same values, different insertion order;
* ``structure_noise`` -- different key composition (rename / add / delete);
* ``new_keys``        -- strict superset of the original keys;
* ``paraphrase``      -- same keys and nesting, reworded string values.

Two stage-2 risks shape the design. R3: there is no LLM behind these
"paraphrases" -- every rewrite is a fixed template (a synonym table plus a
seeded variant choice, or a seeded rephrasing wrapper), so wording quality is a
documented limitation of the report, not a defect of the code. R5: every
generator takes ``seed`` and draws from its own
``random.Random(f"<name>:{seed}")`` stream instead of the global ``random``
state, so one seed reproduces the same output on CI and on a second machine;
seeding with the name:seed string (not ``hash()``) keeps that independent of
``PYTHONHASHSEED``.

Outputs stay inside ``StateValue`` (``str | int | float | bool | None | dict |
list``, ``open_jev/main.py:239``): string leaves may be reworded, non-string
leaves are never touched, no input is mutated -- every function returns a fresh
structure.
"""

from __future__ import annotations

import random
import re

__all__ = ["new_keys", "paraphrase", "shuffle_keys", "structure_noise"]

# Seeded noise templates (risk R3): renames, inserted keys and rephrasing
# wrappers all come from fixed tables picked by seed -- never from a model.
_RENAME = ("{key}_alt", "{key}_v2", "alt_{key}", "{key}_renamed")
_ADDED = (
    "escalation_channel",
    "first_response_minutes",
    "sentiment_score",
    "region",
    "queue_name",
)
_PLACEHOLDER = ("unknown", "n/a", "unspecified")
_REPHRASE = (
    "the record says: {v}",
    "{v} (as recorded)",
    "according to the log: {v}",
    "noted down as: {v}",
    "on file: {v}",
)

# Whole-word synonym table for ``paraphrase`` (risk R3). Keys are lowercase;
# no list ever contains its own key, so a match always changes the string.
_SYNONYMS = {
    "customer": ("client", "account holder", "buyer"),
    "customers": ("clients", "buyers"),
    "client": ("customer", "buyer"),
    "clients": ("customers", "buyers"),
    "order": ("purchase", "transaction"),
    "orders": ("purchases", "transactions"),
    "purchase": ("order", "booking"),
    "transaction": ("charge", "payment"),
    "transactions": ("charges", "payments"),
    "charge": ("payment", "billing entry"),
    "charges": ("payments", "billing entries"),
    "payment": ("charge", "settlement"),
    "refund": ("reimbursement", "chargeback"),
    "refunds": ("reimbursements", "chargebacks"),
    "refundable": ("reimbursable", "reversible"),
    "status": ("state", "condition"),
    "state": ("status", "situation"),
    "issue": ("problem", "incident"),
    "issues": ("problems", "incidents"),
    "problem": ("issue", "defect"),
    "error": ("failure", "fault"),
    "failure": ("error", "outage"),
    "message": ("note", "communication"),
    "messages": ("notes", "communications"),
    "note": ("remark", "message"),
    "policy": ("rule", "guideline"),
    "policies": ("rules", "guidelines"),
    "rule": ("policy", "standard"),
    "priority": ("urgency", "importance"),
    "priorities": ("urgency levels", "ranks"),
    "tier": ("level", "grade"),
    "tiers": ("levels", "grades"),
    "level": ("tier", "band"),
    "enterprise": ("business", "corporate"),
    "agent": ("representative", "operator"),
    "agents": ("representatives", "operators"),
    "representative": ("agent", "specialist"),
    "team": ("group", "unit"),
    "teams": ("groups", "units"),
    "support": ("help", "assistance"),
    "open": ("unresolved", "active"),
    "closed": ("resolved", "completed"),
    "resolved": ("closed", "settled"),
    "pending": ("awaiting", "in progress"),
    "captured": ("settled", "posted"),
    "duplicate": ("repeated", "double"),
    "duplicates": ("repeats", "copies"),
    "request": ("appeal", "demand"),
    "requests": ("appeals", "demands"),
    "question": ("query", "inquiry"),
    "questions": ("queries", "inquiries"),
    "account": ("profile", "record"),
    "user": ("account holder", "operator"),
    "amount": ("sum", "value"),
    "value": ("amount", "figure"),
    "name": ("title", "label"),
    "id": ("identifier", "reference"),
    "region": ("zone", "area"),
    "sentiment": ("tone", "mood"),
    "frustrated": ("annoyed", "irritated"),
    "fix": ("resolve", "repair"),
}

# Longest-first so a key can never shadow its own superstring in the
# alternation (e.g. "status" tried before "state" is irrelevant thanks to \b,
# but ordering keeps the pattern predictable across table edits).
_WORD = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_SYNONYMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _rng(name: str, seed: int) -> random.Random:
    """Private RNG stream per generator, so calls cannot perturb each other.

    The name:seed string is hashed by ``random`` with its version-2 scheme
    (content-based, not ``hash()``), which is what keeps a fixed seed stable
    across processes and PYTHONHASHSEED values.
    """
    return random.Random(f"{name}:{seed}")


def _fresh_key(base: str, used: set) -> str:
    """First name derived from ``base`` that is not in ``used``.

    Renames and insertions must not collide with an existing key: a collision
    would collapse into it and silently leave the key set unchanged (or, for
    ``new_keys``, shrink the "superset" back into a replacement).
    """
    name, i = base, 2
    while name in used:
        name, i = f"{base}_{i}", i + 1
    return name


def shuffle_keys(s: dict, seed: int) -> dict:
    """Same dict, same keys and values, new insertion order -- order only.

    Args:
        s: state to permute. Nested dicts are permuted too; lists keep their
           order, because position inside a list is meaning while position
           among dict keys is not.

    Returns:
        A fresh dict with ``out == s`` (dict equality ignores order) but a
        different serialization order. The one permutation a seed could
        produce that shifts nothing -- the identity -- is rotated by one, so
        any level with >= 2 keys always comes out reordered. Reordering is
        invisible to ``flatten_state``'s positional encoding
        (``open_jev/main.py:255-258``), which is exactly why this generator is
        a consistency probe rather than a semantic change.
    """
    return _shuffle(s, _rng("shuffle_keys", seed))


def _shuffle(value: object, rng: random.Random) -> object:
    if isinstance(value, dict):
        keys = list(value)
        rng.shuffle(keys)
        if keys == list(value) and len(keys) > 1:
            keys = keys[1:] + keys[:1]
        return {k: _shuffle(value[k], rng) for k in keys}
    if isinstance(value, list):
        return [_shuffle(v, rng) for v in value]
    return value


def structure_noise(s: dict, seed: int) -> dict:
    """Change the *composition* of keys with one seeded noise template.

    Exactly one operation is applied, picked by ``seed``: rename a key to a
    fresh template name (value travels with it), delete a key, or insert a key
    carrying a placeholder value. Whatever the seed picks,
    ``set(out) != set(s)`` holds -- renames and insertions always go through
    ``_fresh_key`` and therefore cannot collapse back into an existing key --
    and an empty state only ever gets the insert operation (deleting from it
    is impossible, renaming from it has no source). Same seed, same output;
    surviving values are copied untouched.

    Args:
        s: state to disturb; only the top-level key set changes, nested
           containers ride along as-is so the result stays a valid StateValue.

    Returns:
        A fresh dict differing from ``s`` in its key set.
    """
    rng = _rng("structure_noise", seed)
    ops = ("rename", "add", "delete") if s else ("add",)
    op = rng.choice(ops)
    if op == "delete":
        out = dict(s)
        out.pop(rng.choice(list(out)))
        return out
    if op == "rename":
        src = rng.choice(list(s))
        dst = _fresh_key(rng.choice(_RENAME).format(key=src), set(s))
        return {dst if k == src else k: v for k, v in s.items()}
    out = dict(s)
    out[_fresh_key(rng.choice(_ADDED), set(s))] = rng.choice(_PLACEHOLDER)
    return out


def new_keys(s: dict, n: int = 3, *, seed: int) -> dict:
    """Extend the state with ``n`` brand-new keys -- a superset, never a swap.

    Every original key and value survives untouched; the additions are picked
    from ``_ADDED`` by ``seed`` and deduplicated through ``_fresh_key``, so
    ``set(s) - set(out)`` stays empty and ``set(out) - set(s)`` grows by
    exactly ``n``. ``seed`` is keyword-only because it must never be mistaken
    for ``n`` in a positional call (risk R5: it is mandatory, not defaulted).

    Args:
        s: state to extend.
        n: how many new keys to add; ``n = 0`` returns an equivalent copy.
        seed: keyword-only; fixes which templates and placeholder values are
           drawn, so the same seed yields the same extension.

    Returns:
        A fresh dict: ``s`` plus ``n`` keys the model has never seen.
    """
    if n < 0:
        raise ValueError(f"new_keys needs n >= 0, got {n}")
    rng = _rng("new_keys", seed)
    out, used = dict(s), set(s)
    for _ in range(n):
        out[_fresh_key(rng.choice(_ADDED), used)] = rng.choice(_PLACEHOLDER)
        used = set(out)
    return out


def paraphrase(s: dict, seed: int) -> dict:
    """Reword every non-empty string value; keys, nesting and types survive.

    The recursion preserves shape exactly: ``set(out) == set(s)`` at every
    level (nested dicts included), list lengths hold, and non-string leaves
    (int/float/bool/None) are returned untouched -- only ``str`` leaves are
    rewritten, so the result stays a valid StateValue.

    Rewriting itself is template-driven (risk R3, no LLM): whole words found
    in the ``_SYNONYMS`` table are swapped for a variant chosen by ``seed``
    (capitalization follows the original word), and a value with no dictionary
    hit falls back to a seeded rephrasing wrapper -- hence every non-empty
    string comes out different from the input while keeping its content.
    Whitespace-only strings stay as-is: there is nothing to rephrase.

    Args:
        s: state to reword.

    Returns:
        A fresh dict of the same shape with reworded string leaves.
    """
    return _paraphrase(s, _rng("paraphrase", seed))


def _paraphrase(value: object, rng: random.Random) -> object:
    if isinstance(value, dict):
        return {k: _paraphrase(v, rng) for k, v in value.items()}
    if isinstance(value, list):
        return [_paraphrase(v, rng) for v in value]
    if isinstance(value, str):
        return _reword(value, rng)
    return value


def _reword(text: str, rng: random.Random) -> str:
    if not text.strip():
        return text
    out = _WORD.sub(lambda m: _synonym(m.group(0), rng), text)
    if out == text:  # no dictionary hit: fall back to a wrapper template
        out = rng.choice(_REPHRASE).format(v=text)
    return out


def _synonym(word: str, rng: random.Random) -> str:
    variant = rng.choice(_SYNONYMS[word.lower()])  # never the source word itself
    return variant.capitalize() if word[:1].isupper() else variant
