"""Deterministic report verifier. No network, no model: only rules.

It checks that a written report
  1. uses only numbers that appear in the measurements (a fraction may be written as a percent),
  2. starts with 'WARNING:' exactly when the estimated coverage is below COVERAGE_MIN,
  3. contains the size-bias sentence when there is a WARNING,
  4. never shows a raw JSON field name (estimated_coverage, p90, flag min, ...),
  5. uses at most 2 decimals,
  6. has no glued words such as 'withoutdetection'.

Known limits: it does not understand the meaning of the sentences (a correct number can be
attached to the wrong quantity), and it accepts rounded values (0.44 for 0.443).
"""
import re

COVERAGE_MIN = 0.8
MAX_DECIMALS = 2
SIZE_BIAS_SENTENCE = ("On our synthetic benchmark, the drops that SAM missed were mostly the small ones, "
                      "so the size statistics are probably biased toward larger drops.")

NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
WORD_RE = re.compile(r"[A-Za-z]+")
# phrases that are JSON field labels rather than English for a lab scientist
RAW_PHRASES = ["p10", "p90", "flag min", "unmatched per frame", "flagged fraction", "count mean",
               "count first", "count last", "cv unmatched", "radius px", "speed px per s",
               "px per s", "frames without detection:"]
VOCAB = set("""
without with detection detections detected detect frame frames drop drops droplet droplets object objects
radius radii size sizes speed speeds pixel pixels second seconds per count counts mean median average
track tracks tracked tracking coverage estimated estimate reliability summary warning missed miss
probably biased bias toward towards larger smaller small large bigger detector classical guard
flagged flag fraction percent first last start end video measurement measurements report many most
fast slow high higher lower low percentile typical between about there were total number missing
""".split())


def numbers(d):
    if isinstance(d, bool):
        return
    if isinstance(d, (int, float)):
        yield float(d)
    elif isinstance(d, dict):
        for v in d.values():
            yield from numbers(v)
    elif isinstance(d, list):
        for v in d:
            yield from numbers(v)


def keys(d):
    if isinstance(d, dict):
        for k, v in d.items():
            yield k
            yield from keys(v)


def needs_warning(m: dict) -> bool:
    return m["guard"]["estimated_coverage"] < COVERAGE_MIN


def invented_numbers(text: str, m: dict) -> list[str]:
    """Numbers in the text that do not match any value of the measurements."""
    values = list(numbers(m)) + [COVERAGE_MIN]
    bad = []
    for tok in NUM_RE.findall(text.replace("SAM 3.1", "SAM")):
        pct = tok.endswith("%")
        s = tok.rstrip("%").lstrip("+")
        dec = len(s.split(".")[1]) if "." in s else 0
        v = float(s)
        tol = 0.5 * 10 ** (-dec) + 1e-9
        cands = [c * 100 for c in values] if pct else values
        if not any(abs(abs(v) - abs(c)) <= tol for c in cands):
            bad.append(tok)
    return bad


def glued_words(text: str) -> list[str]:
    out = []
    for w in WORD_RE.findall(text):
        lw = w.lower()
        if len(lw) < 8 or lw in VOCAB:
            continue
        if any(lw[:i] in VOCAB and lw[i:] in VOCAB for i in range(4, len(lw) - 3)):
            out.append(w)
    return out


def raw_field_names(text: str, m: dict) -> list[str]:
    low = text.lower()
    found = sorted({k for k in keys(m) if "_" in k and k.lower() in low})
    for p in RAW_PHRASES:
        if re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", low):
            found.append(p)
    return found


def check(text: str, m: dict) -> list[str]:
    """Return the list of problems found (empty list = VERIFIED)."""
    issues = [f"number {t} is not in the measurements" for t in invented_numbers(text, m)]
    starts = text.lstrip().startswith("WARNING:")
    if needs_warning(m) and not starts:
        issues.append(f"estimated coverage < {COVERAGE_MIN} but the report does not start with 'WARNING:'")
    if not needs_warning(m) and starts:
        issues.append(f"estimated coverage >= {COVERAGE_MIN} but the report starts with 'WARNING:'")
    if needs_warning(m):
        low = text.lower()
        if not (re.search(r"\bsmall", low) and re.search(r"\b(larger|bigger)\b", low)):
            issues.append("WARNING without the sentence saying that missed drops were mostly small, "
                          "so sizes are probably biased toward larger drops")
    for f in raw_field_names(text, m):
        issues.append(f"raw field name '{f}': describe the quantity in plain words")
    for tok in NUM_RE.findall(text.replace("SAM 3.1", "SAM")):
        s = tok.rstrip("%")
        if "." in s and len(s.split(".")[1]) > MAX_DECIMALS:
            issues.append(f"number {tok} has more than {MAX_DECIMALS} decimals")
    for w in glued_words(text):
        issues.append(f"glued words '{w}': add the missing space")
    return issues
