"""Gate criteria (phase 1 spec section 7) and the step assertions of section 6.

Design: an assertion that only raises tells you the first thing that broke.
A gate record tells you everything that broke, in one table, which is what you
actually want at 11pm on day one. So `check` records and returns a bool;
`require` records and then raises.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import PATHS

_GATE_FILE = PATHS.logs / "phase1_gates.json"

# Section 7 go/no-go table. Criterion 6 is the real gate.
CRITERIA = {
    1: "Every FIPS joins (zero unmatched on both sides)",
    2: "Timezones aligned (outage peak within 1 h of ERA5 gust peak)",
    3: "Event table non-empty (20-60 events, 3 verified by eye)",
    4: "Units confirmed (m/s, metres, Kelvin asserted in code)",
    5: "Area weights valid (sum to 1.0 per county, equal-area CRS)",
    6: "HAZARD-CONSEQUENCE CORRELATION POSITIVE (gust_max vs customer_hours > 0.3)",
    7: "All three model stages execute (no exceptions, valid shapes, no NaNs)",
    8: "Monte Carlo produces spread (per-row std > 0)",
    9: "Bias correction active (mapped GEFS mean shifts toward ERA5 climatology)",
    10: "End-to-end single command (`make phase1` runs clean)",
    11: "Volumes and timings recorded (section 8 table filled in)",
}


class GateFailure(AssertionError):
    """A hard stop: the spec says do not proceed on a soft failure."""


@dataclass
class Record:
    name: str
    passed: bool
    detail: str = ""
    criterion: int | None = None
    step: str = ""
    warn: bool = False


@dataclass
class GateBook:
    step: str = ""
    records: list[Record] = field(default_factory=list)

    # ---- primitives ---------------------------------------------------------
    def check(self, name: str, condition: bool, detail: str = "",
              criterion: int | None = None, warn: bool = False) -> bool:
        ok = bool(condition)
        self.records.append(Record(name, ok, str(detail), criterion, self.step, warn))
        return ok

    def require(self, name: str, condition: bool, detail: str = "",
                criterion: int | None = None) -> None:
        if not self.check(name, condition, detail, criterion):
            self.flush()
            raise GateFailure(f"[{self.step}] {name}: {detail}")

    def note(self, name: str, detail: str) -> None:
        """A recorded observation that is not pass/fail (e.g. a measured value)."""
        self.records.append(Record(name, True, str(detail), None, self.step, warn=True))

    # ---- persistence --------------------------------------------------------
    def flush(self) -> None:
        prior = []
        if _GATE_FILE.exists():
            prior = [r for r in json.loads(_GATE_FILE.read_text())
                     if r["step"] != self.step]
        _GATE_FILE.write_text(json.dumps(
            prior + [asdict(r) for r in self.records], indent=2))

    @staticmethod
    def load_all() -> list[dict]:
        if not _GATE_FILE.exists():
            return []
        return json.loads(_GATE_FILE.read_text())

    @staticmethod
    def reset() -> None:
        """Truncate rather than unlink: some sandboxes disallow deletes, and an
        empty gate book is what the next run needs either way."""
        _GATE_FILE.write_text("[]")


def book(step: str) -> GateBook:
    return GateBook(step=step)


def criteria_report() -> tuple[str, bool]:
    """Render the section 7 table. Returns (markdown, all_passed)."""
    recs = GateBook.load_all()
    by_crit: dict[int, list[dict]] = {}
    for r in recs:
        if r.get("criterion"):
            by_crit.setdefault(r["criterion"], []).append(r)

    lines = ["| # | Criterion | Status | Evidence |", "|---|---|---|---|"]
    all_ok = True
    for num, text in CRITERIA.items():
        rs = by_crit.get(num, [])
        if not rs:
            status, evidence, ok = "NOT RUN", "--", False
        else:
            ok = all(r["passed"] for r in rs)
            status = "PASS" if ok else "FAIL"
            evidence = "; ".join(r["detail"] for r in rs if r["detail"])[:180] or "--"
        all_ok &= ok
        lines.append(f"| {num} | {text} | **{status}** | {evidence} |")

    failed = [r for r in recs if not r["passed"]]
    if failed:
        lines += ["", "**Failed checks**", ""]
        lines += [f"- `{r['step']}` / {r['name']}: {r['detail']}" for r in failed]
    return "\n".join(lines), all_ok
