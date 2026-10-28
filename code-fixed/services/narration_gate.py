"""The customer's story must not contradict what their own phone shows.

THE DEFECT, systematic across every airplane-mode case in the 2026-08-20 suite.

    world:     phone.airplane_mode = True
    device:    check_status_bar   -> "Airplane Mode  No Signal  Data Enabled  87%"
               check_network_status -> signal_strength "none", network_type "none"
               run_speed_test     -> "no connection"
    narrated:  "My phone's barely showing any bars, web pages take forever to load,
                and picture messages won't send."

Barely any bars is not No Signal, and takes forever to load is not no connection. The
narrator HAD all three observables - they are in its prompt - and wrote a DEGRADED
service anyway, because the prompt leads with our broken-predicate names
(`has_cellular_service: false`) and a predicate reads like weak, not dead.

Two costs, both measured:

  1. The agent troubleshoots speed. Across the run it went to APN settings and network
     mode while the phone had no radio at all.
  2. The customer simulator, primed by its own opening story, then CONFABULATES to stay
     consistent with it: "Airplane mode is off, mobile data is on" - asserted with no
     check_status_bar call, on a case whose world has airplane mode ON. The agent
     believed it, and toggle_airplane_mode was never asked for again in that
     conversation.

Prompt wording alone did not fix this class before, so it is checked mechanically. The
outage words come from the ENVIRONMENT's own observable values, not from a telecom
vocabulary written here.
"""
from __future__ import annotations

import json
import re

# Values a compiled environment uses to mean "nothing at all". Matched against the
# observables the customer's own read-only tools return.
DEAD_VALUES = ("none", "no signal", "no connection", "not connected", "unavailable",
               "off", "disabled", "no service")

# What a person says when something is SLOW or INTERMITTENT rather than absent. If the
# device reports nothing at all, none of these is true.
PARTIAL = re.compile(
    r"\b(barely|hardly|weak(er)?|spotty|patchy|slow(er|ly)?|crawl|sluggish|"
    r"takes? (for)?ever|a (few|couple of) bars|one bar|two bars|low bars|"
    r"intermittent|on and off|sometimes|drops? out|keeps? dropping|unreliable|"
    r"laggy|delayed|not as fast|less than usual)\b", re.I)

# Saying the connection is entirely gone. One of these must appear instead.
TOTAL = re.compile(
    r"\b(no (signal|service|bars|connection|internet|data)|nothing at all|"
    r"completely (dead|out|off)|zero bars|can'?t connect (to )?(anything|the network)|"
    r"totally (dead|offline)|won'?t connect at all|not connect(ed|ing) at all|"
    r"no network|out of service|dead)\b", re.I)


def outage_evidence(observable: dict) -> list[str]:
    """-> the observables that say the connection is GONE, not merely poor."""
    out = []
    for tool, value in (observable or {}).items():
        blob = json.dumps(value, default=str).lower()
        if any(f'"{d}"' in blob or f": {d}" in blob for d in DEAD_VALUES):
            if any(k in tool for k in ("status_bar", "network", "speed", "signal")):
                out.append(f"{tool}={json.dumps(value, default=str)[:90]}")
    return out


def contradictions(text: str, observable: dict) -> list[str]:
    """-> why this narration contradicts the device. Empty means consistent."""
    ev = outage_evidence(observable)
    if not ev:
        return []
    t = str(text or "")
    hit = PARTIAL.search(t)
    if not hit:
        return []
    if TOTAL.search(t):
        # "no signal at all, and when it does come back it's slow" — the outage IS
        # stated, so a degradation word alongside it is a description, not a denial.
        return []
    return [f"says {hit.group(0)!r} (partial service) but the device reports a total "
            f"outage: {ev[0]}"]


def correction_note(reasons: list[str]) -> str:
    return (
        "YOUR NARRATION WAS REJECTED. It describes service that is DEGRADED, but this "
        "customer's own device reports NO SERVICE AT ALL.\n"
        + "\n".join(f"  - {r}" for r in reasons)
        + "\nRewrite `reason_for_call` so it matches what their phone actually shows. "
          "Say the connection is gone, not that it is slow or weak. Still do not name "
          "any setting, cause or fix — only what the person sees."
    )
