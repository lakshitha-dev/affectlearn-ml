"""Refresh `reports/deployment/aggregates_30d.json`, the deployment record Section 4.12 reports.

WHY THIS EXISTS. The first version of that file was produced by hand from a browser session,
which is how it came to disagree with the figure beside it: Table 4.22 was a reading from one
date and Figure 4.15 a reading from another, and nothing in the repository could reconcile
them because nothing could reproduce either. A record a thesis quotes has to be re-derivable
on demand, or it is a screenshot of a number.

WHAT IT WRITES. Counts only. The window's raw events carry `learner_id`, `session_id` and
adaptation content; none of that reaches the file, and the assertions at the end of `build`
fail rather than write if any of it appears. The thesis states this as a property of the
artefact, so it is enforced here rather than trusted.

WHAT IT DOES NOT DO. It does not filter out the researcher's own sessions, because there is no
way to tell them apart and pretending otherwise would be worse than the caveat. Recruitment
has not opened, so *every* session in the window is the researcher's; the caption on Table
4.22 says exactly that, and the number is a record of what the software does rather than
evidence about learners.

    python scripts/capture_deployment_record.py \
        --api https://affectlearn-api-2026.azurewebsites.net/api/v1 \
        --email admin@example.com --password '...'

Reads AL_API / AL_ADMIN_EMAIL / AL_ADMIN_PASSWORD when the flags are omitted, so a password
need not appear in shell history.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

OUT = pathlib.Path(__file__).resolve().parents[1] / "reports" / "deployment" / "aggregates_30d.json"
WINDOW_HOURS = 720

# The day the reasoning node began generating every adaptation rather than falling back. It is
# a real discontinuity in the record and Section 4.12.2 splits on it, so the split is computed
# here rather than eyeballed off the by-day table.
GENERATION_FROM = "2026-09-05"

# Identifier columns the artefact must never carry. Checked against what is actually written.
FORBIDDEN = ("learner_id", "session_id", "adaptation_id", "section_id")


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(url: str, token: str, raw: bool = False):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer %s" % token})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    return data.decode("utf-8-sig") if raw else json.loads(data)


def by_day(csv_text: str) -> tuple[list[dict], dict, dict]:
    """Deliveries per day, split on whether the reasoning node generated the text.

    Derived from the event stream rather than from the aggregate endpoint, because the
    endpoint reports the window's totals and Section 4.12.2's whole argument is that the
    window's headline rate is not the current rate.
    """
    days: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        if row.get("event_type") != "adaptation_delivered":
            continue
        day = (row.get("iso_time") or "")[:10]
        if not day:
            continue
        d = days.setdefault(day, {"delivered": 0, "generated": 0, "fallback": 0})
        d["delivered"] += 1
        # The column is the string "True"/"False"; anything else is treated as generated,
        # which is the reading that does NOT flatter the result.
        if str(row.get("fallback", "")).strip().lower() == "true":
            d["fallback"] += 1
        else:
            d["generated"] += 1

    rows = [{"date": k, **v} for k, v in sorted(days.items())]
    since = {"from": GENERATION_FROM, "delivered": 0, "generated": 0, "fallback": 0}
    before = {"delivered": 0, "generated": 0, "fallback": 0}
    for r in rows:
        bucket = since if r["date"] >= GENERATION_FROM else before
        for k in ("delivered", "generated", "fallback"):
            bucket[k] += r[k]
    return rows, since, before


def build(api: str, email: str, password: str) -> dict:
    tok = _post("%s/auth/login" % api, {"emailAddress": email, "password": password})
    token = tok.get("accessToken") or tok["access_token"]

    agg = _get("%s/monitor/aggregates?hours=%d" % (api, WINDOW_HOURS), token)
    deliveries = _get(
        "%s/monitor/export.csv?hours=%d&event_types=adaptation_delivered" % (api, WINDOW_HOURS),
        token, raw=True)
    rows, since, before = by_day(deliveries)

    rec = {
        "_note": (
            "Aggregate deployment record over the %d-hour window ending at capture. "
            "Recruitment has not opened, so none of these sessions is a recruited "
            "participant; they are the researcher's own use of the deployed instance. "
            "Aggregates only: no learner identifier, session identifier or content reaches "
            "this file. The window slides, so a later capture drops early sessions and adds "
            "recent ones; the date below is the reading Table 4.22 and Figure 4.15 both quote."
            % WINDOW_HOURS),
        "_source": "GET /api/v1/monitor/aggregates?hours=%d on the deployed instance" % WINDOW_HOURS,
        "_generator": "scripts/capture_deployment_record.py",
        "_captured": dt.date.today().isoformat(),
        "windowHours": agg["windowHours"],
        "totalEvents": agg["totalEvents"],
        "sessions": agg["sessions"],
        "cycles": agg["cycles"],
        "gatedCycles": agg["gatedCycles"],
        "gatePassed": agg.get("gatePassed"),
        "adaptMinConfidence": agg["adaptMinConfidence"],
        "interventions": agg["interventions"],
        "gateReasons": agg["gateReasons"],
        "affectCounts": agg["affectCounts"],
        "modalityStats": agg["modalityStats"],
        "eventsByType": agg["eventsByType"],
        "interventionsByDay": rows,
        "interventionsSince": since,
        "interventionsBefore": before,
    }

    # The by-day derivation and the endpoint's own count are computed from the same events by
    # different paths. If they disagree, one of them is wrong and neither should be published.
    total = sum(r["delivered"] for r in rows)
    assert total == rec["interventions"]["delivered"], (
        "by-day deliveries %d != endpoint total %d" % (total, rec["interventions"]["delivered"]))

    blob = json.dumps(rec)
    for col in FORBIDDEN:
        assert col not in blob, "identifier %r reached the artefact" % col
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("AL_API", ""))
    ap.add_argument("--email", default=os.environ.get("AL_ADMIN_EMAIL", ""))
    ap.add_argument("--password", default=os.environ.get("AL_ADMIN_PASSWORD", ""))
    a = ap.parse_args()
    if not (a.api and a.email and a.password):
        return int(bool(sys.stderr.write(
            "need --api/--email/--password or AL_API/AL_ADMIN_EMAIL/AL_ADMIN_PASSWORD\n")))

    try:
        rec = build(a.api.rstrip("/"), a.email, a.password)
    except urllib.error.HTTPError as e:
        return int(bool(sys.stderr.write("HTTP %s: %s\n" % (e.code, e.read()[:300]))))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(rec, indent=2) + "\n")

    iv = rec["interventions"]
    print("wrote %s" % OUT)
    print("  captured %s over %dh" % (rec["_captured"], rec["windowHours"]))
    print("  sessions %d, cycles %d, gate evaluations %s"
          % (rec["sessions"], rec["cycles"], "{:,}".format(rec["gatedCycles"])))
    print("  delivered %d, fallback %d (%.1f%%), generated %d"
          % (iv["delivered"], iv["fallback"], iv["fallbackRate"] * 100,
             iv["delivered"] - iv["fallback"]))
    for r in rec["interventionsByDay"]:
        print("    %s  delivered %3d  generated %3d  fallback %3d"
              % (r["date"], r["delivered"], r["generated"], r["fallback"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
