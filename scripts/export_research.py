"""Page the research-event API into a single training-set JSON file.

The backend exposes research events only as a paginated admin JSON API (`pageSize` <= 1000);
no CSV/file export exists (deferred to Epic 8 / Story 8.5). This script is the missing bridge
between the live database and `training/behavioral/train_bilstm.py`.

It fetches TWO event sets and concatenates them:

  1. `/admin/research/phase-a-dataset` -> `behavioral_affect_detected` + `self_report`
     (the feature windows and their labels).
  2. `/admin/research/events?eventType=section_started,section_completed`
     -> the section timeline. REQUIRED by the codebook labeller: the frontend never
     populates `section_id` on a `self_report` (the lesson page sends only
     `{affect, skipped, promptIndex}`), so a window can only be attributed to a section by
     falling inside that section's start/complete interval.

Output is `{"items": [...]}`, the shape both `phase_a.load_phase_a_windows` and
`codebook_labels.load_codebook_windows` already accept.

Usage:
    # credentials via env (preferred — avoids shell history)
    export AFFECTLEARN_ADMIN_EMAIL=admin@... AFFECTLEARN_ADMIN_PASSWORD=...
    python scripts/export_research.py

    # explicit, and only today's events
    python scripts/export_research.py --base-url https://affectlearn-api-4905.azurewebsites.net \
        --email admin@... --password ... --start-ts 1755043200000

    # reuse an access token instead of logging in
    python scripts/export_research.py --token "$TOKEN"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_BASE_URL = "https://affectlearn-api-4905.azurewebsites.net"
API_PREFIX = "/api/v1"
PAGE_SIZE = 1000  # server maximum
TIMELINE_EVENT_TYPES = "section_started,section_completed"


def _request(url: str, *, token: str | None = None, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise SystemExit(f"HTTP {e.code} for {url}\n{detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach {url}: {e.reason}") from e


def login(base: str, email: str, password: str) -> str:
    out = _request(
        f"{base}{API_PREFIX}/auth/login",
        body={"emailAddress": email, "password": password},
    )
    token = out.get("accessToken")
    if not token:
        raise SystemExit(f"Login succeeded but returned no accessToken: {out}")
    return token


def fetch_all(base: str, path: str, token: str, params: dict) -> list[dict]:
    """Page through a ResearchEventPage endpoint until every item is collected."""
    items: list[dict] = []
    page = 1
    total = None
    while True:
        q = {**params, "page": page, "pageSize": PAGE_SIZE}
        q = {k: v for k, v in q.items() if v is not None}
        url = f"{base}{API_PREFIX}{path}?{urllib.parse.urlencode(q)}"
        out = _request(url, token=token)
        batch = out.get("items", [])
        items.extend(batch)
        total = out.get("total", len(items))
        print(f"  {path} page {page}: +{len(batch)} ({len(items)}/{total})")
        if len(batch) < PAGE_SIZE or len(items) >= total:
            break
        page += 1
    if total is not None and len(items) != total:
        print(
            f"  WARNING: collected {len(items)} but server reported total={total}",
            file=sys.stderr,
        )
    return items


def fetch_catalogue(base: str, token: str) -> dict[str, dict]:
    """Build `{section_id: {title, course, lesson, sortOrder}}` from the course API.

    Research events carry section UUIDs, but the affect codebook is keyed by section TITLE
    (ids are minted at seed time and differ per database). This map is the bridge, and it is
    written into the export so the training set is self-contained and re-labellable offline.
    """
    courses: list[dict] = []
    page = 1
    while True:
        out = _request(
            f"{base}{API_PREFIX}/courses?page={page}&pageSize=100", token=token
        )
        batch = out.get("items", [])
        courses.extend(batch)
        if not batch or len(courses) >= out.get("total", len(courses)):
            break
        page += 1

    sections: dict[str, dict] = {}
    for c in courses:
        detail = _request(f"{base}{API_PREFIX}/courses/{c['id']}", token=token)
        for mod in detail.get("modules", []):
            for lsn in mod.get("lessons", []):
                for sec in lsn.get("sections", []):
                    sections[str(sec["id"])] = {
                        "title": sec["title"],
                        "course": detail.get("title"),
                        "module": mod.get("title"),
                        "lesson": lsn.get("title"),
                        "sortOrder": sec.get("sortOrder"),
                        "estimatedDurationMinutes": sec.get("estimatedDurationMinutes"),
                    }
    print(f"  {len(courses)} course(s), {len(sections)} section(s)")
    return sections


def summarise(items: list[dict]) -> None:
    """Print the per-export sanity check — run this after every session."""
    by_type = Counter(e.get("eventType") or e.get("event_type") for e in items)
    print("\nEvent types:")
    for t, n in by_type.most_common():
        print(f"  {t:32s} {n:5d}")

    windows = [
        e
        for e in items
        if (e.get("eventType") or e.get("event_type")) == "behavioral_affect_detected"
        and (e.get("payload") or {}).get("features") is not None
    ]
    print(f"\nBehavioural windows carrying `features`: {len(windows)}")

    reports = [
        e for e in items if (e.get("eventType") or e.get("event_type")) == "self_report"
    ]
    labels = Counter()
    skipped = omitted = 0
    for r in reports:
        p = r.get("payload") or {}
        if p.get("skipped"):
            skipped += 1
        elif p.get("omitted"):
            omitted += 1
        else:
            labels[str(p.get("affect") or "none").lower()] += 1
    print(f"Self-reports: {len(reports)} ({skipped} skipped, {omitted} omitted)")
    for a, n in labels.most_common():
        pct = 100 * n / max(sum(labels.values()), 1)
        print(f"  {a:12s} {n:4d}  {pct:5.1f}%")

    # The #1 pilot risk per PHASE_A_RUN_CHECKLIST — and it bites hardest at n=1.
    if labels:
        top, top_n = labels.most_common(1)[0]
        if top_n / sum(labels.values()) >= 0.8:
            print(
                f"\n  *** WARNING: {top_n / sum(labels.values()):.0%} of labels are "
                f"'{top}'. Intervene NOW — do not keep collecting. ***",
                file=sys.stderr,
            )

    sessions = {e.get("sessionId") or e.get("session_id") for e in items}
    print(f"\nSessions: {len(sessions)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.getenv("AFFECTLEARN_API", DEFAULT_BASE_URL))
    ap.add_argument("--email", default=os.getenv("AFFECTLEARN_ADMIN_EMAIL"))
    ap.add_argument("--password", default=os.getenv("AFFECTLEARN_ADMIN_PASSWORD"))
    ap.add_argument("--token", default=os.getenv("AFFECTLEARN_TOKEN"), help="Skip login.")
    ap.add_argument("--learner-id", default=None, help="Restrict to one learner.")
    ap.add_argument("--session-id", default=None, help="Restrict to one session.")
    ap.add_argument("--start-ts", type=int, default=None, help="Unix ms lower bound.")
    ap.add_argument("--end-ts", type=int, default=None, help="Unix ms upper bound.")
    ap.add_argument("--out", default="data/phase_a/export.json")
    ap.add_argument(
        "--check-gaps",
        action="store_true",
        help="Also report per-session missing sequence numbers (NFR23).",
    )
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    token = args.token
    if not token:
        if not (args.email and args.password):
            raise SystemExit(
                "Provide --token, or --email/--password (or set AFFECTLEARN_ADMIN_EMAIL / "
                "AFFECTLEARN_ADMIN_PASSWORD)."
            )
        print(f"Logging in to {base} as {args.email} ...")
        token = login(base, args.email, args.password)

    common = {
        "learnerId": args.learner_id,
        "sessionId": args.session_id,
        "startTs": args.start_ts,
        "endTs": args.end_ts,
    }

    print("\nFetching phase-a dataset (behavioural windows + self-reports) ...")
    items = fetch_all(base, "/admin/research/phase-a-dataset", token, common)

    print("\nFetching section timeline (section_started / section_completed) ...")
    items += fetch_all(
        base,
        "/admin/research/events",
        token,
        {**common, "eventType": TIMELINE_EVENT_TYPES},
    )

    if args.check_gaps:
        print("\nChecking sequence gaps ...")
        q = {k: v for k, v in common.items() if v is not None}
        url = f"{base}{API_PREFIX}/admin/research/events/gaps?{urllib.parse.urlencode(q)}"
        gaps = _request(url, token=token).get("gaps", {})
        nonempty = {s: g for s, g in (gaps or {}).items() if g}
        print(f"  sessions with gaps: {len(nonempty)}")
        for s, g in list(nonempty.items())[:10]:
            print(f"    {s}: {g[:20]}{' ...' if len(g) > 20 else ''}")

    print("\nFetching course catalogue (section id -> title) ...")
    sections = fetch_catalogue(base, token)

    summarise(items)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"items": items, "sections": sections}, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {out} — {len(items)} events, {len(sections)} sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
