#!/usr/bin/env python3
"""Agent-owned browser demo: open a browser on :0, Google a query, click the Nth result.

This drives an AGENT-OWNED Chromium (the same Playwright/CDP DOM stack the
computer-use worker's perception layer uses, chromium-1217) on your real :0.
The browser is THIS script's own child process -> OWNED under the harness
ownership model -> no foreign-window injection, fully inside the safety design.
Clicking the Nth result is done by DOM (deterministic), not pixel guessing.

NOTE (honest scope): this is a direct DOM driver, NOT the full autonomous
perceive->reason->actuate agent loop. That loop can't do this yet — `navigate`
is a stub, agent-launched apps land on the isolated Xvfb, and there's no
DOM-actuation wired. This script is the working prototype of that missing
`navigate`/DOM-click capability.

Usage:
  .venv/bin/python scripts/agent_browser_demo.py "search terms"            # headed on :0 (watch it)
  .venv/bin/python scripts/agent_browser_demo.py "search terms" --rank 19  # pick a different result
  .venv/bin/python scripts/agent_browser_demo.py "search terms" --headless --dry-run  # verify, no window/click

Caveats: Google may show a consent interstitial (handled best-effort) or bot-block
an automated browser (reported honestly — never faked). Reaching rank 19 paginates
past page 1.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from urllib.parse import quote_plus


def _find_chromium() -> str:
    # Prefer 1217 (verified launchable here); fall back to any cached build.
    pref = os.path.expanduser("~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome")
    if os.path.exists(pref):
        return pref
    cands = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")), reverse=True)
    if not cands:
        sys.exit("No cached Playwright chromium found (~/.cache/ms-playwright/chromium-*). Run: playwright install chromium")
    return cands[0]


# Per-engine config. Google fingerprints/bot-blocks clean automated browsers
# (verified: instant /sorry/index). DuckDuckGo permits automation and is the
# reliable default; "more results" loads in-page so rank 19 is easy to reach.
ENGINES = {
    # Bing renders server-side and tolerates automated Chromium (verified). Default.
    "bing": {
        "url": "https://www.bing.com/",
        "serp": "https://www.bing.com/search?q={q}",
        "box": "input[name=q], textarea[name=q]",
        "results": "li.b_algo h2 a",
        "more": "a.sb_pagN, a[aria-label='Next page'], a[title='Next page']",
        "paginates": True,   # Next navigates to a fresh page (~10/page)
    },
    "duckduckgo": {
        "url": "https://duckduckgo.com/",
        "serp": "https://duckduckgo.com/?q={q}",
        "box": "input[name=q]",
        "results": "a[data-testid='result-title-a']",
        "more": "button#more-results, button:has-text('More results')",
        "paginates": False,  # in-page "More results" appends to the DOM
    },
    "google": {
        "url": "https://www.google.com/ncr",
        "serp": "https://www.google.com/search?q={q}",
        "box": "textarea[name=q], input[name=q]",
        "results": "#search a:has(h3)",
        "more": "#pnnext",
        "paginates": True,   # usually bot-blocks an automated browser anyway
    },
}


def _dismiss_consent(page) -> None:
    """Best-effort consent/cookie dismissal (Google + DDG variants)."""
    for sel in ("#L2AGLb", "button:has-text('Accept all')", "button:has-text('I agree')",
                "button:has-text('Reject all')", "button[aria-label*='Accept']",
                "div[role='none'] button:has-text('Accept')"):
        try:
            el = page.locator(sel)
            if el.count() and el.first.is_visible():
                el.first.click(timeout=2000)
                page.wait_for_timeout(700)
                return
        except Exception:
            pass


def _bot_blocked(page) -> bool:
    url = (page.url or "").lower()
    if "/sorry/" in url or "consent.google.com" in url and "search" not in url:
        return "/sorry/" in url
    try:
        body = page.inner_text("body", timeout=1500).lower()
    except Exception:
        body = ""
    return ("unusual traffic" in body or "are not a robot" in body
            or "detected unusual" in body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="programmatic gui automation", help="what to search")
    ap.add_argument("--rank", type=int, default=19, help="which organic result to click (1-based)")
    ap.add_argument("--engine", choices=list(ENGINES), default="bing",
                    help="bing (reliable; default), duckduckgo, or google (usually bot-blocks automated browsers)")
    ap.add_argument("--headless", action="store_true", help="run without a visible window (verify mode)")
    ap.add_argument("--dry-run", action="store_true", help="locate the Nth result but do NOT click it")
    ap.add_argument("--max-pages", type=int, default=8)
    args = ap.parse_args()
    eng = ENGINES[args.engine]

    if not args.headless and not os.environ.get("DISPLAY"):
        sys.exit("No DISPLAY set — run from your graphical session, or pass --headless.")

    from playwright.sync_api import sync_playwright

    exe = _find_chromium()
    print(f"[*] agent-owned chromium: {exe}")
    print(f"[*] engine={args.engine}  query={args.query!r}  rank={args.rank}  "
          f"headless={args.headless}  dry_run={args.dry_run}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless, executable_path=exe,
            slow_mo=0 if args.headless else 450,  # slow enough to watch when headed
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                  "--start-maximized", "--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            viewport=None if not args.headless else {"width": 1280, "height": 900},
            locale="en-US",
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
        page = ctx.new_page()

        print(f"[1] opening {eng['url']} ...")
        page.goto(eng["url"], wait_until="domcontentloaded", timeout=30000)
        _dismiss_consent(page)

        print(f"[2] typing search: {args.query!r}")
        try:
            box = page.locator(eng["box"]).first
            box.click(timeout=8000)
            box.fill(args.query)
            box.press("Enter")
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(1500)
        except Exception as e:
            print(f"    (box submit failed: {type(e).__name__}; using direct SERP url)")
        _dismiss_consent(page)
        # Guarantee we're on a results page (homepage submit can be flaky/JS-gated).
        if page.locator(eng["results"]).count() == 0:
            print("    landing directly on the results URL ...")
            page.goto(eng["serp"].format(q=quote_plus(args.query)),
                      wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1500)
            _dismiss_consent(page)

        if _bot_blocked(page):
            print(f"[!] {args.engine} bot-blocked the automated browser (url={page.url}).")
            print("    Honest result: cannot complete the search this way. "
                  + ("Try --engine duckduckgo." if args.engine == "google" else ""))
            if not args.headless:
                page.wait_for_timeout(4000)
            browser.close()
            return 2

        # Grow the result set until we reach `rank`.
        #  - paginate engines (Bing/Google): "Next" loads a FRESH page of ~10, so
        #    track a running total and index into the page that contains `rank`.
        #  - accumulate engines (DDG): "More results" APPENDS, so the live count grows.
        print(f"[3] collecting organic results until rank {args.rank} ...")
        target = None
        seen = 0  # results on prior pages (paginate mode only)
        for step in range(1, args.max_pages + 1):
            results = page.locator(eng["results"])
            n = results.count()
            total = (seen + n) if eng["paginates"] else n
            print(f"    step {step}: {n} on this page, {total} total")
            if total >= args.rank:
                idx = (args.rank - seen - 1) if eng["paginates"] else (args.rank - 1)
                target = results.nth(idx)
                break
            more = page.locator(eng["more"])
            if not more.count() or not more.first.is_visible():
                print(f"[!] only {total} results available; rank {args.rank} unreachable.")
                browser.close()
                return 3
            if eng["paginates"]:
                seen += n
            more.first.scroll_into_view_if_needed(timeout=4000)
            more.first.click()
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(1200)
            _dismiss_consent(page)
        if target is None:
            print(f"[!] hit max-pages={args.max_pages} before reaching rank {args.rank}.")
            browser.close()
            return 3

        try:
            title = target.inner_text(timeout=3000).strip().splitlines()[0]
        except Exception:
            title = "(title unavailable)"
        href = target.get_attribute("href")
        print(f"[4] rank {args.rank}: {title!r}")
        print(f"    href: {href}")
        if args.dry_run:
            print("[5] --dry-run: located but NOT clicking.")
        else:
            print("[5] clicking it ...")
            target.scroll_into_view_if_needed(timeout=4000)
            before = len(ctx.pages)
            target.click(timeout=8000)
            page.wait_for_timeout(2500)
            # Result links often open a new tab (target=_blank); follow it.
            landed = ctx.pages[-1] if len(ctx.pages) > before else page
            try:
                landed.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass
            print(f"[6] landed on: {landed.url}")
            if not args.headless:
                try:
                    landed.wait_for_timeout(5000)  # let you watch the result page
                except Exception:
                    pass  # operator closed the browser during the pause — expected

        browser.close()
    print("RESULT: PASS — reached and " + ("located" if args.dry_run else "clicked") + f" rank {args.rank}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
