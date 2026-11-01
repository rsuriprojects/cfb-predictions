"""Tiny client for the CollegeFootballData.com API.

The API key comes only from the CFBD_API_KEY environment variable (a GitHub
Actions secret). It is sent in a header, never put in a URL, never printed,
and never written to disk: everything this repo writes is public.

The free plan allows 1,000 calls a month. Every call goes through get(), which
counts them so each run can report how many it used.
"""
import json
import os
import time
import urllib.parse
import urllib.request

BASE = "https://api.collegefootballdata.com"
CALLS = 0


def get(path, **params):
    global CALLS
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise RuntimeError("CFBD_API_KEY is not set")
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    })
    for attempt in range(3):
        try:
            CALLS += 1
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            err = str(e).replace(key, "***")
            if attempt == 2:
                raise RuntimeError(f"{path} {params}: {err}") from None
            print(f"  retrying {path} {params} ({err})")
            time.sleep(5 * (attempt + 1))
