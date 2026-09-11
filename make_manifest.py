#!/usr/bin/env python3
"""
Generates json/manifest.json — a plain list of every *.json record file in
the json/ folder next to this script.

Why this is needed: on GitHub Pages (and most static hosts), a webpage
cannot ask the server "what files are in this folder?" — there's no
directory listing to read, unlike a local `python3 -m http.server`.
ECG_Software2.html knows to fetch json/manifest.json first and use that
list instead, which is how it finds your records once hosted on GitHub
Pages.

Usage:
    python3 make_manifest.py

Run this once, then re-run it any time you add or remove files from json/.
Commit the resulting json/manifest.json to your repo along with the json/
folder itself.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(HERE, "json")
MANIFEST_PATH = os.path.join(JSON_DIR, "manifest.json")


def main():
    if not os.path.isdir(JSON_DIR):
        raise SystemExit(
            f"Couldn't find a json/ folder next to this script "
            f"(looked in: {JSON_DIR}). Put this script in the same folder "
            f"as your json/ folder and try again."
        )

    files = sorted(
        f for f in os.listdir(JSON_DIR)
        if f.lower().endswith(".json") and f.lower() != "manifest.json"
    )

    if not files:
        raise SystemExit(f"No .json record files found in {JSON_DIR}.")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(files, f, indent=2)

    print(f"Wrote {MANIFEST_PATH} listing {len(files)} record file(s):")
    for name in files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
