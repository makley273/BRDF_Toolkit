"""Build one contact-sheet image per database from the RQ5 review bundle.

Each material gets a triptych (measured / analytical / neural), bordered in
the color of its human verdict, grouped into four blocks (analytical /
neural / equal / neither) and alphabetical within each block.

Reads:
    exports/rq5_review/records.jsonl   (renders + fit info)
    exports/rq5_review/verdicts.csv    (human verdicts)

Writes:
    .thesis/latex/figures/rq5_overview_<dataset>.png   one per database

Usage:
    python tools/rq5_overview.py
"""

import csv
import io
import json
import os

from PIL import Image, ImageDraw, ImageFont

BUNDLE = "exports/rq5_review"
OUT_DIR = os.path.join(".thesis", "latex", "figures")

THUMB = 88                 # px, each of the 3 renders in a card
GAP = 2                    # px between the 3 thumbs
BORDER = 4                 # px, verdict-colored frame around the card
CARD_MARGIN = 12           # px between cards
CAPTION_H = 16             # px, one line of material name below the card
BLOCK_HEADER_H = 34        # px, per verdict-block title bar
COLS = 6                   # cards per row

VERDICT_COLOR = {
    "analytical": (52, 152, 219),   # blue
    "neural":     (155, 89, 182),   # purple
    "equal":      (46, 204, 113),   # green
    "neither":    (231, 76, 60),    # red
}
VERDICT_LABEL = {
    "analytical": "Analytical better",
    "neural":     "Neural better",
    "equal":      "Both equally good",
    "neither":    "Neither good",
}
VERDICT_ORDER = ["analytical", "neural", "equal", "neither"]

BG = (251, 251, 250)
FG = (27, 26, 24)
MUTED = (110, 108, 102)

FONT_DIR = "C:/Windows/Fonts"


def load_font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


F_TITLE = load_font("segoeuib.ttf", 30)
F_SUB = load_font("segoeui.ttf", 16)
F_BLOCK = load_font("segoeuib.ttf", 17)
F_CAP = load_font("consola.ttf", 11)


def load_records():
    recs = {}
    path = os.path.join(BUNDLE, "records.jsonl")
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["key"]] = r
    return recs


def load_verdicts():
    path = os.path.join(BUNDLE, "verdicts.csv")
    return {row["key"]: row for row in csv.DictReader(io.open(path, encoding="utf-8"))}


def ellipsize(draw, text, font, max_w):
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "\u2026", font=font) > max_w:
        text = text[:-1]
    return text + "\u2026"


def render_of(rec, which, gain="g1"):
    node = rec["measured"] if which == "measured" else rec[which]
    if node is None:
        return None
    renders = node["renders"]
    if gain not in renders:
        return None
    path = os.path.join(BUNDLE, "renders", rec["key"], renders[gain])
    return Image.open(path).convert("RGB") if os.path.isfile(path) else None


def build_card(rec, verdict, tmp_draw):
    card_w = 3 * THUMB + 2 * GAP + 2 * BORDER
    card_h = THUMB + 2 * BORDER + CAPTION_H
    card = Image.new("RGB", (card_w, card_h), VERDICT_COLOR[verdict])
    inner = Image.new("RGB", (3 * THUMB + 2 * GAP, THUMB), (30, 30, 30))
    for i, which in enumerate(("measured", "analytical", "neural")):
        img = render_of(rec, which)
        if img is None:
            tile = Image.new("RGB", (THUMB, THUMB), (40, 40, 40))
            d = ImageDraw.Draw(tile)
            d.text((THUMB // 2, THUMB // 2), "\u2014", fill=(180, 60, 60),
                   font=F_CAP, anchor="mm")
        else:
            tile = img.resize((THUMB, THUMB), Image.LANCZOS)
        inner.paste(tile, (i * (THUMB + GAP), 0))
    card.paste(inner, (BORDER, BORDER))

    d = ImageDraw.Draw(card)
    d.rectangle([0, THUMB + 2 * BORDER, card_w, card_h], fill=BG)
    label = ellipsize(d, rec["material"], F_CAP, card_w - 6)
    d.text((card_w // 2, THUMB + 2 * BORDER + CAPTION_H // 2), label,
           fill=FG, font=F_CAP, anchor="mm")
    return card


def build_sheet(dataset, recs, verdicts, out_path):
    keys = [k for k, r in recs.items() if r["dataset"] == dataset]
    by_verdict = {v: [] for v in VERDICT_ORDER}
    for k in keys:
        v = verdicts.get(k, {}).get("verdict", "")
        if v in by_verdict:
            by_verdict[v].append(k)
    for v in by_verdict:
        by_verdict[v].sort(key=lambda k: recs[k]["material"])

    card_w = 3 * THUMB + 2 * GAP + 2 * BORDER
    card_h = THUMB + 2 * BORDER + CAPTION_H
    cell_w = card_w + CARD_MARGIN
    cell_h = card_h + CARD_MARGIN

    header_h = 90
    total_h = header_h
    for v in VERDICT_ORDER:
        n = len(by_verdict[v])
        if n == 0:
            continue
        rows = (n + COLS - 1) // COLS
        total_h += BLOCK_HEADER_H + rows * cell_h
    total_w = CARD_MARGIN + COLS * cell_w

    sheet = Image.new("RGB", (total_w, total_h), BG)
    d = ImageDraw.Draw(sheet)

    d.text((CARD_MARGIN, 14), f"RQ5 fit review \u2014 {dataset.upper()}",
           fill=FG, font=F_TITLE)
    d.text((CARD_MARGIN, 54), f"{len(keys)} materials, judged measured / "
           "analytical / neural, {}\u00d7{} thumbnails".format(THUMB, THUMB),
           fill=MUTED, font=F_SUB)
    legend_x = CARD_MARGIN
    legend_y = 78
    for v in VERDICT_ORDER:
        n = len(by_verdict[v])
        sw = 12
        d.rectangle([legend_x, legend_y, legend_x + sw, legend_y + sw],
                    fill=VERDICT_COLOR[v])
        txt = f"{VERDICT_LABEL[v]} ({n})"
        d.text((legend_x + sw + 6, legend_y - 2), txt, fill=FG, font=F_CAP)
        legend_x += sw + 6 + int(d.textlength(txt, font=F_CAP)) + 26

    y = header_h
    for v in VERDICT_ORDER:
        items = by_verdict[v]
        if not items:
            continue
        d.rectangle([0, y, total_w, y + BLOCK_HEADER_H], fill=VERDICT_COLOR[v])
        d.text((CARD_MARGIN, y + BLOCK_HEADER_H // 2),
               f"{VERDICT_LABEL[v]}  \u2014  {len(items)} materials",
               fill=(255, 255, 255), font=F_BLOCK, anchor="lm")
        y += BLOCK_HEADER_H

        for i, k in enumerate(items):
            row, col = divmod(i, COLS)
            x0 = CARD_MARGIN + col * cell_w
            y0 = y + row * cell_h
            card = build_card(recs[k], v, d)
            sheet.paste(card, (x0, y0))
        rows = (len(items) + COLS - 1) // COLS
        y += rows * cell_h

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, optimize=True)
    return sheet.size


def main():
    recs = load_records()
    verdicts = load_verdicts()
    datasets = sorted({r["dataset"] for r in recs.values()})
    for ds in datasets:
        out = os.path.join(OUT_DIR, f"rq5_overview_{ds}.png")
        size = build_sheet(ds, recs, verdicts, out)
        mb = os.path.getsize(out) / 1e6
        print(f"{ds:8s} -> {out}  {size[0]}x{size[1]}  {mb:.1f} MB")


if __name__ == "__main__":
    main()
