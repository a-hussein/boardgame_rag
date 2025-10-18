# src/boardgame_hybrid_qa/data_gen.py
from __future__ import annotations
import argparse, json, random, pathlib, csv
from typing import List, Dict, Any, Optional
import pandas as pd

MECHANICS = [
    "Deck Building","Area Control","Worker Placement","Dice Rolling","Trading",
    "Set Collection","Tile Placement","Hidden Roles","Negotiation","Hand Management",
    "Engine Building","Drafting","Pattern Building","Push Your Luck","Route Building",
]

CATEGORIES = [
    "Economic","Card Game","Eurogame","Ameritrash","Bluffing","Deduction",
    "Abstract","Family","Strategy","Party","Cooperative","Thematic"
]

NAMES = [
    "Catan","Dominion","Carcassonne","Pandemic","Ticket to Ride","7 Wonders",
    "Azul","Splendor","Wingspan","Terraforming Mars","Gloomhaven","Root",
    "The Resistance: Avalon","Brass: Birmingham","Scythe","Spirit Island",
    "Air Land Sea","Eldorado"
]

def _rand_text(name:str, mechs:List[str], cats:List[str], weight:float, t:int)->str:
    bits = []
    if "Deck Building" in mechs: 
        bits.append("players construct engines from a shared market")
    if "Worker Placement" in mechs: 
        bits.append("actions are scarce; turn order tension matters")
    if "Hidden Roles" in mechs: 
        bits.append("social deduction and table talk drive decisions")
    if "Trading" in mechs: 
        bits.append("negotiation yields flexible exchanges and alliances")
    if "Dice Rolling" in mechs: 
        bits.append("variance can be mitigated via rerolls or conversions")
    if not bits: 
        bits.append("strategic choices compound into long-term advantages")
        
    return (f"{name} blends {', '.join(mechs[:2])} within a {cats[0].lower()} frame. "
            f"Typical weight {weight:.1f}; play time around {t} minutes. " + " ".join(bits))

def _row(doc_id:str, name:str)->Dict[str,Any]:
    rng = random.Random(doc_id)
    year = rng.randint(1990, 2024)
    pmin, pmax = rng.choice([(2,4),(2,5),(3,4),(3,5),(4,6)])
    t = rng.choice([30,35,45,60,75,90])
    weight = round(rng.uniform(1.3, 3.8), 1)
    mechs = rng.sample(MECHANICS, k=rng.randint(1,3))
    cats = rng.sample(CATEGORIES, k=rng.randint(1,2))
    text = _rand_text(name, mechs, cats, weight, t)
    return dict(
        doc_id=doc_id, name=name, year=year, players_min=pmin, players_max=pmax,
        play_time=t, weight=weight, mechanics=mechs, categories=cats, text=text
    )

def synthesize(n:int)->List[Dict[str,Any]]:
    rows = []
    for i in range(n):
        name = random.choice(NAMES)
        rows.append(_row(f"G{i:03d}", f"{name} {i%7 if i>0 else ''}".strip()))
    return rows

def load_bgg_csv(path: pathlib.Path) -> List[Dict[str,Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, rec in enumerate(r):
            name = rec.get("name") or f"BGG {i}"
            mechs = [m.strip() for m in (rec.get("mechanics","").replace(";",",").split(",")) if m.strip()]
            cats  = [c.strip() for c in (rec.get("categories","").replace(";",",").split(",")) if c.strip()]
            players_min = int(rec.get("minplayers") or 2)
            players_max = int(rec.get("maxplayers") or max(4, players_min))
            play_time   = int(rec.get("playingtime") or 45)
            weight      = float(rec.get("weight") or 2.3)
            text = _rand_text(name, mechs or ["Set Collection"], cats or ["Family"], weight, play_time)
            rows.append(dict(
                doc_id=f"BGG{i:03d}", name=name, year=int(rec.get("year") or 2000),
                players_min=players_min, players_max=players_max,
                play_time=play_time, weight=weight, mechanics=mechs or ["Set Collection"],
                categories=cats or ["Family"], text=text
            ))
    return rows

def write_jsonl(rows:List[Dict[str,Any]], out:pathlib.Path)->None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def mk_processed(in_jsonl:pathlib.Path, out_parquet:pathlib.Path)->None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in in_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    # simple normalization
    df["mechanics_flat"]  = df["mechanics"].apply(lambda xs: [x.lower() for x in xs])
    df["categories_flat"] = df["categories"].apply(lambda xs: [x.lower() for x in xs])
    df.to_parquet(out_parquet, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, help="Write JSONL corpus here")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bgg-csv", type=pathlib.Path, default=None, help="Optional BGG CSV to augment")
    ap.add_argument("--mk-processed", nargs=2, metavar=("IN_JSONL","OUT_PARQUET"))
    args = ap.parse_args()

    if args.out:
        synth = synthesize(args.n)
        if args.bgg_csv and args.bgg_csv.exists():
            bgg = load_bgg_csv(args.bgg_csv)
            # 70/30 mix
            k = int(len(synth) * 0.7)
            rows = synth[:k] + bgg[: max(0, len(synth)-k)]
        else:
            rows = synth
        write_jsonl(rows, args.out)
        print(f"Wrote {len(rows)} docs -> {args.out}")

    if args.mk_processed:
        in_p = pathlib.Path(args.mk_processed[0])
        out_p = pathlib.Path(args.mk_processed[1])
        mk_processed(in_p, out_p)
        print(f"Processed -> {out_p}")

if __name__ == "__main__":
    main()
