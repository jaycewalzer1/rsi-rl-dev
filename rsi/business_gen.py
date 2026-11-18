"""A seeded, deliberately messy shoe-store export: the sample business dataset.

It stands in for a real company's data dump (synthetic, labelled as such everywhere it
is used). The mess is the kind intake has to prune: customer PII, test rows, exact
duplicate rows, a near-empty legacy column, a constant column, order lines whose order
is missing, and channel values with stray case and spaces.

The returns are packed around the edges the task rules care about: requests near day
30 in store-local time versus UTC, in-store returns stamped in local time in a different
column from mail returns, resent events with the same event id, and quantities that
over-ask what the line bought.

    python -m rsi.business_gen OUT_DIR [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

STORES = [
    ("S01", "Portland Pearl", "America/Los_Angeles", "west"),
    ("S02", "Brooklyn Navy Yard", "America/New_York", "east"),
    ("S03", "Shoreditch", "Europe/London", "intl"),
    ("S04", "Shibuya", "Asia/Tokyo", "intl"),
    ("S05", "Honolulu Kakaako", "Pacific/Honolulu", "west"),
]
PRODUCTS = [
    ("SKU-RUN-01", "Trail runner", "running", 14000, False),
    ("SKU-RUN-02", "Road racer", "running", 18000, False),
    ("SKU-BOOT-01", "Rain boot", "boots", 9500, False),
    ("SKU-BOOT-02", "Hiking boot", "boots", 21000, False),
    ("SKU-SAND-01", "Slide", "sandals", 3500, True),
    ("SKU-SAND-02", "Strap sandal", "sandals", 6000, False),
    ("SKU-DRESS-01", "Oxford", "dress", 16500, True),
    ("SKU-KIDS-01", "Kids sneaker", "kids", 5500, False),
    ("SKU-SOCK-01", "Wool socks 3-pack", "accessories", 2400, True),
    ("SKU-LACE-01", "Laces", "accessories", 600, False),
]
FIRST = ["Ava", "Kai", "Noor", "Leo", "Maya", "Sam", "Iris", "Theo", "Zoe", "Ravi", "Lena", "Omar"]
LAST = ["Park", "Silva", "Okafor", "Nguyen", "Berg", "Haddad", "Ito", "Moreau", "Lopez", "Shah"]
REASONS = ["too_small", "too_large", "changed_mind", "defective", "wrong_item"]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate(seed: int = 7, n_customers: int = 600, n_orders: int = 2400) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    tables: dict[str, list[dict]] = {}

    tables["stores"] = [
        {"store_id": s, "name": n, "timezone": tz, "region": r, "currency": "USD"} for s, n, tz, r in STORES
    ] + [{"store_id": "TEST", "name": "QA sandbox", "timezone": "UTC", "region": "test", "currency": "USD"}]

    tables["products"] = [
        {"sku": s, "name": n, "category": c, "price_cents": p, "final_sale": str(f).lower(),
         "legacy_code": f"L{rng.randint(100, 999)}" if rng.random() < 0.05 else ""}
        for s, n, c, p, f in PRODUCTS
    ]

    customers = []
    for i in range(n_customers):
        fn, ln = rng.choice(FIRST), rng.choice(LAST)
        customers.append({
            "customer_id": f"C{1000 + i}",
            "full_name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}{i}@mail.example.com",
            "phone": f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
            "street_address": f"{rng.randint(1, 9999)} {rng.choice(['Oak', 'Pine', 'Fir', 'Alder'])} St",
            "loyalty_tier": rng.choice(["none", "silver", "gold"]),
            "signup_at": _iso(datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 600))),
        })
    customers.append({"customer_id": "C0000", "full_name": "Test User", "email": "qa@example.test",
                      "phone": "+1-555-000-0000", "street_address": "1 Test Way", "loyalty_tier": "none",
                      "signup_at": "2025-01-01T00:00:00Z"})
    tables["customers"] = customers

    orders, lines, returns = [], [], []
    line_no = event_no = return_no = 0
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    for i in range(n_orders):
        store = rng.choice(STORES)
        placed = start + timedelta(minutes=rng.randint(0, 60 * 24 * 55))
        channel = rng.choice(["online", "store"])
        oid = f"O{50000 + i}"
        orders.append({
            "order_id": oid, "customer_id": rng.choice(customers[:-1])["customer_id"], "store_id": store[0],
            "channel": rng.choice([channel, channel.upper(), f" {channel} "]) if rng.random() < 0.15 else channel,
            "placed_at": _iso(placed), "status": "complete", "source_system": "pos-v2",
        })
        for _ in range(rng.choice([1, 1, 2, 2, 3])):
            sku, _, _, price, _ = rng.choice(PRODUCTS)
            qty = rng.choice([1, 1, 1, 2, 2, 3])
            discount = rng.choice([0, 0, 0, price * qty // 10, 1500, 999])
            line_no += 1
            lid = f"L{line_no:05d}"
            lines.append({"line_id": lid, "order_id": oid, "sku": sku, "qty": qty,
                          "unit_price_cents": price, "discount_cents": discount})
            if rng.random() > 0.55:
                continue
            # one or more return requests on this line, packed around day 30 local vs UTC
            for k in range(rng.choice([1, 1, 2, 2])):
                day = rng.choice([3, 12, 20, 29, 30, 30, 31, 31, 32, 45])
                tz = ZoneInfo(store[2])
                local_placed = placed.astimezone(tz)
                req_local = (local_placed + timedelta(days=day)).replace(
                    hour=rng.choice([0, 1, 7, 12, 18, 22, 23]), minute=rng.randint(0, 59), second=0)
                rchan = rng.choice(["mail", "mail", "store"])
                event_no += 1
                return_no += 1
                row = {
                    "return_id": f"R{return_no:05d}", "event_id": f"EV{event_no:06d}", "line_id": lid,
                    "qty": rng.choice([1, 1, qty, qty + 1]),
                    "channel": rchan, "reason": rng.choice(REASONS),
                    "requested_at": _iso(req_local) if rchan == "mail" else "",
                    "requested_local": req_local.strftime("%Y-%m-%d %H:%M") if rchan == "store" else "",
                    "kind": "exchange" if k == 0 and rng.random() < 0.4 else rng.choice(["refund", "refund", "refund", "exchange"]),
                }
                returns.append(row)
                if rng.random() < 0.2:  # the webhook resent it: same event, new row id, a little later
                    return_no += 1
                    returns.append({**row, "return_id": f"R{return_no:05d}"})

    # orphan lines (their order was purged upstream) and exact duplicate rows
    for k in range(4):
        line_no += 1
        lines.append({"line_id": f"L{line_no:05d}", "order_id": f"O9{k:04d}", "sku": "SKU-LACE-01", "qty": 1,
                      "unit_price_cents": 600, "discount_cents": 0})
    for table in (orders, lines):
        for row in rng.sample(table, 5):
            table.append(dict(row))
    orders.append({"order_id": "O00000", "customer_id": "C0000", "store_id": "TEST", "channel": "online",
                   "placed_at": "2026-09-01T00:00:00Z", "status": "test", "source_system": "pos-v2"})

    tables["orders"] = orders
    tables["order_lines"] = lines
    tables["returns"] = returns
    tables["support_tickets"] = [
        {"ticket_id": f"T{i:04d}", "order_id": rng.choice(orders)["order_id"],
         "customer_email": "redacted-later@mail.example.com",
         "body": rng.choice(["where is my refund", "size swap please", "box arrived wet", "charged twice?"]),
         "created_at": _iso(start + timedelta(days=rng.randint(0, 70)))}
        for i in range(40)
    ]
    return tables


def write_csv(tables: dict[str, list[dict]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (out / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    write_csv(generate(a.seed), a.out)
    print(f"wrote synthetic shoe-store export (seed {a.seed}) -> {a.out}")


if __name__ == "__main__":
    main()
