"""Refunds owed per order, over the synthetic shoe-store export (rsi/business_gen.py).

The hand-written family the framework started from; kept as the worked example of the
contract in rsi/family.py and as a control benchmark from a dataset no model trains on.
"""

from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

FAMILY = {
    "id": "shoe-refunds",
    "background": "You maintain the refunds job for a shoe retailer with stores in several time zones. "
                  "Each returns row is one return request for an order line. Compute the refund owed per order.",
    "pins": [
        "Return `{order_id: refund_cents}` with an int for every order whose refund is above 0, and no other keys.",
        "Order time is `orders.placed_at` (UTC, ISO 8601 with Z); a store's IANA zone is `stores.timezone`.",
    ],
    "tables": ["stores", "products", "orders", "order_lines", "returns"],
}

RULES = [
    {"id": "in_store", "mode": "sibling-path-missed",
     "present": "In-store returns follow every rule exactly as mail returns do. Their request time is the store's local wall-clock time, in requested_local; mail returns carry UTC in requested_at.",
     "absent": "Only mail returns (channel mail) count; ignore in-store returns."},
    {"id": "window", "mode": "timezone-math",
     "present": "A return is accepted only within 30 days of the order, counted in whole calendar days in the store's own time zone: day 30 is accepted, day 31 is not.",
     "absent": "There is no time limit on returns."},
    {"id": "final_sale", "mode": "seam-assumption",
     "present": "A product marked final_sale is returnable only when the reason is defective; any other final-sale return is rejected.",
     "absent": "Final-sale products are returnable like any other."},
    {"id": "dedupe", "mode": "idempotency-under-failure",
     "present": "The webhook sometimes delivers a return twice: rows sharing an event_id are one return and count once.",
     "absent": "Every row is its own return, even when event_ids repeat."},
    {"id": "qty_cap", "mode": "requirements-ignored",
     "present": "Process returns in request-time order (ties by return_id). A return that would bring its line's accepted quantity above the quantity bought is rejected whole; rejected returns use up nothing.",
     "absent": "There is no limit on returned quantity."},
    {"id": "discount", "mode": "derived-surface-omitted",
     "present": "Refund the net price: unit_price_cents * qty minus the line discount's share, discount_cents * qty // line qty (integer division).",
     "absent": "Refund unit_price_cents * qty; ignore discounts."},
    {"id": "exchange", "mode": "single-manifestation-fix",
     "present": "Exchanges (kind exchange) refund nothing, yet must pass every other rule and use up returnable quantity like a refund.",
     "absent": "There is no special case for exchanges: rows whose kind is exchange are refunded like any return."},
    {"id": "tenant", "mode": "authz-by-id",
     "present": "Report only orders placed at store params['store_id']; another store's returns never appear, whatever line_id they cite.",
     "absent": "Report orders from every store."},
]


def params_choices(db):
    return [{"store_id": s["store_id"]} for s in db["stores"]]


def _utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def reference(db, params, rules):
    rules = set(rules)
    stores = {s["store_id"]: ZoneInfo(s["timezone"]) for s in db["stores"]}
    orders = {o["order_id"]: o for o in db["orders"]}
    lines = {ln["line_id"]: ln for ln in db["order_lines"]}
    final = {p["sku"]: p["final_sale"] in (True, "true") for p in db["products"]}

    rows = []
    for r in db["returns"]:
        line = lines.get(r["line_id"])
        if line is None or line["order_id"] not in orders:
            continue
        order = orders[line["order_id"]]
        tz = stores[order["store_id"]]
        if r["channel"] == "store":
            if "in_store" not in rules:
                continue
            req = datetime.strptime(r["requested_local"], "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        else:
            req = _utc(r["requested_at"])
        if "tenant" in rules and order["store_id"] != params.get("store_id"):
            continue
        rows.append((req.astimezone(timezone.utc), r["return_id"], r, line, order, tz))
    rows.sort(key=lambda x: (x[0], x[1]))

    seen_events, used, refund = set(), defaultdict(int), defaultdict(int)
    for req, _, r, line, order, tz in rows:
        if "dedupe" in rules:
            if r["event_id"] in seen_events:
                continue
            seen_events.add(r["event_id"])
        if "window" in rules:
            placed = _utc(order["placed_at"]).astimezone(tz).date()
            if (req.astimezone(tz).date() - placed).days > 30:
                continue
        if "final_sale" in rules and final[line["sku"]] and r["reason"] != "defective":
            continue
        qty = int(r["qty"])
        if "qty_cap" in rules:
            if used[line["line_id"]] + qty > int(line["qty"]):
                continue
            used[line["line_id"]] += qty
        if r["kind"] == "exchange" and "exchange" in rules:
            continue
        amount = int(line["unit_price_cents"]) * qty
        if "discount" in rules:
            amount -= int(line["discount_cents"]) * qty // int(line["qty"])
        refund[order["order_id"]] += amount
    return {k: v for k, v in sorted(refund.items()) if v > 0}
