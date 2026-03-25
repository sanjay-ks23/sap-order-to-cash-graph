"""SQLite database: JSONL ingestion, graph construction, and query functions.

Replaces the former Neo4j dependency with a self-contained in-memory SQLite
database, making the application zero-config and easy to deploy.
"""

from __future__ import annotations

import functools
import json
import logging
import sqlite3
import threading
from pathlib import Path

from config import DATASET_DIR

logger = logging.getLogger(__name__)

_local = threading.local()
_init_sql: str | None = None
_data_statements: list[tuple[str, list]] = []
_ready = False
_initializing = False
_cache: dict = {}


def _cached(fn):
    """Read-only cache. Dataset never changes so no invalidation needed."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (fn.__name__,) + args + tuple(sorted(kwargs.items()))
        if key not in _cache:
            _cache[key] = fn(*args, **kwargs)
        return _cache[key]
    return wrapper


# ---------------------------------------------------------------------------
# Connection handling (thread-safe in-memory DB via shared cache)
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    if not _ready and not _initializing:
        init_db()
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect("file:o2c?mode=memory&cache=shared", uri=True,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> dict:
    """Create schema, ingest JSONL data, return row-count metrics."""
    global _ready, _initializing
    if _ready:
        return get_stats()
    _initializing = True
    conn = _get_conn()
    _create_tables(conn)
    metrics = _ingest_all(conn)
    _create_indexes(conn)
    _ready = True
    _initializing = False
    logger.info("SQLite ingestion complete: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY, name TEXT, full_name TEXT, category TEXT,
    industry TEXT, country TEXT, city TEXT, region TEXT, street TEXT, postal_code TEXT
);
CREATE TABLE IF NOT EXISTS sales_orders (
    id TEXT PRIMARY KEY, customer_id TEXT, order_type TEXT, sales_org TEXT,
    creation_date TEXT, total_amount REAL, currency TEXT,
    delivery_status TEXT, billing_status TEXT, payment_terms TEXT
);
CREATE TABLE IF NOT EXISTS sales_order_items (
    id TEXT PRIMARY KEY, order_id TEXT, item_number TEXT, material_id TEXT,
    quantity REAL, amount REAL, currency TEXT, plant TEXT, item_category TEXT
);
CREATE TABLE IF NOT EXISTS deliveries (
    id TEXT PRIMARY KEY, creation_date TEXT, shipping_point TEXT,
    goods_movement_status TEXT, picking_status TEXT
);
CREATE TABLE IF NOT EXISTS delivery_items (
    id TEXT PRIMARY KEY, delivery_id TEXT, item_number TEXT,
    order_id TEXT, order_item TEXT, plant TEXT, quantity REAL
);
CREATE TABLE IF NOT EXISTS billing_documents (
    id TEXT PRIMARY KEY, doc_type TEXT, creation_date TEXT,
    total_amount REAL, currency TEXT, customer_id TEXT,
    is_cancelled INTEGER DEFAULT 0, accounting_doc TEXT
);
CREATE TABLE IF NOT EXISTS billing_items (
    id TEXT PRIMARY KEY, billing_id TEXT, item_number TEXT,
    material_id TEXT, quantity REAL, amount REAL, currency TEXT,
    delivery_id TEXT, delivery_item TEXT
);
CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY, item_number TEXT, reference_doc TEXT,
    clearing_doc TEXT, amount REAL, currency TEXT,
    customer_id TEXT, posting_date TEXT, doc_type TEXT
);
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY, item_number TEXT, clearing_doc TEXT,
    amount REAL, currency TEXT, customer_id TEXT,
    posting_date TEXT, clearing_date TEXT
);
CREATE TABLE IF NOT EXISTS plants (
    id TEXT PRIMARY KEY, name TEXT, sales_org TEXT, language TEXT
);
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY, product_type TEXT, description TEXT,
    gross_weight REAL, net_weight REAL, weight_unit TEXT,
    product_group TEXT, base_unit TEXT
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_so_customer   ON sales_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_soi_order     ON sales_order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_soi_material  ON sales_order_items(material_id);
CREATE INDEX IF NOT EXISTS idx_di_delivery   ON delivery_items(delivery_id);
CREATE INDEX IF NOT EXISTS idx_di_order      ON delivery_items(order_id);
CREATE INDEX IF NOT EXISTS idx_bd_customer   ON billing_documents(customer_id);
CREATE INDEX IF NOT EXISTS idx_bi_billing    ON billing_items(billing_id);
CREATE INDEX IF NOT EXISTS idx_bi_delivery   ON billing_items(delivery_id);
CREATE INDEX IF NOT EXISTS idx_bi_material   ON billing_items(material_id);
CREATE INDEX IF NOT EXISTS idx_je_ref        ON journal_entries(reference_doc);
CREATE INDEX IF NOT EXISTS idx_je_clearing   ON journal_entries(clearing_doc);
CREATE INDEX IF NOT EXISTS idx_pay_clearing  ON payments(clearing_doc);
CREATE INDEX IF NOT EXISTS idx_pay_customer  ON payments(customer_id);
"""


def _create_tables(conn: sqlite3.Connection):
    conn.executescript(_SCHEMA)


def _create_indexes(conn: sqlite3.Connection):
    conn.executescript(_INDEXES)


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def _load_jsonl(folder_name: str) -> list[dict]:
    folder = DATASET_DIR / folder_name
    if not folder.exists():
        logger.warning("Dataset folder missing: %s", folder)
        return []
    records: list[dict] = []
    for fpath in sorted(folder.glob("*.jsonl")):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _safe(record: dict, *keys, default="") -> str:
    for k in keys:
        val = record.get(k)
        if val is not None and str(val).strip():
            return str(val).strip()
    return default


def _safe_float(record: dict, *keys) -> float:
    for k in keys:
        val = record.get(k)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _ingest_all(conn: sqlite3.Connection) -> dict:
    metrics: dict[str, int] = {}

    # --- Customers + Addresses ---
    bp_records = _load_jsonl("business_partners")
    addr_records = _load_jsonl("business_partner_addresses")
    addr_map: dict[str, dict] = {}
    for a in addr_records:
        bp_id = _safe(a, "businessPartner")
        if bp_id:
            addr_map[bp_id] = a

    rows = []
    for r in bp_records:
        bp_id = _safe(r, "businessPartner", "customer")
        if not bp_id:
            continue
        addr = addr_map.get(bp_id, {})
        rows.append((
            bp_id,
            _safe(r, "businessPartnerName"),
            _safe(r, "businessPartnerFullName"),
            _safe(r, "businessPartnerCategory"),
            _safe(r, "industry"),
            _safe(addr, "country"),
            _safe(addr, "cityName"),
            _safe(addr, "region"),
            _safe(addr, "streetName"),
            _safe(addr, "postalCode"),
        ))
    conn.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    metrics["customers"] = len(rows)

    # --- Sales Orders ---
    rows = []
    for r in _load_jsonl("sales_order_headers"):
        so_id = _safe(r, "salesOrder")
        if not so_id:
            continue
        rows.append((
            so_id, _safe(r, "soldToParty"), _safe(r, "salesOrderType"),
            _safe(r, "salesOrganization"), _safe(r, "creationDate"),
            _safe_float(r, "totalNetAmount"), _safe(r, "transactionCurrency"),
            _safe(r, "overallDeliveryStatus"), _safe(r, "overallOrdReltdBillgStatus"),
            _safe(r, "customerPaymentTerms"),
        ))
    conn.executemany("INSERT OR IGNORE INTO sales_orders VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    metrics["sales_orders"] = len(rows)

    # --- Sales Order Items ---
    rows = []
    for r in _load_jsonl("sales_order_items"):
        so_id = _safe(r, "salesOrder")
        item = _safe(r, "salesOrderItem")
        if not so_id or not item:
            continue
        rows.append((
            f"{so_id}-{item}", so_id, item,
            _safe(r, "material"), _safe_float(r, "requestedQuantity"),
            _safe_float(r, "netAmount"), _safe(r, "transactionCurrency"),
            _safe(r, "productionPlant"), _safe(r, "salesOrderItemCategory"),
        ))
    conn.executemany("INSERT OR IGNORE INTO sales_order_items VALUES (?,?,?,?,?,?,?,?,?)", rows)
    metrics["sales_order_items"] = len(rows)

    # --- Deliveries ---
    rows = []
    for r in _load_jsonl("outbound_delivery_headers"):
        d_id = _safe(r, "deliveryDocument")
        if not d_id:
            continue
        rows.append((
            d_id, _safe(r, "creationDate"), _safe(r, "shippingPoint"),
            _safe(r, "overallGoodsMovementStatus"), _safe(r, "overallPickingStatus"),
        ))
    conn.executemany("INSERT OR IGNORE INTO deliveries VALUES (?,?,?,?,?)", rows)
    metrics["deliveries"] = len(rows)

    # --- Delivery Items ---
    rows = []
    for r in _load_jsonl("outbound_delivery_items"):
        d_id = _safe(r, "deliveryDocument")
        item = _safe(r, "deliveryDocumentItem")
        if not d_id or not item:
            continue
        rows.append((
            f"{d_id}-{item}", d_id, item,
            _safe(r, "referenceSdDocument"), _safe(r, "referenceSdDocumentItem"),
            _safe(r, "plant"), _safe_float(r, "actualDeliveryQuantity"),
        ))
    conn.executemany("INSERT OR IGNORE INTO delivery_items VALUES (?,?,?,?,?,?,?)", rows)
    metrics["delivery_items"] = len(rows)

    # --- Billing Documents ---
    rows = []
    for r in _load_jsonl("billing_document_headers"):
        b_id = _safe(r, "billingDocument")
        if not b_id:
            continue
        rows.append((
            b_id, _safe(r, "billingDocumentType"), _safe(r, "creationDate"),
            _safe_float(r, "totalNetAmount"), _safe(r, "transactionCurrency"),
            _safe(r, "soldToParty"),
            1 if r.get("billingDocumentIsCancelled") else 0,
            _safe(r, "accountingDocument"),
        ))
    conn.executemany("INSERT OR IGNORE INTO billing_documents VALUES (?,?,?,?,?,?,?,?)", rows)
    metrics["billing_documents"] = len(rows)

    # --- Billing Items ---
    rows = []
    for r in _load_jsonl("billing_document_items"):
        b_id = _safe(r, "billingDocument")
        item = _safe(r, "billingDocumentItem")
        if not b_id or not item:
            continue
        rows.append((
            f"{b_id}-{item}", b_id, item,
            _safe(r, "material"), _safe_float(r, "billingQuantity"),
            _safe_float(r, "netAmount"), _safe(r, "transactionCurrency"),
            _safe(r, "referenceSdDocument"), _safe(r, "referenceSdDocumentItem"),
        ))
    conn.executemany("INSERT OR IGNORE INTO billing_items VALUES (?,?,?,?,?,?,?,?,?)", rows)
    metrics["billing_items"] = len(rows)

    # --- Journal Entries ---
    rows = []
    for r in _load_jsonl("journal_entry_items_accounts_receivable"):
        je_id = _safe(r, "accountingDocument")
        item = _safe(r, "accountingDocumentItem")
        if not je_id:
            continue
        rows.append((
            je_id, item, _safe(r, "referenceDocument"),
            _safe(r, "clearingAccountingDocument"),
            _safe_float(r, "amountInTransactionCurrency"),
            _safe(r, "transactionCurrency"), _safe(r, "customer"),
            _safe(r, "postingDate"), _safe(r, "accountingDocumentType"),
        ))
    conn.executemany("INSERT OR REPLACE INTO journal_entries VALUES (?,?,?,?,?,?,?,?,?)", rows)
    metrics["journal_entries"] = len(rows)

    # --- Payments ---
    rows = []
    for r in _load_jsonl("payments_accounts_receivable"):
        p_id = _safe(r, "accountingDocument")
        item = _safe(r, "accountingDocumentItem")
        if not p_id:
            continue
        rows.append((
            p_id, item, _safe(r, "clearingAccountingDocument"),
            _safe_float(r, "amountInTransactionCurrency"),
            _safe(r, "transactionCurrency"), _safe(r, "customer"),
            _safe(r, "postingDate"), _safe(r, "clearingDate"),
        ))
    conn.executemany("INSERT OR REPLACE INTO payments VALUES (?,?,?,?,?,?,?,?)", rows)
    metrics["payments"] = len(rows)

    # --- Plants ---
    rows = []
    for r in _load_jsonl("plants"):
        p_id = _safe(r, "plant")
        if not p_id:
            continue
        rows.append((p_id, _safe(r, "plantName"), _safe(r, "salesOrganization"), _safe(r, "language")))
    conn.executemany("INSERT OR IGNORE INTO plants VALUES (?,?,?,?)", rows)
    metrics["plants"] = len(rows)

    # --- Products + Descriptions ---
    desc_map: dict[str, str] = {}
    for r in _load_jsonl("product_descriptions"):
        pid = _safe(r, "product")
        if pid:
            desc_map[pid] = _safe(r, "productDescription")

    rows = []
    for r in _load_jsonl("products"):
        p_id = _safe(r, "product")
        if not p_id:
            continue
        rows.append((
            p_id, _safe(r, "productType"), desc_map.get(p_id, ""),
            _safe_float(r, "grossWeight"), _safe_float(r, "netWeight"),
            _safe(r, "weightUnit"), _safe(r, "productGroup"), _safe(r, "baseUnit"),
        ))
    conn.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?)", rows)
    metrics["products"] = len(rows)

    conn.commit()
    return metrics


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

@_cached
def build_graph_root() -> dict:
    """Build the initial graph with all core-flow entity nodes and relationships."""
    db = _get_conn()
    nodes: list[dict] = []
    edges: list[dict] = []
    node_ids: set[str] = set()

    def _add(nid: str, node: dict):
        if nid not in node_ids:
            node_ids.add(nid)
            nodes.append(node)

    for r in db.execute("SELECT * FROM customers").fetchall():
        nid = f"CUST-{r['id']}"
        _add(nid, {"id": nid, "raw_id": r["id"], "node_type": "Customer",
                    "label": r["name"] or r["id"], "name": r["name"],
                    "industry": r["industry"], "country": r["country"], "city": r["city"]})

    for r in db.execute("SELECT * FROM sales_orders").fetchall():
        nid = f"SO-{r['id']}"
        _add(nid, {"id": nid, "raw_id": r["id"], "node_type": "SalesOrder",
                    "label": f"SO-{r['id']}", "total_amount": r["total_amount"],
                    "currency": r["currency"], "delivery_status": r["delivery_status"],
                    "creation_date": r["creation_date"]})
        if r["customer_id"]:
            edges.append({"source": f"CUST-{r['customer_id']}", "target": nid, "type": "PLACED"})

    for r in db.execute("SELECT * FROM deliveries").fetchall():
        nid = f"DLV-{r['id']}"
        _add(nid, {"id": nid, "raw_id": r["id"], "node_type": "Delivery",
                    "label": f"DLV-{r['id']}", "creation_date": r["creation_date"],
                    "goods_movement_status": r["goods_movement_status"]})

    for r in db.execute(
        "SELECT DISTINCT di.order_id, di.delivery_id FROM delivery_items di "
        "WHERE di.order_id != '' AND di.delivery_id != ''"
    ).fetchall():
        edges.append({"source": f"SO-{r['order_id']}", "target": f"DLV-{r['delivery_id']}", "type": "DELIVERED_IN"})

    for r in db.execute("SELECT * FROM billing_documents").fetchall():
        nid = f"BILL-{r['id']}"
        _add(nid, {"id": nid, "raw_id": r["id"], "node_type": "BillingDocument",
                    "label": f"BILL-{r['id']}", "total_amount": r["total_amount"],
                    "currency": r["currency"], "creation_date": r["creation_date"]})

    for r in db.execute(
        "SELECT DISTINCT bi.delivery_id, bi.billing_id FROM billing_items bi "
        "WHERE bi.delivery_id != '' AND bi.billing_id != ''"
    ).fetchall():
        edges.append({"source": f"DLV-{r['delivery_id']}", "target": f"BILL-{r['billing_id']}", "type": "BILLED_IN"})

    for r in db.execute(
        "SELECT DISTINCT id, reference_doc, clearing_doc, amount, currency, posting_date "
        "FROM journal_entries"
    ).fetchall():
        nid = f"JE-{r['id']}"
        _add(nid, {"id": nid, "raw_id": r["id"], "node_type": "JournalEntry",
                    "label": f"JE-{r['id']}", "amount": r["amount"], "currency": r["currency"]})
        if r["reference_doc"]:
            edges.append({"source": f"BILL-{r['reference_doc']}", "target": nid, "type": "GENERATES_JOURNAL"})

    for r in db.execute(
        "SELECT DISTINCT id, clearing_doc, amount, currency, customer_id, clearing_date "
        "FROM payments"
    ).fetchall():
        nid = f"PAY-{r['id']}"
        _add(nid, {"id": nid, "raw_id": r["id"], "node_type": "Payment",
                    "label": f"PAY-{r['id']}", "amount": r["amount"], "currency": r["currency"],
                    "clearing_date": r["clearing_date"]})

    for r in db.execute(
        "SELECT DISTINCT j.id AS je_id, p.id AS pay_id FROM journal_entries j "
        "JOIN payments p ON j.clearing_doc = p.clearing_doc "
        "WHERE j.clearing_doc != '' AND j.clearing_doc IS NOT NULL"
    ).fetchall():
        edges.append({"source": f"JE-{r['je_id']}", "target": f"PAY-{r['pay_id']}", "type": "CLEARED_BY"})

    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]
    return {"nodes": nodes, "edges": edges}


def expand_node(node_id: str) -> dict:
    """Return immediate neighbors of a node for progressive graph exploration."""
    db = _get_conn()
    prefix, _, raw_id = node_id.partition("-")
    if not raw_id:
        return {"nodes": [], "edges": [], "error": f"Invalid node ID: {node_id}"}

    center = get_node_detail(node_id)
    if "error" in center:
        return {"nodes": [], "edges": [], "error": center["error"]}

    nodes: dict[str, dict] = {node_id: center}
    edges: list[dict] = []

    def _add_nodes(rows, prefix_str, node_type, label_fn=None, extra=None):
        for r in rows:
            nid = f"{prefix_str}-{r['id']}"
            node = {"id": nid, "raw_id": r["id"], "node_type": node_type,
                    "label": label_fn(r) if label_fn else nid}
            if extra:
                for k in extra:
                    if k in r.keys():
                        node[k] = r[k]
            nodes[nid] = node
            return nid
        return None

    if prefix == "CUST":
        for r in db.execute("SELECT * FROM sales_orders WHERE customer_id=?", (raw_id,)):
            nid = f"SO-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "SalesOrder",
                          "label": f"SO-{r['id']}", "total_amount": r["total_amount"],
                          "currency": r["currency"]}
            edges.append({"source": node_id, "target": nid, "type": "PLACED"})

    elif prefix == "SO":
        for r in db.execute(
            "SELECT c.* FROM customers c JOIN sales_orders so ON so.customer_id=c.id WHERE so.id=?",
            (raw_id,),
        ):
            nid = f"CUST-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "Customer",
                          "label": r["name"] or r["id"]}
            edges.append({"source": nid, "target": node_id, "type": "PLACED"})

        for r in db.execute("SELECT * FROM sales_order_items WHERE order_id=?", (raw_id,)):
            nid = f"SOI-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "SalesOrderItem",
                          "label": f"SOI-{r['id']}", "material_id": r["material_id"],
                          "quantity": r["quantity"], "amount": r["amount"]}
            edges.append({"source": node_id, "target": nid, "type": "HAS_ITEM"})

        for r in db.execute(
            "SELECT DISTINCT d.* FROM deliveries d "
            "JOIN delivery_items di ON di.delivery_id=d.id WHERE di.order_id=?",
            (raw_id,),
        ):
            nid = f"DLV-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "Delivery",
                          "label": f"DLV-{r['id']}"}
            edges.append({"source": node_id, "target": nid, "type": "DELIVERED_IN"})

    elif prefix == "DLV":
        for r in db.execute(
            "SELECT DISTINCT so.* FROM sales_orders so "
            "JOIN delivery_items di ON di.order_id=so.id WHERE di.delivery_id=?",
            (raw_id,),
        ):
            nid = f"SO-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "SalesOrder",
                          "label": f"SO-{r['id']}"}
            edges.append({"source": nid, "target": node_id, "type": "DELIVERED_IN"})

        for r in db.execute(
            "SELECT DISTINCT bd.* FROM billing_documents bd "
            "JOIN billing_items bi ON bi.billing_id=bd.id WHERE bi.delivery_id=?",
            (raw_id,),
        ):
            nid = f"BILL-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "BillingDocument",
                          "label": f"BILL-{r['id']}"}
            edges.append({"source": node_id, "target": nid, "type": "BILLED_IN"})

        for r in db.execute("SELECT * FROM delivery_items WHERE delivery_id=?", (raw_id,)):
            nid = f"DLI-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "DeliveryItem",
                          "label": f"DLI-{r['id']}"}
            edges.append({"source": nid, "target": node_id, "type": "PART_OF_DELIVERY"})

    elif prefix == "BILL":
        for r in db.execute(
            "SELECT DISTINCT d.* FROM deliveries d "
            "JOIN billing_items bi ON bi.delivery_id=d.id WHERE bi.billing_id=?",
            (raw_id,),
        ):
            nid = f"DLV-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "Delivery",
                          "label": f"DLV-{r['id']}"}
            edges.append({"source": nid, "target": node_id, "type": "BILLED_IN"})

        for r in db.execute("SELECT * FROM journal_entries WHERE reference_doc=?", (raw_id,)):
            nid = f"JE-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "JournalEntry",
                          "label": f"JE-{r['id']}"}
            edges.append({"source": node_id, "target": nid, "type": "GENERATES_JOURNAL"})

        for r in db.execute("SELECT * FROM billing_items WHERE billing_id=?", (raw_id,)):
            nid = f"BLI-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "BillingItem",
                          "label": f"BLI-{r['id']}", "material_id": r["material_id"]}
            edges.append({"source": nid, "target": node_id, "type": "PART_OF_BILLING"})

    elif prefix == "JE":
        for r in db.execute(
            "SELECT bd.* FROM billing_documents bd "
            "JOIN journal_entries je ON je.reference_doc=bd.id WHERE je.id=?",
            (raw_id,),
        ):
            nid = f"BILL-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "BillingDocument",
                          "label": f"BILL-{r['id']}"}
            edges.append({"source": nid, "target": node_id, "type": "GENERATES_JOURNAL"})

        for r in db.execute(
            "SELECT p.* FROM payments p "
            "JOIN journal_entries je ON je.clearing_doc=p.clearing_doc "
            "WHERE je.id=? AND je.clearing_doc != ''",
            (raw_id,),
        ):
            nid = f"PAY-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "Payment",
                          "label": f"PAY-{r['id']}"}
            edges.append({"source": node_id, "target": nid, "type": "CLEARED_BY"})

    elif prefix == "PAY":
        for r in db.execute(
            "SELECT je.* FROM journal_entries je "
            "JOIN payments p ON p.clearing_doc=je.clearing_doc "
            "WHERE p.id=? AND p.clearing_doc != ''",
            (raw_id,),
        ):
            nid = f"JE-{r['id']}"
            nodes[nid] = {"id": nid, "raw_id": r["id"], "node_type": "JournalEntry",
                          "label": f"JE-{r['id']}"}
            edges.append({"source": nid, "target": node_id, "type": "CLEARED_BY"})

    elif prefix in ("SOI", "BLI", "DLI"):
        if prefix == "SOI":
            r = db.execute("SELECT * FROM sales_order_items WHERE id=?", (raw_id,)).fetchone()
            if r and r["material_id"]:
                pr = db.execute("SELECT * FROM products WHERE id=?", (r["material_id"],)).fetchone()
                if pr:
                    nid = f"PRD-{pr['id']}"
                    nodes[nid] = {"id": nid, "raw_id": pr["id"], "node_type": "Product",
                                  "label": pr["description"] or pr["id"]}
                    edges.append({"source": node_id, "target": nid, "type": "PRODUCT_OF"})

    return {"nodes": list(nodes.values()), "edges": edges,
            "truncated": len(nodes) > 20}


# ---------------------------------------------------------------------------
# Node detail
# ---------------------------------------------------------------------------

_TABLE_MAP = {
    "CUST": ("customers", "Customer"),
    "SO":   ("sales_orders", "SalesOrder"),
    "SOI":  ("sales_order_items", "SalesOrderItem"),
    "DLV":  ("deliveries", "Delivery"),
    "DLI":  ("delivery_items", "DeliveryItem"),
    "BILL": ("billing_documents", "BillingDocument"),
    "BLI":  ("billing_items", "BillingItem"),
    "JE":   ("journal_entries", "JournalEntry"),
    "PAY":  ("payments", "Payment"),
    "PLT":  ("plants", "Plant"),
    "PRD":  ("products", "Product"),
}


def get_node_detail(node_id: str) -> dict:
    db = _get_conn()
    prefix, _, raw_id = node_id.partition("-")
    if not raw_id or prefix not in _TABLE_MAP:
        return {"error": f"Unknown node: {node_id}"}

    table, node_type = _TABLE_MAP[prefix]
    row = db.execute(f"SELECT * FROM {table} WHERE id=?", (raw_id,)).fetchone()
    if not row:
        return {"error": f"Node {node_id} not found"}

    result = dict(row)
    result["id"] = node_id
    result["node_type"] = node_type
    result["label"] = node_id
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@_cached
def get_stats() -> dict:
    db = _get_conn()
    by_type: dict[str, int] = {}
    for table in ["customers", "sales_orders", "sales_order_items", "deliveries",
                   "delivery_items", "billing_documents", "billing_items",
                   "journal_entries", "payments", "plants", "products"]:
        by_type[table] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    total_nodes = sum(by_type.values())

    edge_count = (
        db.execute("SELECT count(*) FROM sales_orders WHERE customer_id != ''").fetchone()[0]
        + db.execute("SELECT count(DISTINCT order_id||'-'||delivery_id) FROM delivery_items WHERE order_id!=''").fetchone()[0]
        + db.execute("SELECT count(DISTINCT delivery_id||'-'||billing_id) FROM billing_items WHERE delivery_id!=''").fetchone()[0]
        + db.execute("SELECT count(*) FROM journal_entries WHERE reference_doc!=''").fetchone()[0]
        + db.execute(
            "SELECT count(*) FROM journal_entries j JOIN payments p ON j.clearing_doc=p.clearing_doc "
            "WHERE j.clearing_doc!=''"
        ).fetchone()[0]
    )

    return {"total_nodes": total_nodes, "total_edges": edge_count, "by_type": by_type}


# ---------------------------------------------------------------------------
# Business query functions (used by chat intents)
# ---------------------------------------------------------------------------

def trace_order_flow(order_id: str) -> dict:
    db = _get_conn()
    order = db.execute("SELECT * FROM sales_orders WHERE id=?", (order_id,)).fetchone()
    if not order:
        return {"error": f"Sales order {order_id} not found"}

    customer = db.execute("SELECT * FROM customers WHERE id=?", (order["customer_id"],)).fetchone()
    items = db.execute("SELECT * FROM sales_order_items WHERE order_id=?", (order_id,)).fetchall()

    deliveries = db.execute(
        "SELECT DISTINCT d.* FROM deliveries d "
        "JOIN delivery_items di ON di.delivery_id=d.id WHERE di.order_id=?",
        (order_id,),
    ).fetchall()

    d_ids = [d["id"] for d in deliveries]
    billings = _in_query(db, "SELECT DISTINCT bd.* FROM billing_documents bd "
                         "JOIN billing_items bi ON bi.billing_id=bd.id WHERE bi.delivery_id IN",
                         d_ids)

    b_ids = [b["id"] for b in billings]
    journals = _in_query(db, "SELECT DISTINCT * FROM journal_entries WHERE reference_doc IN", b_ids)

    clearing = [j["clearing_doc"] for j in journals if j["clearing_doc"]]
    pays = _in_query(db, "SELECT DISTINCT * FROM payments WHERE clearing_doc IN", clearing)

    traversed = [f"SO-{order_id}"]
    if customer:
        traversed.append(f"CUST-{customer['id']}")
    traversed += [f"DLV-{d['id']}" for d in deliveries]
    traversed += [f"BILL-{b['id']}" for b in billings]
    traversed += [f"JE-{j['id']}" for j in journals]
    traversed += [f"PAY-{p['id']}" for p in pays]

    return {
        "order": _row_dict(order, "SalesOrder"),
        "customer": _row_dict(customer, "Customer") if customer else None,
        "items": [_row_dict(i, "SalesOrderItem") for i in items],
        "deliveries": [_row_dict(d, "Delivery") for d in deliveries],
        "billings": [_row_dict(b, "BillingDocument") for b in billings],
        "journals": [_row_dict(j, "JournalEntry") for j in journals],
        "payments": [_row_dict(p, "Payment") for p in pays],
        "nodes_traversed": traversed,
    }


def trace_billing_flow(billing_id: str) -> dict:
    db = _get_conn()
    billing = db.execute("SELECT * FROM billing_documents WHERE id=?", (billing_id,)).fetchone()
    if not billing:
        return {"error": f"Billing document {billing_id} not found"}

    items = db.execute("SELECT * FROM billing_items WHERE billing_id=?", (billing_id,)).fetchall()
    d_ids = list({i["delivery_id"] for i in items if i["delivery_id"]})
    deliveries = _in_query(db, "SELECT * FROM deliveries WHERE id IN", d_ids)
    orders = _in_query(db,
        "SELECT DISTINCT so.* FROM sales_orders so "
        "JOIN delivery_items di ON di.order_id=so.id WHERE di.delivery_id IN", d_ids)

    customers = []
    if billing["customer_id"]:
        c = db.execute("SELECT * FROM customers WHERE id=?", (billing["customer_id"],)).fetchone()
        if c:
            customers = [c]

    journals = db.execute("SELECT * FROM journal_entries WHERE reference_doc=?", (billing_id,)).fetchall()
    clearing = [j["clearing_doc"] for j in journals if j["clearing_doc"]]
    pays = _in_query(db, "SELECT * FROM payments WHERE clearing_doc IN", clearing)

    traversed = [f"BILL-{billing_id}"]
    traversed += [f"DLV-{d['id']}" for d in deliveries]
    traversed += [f"SO-{o['id']}" for o in orders]
    traversed += [f"CUST-{c['id']}" for c in customers]
    traversed += [f"JE-{j['id']}" for j in journals]
    traversed += [f"PAY-{p['id']}" for p in pays]

    return {
        "billing": _row_dict(billing, "BillingDocument"),
        "items": [_row_dict(i, "BillingItem") for i in items],
        "deliveries": [_row_dict(d, "Delivery") for d in deliveries],
        "orders": [_row_dict(o, "SalesOrder") for o in orders],
        "customers": [_row_dict(c, "Customer") for c in customers],
        "journals": [_row_dict(j, "JournalEntry") for j in journals],
        "payments": [_row_dict(p, "Payment") for p in pays],
        "nodes_traversed": traversed,
    }


@_cached
def top_products_by_billing(limit: int = 10) -> dict:
    db = _get_conn()
    rows = db.execute(
        "SELECT p.id, p.description, p.product_type, COUNT(DISTINCT bi.billing_id) AS billing_count "
        "FROM products p JOIN billing_items bi ON bi.material_id=p.id "
        "GROUP BY p.id ORDER BY billing_count DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {"products": [_row_dict(r, "Product") for r in rows], "total": len(rows)}


@_cached
def find_incomplete_orders() -> dict:
    db = _get_conn()
    rows = db.execute("""
        SELECT so.id, so.customer_id, so.total_amount, so.currency,
               so.creation_date, so.delivery_status,
               MAX(CASE WHEN di.delivery_id IS NOT NULL THEN 1 ELSE 0 END) AS has_delivery,
               MAX(CASE WHEN bi.billing_id IS NOT NULL THEN 1 ELSE 0 END) AS has_billing
        FROM sales_orders so
        LEFT JOIN delivery_items di ON di.order_id = so.id
        LEFT JOIN billing_items bi ON bi.delivery_id = di.delivery_id
        GROUP BY so.id
        HAVING has_delivery = 0 OR has_billing = 0
    """).fetchall()

    incomplete = []
    for r in rows:
        issues = []
        if not r["has_delivery"]:
            issues.append("no delivery")
        if not r["has_billing"]:
            issues.append("no billing")
        d = _row_dict(r, "SalesOrder")
        d["issue"] = ", ".join(issues)
        incomplete.append(d)

    return {"incomplete_orders": incomplete, "total": len(incomplete)}


def get_customer_orders(customer_id: str) -> dict:
    db = _get_conn()
    customer = db.execute("SELECT * FROM customers WHERE id=?", (customer_id,)).fetchone()
    if not customer:
        return {"error": f"Customer {customer_id} not found"}
    orders = db.execute("SELECT * FROM sales_orders WHERE customer_id=?", (customer_id,)).fetchall()
    traversed = [f"CUST-{customer_id}"] + [f"SO-{o['id']}" for o in orders]
    return {
        "customer": _row_dict(customer, "Customer"),
        "orders": [_row_dict(o, "SalesOrder") for o in orders],
        "total_orders": len(orders),
        "nodes_traversed": traversed,
    }


def get_customer_info(customer_id: str) -> dict:
    result = get_customer_orders(customer_id)
    if "error" in result:
        return result
    db = _get_conn()
    bc = db.execute("SELECT count(DISTINCT id) FROM billing_documents WHERE customer_id=?",
                    (customer_id,)).fetchone()[0]
    return {
        "customer": result["customer"],
        "total_orders": result["total_orders"],
        "total_billings": bc,
        "nodes_traversed": result["nodes_traversed"],
    }


def get_product_info(product_id: str) -> dict:
    db = _get_conn()
    product = db.execute("SELECT * FROM products WHERE id=?", (product_id,)).fetchone()
    if not product:
        return {"error": f"Product {product_id} not found"}
    soi = db.execute("SELECT count(DISTINCT id) FROM sales_order_items WHERE material_id=?",
                     (product_id,)).fetchone()[0]
    bi = db.execute("SELECT count(DISTINCT id) FROM billing_items WHERE material_id=?",
                    (product_id,)).fetchone()[0]
    return {
        "product": _row_dict(product, "Product"),
        "order_item_count": soi,
        "billing_item_count": bi,
        "nodes_traversed": [f"PRD-{product_id}"],
    }


def find_customer_by_name(name_fragment: str) -> list[dict]:
    """Fuzzy search for customers by name (case-insensitive partial match)."""
    conn = _get_conn()
    pattern = f"%{name_fragment}%"
    rows = conn.execute(
        "SELECT id, name, full_name FROM customers WHERE "
        "name LIKE ? COLLATE NOCASE OR full_name LIKE ? COLLATE NOCASE",
        (pattern, pattern),
    ).fetchall()
    return [dict(r) for r in rows]


def find_product_by_name(name_fragment: str) -> list[dict]:
    """Fuzzy search for products by description (case-insensitive partial match)."""
    conn = _get_conn()
    pattern = f"%{name_fragment}%"
    rows = conn.execute(
        "SELECT id, description FROM products WHERE "
        "description LIKE ? COLLATE NOCASE OR id LIKE ? COLLATE NOCASE",
        (pattern, pattern),
    ).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    return get_stats()


def run_sql(sql: str) -> dict:
    """Execute a read-only SQL query safely."""
    db = _get_conn()
    sql_stripped = sql.strip()
    sql_lower = sql_stripped.lower()

    if not sql_lower.startswith("select"):
        return {"error": "Only SELECT queries are allowed."}

    for kw in ("drop", "delete", "update", "insert", "alter", "create", "pragma", "attach"):
        if kw in sql_lower:
            return {"error": f"Forbidden SQL keyword: {kw}"}

    try:
        rows = db.execute(sql_stripped).fetchall()
        results = [dict(r) for r in rows[:500]]
        return {"rows": results, "count": len(results)}
    except Exception as e:
        return {"error": f"SQL error: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_dict(row, node_type: str) -> dict:
    if row is None:
        return {}
    d = dict(row)
    d["node_type"] = node_type
    return d


def _in_query(db: sqlite3.Connection, base_sql: str, values: list) -> list:
    if not values:
        return []
    placeholders = ",".join("?" * len(values))
    return db.execute(f"{base_sql} ({placeholders})", values).fetchall()
