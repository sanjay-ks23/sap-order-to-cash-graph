"""Gemini LLM integration: intent classification, NL-to-SQL, response formatting, guardrails.

Uses rule-based intent matching first (sub-millisecond), falling back to
Gemini only when patterns don't match. Response formatting is fully
template-based — no LLM call required.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import google.generativeai as genai

import database as db_module
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

_model: genai.GenerativeModel | None = None
_executor = ThreadPoolExecutor(max_workers=2)
_LLM_TIMEOUT = 15


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(GEMINI_MODEL)
    return _model


def _call_llm(prompt: str, max_tokens: int = 2048) -> str:
    """Call Gemini with a timeout to prevent hanging."""
    model = _get_model()

    def _do():
        return model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=max_tokens),
        )

    future = _executor.submit(_do)
    response = future.result(timeout=_LLM_TIMEOUT)
    return response.text


# ---------------------------------------------------------------------------
# Rule-based intent classification (Change 1)
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks?|"
    r"thank\s+you|bye|goodbye|cheers)\s*[!.?]*$", re.I,
)

_CREATIVE_RE = re.compile(
    r"(?:write\s+(?:a\s+)?(?:poem|story|essay|song|code)|"
    r"tell\s+(?:me\s+)?a\s+joke|"
    r"what\s+(?:is|are)\s+(?:the\s+)?(?:meaning\s+of\s+life|capital\s+of|weather|population)|"
    r"who\s+(?:is|was|are)\s+(?!customer))", re.I,
)

_CONTEXT_WORDS = {"it", "that", "this", "same", "previous", "above", "those", "them"}


_ENTITY_WORDS_RE = re.compile(
    r"\b(?:customers?|cust|orders?|products?|materials?|"
    r"billings?|invoices?|deliver(?:y|ies)|payments?|"
    r"journals?|plants?)\b", re.I,
)

_ANALYTICAL_RE = re.compile(
    r"\b(?:how many|count|total|average|avg|sum|max|min|"
    r"highest|lowest|most|least|compare|ratio|percent|"
    r"no\s+of|number\s+of|"
    r"rank|top\s+\d+\s+\w+\s+(?:by|for|per))\b", re.I,
)


def _resolve_customer_ref(query: str) -> tuple[str | None, str | None]:
    """Extract and resolve customer reference (by ID or name). Returns (id, name)."""
    ql = query.lower()

    m = re.search(r"(?:customer|cust)\s+(?:whose\s+)?(?:id\s+(?:is\s+)?)?(\d{6,})", ql)
    if not m:
        m = re.search(r"(?:id\s*(?:is|=|:)\s*)(\d{6,})", ql)
    if m:
        return m.group(1), None

    name_patterns = [
        r"(?:customer|cust)[:\s]+(?:customer\s+)?(.+?)(?:$|\s+(?:order|bill|deliver|pay|how|what|which|have|has))",
        r"(?:orders?|billings?|deliver(?:y|ies)|payments?)\s+(?:for|by|of|from|made\s+by|placed\s+by)\s+(?:customer[:\s]*)?(?:customer\s+)?(.+?)$",
        r"(?:what|which|how many|total|sum|average|avg|count|no\s+of|number\s+of).+?(?:for|by|from|of)\s+(?:customer[:\s]*)?(?:customer\s+)?(.+?)$",
        r"(?:did|does|has)\s+(?:customer\s+)?(.+?)\s+(?:order|place|buy|purchase|have|made?)",
        r"(?:for|by|from|of)\s+(?:customer[:\s]*)?(.+?)$",
    ]
    for pat in name_patterns:
        m = re.search(pat, query, re.I)
        if m:
            name = m.group(1).strip().rstrip(".!?,;:")
            if len(name) < 2 or re.match(r"^(?:the|a|an|any|some|all|each|every)$", name, re.I):
                continue
            if name.isdigit():
                cid = _lookup_customer_by_short_id(name)
                if cid:
                    return cid, None
                continue
            matches = db_module.find_customer_by_name(name)
            if matches:
                return matches[0]["id"], matches[0].get("name", name)

    m = re.search(r"(?:customer|cust)\s+(?:whose\s+)?(?:id\s+(?:is\s+)?)?(\d+)", ql)
    if not m:
        m = re.search(r"(?:id\s*(?:is|=|:)\s*)(\d+)", ql)
    if m:
        cid = _lookup_customer_by_short_id(m.group(1))
        if cid:
            return cid, None
        return m.group(1), None

    return None, None


def _lookup_customer_by_short_id(ref: str) -> str | None:
    """Resolve short customer ID references by searching the database."""
    conn = db_module._get_conn()
    row = conn.execute("SELECT id FROM customers WHERE id=?", (ref,)).fetchone()
    if row:
        return row["id"]
    rows = conn.execute("SELECT id FROM customers WHERE id LIKE ?", (f"%{ref}",)).fetchall()
    if len(rows) == 1:
        return rows[0]["id"]
    rows = conn.execute("SELECT id FROM customers WHERE id LIKE ?", (f"%{ref}%",)).fetchall()
    if len(rows) == 1:
        return rows[0]["id"]
    return None


def _try_product_name_match(q: str, ql: str) -> dict | None:
    """Resolve product names in queries and generate appropriate SQL."""
    m = re.search(r"(?:product|material)\s+([A-Z][\w\s]+)", q)
    if not m:
        m = re.search(r"(?:for|about|of)\s+(?:product\s+)?([A-Z][A-Z\d\s+\-]+)", q)
    if not m:
        return None
    name = m.group(1).strip()
    if len(name) < 3:
        return None
    matches = db_module.find_product_by_name(name)
    if not matches:
        return None
    pid = matches[0]["id"]
    if _ANALYTICAL_RE.search(ql):
        if re.search(r"\b(?:how many|count|no\s+of|number\s+of)\b.*\b(?:order|billing|deliver)", ql):
            entity = "sales_order_items" if "order" in ql else ("billing_items" if "billing" in ql else "delivery_items")
            return {"intent": "SQL_QUERY", "params": {
                "query": q,
                "generated_sql": f"SELECT COUNT(*) as count FROM {entity} WHERE material_id='{pid}'"
            }, "source": "rules"}
    return {"intent": "PRODUCT_INFO", "params": {"product_id": pid}, "source": "rules"}


def _identify_document_type(doc_id: str) -> str | None:
    """Determine entity type by checking which table contains the ID."""
    conn = db_module._get_conn()
    for table, etype in [
        ("billing_documents", "billing"), ("sales_orders", "order"),
        ("deliveries", "delivery"), ("journal_entries", "journal"),
        ("payments", "payment"),
    ]:
        if conn.execute(f"SELECT 1 FROM {table} WHERE id=?", (doc_id,)).fetchone():
            return etype
    return None


def _try_entity_link(q: str, ql: str) -> dict | None:
    """Handle 'find X linked to Y' and '[number] - find X linked to this' queries."""
    m = re.search(r"(\d{6,})\s*[-–—:]\s*.+?(?:link|connect|relat|associat|tied|from|for)", ql)
    if not m:
        m = re.search(
            r"(?:find|what|which|get|show)\s+.*?"
            r"(?:link|connect|relat|associat|tied)\s+(?:to|with)\s+.*?(\d{6,})", ql
        )
    if not m:
        return None

    doc_id = m.group(1)
    target = None
    for entity in ["journal", "billing", "order", "delivery", "payment"]:
        if entity in ql:
            target = entity
            break
    if not target:
        return None

    source = None
    for kw, etype in [("billing", "billing"), ("invoice", "billing"),
                       ("order", "order"), ("delivery", "delivery"),
                       ("journal", "journal"), ("payment", "payment")]:
        if kw in ql and etype != target:
            source = etype
            break
    if not source:
        source = _identify_document_type(doc_id)
    if not source:
        return None

    link_sql = {
        ("billing", "journal"): f"SELECT je.id, je.amount, je.currency, je.posting_date FROM journal_entries je WHERE je.reference_doc='{doc_id}'",
        ("billing", "order"): f"SELECT DISTINCT so.id, so.total_amount, so.currency FROM sales_orders so JOIN delivery_items di ON di.order_id=so.id JOIN billing_items bi ON bi.delivery_id=di.delivery_id WHERE bi.billing_id='{doc_id}'",
        ("billing", "delivery"): f"SELECT DISTINCT d.id, d.creation_date FROM deliveries d JOIN billing_items bi ON bi.delivery_id=d.id WHERE bi.billing_id='{doc_id}'",
        ("billing", "payment"): f"SELECT DISTINCT p.id, p.amount, p.currency FROM payments p JOIN journal_entries je ON je.clearing_doc=p.clearing_doc WHERE je.reference_doc='{doc_id}'",
        ("order", "delivery"): f"SELECT DISTINCT d.id, d.creation_date FROM deliveries d JOIN delivery_items di ON di.delivery_id=d.id WHERE di.order_id='{doc_id}'",
        ("order", "billing"): f"SELECT DISTINCT bd.id, bd.total_amount, bd.currency FROM billing_documents bd JOIN billing_items bi ON bi.billing_id=bd.id JOIN delivery_items di ON bi.delivery_id=di.delivery_id WHERE di.order_id='{doc_id}'",
        ("delivery", "order"): f"SELECT DISTINCT so.id, so.total_amount FROM sales_orders so JOIN delivery_items di ON di.order_id=so.id WHERE di.delivery_id='{doc_id}'",
        ("delivery", "billing"): f"SELECT DISTINCT bd.id, bd.total_amount FROM billing_documents bd JOIN billing_items bi ON bi.billing_id=bd.id WHERE bi.delivery_id='{doc_id}'",
        ("journal", "billing"): f"SELECT je.reference_doc as billing_id FROM journal_entries je WHERE je.id='{doc_id}'",
        ("journal", "payment"): f"SELECT p.id, p.amount FROM payments p JOIN journal_entries je ON je.clearing_doc=p.clearing_doc WHERE je.id='{doc_id}'",
        ("payment", "journal"): f"SELECT je.id, je.reference_doc, je.amount FROM journal_entries je WHERE je.clearing_doc=(SELECT clearing_doc FROM payments WHERE id='{doc_id}')",
    }

    sql = link_sql.get((source, target))
    if sql:
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}
    return None


def _rule_based_classify(query: str) -> dict | None:
    """Try regex rules. Return {intent, params} or None to fall through to LLM."""
    q = query.strip()

    if _GREETING_RE.match(q):
        return {"intent": "UNKNOWN", "params": {}}
    if _CREATIVE_RE.search(q):
        return {"intent": "UNKNOWN", "params": {}}

    ql = q.lower()

    # --- TRACE_BILLING_FLOW (trace + billing + id) ---
    m = re.search(r"(?:trace|flow|track|full\s+flow).*?billing.*?(\d{6,})", ql)
    if not m:
        m = re.search(r"billing.*?(\d{6,}).*?(?:trace|flow|track|full\s+flow)", ql)
    if m:
        return {"intent": "TRACE_BILLING_FLOW", "params": {"billing_id": m.group(1)}}

    # --- TRACE_ORDER_FLOW (trace + order + id) ---
    m = re.search(r"(?:trace|flow|track|full\s+flow).*?order.*?(\d{5,})", ql)
    if not m:
        m = re.search(r"order.*?(\d{5,}).*?(?:trace|flow|track|full\s+flow)", ql)
    if m:
        return {"intent": "TRACE_ORDER_FLOW", "params": {"order_id": m.group(1)}}

    # --- TOP_PRODUCTS ---
    if re.search(r"top\s+product|most\s+billing|product.*?(?:billing|highest|most)|"
                 r"highest.*?(?:billing|product)", ql):
        m = re.search(r"top\s+(\d+)", ql)
        return {"intent": "TOP_PRODUCTS", "params": {"limit": int(m.group(1)) if m else 10}}

    # --- BROKEN_FLOWS ---
    if re.search(r"incomplete|broken|missing.*?flow|anomal|gap|not\s+billed|not\s+delivered", ql):
        return {"intent": "BROKEN_FLOWS", "params": {}}

    # --- ORDER_DETAILS (order + id, without trace words) ---
    m = re.search(r"(?:details?|info|show|about)\s+.*?order\s+(\d{5,})", ql)
    if not m:
        m = re.search(r"order\s+(\d{5,})", ql)
    if m:
        return {"intent": "ORDER_DETAILS", "params": {"order_id": m.group(1)}}

    # --- BILLING_DETAILS (billing + id, without trace words) ---
    m = re.search(r"billing\s*(?:document\s*)?(\d{6,})|invoice\s+(\d{6,})", ql)
    if m:
        if not re.search(r"(?:link|connect|relat|associat|tied|from|for)\b.*(?:journal|order|delivery|payment)", ql):
            return {"intent": "BILLING_DETAILS", "params": {"billing_id": m.group(1) or m.group(2)}}

    # --- Entity linking (find X linked/connected to Y) ---
    link_result = _try_entity_link(q, ql)
    if link_result:
        return link_result

    # --- Entity name resolution (customer) ---
    cust_id, cust_name = _resolve_customer_ref(q)
    if cust_id:
        return _classify_with_customer(q, ql, cust_id)

    # --- Product name resolution ---
    prod_result = _try_product_name_match(q, ql)
    if prod_result:
        return prod_result

    # --- CUSTOMER_ORDERS (customer + numeric id + order context) ---
    if not _ANALYTICAL_RE.search(ql):
        m = re.search(r"(?:customer|cust)\s+(\d+).*?order|order.*?(?:customer|cust)\s+(\d+)", ql)
        if m:
            return {"intent": "CUSTOMER_ORDERS", "params": {"customer_id": m.group(1) or m.group(2)}}

    # --- PRODUCT_INFO (simple lookup only) ---
    if not _ANALYTICAL_RE.search(ql) and not re.search(r"\b(?:by|for)\b", ql):
        m = re.search(r"product\s+(\w+)|material\s+(\w+)", ql)
        if m:
            return {"intent": "PRODUCT_INFO", "params": {"product_id": m.group(1) or m.group(2)}}

    # --- CUSTOMER_INFO (simple lookup, numeric id) ---
    if not _ANALYTICAL_RE.search(ql) and not re.search(r"\b(?:by|for)\b", ql):
        m = re.search(r"(?:customer|cust)\s+(\d+)", ql)
        if m:
            return {"intent": "CUSTOMER_INFO", "params": {"customer_id": m.group(1)}}

    # --- SUMMARY_STATS (only truly generic overview questions) ---
    if re.search(r"\b(?:stats|statistic|summary|overview|dataset|how big)\b", ql):
        return {"intent": "SUMMARY_STATS", "params": {}}

    # --- ANALYTICAL / SQL QUERY (try rule-based SQL before LLM) ---
    sql = _try_rule_based_sql(ql)
    if sql:
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    # --- Catch-all: if query references O2C entities, route to SQL_QUERY for LLM ---
    if _ENTITY_WORDS_RE.search(ql):
        return {"intent": "SQL_QUERY", "params": {"query": q}, "source": "rules"}

    return None


def _classify_with_customer(q: str, ql: str, cust_id: str) -> dict:
    """Route customer-related queries to the right intent with the resolved ID."""
    has_analytical = _ANALYTICAL_RE.search(ql)

    if re.search(r"\b(?:product|material)\w*\s+(?:ordered|bought|purchased)", ql) or \
       re.search(r"\b(?:how many|count)\s+(?:\w+\s+)*(?:product|material)", ql):
        if has_analytical:
            sql = (f"SELECT COUNT(DISTINCT soi.material_id) as count "
                   f"FROM sales_order_items soi JOIN sales_orders so ON soi.order_id=so.id "
                   f"WHERE so.customer_id='{cust_id}'")
        else:
            sql = (f"SELECT DISTINCT p.id, p.description "
                   f"FROM products p JOIN sales_order_items soi ON soi.material_id=p.id "
                   f"JOIN sales_orders so ON soi.order_id=so.id "
                   f"WHERE so.customer_id='{cust_id}'")
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:total|sum)\b.*\b(?:amount|value|revenue|spend)\b", ql) or \
       re.search(r"\b(?:order|billing)\b.*\b(?:value|amount)\b.*\b(?:total|sum)\b", ql) or \
       re.search(r"\b(?:total|sum)\b.*\border\s+value\b", ql):
        sql = (f"SELECT ROUND(SUM(total_amount), 2) as total_amount, currency "
               f"FROM sales_orders WHERE customer_id='{cust_id}' GROUP BY currency")
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:average|avg)\b.*\b(?:amount|value|order)", ql):
        sql = (f"SELECT ROUND(AVG(total_amount), 2) as avg_amount "
               f"FROM sales_orders WHERE customer_id='{cust_id}'")
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:how many|count|number\s+of|no\s+of)\b.*\b(?:order|orders)\b", ql):
        sql = f"SELECT COUNT(*) as count FROM sales_orders WHERE customer_id='{cust_id}'"
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:how many|count|total)\b.*\b(?:billing|invoice)", ql):
        sql = f"SELECT COUNT(*) as count FROM billing_documents WHERE customer_id='{cust_id}'"
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:how many|count|total)\b.*\b(?:deliver)", ql):
        sql = (f"SELECT COUNT(DISTINCT d.id) as count FROM deliveries d "
               f"JOIN delivery_items di ON di.delivery_id=d.id "
               f"JOIN sales_orders so ON di.order_id=so.id "
               f"WHERE so.customer_id='{cust_id}'")
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:order|orders|ordered|bought|purchased|placed)\b", ql):
        return {"intent": "CUSTOMER_ORDERS", "params": {"customer_id": cust_id}}

    if re.search(r"\b(?:billing|invoice|billed)\b", ql):
        sql = (f"SELECT * FROM billing_documents WHERE customer_id='{cust_id}' "
               f"ORDER BY creation_date DESC LIMIT 50")
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:deliver)", ql):
        sql = (f"SELECT DISTINCT d.* FROM deliveries d "
               f"JOIN delivery_items di ON di.delivery_id=d.id "
               f"JOIN sales_orders so ON di.order_id=so.id "
               f"WHERE so.customer_id='{cust_id}'")
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if re.search(r"\b(?:payments?|paid|pay)\b", ql):
        sql = (f"SELECT DISTINCT p.* FROM payments p "
               f"JOIN journal_entries je ON je.clearing_doc=p.clearing_doc "
               f"WHERE je.customer_id='{cust_id}'")
        return {"intent": "SQL_QUERY", "params": {"query": q, "generated_sql": sql}, "source": "rules"}

    if has_analytical:
        return {"intent": "CUSTOMER_ORDERS", "params": {"customer_id": cust_id}}

    return {"intent": "CUSTOMER_INFO", "params": {"customer_id": cust_id}}


# ---------------------------------------------------------------------------
# Rule-based SQL generator (handles common analytical patterns without LLM)
# ---------------------------------------------------------------------------

_TABLE_MAP = {
    "customer": "customers", "cust": "customers", "customers": "customers",
    "order": "sales_orders", "orders": "sales_orders", "sales order": "sales_orders",
    "product": "products", "products": "products", "material": "products", "materials": "products",
    "billing": "billing_documents", "billings": "billing_documents",
    "invoice": "billing_documents", "invoices": "billing_documents",
    "delivery": "deliveries", "deliveries": "deliveries", "deliver": "deliveries",
    "payment": "payments", "payments": "payments",
    "journal": "journal_entries", "journals": "journal_entries",
    "plant": "plants", "plants": "plants",
    "item": "sales_order_items", "items": "sales_order_items",
    "entr": "journal_entries", "entrie": "journal_entries",
}

_AMOUNT_FIELDS = {"sales_orders": "total_amount", "billing_documents": "total_amount",
                  "journal_entries": "amount", "payments": "amount"}

_NAME_FIELDS = {"customers": "name", "products": "description", "plants": "name"}


def _resolve_entity(word: str) -> str | None:
    return _TABLE_MAP.get(word.rstrip("s")) or _TABLE_MAP.get(word)


def _try_rule_based_sql(ql: str) -> str | None:
    """Try to generate SQL from common analytical patterns. Return SQL string or None."""

    # "which/what customer has the most orders"
    m = re.search(
        r"(?:which|what)\s+(customer|cust|product|material|plant)\w*"
        r"\s+(?:has|have|had)\s+(?:the\s+)?(?:most|highest|largest|maximum)\s+"
        r"(order|billing|deliver|payment|invoice|product)\w*", ql
    )
    if m:
        return _sql_ranking(m.group(1), m.group(2), "DESC")

    # "which customer has the fewest/least orders"
    m = re.search(
        r"(?:which|what)\s+(customer|cust|product|material|plant)\w*"
        r"\s+(?:has|have|had)\s+(?:the\s+)?(?:fewest|least|lowest|minimum)\s+"
        r"(order|billing|deliver|payment|invoice|product)\w*", ql
    )
    if m:
        return _sql_ranking(m.group(1), m.group(2), "ASC")

    # "average/avg order value/amount"
    m = re.search(r"(?:average|avg|mean)\s+(?:of\s+)?(?:the\s+)?(order|billing|invoice|payment|delivery)\w*"
                  r"\s*(?:value|amount|price|total)?", ql)
    if m:
        table = _resolve_entity(m.group(1))
        if table and table in _AMOUNT_FIELDS:
            return f"SELECT ROUND(AVG({_AMOUNT_FIELDS[table]}), 2) as average_amount FROM {table}"

    # "average value/amount of orders"
    m = re.search(r"(?:average|avg|mean)\s+(?:value|amount|price|total)\s+(?:of|for|per)\s+"
                  r"(order|billing|invoice|payment)\w*", ql)
    if m:
        table = _resolve_entity(m.group(1))
        if table and table in _AMOUNT_FIELDS:
            return f"SELECT ROUND(AVG({_AMOUNT_FIELDS[table]}), 2) as average_amount FROM {table}"

    # "how many [entity]" (specific entity count, handles multi-word like "sales orders")
    m = re.search(r"how many\s+(?:sales\s+)?(customer|cust|order|product|material|billing|"
                  r"deliver|payment|invoice|journal|plant|"
                  r"sales\s+order|billing\s+document|journal\s+entr|delivery\s+item|"
                  r"order\s+item|billing\s+item)\w*"
                  r"\s*(?:are there|exist|do we have|in total|in the (?:dataset|system|database))?\.?$", ql)
    if m:
        table = _resolve_entity(m.group(1).replace("sales ", "").replace("billing ", "").replace("journal ", "").replace("delivery ", "").replace("order ", "").strip())
        if table:
            return f"SELECT COUNT(*) as total FROM {table}"

    # "how many [things] for/by/of customer [id]"
    m = re.search(r"how many\s+(\w+)\s+.*?(?:for|by|of|from)\s+(?:customer|cust)\s*(\w+)", ql)
    if m:
        thing, cust_ref = m.group(1), m.group(2)
        return _sql_count_for_customer(thing, cust_ref)

    # "how many [things] for/by product [id]"
    m = re.search(r"how many\s+(\w+)\s+.*?(?:for|by|of|from)\s+(?:product|material)\s*(\w+)", ql)
    if m:
        thing, prod_ref = m.group(1), m.group(2)
        return _sql_count_for_product(thing, prod_ref)

    # "total [amount/value] of/for [entity]"
    m = re.search(r"(?:total|sum)\s+(?:of\s+)?(?:the\s+)?(?:amount|value|revenue|sales)\s+"
                  r"(?:of|for|from|in)\s+(order|billing|invoice|payment)\w*", ql)
    if m:
        table = _resolve_entity(m.group(1))
        if table and table in _AMOUNT_FIELDS:
            return f"SELECT ROUND(SUM({_AMOUNT_FIELDS[table]}), 2) as total_amount FROM {table}"

    # "total [entity] [amount/value]"
    m = re.search(r"(?:total|sum)\s+(order|billing|invoice|payment)\w*\s+(?:amount|value|revenue)", ql)
    if m:
        table = _resolve_entity(m.group(1))
        if table and table in _AMOUNT_FIELDS:
            return f"SELECT ROUND(SUM({_AMOUNT_FIELDS[table]}), 2) as total_amount FROM {table}"

    # "list/show all [entity] where/with amount > N"
    m = re.search(r"(?:list|show|find|get)\s+(?:all\s+)?(order|billing|delivery|payment|customer|product)\w*\s+"
                  r"(?:where|with|having)\s+(?:amount|value|total)\s*(>|<|>=|<=)\s*(\d+(?:\.\d+)?)", ql)
    if m:
        table = _resolve_entity(m.group(1))
        op, val = m.group(2), m.group(3)
        if table and table in _AMOUNT_FIELDS:
            return f"SELECT * FROM {table} WHERE {_AMOUNT_FIELDS[table]} {op} {val} LIMIT 100"

    # "orders/billings with amount greater/more/less than N"
    m = re.search(r"(order|billing|payment|invoice)\w*\s+(?:with\s+)?(?:amount|value|total)\s+"
                  r"(?:greater|more|higher|above|over)\s+than\s+(\d+(?:\.\d+)?)", ql)
    if m:
        table = _resolve_entity(m.group(1))
        val = m.group(2)
        if table and table in _AMOUNT_FIELDS:
            return f"SELECT * FROM {table} WHERE {_AMOUNT_FIELDS[table]} > {val} ORDER BY {_AMOUNT_FIELDS[table]} DESC LIMIT 100"

    m = re.search(r"(order|billing|payment|invoice)\w*\s+(?:with\s+)?(?:amount|value|total)\s+"
                  r"(?:less|lower|below|under)\s+than\s+(\d+(?:\.\d+)?)", ql)
    if m:
        table = _resolve_entity(m.group(1))
        val = m.group(2)
        if table and table in _AMOUNT_FIELDS:
            return f"SELECT * FROM {table} WHERE {_AMOUNT_FIELDS[table]} < {val} ORDER BY {_AMOUNT_FIELDS[table]} ASC LIMIT 100"

    return None


def _sql_ranking(entity_word: str, counted_word: str, direction: str) -> str | None:
    """Generate ranking SQL like 'which customer has the most orders'."""
    entity_table = _resolve_entity(entity_word)
    counted_table = _resolve_entity(counted_word)
    if not entity_table or not counted_table:
        return None

    name_col = _NAME_FIELDS.get(entity_table, "id")

    joins = {
        ("customers", "sales_orders"): (
            f"SELECT c.id, c.{name_col}, COUNT(so.id) as cnt "
            f"FROM customers c JOIN sales_orders so ON so.customer_id=c.id "
            f"GROUP BY c.id ORDER BY cnt {direction} LIMIT 10"
        ),
        ("customers", "billing_documents"): (
            f"SELECT c.id, c.{name_col}, COUNT(bd.id) as cnt "
            f"FROM customers c JOIN billing_documents bd ON bd.customer_id=c.id "
            f"GROUP BY c.id ORDER BY cnt {direction} LIMIT 10"
        ),
        ("customers", "payments"): (
            f"SELECT c.id, c.{name_col}, COUNT(p.id) as cnt "
            f"FROM customers c JOIN journal_entries je ON je.customer_id=c.id "
            f"JOIN payments p ON p.clearing_doc=je.clearing_doc "
            f"GROUP BY c.id ORDER BY cnt {direction} LIMIT 10"
        ),
        ("customers", "deliveries"): (
            f"SELECT c.id, c.{name_col}, COUNT(DISTINCT d.id) as cnt "
            f"FROM customers c JOIN sales_orders so ON so.customer_id=c.id "
            f"JOIN delivery_items di ON di.order_id=so.id "
            f"JOIN deliveries d ON d.id=di.delivery_id "
            f"GROUP BY c.id ORDER BY cnt {direction} LIMIT 10"
        ),
        ("customers", "products"): (
            f"SELECT c.id, c.{name_col}, COUNT(DISTINCT soi.material_id) as cnt "
            f"FROM customers c JOIN sales_orders so ON so.customer_id=c.id "
            f"JOIN sales_order_items soi ON soi.order_id=so.id "
            f"GROUP BY c.id ORDER BY cnt {direction} LIMIT 10"
        ),
        ("products", "sales_orders"): (
            f"SELECT p.id, p.description, COUNT(DISTINCT soi.order_id) as cnt "
            f"FROM products p JOIN sales_order_items soi ON soi.material_id=p.id "
            f"GROUP BY p.id ORDER BY cnt {direction} LIMIT 10"
        ),
        ("products", "billing_documents"): (
            f"SELECT p.id, p.description, COUNT(DISTINCT bi.billing_id) as cnt "
            f"FROM products p JOIN billing_items bi ON bi.material_id=p.id "
            f"GROUP BY p.id ORDER BY cnt {direction} LIMIT 10"
        ),
    }
    return joins.get((entity_table, counted_table))


def _sql_count_for_customer(thing: str, cust_ref: str) -> str:
    """Generate count SQL for 'how many X for customer Y'."""
    cust_id = _resolve_customer_id(cust_ref)
    cond = f"LIKE '{cust_id}%'" if len(cust_ref) < 6 else f"= '{cust_id}'"

    thing_l = thing.lower().rstrip("s")
    if thing_l in ("order", "sales order"):
        return f"SELECT COUNT(*) as count FROM sales_orders WHERE customer_id {cond}"
    if thing_l in ("product", "material"):
        return (f"SELECT COUNT(DISTINCT soi.material_id) as count "
                f"FROM sales_order_items soi JOIN sales_orders so ON soi.order_id=so.id "
                f"WHERE so.customer_id {cond}")
    if thing_l in ("billing", "invoice"):
        return f"SELECT COUNT(*) as count FROM billing_documents WHERE customer_id {cond}"
    if thing_l in ("delivery", "deliverie"):
        return (f"SELECT COUNT(DISTINCT d.id) as count FROM deliveries d "
                f"JOIN delivery_items di ON di.delivery_id=d.id "
                f"JOIN sales_orders so ON di.order_id=so.id "
                f"WHERE so.customer_id {cond}")
    if thing_l in ("payment",):
        return (f"SELECT COUNT(DISTINCT p.id) as count FROM payments p "
                f"JOIN journal_entries je ON je.clearing_doc=p.clearing_doc "
                f"WHERE je.customer_id {cond}")
    return (f"SELECT COUNT(*) as count FROM sales_orders WHERE customer_id {cond}")


def _sql_count_for_product(thing: str, prod_ref: str) -> str:
    """Generate count SQL for 'how many X for product Y'."""
    thing_l = thing.lower().rstrip("s")
    if thing_l in ("order", "sales order"):
        return (f"SELECT COUNT(DISTINCT soi.order_id) as count "
                f"FROM sales_order_items soi WHERE soi.material_id LIKE '%{prod_ref}%'")
    if thing_l in ("billing", "invoice"):
        return (f"SELECT COUNT(DISTINCT bi.billing_id) as count "
                f"FROM billing_items bi WHERE bi.material_id LIKE '%{prod_ref}%'")
    return (f"SELECT COUNT(DISTINCT soi.order_id) as count "
            f"FROM sales_order_items soi WHERE soi.material_id LIKE '%{prod_ref}%'")


def _resolve_customer_id(ref: str) -> str:
    """Resolve abbreviated customer references: '3' -> '3', '310000108' -> '310000108'."""
    ref = ref.strip()
    if ref.isdigit() and len(ref) < 6:
        return f"3{ref.zfill(8)}" if len(ref) == 1 else ref
    return ref


# ---------------------------------------------------------------------------
# Intent classification (rules first, LLM fallback)
# ---------------------------------------------------------------------------

INTENT_DESCRIPTIONS = {
    "TRACE_BILLING_FLOW": "Trace full O2C flow for a billing document. Params: billing_id",
    "TRACE_ORDER_FLOW": "Trace full O2C flow for a sales order. Params: order_id",
    "TOP_PRODUCTS": "Products ranked by billing document count. Params: limit (optional)",
    "BROKEN_FLOWS": "Sales orders with broken/incomplete flows. No params",
    "ORDER_DETAILS": "Details for a specific sales order. Params: order_id",
    "CUSTOMER_ORDERS": "All orders for a customer. Params: customer_id",
    "BILLING_DETAILS": "Billing document details. Params: billing_id",
    "PRODUCT_INFO": "Product details and associations. Params: product_id",
    "CUSTOMER_INFO": "Customer details and activity summary. Params: customer_id",
    "SUMMARY_STATS": "Overall dataset statistics. No params",
    "SQL_QUERY": "Complex natural-language question answered via SQL. Params: query",
}

REQUIRED_PARAMS: dict[str, list[str]] = {
    "TRACE_BILLING_FLOW": ["billing_id"],
    "TRACE_ORDER_FLOW": ["order_id"],
    "TOP_PRODUCTS": [],
    "BROKEN_FLOWS": [],
    "ORDER_DETAILS": ["order_id"],
    "CUSTOMER_ORDERS": ["customer_id"],
    "BILLING_DETAILS": ["billing_id"],
    "PRODUCT_INFO": ["product_id"],
    "CUSTOMER_INFO": ["customer_id"],
    "SUMMARY_STATS": [],
    "SQL_QUERY": ["query"],
}


def classify_intent(user_query: str, history: list[dict] | None = None) -> dict:
    """Classify intent: try rules first, fall back to Gemini if needed."""
    needs_context = bool(
        history and len(history) > 1
        and any(w in user_query.lower().split() for w in _CONTEXT_WORDS)
    )

    if not needs_context:
        result = _rule_based_classify(user_query)
        if result is not None:
            logger.info("Rule-match intent: %s for: %s", result["intent"], user_query[:80])
            if "source" not in result:
                result["source"] = "rules"
            return result

    llm_result = _llm_classify(user_query, history)

    if llm_result.get("source") == "llm_error":
        sql = _try_rule_based_sql(user_query.lower())
        if sql:
            logger.info("LLM failed, falling back to rule-based SQL for: %s", user_query[:80])
            return {"intent": "SQL_QUERY", "params": {"query": user_query, "generated_sql": sql}, "source": "rules_fallback"}

    return llm_result


def _llm_classify(user_query: str, history: list[dict] | None = None) -> dict:
    """LLM-based intent classification (fallback) with timeout."""
    try:
        intent_block = "\n".join(f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())
        prompt = (
            "You are a query router for an SAP Order-to-Cash dataset system.\n"
            "Classify the user's query into one intent and extract parameters.\n\n"
            f"Available intents:\n{intent_block}\n\n"
            "Rules:\n"
            '1. Return ONLY valid JSON: {"intent":"...","params":{...}}\n'
            '2. For unrelated queries, return {"intent":"UNKNOWN","params":{}}\n'
            "3. Extract numeric IDs. Example: 'billing document 90504248' -> billing_id:'90504248'\n"
            "4. Use SQL_QUERY for any analytical question (averages, comparisons, "
            "aggregations, filtered counts, rankings, joins across entities)\n"
            "5. SUMMARY_STATS is ONLY for 'show me the overall stats/overview'. "
            "Any question that needs computation (average, sum, etc.) must use SQL_QUERY\n"
            "6. Return ONLY the JSON object, no markdown\n"
        )
        if history:
            prompt += "\nRecent conversation:\n"
            for msg in history[-4:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                prompt += f"{role}: {msg.get('content', '')}\n"
        prompt += f"\nCurrent query: {user_query}\nJSON:"

        raw = _strip_code_fence(_call_llm(prompt, max_tokens=2048))
        result = json.loads(raw)
        result.setdefault("intent", "UNKNOWN")
        result.setdefault("params", {})
        result["source"] = "llm"
        logger.info("LLM intent: %s for: %s", result["intent"], user_query[:80])
        return result
    except FuturesTimeoutError:
        logger.error("LLM classification timed out for: %s", user_query[:80])
        return {"intent": "UNKNOWN", "params": {}, "source": "llm_error"}
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return {"intent": "UNKNOWN", "params": {}, "source": "llm_error"}


# ---------------------------------------------------------------------------
# Validation / Guardrails
# ---------------------------------------------------------------------------

_CLARIFICATIONS = {
    "billing_id": "Which billing document? Please provide the billing document number.",
    "order_id": "Which sales order? Please provide the sales order number.",
    "customer_id": "Which customer? Please provide the customer ID.",
    "product_id": "Which product? Please provide the product ID.",
    "query": "Please provide your business question.",
}


def validate_intent(intent: str, params: dict) -> tuple[bool, str | None]:
    if intent == "UNKNOWN":
        return False, (
            "This system is designed to answer questions related to the SAP "
            "Order-to-Cash dataset only. Try asking about sales orders, "
            "deliveries, billing documents, products, or customers."
        )
    if intent not in REQUIRED_PARAMS:
        return False, (
            "I didn't understand that query. You can ask about:\n"
            "- Tracing billing or order flows\n"
            "- Top products by billing count\n"
            "- Incomplete/broken order flows\n"
            "- Details on specific orders, customers, or products"
        )
    for p in REQUIRED_PARAMS[intent]:
        if not params.get(p):
            return False, _CLARIFICATIONS.get(p, f"Missing parameter: {p}")
    return True, None


# ---------------------------------------------------------------------------
# NL-to-SQL (only LLM call that remains for data retrieval)
# ---------------------------------------------------------------------------

_SQL_SCHEMA = """
Tables:
- customers (id, name, full_name, category, industry, country, city, region, street, postal_code)
- sales_orders (id, customer_id, order_type, sales_org, creation_date, total_amount, currency, delivery_status, billing_status, payment_terms)
- sales_order_items (id, order_id, item_number, material_id, quantity, amount, currency, plant, item_category)
- deliveries (id, creation_date, shipping_point, goods_movement_status, picking_status)
- delivery_items (id, delivery_id, item_number, order_id, order_item, plant, quantity)
- billing_documents (id, doc_type, creation_date, total_amount, currency, customer_id, is_cancelled, accounting_doc)
- billing_items (id, billing_id, item_number, material_id, quantity, amount, currency, delivery_id, delivery_item)
- journal_entries (id, item_number, reference_doc, clearing_doc, amount, currency, customer_id, posting_date, doc_type)
- payments (id, item_number, clearing_doc, amount, currency, customer_id, posting_date, clearing_date)
- plants (id, name, sales_org, language)
- products (id, product_type, description, gross_weight, net_weight, weight_unit, product_group, base_unit)

Key relationships:
- sales_orders.customer_id -> customers.id
- sales_order_items.order_id -> sales_orders.id
- sales_order_items.material_id -> products.id
- delivery_items.order_id -> sales_orders.id
- delivery_items.delivery_id -> deliveries.id
- billing_items.billing_id -> billing_documents.id
- billing_items.delivery_id -> deliveries.id
- billing_items.material_id -> products.id
- billing_documents.customer_id -> customers.id
- journal_entries.reference_doc -> billing_documents.id
- journal_entries.clearing_doc <-> payments.clearing_doc
"""


def generate_sql(question: str, history: list[dict] | None = None) -> str:
    prompt = (
        "You are a SQL expert for an SAP Order-to-Cash SQLite database.\n"
        f"{_SQL_SCHEMA}\n"
    )
    if history:
        prompt += "\nRecent conversation for context:\n"
        for msg in history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            prompt += f"{role}: {msg.get('content', '')[:200]}\n"
    prompt += (
        f'\nWrite a SQLite SELECT query to answer: "{question}"\n'
        "Return ONLY the SQL query. Use LIMIT 100. No explanation."
    )
    return _strip_code_fence(_call_llm(prompt, max_tokens=4096))


def summarize_sql_result(user_query: str, rows: list, count: int) -> str | None:
    """Use LLM to generate a brief NL answer from SQL results (for complex cases)."""
    if not rows or not GEMINI_API_KEY:
        return None
    try:
        sample = rows[:8]
        prompt = (
            f'User asked: "{user_query}"\n'
            f"The database query returned {count} result(s). Data:\n"
            f"{json.dumps(sample, default=str)}\n\n"
            "Write a concise (1-3 sentence) natural language answer that directly "
            "answers the user's question. Include specific numbers, names, and IDs "
            "from the data. Be factual, do not speculate."
        )
        return _call_llm(prompt, max_tokens=512)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Template-based response formatting (Change 2 — no LLM)
# ---------------------------------------------------------------------------

def format_structured_response(intent: str, result: dict, user_query: str = "") -> dict:
    """Build a structured response dict using deterministic templates."""
    if "error" in result:
        return {
            "summary": f"Error: {result['error']}",
            "details": {},
            "insights": [],
            "nodes_traversed": result.get("nodes_traversed", []),
            "generated_sql": result.get("generated_sql"),
        }
    if intent == "SQL_QUERY":
        result["user_query"] = user_query
    formatter = _TEMPLATES.get(intent, _tmpl_generic)
    try:
        return formatter(result)
    except Exception as e:
        logger.error("Template formatting failed for %s: %s", intent, e)
        return _tmpl_generic(result)


def _tmpl_trace_billing(r: dict) -> dict:
    b = r.get("billing", {})
    bid = b.get("id", "?")
    stages = {
        "Sales Orders": r.get("orders", []),
        "Deliveries": r.get("deliveries", []),
        "Billing": [b] if b else [],
        "Journal Entries": r.get("journals", []),
        "Payments": r.get("payments", []),
    }
    complete = sum(1 for v in stages.values() if v)
    flow = []
    if stages["Sales Orders"]:
        flow.append(f"Order(s) {', '.join(o['id'] for o in stages['Sales Orders'])}")
    if stages["Deliveries"]:
        flow.append(f"Delivery(s) {', '.join(d['id'] for d in stages['Deliveries'])}")
    flow.append(f"Billing {bid}")
    if stages["Journal Entries"]:
        flow.append(f"Journal(s) {', '.join(j['id'] for j in stages['Journal Entries'])}")
    if stages["Payments"]:
        flow.append(f"Payment(s) {', '.join(p['id'] for p in stages['Payments'])}")

    cust = r.get("customers", [])
    cust_str = f" Customer: {cust[0].get('name', cust[0].get('id'))}." if cust else ""
    summary = (
        f"**Billing {bid}** traces: {' → '.join(flow)}. "
        f"{complete}/5 stages complete.{cust_str}"
    )
    if b.get("total_amount"):
        summary += f" Amount: {b['total_amount']} {b.get('currency', '')}."

    insights = []
    if not stages["Payments"]:
        insights.append("No payment found — potential open receivable.")
    if not stages["Journal Entries"]:
        insights.append("No journal entry — accounting document may be pending.")
    if not stages["Sales Orders"]:
        insights.append("No linked sales order found.")
    if b.get("is_cancelled"):
        insights.append("This billing document has been cancelled.")

    return {
        "summary": summary,
        "details": {k: [dict(row) if hasattr(row, "keys") else row for row in v] for k, v in stages.items()},
        "insights": insights,
        "nodes_traversed": r.get("nodes_traversed", []),
        "generated_sql": None,
    }


def _tmpl_trace_order(r: dict) -> dict:
    order = r.get("order", {})
    oid = order.get("id", "?")
    stages = {
        "Customer": [r["customer"]] if r.get("customer") else [],
        "Sales Order": [order] if order else [],
        "Deliveries": r.get("deliveries", []),
        "Billings": r.get("billings", []),
        "Journal Entries": r.get("journals", []),
        "Payments": r.get("payments", []),
    }
    complete = sum(1 for v in stages.values() if v)
    flow = [f"Order {oid}"]
    if stages["Deliveries"]:
        flow.append(f"Delivery(s) {', '.join(d['id'] for d in stages['Deliveries'])}")
    if stages["Billings"]:
        flow.append(f"Billing(s) {', '.join(b['id'] for b in stages['Billings'])}")
    if stages["Journal Entries"]:
        flow.append(f"Journal(s) {', '.join(j['id'] for j in stages['Journal Entries'])}")
    if stages["Payments"]:
        flow.append(f"Payment(s) {', '.join(p['id'] for p in stages['Payments'])}")

    cust = r.get("customer")
    cust_str = f" Customer: {cust.get('name', cust.get('id'))}." if cust else ""
    summary = (
        f"**Order {oid}** flow: {' → '.join(flow)}. "
        f"{complete}/6 stages complete.{cust_str}"
    )
    if order.get("total_amount"):
        summary += f" Amount: {order['total_amount']} {order.get('currency', '')}."

    insights = []
    if not stages["Deliveries"]:
        insights.append("No delivery found — order may not have shipped yet.")
    if not stages["Billings"]:
        insights.append("No billing document — potential revenue leakage.")
    if stages["Deliveries"] and not stages["Billings"]:
        insights.append("Delivered but not billed — revenue leakage risk.")
    if not stages["Payments"]:
        insights.append("No payment recorded — open receivable.")

    return {
        "summary": summary,
        "details": {k: [dict(row) if hasattr(row, "keys") else row for row in v] for k, v in stages.items()},
        "insights": insights,
        "nodes_traversed": r.get("nodes_traversed", []),
        "generated_sql": None,
    }


def _tmpl_top_products(r: dict) -> dict:
    products = r.get("products", [])
    lines = []
    for i, p in enumerate(products, 1):
        desc = p.get("description") or p.get("id", "?")
        count = p.get("billing_count", 0)
        lines.append(f"{i}. **{desc}** (ID: {p.get('id', '?')}) — {count} billing documents")
    summary = f"**Top {len(products)} products** by billing document frequency:\n" + "\n".join(lines)
    return {
        "summary": summary,
        "details": {"products": products},
        "insights": [],
        "nodes_traversed": [],
        "generated_sql": None,
    }


def _tmpl_broken_flows(r: dict) -> dict:
    orders = r.get("incomplete_orders", [])
    total = r.get("total", len(orders))

    no_delivery = sum(1 for o in orders if "no delivery" in o.get("issue", ""))
    no_billing = sum(1 for o in orders if "no billing" in o.get("issue", ""))
    delivered_not_billed = sum(
        1 for o in orders
        if "no billing" in o.get("issue", "") and "no delivery" not in o.get("issue", "")
    )
    neither = sum(
        1 for o in orders
        if "no delivery" in o.get("issue", "") and "no billing" in o.get("issue", "")
    )

    summary = f"**{total} sales orders** with incomplete flows found."
    if delivered_not_billed:
        summary += f"\n- {delivered_not_billed} orders delivered but not billed"
    if neither:
        summary += f"\n- {neither} orders have no delivery and no billing"
    if no_delivery and not neither:
        summary += f"\n- {no_delivery} orders have no delivery"

    sample = orders[:8]
    if sample:
        summary += "\n\nSample orders:\n" + "\n".join(
            f"- **{o.get('id')}**: {o.get('issue', 'unknown')} "
            f"(amount: {o.get('total_amount', 0)} {o.get('currency', '')})"
            for o in sample
        )
        if total > 8:
            summary += f"\n... and {total - 8} more"

    insights = []
    if delivered_not_billed:
        insights.append(f"{delivered_not_billed} orders delivered but not billed — potential revenue leakage.")
    if neither:
        insights.append(f"{neither} orders have neither delivery nor billing — may be pending or cancelled.")

    return {
        "summary": summary,
        "details": {"orders": orders[:50]},
        "insights": insights,
        "nodes_traversed": [],
        "generated_sql": None,
    }


def _tmpl_summary_stats(r: dict) -> dict:
    by_type = r.get("by_type", {})
    total_n = r.get("total_nodes", 0)
    total_e = r.get("total_edges", 0)

    type_lines = "\n".join(f"- **{k.replace('_', ' ').title()}**: {v:,}" for k, v in by_type.items())
    summary = (
        f"**{total_n:,} entities** and **{total_e:,} relationships** "
        f"across {len(by_type)} entity types.\n\n{type_lines}"
    )
    return {
        "summary": summary,
        "details": r,
        "insights": [],
        "nodes_traversed": [],
        "generated_sql": None,
    }


def _tmpl_customer_orders(r: dict) -> dict:
    cust = r.get("customer", {})
    orders = r.get("orders", [])
    total = r.get("total_orders", len(orders))
    name = cust.get("name") or cust.get("id", "?")
    summary = f"**Customer {name}** (ID: {cust.get('id', '?')}) has **{total} sales orders**."
    if orders:
        total_value = sum(float(o.get("total_amount", 0) or 0) for o in orders)
        currencies = {o.get("currency", "") for o in orders if o.get("currency")}
        cur = next(iter(currencies), "")
        summary += f"\nTotal order value: **{total_value:,.2f} {cur}**."
        summary += "\n\nRecent orders:\n" + "\n".join(
            f"- **SO-{o['id']}**: {o.get('total_amount', 0)} {o.get('currency', '')} "
            f"({o.get('creation_date', 'n/a')})"
            for o in orders[:8]
        )
    return {
        "summary": summary,
        "details": {"customer": cust, "orders": orders},
        "insights": [],
        "nodes_traversed": r.get("nodes_traversed", []),
        "generated_sql": None,
    }


def _tmpl_customer_info(r: dict) -> dict:
    cust = r.get("customer", {})
    name = cust.get("name") or cust.get("id", "?")
    summary = (
        f"**Customer {name}** (ID: {cust.get('id', '?')})\n"
        f"- Location: {cust.get('city', 'N/A')}, {cust.get('country', 'N/A')}\n"
        f"- Category: {cust.get('category', 'N/A')}\n"
        f"- Total orders: **{r.get('total_orders', 0)}**\n"
        f"- Total billings: **{r.get('total_billings', 0)}**"
    )
    return {
        "summary": summary,
        "details": r,
        "insights": [],
        "nodes_traversed": r.get("nodes_traversed", []),
        "generated_sql": None,
    }


def _tmpl_product_info(r: dict) -> dict:
    p = r.get("product", {})
    desc = p.get("description") or p.get("id", "?")
    summary = (
        f"**Product {desc}** (ID: {p.get('id', '?')})\n"
        f"- Type: {p.get('product_type', 'N/A')}\n"
        f"- Product group: {p.get('product_group', 'N/A')}\n"
        f"- Weight: {p.get('net_weight', 0)} {p.get('weight_unit', '')}\n"
        f"- Used in **{r.get('order_item_count', 0)}** order items\n"
        f"- Appears in **{r.get('billing_item_count', 0)}** billing items"
    )
    return {
        "summary": summary,
        "details": r,
        "insights": [],
        "nodes_traversed": r.get("nodes_traversed", []),
        "generated_sql": None,
    }


def _tmpl_sql_query(r: dict) -> dict:
    rows = r.get("rows", [])
    count = r.get("count", len(rows))
    sql = r.get("generated_sql", "")
    user_query = r.get("user_query", "")

    if not rows:
        return {
            "summary": "No matching data found for your query.",
            "details": {"rows": [], "count": 0},
            "insights": [],
            "nodes_traversed": [],
            "generated_sql": sql,
        }

    cols = list(rows[0].keys())

    if count == 1 and len(cols) == 1:
        summary = _nl_scalar(cols[0], rows[0][cols[0]], user_query, sql)
    elif count == 1:
        summary = _nl_single_row(rows[0], user_query, sql)
    elif count <= 20:
        summary = _nl_small_result(rows, cols, count, user_query, sql)
    else:
        summary = _nl_large_result(rows, cols, count, user_query)

    return {
        "summary": summary,
        "details": {"rows": rows[:50], "count": count},
        "insights": [],
        "nodes_traversed": [],
        "generated_sql": sql,
    }


def _entity_label(query: str, sql: str) -> str:
    q = query.lower()
    if "product" in q or "material" in q: return "products"
    if "billing" in q or "invoice" in q: return "billing documents"
    if "deliver" in q: return "deliveries"
    if "payment" in q: return "payments"
    if "journal" in q: return "journal entries"
    if "customer" in q: return "customers"
    if "plant" in q: return "plants"
    if "order" in q: return "orders"
    ctx = sql.lower()
    if "products" in ctx: return "products"
    if "billing" in ctx: return "billing documents"
    if "deliveries" in ctx: return "deliveries"
    if "payments" in ctx: return "payments"
    if "journal" in ctx: return "journal entries"
    if "sales_orders" in ctx: return "orders"
    return "records"


def _fmt_val(v) -> str:
    if isinstance(v, float): return f"{v:,.2f}"
    if isinstance(v, int): return f"{v:,}"
    return str(v) if v is not None else "N/A"


def _nl_scalar(key: str, val, query: str, sql: str) -> str:
    k = key.lower()
    v = _fmt_val(val)
    entity = _entity_label(query, sql)
    if "count" in k or k == "total":
        return f"There are **{v}** {entity} matching your query."
    if "avg" in k or "average" in k:
        return f"The average {entity.rstrip('s')} amount is **{v}**."
    if "total_amount" in k or "sum" in k:
        return f"The total amount is **{v}**."
    if "min" in k: return f"The minimum is **{v}**."
    if "max" in k: return f"The maximum is **{v}**."
    return f"**{key.replace('_', ' ').title()}**: {v}"


def _singular(label: str) -> str:
    if label.endswith("ies"): return label[:-3] + "y"
    if label.endswith("ments"): return label[:-1]
    if label.endswith("s"): return label[:-1]
    return label


def _nl_single_row(row: dict, query: str, sql: str) -> str:
    q = query.lower()
    vals = list(row.values())
    keys = list(row.keys())

    if any(w in q for w in ["linked", "connected", "related", "associated", "tied"]):
        entity = _singular(_entity_label(q, sql))
        primary = vals[0]
        if len(vals) >= 2:
            extra = ", ".join(f"{_fmt_val(v)}" for v in vals[1:] if v is not None)
            return f"The {entity} linked to this document is **{primary}** ({extra})."
        return f"The {entity} linked to this document is **{primary}**."

    amt_key = next((k for k in keys if "amount" in k.lower() or "total" in k.lower() or "sum" in k.lower()), None)
    cur_key = next((k for k in keys if "currency" in k.lower()), None)
    if amt_key and cur_key and len(keys) <= 3:
        amt = _fmt_val(row[amt_key])
        cur = row.get(cur_key, "")
        return f"The total amount is **{amt} {cur}**."

    if len(keys) <= 4:
        pairs = [f"**{k.replace('_', ' ').title()}**: {_fmt_val(v)}" for k, v in row.items() if v is not None]
        return " | ".join(pairs)

    lines = [f"- **{k.replace('_', ' ').title()}**: {_fmt_val(v)}" for k, v in row.items() if v is not None]
    return "\n".join(lines)


def _nl_small_result(rows: list, cols: list, count: int, query: str, sql: str) -> str:
    cnt_col = next((c for c in cols if c in ("cnt", "count", "total", "num")), None)
    name_col = next((c for c in cols if c in ("name", "description", "full_name")), None)

    if cnt_col and (name_col or "id" in cols):
        label_col = name_col or "id"
        lines = []
        for i, row in enumerate(rows, 1):
            label = row.get(label_col, row.get("id", "?"))
            cv = row.get(cnt_col, 0)
            lines.append(f"{i}. **{label}** — {_fmt_val(cv)}")
        return "\n".join(lines)

    return _build_md_table(rows, cols, count)


def _nl_large_result(rows: list, cols: list, count: int, query: str) -> str:
    header = f"Found **{count:,}** results."
    return header + "\n\n" + _build_md_table(rows[:20], cols, count)


def _build_md_table(rows: list, cols: list, total: int) -> str:
    header = " | ".join(f"**{c}**" for c in cols)
    divider = " | ".join("---" for _ in cols)
    body = "\n".join(" | ".join(str(row.get(c, "")) for c in cols) for row in rows[:20])
    table = f"{header}\n{divider}\n{body}"
    if total > 20:
        table += f"\n\n... showing first 20 of {total} rows."
    return table


def _tmpl_generic(r: dict) -> dict:
    lines = []
    for key, value in r.items():
        if key in ("nodes_traversed", "generated_sql"):
            continue
        if isinstance(value, list):
            lines.append(f"**{key.replace('_', ' ').title()}** ({len(value)} items)")
            for item in value[:5]:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('id', item.get('label', str(item)[:60]))}")
                else:
                    lines.append(f"  - {item}")
            if len(value) > 5:
                lines.append(f"  ... and {len(value) - 5} more")
        elif isinstance(value, dict):
            lines.append(f"**{key.replace('_', ' ').title()}**: {value.get('id', value.get('name', ''))}")
        else:
            lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
    return {
        "summary": "\n".join(lines) if lines else "Query completed.",
        "details": r,
        "insights": [],
        "nodes_traversed": r.get("nodes_traversed", []),
        "generated_sql": r.get("generated_sql"),
    }


_TEMPLATES = {
    "TRACE_BILLING_FLOW": _tmpl_trace_billing,
    "BILLING_DETAILS": _tmpl_trace_billing,
    "TRACE_ORDER_FLOW": _tmpl_trace_order,
    "ORDER_DETAILS": _tmpl_trace_order,
    "TOP_PRODUCTS": _tmpl_top_products,
    "BROKEN_FLOWS": _tmpl_broken_flows,
    "SUMMARY_STATS": _tmpl_summary_stats,
    "CUSTOMER_ORDERS": _tmpl_customer_orders,
    "CUSTOMER_INFO": _tmpl_customer_info,
    "PRODUCT_INFO": _tmpl_product_info,
    "SQL_QUERY": _tmpl_sql_query,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text
