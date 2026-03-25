"""FastAPI application: startup, API routes, and SSE chat streaming.

Single entry point that consolidates all routes previously split across
routers/graph_router.py, routers/chat_router.py, and queries/executor.py.
"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import database as db
import llm
from config import GEMINI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("o2c")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    logger.info("Initializing SQLite database from JSONL dataset...")
    try:
        metrics = db.init_db()
        app.state.db_ready = True
        logger.info("Database ready: %s", metrics)
    except Exception as exc:
        app.state.db_ready = False
        app.state.startup_error = str(exc)
        logger.exception("Database initialization failed")

    elapsed = (time.perf_counter() - start) * 1000
    logger.info("Startup complete in %.0fms", elapsed)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SAP O2C Graph Query System",
    description="Graph-based data modeling and conversational query system for SAP Order-to-Cash data.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    if getattr(app.state, "db_ready", False):
        return {"status": "ok", "database": "ready", "llm": "configured" if GEMINI_API_KEY else "not_configured"}
    return {"status": "degraded", "error": getattr(app.state, "startup_error", None)}


# ---------------------------------------------------------------------------
# Graph API
# ---------------------------------------------------------------------------

@app.get("/api/graph/root")
def graph_root():
    return db.build_graph_root()


@app.get("/api/graph/expand/{node_id:path}")
def graph_expand(node_id: str):
    return db.expand_node(node_id)


@app.get("/api/graph/node/{node_id:path}")
def graph_node_detail(node_id: str):
    return db.get_node_detail(node_id)


@app.get("/api/graph/stats")
def graph_stats():
    return db.get_stats()


@app.get("/api/graph/flow/{billing_id}")
def graph_billing_flow(billing_id: str):
    return db.trace_billing_flow(billing_id)


@app.get("/api/graph/order-flow/{order_id}")
def graph_order_flow(order_id: str):
    return db.trace_order_flow(order_id)


@app.get("/api/graph/products/top")
def graph_top_products(limit: int = 10):
    return db.top_products_by_billing(limit)


@app.get("/api/graph/orders/incomplete")
def graph_incomplete_orders():
    return db.find_incomplete_orders()


# ---------------------------------------------------------------------------
# Chat API (SSE streaming)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []


QUERY_FUNCTIONS = {
    "TRACE_BILLING_FLOW": lambda p: db.trace_billing_flow(p.get("billing_id", "")),
    "TRACE_ORDER_FLOW": lambda p: db.trace_order_flow(p.get("order_id", "")),
    "TOP_PRODUCTS": lambda p: db.top_products_by_billing(int(p.get("limit", 10))),
    "BROKEN_FLOWS": lambda _: db.find_incomplete_orders(),
    "ORDER_DETAILS": lambda p: db.trace_order_flow(p.get("order_id", "")),
    "CUSTOMER_ORDERS": lambda p: db.get_customer_orders(p.get("customer_id", "")),
    "BILLING_DETAILS": lambda p: db.trace_billing_flow(p.get("billing_id", "")),
    "PRODUCT_INFO": lambda p: db.get_product_info(p.get("product_id", "")),
    "CUSTOMER_INFO": lambda p: db.get_customer_info(p.get("customer_id", "")),
    "SUMMARY_STATS": lambda _: db.get_summary_stats(),
    "SQL_QUERY": lambda p: _run_sql_query(p),
}


def _run_sql_query(params: dict) -> dict:
    query = params.get("query", "")
    if not query:
        return {"error": "No query provided"}
    try:
        sql = params.get("generated_sql") or llm.generate_sql(query, params.get("conversation_history"))
        result = db.run_sql(sql)
        result["generated_sql"] = sql
        return result
    except Exception as e:
        return {"error": f"SQL generation failed: {e}"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _chat_stream(req.message, req.conversation_history),
        media_type="text/event-stream",
    )


async def _chat_stream(message: str, history: list[dict]):
    total_start = time.perf_counter()

    if not GEMINI_API_KEY:
        yield _sse({
            "type": "answer",
            "content": "Please set GEMINI_API_KEY in server/.env to enable the chat interface.",
            "meta": {"intent": "SYSTEM_ERROR"},
        })
        yield _sse({"type": "done"})
        return

    # 1) Intent classification
    yield _sse({"type": "status", "message": "Classifying intent..."})
    llm_start = time.perf_counter()
    intent_result = llm.classify_intent(message, history)
    llm_ms = (time.perf_counter() - llm_start) * 1000
    intent = intent_result.get("intent", "UNKNOWN")
    params = intent_result.get("params", {})

    if intent == "SQL_QUERY":
        params.setdefault("query", message)
        params["conversation_history"] = history

    yield _sse({"type": "intent", "intent": intent, "params": params, "llm_ms": round(llm_ms, 2)})

    # 2) Validation / guardrails
    is_valid, clarification = llm.validate_intent(intent, params)
    if not is_valid:
        yield _sse({
            "type": "answer",
            "content": clarification,
            "meta": {"intent": intent, "llm_ms": round(llm_ms, 2)},
        })
        yield _sse({"type": "done"})
        return

    # 3) Execute query
    yield _sse({"type": "status", "message": f"Executing {intent}..."})
    q_start = time.perf_counter()
    fn = QUERY_FUNCTIONS.get(intent)
    result = fn(params) if fn else {"error": f"Unknown intent: {intent}"}
    q_ms = (time.perf_counter() - q_start) * 1000

    yield _sse({
        "type": "result",
        "data": result,
        "meta": {"nodes_traversed": result.get("nodes_traversed", [])},
    })

    # 4) Format response via templates + optional LLM NL summary for SQL
    fmt_start = time.perf_counter()
    structured = llm.format_structured_response(intent, result, message)

    if intent == "SQL_QUERY" and result.get("rows"):
        rows = result["rows"]
        count = result.get("count", len(rows))
        cols = list(rows[0].keys()) if rows else []
        needs_llm = count > 10 and len(cols) > 4
        if needs_llm:
            nl_answer = llm.summarize_sql_result(message, rows, count)
            if nl_answer:
                structured["summary"] = nl_answer

    fmt_ms = (time.perf_counter() - fmt_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    intent_source = intent_result.get("source", "llm")

    yield _sse({
        "type": "answer",
        "content": structured["summary"],
        "structured": structured,
        "meta": {
            "intent": intent,
            "intent_source": intent_source,
            "params": {k: v for k, v in params.items() if k != "conversation_history"},
            "nodes_traversed": structured.get("nodes_traversed", []),
            "generated_sql": structured.get("generated_sql"),
            "llm_ms": round(llm_ms, 2),
            "query_ms": round(q_ms, 2),
            "format_ms": round(fmt_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    })
    yield _sse({"type": "done"})

    logger.info(
        "Chat: query=%r intent=%s[%s] llm=%.0fms query=%.0fms fmt=%.0fms total=%.0fms",
        message[:60], intent, intent_source, llm_ms, q_ms, fmt_ms, total_ms,
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"
