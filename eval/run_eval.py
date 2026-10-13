"""Quality eval: the README facts through the real MCP server over HTTP.

Run against a live server with a real LLM (``docker compose up``):

    uv run python eval/run_eval.py                      # index must be complete
    uv run python eval/run_eval.py --index ./sample_docs  # index first and wait
    uv run python eval/run_eval.py --ids 3,6 --phrasings question --json out.json

Every fact is asked in two phrasings (natural question, keywords) and scored on three
separate stages, so a failure points at the stage that caused it:

  retrieval — find_relevant_docs(query, top_k=RETRIEVE_K) has the expected file;
  grader    — ask_question used the expected file as a chunk the grader accepted
              (its sources, not a low-confidence fallback);
  answer    — the final answer contains every expected value.

The grader number is read from the sources of the answer, which are the accepted chunks
that fit the context budget (MAX_CONTEXT_CHARS); with the default budget that is all of
them for these questions.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from fastmcp import Client

FACTS = Path(__file__).with_name("facts.yaml")
PHRASINGS = ("question", "keywords")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[   ]", " ", text)
    text = re.sub(r"(\d) (?=\d{3}\b)", r"\1", text)  # 18 600 -> 18600
    return re.sub(r"(\d),(\d)", r"\1.\2", text)  # 4,35 -> 4.35


def has_values(answer: str, expect: list[list[str]]) -> bool:
    text = normalize(answer)
    return all(any(normalize(v) in text for v in group) for group in expect)


def matches(source: str | None, files: list[str]) -> bool:
    return bool(source) and any(source.replace("\\", "/").endswith(f) for f in files)


def data(result) -> dict[str, Any]:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def ensure_index(client: Client, index: str | None) -> dict:
    status = data(await client.call_tool("index_status"))
    if index and (status.get("empty") or status.get("indexing", {}).get("state") != "running"):
        await client.call_tool("index_folder", {"path": index})
    while True:
        status = data(await client.call_tool("index_status"))
        job = status.get("indexing") or {}
        if job.get("state") != "running":
            break
        print(f"indexing: {job.get('percent')}% (eta {job.get('eta_s')} s)", file=sys.stderr)
        await asyncio.sleep(15)
    if status.get("empty"):
        sys.exit("The knowledge base is empty: pass --index ./sample_docs")
    return status


async def run_case(client: Client, fact: dict, phrasing: str, top_k: int) -> dict:
    query = fact[phrasing]
    files = fact["files"]

    found = data(await client.call_tool("find_relevant_docs", {"query": query, "top_k": top_k}))
    ranks = [r["rank"] for r in found["results"] if matches(r["source"], files)]

    started = time.monotonic()
    try:
        answer = data(await client.call_tool("ask_question", {"question": query}))
    except Exception as exc:  # a timeout or tool error is a failed case, not a failed run
        answer = {"answer": "", "error": f"{type(exc).__name__}: {exc}"}
    low = bool(answer.get("low_confidence"))
    accepted = not low and any(matches(s["source"], files) for s in answer.get("sources", []))
    return {
        "id": fact["id"],
        "phrasing": phrasing,
        "query": query,
        "retrieval": bool(ranks),
        "retrieval_rank": ranks[0] if ranks else None,
        "grader": accepted,
        "answer_ok": has_values(answer.get("answer", ""), fact["expect"]),
        "low_confidence": low,
        "seconds": round(time.monotonic() - started, 1),
        "answer": answer.get("answer", ""),
        "sources": answer.get("sources", []),
        "steps": answer.get("steps", []),
        "error": answer.get("error"),
    }


def mark(ok: bool) -> str:
    return "+" if ok else "-"


def report(rows: list[dict], llm: dict, top_k: int) -> str:
    lines = [
        f"LLM: {llm.get('provider')} {llm.get('model')} at {llm.get('base_url')}",
        "",
        "| # | phrasing | retrieval (rank) | grader | answer | low conf. | s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rank = f" ({r['retrieval_rank']})" if r["retrieval_rank"] else ""
        lines.append(
            f"| {r['id']} | {r['phrasing']} | {mark(r['retrieval'])}{rank} | {mark(r['grader'])} "
            f"| {mark(r['answer_ok'])} | {'yes' if r['low_confidence'] else ''} | {r['seconds']}"
            + (f" ({r['error']})" if r.get("error") else "")
            + " |"
        )
    lines += ["", f"| phrasing | retrieval@{top_k} | grader | answer |", "|---|---|---|---|"]
    for phrasing in [*PHRASINGS, "all"]:
        sel = [r for r in rows if phrasing in ("all", r["phrasing"])]
        if not sel:
            continue
        cells = [
            f"{sum(r[k] for r in sel)}/{len(sel)}" for k in ("retrieval", "grader", "answer_ok")
        ]
        lines.append(f"| {phrasing} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000/mcp")
    parser.add_argument("--index", help="index this server-side folder first, e.g. ./sample_docs")
    parser.add_argument("--ids", help="comma-separated fact ids (default: all)")
    parser.add_argument("--phrasings", default=",".join(PHRASINGS))
    parser.add_argument("--top-k", type=int, default=6, help="retrieval depth (RETRIEVE_K)")
    parser.add_argument("--json", type=Path, help="write raw results here (after every case)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per tool call")
    args = parser.parse_args()

    facts = yaml.safe_load(FACTS.read_text(encoding="utf-8"))["facts"]
    if args.ids:
        wanted = {int(i) for i in args.ids.split(",")}
        facts = [f for f in facts if f["id"] in wanted]
    phrasings = [p for p in args.phrasings.split(",") if p in PHRASINGS]

    rows = []
    async with Client(args.url, timeout=args.timeout) as client:
        status = await ensure_index(client, args.index)
        llm = status.get("llm", {})
        if not (llm.get("reachable") and llm.get("model_available")):
            sys.exit(f"LLM is not available: {llm}")
        for fact in facts:
            for phrasing in phrasings:
                row = await run_case(client, fact, phrasing, args.top_k)
                print(
                    f"#{row['id']} {phrasing}: retrieval {mark(row['retrieval'])} "
                    f"grader {mark(row['grader'])} answer {mark(row['answer_ok'])} "
                    f"({row['seconds']} s){' ' + row['error'] if row['error'] else ''}",
                    file=sys.stderr,
                )
                rows.append(row)
                if args.json:  # keep partial results if the run is interrupted
                    args.json.write_text(
                        json.dumps({"llm": llm, "results": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    print(report(rows, llm, args.top_k))


if __name__ == "__main__":
    asyncio.run(main())
