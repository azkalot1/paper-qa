#!/usr/bin/env python3
"""Test PaperQA end-to-end with a single PDF using NVIDIA NIM endpoints.

Runs six stages:
  1. add    -- parse one PDF and add to Docs (validates parser + embedding)
  2. query  -- Docs.aquery() on single paper (non-agent, no tools)
  3. agent  -- agent_query() with pre-loaded Docs (agent loop + tools)
  4. ask    -- ask() from scratch (full index build + agent)
  5. verify -- inspect parsing metadata (media detection, enrichment)
  6. multi  -- add ALL PDFs from --pdf-dir, then cross-paper query + agent

Each NIM endpoint (parse, embedding, VLM) can point to localhost or a remote
inference URL.  Defaults are localhost NIMs from launch_NIMs.sh.

Examples:
  # All local NIMs, default PDF
  python test_PQA_singlePDF.py

  # Specific PDF, skip slow stages
  python test_PQA_singlePDF.py --pdf papers/my_paper.pdf --stages add query

  # Local parse, remote embedding + VLM
  NVIDIA_INFERENCE_KEY=nvapi-... python test_PQA_singlePDF.py \\
    --parse-base-url http://localhost:8002/v1 \\
    --embedding-base-url https://inference-api.nvidia.com/v1 \\
    --vlm-base-url https://inference-api.nvidia.com/v1

  # Custom models
  python test_PQA_singlePDF.py \\
    --vlm-model nvidia/llama-3.3-nemotron-super-49b-v1 \\
    --embedding-model nvidia/llama-3.2-nv-embedqa-1b-v2
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
import pathlib
import sys
import traceback

# ---------------------------------------------------------------------------
# Logging setup (match notebook)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
for _name in ("LiteLLM", "litellm"):
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        _h = logging.StreamHandler(sys.stdout)
        _h.setLevel(logging.INFO)
        _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        _log.addHandler(_h)
    _log.propagate = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PARSE_BASE = "http://localhost:8002/v1"
DEFAULT_EMBED_BASE = "http://localhost:8003/v1"
DEFAULT_VLM_BASE = "http://localhost:8004/v1"
DEFAULT_PARSE_MODEL = "nvidia/nemotron-parse"
DEFAULT_EMBED_MODEL = "nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_VLM_MODEL = "nvidia/nvidia/nemotron-nano-12b-v2-vl"

DEFAULT_QUESTION = "What experiments are carried out?"
ALL_STAGES = ("add", "query", "agent", "ask", "verify", "multi")


# ---------------------------------------------------------------------------
# Settings builder
# ---------------------------------------------------------------------------
def _make_router(alias: str, model: str, base_url: str, api_key: str, **extra_params) -> dict:
    """Build a LiteLLM Router config for one model role."""
    litellm_params = {
        "model": f"openai/{model}" if not model.startswith("openai/") else model,
        "api_base": base_url,
        "api_key": api_key,
        "temperature": extra_params.pop("temperature", 0),
        "max_tokens": extra_params.pop("max_tokens", 12*2048),
        **extra_params,
    }
    return {
        "model_list": [{
            "model_name": alias,
            "litellm_params": litellm_params,
        }]
    }


def build_settings(args: argparse.Namespace) -> "Settings":
    from paperqa import Settings
    from paperqa.settings import AgentSettings, AnswerSettings, IndexSettings, ParsingSettings

    parse_pdf_fn = None
    reader_config: dict = {
        "chunk_chars": args.chunk_chars,
        "overlap": args.overlap,
    }

    if args.parser == "nemotron":
        from paperqa_nemotron import parse_pdf_to_pages
        parse_pdf_fn = parse_pdf_to_pages
        reader_config.update({
            "dpi": args.dpi,
            "api_params": {
                "api_base": args.parse_base_url,
                "api_key": args.parse_api_key,
                "model_name": args.parse_model,
                "temperature": 0,
                "max_tokens": 8995,
            },
        })

    # Per-role router configs
    llm_alias = "pqa-llm"
    llm_router = _make_router(
        llm_alias, args.llm_model, args.llm_base_url, args.llm_api_key,
    )

    summary_alias = "pqa-summary"
    summary_router = _make_router(
        summary_alias, args.summary_llm_model, args.summary_llm_base_url, args.summary_llm_api_key,
    )

    agent_alias = "pqa-agent"
    agent_router = _make_router(
        agent_alias, args.agent_llm_model, args.agent_llm_base_url, args.agent_llm_api_key,
    )

    enrichment_alias = "pqa-enrichment"
    enrichment_router = _make_router(
        enrichment_alias, args.enrichment_llm_model, args.enrichment_llm_base_url, args.enrichment_llm_api_key,
    )

    embedding_config = {
        "kwargs": {
            "api_base": args.embedding_base_url,
            "api_key": args.embedding_api_key,
            "encoding_format": "float",
            "input_type": "passage",
        }
    }

    parsing_settings = ParsingSettings(
        use_doc_details=False,
        reader_config=reader_config,
        enrichment_llm=enrichment_alias,
        enrichment_llm_config=enrichment_router,
        multimodal=True,
    )
    if parse_pdf_fn is not None:
        parsing_settings.parse_pdf = parse_pdf_fn

    papers_dir = pathlib.Path(args.pdf).resolve().parent

    settings = Settings(
        llm=llm_alias,
        llm_config=llm_router,
        summary_llm=summary_alias,
        summary_llm_config=summary_router,
        embedding=f"openai/{args.embedding_model}",
        embedding_config=embedding_config,
        temperature=0,
        verbosity=args.verbosity,
        answer=AnswerSettings(evidence_k=args.evidence_k, answer_max_sources=args.max_sources),
        parsing=parsing_settings,
        agent=AgentSettings(
            agent_llm=agent_alias,
            agent_llm_config=agent_router,
            index=IndexSettings(paper_directory=papers_dir),
        ),
    )
    return settings


# ---------------------------------------------------------------------------
# LLM call tracer -- monkey-patches litellm to log every call
# ---------------------------------------------------------------------------
def _fix_empty_content(messages: list[dict]) -> list[dict]:
    """Replace empty-string content with None on assistant messages.

    NVIDIA endpoints reject content="" on assistant messages that carry
    tool_calls.  OpenAI and most providers accept it, but NVIDIA requires
    content to be null/absent or at least 1 char.  This patch makes the
    messages compatible without changing semantics.
    """
    fixed = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            if content is not None and isinstance(content, str) and content.strip() == "":
                m = {**m, "content": None}
        fixed.append(m)
    return fixed


class LiteLLMCallTracer:
    """Wraps litellm.acompletion and litellm.aembedding to print what goes where."""

    def __init__(self, enabled: bool = True, fix_empty_content: bool = True):
        self.enabled = enabled
        self.fix_empty_content = fix_empty_content
        self._call_num = 0
        self._orig_acompletion = None
        self._orig_aembedding = None

    def install(self) -> None:
        if not self.enabled and not self.fix_empty_content:
            return
        import litellm
        self._orig_acompletion = litellm.acompletion
        self._orig_aembedding = litellm.aembedding

        tracer = self

        async def traced_acompletion(*args, **kwargs):
            tracer._call_num += 1
            n = tracer._call_num
            model = kwargs.get("model", args[0] if args else "?")
            api_base = kwargs.get("api_base", "?")
            messages = kwargs.get("messages", [])

            # Fix NVIDIA endpoint compat: replace empty-string content
            # with None on assistant messages (tool-call responses have
            # content="" which OpenAI accepts but NVIDIA rejects).
            if tracer.fix_empty_content:
                messages = _fix_empty_content(messages)
                kwargs["messages"] = messages

            if tracer.enabled:
                role_hint = _guess_role(messages, model)
                _print_header(n, "LLM", model, api_base, role_hint)
                _print_messages_preview(messages)
            result = await tracer._orig_acompletion(*args, **kwargs)
            if tracer.enabled:
                _print_response_preview(n, result)
            return result

        async def traced_aembedding(*args, **kwargs):
            tracer._call_num += 1
            n = tracer._call_num
            model = kwargs.get("model", args[0] if args else "?")
            api_base = kwargs.get("api_base", "?")
            inp = kwargs.get("input", [])
            count = len(inp) if isinstance(inp, list) else 1
            if tracer.enabled:
                _print_header(n, "EMBED", model, api_base, f"{count} text(s)")
                if isinstance(inp, list) and inp:
                    preview = str(inp[0])[:120]
                    print(f"  | input[0]: {preview}...")
            result = await tracer._orig_aembedding(*args, **kwargs)
            if tracer.enabled and hasattr(result, "data") and result.data:
                dim = len(result.data[0].get("embedding", [])) if isinstance(result.data[0], dict) else len(getattr(result.data[0], "embedding", []))
                print(f"  | -> dim={dim}, count={len(result.data)}")
            return result

        litellm.acompletion = traced_acompletion
        litellm.aembedding = traced_aembedding
        parts = []
        if self.enabled:
            parts.append("tracing")
        if self.fix_empty_content:
            parts.append("empty-content fix")
        print(f"[tracer] LiteLLM patched: {', '.join(parts)}.\n")

    def uninstall(self) -> None:
        if self._orig_acompletion is None:
            return
        import litellm
        litellm.acompletion = self._orig_acompletion
        litellm.aembedding = self._orig_aembedding
        self._orig_acompletion = None
        self._orig_aembedding = None


def _content_to_str(content) -> str:
    """Flatten any message content shape to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", "") or f"[{p.get('type', '')}]")
            else:
                parts.append(str(p))
        return " ".join(parts)
    return str(content)


def _guess_role(messages: list, model: str) -> str:
    """Best-effort guess which PaperQA role this call belongs to."""
    if not messages:
        return "unknown"
    combined = " ".join(
        _content_to_str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
        for m in messages[:2]
    ).lower()[:600]
    if "provide the citation" in combined or "mla format" in combined:
        return "MAIN-LLM (citation)"
    if "answer the question below" in combined or "answer in a direct" in combined:
        return "MAIN-LLM (answer)"
    if "relevance_score" in combined or '"summary"' in combined:
        return "SUMMARY-LLM (evidence)"
    if "you are analyzing an image" in combined or "irrelevant" in combined:
        return "ENRICHMENT-LLM (media)"
    if "paper_search" in combined or "gather_evidence" in combined or "gen_answer" in combined:
        return "AGENT-LLM (tool select)"
    if "search query" in combined:
        return "MAIN-LLM (search gen)"
    return "LLM"


def _print_header(n: int, kind: str, model: str, api_base: str, role: str) -> None:
    print(f"\n{'~'*60}")
    print(f"  [{n}] {kind}  role={role}")
    print(f"       model={model}")
    print(f"       api_base={api_base}")


def _print_messages_preview(messages: list, max_chars: int = 300) -> None:
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role", "?"), m.get("content", "")
        else:
            role, content = getattr(m, "role", "?"), getattr(m, "content", "")
        text = _content_to_str(content)
        # Mark images explicitly in the preview
        if isinstance(content, list):
            has_image = any(
                (isinstance(p, dict) and p.get("type") == "image_url")
                for p in content
            )
            if has_image:
                text = "[+IMAGE] " + text
        print(f"  | {role}: {text[:max_chars]}{'...' if len(text) > max_chars else ''}")


def _print_response_preview(n: int, result) -> None:
    try:
        choice = result.choices[0] if result.choices else None
        if choice is None:
            print(f"  | -> (no choices)")
            return
        msg = choice.message
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                fn = getattr(tc, "function", tc)
                print(f"  | -> tool_call: {getattr(fn,'name','?')}({str(getattr(fn,'arguments',''))[:200]})")
        else:
            text = getattr(msg, "content", None) or ""
            print(f"  | -> {text[:300]}{'...' if len(text) > 300 else ''}")
    except Exception as exc:
        print(f"  | -> (response parse error: {exc})")


def print_settings_map(args: argparse.Namespace) -> None:
    """Print a table showing which model/endpoint is assigned to each role."""
    print("\n" + "=" * 70)
    print("  Model-to-role mapping")
    print("=" * 70)
    rows = [
        ("Main LLM (answer)", args.llm_model, args.llm_base_url),
        ("Summary LLM (evidence)", args.summary_llm_model, args.summary_llm_base_url),
        ("Agent LLM (tool select)", args.agent_llm_model, args.agent_llm_base_url),
        ("Enrichment LLM (media)", args.enrichment_llm_model, args.enrichment_llm_base_url),
        ("Embedding", args.embedding_model, args.embedding_base_url),
        ("Parse NIM", args.parse_model, args.parse_base_url),
    ]
    for role, model, url in rows:
        print(f"  {role:30s}  {model}")
        print(f"  {'':30s}  -> {url}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
async def stage_add(pdf_path: str, settings: "Settings") -> "Docs":
    from paperqa import Docs

    print("\n" + "=" * 60)
    print("STAGE: add -- parse PDF and add to Docs")
    print("=" * 60)
    docs = Docs()
    print(f"Adding {pdf_path} ...")
    await docs.aadd(pdf_path, settings=settings)
    print(f"  Total docs: {len(docs.docs)}")
    for doc in docs.docs.values():
        print(f"  - {doc.docname}: {doc.citation[:100]}...")
    print(f"  Total text chunks: {len(docs.texts)}")
    return docs


async def stage_query(docs: "Docs", question: str, settings: "Settings") -> None:
    print("\n" + "=" * 60)
    print("STAGE: query -- Docs.aquery() (no agent)")
    print("=" * 60)
    print(f"  LLM (answer gen):  {settings.llm}")
    print(f"  Summary LLM:       {settings.summary_llm}")
    print(f"  Embedding:         {settings.embedding}")
    print(f"  Texts in index:    {len(docs.texts)}")
    print()
    session = await docs.aquery(question, settings=settings)
    print(f"\nQuestion: {session.question}")
    print(f"\nAnswer ({len(session.answer) if session.answer else 0} chars):")
    print(session.answer or "(None -- model returned empty response)")
    print(f"\nContexts used: {len(session.contexts)}")
    for i, ctx in enumerate(session.contexts[:5], 1):
        print(f"  [{i}] score={ctx.score}  source={ctx.text.name}")
        print(f"      summary: {ctx.context[:150]}...")
    print(f"\nReferences:\n{session.references}")
    print(f"\nTokens: {session.token_counts}")


async def stage_agent(docs: "Docs", question: str, settings: "Settings") -> None:
    from paperqa.agents.main import agent_query

    print("\n" + "=" * 60)
    print("STAGE: agent -- agent_query() with pre-loaded Docs")
    print("=" * 60)
    print(f"  Agent LLM:  {settings.agent.agent_llm}")
    print(f"  Agent type: {settings.agent.agent_type}")
    print()
    response = await agent_query(question, settings, docs=docs)
    print(f"\nStatus: {response.status}")
    print(f"Answer ({len(response.session.answer) if response.session.answer else 0} chars):")
    print(response.session.answer or "(None -- model returned empty response)")
    print(f"\nTool history: {response.session.tool_history}")
    print(f"Contexts: {len(response.session.contexts)}")
    for i, ctx in enumerate(response.session.contexts[:3], 1):
        print(f"  [{i}] score={ctx.score}  source={ctx.text.name}")


async def stage_ask(question: str, settings: "Settings") -> None:
    from paperqa import ask

    print("\n" + "=" * 60)
    print("STAGE: ask -- full pipeline from scratch")
    print("=" * 60)
    print(f"  Paper dir: {settings.agent.index.paper_directory}")
    print()
    response = await ask(question, settings=settings)
    print(f"\nStatus: {response.status}")
    print(f"Question: {response.session.question}")
    print(f"Answer ({len(response.session.answer) if response.session.answer else 0} chars):")
    print(response.session.answer or "(None -- model returned empty response)")
    print(f"\nTool history: {response.session.tool_history}")
    print(f"Contexts: {len(response.session.contexts)}")
    for i, ctx in enumerate(response.session.contexts[:3], 1):
        print(f"  [{i}] score={ctx.score}  source={ctx.text.name}")
        print(f"      {ctx.context[:200]}...")


async def stage_verify(pdf_path: str, settings: "Settings") -> None:
    from paperqa import Docs

    print("\n" + "=" * 60)
    print("STAGE: verify -- inspect parsing metadata")
    print("=" * 60)
    docs = Docs()
    await docs.aadd(pdf_path, settings=settings)

    total_media = 0
    media_types: dict[str, int] = {}
    enriched_count = 0

    for text in docs.texts:
        for media in text.media:
            total_media += 1
            mt = media.info.get("type", "unknown")
            media_types[mt] = media_types.get(mt, 0) + 1
            if media.info.get("enriched_description"):
                enriched_count += 1

    print(f"\nMedia detection:")
    print(f"  Total media items: {total_media}")
    print(f"  With enrichment:   {enriched_count}")
    if media_types:
        print("  Types:")
        for mt, count in sorted(media_types.items()):
            print(f"    {mt}: {count}")

    if docs.texts:
        print(f"\nSample parsed text (first chunk, 500 chars):")
        print(f"  {docs.texts[0].text[:500]}...")

    if total_media > 0 and media_types:
        print("\nCONFIRMED: Nemotron-parse media detection present.")
    else:
        print("\nNo structured media detected; check NIM logs for POST evidence.")


async def stage_multi(pdf_dir: str, question: str, settings: "Settings") -> None:
    """Add ALL PDFs in a directory, then query across them.

    This tests cross-document retrieval: the agent must search a Tantivy
    index built over multiple papers, retrieve chunks from different sources,
    and synthesize an answer that draws on evidence from several documents.

    Stages inside:
      1. Discover all PDFs in pdf_dir.
      2. Create a single Docs object and aadd() each PDF (parse + embed).
      3. Run docs.aquery() over the combined collection (no agent, tests
         retrieval across papers).
      4. Run agent_query() with the combined Docs (agent loop picks tools,
         searches across all loaded papers).
    """
    from paperqa import Docs
    from paperqa.agents.main import agent_query

    pdf_dir_path = pathlib.Path(pdf_dir)
    pdfs = sorted(pdf_dir_path.glob("*.pdf"))

    print("\n" + "=" * 60)
    print("STAGE: multi -- add all PDFs, then cross-paper query")
    print("=" * 60)
    print(f"  Directory: {pdf_dir_path}")
    print(f"  PDFs found: {len(pdfs)}")
    for i, p in enumerate(pdfs):
        size_kb = p.stat().st_size / 1024
        print(f"    [{i}] {p.name}  ({size_kb:.0f} KB)")

    if len(pdfs) < 2:
        print("\n  Need at least 2 PDFs for a meaningful multi-paper test.")
        print("  Add more PDFs to the directory or use download_papers.py.")
        return

    # -- Step 1: add all PDFs --
    docs = Docs()
    added, failed = 0, 0
    for i, pdf_path in enumerate(pdfs):
        tag = f"[{i + 1}/{len(pdfs)}]"
        try:
            await docs.aadd(str(pdf_path), settings=settings)
            added += 1
            print(f"  {tag} added: {pdf_path.name}")
        except Exception as exc:
            failed += 1
            print(f"  {tag} FAILED: {pdf_path.name} -- {exc}")

    print(f"\n  Added {added}/{len(pdfs)} papers, {failed} failed.")
    print(f"  Total docs: {len(docs.docs)}")
    print(f"  Total text chunks: {len(docs.texts)}")
    for doc in docs.docs.values():
        print(f"    - {doc.docname}: {doc.citation[:80]}...")

    # -- Step 2: cross-paper aquery (no agent) --
    print(f"\n  --- Cross-paper aquery (no agent) ---")
    print(f"  Question: {question}")
    session = await docs.aquery(question, settings=settings)
    print(f"  Answer ({len(session.answer) if session.answer else 0} chars):")
    print(f"  {(session.answer or '(empty)')[:500]}")
    print(f"  Contexts: {len(session.contexts)}")
    sources_seen = set()
    for ctx in session.contexts[:5]:
        sources_seen.add(ctx.text.doc.docname)
        print(f"    score={ctx.score}  source={ctx.text.name}")

    cross_paper = len(sources_seen) > 1
    print(f"\n  Sources in answer: {sources_seen}")
    print(f"  Cross-paper retrieval: {'YES' if cross_paper else 'NO (all from one paper)'}")

    # -- Step 3: cross-paper agent_query --
    print(f"\n  --- Cross-paper agent_query (with tools) ---")
    response = await agent_query(question, settings, docs=docs)
    print(f"  Status: {response.status}")
    print(f"  Answer ({len(response.session.answer) if response.session.answer else 0} chars):")
    print(f"  {(response.session.answer or '(empty)')[:500]}")
    print(f"  Tool history: {response.session.tool_history}")
    agent_sources = {ctx.text.doc.docname for ctx in response.session.contexts}
    print(f"  Sources: {agent_sources}")
    print(f"  Cross-paper: {'YES' if len(agent_sources) > 1 else 'NO'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def find_default_pdf() -> str | None:
    script_dir = pathlib.Path(__file__).resolve().parent
    pdfs = sorted(glob.glob(str(script_dir / "papers" / "*.pdf")))
    return pdfs[0] if pdfs else None


def parse_args() -> argparse.Namespace:
    default_pdf = find_default_pdf()

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pdf", default=default_pdf,
                    help="Path to PDF file (for add/query/agent/verify stages). Default: first PDF in papers/.")
    p.add_argument("--pdf-dir", default=str(pathlib.Path(__file__).resolve().parent / "papers"),
                    help="Directory of PDFs for the 'multi' stage. Default: papers/ next to this script.")
    p.add_argument("--question", default=DEFAULT_QUESTION,
                    help="Question to ask the paper(s).")
    p.add_argument("--stages", nargs="+", default=list(ALL_STAGES),
                    choices=ALL_STAGES,
                    help="Which stages to run (default: all).")

    # -- Parse NIM (only used for PDF parsing, not an LLM role) ---------------
    g = p.add_argument_group("Parse NIM")
    g.add_argument("--parse-base-url", default=os.getenv("PQA_PARSE_API_BASE", DEFAULT_PARSE_BASE))
    g.add_argument("--parse-api-key", default=os.getenv("PQA_PARSE_API_KEY", "dummy"))
    g.add_argument("--parse-model", default=DEFAULT_PARSE_MODEL)

    # -- Embedding -----------------------------------------------------------
    g = p.add_argument_group("Embedding model")
    g.add_argument("--embedding-base-url", default=os.getenv("PQA_EMBEDDING_API_BASE", DEFAULT_EMBED_BASE))
    g.add_argument("--embedding-api-key",
                    default=os.getenv("PQA_EMBEDDING_API_KEY", os.getenv("NVIDIA_INFERENCE_KEY", "dummy")))
    g.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)

    # -- Per-role LLM endpoints ----------------------------------------------
    # Each role defaults to the VLM endpoint / model so a single --vlm-* set
    # still works.  Override individual roles to use different models.
    _vlm_base = os.getenv("PQA_VLM_API_BASE", DEFAULT_VLM_BASE)
    _vlm_key = os.getenv("PQA_VLM_API_KEY", os.getenv("NVIDIA_INFERENCE_KEY", "dummy"))

    g = p.add_argument_group(
        "LLM roles",
        "Each role defaults to --vlm-* values. Override per-role to split models.",
    )
    g.add_argument("--vlm-base-url", default=_vlm_base,
                    help="Shared default base URL for all LLM roles.")
    g.add_argument("--vlm-api-key", default=_vlm_key,
                    help="Shared default API key for all LLM roles.")
    g.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL,
                    help="Shared default model for all LLM roles.")

    g.add_argument("--llm-model", default=None, help="Main (answer) LLM model. Default: --vlm-model.")
    g.add_argument("--llm-base-url", default=None, help="Main LLM base URL. Default: --vlm-base-url.")
    g.add_argument("--llm-api-key", default=None, help="Main LLM API key. Default: --vlm-api-key.")

    g.add_argument("--summary-llm-model", default=None, help="Summary LLM model. Default: --vlm-model.")
    g.add_argument("--summary-llm-base-url", default=None, help="Summary LLM base URL. Default: --vlm-base-url.")
    g.add_argument("--summary-llm-api-key", default=None, help="Summary LLM API key. Default: --vlm-api-key.")

    g.add_argument("--agent-llm-model", default=None, help="Agent (tool-selection) LLM model. Default: --vlm-model.")
    g.add_argument("--agent-llm-base-url", default=None, help="Agent LLM base URL. Default: --vlm-base-url.")
    g.add_argument("--agent-llm-api-key", default=None, help="Agent LLM API key. Default: --vlm-api-key.")

    g.add_argument("--enrichment-llm-model", default=None, help="Enrichment (image captioning) LLM model. Default: --vlm-model.")
    g.add_argument("--enrichment-llm-base-url", default=None, help="Enrichment LLM base URL. Default: --vlm-base-url.")
    g.add_argument("--enrichment-llm-api-key", default=None, help="Enrichment LLM API key. Default: --vlm-api-key.")

    g = p.add_argument_group("Tracing / compat")
    g.add_argument("--trace", action="store_true",
                    help="Trace every LiteLLM call: print role, model, endpoint, input/output preview.")
    g.add_argument("--fix-empty-content", action="store_true", default=True,
                    help="Patch empty assistant content to None for NVIDIA endpoint compat (default: on).")
    g.add_argument("--no-fix-empty-content", dest="fix_empty_content", action="store_false",
                    help="Disable the empty-content fix.")

    g = p.add_argument_group("Parser / RAG tuning")
    g.add_argument("--parser", choices=("nemotron", "default"), default="nemotron",
                    help="PDF parser to use. 'default' uses installed fallback (pypdf/pymupdf).")
    g.add_argument("--chunk-chars", type=int, default=5000)
    g.add_argument("--overlap", type=int, default=250)
    g.add_argument("--dpi", type=int, default=150, help="Page render DPI for nemotron parser.")
    g.add_argument("--evidence-k", type=int, default=5)
    g.add_argument("--max-sources", type=int, default=3)
    g.add_argument("--verbosity", type=int, default=3, choices=(0, 1, 2, 3))

    args = p.parse_args()

    # Fall back per-role values to the shared --vlm-* defaults
    for role in ("llm", "summary_llm", "agent_llm", "enrichment_llm"):
        for suffix, vlm_attr in (("model", "vlm_model"), ("base_url", "vlm_base_url"), ("api_key", "vlm_api_key")):
            attr = f"{role}_{suffix}"
            if getattr(args, attr) is None:
                setattr(args, attr, getattr(args, vlm_attr))

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main() -> int:
    args = parse_args()

    needs_single_pdf = any(s in args.stages for s in ("add", "query", "agent", "verify"))
    if needs_single_pdf:
        if not args.pdf:
            print("No PDF found. Provide --pdf or place a PDF in papers/.")
            return 2
        if not pathlib.Path(args.pdf).exists():
            print(f"PDF not found: {args.pdf}")
            return 2

    print("=" * 60)
    print("PaperQA single-PDF test")
    print("=" * 60)
    print(f"PDF:           {args.pdf}")
    print(f"Question:      {args.question}")
    print(f"Stages:        {', '.join(args.stages)}")
    print(f"Parser:        {args.parser}")
    print(f"Parse NIM:     {args.parse_base_url}  model={args.parse_model}")
    print(f"Embedding:     {args.embedding_base_url}  model={args.embedding_model}")
    print(f"LLM (answer):  {args.llm_base_url}  model={args.llm_model}")
    print(f"Summary LLM:   {args.summary_llm_base_url}  model={args.summary_llm_model}")
    print(f"Agent LLM:     {args.agent_llm_base_url}  model={args.agent_llm_model}")
    print(f"Enrichment:    {args.enrichment_llm_base_url}  model={args.enrichment_llm_model}")
    print(f"Trace:         {args.trace}")

    print_settings_map(args)

    needs_patch = args.trace or args.fix_empty_content
    tracer = LiteLLMCallTracer(enabled=needs_patch, fix_empty_content=args.fix_empty_content)
    tracer.install()

    settings = build_settings(args)
    docs = None
    results: dict[str, str] = {}

    for stage in args.stages:
        try:
            if stage == "add":
                docs = await stage_add(args.pdf, settings)
                results[stage] = "PASS"

            elif stage == "query":
                if docs is None:
                    docs = await stage_add(args.pdf, settings)
                await stage_query(docs, args.question, settings)
                results[stage] = "PASS"

            elif stage == "agent":
                if docs is None:
                    docs = await stage_add(args.pdf, settings)
                await stage_agent(docs, args.question, settings)
                results[stage] = "PASS"

            elif stage == "ask":
                await stage_ask(args.question, settings)
                results[stage] = "PASS"

            elif stage == "verify":
                await stage_verify(args.pdf, settings)
                results[stage] = "PASS"

            elif stage == "multi":
                await stage_multi(args.pdf_dir, args.question, settings)
                results[stage] = "PASS"

        except Exception:
            traceback.print_exc()
            results[stage] = "FAIL"

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for stage, status in results.items():
        print(f"  {stage:8s} {status}")
    return 0 if all(v == "PASS" for v in results.values()) else 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
