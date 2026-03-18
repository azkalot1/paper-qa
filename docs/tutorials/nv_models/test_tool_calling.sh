#!/usr/bin/env bash
# ============================================================================
# Test: NVIDIA endpoint tool-calling compatibility for PaperQA agent loop
# ============================================================================
#
# PaperQA's agent uses the OpenAI function-calling protocol:
#   - Sends tools=[...] + tool_choice="required"
#   - Agent picks a tool (e.g. paper_search)
#   - Response has content="" and tool_calls=[{...}]
#   - On the NEXT turn, that assistant message (content="") is included in
#     the conversation history
#
# BUG: NVIDIA endpoints reject content="" on assistant messages:
#   "String should have at least 1 character"
# OpenAI and most providers accept content="" or content=null on assistant
# messages that carry tool_calls.
#
# This script reproduces the exact two-turn flow.
#
# WORKAROUND: Replace content:"" with content:null on assistant messages
# before sending (implemented in test_PQA_singlePDF.py --fix-empty-content).
#
# Usage:
#   # Test against NVIDIA hosted endpoint
#   export NVIDIA_INFERENCE_KEY=nvapi-...
#   bash test_tool_calling.sh https://inference-api.nvidia.com/v1 nvidia/nvidia/nemotron-nano-12b-v2-vl
#
#   # Test against local NIM
#   bash test_tool_calling.sh http://localhost:8004/v1 nvidia/nemotron-nano-12b-v2-vl dummy
#
# ============================================================================

set -euo pipefail

BASE_URL="${1:?Usage: $0 <base_url> <model> [api_key]}"
MODEL="${2:?Usage: $0 <base_url> <model> [api_key]}"
API_KEY="${3:-${NVIDIA_INFERENCE_KEY:-dummy}}"

TOOLS='[
  {"type":"function","function":{"name":"paper_search","description":"Search for papers.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
  {"type":"function","function":{"name":"gather_evidence","description":"Gather evidence from papers.","parameters":{"type":"object","properties":{"question":{"type":"string"}},"required":["question"]}}},
  {"type":"function","function":{"name":"gen_answer","description":"Generate answer from evidence.","parameters":{"type":"object","properties":{},"required":[]}}},
  {"type":"function","function":{"name":"complete","description":"Signal completion.","parameters":{"type":"object","properties":{"has_successful_answer":{"type":"boolean"}},"required":["has_successful_answer"]}}},
  {"type":"function","function":{"name":"reset","description":"Reset evidence.","parameters":{"type":"object","properties":{},"required":[]}}}
]'

echo "============================================"
echo "STEP 1: First agent turn (initial tool call)"
echo "============================================"
echo "POST ${BASE_URL}/chat/completions"
echo "Model: ${MODEL}"
echo ""

STEP1=$(curl -s "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
  "model": "'"${MODEL}"'",
  "messages": [
    {"role":"system","content":"You are a helpful AI assistant."},
    {"role":"user","content":"Use the tools to answer the question: What experiments are carried out?\n\nThe current status is: Paper Count=0 | Relevant Papers=0 | Current Evidence=0"}
  ],
  "tools": '"${TOOLS}"',
  "tool_choice": "required"
}')

echo "Response:"
echo "${STEP1}" | python3 -m json.tool 2>/dev/null || echo "${STEP1}"

# Extract finish_reason and tool call
FINISH=$(echo "${STEP1}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['finish_reason'])" 2>/dev/null || echo "PARSE_ERROR")
echo ""
echo "finish_reason: ${FINISH}"

if [ "${FINISH}" != "tool_calls" ]; then
    echo "FAIL: Expected finish_reason=tool_calls, got ${FINISH}"
    echo "This model may not support OpenAI function calling."
    exit 1
fi

echo ""
echo "PASS: Step 1 succeeded (model returned tool_calls)."

# ============================================================================
echo ""
echo "========================================================"
echo "STEP 2a: Second turn WITH content:\"\" (the bug trigger)"
echo "========================================================"
echo ""
echo "Sending assistant message with content:\"\" (empty string)..."
echo "NVIDIA endpoints should reject this with 400."
echo ""

STEP2_BUG=$(curl -s -w "\n%{http_code}" "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
  "model": "'"${MODEL}"'",
  "messages": [
    {"role":"system","content":"You are a helpful AI assistant."},
    {"role":"user","content":"Use the tools to answer: What experiments are carried out?"},
    {"role":"assistant","content":"","tool_calls":[{"id":"call_001","type":"function","function":{"name":"paper_search","arguments":"{\"query\":\"experiments\"}"}}]},
    {"role":"tool","tool_call_id":"call_001","content":"Status: Paper Count=2 | Relevant Papers=0 | Current Evidence=0"}
  ],
  "tools": '"${TOOLS}"',
  "tool_choice": "required"
}')

HTTP_CODE=$(echo "${STEP2_BUG}" | tail -1)
BODY=$(echo "${STEP2_BUG}" | sed '$d')

echo "HTTP status: ${HTTP_CODE}"
if [ "${HTTP_CODE}" = "200" ]; then
    echo "PASS: Endpoint accepts content:\"\" (OpenAI-compatible behavior)."
    echo "Response:"
    echo "${BODY}" | python3 -m json.tool 2>/dev/null || echo "${BODY}"
else
    echo "FAIL: Endpoint rejected content:\"\" with HTTP ${HTTP_CODE}."
    echo "Error:"
    echo "${BODY}" | python3 -m json.tool 2>/dev/null || echo "${BODY}"
    echo ""
    echo "This is the bug PaperQA hits: aviary ToolSelector includes the"
    echo "assistant message with content:\"\" in the conversation history."
fi

# ============================================================================
echo ""
echo "========================================================"
echo "STEP 2b: Second turn WITH content:null (the workaround)"
echo "========================================================"
echo ""
echo "Sending assistant message with content:null..."
echo ""

STEP2_FIX=$(curl -s -w "\n%{http_code}" "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
  "model": "'"${MODEL}"'",
  "messages": [
    {"role":"system","content":"You are a helpful AI assistant."},
    {"role":"user","content":"Use the tools to answer: What experiments are carried out?"},
    {"role":"assistant","content":null,"tool_calls":[{"id":"call_001","type":"function","function":{"name":"paper_search","arguments":"{\"query\":\"experiments\"}"}}]},
    {"role":"tool","tool_call_id":"call_001","content":"Status: Paper Count=2 | Relevant Papers=0 | Current Evidence=0"}
  ],
  "tools": '"${TOOLS}"',
  "tool_choice": "required"
}')

HTTP_CODE_FIX=$(echo "${STEP2_FIX}" | tail -1)
BODY_FIX=$(echo "${STEP2_FIX}" | sed '$d')

echo "HTTP status: ${HTTP_CODE_FIX}"
if [ "${HTTP_CODE_FIX}" = "200" ]; then
    echo "PASS: Endpoint accepts content:null (workaround works)."
    echo "Response:"
    echo "${BODY_FIX}" | python3 -m json.tool 2>/dev/null || echo "${BODY_FIX}"
else
    echo "FAIL: Endpoint also rejected content:null with HTTP ${HTTP_CODE_FIX}."
    echo "Error:"
    echo "${BODY_FIX}" | python3 -m json.tool 2>/dev/null || echo "${BODY_FIX}"
fi

# ============================================================================
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "  Model:            ${MODEL}"
echo "  Endpoint:         ${BASE_URL}"
echo "  Step 1 (initial): ${FINISH}"
echo "  Step 2a (bug):    HTTP ${HTTP_CODE}"
echo "  Step 2b (fix):    HTTP ${HTTP_CODE_FIX}"
echo ""
if [ "${HTTP_CODE}" != "200" ] && [ "${HTTP_CODE_FIX}" = "200" ]; then
    echo "CONFIRMED: This endpoint has the empty-content bug."
    echo "Use --fix-empty-content (default on) in test_PQA_singlePDF.py."
elif [ "${HTTP_CODE}" = "200" ]; then
    echo "No bug: This endpoint accepts content:\"\" (fully OpenAI-compatible)."
else
    echo "Both variants failed. This endpoint may not support tool calling."
fi
