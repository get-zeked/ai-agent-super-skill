---
name: ai-agent-super-skill
description: Comprehensive AI agent building skill merging Perplexity Computer's skill creation, webserver, and automation capabilities with Claude Code's agent orchestration, MCP server building, RAG system construction, subagent coordination, parallel agent dispatching, prompt optimization, and execution planning. Covers designing AI agents, building MCP servers, creating RAG pipelines, orchestrating multi-agent systems, optimizing prompts, and deploying AI-powered workflows. Use when building AI agents, creating MCP servers, designing RAG systems, coordinating subagents, optimizing prompts, or architecting any AI-powered automation.
license: MIT
metadata:
  author: get-zeked
  version: '1.0'
---

# AI Agent Builder Super-Skill

A comprehensive reference for designing, building, and deploying AI agents — from single-tool bots to production multi-agent systems — merging best practices from Claude Code's agent orchestration patterns with Perplexity Computer's deployment infrastructure.

---

## Table of Contents

1. [Gap Analysis Table](#1-gap-analysis-table)
2. [Agent Architecture & Design Patterns](#2-agent-architecture--design-patterns)
3. [MCP Server Development](#3-mcp-server-development)
4. [RAG System Construction](#4-rag-system-construction)
5. [Subagent Coordination](#5-subagent-coordination)
6. [Execution Planning & Verification](#6-execution-planning--verification)
7. [Prompt Engineering & Optimization](#7-prompt-engineering--optimization)
8. [ML Integration for Agents](#8-ml-integration-for-agents)
9. [Skill & Capability Creation](#9-skill--capability-creation)
10. [Backend Infrastructure for Agents](#10-backend-infrastructure-for-agents)
11. [Agent Deployment & Monitoring](#11-agent-deployment--monitoring)
12. [Unique Perplexity Computer Capabilities](#12-unique-perplexity-computer-capabilities)

---

## 1. Gap Analysis Table

This table maps each capability domain to its source skill, coverage level, and any gaps filled by this super-skill.

| Capability | Source Skill(s) | Coverage | Gaps Filled Here |
|------------|----------------|----------|-----------------|
| Agent architecture (ReAct, Plan-Execute) | senior-prompt-engineer | Partial — workflow diagrams only | Full pattern library with code |
| Multi-agent orchestration | subagent-driven-development, dispatching-parallel-agents | Strong process, no code | Integration patterns, conflict detection |
| MCP server building | mcp-builder | Full (4-phase process) | Perplexity-compatible CGI deployment |
| RAG pipeline construction | senior-ml-engineer, senior-prompt-engineer | Chunking + DB selection tables | End-to-end pipeline code |
| Prompt engineering | senior-prompt-engineer | Pattern reference table | Advanced chain-of-thought + meta-prompting |
| Subagent task dispatch | subagent-driven-development | Process diagrams | Template prompts with full context injection |
| Parallel agent dispatch | dispatching-parallel-agents | Decision tree + examples | Conflict detection, state isolation |
| Plan execution with checkpoints | executing-plans | Step-by-step process | Batch sizing, rollback strategies |
| MLOps / model deployment | senior-ml-engineer | Docker + k8s templates | Agent-specific serving patterns |
| Backend webhooks/SQLite | webserver | Full CGI-bin reference | Agent memory persistence layer |
| Skill packaging (SKILL.md) | create-skill (Perplexity) | YAML frontmatter format | Validation pipeline, versioning |
| Deployment & observability | website-building (Perplexity) | UI deployment only | Agent health checks, trace logging |
| Perplexity 400+ integrations | Perplexity Computer native | Available but undocumented | Integration mapping for agent use |
| Scheduled monitoring | Perplexity Computer native | Not in any skill | Agent heartbeat and drift triggers |

---

## 2. Agent Architecture & Design Patterns

### 2.1 Core Agent Loop

Every agent follows a fundamental observe → think → act → observe loop. The differences between architectures lie in how deeply they plan before acting and how they handle tool results.

```
+---------------------------------------------+
|                 AGENT LOOP                  |
|                                             |
|   Input/Observation                         |
|         |                                   |
|         v                                   |
|   +-------------+                           |
|   |    Think    | <---------------------+   |
|   |  (Reason)   |                       |   |
|   +------+------+                       |   |
|          |                              |   |
|          v                              |   |
|   +-------------+    No more tools      |   |
|   |  Select     |------------------>    |   |
|   |  Action     |                  |    |   |
|   +------+------+                  |    |   |
|          |                         |    |   |
|          v                         v    |   |
|   +-------------+           +----------+|   |
|   |  Execute    |           |  Final   ||   |
|   |  Tool/API   |           |  Answer  ||   |
|   +------+------+           +----------+|   |
|          |                              |   |
|          v                              |   |
|   +-------------+                       |   |
|   |  Observe    |-----------------------+   |
|   |  Result     |                           |
|   +-------------+                           |
+---------------------------------------------+
```

### 2.2 Architecture Patterns

#### Pattern A: ReAct (Reason + Act)

Best for: open-ended research, customer support, tool-use agents.

**How it works:** The agent interleaves reasoning traces (Thought:) with actions (Action:) and observations (Observation:) in a single conversation thread.

```python
REACT_SYSTEM_PROMPT = """
You are a research agent. For every task:

1. THOUGHT: Reason about what you know and what you need
2. ACTION: Choose one tool to call
3. OBSERVATION: Read the tool result
4. Repeat until you have enough information
5. FINAL ANSWER: Synthesize and respond

Available tools: {tool_list}

Format strictly:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <tool_arguments as JSON>
Observation: <tool result — filled by system>
... (repeat)
Final Answer: <your complete response>
"""

def react_agent(query: str, tools: dict, llm, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT.format(
            tool_list="\n".join(f"- {k}: {v['description']}" for k, v in tools.items())
        )},
        {"role": "user", "content": query}
    ]
    
    for iteration in range(max_iterations):
        response = llm.complete(messages)
        
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
        
        # Parse Action / Action Input
        action_line = [l for l in response.split("\n") if l.startswith("Action:")]
        input_line  = [l for l in response.split("\n") if l.startswith("Action Input:")]
        
        if not action_line:
            break
        
        tool_name = action_line[0].replace("Action:", "").strip()
        tool_input = json.loads(input_line[0].replace("Action Input:", "").strip())
        
        # Execute tool
        if tool_name in tools:
            observation = tools[tool_name]["fn"](**tool_input)
        else:
            observation = f"Error: Unknown tool '{tool_name}'"
        
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    
    return "Agent reached max iterations without a final answer."
```

#### Pattern B: Plan-and-Execute

Best for: complex multi-step workflows, code generation, structured report creation.

**How it works:** A planner LLM generates a complete task list first; executor agents complete each step sequentially or in parallel.

```python
PLANNER_PROMPT = """
Given this goal: {goal}

Create a numbered execution plan. Each step must be:
- Atomic: one clear action
- Verifiable: has a concrete success criterion
- Independent (where possible): can run without other steps completing first

Output format:
PLAN:
1. [Step description] | SUCCESS: [verification criterion] | DEPS: [step numbers or NONE]
2. ...
"""

EXECUTOR_PROMPT = """
Execute this step exactly:
{step}

Context from previous steps:
{context}

Available tools: {tools}

Return:
- RESULT: what you produced
- STATUS: SUCCESS or FAILED
- NOTES: any issues or observations
"""

class PlanExecuteAgent:
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools
    
    def run(self, goal: str) -> dict:
        # Phase 1: Plan
        plan_response = self.planner.complete(
            PLANNER_PROMPT.format(goal=goal)
        )
        steps = self._parse_plan(plan_response)
        
        # Phase 2: Execute
        results = {}
        for step in self._topological_sort(steps):
            context = {k: v["result"] for k, v in results.items() if v["status"] == "SUCCESS"}
            result = self.executor.complete(
                EXECUTOR_PROMPT.format(
                    step=step["description"],
                    context=json.dumps(context, indent=2),
                    tools=list(self.tools.keys())
                )
            )
            results[step["id"]] = self._parse_result(result)
        
        return results
```

#### Pattern C: Reflexion

Best for: code debugging, essay writing, tasks that benefit from self-critique.

**How it works:** After each attempt, the agent evaluates its own output, stores a reflection in memory, and retries.

```python
REFLEXION_EVALUATOR_PROMPT = """
Task: {task}
Attempt: {attempt}

Evaluate this attempt:
1. What did it get RIGHT? (be specific)
2. What did it get WRONG or MISS? (be specific)
3. What should the NEXT attempt do differently?

Score (0-10): 
Reflection: 
"""

class ReflexionAgent:
    def __init__(self, llm, max_attempts: int = 3, pass_threshold: float = 8.0):
        self.llm = llm
        self.max_attempts = max_attempts
        self.threshold = pass_threshold
        self.memory = []  # Persisted reflections
    
    def run(self, task: str) -> str:
        for attempt_num in range(self.max_attempts):
            # Inject prior reflections into context
            reflection_context = "\n".join(
                f"Attempt {i+1} reflection: {r}" for i, r in enumerate(self.memory)
            )
            
            attempt = self.llm.complete(
                f"Task: {task}\n\nPrior attempt learnings:\n{reflection_context}\n\nYour attempt:"
            )
            
            # Evaluate
            eval_response = self.llm.complete(
                REFLEXION_EVALUATOR_PROMPT.format(task=task, attempt=attempt)
            )
            score = float(re.search(r"Score \(0-10\):\s*([\d.]+)", eval_response).group(1))
            reflection = re.search(r"Reflection:\s*(.+)", eval_response, re.DOTALL).group(1).strip()
            
            self.memory.append(reflection)
            
            if score >= self.threshold:
                return attempt
        
        return attempt  # Return best attempt after max tries
```

#### Pattern D: Tool-Use Agent (Function Calling)

Best for: API integrations, data retrieval, modern LLM APIs that support native tool calling.

```python
import anthropic

def build_tool_agent(tools: list[dict], system: str = "") -> callable:
    """
    tools: list of Anthropic-format tool definitions
    Returns a function that runs the agent for a given query.
    """
    client = anthropic.Anthropic()
    
    def run(query: str, tool_executors: dict[str, callable]) -> str:
        messages = [{"role": "user", "content": query}]
        
        while True:
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages
            )
            
            # No tool calls — final answer
            if response.stop_reason == "end_turn":
                return response.content[0].text
            
            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    executor = tool_executors.get(block.name)
                    if executor:
                        result = executor(**block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })
            
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
    
    return run
```

### 2.3 Architecture Selection Guide

| Goal | Pattern | Reason |
|------|---------|--------|
| Open-ended research | ReAct | Flexible, self-correcting, handles unknown paths |
| Multi-step report generation | Plan-Execute | Predictable, auditable, checkpointable |
| Code writing / debugging | Reflexion | Self-critique loop improves quality over iterations |
| API integration / tool calling | Tool-Use | Native LLM feature, lower latency, less prompt engineering |
| Customer support bot | ReAct + Tool-Use | Hybrid: structured tools with flexible reasoning |
| Batch data processing | Plan-Execute with parallel dispatch | Speed via parallelism, structured output |
| Creative tasks (writing, design) | Reflexion | Quality improves with each self-critique cycle |

### 2.4 Multi-Agent System Topologies

```
TOPOLOGY 1: Hub-and-Spoke (Orchestrator + Specialists)

                    +----------------+
                    |  Orchestrator  |
                    |  (Coordinator) |
                    +-------+--------+
          +-----------------+-----------------+
          v                 v                 v
   +------------+   +------------+   +------------+
   |  Research  |   |   Coder    |   |  Writer    |
   |   Agent    |   |   Agent    |   |   Agent    |
   +------------+   +------------+   +------------+

Use for: complex tasks needing specialized expertise per sub-domain.
Orchestrator decomposes goal -> dispatches -> aggregates results.

TOPOLOGY 2: Pipeline (Assembly Line)

  Input -> [Extractor] -> [Transformer] -> [Validator] -> [Writer] -> Output

Use for: ETL, document processing, multi-stage generation tasks.
Each agent only sees the previous stage's output.

TOPOLOGY 3: Competitive / Debate

  Query -> Agent A --+
                    +---> Judge Agent ---> Final Answer
  Query -> Agent B --+

Use for: decisions requiring multiple perspectives, factual verification,
high-stakes outputs where consensus improves reliability.

TOPOLOGY 4: Peer Network (Gossip/Consensus)

  Agent 1 <---> Agent 2
     ^  \          ^
     |   \         |
     v    \        v
  Agent 4 <---> Agent 3

Use for: simulation, emergent behavior research, distributed problem solving.
High coordination overhead — avoid for production automation.
```

---

## 3. MCP Server Development

### 3.1 Four-Phase MCP Build Process

Building a production MCP server follows four phases. Do not skip phases — each builds on the previous.

**Phase 1: Research & Planning**
1. Fetch MCP spec: `https://modelcontextprotocol.io/sitemap.xml` then pages with `.md` suffix
2. Study the target API's documentation — auth requirements, rate limits, key endpoints
3. Decide: TypeScript (recommended) or Python (FastMCP)
4. List all tools, prioritizing comprehensive API coverage over convenience wrappers

**Phase 2: Implementation**

**Phase 3: Review & Test** — use MCP Inspector

**Phase 4: Create Evaluations** — 10 read-only, complex, verifiable questions

### 3.2 TypeScript MCP Server Template

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// --- Server Initialization ---------------------------------------------------
const server = new McpServer({
  name: "my-service-mcp",
  version: "1.0.0",
});

// --- Shared API Client --------------------------------------------------------
interface ApiConfig {
  baseUrl: string;
  apiKey: string;
}

class ServiceClient {
  constructor(private config: ApiConfig) {}

  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        "Authorization": `Bearer ${this.config.apiKey}`,
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(
        `API error ${response.status}: ${error}. ` +
        `Suggestion: Check your API key or verify the resource exists.`
      );
    }

    return response.json() as Promise<T>;
  }
}

const client = new ServiceClient({
  baseUrl: process.env.SERVICE_BASE_URL ?? "https://api.example.com",
  apiKey: process.env.SERVICE_API_KEY ?? "",
});

// --- Tool: List Items ---------------------------------------------------------
server.registerTool(
  "service_list_items",
  {
    description: "List items with optional filtering and pagination.",
    inputSchema: z.object({
      page:     z.number().int().min(1).default(1).describe("Page number (1-indexed)"),
      per_page: z.number().int().min(1).max(100).default(20).describe("Items per page"),
      filter:   z.string().optional().describe("Optional keyword filter"),
    }),
    annotations: {
      readOnlyHint:  true,
      destructiveHint: false,
      idempotentHint: true,
    },
  },
  async (params) => {
    const query = new URLSearchParams({
      page:     String(params.page),
      per_page: String(params.per_page),
      ...(params.filter ? { q: params.filter } : {}),
    });

    const data = await client.request<{ items: unknown[]; total: number }>(
      `/items?${query}`
    );

    return {
      content: [{
        type: "text",
        text: JSON.stringify(data, null, 2),
      }],
      structuredContent: data,
    };
  }
);

// --- Tool: Get Item -----------------------------------------------------------
server.registerTool(
  "service_get_item",
  {
    description: "Get a single item by ID.",
    inputSchema: z.object({
      id: z.string().describe("Item ID"),
    }),
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
    },
  },
  async (params) => {
    const item = await client.request(`/items/${params.id}`);
    return {
      content: [{ type: "text", text: JSON.stringify(item, null, 2) }],
      structuredContent: item,
    };
  }
);

// --- Tool: Create Item --------------------------------------------------------
server.registerTool(
  "service_create_item",
  {
    description: "Create a new item. Returns the created item with its assigned ID.",
    inputSchema: z.object({
      name:        z.string().min(1).describe("Item name"),
      description: z.string().optional().describe("Optional item description"),
      tags:        z.array(z.string()).optional().describe("Tag list"),
    }),
    annotations: {
      readOnlyHint:    false,
      destructiveHint: false,
      idempotentHint:  false,
    },
  },
  async (params) => {
    const item = await client.request("/items", {
      method: "POST",
      body: JSON.stringify(params),
    });
    return {
      content: [{ type: "text", text: JSON.stringify(item, null, 2) }],
      structuredContent: item,
    };
  }
);

// --- Transport ----------------------------------------------------------------
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("MCP server running on stdio");
```

### 3.3 Python FastMCP Server Template

```python
#!/usr/bin/env python3
"""FastMCP server template for Python-based MCP servers."""

import json
import os
from typing import Any

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# --- Server Initialization ---------------------------------------------------
mcp = FastMCP("my-service-mcp")

# --- API Client ---------------------------------------------------------------
BASE_URL = os.environ.get("SERVICE_BASE_URL", "https://api.example.com")
API_KEY  = os.environ.get("SERVICE_API_KEY", "")

async def api_request(endpoint: str, method: str = "GET", body: dict | None = None) -> Any:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method,
            f"{BASE_URL}{endpoint}",
            headers=headers,
            json=body,
        )
        if not response.is_success:
            raise ValueError(
                f"API error {response.status_code}: {response.text}. "
                "Check your credentials or verify the resource exists."
            )
        return response.json()

# --- Input Models -------------------------------------------------------------
class ListParams(BaseModel):
    page:     int   = Field(default=1,  ge=1,          description="Page number (1-indexed)")
    per_page: int   = Field(default=20, ge=1,  le=100, description="Items per page")
    filter:   str | None = Field(default=None,         description="Keyword filter")

class CreateParams(BaseModel):
    name:        str        = Field(...,       description="Item name")
    description: str | None = Field(None,      description="Optional description")
    tags:        list[str]  = Field(default=[], description="Tag list")

# --- Tools --------------------------------------------------------------------
@mcp.tool(description="List items with optional filtering and pagination.")
async def service_list_items(params: ListParams) -> str:
    query = f"?page={params.page}&per_page={params.per_page}"
    if params.filter:
        query += f"&q={params.filter}"
    data = await api_request(f"/items{query}")
    return json.dumps(data, indent=2)


@mcp.tool(description="Get a single item by ID.")
async def service_get_item(id: str) -> str:
    """id: Item ID to retrieve"""
    item = await api_request(f"/items/{id}")
    return json.dumps(item, indent=2)


@mcp.tool(description="Create a new item. Returns the created item with its assigned ID.")
async def service_create_item(params: CreateParams) -> str:
    item = await api_request("/items", method="POST", body=params.model_dump())
    return json.dumps(item, indent=2)


if __name__ == "__main__":
    mcp.run()
```

### 3.4 MCP Tool Design Checklist

Before shipping any MCP server, verify every tool against this checklist:

- [ ] Tool name uses `service_verb_noun` convention (e.g., `github_create_issue`)
- [ ] Description is a single sentence — concise, action-oriented
- [ ] All input fields have `description` populated
- [ ] Required vs. optional fields are correctly marked
- [ ] Numeric fields have `min`/`max` constraints
- [ ] Enum fields use `z.enum()`/`Literal` instead of free strings
- [ ] Annotations set: `readOnlyHint`, `destructiveHint`, `idempotentHint`
- [ ] Error messages suggest a remediation action
- [ ] Pagination supported for list endpoints
- [ ] `structuredContent` returned alongside text content
- [ ] Build compiles without errors: `npm run build` or `python -m py_compile`
- [ ] Tested with MCP Inspector: `npx @modelcontextprotocol/inspector`

### 3.5 MCP Error Message Patterns

Good error messages are diagnostic and actionable:

```typescript
// Bad
throw new Error("Not found");

// Good
throw new Error(
  `Item '${id}' not found. ` +
  `Use service_list_items to find valid IDs, or verify the item exists in the service.`
);

// Bad
throw new Error("Unauthorized");

// Good
throw new Error(
  `Authentication failed. ` +
  `Check that SERVICE_API_KEY is set and has the required 'items:read' scope. ` +
  `Generate a new key at https://service.example.com/settings/api-keys`
);
```

---

## 4. RAG System Construction

### 4.1 Complete RAG Pipeline

```
+---------------------------------------------------------------------+
|                       INGESTION PIPELINE                            |
|                                                                     |
|  Documents -> [Loader] -> [Chunker] -> [Embedder] -> [Vector Store] |
|                                          |                          |
|                                   [Metadata Store]                  |
+---------------------------------------------------------------------+
                                    |
                              (index built)
                                    |
+---------------------------------------------------------------------+
|                        QUERY PIPELINE                               |
|                                                                     |
|  Query -> [Query Embed] -> [Vector Search] -> [Reranker] -> [LLM]  |
|              |                  |                |                  |
|         [HyDE opt.]        [Metadata          [Context             |
|                              Filter]           Format]              |
+---------------------------------------------------------------------+
```

### 4.2 Full Python RAG Implementation

```python
#!/usr/bin/env python3
"""
Production RAG pipeline with chunking, embedding, retrieval, and reranking.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Data Models -------------------------------------------------------------
@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default="")
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]

@dataclass
class Chunk:
    content: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rerank_score: float = 0.0

# --- Chunking Strategies -----------------------------------------------------
class ChunkingStrategy:
    """Base class — override chunk()."""
    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError

class FixedSizeChunker(ChunkingStrategy):
    """Fixed token-count chunks with overlap. Good for general text."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap    = overlap
    
    def chunk(self, doc: Document) -> list[Chunk]:
        # Simple word-based split (use tiktoken for production token counting)
        words = doc.content.split()
        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                break
            chunks.append(Chunk(
                content=" ".join(chunk_words),
                doc_id=doc.doc_id,
                chunk_index=len(chunks),
                metadata=doc.metadata,
            ))
        return chunks

class SentenceChunker(ChunkingStrategy):
    """Sentence-boundary-aware chunking. Better for structured prose."""
    
    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.n  = sentences_per_chunk
        self.ov = overlap_sentences
    
    def chunk(self, doc: Document) -> list[Chunk]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', doc.content.strip())
        chunks = []
        step = self.n - self.ov
        
        for i in range(0, len(sentences), step):
            batch = sentences[i:i + self.n]
            if not batch:
                break
            chunks.append(Chunk(
                content=" ".join(batch),
                doc_id=doc.doc_id,
                chunk_index=len(chunks),
                metadata=doc.metadata,
            ))
        return chunks

class RecursiveChunker(ChunkingStrategy):
    """Hierarchical chunking — preserves section structure. Best for long docs."""
    
    SEPARATORS = ["\n\n", "\n", ". ", " "]
    
    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 100):
        self.max_size = max_chunk_size
        self.min_size = min_chunk_size
    
    def chunk(self, doc: Document) -> list[Chunk]:
        chunks = []
        self._split(doc.content, doc.doc_id, doc.metadata, 0, chunks, [])
        return chunks
    
    def _split(self, text, doc_id, metadata, depth, chunks, idx_counter):
        if len(text.split()) <= self.max_size or depth >= len(self.SEPARATORS):
            idx_counter.append(None)
            chunks.append(Chunk(
                content=text,
                doc_id=doc_id,
                chunk_index=len(chunks),
                metadata=metadata,
            ))
            return
        
        sep = self.SEPARATORS[depth]
        parts = text.split(sep)
        current = ""
        
        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if len(candidate.split()) <= self.max_size:
                current = candidate
            else:
                if current and len(current.split()) >= self.min_size:
                    self._split(current, doc_id, metadata, depth + 1, chunks, idx_counter)
                current = part
        
        if current:
            self._split(current, doc_id, metadata, depth + 1, chunks, idx_counter)

# --- Vector Store (using Chroma for local dev) --------------------------------
class VectorStore:
    """Minimal abstract vector store interface."""
    
    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError
    
    def query(self, embedding: list[float], top_k: int = 10,
              filter_metadata: dict | None = None) -> list[RetrievedChunk]:
        raise NotImplementedError

class ChromaVectorStore(VectorStore):
    """Local development store using Chroma."""
    
    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db"):
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def upsert(self, chunks: list[Chunk]) -> None:
        self.collection.upsert(
            ids=[f"{c.doc_id}_{c.chunk_index}" for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=[c.embedding for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
    
    def query(self, embedding: list[float], top_k: int = 10,
              filter_metadata: dict | None = None) -> list[RetrievedChunk]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata
        
        results = self.collection.query(**kwargs)
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(RetrievedChunk(
                chunk=Chunk(content=doc, doc_id=meta.get("doc_id",""), chunk_index=0, metadata=meta),
                score=1.0 - dist,  # cosine: distance -> similarity
            ))
        return retrieved

# --- Embedder ----------------------------------------------------------------
class Embedder:
    """Embed text using OpenAI-compatible API."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        import openai
        self.client = openai.OpenAI()
        self.model  = model
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
    
    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

# --- Reranker ----------------------------------------------------------------
class CrossEncoderReranker:
    """Rerank retrieved chunks with a cross-encoder for precision improvement."""
    
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)
    
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        pairs  = [(query, c.chunk.content) for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]

# --- RAG Pipeline ------------------------------------------------------------
class RAGPipeline:
    def __init__(
        self,
        chunker: ChunkingStrategy,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: CrossEncoderReranker | None = None,
        llm_fn: callable = None,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
    ):
        self.chunker      = chunker
        self.embedder     = embedder
        self.store        = vector_store
        self.reranker     = reranker
        self.llm          = llm_fn
        self.retrieval_k  = retrieval_top_k
        self.rerank_k     = rerank_top_k
    
    # -- Ingestion ------------------------------------------------------------
    def ingest(self, documents: list[Document]) -> int:
        all_chunks: list[Chunk] = []
        
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            texts  = [c.content for c in chunks]
            embeddings = self.embedder.embed(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            all_chunks.extend(chunks)
        
        self.store.upsert(all_chunks)
        return len(all_chunks)
    
    # -- Query -----------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        metadata_filter: dict | None = None,
        use_hyde: bool = False,
    ) -> list[RetrievedChunk]:
        # Optional HyDE: generate a hypothetical document to improve retrieval
        embed_target = query
        if use_hyde and self.llm:
            hypothetical = self.llm(
                f"Write a short passage that would answer this question:\n{query}"
            )
            embed_target = hypothetical
        
        q_embedding = self.embedder.embed_one(embed_target)
        candidates  = self.store.query(q_embedding, top_k=self.retrieval_k,
                                       filter_metadata=metadata_filter)
        
        if self.reranker and candidates:
            return self.reranker.rerank(query, candidates, top_k=self.rerank_k)
        return candidates[:self.rerank_k]
    
    def answer(self, query: str, metadata_filter: dict | None = None) -> dict:
        chunks = self.retrieve(query, metadata_filter=metadata_filter)
        
        context = "\n\n---\n\n".join(
            f"[Source {i+1}] {c.chunk.content}"
            for i, c in enumerate(chunks)
        )
        
        prompt = f"""Answer the question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm(prompt) if self.llm else "[No LLM configured]"
        
        return {
            "answer": response,
            "sources": [
                {
                    "content":  c.chunk.content[:200] + "...",
                    "score":    c.score,
                    "rerank":   c.rerank_score,
                    "metadata": c.chunk.metadata,
                }
                for c in chunks
            ],
        }
```

### 4.3 Vector Database Selection

| Database | Hosting | Scale | Latency | Best For |
|----------|---------|-------|---------|----------|
| Pinecone | Managed cloud | High | Low | Production, zero-ops |
| Qdrant | Self-hosted / cloud | High | Very Low | Performance-critical |
| Weaviate | Both | High | Low | Hybrid (keyword + vector) |
| Chroma | Self-hosted | Medium | Low | Local dev / prototyping |
| pgvector | Self-hosted (Postgres) | Medium | Medium | Existing Postgres stacks |
| Redis VSS | Both | Medium | Very Low | Real-time / cache-adjacent |
| Milvus | Self-hosted / cloud | Very High | Low | Enterprise scale |

### 4.4 Chunking Strategy Selection

| Strategy | Chunk Size | Overlap | Best For |
|----------|------------|---------|----------|
| Fixed-size | 500-1000 tokens | 50-100 tokens | General text, unknown structure |
| Sentence | 3-5 sentences | 1 sentence | News articles, documentation |
| Semantic | Variable | Meaning-based | Research papers, books |
| Recursive | Hierarchical | Parent-child | Long documents with headers |

### 4.5 RAG Evaluation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Context Relevance | % of retrieved chunks relevant to query | > 0.80 |
| Answer Faithfulness | % of answer grounded in context | > 0.90 |
| Retrieval Precision@5 | Relevant chunks in top 5 / 5 | > 0.70 |
| Context Coverage | % of questions with >=1 relevant chunk in top-5 | > 0.85 |
| End-to-end Accuracy | Correct answers / total questions | > 0.80 |

```bash
# Evaluate a RAG pipeline
python scripts/rag_evaluator.py \
  --contexts retrieved_contexts.json \
  --questions eval_questions.json \
  --metrics relevance,faithfulness,coverage \
  --output report.json
```

---

## 5. Subagent Coordination

### 5.1 Subagent-Driven Development

**Core principle:** Fresh subagent per task + two-stage review (spec compliance, then code quality) = high quality, fast iteration.

This pattern runs entirely within the current session — no context switch to parallel sessions.

```
PROCESS FLOW:

1. Read plan -> extract all tasks with full text -> create TodoList

2. FOR EACH TASK:
   a. Dispatch Implementer subagent (full task text + context injected)
      +-> Subagent asks questions? -> Answer -> Re-dispatch
      +-> Subagent implements, tests, self-reviews, signals done
   
   b. Dispatch Spec Compliance Reviewer
      +-> Reviewer finds issues? -> Implementer fixes -> Re-review
      +-> OK Spec compliant -> proceed
   
   c. Dispatch Code Quality Reviewer (ONLY after spec review passes)
      +-> Reviewer finds issues? -> Implementer fixes -> Re-review
      +-> OK Quality approved -> mark task complete
   
3. After all tasks: Dispatch Final Code Reviewer for full implementation

4. Use finishing-a-development-branch workflow
```

### 5.2 Implementer Subagent Prompt Template

```markdown
# Implementer Subagent

## Context
You are implementing one task from a larger plan. You have been given full task text below.
Do NOT read plan files — the controller has already provided all necessary context.

## Project Context
{project_description}
Repository: {repo_path}
Branch: {branch_name}
Tech stack: {stack}

## Your Task
{full_task_text}

## Requirements
1. Ask questions BEFORE beginning if anything is unclear
2. Follow TDD: write failing test first, then implementation
3. Run all tests and verify they pass
4. Self-review: check for edge cases, naming, error handling
5. Commit with a descriptive message

## Output When Done
- Summary of what you implemented
- Test results (pass/fail counts)
- Any concerns or trade-offs you made
- Commit SHA
```

### 5.3 Spec Reviewer Prompt Template

```markdown
# Spec Compliance Reviewer

## Your Role
You are a spec compliance reviewer — NOT a code quality reviewer.
Your ONLY job: verify the implementation matches the spec. Nothing more.

## Task Spec
{task_spec}

## Implementation to Review
Git SHAs of new commits: {commit_shas}

## Review Criteria
Check for:
1. MISSING: Requirements in the spec not implemented
2. EXTRA: Features implemented that were NOT requested (scope creep)
3. WRONG: Implementation that contradicts the spec

## Output Format
STATUS: OK COMPLIANT or NOT COMPLIANT

If non-compliant, list each issue as:
- MISSING: [description]
- EXTRA: [description]  
- WRONG: [description]

Do NOT comment on code quality, style, or performance.
```

### 5.4 Code Quality Reviewer Prompt Template

```markdown
# Code Quality Reviewer

## Your Role
You are a code quality reviewer. The spec compliance reviewer has already confirmed
this implementation matches the spec — your job is code quality ONLY.

## Implementation to Review
Git SHAs: {commit_shas}

## Review Criteria
For each finding, classify as:
- CRITICAL: Must fix before merge (security, correctness, data loss)
- IMPORTANT: Should fix (maintainability, performance)
- SUGGESTION: Nice to have (style, naming)

## What to Check
- Error handling completeness
- Edge cases (null, empty, boundary values)
- Naming clarity
- Magic numbers/strings (extract to constants)
- DRY violations
- Security issues (injection, auth bypass, data exposure)
- Test coverage adequacy

## Output Format
STRENGTHS: [what's well done]
CRITICAL ISSUES: [list or "None"]
IMPORTANT ISSUES: [list or "None"]
VERDICT: OK APPROVED or CHANGES REQUIRED
```

### 5.5 Red Flags in Subagent Coordination

**Never do these:**
- Start code quality review before spec compliance passes — wrong order produces wasted cycles
- Dispatch multiple implementer subagents in parallel on the same codebase — merge conflicts guaranteed
- Let subagent read plan files — provide full task text in the prompt instead (eliminates file-reading overhead)
- Accept "close enough" spec compliance — reviewer found issues means the task is not done
- Skip the re-review after fixes — don't trust the fix without verification
- Skip scene-setting context in subagent prompts — subagent needs to understand where the task fits

---

## 6. Execution Planning & Verification

### 6.1 Parallel Agent Dispatch Pattern

Use when 2+ independent tasks can proceed without shared state or sequential dependencies.

**Decision tree:**
```
Multiple independent tasks?
+-- YES: Can they write to the same files/resources?
|   +-- YES -> Sequential agents (avoid conflict)
|   +-- NO  -> Parallel dispatch OK
+-- NO: Tasks are related -> Single agent investigates all
```

**Parallel dispatch template:**
```python
import asyncio
from typing import Callable, Any

async def dispatch_parallel_agents(
    tasks: list[dict],
    agent_fn: Callable[[dict], Any],
    max_concurrent: int = 5,
) -> list[dict]:
    """
    Dispatch multiple agent tasks in parallel with a concurrency limit.
    
    tasks: list of dicts, each with 'id', 'description', 'context', 'constraints'
    agent_fn: async function(task) -> result
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(task: dict) -> dict:
        async with semaphore:
            try:
                result = await agent_fn(task)
                return {"task_id": task["id"], "status": "success", "result": result}
            except Exception as e:
                return {"task_id": task["id"], "status": "failed", "error": str(e)}
    
    results = await asyncio.gather(*[run_with_semaphore(t) for t in tasks])
    return list(results)


def check_result_conflicts(results: list[dict]) -> list[str]:
    """
    Scan parallel agent results for potential conflicts before integration.
    Checks: same files modified, same database records mutated.
    """
    conflicts = []
    files_modified = {}
    
    for result in results:
        if result["status"] != "success":
            continue
        files = result.get("result", {}).get("files_modified", [])
        for f in files:
            if f in files_modified:
                conflicts.append(
                    f"Conflict: '{f}' modified by both task "
                    f"{files_modified[f]} and {result['task_id']}"
                )
            else:
                files_modified[f] = result["task_id"]
    
    return conflicts
```

### 6.2 Focused Agent Task Prompt Structure

Good parallel agent task prompts are self-contained, specific, and constrained:

```markdown
## Task: {task_title}

### Problem Statement
{specific_error_messages_or_failure_description}

### Scope
Files/subsystems in scope: {explicit_list}
Files/subsystems OUT of scope: {explicit_exclusions}

### Goal
{single_clear_success_criterion}

### Constraints
- Do NOT modify: {protected_files}
- Do NOT add new dependencies without flagging it
- {other_constraints}

### Required Output
Return:
1. Root cause analysis
2. What you changed and why
3. Verification output (test results, logs)
4. Summary of changes as a git diff or commit SHA
```

### 6.3 Batch Execution with Checkpoints

```
EXECUTING-PLANS PROCESS:

1. LOAD & REVIEW
   - Read plan file once
   - Identify concerns or blockers -> raise with human BEFORE starting
   - Create TodoList from all tasks
   - Announce: "Using executing-plans to implement this plan."

2. EXECUTE BATCH (default: 3 tasks per batch)
   For each task in batch:
   - Mark in_progress
   - Follow steps exactly as written
   - Run all verifications specified in plan
   - Mark completed

3. CHECKPOINT REPORT
   - Show: what was implemented, verification output
   - Say: "Ready for feedback."
   - Wait for approval before next batch

4. REPEAT until all tasks complete

5. FINISH
   - Use finishing-a-development-branch workflow
   - Verify all tests pass, no regressions

STOP IMMEDIATELY if:
- A blocker appears mid-batch
- Verification fails repeatedly
- Instructions are ambiguous
-> Ask for clarification; never guess.
```

### 6.4 Verification Workflow

Every implementation task should define explicit verification steps:

```python
VERIFICATION_LEVELS = {
    "smoke": [
        "Application starts without errors",
        "All previously-passing tests still pass",
        "No new lint errors",
    ],
    "functional": [
        "New unit tests written and passing",
        "Integration tests pass against staging",
        "Edge cases tested (null, empty, boundary)",
    ],
    "acceptance": [
        "Feature works end-to-end as specified",
        "Performance within targets (latency, throughput)",
        "Security review passed",
        "Documentation updated",
    ],
}

def build_verification_prompt(task: str, level: str = "functional") -> str:
    checks = "\n".join(f"- [ ] {c}" for c in VERIFICATION_LEVELS[level])
    return f"""
After implementing: {task}

Run these verification checks:
{checks}

Report each check as OK PASS, FAIL, or SKIPPED (with reason).
If any check FAILS, stop and report before proceeding.
"""
```

---

## 7. Prompt Engineering & Optimization

### 7.1 Core Prompt Patterns

| Pattern | When to Use | Token Cost | Quality Gain |
|---------|-------------|------------|-------------|
| Zero-shot | Simple, well-defined tasks | Lowest | Baseline |
| Few-shot (3-5 examples) | Complex tasks, consistent format needed | Medium | High |
| Chain-of-Thought (CoT) | Reasoning, math, multi-step logic | Medium | High |
| Role Prompting | Domain expertise, specific perspective | Low | Medium |
| Structured Output | Need parseable JSON/XML | Low | High (reliability) |
| Tree-of-Thought | Complex problem solving, backtracking | High | Very High |
| Meta-prompting | Generating/optimizing other prompts | High | Very High |
| Self-consistency | High-stakes decisions (majority vote) | Very High | High |

### 7.2 Chain-of-Thought Implementation

```python
COT_TEMPLATES = {
    # Standard CoT
    "standard": """
{task}

Think step by step:
1. First, identify what information is given
2. Determine what needs to be found
3. Work through the reasoning systematically
4. State your conclusion

Reasoning:
""",

    # Few-shot CoT
    "few_shot": """
Solve problems by thinking step by step.

Example 1:
Problem: {example_problem_1}
Reasoning: {example_reasoning_1}
Answer: {example_answer_1}

Example 2:
Problem: {example_problem_2}
Reasoning: {example_reasoning_2}
Answer: {example_answer_2}

Now solve:
Problem: {problem}
Reasoning:
""",

    # Zero-shot CoT (Kojima et al.)
    "zero_shot": "{task}\n\nLet's think step by step.",

    # Plan-then-execute CoT
    "plan_execute": """
{task}

Step 1 - Make a plan: List the sub-problems you need to solve, in order.
Step 2 - Execute: Work through each sub-problem, showing your reasoning.
Step 3 - Verify: Check your answer against the original question.

Begin:
""",
}
```

### 7.3 Structured Output Design

```python
from typing import Literal
from pydantic import BaseModel, Field

# --- Define Output Schema -----------------------------------------------------
class SentimentAnalysis(BaseModel):
    summary:    str     = Field(..., max_length=200, description="Brief content summary")
    sentiment:  Literal["positive", "negative", "neutral", "mixed"]
    confidence: float   = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    key_points: list[str] = Field(..., max_items=5, description="Up to 5 key points")

# --- Build Prompt with Schema -------------------------------------------------
def build_structured_prompt(content: str, schema: type[BaseModel]) -> str:
    schema_json = schema.model_json_schema()
    
    return f"""Analyze the following content.

Respond ONLY with valid JSON matching this schema:
{json.dumps(schema_json, indent=2)}

IMPORTANT:
- Start your response with {{
- End your response with }}
- No markdown code fences, no explanation outside the JSON

Content to analyze:
{content}

JSON response:"""

# --- Parse and Validate Output ------------------------------------------------
def parse_structured_output(response: str, schema: type[BaseModel]) -> BaseModel:
    # Strip markdown fences if present
    import re
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
    
    # Find outermost JSON object
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{response}")
    
    data = json.loads(cleaned[start:end])
    return schema.model_validate(data)
```

### 7.4 Prompt Optimization Workflow

```
STEP 1: Baseline
  python scripts/prompt_optimizer.py current_prompt.txt --analyze --output baseline.json
  Capture: token count, clarity score, issues found

STEP 2: Identify Problems
  | Issue              | Apply This Pattern              |
  |--------------------|----------------------------------|
  | Ambiguous output   | Add explicit format specification |
  | Too verbose        | Extract to few-shot examples      |
  | Inconsistent results| Add role/persona framing         |
  | Missing edge cases | Add constraint boundaries         |
  | Poor reasoning     | Add chain-of-thought trigger      |
  | Wrong format       | Add schema + format enforcement   |

STEP 3: Apply Optimizations
  python scripts/prompt_optimizer.py current_prompt.txt --optimize --output optimized.txt

STEP 4: Compare
  python scripts/prompt_optimizer.py optimized.txt --analyze --compare baseline.json

STEP 5: A/B Test
  Run both prompts against held-out evaluation set.
  Accept optimization only if: quality up AND cost <= 1.2x baseline.
```

### 7.5 Meta-Prompting (Prompt Generation)

```python
META_PROMPT_GENERATOR = """
You are an expert prompt engineer. Generate an optimized prompt for the following use case.

## Use Case
Task: {task_description}
Model: {model}
Expected input format: {input_format}
Expected output format: {output_format}
Edge cases to handle: {edge_cases}
Constraints: {constraints}

## Generate a prompt that:
1. Uses role framing appropriate for the task
2. Provides clear, unambiguous instructions
3. Includes 2-3 few-shot examples if appropriate
4. Specifies exact output format
5. Handles the listed edge cases
6. Is token-efficient (no redundancy)

Return the complete prompt, ready to use.
"""

def generate_prompt(
    task_description: str,
    model: str = "claude-opus-4-5",
    output_format: str = "JSON",
    edge_cases: str = "empty input, ambiguous cases",
    constraints: str = "respond in English only",
    input_format: str = "plain text",
) -> str:
    """Use an LLM to generate an optimized prompt for a given task."""
    import anthropic
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": META_PROMPT_GENERATOR.format(
                task_description=task_description,
                model=model,
                input_format=input_format,
                output_format=output_format,
                edge_cases=edge_cases,
                constraints=constraints,
            )
        }]
    )
    return response.content[0].text
```

### 7.6 Few-Shot Example Design

```
EXAMPLE DESIGN CHECKLIST:

[ ] 3-5 examples (more = diminishing returns + token cost)
[ ] Covers: simple case, edge case, complex case, negative case
[ ] Consistent format across all examples
[ ] Output format matches expected production output exactly
[ ] Examples do NOT appear in test set (data contamination)
[ ] Ordered: simple -> complex (progressive difficulty)

EXAMPLE TEMPLATE:

Input: {diverse_input}
Output: {correctly_formatted_output}

[Repeat for each example with blank line between]

Now apply to:
Input: {actual_input}
Output:
```

---

## 8. ML Integration for Agents

### 8.1 LLM Provider Abstraction Layer

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float

# Cost per 1K tokens (update as pricing changes)
PRICING = {
    "claude-opus-4-5":     {"input": 0.015,    "output": 0.075},
    "claude-haiku-3":      {"input": 0.00025,  "output": 0.00125},
    "gpt-4o":              {"input": 0.005,    "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015,  "output": 0.0006},
}

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        pass
    
    def estimate_cost(self, prompt: str, output_tokens_estimate: int = 500) -> float:
        model = getattr(self, "model", "unknown")
        pricing = PRICING.get(model, {"input": 0.01, "output": 0.03})
        input_tokens = len(prompt.split()) * 1.3  # rough approximation
        return (input_tokens / 1000 * pricing["input"] +
                output_tokens_estimate / 1000 * pricing["output"])


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-opus-4-5"):
        self.client = anthropic.Anthropic()
        self.model  = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, max_tokens: int = 2048, **kwargs) -> LLMResponse:
        start = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency = (time.time() - start) * 1000
        pricing = PRICING.get(self.model, {"input": 0.015, "output": 0.075})
        cost = (response.usage.input_tokens  / 1000 * pricing["input"] +
                response.usage.output_tokens / 1000 * pricing["output"])
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            latency_ms=latency,
        )


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI()
        self.model  = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, max_tokens: int = 2048, **kwargs) -> LLMResponse:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency = (time.time() - start) * 1000
        usage   = response.usage
        pricing = PRICING.get(self.model, {"input": 0.005, "output": 0.015})
        cost = (usage.prompt_tokens     / 1000 * pricing["input"] +
                usage.completion_tokens / 1000 * pricing["output"])
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
            latency_ms=latency,
        )


class FallbackProvider(LLMProvider):
    """Try primary provider; fall back to secondary on failure."""
    
    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self.primary  = primary
        self.fallback = fallback
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            return self.primary.complete(prompt, **kwargs)
        except Exception as e:
            print(f"Primary LLM failed ({e}), falling back...")
            return self.fallback.complete(prompt, **kwargs)
```

### 8.2 Model Deployment for Agent Serving

```dockerfile
# Dockerfile for agent service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY config/ ./config/

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
CMD ["uvicorn", "src.agent_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

```python
# src/agent_server.py — FastAPI agent serving wrapper
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import uuid

app = FastAPI(title="Agent Service", version="1.0.0")

class AgentRequest(BaseModel):
    query:        str
    session_id:   str | None = None
    max_steps:    int        = 10
    metadata:     dict       = {}

class AgentResponse(BaseModel):
    result:     str
    session_id: str
    steps_used: int
    cost_usd:   float
    latency_ms: float

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    session_id = request.session_id or str(uuid.uuid4())
    # ... agent execution logic ...
    return AgentResponse(
        result="...",
        session_id=session_id,
        steps_used=0,
        cost_usd=0.0,
        latency_ms=0.0,
    )

@app.post("/agent/run-async")
async def run_agent_async(request: AgentRequest, background: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background.add_task(_run_agent_task, job_id, request)
    return {"job_id": job_id, "status": "queued"}

@app.get("/agent/status/{job_id}")
async def get_status(job_id: str):
    # Fetch from job store (Redis, DB, etc.)
    return {"job_id": job_id, "status": "unknown"}

async def _run_agent_task(job_id: str, request: AgentRequest):
    # Run agent and persist result
    pass
```

### 8.3 Model Monitoring for Agents

```python
# Drift detection for agent input distributions
from scipy.stats import ks_2samp
import numpy as np

def detect_input_drift(
    reference_inputs: list[str],
    current_inputs: list[str],
    threshold_p: float = 0.05,
) -> dict:
    """
    Detect distribution shift in agent input queries using
    token length distribution as a proxy metric.
    """
    ref_lengths = np.array([len(t.split()) for t in reference_inputs])
    cur_lengths = np.array([len(t.split()) for t in current_inputs])
    
    stat, p_value = ks_2samp(ref_lengths, cur_lengths)
    
    return {
        "drift_detected": p_value < threshold_p,
        "ks_statistic":   float(stat),
        "p_value":        float(p_value),
        "ref_mean_tokens": float(ref_lengths.mean()),
        "cur_mean_tokens": float(cur_lengths.mean()),
        "recommendation": (
            "Retrain or re-evaluate agent prompts — input distribution shifted significantly."
            if p_value < threshold_p else
            "No significant drift detected."
        ),
    }

# Alert thresholds for agent monitoring
AGENT_ALERT_THRESHOLDS = {
    "p95_latency_ms":     {"warning": 2000,  "critical": 5000},
    "error_rate_pct":     {"warning": 1.0,   "critical": 5.0},
    "cost_per_query_usd": {"warning": 0.05,  "critical": 0.20},
    "tool_failure_rate":  {"warning": 0.05,  "critical": 0.15},
    "token_overflow_rate":{"warning": 0.02,  "critical": 0.10},
}
```

### 8.4 Serving Strategy Selection

| Strategy | Latency | Throughput | Cost | Use Case |
|---------|---------|-----------|------|----------|
| FastAPI + Uvicorn | Low | Medium | Low | REST agent APIs, single-model |
| Ray Serve | Medium | Very High | Medium | Multi-model pipelines, scaling |
| Triton Inference | Very Low | Very High | Medium | GPU batch inference |
| Serverless (Lambda/Cloud Run) | Cold-start medium | Auto-scale | Pay-per-use | Bursty agent tasks |
| Streaming (SSE/WebSocket) | Apparent Low | Medium | Low | Conversational agents |

---

## 9. Skill & Capability Creation

### 9.1 SKILL.md Format Specification

Every Perplexity Computer skill must follow this exact format:

```markdown
---
name: skill-name-with-hyphens
description: One or two sentences describing when to use this skill. Start with "Use when..." or describe the trigger conditions clearly.
license: MIT
metadata:
  author: your-username
  version: '1.0'
---

# Skill Title

Brief one-paragraph overview of the skill's purpose.

## When to Use
...

## Core Concepts
...

## Step-by-Step Process
...

## Examples
...

## Common Mistakes
...
```

### 9.2 Validation Pipeline

```bash
# Validate skill structure and frontmatter
cd /home/user/workspace && uvx --from skills-ref agentskills validate <skill-name>

# What the validator checks:
# OK First line is exactly ---
# OK YAML frontmatter present and parseable
# OK Required fields: name, description, license, metadata.author, metadata.version
# OK name matches directory name
# OK version is quoted string ('1.0' not 1.0)
# OK Skill directory exists at workspace/<skill-name>/SKILL.md
# OK No syntax errors in YAML block
```

### 9.3 Skill Quality Checklist

Before publishing any skill:

- [ ] First line of SKILL.md is exactly `---` (three dashes, no spaces)
- [ ] All required YAML fields present (`name`, `description`, `license`, `metadata.author`, `metadata.version`)
- [ ] `version` is quoted: `'1.0'` not `1.0`
- [ ] Description tells the agent WHEN to load the skill (trigger conditions)
- [ ] `## When to Use` section with clear positive AND negative cases
- [ ] At least one working code example
- [ ] Common mistakes / red flags section
- [ ] Validation passes: `uvx --from skills-ref agentskills validate <name>`

### 9.4 Skill Packaging for Distribution

```
skill-directory/
+-- SKILL.md          <- Main skill file (required)
+-- README.md         <- Human-readable docs (optional)
+-- examples/
|   +-- basic.md      <- Annotated simple example
|   +-- advanced.md   <- Complex workflow example
+-- templates/
|   +-- prompt_template.md
|   +-- config.yaml
+-- scripts/
    +-- validate.py   <- Skill-specific validation helpers
```

### 9.5 Skill Trigger Design

The description field determines when the skill is loaded. Write it to trigger on the right signals:

```yaml
# Bad — too vague, triggers on everything
description: Helps build things with AI.

# Bad — too narrow, misses many triggers
description: Use when the user types "build an MCP server".

# Good — triggers on intent, not exact phrasing
description: >
  Use when building AI agents, creating MCP servers, designing RAG systems,
  coordinating subagents, optimizing prompts, or architecting any AI-powered
  automation workflow. Covers agent design patterns, multi-agent orchestration,
  and production deployment.
```

---

## 10. Backend Infrastructure for Agents

### 10.1 Agent Memory Persistence with SQLite (CGI-bin)

Agents need persistent memory across sessions. The CGI-bin pattern lets agents store and retrieve state via HTTP endpoints without a dedicated backend server.

```python
#!/usr/bin/env python3
# cgi-bin/agent_memory.py
# Agent memory store: conversations, tool results, learned facts

import json
import os
import sqlite3
import sys
from datetime import datetime

DB_PATH = "agent_memory.db"

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata    TEXT DEFAULT '{}'
        );
        
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('user','assistant','tool','system')),
            content     TEXT NOT NULL,
            tool_name   TEXT,
            tool_input  TEXT,
            tool_result TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        
        CREATE TABLE IF NOT EXISTS facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            key         TEXT NOT NULL,
            value       TEXT NOT NULL,
            confidence  REAL DEFAULT 1.0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at  TIMESTAMP,
            UNIQUE(session_id, key)
        );
        
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
    """)
    conn.commit()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
init_db(conn)

method      = os.environ.get("REQUEST_METHOD", "GET")
query       = os.environ.get("QUERY_STRING", "")
path_info   = os.environ.get("PATH_INFO", "")

def respond(data, status=200):
    print(f"Status: {status}")
    print("Content-Type: application/json")
    print()
    print(json.dumps(data))

def parse_qs(qs):
    params = {}
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    return params

# -- Routes --------------------------------------------------------------------
if path_info == "/sessions" and method == "POST":
    body = json.loads(sys.stdin.read() or "{}")
    sid  = body.get("session_id") or f"sess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    conn.execute("INSERT OR IGNORE INTO sessions (session_id, metadata) VALUES (?,?)",
                 [sid, json.dumps(body.get("metadata", {}))])
    conn.commit()
    respond({"session_id": sid}, 201)

elif path_info == "/messages" and method == "POST":
    body = json.loads(sys.stdin.read())
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_name, tool_input, tool_result) "
        "VALUES (?,?,?,?,?,?)",
        [body["session_id"], body["role"], body["content"],
         body.get("tool_name"), body.get("tool_input"), body.get("tool_result")]
    )
    conn.commit()
    msg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    respond({"id": msg_id}, 201)

elif path_info == "/messages" and method == "GET":
    params = parse_qs(query)
    sid    = params.get("session_id", "")
    limit  = int(params.get("limit", 50))
    rows   = conn.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY created_at LIMIT ?",
        [sid, limit]
    ).fetchall()
    respond([dict(r) for r in rows])

elif path_info == "/facts" and method == "PUT":
    body = json.loads(sys.stdin.read())
    conn.execute(
        "INSERT OR REPLACE INTO facts (session_id, key, value, confidence) VALUES (?,?,?,?)",
        [body.get("session_id"), body["key"], json.dumps(body["value"]),
         body.get("confidence", 1.0)]
    )
    conn.commit()
    respond({"status": "ok"})

elif path_info == "/facts" and method == "GET":
    params = parse_qs(query)
    sid    = params.get("session_id", "")
    rows   = conn.execute(
        "SELECT key, value, confidence FROM facts WHERE session_id=? OR session_id IS NULL",
        [sid]
    ).fetchall()
    respond({r["key"]: {"value": json.loads(r["value"]), "confidence": r["confidence"]}
             for r in rows})

else:
    respond({"error": f"Unknown route: {method} {path_info}"}, 400)
```

### 10.2 Webhook Receiver for Agent Triggers

```python
#!/usr/bin/env python3
# cgi-bin/webhook_receiver.py
# Receives external events and queues them for agent processing

import hashlib
import hmac
import json
import os
import sqlite3
import sys
import time

DB_PATH     = "webhook_events.db"
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        source      TEXT NOT NULL,
        event_type  TEXT NOT NULL,
        payload     TEXT NOT NULL,
        processed   INTEGER DEFAULT 0,
        received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

method = os.environ.get("REQUEST_METHOD", "GET")

def verify_signature(body: str, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 webhook signature."""
    if not secret:
        return True  # Skip verification if no secret configured
    expected = "sha256=" + hmac.new(
        secret.encode(), body.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

if method == "POST":
    raw_body = sys.stdin.read()
    sig      = os.environ.get("HTTP_X_HUB_SIGNATURE_256", "")
    
    if not verify_signature(raw_body, sig, WEBHOOK_SECRET):
        print("Status: 401")
        print("Content-Type: application/json")
        print()
        print('{"error": "Invalid signature"}')
        sys.exit(0)
    
    body = json.loads(raw_body)
    conn.execute(
        "INSERT INTO events (source, event_type, payload) VALUES (?,?,?)",
        [body.get("source", "unknown"), body.get("type", "unknown"), raw_body]
    )
    conn.commit()
    
    print("Status: 202")
    print("Content-Type: application/json")
    print()
    print('{"status": "accepted"}')

elif method == "GET":
    # Dequeue unprocessed events for agent polling
    rows = conn.execute(
        "SELECT * FROM events WHERE processed=0 ORDER BY received_at LIMIT 50"
    ).fetchall()
    
    events = [
        {"id": r[0], "source": r[1], "event_type": r[2],
         "payload": json.loads(r[3]), "received_at": r[5]}
        for r in rows
    ]
    print("Content-Type: application/json")
    print()
    print(json.dumps(events))
```

### 10.3 Agent-to-Agent Communication via Message Bus

```python
#!/usr/bin/env python3
# cgi-bin/message_bus.py
# Simple pub/sub message bus for multi-agent coordination

import json
import os
import sqlite3
import sys
import uuid

DB_PATH = "message_bus.db"

conn = sqlite3.connect(DB_PATH)
conn.executescript("""
    CREATE TABLE IF NOT EXISTS messages (
        id          TEXT PRIMARY KEY,
        from_agent  TEXT NOT NULL,
        to_agent    TEXT,           -- NULL = broadcast
        topic       TEXT NOT NULL,
        payload     TEXT NOT NULL,
        ack         INTEGER DEFAULT 0,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS subscriptions (
        agent_id    TEXT NOT NULL,
        topic       TEXT NOT NULL,
        PRIMARY KEY (agent_id, topic)
    );
""")
conn.commit()

method    = os.environ.get("REQUEST_METHOD", "GET")
path_info = os.environ.get("PATH_INFO", "")
query     = os.environ.get("QUERY_STRING", "")

def respond(data, status=200):
    print(f"Status: {status}")
    print("Content-Type: application/json")
    print()
    print(json.dumps(data))

if path_info == "/publish" and method == "POST":
    body = json.loads(sys.stdin.read())
    msg_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO messages (id, from_agent, to_agent, topic, payload) VALUES (?,?,?,?,?)",
        [msg_id, body["from_agent"], body.get("to_agent"),
         body["topic"], json.dumps(body["payload"])]
    )
    conn.commit()
    respond({"message_id": msg_id}, 201)

elif path_info == "/subscribe" and method == "POST":
    body = json.loads(sys.stdin.read())
    conn.execute(
        "INSERT OR IGNORE INTO subscriptions VALUES (?,?)",
        [body["agent_id"], body["topic"]]
    )
    conn.commit()
    respond({"status": "subscribed"})

elif path_info == "/poll" and method == "GET":
    params = dict(p.split("=") for p in query.split("&") if "=" in p)
    agent_id = params.get("agent_id", "")
    
    # Get messages for this agent (direct + subscribed topics)
    rows = conn.execute("""
        SELECT m.* FROM messages m
        LEFT JOIN subscriptions s ON s.agent_id=? AND s.topic=m.topic
        WHERE m.ack=0 AND (m.to_agent=? OR (m.to_agent IS NULL AND s.agent_id IS NOT NULL))
        ORDER BY m.created_at LIMIT 20
    """, [agent_id, agent_id]).fetchall()
    
    messages = [{"id": r[0], "from": r[1], "topic": r[3],
                 "payload": json.loads(r[4])} for r in rows]
    
    # Mark as acked
    if messages:
        ids = [m["id"] for m in messages]
        conn.execute(f"UPDATE messages SET ack=1 WHERE id IN ({','.join('?'*len(ids))})", ids)
        conn.commit()
    
    respond(messages)
```

### 10.4 JavaScript Client for Agent Backend

```javascript
// Agent backend client — use __CGI_BIN__ placeholder in production
const CGI_BIN = "__CGI_BIN__";

class AgentMemoryClient {
  async createSession(metadata = {}) {
    const res = await fetch(`${CGI_BIN}/agent_memory.py/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata }),
    });
    return res.json();
  }

  async addMessage(sessionId, role, content, toolData = {}) {
    const res = await fetch(`${CGI_BIN}/agent_memory.py/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, role, content, ...toolData }),
    });
    return res.json();
  }

  async getHistory(sessionId, limit = 50) {
    const res = await fetch(
      `${CGI_BIN}/agent_memory.py/messages?session_id=${sessionId}&limit=${limit}`
    );
    return res.json();
  }

  async setFact(sessionId, key, value, confidence = 1.0) {
    const res = await fetch(`${CGI_BIN}/agent_memory.py/facts`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, key, value, confidence }),
    });
    return res.json();
  }

  async getFacts(sessionId) {
    const res = await fetch(
      `${CGI_BIN}/agent_memory.py/facts?session_id=${sessionId}`
    );
    return res.json();
  }
}

class MessageBusClient {
  async publish(fromAgent, topic, payload, toAgent = null) {
    const res = await fetch(`${CGI_BIN}/message_bus.py/publish`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_agent: fromAgent, topic, payload, to_agent: toAgent }),
    });
    return res.json();
  }

  async subscribe(agentId, topic) {
    const res = await fetch(`${CGI_BIN}/message_bus.py/subscribe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: agentId, topic }),
    });
    return res.json();
  }

  async poll(agentId) {
    const res = await fetch(`${CGI_BIN}/message_bus.py/poll?agent_id=${agentId}`);
    return res.json();
  }
}
```

---

## 11. Agent Deployment & Monitoring

### 11.1 Deployment Checklist

Before deploying any agent to production:

**Infrastructure**
- [ ] Health check endpoint responds: `GET /health` -> `{"status": "ok"}`
- [ ] Graceful shutdown handles in-flight requests
- [ ] Rate limiting configured (protect upstream APIs)
- [ ] Request timeouts set (prevent runaway agents)
- [ ] Retry logic with exponential backoff on transient failures
- [ ] Circuit breaker prevents cascade failures

**Observability**
- [ ] Structured logging (JSON) with correlation IDs
- [ ] Trace propagation across agent calls
- [ ] Metrics exported: request count, latency p50/p95/p99, error rate, token usage, cost
- [ ] Alerts configured for critical thresholds

**Security**
- [ ] API keys stored in environment variables, never in code
- [ ] Input sanitization before passing to LLM (prompt injection defense)
- [ ] Output filtering for PII / sensitive data
- [ ] Rate limits per user/tenant
- [ ] Audit log of all tool executions

**Cost Controls**
- [ ] Per-request token budget enforced
- [ ] Daily/monthly spend limits with alerting
- [ ] Model fallback chain configured (expensive -> cheap)

### 11.2 Structured Agent Logging

```python
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass

@dataclass
class AgentTraceEvent:
    trace_id:    str
    span_id:     str
    event_type:  str   # "llm_call" | "tool_call" | "agent_start" | "agent_end"
    timestamp:   float
    agent_id:    str
    model:       str | None = None
    tool_name:   str | None = None
    input_tokens:  int = 0
    output_tokens: int = 0
    cost_usd:    float = 0.0
    latency_ms:  float = 0.0
    error:       str | None = None
    metadata:    dict | None = None

class AgentTracer:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger   = logging.getLogger("agent.trace")
    
    def log(self, event: AgentTraceEvent):
        self.logger.info(json.dumps(asdict(event)))
    
    @contextmanager
    def trace_llm_call(self, trace_id: str, model: str):
        span_id = str(uuid.uuid4())[:8]
        start   = time.time()
        event   = AgentTraceEvent(
            trace_id=trace_id, span_id=span_id,
            event_type="llm_call", timestamp=start,
            agent_id=self.agent_id, model=model,
        )
        try:
            yield event
        except Exception as e:
            event.error = str(e)
            raise
        finally:
            event.latency_ms = (time.time() - start) * 1000
            self.log(event)
    
    @contextmanager
    def trace_tool_call(self, trace_id: str, tool_name: str):
        span_id = str(uuid.uuid4())[:8]
        start   = time.time()
        event   = AgentTraceEvent(
            trace_id=trace_id, span_id=span_id,
            event_type="tool_call", timestamp=start,
            agent_id=self.agent_id, tool_name=tool_name,
        )
        try:
            yield event
        except Exception as e:
            event.error = str(e)
            raise
        finally:
            event.latency_ms = (time.time() - start) * 1000
            self.log(event)
```

### 11.3 Agent Health Dashboard Schema

```python
# Metrics to expose at /metrics (Prometheus-compatible)
AGENT_METRICS = """
# HELP agent_requests_total Total requests processed
# TYPE agent_requests_total counter
agent_requests_total{agent_id="{agent_id}",status="success"} {success_count}
agent_requests_total{agent_id="{agent_id}",status="error"} {error_count}

# HELP agent_latency_seconds Request latency
# TYPE agent_latency_seconds histogram
agent_latency_seconds_bucket{{agent_id="{agent_id}",le="0.5"}} {bucket_0_5}
agent_latency_seconds_bucket{{agent_id="{agent_id}",le="1.0"}} {bucket_1_0}
agent_latency_seconds_bucket{{agent_id="{agent_id}",le="5.0"}} {bucket_5_0}
agent_latency_seconds_bucket{{agent_id="{agent_id}",le="+Inf"}} {bucket_inf}

# HELP agent_tokens_total Total tokens consumed
# TYPE agent_tokens_total counter
agent_tokens_total{{agent_id="{agent_id}",type="input"}} {input_tokens}
agent_tokens_total{{agent_id="{agent_id}",type="output"}} {output_tokens}

# HELP agent_cost_usd_total Total cost in USD
# TYPE agent_cost_usd_total counter
agent_cost_usd_total{{agent_id="{agent_id}"}} {total_cost}

# HELP agent_tool_calls_total Tool calls by name
# TYPE agent_tool_calls_total counter
{tool_call_metrics}
"""
```

### 11.4 Scaling Strategies

| Scale Level | Approach | Infrastructure |
|------------|---------|---------------|
| Single user | Single process, local SQLite | Dev machine or single VM |
| Small team (< 50 users) | Multi-worker Uvicorn, shared PostgreSQL | Single server, 4-8 CPU |
| Medium (50-500 users) | Horizontal pod autoscaling, Redis cache | Kubernetes, load balancer |
| Large (500+ users) | Async task queue (Celery/Arq), vector DB cluster | Multi-region, CDN |
| Enterprise | Dedicated LLM endpoints, tenant isolation, SOC2 | Managed cloud |

---

## 12. Unique Perplexity Computer Capabilities

### 12.1 400+ Service Integrations

Perplexity Computer exposes connections to 400+ external services via the `list_external_tools` / `call_external_tool` pattern. This makes it uniquely suited for building **integration-heavy agents** without writing custom connectors.

**Discovery pattern:**
```
1. list_external_tools(queries=["github", "repo"])
   -> Returns connected tools with source_id and status

2. describe_external_tools(source_id="github", tool_names=["create_issue"])
   -> Returns full JSON schema for tool inputs

3. call_external_tool(tool_name="create_issue", source_id="github", arguments={...})
   -> Executes against live service
```

**Key integration categories available:**
| Category | Example Services |
|---------|-----------------|
| Communication | Gmail, Slack, Teams, Discord, Outlook |
| Project management | GitHub, Jira, Linear, Asana, Notion |
| CRM/Sales | Salesforce, HubSpot, Pipedrive |
| Data/Analytics | Google Sheets, Airtable, BigQuery |
| Storage | Google Drive, Dropbox, S3 |
| Calendar | Google Calendar, Outlook Calendar |
| Social | Twitter/X, LinkedIn |
| Payments | Stripe |
| Databases | PostgreSQL, MySQL, MongoDB |

### 12.2 Building Integration-Driven Agents

An agent in Perplexity Computer that uses external integrations:

```python
# Pattern: Research -> Enrich -> Store -> Notify
# Example: Competitor monitoring agent

COMPETITOR_AGENT_FLOW = """
1. search_web(queries=["[competitor] product update", "[competitor] pricing 2026"])
   -> Gather news about target competitor

2. search_social(query="from:[competitor_twitter] -is:retweet", only_recent=True)
   -> Collect recent social posts

3. call_external_tool(source_id="notion_mcp", tool_name="create_page", arguments={
       "title": f"Competitor Intel: {date}",
       "content": compiled_findings
   })
   -> Store findings in Notion

4. call_external_tool(source_id="gmail", tool_name="send_email", arguments={
       "to": "team@company.com",
       "subject": f"Weekly Competitor Brief: {date}",
       "body": executive_summary
   })
   -> Notify stakeholders
"""
```

### 12.3 Scheduled Monitoring Agents

Perplexity Computer supports recurring tasks that trigger agents on a schedule. Use this for:

- Price monitoring: scrape competitor pricing daily, alert on changes
- Brand mention tracking: search social + news, weekly digest
- Performance monitoring: check site metrics, alert on regressions
- Data drift detection: compare model input distributions, trigger retraining

**Scheduled agent pattern:**
```python
MONITORING_AGENT_PROMPT = """
You are a scheduled monitoring agent running at: {timestamp}
Your monitoring target: {target_description}

## Steps
1. Collect current data from: {data_sources}
2. Compare to baseline stored at: {baseline_reference}
3. Calculate delta metrics
4. If any metric exceeds threshold: {alert_thresholds}
   -> Call alert tool with details
5. Update baseline with today's snapshot
6. Output summary report

## Output
Provide a structured report:
- Status: NORMAL | WARNING | CRITICAL
- Changes detected: [list]
- Metrics compared to baseline: [table]
- Actions taken: [list]
- Next check: {next_check_time}
"""
```

### 12.4 Research-Backed Agent Responses

Perplexity Computer's `search_web`, `search_vertical`, and `fetch_url` tools give agents access to current, real-world information that LLMs alone cannot provide.

**Research agent pattern:**
```python
RESEARCH_AGENT_STEPS = [
    # Step 1: Broad search
    "search_web(queries=[query, related_query_1, related_query_2])",
    
    # Step 2: Deep dive on key sources
    "fetch_url(url=top_result_url, prompt='Extract key facts and data')",
    
    # Step 3: Academic grounding (for technical claims)
    "search_vertical(vertical='academic', query=technical_query)",
    
    # Step 4: Visual evidence
    "search_vertical(vertical='image', query=visual_query)",
    
    # Step 5: Synthesize with citations
    "write synthesis with inline markdown citations linking to sources",
]
```

### 12.5 Live Deployment for Agent Interfaces

Agents built on Perplexity Computer can be deployed as live, publicly accessible web applications using the `deploy_website` workflow.

```
Agent Interface Deployment Stack:

Frontend (HTML/JS) --> deploy_website(project_path)
     |
     | fetch(__CGI_BIN__/agent_memory.py/...)
     |
Backend (CGI-bin Python) --> auto-deployed with frontend
     |
     | sqlite3 / flat files
     |
Persistent Storage --> lives in project directory

Result: A live URL with full agent interface accessible from anywhere.
No separate server provisioning needed.
```

**Full-stack agent UI deployment checklist:**
- [ ] `index.html` contains the agent chat interface
- [ ] `cgi-bin/agent_memory.py` handles session/message persistence
- [ ] `cgi-bin/agent_api.py` proxies LLM calls (keeps API key server-side)
- [ ] All CGI scripts are marked executable: `chmod +x cgi-bin/*.py`
- [ ] Client JavaScript uses `__CGI_BIN__` as base URL (replaced at deploy time)
- [ ] CORS is handled by the CGI proxy layer (no browser CORS issues)
- [ ] Deployed with `deploy_website` tool

---

## Appendix A: Quick Reference — When to Use Which Pattern

| Situation | Use This |
|-----------|----------|
| Need to build an agent from scratch | §2.2 Architecture Patterns — pick ReAct, Plan-Execute, or Reflexion |
| Need to integrate an external API as an agent tool | §3 MCP Server Development |
| Need the agent to answer from private documents | §4 RAG System Construction |
| Have multiple tasks that can run independently | §5.5 Parallel Agent Dispatch (dispatching-parallel-agents) |
| Have sequential tasks with quality gates | §5.1 Subagent-Driven Development |
| Have a written plan to implement | §6.3 Batch Execution with Checkpoints (executing-plans) |
| Prompt producing inconsistent or low-quality output | §7 Prompt Engineering & Optimization |
| Need to swap LLM providers or add cost controls | §8.1 LLM Provider Abstraction Layer |
| Need to package a workflow as a reusable skill | §9.1 SKILL.md Format Specification |
| Need persistent agent state across sessions | §10.1 Agent Memory Persistence (SQLite/CGI-bin) |
| Need agents to communicate with each other | §10.3 Agent Message Bus |
| Deploying an agent to production | §11 Agent Deployment & Monitoring |
| Using Perplexity Computer for agent automation | §12 Unique Perplexity Computer Capabilities |

---

## Appendix B: Architecture Decision Records (ADR)

### ADR-001: ReAct vs. Plan-Execute
**Context:** Choosing architecture for a new agent.
**Decision:** Use ReAct for open-ended tasks; Plan-Execute for structured workflows.
**Rationale:** ReAct handles unknowns gracefully. Plan-Execute gives auditability and checkpointing needed for long-running structured tasks.

### ADR-002: Vector DB Selection for RAG
**Context:** Choosing vector database for production RAG.
**Decision:** Pinecone for managed production; Chroma for local development.
**Rationale:** Pinecone eliminates ops burden at production scale. Chroma has zero-setup for dev/test.

### ADR-003: MCP Language Choice
**Context:** TypeScript vs. Python for MCP servers.
**Decision:** TypeScript as default; Python (FastMCP) when team is Python-only.
**Rationale:** TypeScript SDK has broader client compatibility. Static typing catches tool schema errors at compile time.

### ADR-004: Subagent Review Order
**Context:** Should spec review or code quality review happen first?
**Decision:** Spec compliance ALWAYS before code quality.
**Rationale:** Code quality review on spec-non-compliant code wastes review cycles. Fixing spec gaps may invalidate quality feedback.

### ADR-005: Agent Memory Architecture
**Context:** How should agents persist state in Perplexity Computer?
**Decision:** SQLite via CGI-bin for development; PostgreSQL/Redis for production.
**Rationale:** CGI-bin SQLite requires zero infrastructure, deploys with the frontend. Swap to PostgreSQL when multi-instance or high-traffic.

---

## Appendix C: Evaluation Question Patterns for MCP Servers

When creating the 10 evaluation questions required by Phase 4 of MCP development:

```
GOOD EVALUATION QUESTIONS:

OK Multi-step: requires 3+ tool calls to answer
OK Read-only: only non-destructive operations
OK Verifiable: single correct answer, checkable by string comparison
OK Realistic: something a real user would ask
OK Stable: answer won't change over time
OK Independent: not dependent on other questions

BAD EVALUATION QUESTIONS:

NO Single-step: answerable with one tool call
NO Write operations: creates, updates, or deletes data
NO Ambiguous: multiple valid answers
NO Unstable: answer changes (e.g., "latest version")
NO Dependent: requires previous question's state
```

**Example good evaluation question:**
```xml
<qa_pair>
  <question>
    Find all issues labeled "bug" and "high-priority" in the repository.
    What is the title of the oldest open one, and which user has been assigned
    the most bugs in that same repository?
  </question>
  <answer>Oldest: "Memory leak in session handler" | Most assigned: alice (7 bugs)</answer>
</qa_pair>
```

---

## Appendix D: Common Failure Modes & Fixes

| Failure Mode | Symptoms | Fix |
|-------------|---------|-----|
| Context window overflow | Agent truncates history, loses tool results | Implement sliding window memory, summarize old turns |
| Tool call hallucination | Agent calls nonexistent tools | Enumerate tools explicitly in system prompt; use native tool calling API |
| Prompt injection | User input overrides agent instructions | Wrap user input in XML tags: `<user_input>{input}</user_input>` |
| Infinite ReAct loop | Agent never reaches Final Answer | Add explicit iteration counter; add "if unsure after N steps, state limitations" |
| Parallel agent conflicts | Two agents edit same file | Map files to agents before dispatch; check `check_result_conflicts()` |
| RAG hallucination | Agent answers outside retrieved context | Add "Only answer from context. Say 'I don't know' if not in context." |
| Spec creep | Implementer adds unasked-for features | Spec reviewer checks for EXTRA features; reject scope creep explicitly |
| Cost overrun | Agent exceeds budget | Set `max_tokens` per request; add total-session token budget |
| Stale memory | Agent uses outdated facts | Add TTL to fact store; validate facts against current context |
| MCP tool discovery failure | Agent can't find right tool | Use consistent naming: `service_verb_noun`; add synonyms to description |
