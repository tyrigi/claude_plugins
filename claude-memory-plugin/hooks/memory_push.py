#!/usr/bin/env python3
"""
Memory Push Hook Script - Multi-Extractor Architecture

Uses specialized extractors to reliably identify different types of
memorable information from conversation transcripts:

1. Preferences & Requirements - User preferences, tool choices, style
2. Decisions & Rationale - Architecture choices, tradeoffs, reasoning
3. Insights & Corrections - Breakthroughs, debugging discoveries
4. Project State (optional) - Blockers, next steps, environmental facts

Each extractor has a narrow focus for reliability. Results are tagged
with categories for filtered retrieval later.

Usage:
    echo '{"transcript": "..."}' | python memory_push.py

Environment:
    MEM0_LLM_URL - Override LLM endpoint (default: auto-detect)
    MEM0_EXTRACT_MODEL - Model for extraction (default: qwen3:4b-instruct)
    MEM0_SKIP_STATE - Set to "1" to skip project state extractor
    MEM0_PARALLEL - Set to "1" to run extractors in parallel
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add lib dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import httpx
from claude_memory import ClaudeMemory
from platform_utils import get_queue_path, get_log_dir, ensure_data_dir, ensure_log_dir, file_lock

# Configure logging
ensure_log_dir()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [memory_push] %(message)s',
    handlers=[
        logging.FileHandler(get_log_dir() / 'memory_push.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
log = logging.getLogger(__name__)


# =============================================================================
# Extractor Prompts - Each has ONE job for reliability
# =============================================================================

EXTRACTOR_PREFERENCES = """Does this transcript contain any user preferences, requirements, or working style information?

Look for:
- Explicit statements of preference ("I prefer...", "I always want...", "Don't do X")
- Tool/technology choices with reasoning
- Communication style preferences
- Project constraints or requirements

If none found, return empty JSON array: []
Otherwise return JSON array of preference strings.

Transcript:
---
{transcript}
---

Return ONLY a JSON array of strings, nothing else:"""


EXTRACTOR_DECISIONS = """Does this transcript contain any decisions that were made, along with why?

Look for:
- Architecture or design choices
- Tradeoff evaluations
- "We'll go with X because Y"
- Rejected alternatives and why

Capture the decision AND the reasoning as a single statement.

If none found, return empty JSON array: []
Otherwise return JSON array of decision strings.

Transcript:
---
{transcript}
---

Return ONLY a JSON array of strings, nothing else:"""


EXTRACTOR_INSIGHTS = """Does this transcript contain any breakthroughs, realizations, or corrections?

Look for:
- Problems solved with the key insight that fixed it
- Debugging discoveries ("the issue was actually X")
- User correcting a misunderstanding
- "Aha" moments that changed approach

Capture what was learned, not the steps to get there.

If none found, return empty JSON array: []
Otherwise return JSON array of insight strings.

Transcript:
---
{transcript}
---

Return ONLY a JSON array of strings, nothing else:"""


EXTRACTOR_STATE = """Does this transcript contain any project state information that would be lost if this session ended?

Look for:
- Current blockers or open questions
- Next steps that were identified
- Environmental facts (paths, configurations, versions)
- Work in progress that's incomplete

If none found, return empty JSON array: []
Otherwise return JSON array of state fact strings.

Transcript:
---
{transcript}
---

Return ONLY a JSON array of strings, nothing else:"""


@dataclass
class ExtractorConfig:
    """Configuration for a single extractor."""
    name: str
    prompt_template: str
    category: str


EXTRACTORS = [
    ExtractorConfig("preferences", EXTRACTOR_PREFERENCES, "preference"),
    ExtractorConfig("decisions", EXTRACTOR_DECISIONS, "decision"),
    ExtractorConfig("insights", EXTRACTOR_INSIGHTS, "insight"),
]

# State extractor is optional - it goes stale fast
STATE_EXTRACTOR = ExtractorConfig("state", EXTRACTOR_STATE, "state")


def get_llm_endpoint() -> str:
    """Get LLM endpoint, preferring localhost (DDR5 CPU faster than desktop DDR4)."""
    override = os.environ.get("MEM0_LLM_URL")
    if override:
        return override

    # Prefer localhost - DDR5 bandwidth makes CPU inference faster here
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return "http://localhost:11434"
    except:
        pass

    # Fall back to desktop (slower DDR4 CPU, but available)
    return "http://192.168.50.3:11434"


def parse_json_array(text: str) -> list[str]:
    """Parse LLM response as JSON array, handling common issues."""
    text = text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item) for item in result if item]
        return []
    except json.JSONDecodeError:
        pass

    # Try to extract array from response
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item) for item in result if item]
        except json.JSONDecodeError:
            pass

    return []


def run_extractor(
    extractor: ExtractorConfig,
    transcript: str,
    llm_url: str,
    model: str
) -> tuple[str, list[str]]:
    """
    Run a single extractor against the transcript.

    Returns:
        Tuple of (category, list of extracted facts)
    """
    prompt = extractor.prompt_template.format(transcript=transcript[:12000])

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{llm_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 500,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "")

            facts = parse_json_array(text)
            log.info(f"Extractor '{extractor.name}' found {len(facts)} items")
            return extractor.category, facts

    except Exception as e:
        log.error(f"Extractor '{extractor.name}' failed: {e}")
        return extractor.category, []


def extract_all(
    transcript: str,
    llm_url: str,
    model: str,
    include_state: bool = False,
    parallel: bool = False
) -> list[dict]:
    """
    Run all extractors and aggregate results.

    Args:
        transcript: Conversation transcript to analyze
        llm_url: Ollama endpoint
        model: Model name
        include_state: Whether to run the state extractor
        parallel: Whether to run extractors in parallel

    Returns:
        List of {"content": str, "category": str} dicts
    """
    extractors = list(EXTRACTORS)
    if include_state:
        extractors.append(STATE_EXTRACTOR)

    memories = []

    if parallel:
        # Run extractors concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_extractor, ext, transcript, llm_url, model): ext
                for ext in extractors
            }
            for future in as_completed(futures):
                category, facts = future.result()
                for fact in facts:
                    memories.append({"content": fact, "category": category})
    else:
        # Run sequentially
        for extractor in extractors:
            category, facts = run_extractor(extractor, transcript, llm_url, model)
            for fact in facts:
                memories.append({"content": fact, "category": category})

    return memories


def store_memories(memories: list[dict], memory: ClaudeMemory) -> int:
    """
    Store extracted memories in mem0 with category metadata.

    Args:
        memories: List of {"content": str, "category": str} dicts
        memory: ClaudeMemory instance

    Returns:
        Number of memories successfully stored
    """
    stored = 0
    for mem in memories:
        try:
            result = memory.remember(
                mem["content"],
                category=mem["category"]
            )
            # Check if stored in either primary or secondary
            if result.get("primary", {}).get("results") or result.get("secondary", {}).get("results"):
                stored += 1
                log.info(f"[{mem['category']}] {mem['content'][:60]}...")
        except Exception as e:
            log.error(f"Failed to store: {e}")

    return stored


def strip_tool_calls(content) -> str:
    """
    Strip tool calls and results from message content.

    Tool calls are noise for memory extraction - we want the
    conversational content (preferences, decisions, insights).
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Content blocks - filter to just text
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    # Optionally keep a minimal trace
                    tool_name = block.get("name", "unknown")
                    text_parts.append(f"[Used tool: {tool_name}]")
                elif block_type == "tool_result":
                    # Skip tool results entirely - this is the noise
                    pass
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)

    return str(content) if content else ""


def process_hook_input(input_data: dict) -> str:
    """
    Extract transcript from hook input data, stripping tool calls.

    Tool results (file contents, bash output, etc.) are filtered out
    to reduce noise for the extractors.
    """
    # Claude Code Stop hook provides transcript_path - read JSONL file
    # Only process the LAST TURN (last user message + subsequent assistant responses)
    # since we process on every stop hook, not just session end
    if "transcript_path" in input_data:
        transcript_path = Path(input_data["transcript_path"])
        if transcript_path.exists():
            try:
                # Read all entries to find the last turn
                entries = []
                with open(transcript_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                # Find the last REAL user message (with text, not just tool_result)
                last_user_idx = -1
                for i in range(len(entries) - 1, -1, -1):
                    entry = entries[i]
                    if entry.get("type") == "user":
                        content = entry.get("message", {}).get("content", "")
                        # Check if this has actual text content (not just tool_result)
                        has_text = False
                        if isinstance(content, str) and content.strip():
                            has_text = True
                        elif isinstance(content, list):
                            for block in content:
                                if block.get("type") == "text" and block.get("text", "").strip():
                                    has_text = True
                                    break
                        if has_text:
                            last_user_idx = i
                            break

                if last_user_idx == -1:
                    log.info("No user message found in transcript")
                    return ""

                # Only process from last user message onward (the current turn)
                filtered_parts = []
                for entry in entries[last_user_idx:]:
                    if entry.get("type") in ("user", "assistant"):
                        msg = entry.get("message", {})
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        clean_content = strip_tool_calls(content)
                        if clean_content.strip():
                            filtered_parts.append(f"{role}: {clean_content}")

                if filtered_parts:
                    log.info(f"Processing last turn only: {len(filtered_parts)} messages")
                    return "\n\n".join(filtered_parts)
            except Exception as e:
                log.error(f"Failed to read transcript file: {e}")

    # Direct transcript (already processed)
    if "transcript" in input_data:
        return input_data["transcript"]

    # Context string
    if "context" in input_data:
        return input_data["context"]

    # Conversation format
    if "conversation" in input_data:
        return input_data["conversation"]

    # Messages array - filter out tool noise
    if "messages" in input_data:
        messages = input_data["messages"]
        filtered_parts = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")

            # Strip tool calls from content
            clean_content = strip_tool_calls(content)

            if clean_content.strip():
                filtered_parts.append(f"{role}: {clean_content}")

        return "\n\n".join(filtered_parts)

    # Summary format
    if "summary" in input_data:
        return input_data["summary"]

    # Raw text
    if "text" in input_data:
        return input_data["text"]

    # Serialize the whole thing (fallback)
    return json.dumps(input_data, indent=2)


def main():
    """Main entry point for hook script."""
    log.info("Memory push hook triggered (multi-extractor)")

    # Read input
    try:
        input_text = sys.stdin.read()
        if not input_text.strip():
            log.info("No input provided, exiting")
            return 0

        # DEBUG: Write raw input to file for inspection
        debug_path = Path("/tmp/memory_push_debug.json")
        debug_path.write_text(input_text[:50000])  # Cap at 50k
        log.info(f"DEBUG: Wrote raw input to {debug_path} ({len(input_text)} bytes)")

        try:
            input_data = json.loads(input_text)
        except json.JSONDecodeError:
            input_data = {"text": input_text}

        transcript = process_hook_input(input_data)

        # DEBUG: Write processed transcript
        transcript_path = Path("/tmp/memory_push_transcript.txt")
        transcript_path.write_text(transcript[:50000])
        log.info(f"DEBUG: Wrote transcript to {transcript_path} ({len(transcript)} chars)")

        if not transcript or len(transcript) < 100:
            log.info("Transcript too short, skipping")
            return 0

    except Exception as e:
        log.error(f"Failed to read input: {e}")
        return 1

    # Configuration
    llm_url = get_llm_endpoint()
    model = os.environ.get("MEM0_EXTRACT_MODEL", "qwen3:4b-instruct")
    include_state = os.environ.get("MEM0_SKIP_STATE", "1") != "1"  # Disabled by default - goes stale fast
    parallel = os.environ.get("MEM0_PARALLEL", "1") == "1"  # Enabled by default for speed

    log.info(f"Using LLM: {llm_url}, model: {model}")
    log.info(f"State extractor: {'enabled' if include_state else 'disabled'}")
    log.info(f"Parallel mode: {'enabled' if parallel else 'disabled'}")

    # Run extractors
    memories = extract_all(
        transcript=transcript,
        llm_url=llm_url,
        model=model,
        include_state=include_state,
        parallel=parallel
    )

    log.info(f"Total extracted: {len(memories)} memories")

    if not memories:
        print(json.dumps({"status": "ok", "extracted": 0, "queued": 0}))
        return 0

    # Queue memories for async processing instead of blocking
    import time
    queue_path = get_queue_path()
    ensure_data_dir()

    queued = 0
    try:
        with open(queue_path, "a") as f:
            with file_lock(f, exclusive=True):  # Exclusive lock for appending
                for mem in memories:
                    entry = {
                        "content": mem["content"],
                        "category": mem["category"],
                        "timestamp": time.time(),
                        "llm_url": llm_url,
                    }
                    f.write(json.dumps(entry) + "\n")
                    queued += 1

        log.info(f"Queued {queued}/{len(memories)} memories for async storage")
    except Exception as e:
        log.error(f"Failed to queue memories: {e}")

    # Output result - return immediately without waiting for storage
    result = {
        "status": "ok",
        "extracted": len(memories),
        "queued": queued,
        "by_category": {}
    }
    for mem in memories:
        cat = mem["category"]
        result["by_category"][cat] = result["by_category"].get(cat, 0) + 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
