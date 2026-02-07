#!/usr/bin/env python3
"""Validate all agent markdown files in the repository.

Checks:
- YAML frontmatter is well-formed (name, description required)
- Referenced tools are valid Claude Code tools
- Agents referenced in delegation sections exist
- File size warnings (> 500 lines)

Usage:
    python tests/validate_agents.py
"""

import os
import re
import sys
from pathlib import Path

VALID_TOOLS = {
    "Read", "Write", "Edit", "MultiEdit", "Bash", "Grep", "Glob", "LS",
    "WebFetch", "WebSearch", "Task", "TodoRead", "TodoWrite",
}

AGENTS_DIR = Path(__file__).parent.parent / "agents"

LINE_WARNING_THRESHOLD = 500


def find_agent_files(base_dir: Path) -> list[Path]:
    """Find all .md files under agents/."""
    return sorted(base_dir.rglob("*.md"))


def parse_frontmatter(content: str) -> dict | None:
    """Extract YAML frontmatter from markdown content."""
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---", content, re.DOTALL)
    if not match:
        return None

    frontmatter = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key in frontmatter:
                frontmatter[key] += " " + value
            else:
                frontmatter[key] = value

    return frontmatter


def get_all_agent_names(base_dir: Path) -> set[str]:
    """Collect all agent names from frontmatter."""
    names = set()
    for filepath in find_agent_files(base_dir):
        content = filepath.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)
        if fm and "name" in fm:
            names.add(fm["name"])
    return names


def validate_agent(filepath: Path, known_agents: set[str]) -> list[str]:
    """Validate a single agent file. Returns list of issues."""
    issues = []
    rel_path = filepath.relative_to(AGENTS_DIR.parent)

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return [f"  ERROR: Cannot read file: {e}"]

    lines = content.splitlines()

    # Check line count
    if len(lines) > LINE_WARNING_THRESHOLD:
        issues.append(
            f"  WARNING: {len(lines)} lines (>{LINE_WARNING_THRESHOLD} threshold)"
        )

    # Check frontmatter
    fm = parse_frontmatter(content)
    if fm is None:
        issues.append("  ERROR: No YAML frontmatter found (missing --- delimiters)")
        return issues

    # Required fields
    if "name" not in fm or not fm["name"]:
        issues.append("  ERROR: Missing required field 'name' in frontmatter")

    if "description" not in fm or not fm["description"]:
        issues.append("  ERROR: Missing required field 'description' in frontmatter")

    # Validate tools if specified
    if "tools" in fm and fm["tools"]:
        tools = [t.strip() for t in fm["tools"].split(",")]
        for tool in tools:
            if tool and tool not in VALID_TOOLS:
                issues.append(f"  WARNING: Unknown tool '{tool}' in frontmatter")

        # Check for duplicates
        seen = set()
        for tool in tools:
            if tool in seen:
                issues.append(f"  WARNING: Duplicate tool '{tool}' in frontmatter")
            seen.add(tool)

    # Check for references to non-existent agents in delegation sections
    agent_refs = re.findall(r"`([\w-]+(?:-[\w-]+)*)`", content)
    delegation_keywords = {
        "security-guardian", "refactoring-expert", "testing-specialist",
    }
    for ref in agent_refs:
        if ref in delegation_keywords and ref not in known_agents:
            issues.append(
                f"  WARNING: References non-existent agent '{ref}'"
            )

    return issues


def main():
    if not AGENTS_DIR.exists():
        print(f"ERROR: Agents directory not found at {AGENTS_DIR}")
        sys.exit(1)

    agent_files = find_agent_files(AGENTS_DIR)
    print(f"Found {len(agent_files)} agent files\n")

    # First pass: collect all agent names
    known_agents = get_all_agent_names(AGENTS_DIR)
    print(f"Known agents: {len(known_agents)}")
    print(f"  {', '.join(sorted(known_agents))}\n")

    # Second pass: validate each agent
    total_errors = 0
    total_warnings = 0

    for filepath in agent_files:
        rel_path = filepath.relative_to(AGENTS_DIR.parent)
        issues = validate_agent(filepath, known_agents)

        if issues:
            print(f"{rel_path}:")
            for issue in issues:
                print(issue)
                if "ERROR" in issue:
                    total_errors += 1
                elif "WARNING" in issue:
                    total_warnings += 1
            print()

    # Summary
    print("=" * 60)
    print(f"Validation complete: {len(agent_files)} files checked")
    print(f"  Errors:   {total_errors}")
    print(f"  Warnings: {total_warnings}")

    if total_errors > 0:
        print("\nValidation FAILED - fix errors above")
        sys.exit(1)
    else:
        print("\nValidation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
