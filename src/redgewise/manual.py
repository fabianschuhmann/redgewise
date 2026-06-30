from __future__ import annotations

import argparse
import re
import shutil
import textwrap
from importlib import resources
from pathlib import Path


class ManualAction(argparse.Action):
    """Argparse action that prints a command manual and exits immediately."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, **kwargs):
        self.manual_name = kwargs.pop("manual_name")
        kwargs.setdefault("nargs", 0)
        super().__init__(option_strings, dest=dest, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(render_manual(self.manual_name))
        parser.exit(0)


def render_manual(name: str) -> str:
    """Load a Markdown manual from redgewise/manuals and render it for terminal output."""
    markdown = load_manual_markdown(name)
    return render_markdown_for_terminal(markdown)


def load_manual_markdown(name: str) -> str:
    filename = f"{name}.md"

    try:
        package_files = resources.files("redgewise.manuals")
        return package_files.joinpath(filename).read_text(encoding="utf-8")
    except Exception:
        # Fallback for source-tree use if package data was not installed correctly.
        local_path = Path(__file__).resolve().parent / "manuals" / filename
        if local_path.exists():
            return local_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"redgewise manual not found: {filename}")


def render_markdown_for_terminal(markdown: str) -> str:
    """Small dependency-free Markdown renderer for command manuals.

    It keeps code blocks readable, converts headings to terminal-style section
    headers, and removes the most common inline Markdown markers. It is not a
    general Markdown renderer; it is intentionally conservative.
    """
    width = shutil.get_terminal_size((100, 24)).columns
    width = max(72, min(width, 120))

    lines: list[str] = []
    in_code = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            in_code = not in_code
            if not in_code:
                lines.append("")
            continue

        if in_code:
            lines.append(f"  {line}")
            continue

        if not line.strip():
            if lines and lines[-1] != "":
                lines.append("")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            text = strip_inline_markdown(heading.group(2)).strip()
            if level == 1:
                lines.append(text.upper())
                lines.append("=" * min(len(text), width))
            elif level == 2:
                lines.append(text)
                lines.append("-" * min(len(text), width))
            else:
                lines.append(text + ":")
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            text = strip_inline_markdown(bullet.group(1)).strip()
            wrapped = textwrap.wrap(text, width=width - 4) or [""]
            lines.append(f"  • {wrapped[0]}")
            for cont in wrapped[1:]:
                lines.append(f"    {cont}")
            continue

        numbered = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if numbered:
            prefix = f"  {numbered.group(1)}. "
            text = strip_inline_markdown(numbered.group(2)).strip()
            wrapped = textwrap.wrap(text, width=width - len(prefix)) or [""]
            lines.append(prefix + wrapped[0])
            for cont in wrapped[1:]:
                lines.append(" " * len(prefix) + cont)
            continue

        text = strip_inline_markdown(line)
        wrapped = textwrap.wrap(text, width=width) or [""]
        lines.extend(wrapped)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def strip_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"__([^_]*)__", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"_([^_]*)_", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text
