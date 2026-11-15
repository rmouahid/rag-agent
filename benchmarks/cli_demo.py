"""Record a real CLI query and render its unedited stdout as a terminal SVG."""

import argparse
import html
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def render(transcript: str) -> str:
    lines = [
        wrapped
        for line in transcript.splitlines()
        for wrapped in (textwrap.wrap(line, width=100, replace_whitespace=False) or [""])
    ]
    height = 100 + 23 * len(lines)
    rows = []
    for i, line in enumerate(lines):
        color = "#8ce99a" if line.startswith("$ ") else "#e6edf3"
        rows.append(f'<text x="24" y="{85 + i * 23}" fill="{color}">{html.escape(line)}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
        f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">\n'
        '<title id="title">RAG Agent: real CLI ingestion and query</title>\n'
        '<desc id="desc">Recorded stdout rendered as a terminal transcript. '
        "No answers are mocked or edited.</desc>\n"
        f'<rect width="1120" height="{height}" rx="12" fill="#0d1117"/>\n'
        '<g fill="#8b949e" font-family="monospace" font-size="15">'
        '<text x="24" y="34">RAG Agent · recorded CLI output</text></g>\n'
        '<g font-family="DejaVu Sans Mono, monospace" font-size="17" xml:space="preserve">\n'
        + "\n".join(rows)
        + "\n</g></svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tests/fixtures/knowledge.md")
    parser.add_argument("--question", default="Who created Python and when was it first released?")
    parser.add_argument("--output", type=Path, default=Path("docs/assets/cli-demo"))
    args = parser.parse_args()
    if not Path(args.source).is_file():
        parser.error("source must be an existing document")
    entry = "from rag_agent.cli import main; raise SystemExit(main())"
    transcript = ""
    with tempfile.TemporaryDirectory(prefix="rag-cli-demo-") as store:
        env = dict(os.environ, VECTOR_STORE_PATH=store)
        for command in [["ingest", args.source], ["ask", args.question]]:
            result = subprocess.run(
                [sys.executable, "-c", entry, *command],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            transcript += "$ rag-agent " + shlex.join(command) + "\n" + result.stdout + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".txt").write_text(transcript, encoding="utf-8")
    args.output.with_suffix(".svg").write_text(render(transcript), encoding="utf-8")


if __name__ == "__main__":
    main()
