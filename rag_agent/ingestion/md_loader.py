import re

from rag_agent.ingestion.document import Document
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_CODE_FENCE_RE = re.compile(r"^```")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BOLD_RE = re.compile(r"(\*\*|__)(.*?)\1")
_ITALIC_RE = re.compile(r"(\*|_)(.*?)\1")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+")


class MarkdownLoader:
    """
    Parse un fichier Markdown en Document : le texte prosaïque est
    débarrassé de sa syntaxe (gras, italique, liens, listes...), les
    titres sont extraits en metadata structurelle, et les blocs de
    code (```...```) sont conservés tels quels.
    """

    def load(self, path: str) -> Document:
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        text, headings = self._parse(raw)

        logger.info("Markdown chargé : %s", path)

        return Document(
            text=text,
            source=path,
            metadata={"type": "markdown", "headings": headings},
        )

    def _parse(self, raw: str) -> tuple[str, list[dict]]:
        out_lines = []
        headings = []
        in_code_block = False

        for line in raw.splitlines():
            if _CODE_FENCE_RE.match(line):
                in_code_block = not in_code_block
                out_lines.append(line)
                continue

            if in_code_block:
                out_lines.append(line)
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                headings.append({"level": level, "text": heading_text})
                out_lines.append(heading_text)
                continue

            out_lines.append(_strip_inline_markdown(line))

        return "\n".join(out_lines), headings


def _strip_inline_markdown(line: str) -> str:
    line = _IMAGE_RE.sub(r"\1", line)
    line = _LINK_RE.sub(r"\1", line)
    line = _BOLD_RE.sub(r"\2", line)
    line = _ITALIC_RE.sub(r"\2", line)
    line = _INLINE_CODE_RE.sub(r"\1", line)
    line = _BLOCKQUOTE_RE.sub("", line)
    line = _BULLET_RE.sub("", line)
    line = _NUMBERED_RE.sub("", line)
    return line
