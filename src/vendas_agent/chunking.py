"""Split documents into retrievable chunks.

Strategy: split by Markdown heading (keeps each chunk about one topic and gives us
a section title to cite), pack paragraphs up to ``max_chars``, and only slice a
paragraph (with overlap) when a single paragraph is too large.
"""

import re

from vendas_agent.models import Chunk, Document

_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_BLANK_LINE = re.compile(r"\n\s*\n")
DEFAULT_SECTION = "Geral"


def _validate(max_chars: int, overlap: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return ``(section_title, body)`` pairs, skipping empty sections."""
    sections: list[tuple[str, str]] = []
    title = DEFAULT_SECTION
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((title, body))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            title = match.group(1)
            buffer = []
        else:
            buffer.append(line)
    flush()
    return sections


def _slice_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # prefer cutting at a space so words are not split in half
            cut = text.rfind(" ", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)  # always make progress
        if text[next_start - 1] != " ":
            # do not begin the next piece in the middle of a word
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return pieces


def split_body(body: str, max_chars: int, overlap: int) -> list[str]:
    _validate(max_chars, overlap)
    pieces: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in _BLANK_LINE.split(body)):
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_slice_with_overlap(paragraph, max_chars, overlap))
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            pieces.append(current)
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc: Document, max_chars: int = 700, overlap: int = 80) -> list[Chunk]:
    _validate(max_chars, overlap)
    chunks: list[Chunk] = []
    for title, body in split_sections(doc.text):
        for piece in split_body(body, max_chars, overlap):
            chunks.append(
                Chunk(text=piece, source=doc.source, section=title, position=len(chunks))
            )
    return chunks
