
import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Structure-aware Markdown chunking.

    chunk_size and overlap are measured in words,
    not model tokens.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and < chunk_size"
        )

    sections = re.split(
        r"(?m)(?=^#{1,6}\s+)",
        text.strip(),
    )

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    def flush():
        nonlocal current_parts, current_words

        if current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_words = 0

    for section in sections:
        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()
        heading = ""

        if lines and re.match(r"^#{1,6}\s+", lines[0]):
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            body = section

        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", body)
            if p.strip()
        ]

        # Keep the heading attached to the section.
        prefix = heading
        if not paragraphs and prefix:
            paragraphs = [prefix]
            prefix = ""

        for paragraph in paragraphs:
            piece = (
                f"{prefix}\n\n{paragraph}"
                if prefix
                else paragraph
            )
            prefix = ""

            words = piece.split()

            if len(words) > chunk_size:
                flush()

                # Split exceptionally large paragraphs.
                step = chunk_size - overlap

                for start in range(0, len(words), step):
                    part = " ".join(
                        words[start:start + chunk_size]
                    )
                    chunks.append(part)

                continue

            if (
                current_parts
                and current_words + len(words) > chunk_size
            ):
                flush()

            current_parts.append(piece)
            current_words += len(words)

    flush()
    return chunks
