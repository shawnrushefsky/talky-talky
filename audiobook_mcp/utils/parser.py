"""Text parsing utilities for detecting dialogue and splitting prose."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedSegment:
    text: str
    is_dialogue: bool
    dialogue_attribution: Optional[str] = None


# Speech verbs for dialogue detection
SPEECH_VERBS = [
    "said", "asked", "replied", "exclaimed", "whispered", "shouted", "muttered",
    "answered", "called", "cried", "yelled", "demanded", "inquired", "stated",
    "declared", "announced", "continued", "added", "interrupted", "suggested",
    "murmured", "sighed", "groaned", "laughed", "chuckled", "growled", "hissed",
    "screamed", "bellowed", "pleaded", "begged", "insisted", "protested",
    "objected", "agreed", "admitted", "confessed", "explained", "warned",
    "threatened", "promised", "vowed", "swore", "lied", "joked", "teased",
    "mocked", "sneered", "snapped", "barked", "snarled", "cooed", "purred", "crooned"
]

SPEECH_VERBS_PATTERN = "|".join(SPEECH_VERBS)


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def parse_dialogue(text: str) -> list[ParsedSegment]:
    """Split a paragraph into dialogue and narration segments.

    Handles quoted dialogue with optional attribution.
    """
    segments: list[ParsedSegment] = []

    # Match quoted text with optional attribution
    # Quote characters: " " « ' '
    open_quotes = '""«\'\''
    close_quotes = '""»\'\''
    dialogue_pattern = re.compile(
        rf'([{open_quotes}])([^{close_quotes}]+)([{close_quotes}]+)'
        rf'(\s*[,.]?\s*(?:{SPEECH_VERBS_PATTERN})?\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)?)?',
        re.IGNORECASE
    )

    last_index = 0

    for match in dialogue_pattern.finditer(text):
        # Add any narration before this dialogue
        if match.start() > last_index:
            narration = text[last_index:match.start()].strip()
            if narration:
                segments.append(ParsedSegment(text=narration, is_dialogue=False))

        # Add the dialogue
        dialogue_text = match.group(2)
        attribution = match.group(4)

        segments.append(ParsedSegment(
            text=dialogue_text,
            is_dialogue=True,
            dialogue_attribution=attribution.strip() if attribution else None
        ))

        last_index = match.end()

    # Add any remaining narration
    if last_index < len(text):
        remaining = text[last_index:].strip()
        if remaining:
            segments.append(ParsedSegment(text=remaining, is_dialogue=False))

    # If no dialogue was found, return the whole text as narration
    if not segments:
        segments.append(ParsedSegment(text=text, is_dialogue=False))

    return segments


def parse_text(text: str) -> list[ParsedSegment]:
    """Parse an entire text into segments, preserving paragraph structure
    and splitting dialogue from narration within each paragraph.
    """
    paragraphs = split_into_paragraphs(text)
    all_segments: list[ParsedSegment] = []

    for paragraph in paragraphs:
        segments = parse_dialogue(paragraph)
        all_segments.extend(segments)

    return all_segments


def extract_character_names(text: str) -> list[str]:
    """Extract character names mentioned in dialogue attributions."""
    names: set[str] = set()

    # Pattern to find dialogue attributions: "..." said Name
    close_quotes = '""»\'\''
    open_quotes = '""«\'\''
    attribution_pattern = re.compile(
        rf'[{close_quotes}]\s*[,.]?\s*(?:{SPEECH_VERBS_PATTERN})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        re.IGNORECASE
    )

    for match in attribution_pattern.finditer(text):
        if match.group(1):
            names.add(match.group(1))

    # Also look for "Name said" patterns
    prefix_pattern = re.compile(
        rf'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:{SPEECH_VERBS_PATTERN})[,.]?\s*[{open_quotes}]',
        re.IGNORECASE
    )

    for match in prefix_pattern.finditer(text):
        if match.group(1):
            names.add(match.group(1))

    return sorted(names)


def clean_for_tts(text: str) -> str:
    """Clean text for TTS (remove excessive whitespace, normalize quotes, etc.)."""
    result = text
    result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
    result = result.replace('"', '"').replace('"', '"')  # Normalize double quotes
    result = result.replace(''', "'").replace(''', "'")  # Normalize single quotes
    result = result.replace('«', '"').replace('»', '"')  # Convert guillemets
    result = result.replace('—', ' - ')  # Em dash with spaces
    result = result.replace('–', ' - ')  # En dash with spaces
    result = result.replace('...', '…')  # Normalize ellipsis
    return result.strip()
