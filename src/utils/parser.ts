/**
 * Text parsing utilities for detecting dialogue and splitting prose
 */

export interface ParsedSegment {
  text: string;
  is_dialogue: boolean;
  dialogue_attribution?: string; // e.g., "said John" or "John asked"
}

/**
 * Split text into paragraphs
 */
export function splitIntoParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
}

/**
 * Split a paragraph into dialogue and narration segments.
 * Handles quoted dialogue with optional attribution.
 */
export function parseDialogue(text: string): ParsedSegment[] {
  const segments: ParsedSegment[] = [];

  // Match quoted text with optional attribution
  // This regex captures:
  // - Opening quote (", ', ", ", « )
  // - The quoted content
  // - Closing quote
  // - Optional attribution phrase (said X, X replied, etc.)
  const dialoguePattern =
    /([""«''])([^""»'']+)([""»''])(\s*[,.]?\s*(?:said|asked|replied|exclaimed|whispered|shouted|muttered|answered|called|cried|yelled|demanded|inquired|stated|declared|announced|continued|added|interrupted|suggested|murmured|sighed|groaned|laughed|chuckled|growled|hissed|screamed|bellowed|pleaded|begged|insisted|protested|objected|agreed|admitted|confessed|explained|warned|threatened|promised|vowed|swore|lied|joked|teased|mocked|sneered|snapped|barked|snarled|cooed|purred|crooned)?\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)?)?/gi;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = dialoguePattern.exec(text)) !== null) {
    // Add any narration before this dialogue
    if (match.index > lastIndex) {
      const narration = text.slice(lastIndex, match.index).trim();
      if (narration) {
        segments.push({
          text: narration,
          is_dialogue: false,
        });
      }
    }

    // Add the dialogue
    const dialogueText = match[2];
    const attribution = match[4]?.trim();

    segments.push({
      text: dialogueText,
      is_dialogue: true,
      dialogue_attribution: attribution || undefined,
    });

    lastIndex = match.index + match[0].length;
  }

  // Add any remaining narration
  if (lastIndex < text.length) {
    const remaining = text.slice(lastIndex).trim();
    if (remaining) {
      segments.push({
        text: remaining,
        is_dialogue: false,
      });
    }
  }

  // If no dialogue was found, return the whole text as narration
  if (segments.length === 0) {
    segments.push({
      text: text,
      is_dialogue: false,
    });
  }

  return segments;
}

/**
 * Parse an entire text into segments, preserving paragraph structure
 * and splitting dialogue from narration within each paragraph.
 */
export function parseText(text: string): ParsedSegment[] {
  const paragraphs = splitIntoParagraphs(text);
  const allSegments: ParsedSegment[] = [];

  for (const paragraph of paragraphs) {
    const segments = parseDialogue(paragraph);
    allSegments.push(...segments);
  }

  return allSegments;
}

/**
 * Extract character names mentioned in dialogue attributions
 */
export function extractCharacterNames(text: string): string[] {
  const names = new Set<string>();

  // Pattern to find dialogue attributions
  const attributionPattern =
    /[""»'']\s*[,.]?\s*(?:said|asked|replied|exclaimed|whispered|shouted|muttered|answered|called|cried|yelled|demanded|inquired|stated|declared|announced|continued|added|interrupted|suggested|murmured|sighed|groaned|laughed|chuckled|growled|hissed|screamed|bellowed|pleaded|begged|insisted|protested|objected|agreed|admitted|confessed|explained|warned|threatened|promised|vowed|swore|lied|joked|teased|mocked|sneered|snapped|barked|snarled|cooed|purred|crooned)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/gi;

  let match: RegExpExecArray | null;
  while ((match = attributionPattern.exec(text)) !== null) {
    if (match[1]) {
      names.add(match[1]);
    }
  }

  // Also look for "Name said" patterns
  const prefixPattern =
    /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:said|asked|replied|exclaimed|whispered|shouted|muttered|answered|called|cried|yelled|demanded|inquired|stated|declared|announced|continued|added|interrupted|suggested|murmured|sighed|groaned|laughed|chuckled|growled|hissed|screamed|bellowed|pleaded|begged|insisted|protested|objected|agreed|admitted|confessed|explained|warned|threatened|promised|vowed|swore|lied|joked|teased|mocked|sneered|snapped|barked|snarled|cooed|purred|crooned)[,.]?\s*[""«'']/gi;

  while ((match = prefixPattern.exec(text)) !== null) {
    if (match[1]) {
      names.add(match[1]);
    }
  }

  return Array.from(names).sort();
}

/**
 * Clean text for TTS (remove excessive whitespace, normalize quotes, etc.)
 */
export function cleanForTTS(text: string): string {
  return text
    .replace(/\s+/g, " ") // Normalize whitespace
    .replace(/[""]/g, '"') // Normalize double quotes
    .replace(/['']/g, "'") // Normalize single quotes
    .replace(/[«»]/g, '"') // Convert guillemets
    .replace(/—/g, " - ") // Em dash with spaces
    .replace(/–/g, " - ") // En dash with spaces
    .replace(/\.\.\./g, "…") // Normalize ellipsis
    .trim();
}
