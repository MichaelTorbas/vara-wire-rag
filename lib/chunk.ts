// Char-based chunker with overlap to avoid mid-sentence cuts
// Defaults: 1200 chars, 250 overlap
export function chunkTextByChars(
  text: string,
  size = 1200,
  overlap = 250
): string[] {
  const clean = (text || "").replace(/\s+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (!clean) return [];

  const chunks: string[] = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + size, clean.length);

    // try to end on a sentence boundary within the last 120 chars
    const windowStart = Math.max(start, end - 120);
    const slice = clean.slice(windowStart, end);
    const lastStop = Math.max(slice.lastIndexOf(". "), slice.lastIndexOf("\n"));
    if (lastStop > -1 && end < clean.length) {
      end = windowStart + lastStop + 1;
    }

    chunks.push(clean.slice(start, end).trim());
    if (end === clean.length) break;
    start = Math.max(0, end - overlap);
  }
  return chunks.filter(Boolean);
}

