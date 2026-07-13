// /api/ingest.ts
import type { VercelRequest, VercelResponse } from "@vercel/node";
import OpenAI from "openai";
import { randomUUID } from "node:crypto";
import { qdrant, COLLECTION } from "../lib/qdrant.js";
import { chunkTextByChars } from "../lib/chunk.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = "text-embedding-3-small"; // 1536-dim

type CorpusItem = {
  id?: string;
  title?: string;
  url?: string;
  source?: string;
  tags?: string[];
  date?: string;
  text?: string;         // optional; if missing and url present, we will fetch
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    if (req.method !== "POST") return res.status(405).json({ error: "Use POST" });

    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    const { items } = (body || {}) as { items: CorpusItem[] };
    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: "Body must be { items: CorpusItem[] }" });
    }
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "INDEX_ERROR", detail: "Missing OPENAI_API_KEY" });
    }

    let itemsProcessed = 0;
    let fetchedFromUrl = 0;
    let insertedChunks = 0;
    let pointsWritten = 0;

    const allPoints: any[] = [];

    for (const item of items) {
      itemsProcessed++;

      // 1) Ensure we have text. If not, fetch from URL
      let text = (item.text || "").trim();
      if (!text && item.url) {
        try {
          text = await fetchAndExtractText(item.url);
          if (text) fetchedFromUrl++;
        } catch (e) {
          console.warn("Fetch failed for", item.url, e);
        }
      }
      if (!text) continue; // skip empty

      // 2) Chunk
      const chunks = chunkTextByChars(text);
      insertedChunks += chunks.length;

      // 3) Embed in one shot
      const emb = await openai.embeddings.create({
        model: EMB_MODEL,
        input: chunks,
      });

      // 4) Build points
      const docId = item.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      emb.data.forEach((ed, idx) => {
        const vec = ed.embedding as number[];
        if (!vec || vec.length !== 1536) {
          throw new Error(`Embedding dimension mismatch: got ${vec?.length ?? 0}, expected 1536`);
        }
        allPoints.push({
          id: randomUUID(),
          vector: vec,
          payload: {
            text: chunks[idx],
            title: item.title || item.url || "",
            url: item.url || "",
            source: item.source || "",
            tags: item.tags || [],
            doc_id: docId,
            chunk_index: idx,
            date: item.date || null,
          },
        });
      });
    }

    // 5) Upsert once
    if (allPoints.length) {
      await qdrant.upsert(COLLECTION, { wait: true, points: allPoints });
      pointsWritten = allPoints.length;
    }

    return res.status(200).json({
      ok: true,
      items_processed: itemsProcessed,
      fetched_from_url: fetchedFromUrl,
      inserted_chunks: insertedChunks,
      points_written: pointsWritten,
      collection: COLLECTION,
    });
  } catch (err: any) {
    console.error("INDEX_ERROR RAW:", err?.message || err);
    if (err?.response) {
      try {
        const data = err.response.data ?? (await err.response.text?.());
        console.error("INDEX_ERROR RESP:", data);
      } catch {}
    }
    return res.status(500).json({ error: "INDEX_ERROR", detail: String(err?.message || err) });
  }
}

/** Minimal HTML → text extraction without extra deps */
async function fetchAndExtractText(url: string): Promise<string> {
  const resp = await fetch(url, { headers: { "user-agent": "Mozilla/5.0" } });
  const html = await resp.text();

  // Strip scripts/styles, then tags, then collapse whitespace.
  const cleaned = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<\/?[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return cleaned;
}
