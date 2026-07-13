// api/list.ts
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { qdrant, COLLECTION } from "../lib/qdrant.js";

export default async (req: VercelRequest, res: VercelResponse) => {
  try {
    const limit = Number(req.query.limit ?? 20);
    const r = await qdrant.scroll(COLLECTION, { limit, with_payload: true, with_vector: false });
    const rows = (r.points || []).map(p => ({
      id: p.id, title: p.payload?.title, source: p.payload?.source,
      tags: p.payload?.tags, doc_id: p.payload?.doc_id, chunk_index: p.payload?.chunk_index
    }));
    res.json({ ok: true, rows });
  } catch (e:any) {
    console.error("LIST_ERR", e?.message || e);
    res.status(500).json({ error: "LIST_ERR", detail: String(e?.message || e) });
  }
};
