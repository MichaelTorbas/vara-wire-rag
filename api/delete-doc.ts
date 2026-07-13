// api/delete-doc.ts
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { qdrant, COLLECTION } from "../lib/qdrant.js";

export default async (req: VercelRequest, res: VercelResponse) => {
  try {
    if (req.method !== "POST") return res.status(405).json({ error: "Use POST" });
    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    const { doc_id } = body || {};
    if (!doc_id) return res.status(400).json({ error: "doc_id required" });

    await qdrant.delete(COLLECTION, { filter: { must: [{ key: "doc_id", match: { value: doc_id } }] } });
    return res.json({ ok: true, deleted_doc_id: doc_id });
  } catch (e:any) {
    console.error("DELETE_DOC_ERR", e?.message || e);
    return res.status(500).json({ error: "DELETE_DOC_ERR", detail: String(e?.message || e) });
  }
};
