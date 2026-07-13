import type { VercelRequest, VercelResponse } from "@vercel/node";

export default function handler(req: VercelRequest, res: VercelResponse) {
  console.log("PING HIT");
  res.status(200).json({ ok: true, route: "/api/ping", method: req.method });
}
