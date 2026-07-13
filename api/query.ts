// /api/query.ts
import type { VercelRequest, VercelResponse } from "@vercel/node";
import OpenAI from "openai";
import { qdrant, COLLECTION } from "../lib/qdrant.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Embeddings: lean & fast
const EMB_MODEL = "text-embedding-3-small";
// Default generation model: 4o primary, env can override (e.g., "gpt-5")
// Fallback remains 4o-mini.
const DEFAULT_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL ?? "gpt-4o";

// ---- helpers ----
const isGpt5 = (m: string) => /^gpt-5/i.test(m);
const toNumber = (v: any, d: number) => (Number.isFinite(Number(v)) ? Number(v) : d);

// --- length helpers ---
const clamp = (n: number, lo: number, hi: number) => Math.min(Math.max(n, lo), hi);
const wordsToTokens = (w: number) => Math.round(w * 1.35); // ~1.3–1.5 tok/word
const enforceWordTarget = (text: string, target: number) => {
  if (!text) return text;
  const maxAllowed = Math.round(target * 1.10); // +10% tolerance
  const words = text.trim().split(/\s+/);
  if (words.length <= maxAllowed) return text;
  const slice = words.slice(0, maxAllowed).join(" ");
  const lastPeriod = slice.lastIndexOf(".");
  if (lastPeriod > slice.length - 240) return slice.slice(0, lastPeriod + 1);
  return slice + "…";
};

// --- continuation helpers (for articles/qa) ---
const countWords = (s: string) => (s.trim().match(/\b\S+\b/g) || []).length;
const tail = (s: string, n = 2000) => s.slice(Math.max(0, s.length - n));
const hasConclusion = (s: string) =>
  /\b(conclusion|final thoughts|wrapping up|in summary|to conclude)\b/i.test(s);

// Normalize SDK outputs
function normalizeParts(content: any): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((p) => (typeof p === "string" ? p : p?.text ?? p?.content ?? "")).join("");
  }
  return "";
}
function extractResponseText(resp: any): string {
  try {
    if (typeof resp?.output_text === "string") return resp.output_text;
    const out = resp?.output ?? resp?.content ?? [];
    const parts = Array.isArray(out) ? out : [out];
    const chunks: string[] = [];
    for (const item of parts) {
      const c = item?.content ?? item;
      if (Array.isArray(c)) {
        for (const part of c) {
          if (typeof part === "string") chunks.push(part);
          else if (part?.text) chunks.push(String(part.text));
          else if (part?.output_text) chunks.push(String(part.output_text));
        }
      } else if (typeof c === "string") chunks.push(c);
      else if (c?.text) chunks.push(String(c.text));
    }
    return chunks.join("");
  } catch {
    return "";
  }
}

// tweet clamps
function clampTweet(s: string): string {
  if (!s) return s;
  const oneLine = s.replace(/\s+/g, " ").trim();
  return oneLine.length <= 280 ? oneLine : oneLine.slice(0, 277) + "...";
}
function clampTweetsList(s: string): string {
  if (!s) return s.trim();
  const lines = s.split(/\r?\n/);
  const numbered = lines.filter(l => /^\s*\d+\.\s+/.test(l)).length >= 1;
  const dashed = lines.filter(l => /^\s*[-*]\s+/.test(l)).length >= 1;

  if (numbered || dashed) {
    let idx = 0;
    return lines.map((l) => {
      const mNum = l.match(/^(\s*\d+\.\s+)([\s\S]*)$/);
      const mDash = l.match(/^(\s*[-*]\s+)([\s\S]*)$/);
      if (mNum) {
        const prefix = mNum[1];
        const body = mNum[2].replace(/\s+/g, " ").trim();
        const clamped = body.length <= 280 ? body : (body.slice(0, 277) + "…");
        return prefix + clamped;
      } else if (mDash) {
        const prefix = `${++idx}. `;
        const body = mDash[2].replace(/\s+/g, " ").trim();
        const clamped = body.length <= 280 ? body : (body.slice(0, 277) + "…");
        return prefix + clamped;
      } else {
        return l;
      }
    }).join("\n");
  }
  return clampTweet(s);
}

// --- FACET EXPANSION (tweet-mode) ---
// fixed: 'gasless' spelling
const TECH_HINT_RE = /(gear|actor model|wasm|message automation|light client|sync committee|beacon|zk|plonky|gnark|merkle|relayer|dao|governance|gasless)/i;

// GENERIC-ONLY facet strings (mechanism-level, not marketing)
const FACET_GROUPS: Record<string, string[]> = {
  generic: [
    "Gear actor model persistent actors",
    "message queues async messaging patterns",
    "WASM execution model gas metering",
    "system mailbox scheduling inbox/outbox",
    "light client finality verification",
    "permissionless relayers retry/ack",
    "Merkle commitments inclusion proofs",
    "zk proofs Plonky2 gnark circuits",
  ],
};

const hasTechHint = (s: string) => TECH_HINT_RE.test(s);

// hit shaping
const hitMap = (h: any) => ({
  score: Number(h.score) || 0,
  text: String(h.payload?.text ?? ""),
  title: String(h.payload?.title ?? ""),
  url: String(h.payload?.url ?? ""),
  source: String(h.payload?.source ?? ""),
  chunk_index: h.payload?.chunk_index as number | undefined,
});
const keyOf = (h: any) =>
  `${h.url || "no-url"}#${h.chunk_index ?? ""}|${(h.text || "").slice(0, 64)}`;
function dedupeHits(arr: any[]) {
  const seen = new Set<string>();
  return arr.filter((h) => {
    const k = keyOf(h);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}
// Always return only "generic" topics
function topicHintsFromQ(_q: string): string[] {
  return ["generic"];
}

// lightweight boosts to prefer authoritative/technical chunks
const DOMAIN_BOOST: Array<{ rx: RegExp; bump: number }> = [
  { rx: /wiki\.gear\.foundation/i, bump: 0.08 },
  { rx: /wiki\.vara\.network/i, bump: 0.08 },
  { rx: /medium\.com\/@VaraNetwork/i, bump: 0.04 },
];
function boostScore(h: any) {
  let s = h.score;
  for (const r of DOMAIN_BOOST) if (r.rx.test(h.url || "")) s += r.bump;
  if (TECH_HINT_RE.test(h.text)) s += 0.02;
  return s;
}
function boostedSort(arr: any[]) {
  return arr.slice().sort((a, b) => boostScore(b) - boostScore(a));
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    const q = String(body?.q ?? "").trim();
    if (!q) return res.status(400).json({ error: "Missing q" });

    // Model selection (env or per-request)
    const MODEL = String(body?.model ?? DEFAULT_CHAT_MODEL);

    // ---- params (camel + snake accepted) ----
    const modeRaw = (body?.mode ?? "qa").toString().toLowerCase();
    const mode: "qa" | "article" | "tweet" =
      (["qa", "article", "tweet"].includes(modeRaw) ? modeRaw : "qa") as any;

    const topK       = Math.max(1, Math.min(12, toNumber(body?.topK ?? body?.top_k, 6)));
    const retrieveN  = Math.max(topK, Math.min(30, toNumber(body?.retrieveN ?? body?.retrieve_n, 24)));
    const minScore   = Number.isFinite(body?.minScore ?? body?.min_score)
      ? Number(body?.minScore ?? body?.min_score) : 0.15;

    const contextCharsCap = mode === "article" ? 60000 : 24000;
    const contextChars = Math.max(
      2000,
      Math.min(
        contextCharsCap,
        toNumber(body?.contextChars ?? body?.context_chars, mode === "article" ? 40000 : 6000),
      ),
    );

    // capture raw requested words
    const rawRequestedWords = toNumber(body?.targetWords ?? body?.target_words, NaN);

    // target words defaults & clamping
    let targetWords = toNumber(body?.targetWords ?? body?.target_words, undefined as any);
    if (mode === "article") {
      if (!Number.isFinite(targetWords)) targetWords = 4000;
      targetWords = clamp(targetWords!, 200, 6000);
    } else if (mode === "qa") {
      if (!Number.isFinite(targetWords)) targetWords = 200;
      targetWords = clamp(targetWords!, 120, 400);
    } else {
      targetWords = undefined as any; // tweet ignores words
    }

    // tweet-mode knobs
    const tweetMode = mode === "tweet";
    const evidenceRaw = String((body as any)?.evidence ?? (body as any)?.ctx ?? "balanced").toLowerCase();
    const evidenceMode = ["tight","balanced","loose"].includes(evidenceRaw) ? evidenceRaw : "balanced";

    const expandRaw = String((body as any)?.expand ?? "").toLowerCase();
    const expandMode = ["off","light","balanced","aggressive"].includes(expandRaw) ? expandRaw : "balanced";

    // Effective retrieval params
    let effMinScore = minScore;
    let effTopK     = topK;
    let effRetrieve = retrieveN;
    let effCtxCap   = contextChars;

    if (tweetMode) {
      if (evidenceMode === "tight") {
        effMinScore = Math.max(0.28, minScore);
        effTopK     = clamp(topK, 5, 6);
        effRetrieve = Math.max(effTopK, Math.min(24, retrieveN));
        effCtxCap   = clamp(contextChars, 6000, 9000);
      } else if (evidenceMode === "loose") {
        effMinScore = Math.max(0.15, Math.min(minScore, 0.20));
        effTopK     = clamp(topK, 8, 10);
        effRetrieve = Math.max(effTopK, Math.min(48, retrieveN));
        effCtxCap   = clamp(contextChars, 10000, 16000);
      } else {
        effMinScore = Math.max(0.20, minScore);
        effTopK     = clamp(topK, 6, 8);
        effRetrieve = Math.max(effTopK, Math.min(36, retrieveN));
        effCtxCap   = clamp(contextChars, 8000, 12000);
      }
    }

    // ---- 1) Embed query ----
    const emb = await openai.embeddings.create({ model: EMB_MODEL, input: q });
    const queryVec = emb.data[0].embedding as number[];

    // ---- 2) Vector search ----
    const raw = await qdrant.search(COLLECTION, {
      vector: queryVec,
      limit: tweetMode ? effRetrieve : retrieveN,
      with_payload: true,
      with_vector: false,
      score_threshold: 0.0,
    });

    // ---- 3) Tidy + base strong ----
    const hits = raw
      .map(hitMap)
      .filter((h) => h.text.length > 0)
      .sort((a, b) => b.score - a.score);

    let baseStrong = hits.filter((h) => h.score >= (tweetMode ? effMinScore : minScore));

    // ---- 3b) FACET EXPANSION (always for tweets unless expand=off) ----
    let expanded: any[] = [];
    if (tweetMode && expandMode !== "off") {
      const topics = topicHintsFromQ(q); // ["generic"]
      const perFacet = expandMode === "aggressive" ? 3 : (expandMode === "light" ? 1 : 2);
      const facetMin = Math.max(0.14, (tweetMode ? effMinScore : minScore) - 0.03);

      const groups = Array.from(new Set(topics));
      for (const g of groups) {
        const queries = FACET_GROUPS[g] || [];
        for (const fQuery of queries) {
          const subq = `${q} ${fQuery}`;
          const emb2 = await openai.embeddings.create({ model: EMB_MODEL, input: subq });
          const vec2 = emb2.data[0].embedding as number[];
          const raw2 = await qdrant.search(COLLECTION, {
            vector: vec2,
            limit: Math.max(perFacet * 3, 8),
            with_payload: true,
            with_vector: false,
            score_threshold: 0.0,
          });
          const cleaned2 = raw2.map(hitMap).filter(h => h.text.length > 0).sort((a,b)=>b.score-a.score);
          const strong2  = cleaned2.filter(h => h.score >= facetMin).slice(0, perFacet);
          expanded.push(...strong2);
        }
      }
      // merge & re-rank
      baseStrong = boostedSort(dedupeHits([...baseStrong, ...expanded]));
    }

    // ---- 4) Pack bounded context (tweet uses bucket; article/qa default) ----
    let returnedHits: any[] = [];
    let contextBlocks = "";

    if (tweetMode) {
      // split into pools
      const baseList  = boostedSort(baseStrong);
      const facetList = boostedSort(dedupeHits(expanded));

      // budget split 60% base / 40% facet
      const baseBudgetChars  = Math.floor(effCtxCap * 0.60);
      const facetBudgetChars = effCtxCap - baseBudgetChars;

      let usedChars = 0;
      let baseUsed = 0;
      let facetUsed = 0;
      const chosen: any[] = [];

      // fill base bucket
      for (const h of baseList) {
        const add = h.text.length + 300;
        if (baseUsed + add > baseBudgetChars) break;
        if (usedChars + add > effCtxCap) break;
        chosen.push(h);
        baseUsed += add;
        usedChars += add;
        if (chosen.length >= effTopK) break;
      }

      // then facet bucket
      const chosenKeys = new Set(chosen.map(keyOf));
      for (const h of facetList) {
        if (chosenKeys.has(keyOf(h))) continue;
        const add = h.text.length + 300;
        if (facetUsed + add > facetBudgetChars) break;
        if (usedChars + add > effCtxCap) break;
        chosen.push(h);
        chosenKeys.add(keyOf(h));
        facetUsed += add;
        usedChars += add;
        if (chosen.length >= effTopK) break;
      }

      // fill remaining from best of both
      if (chosen.length < effTopK) {
        const pool = boostedSort(
          dedupeHits([...baseList, ...facetList]).filter(h => !chosenKeys.has(keyOf(h)))
        );
        for (const h of pool) {
          const add = h.text.length + 300;
          if (usedChars + add > effCtxCap) break;
          chosen.push(h);
          chosenKeys.add(keyOf(h));
          usedChars += add;
          if (chosen.length >= effTopK) break;
        }
      }

      if (chosen.length === 0) {
        return res.status(200).json({
          ok: true,
          hits: hits.slice(0, Math.min(effTopK, hits.length)),
          answer: `No sufficiently relevant context found (minScore=${effMinScore}).`,
        });
      }

      returnedHits = chosen.slice(0, effTopK);
      contextBlocks = returnedHits
        .map((h, i) => {
          const tag = `[S${i + 1}]`;
          const header = `${tag} • ${h.title || h.source || "source"} • score=${h.score.toFixed(3)} • ${h.url || "no-url"}`;
          return `${header}\n${h.text}`;
        })
        .join("\n\n---\n\n");

      // optional diagnostics
      // @ts-ignore
      (globalThis as any).__tweet_ctx = { baseBudgetChars, facetBudgetChars };
    } else {
      // default non-tweet packing
      let used = 0;
      const chosen: any[] = [];
      for (const h of baseStrong) {
        const add = h.text.length + 300;
        if (used + add > contextChars) break;
        chosen.push(h);
        used += add;
      }
      returnedHits = chosen.slice(0, topK);
      contextBlocks = chosen
        .map((h, i) => {
          const tag = `[S${i + 1}]`;
          const header = `${tag} • ${h.title || h.source || "source"} • score=${h.score.toFixed(3)} • ${h.url || "no-url"}`;
          return `${header}\n${h.text}`;
        })
        .join("\n\n---\n\n");
    }

    // ---- 5) Prompts per mode ----
    const isArticle = mode === "article";
    const isTweet   = mode === "tweet";

    const lengthGuard = isTweet
      ? `Each tweet must be ≤280 characters.`
      : isArticle
      ? `Write about ${targetWords} words (±10%). Do NOT exceed ~${Math.round((targetWords as number) * 1.10)} words.`
      : `Answer in about ${targetWords} words (hard cap ~${Math.round((targetWords as number) * 1.10)}).`;

    const prompt = isTweet
      ? [
          `Write a set of tweets.`,
          `Obey the user's prompt exactly — audience, topic, tone, and style — above all else.`,
          `Format as a numbered list (1., 2., 3., …).`,
          `Use ONLY the "Context" below for facts; do not invent. Cite with [S#] if you reference the context.`,
          `Keep each tweet punchy, standalone, and ≤280 chars. One emoji max per tweet.`,
          lengthGuard,
          ``,
          `User Prompt: ${q}`,
          ``,
          `Context:\n${contextBlocks}`,
        ].join("\n")
      : isArticle
      ? [
          `You are writing a long-form article for a technical audience.`,
          `Use ONLY the "Context" below for facts. Do NOT invent or import outside knowledge.`,
          `If the Task requests details not in Context, omit them or briefly state: "I don't know from the provided context."`,
          `Cite sources with [S#] immediately after the specific claims they support.`,
          `Follow the Task exactly for subject, angle, audience, tone, language, structure, and length.`,
          `If the Task lacks structure, create a clear outline with H2/H3 headings, an introduction, body sections, and a short conclusion.`,
          `Aim for ~${targetWords} words (soft target; prefer the Task if it sets a length).`,
          lengthGuard,
          `Write in the same language as the Task.`,
          `Output Markdown only (no preface, no metadata, no extra commentary).`,
          ``,
          `Task: ${q}`,
          ``,
          `Context:\n${contextBlocks}`,
        ].join("\n")
      : [
          `Answer using ONLY the "Context" below. If not found, say "I don't know from the provided context."`,
          `Cite with [S#]. Be concise, 3–6 sentences.`,
          lengthGuard,
          ``,
          `Question: ${q}`,
          ``,
          `Context:\n${contextBlocks}`,
        ].join("\n");

    // ---- 6) Generation (4o primary → 4o-mini fallback; 5.x via responses) ----
    let maxTokens = isTweet
      ? 480
      : isArticle
      ? clamp(wordsToTokens((targetWords as number) * 1.12), 512, 12000)
      : clamp(wordsToTokens((targetWords as number) * 1.05), 256, 2000);

    let answer = "";
    let finishReason: string | null = null;
    let modelUsed = MODEL;
    let fallbackReason: string | null = null;

    async function chat4oMini(msg: string, mt: number) {
      const chat = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: isTweet ? "You produce one or more tweets (numbered list), each ≤280 chars." : "You answer briefly and cite [S#]." },
          { role: "user", content: msg },
        ],
        temperature: isArticle ? 0.3 : 0.1,
        max_tokens: mt,
      });
      const raw = chat.choices?.[0]?.message?.content as any;
      const text = typeof raw === "string" ? raw : normalizeParts(raw);
      return { text, reason: chat.choices?.[0]?.finish_reason ?? null };
    }

    // Primary try
    try {
      if (isGpt5(MODEL)) {
        const resp = await openai.responses.create({
          model: MODEL,
          input: prompt,
          max_output_tokens: maxTokens,
        });
        // @ts-ignore
        answer = (resp as any).output_text ?? extractResponseText(resp);
        // @ts-ignore
        finishReason = (resp as any)?.output?.[0]?.finish_reason ?? null;
      } else {
        // Treat whatever came in (default 4o) as primary
        const chat = await openai.chat.completions.create({
          model: MODEL,
          messages: [{ role: "user", content: prompt }],
          temperature: isArticle ? 0.3 : 0.1,
          max_tokens: maxTokens,
        });
        const rawChat = chat.choices?.[0]?.message?.content as any;
        answer = typeof rawChat === "string" ? rawChat : normalizeParts(rawChat);
        finishReason = chat.choices?.[0]?.finish_reason ?? null;
      }
    } catch (err: any) {
      fallbackReason = `primary_failed:${String(err?.message || err)}`;
    }

    // Fallback if empty or error
    if (!answer || !answer.trim()) {
      const r = await chat4oMini(prompt, maxTokens);
      answer = r.text;
      finishReason = r.reason;
      modelUsed = "gpt-4o-mini";
      if (!fallbackReason) fallbackReason = "empty_primary_output";
    }

    // tweet-specific lite fallback if still empty
    // (extremely unlikely; keep for extra resilience)
    if (isTweet && (!answer || !answer.trim())) {
      const top2 = returnedHits.slice(0, 2).map((h, i) => {
        const tag = `[S${i + 1}]`;
        const head = `${tag} • ${h.title || h.source || "source"} • ${h.url || "no-url"}`;
        const body = (h.text || "").slice(0, 1000);
        return `${head}\n${body}`;
      }).join("\n\n---\n\n");

      const litePrompt = [
        `Write a set of tweets.`,
        `Obey the user's prompt exactly — audience, topic, tone, and style — above all else.`,
        `Format as a numbered list (1., 2., 3., …).`,
        `Use ONLY the snippets below; one emoji max per tweet; do not invent facts.`,
        `Keep each tweet ≤280 chars.`,
        ``,
        `User Prompt: ${q}`,
        ``,
        `Context:\n${top2}`,
      ].join("\n");

      const r2 = await chat4oMini(litePrompt, 360);
      answer = r2.text || "I couldn't generate tweets from the provided context.";
      finishReason = r2.reason;
      modelUsed = "gpt-4o-mini";
      if (!fallbackReason) fallbackReason = "tweet_lite_fallback";
    }

    // tweet clamp
    if (isTweet) answer = clampTweetsList(answer);

    // --- continuation + body-only enforcement for article/qa (ALWAYS FINISH) ---
    if (!isTweet && Number.isFinite(targetWords as any)) {
      const splitOnSources = /^---\s*Sources:/m;
      let bodyOnly = answer?.trim() || "";
      let sourcesBlock = "";

      // Split out any sources block if present
      if (splitOnSources.test(bodyOnly)) {
        const idx = bodyOnly.search(splitOnSources);
        bodyOnly = bodyOnly.slice(0, idx).trim();
        sourcesBlock = answer.slice(idx);
      }

      // 1) Expand body until ~target hit OR we detect a conclusion.
      const TARGET = targetWords as number;
      const MIN_RATIO = 0.98;         // push closer to target
      const MAX_LOOPS = 6;            // more aggressive than before
      let wc = countWords(bodyOnly);
      let loops = 0;

      while (
        (wc < Math.round(TARGET * MIN_RATIO) || !hasConclusion(bodyOnly)) &&
        loops < MAX_LOOPS
      ) {
        const missingWords = Math.max(120, TARGET - wc);
        const askFor = clamp(Math.round(missingWords * 0.75), 120, 1200);

        const continuePrompt = [
          `Continue the following ${mode} from where it stopped.`,
          `Do NOT repeat previous text, headings, intro, or already-covered points.`,
          `Write about ${askFor} more words (±10%).`,
          `Keep the same tone/structure. Use H2/H3 as needed. Keep [S#] citations where applicable.`,
          `If a conclusion already exists, do NOT add another; otherwise add a short conclusion at the end.`,
          `Return ONLY the continuation (no title, no preface, no Sources section).`,
          ``,
          `Task: ${q}`,
          ``,
          `Context:\n${contextBlocks}`,
          ``,
          `Article so far (tail):\n${tail(bodyOnly, 2000)}`
        ].join("\n");

        const mt = clamp(wordsToTokens(askFor * 1.20), 256, 6000);
        let cont = "";

        try {
          if (isGpt5(MODEL)) {
            const r = await openai.responses.create({
              model: modelUsed || MODEL,
              input: continuePrompt,
              max_output_tokens: mt,
            });
            // @ts-ignore
            cont = (r as any).output_text ?? extractResponseText(r);
          } else {
            const c = await openai.chat.completions.create({
              model: modelUsed || MODEL, // if we already fell back, keep that model
              messages: [{ role: "user", content: continuePrompt }],
              temperature: isArticle ? 0.3 : 0.1,
              max_tokens: mt,
            });
            const rawC = c.choices?.[0]?.message?.content as any;
            cont = typeof rawC === "string" ? rawC : normalizeParts(rawC);
          }
        } catch (e: any) {
          // try mini one time for the continuation if primary fails
          const cMini = await chat4oMini(continuePrompt, mt);
          cont = cMini.text || "";
          if (!fallbackReason) fallbackReason = `continue_failed_used_mini:${String(e?.message || e)}`;
          modelUsed = "gpt-4o-mini";
        }

        if (!cont || cont.trim().length < 50) break;

        bodyOnly = (bodyOnly + "\n\n" + cont.trim()).trim();
        wc = countWords(bodyOnly);
        loops++;
      }

      // 2) Finalization pass — guarantee a conclusion
      if (!hasConclusion(bodyOnly)) {
        const finalizePrompt = [
          `Write a SHORT concluding section (3–6 sentences) that wraps the article naturally.`,
          `Do NOT repeat prior sentences verbatim. No new facts beyond Context.`,
          `Use the same language as the Task and keep it technical yet accessible.`,
          `Return ONLY the conclusion (no title, no Sources).`,
          ``,
          `Task: ${q}`,
          ``,
          `Context:\n${contextBlocks}`,
          ``,
          `Article so far (tail):\n${tail(bodyOnly, 1600)}`
        ].join("\n");

        const mt2 = 320;
        let concl = "";
        try {
          if (isGpt5(MODEL)) {
            const r = await openai.responses.create({
              model: modelUsed || MODEL,
              input: finalizePrompt,
              max_output_tokens: mt2,
            });
            // @ts-ignore
            concl = (r as any).output_text ?? extractResponseText(r);
          } else {
            const c = await openai.chat.completions.create({
              model: modelUsed || MODEL,
              messages: [{ role: "user", content: finalizePrompt }],
              temperature: 0.2,
              max_tokens: mt2,
            });
            const rawC = c.choices?.[0]?.message?.content as any;
            concl = typeof rawC === "string" ? rawC : normalizeParts(rawC);
          }
        } catch (e: any) {
          const cMini2 = await chat4oMini(finalizePrompt, mt2);
          concl = cMini2.text || "";
          if (!fallbackReason) fallbackReason = `finalize_failed_used_mini:${String(e?.message || e)}`;
          modelUsed = "gpt-4o-mini";
        }
        if (concl && concl.trim().length > 40) {
          bodyOnly = (bodyOnly.replace(/\s+$/, "") + "\n\n" + concl.trim()).trim();
        }
      }

      // 3) Respect word target cap
      const TARGET = targetWords as number;
      bodyOnly = enforceWordTarget(bodyOnly, TARGET);

      // Reattach sources if they existed
      answer = sourcesBlock ? `${bodyOnly}\n${sourcesBlock}` : bodyOnly;

      // @ts-ignore
      (globalThis as any).__len_meta = { loops: 0, wc_after: countWords(bodyOnly) };
    }

    // ---- 7) Citations map ----
    const citations = returnedHits.map((h, i) => ({
      tag: `S${i + 1}`,
      title: h.title || h.source || "source",
      url: h.url,
      score: h.score,
    }));

    return res.status(200).json({
      ok: true,
      hits: returnedHits,
      answer,
      citations,
      meta: {
        mode,
        modelRequested: MODEL,
        modelUsed,
        fallbackReason, // <—— visible reason if we used mini or had to recover
        retrieved: hits.length,
        chosen: returnedHits.length,
        minScore: tweetMode ? effMinScore : minScore,
        contextChars: tweetMode ? effCtxCap : contextChars,
        topK: tweetMode ? effTopK : topK,
        retrieveN: tweetMode ? effRetrieve : retrieveN,
        evidenceMode: tweetMode ? evidenceMode : undefined,
        expandMode: tweetMode ? expandMode : undefined,
        maxTokens,
        targetWords,
        requestedWords: rawRequestedWords,
        continuations: (globalThis as any).__len_meta?.loops ?? 0,
        bodyWordCount: countWords((answer.split(/^---\s*Sources:/m)[0] || "")),
        finishReason,
      },
    });
  } catch (err: any) {
    console.error("QUERY_ERROR:", err?.message || err);
    return res.status(500).json({ error: "QUERY_ERROR", detail: String(err?.message || err) });
  }
}
