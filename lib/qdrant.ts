import { QdrantClient } from "@qdrant/js-client-rest";

export const COLLECTION = process.env.QDRANT_COLLECTION || "vara_wire_rag";

export const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL!,
  apiKey: process.env.QDRANT_API_KEY!,
});

export async function ensureCollection(vectorSize: number) {
  const collections = await qdrant.getCollections();
  const exists = collections.collections?.some(c => c.name === COLLECTION);
  if (!exists) {
    await qdrant.createCollection(COLLECTION, {
      vectors: { size: vectorSize, distance: "Cosine" }
    });
  }
}
