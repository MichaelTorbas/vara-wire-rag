export type CorpusItem = {
  id?: string;                 // optional client-supplied id
  title: string;
  url?: string;
  source?: string;             // WireNetwork, GearProtocol, etc.
  tags?: string[];             // ["runtime","bridge","ethereum"]
  date?: string;               // ISO date string
  text: string;                // full, cleaned article text
};

export type IndexedChunk = {
  id: string;
  text: string;
  metadata: Omit<CorpusItem, "text"> & {
    doc_id: string;
    chunk_index: number;
  };
};

export type GenerateMode = "tweet" | "article";
