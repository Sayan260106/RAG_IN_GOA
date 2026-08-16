export interface RetrievedChunk {
  id: string;
  chunkNumber: number;
  content: string;
  source: string;
  category: string;

  similarityScore: number;

  bm25Score?: number;
  vectorSimilarity?: number;
  rrfScore?: number;
  rerankScore?: number;

  keywords?: string[];
  docTitle?: string;
}


export interface DocumentItem {
  id: string;
  title: string;
  category: string;
  source: string;
  content: string;
  chunkCount: number;
  createdAt: string;
  isCustom?: boolean;
}


/**
 * Response returned by the FastAPI backend.
 *
 * Backend endpoint:
 * POST /api/rag
 */
export interface BackendRAGResponse {
  transcript: string;
  answer: string;
  confidence: number;
  latencyMs: number;

  groundingScore: number;
  modelEngine: string;
  llmProvider?: string;
  isGrounded?: boolean;
  guardrailStatus?: "VERIFIED_GROUNDED" | "FLAGGED_LOW_SIMILARITY" | "FLAGGED_LOW_GROUNDING";
  guardrailWarning?: string | null;
  retrievalBreakdown?: {
    queryTokens: string[];
    bm25TopScore: number;
    vectorTopScore: number;
    totalDocsIndexed: number;
    totalChunksIndexed: number;
    searchMethod: string;
  };

  retrievedChunks: RetrievedChunk[];

  timestamp: string;
}


/**
 * Normalized response used by the React frontend.
 *
 * App.tsx works with this structure after
 * converting the backend response.
 */
export interface RAGResponse {
  query: string;
  answer: string;

  confidence: number;
  originalConfidence?: number;

  latencyMs: number;

  sourceFile: string;
  indexRef: string;

  chunks: RetrievedChunk[];

  minSimilarity: number;
  isGrounded: boolean;

  guardrailStatus:
    | 'VERIFIED_GROUNDED'
    | 'FLAGGED_LOW_SIMILARITY';

  guardrailWarning: string | null;

  retrievalBreakdown?: {
    queryTokens: string[];
    bm25TopScore: number;
    vectorTopScore: number;
    totalDocsIndexed: number;
    totalChunksIndexed: number;
    searchMethod: string;
  };

  stats?: {
    p50?: number;
    p75?: number;
    p100?: number;
    avg?: number;
    min?: number;
    max?: number;
    total?: number;
    budgetLimitMs?: number;
    underBudgetRatio?: number;
  };

  /**
   * Information directly returned by the backend.
   */
  groundingScore?: number;
  modelEngine?: string;
  timestamp?: string;
}


/**
 * Knowledge-base document response.
 */
export interface DocumentsResponse {
  documents: DocumentItem[];
  totalChunks: number;
}


/**
 * Benchmark response.
 */
export interface BenchmarkResponse {
  recent_queries?: LatencyQueryRecord[];
  stats?: {
    p50?: number;
    p75?: number;
    p100?: number;
    avg?: number;
    min?: number;
    max?: number;
    total?: number;
    budgetLimitMs?: number;
    underBudgetRatio?: number;
  };
}


/**
 * Frontend telemetry log.
 */
export interface PipelineLog {
  timestamp: string;
  level: string;
  component: string;
  message: string;
  color?: string;
}


/**
 * Latency history record.
 */
export interface LatencyQueryRecord {
  id: string;
  queryNumber: number;
  query: string;

  latencyMs: number;

  p50Benchmark: number;
  p75Benchmark: number;
  p100Benchmark: number;

  confidence: number;
  minSimilarity: number;
  isGrounded: boolean;

  timestamp: string;
}


/**
 * Latency statistics.
 */
export interface LatencyStats {
  p50: number;
  p75: number;
  p100: number;

  avg: number;
  min: number;
  max: number;

  total: number;

  budgetLimitMs: number;
  underBudgetRatio: number;
}