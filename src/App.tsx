/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
} from "react";

import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Share2,
  Download,
  Sparkles,
  Search,
  Check,
  AlertTriangle,
  ShieldCheck,
  Activity,
  Loader2,
  Send,
  Edit3,
  X,
  Layers,
  Database,
  Cpu,
} from "lucide-react";

import {
  LatencyMetricsModal,
  LatencyQueryRecord,
  LatencyStats,
} from "./components/LatencyMetricsModal";

import { KnowledgeBaseModal } from "./components/KnowledgeBaseModal";
import { PipelineInspector } from "./components/PipelineInspector";

import {
  RetrievedChunk,
  DocumentItem,
  RAGResponse,
  BackendRAGResponse,
  PipelineLog,
} from "./types";


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const STOP_WORDS = new Set([
  "what",
  "is",
  "are",
  "the",
  "in",
  "of",
  "and",
  "a",
  "an",
  "for",
  "to",
  "how",
  "do",
  "i",
  "on",
  "at",
  "by",
  "with",
  "from",
  "about",
  "why",
  "when",
  "where",
  "which",
  "can",
  "you",
  "me",
  "tell",
  "explain",
  "give",
  "its",
  "their",
  "there",
  "that",
  "this",
  "these",
  "those",
  "some",
  "any",
  "does",
  "been",
  "being",
  "have",
  "has",
  "details",
]);


// ============================================================
// HELPERS
// ============================================================

function extractKeywords(query: string): string[] {
  if (!query) return [];

  const words = query
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(
      (word) => word.length > 2 && !STOP_WORDS.has(word)
    );

  return Array.from(new Set(words));
}


function normalizeChunk(
  chunk: any,
  index: number
): RetrievedChunk {
  return {
    id:
      chunk?.id ||
      `chunk-${index + 1}-${Date.now()}`,

    chunkNumber:
      typeof chunk?.chunkNumber === "number"
        ? chunk.chunkNumber
        : index + 1,

    content: chunk?.content || "",

    source:
      chunk?.source ||
      chunk?.metadata?.source ||
      "Unknown Source",

    category:
      chunk?.category ||
      chunk?.metadata?.category ||
      "general",

    similarityScore:
      typeof chunk?.similarityScore === "number"
        ? chunk.similarityScore
        : typeof chunk?.similarity === "number"
        ? chunk.similarity
        : typeof chunk?.score === "number"
        ? chunk.score
        : 0,

    bm25Score:
      typeof chunk?.bm25Score === "number"
        ? chunk.bm25Score
        : undefined,

    vectorSimilarity:
      typeof chunk?.vectorSimilarity === "number"
        ? chunk.vectorSimilarity
        : undefined,

    rrfScore:
      typeof chunk?.rrfScore === "number"
        ? chunk.rrfScore
        : undefined,

    rerankScore:
      typeof chunk?.rerankScore === "number"
        ? chunk.rerankScore
        : undefined,

    keywords:
      Array.isArray(chunk?.keywords)
        ? chunk.keywords
        : [],

    docTitle:
      chunk?.docTitle ||
      chunk?.title ||
      chunk?.metadata?.title ||
      undefined,
  };
}


function normalizeBackendResponse(
  data: BackendRAGResponse,
  query: string
): RAGResponse {
  const rawChunks = Array.isArray(data.retrievedChunks)
    ? data.retrievedChunks
    : [];

  const chunks = rawChunks.map(normalizeChunk);

  const calculatedMinSimilarity =
    chunks.length > 0
      ? Math.min(
          ...chunks.map((chunk) =>
            typeof chunk.similarityScore === "number"
              ? chunk.similarityScore
              : 0
          )
        )
      : data.groundingScore || 0;

  const minSimilarity =
    chunks.length > 0
      ? calculatedMinSimilarity
      : data.groundingScore || 0;

  const isGrounded =
    typeof data.isGrounded === "boolean"
      ? data.isGrounded
      : minSimilarity >= 0.7;

  return {
    query:
      data.transcript ||
      query,

    answer:
      data.answer ||
      "No answer was generated.",

    confidence:
      typeof data.confidence === "number"
        ? data.confidence
        : 0,

    originalConfidence: data.confidence,

    latencyMs:
      typeof data.latencyMs === "number"
        ? data.latencyMs
        : 0,

    sourceFile:
      "src/rag/orchestration/orchestrator.py",

    indexRef:
      "FAISS + BM25 Hybrid RRF",

    chunks,

    minSimilarity,

    isGrounded,

    guardrailStatus:
      (data.guardrailStatus === "VERIFIED_GROUNDED" ||
       data.guardrailStatus === "FLAGGED_LOW_SIMILARITY"
        ? data.guardrailStatus
        : isGrounded
        ? "VERIFIED_GROUNDED"
        : "FLAGGED_LOW_SIMILARITY"),

    guardrailWarning:
      data.guardrailWarning ??
      (isGrounded
        ? null
        : `Warning: minimum chunk similarity (${Math.round(
            minSimilarity * 100
          )}%) is below the 70% grounding threshold.`),

    groundingScore:
      typeof data.groundingScore === "number"
        ? data.groundingScore
        : data.confidence,

    modelEngine:
      data.modelEngine || "unknown",

    timestamp:
      data.timestamp,

    retrievalBreakdown: {
      queryTokens: extractKeywords(query),

      bm25TopScore:
        chunks.length > 0
          ? Math.max(
              ...chunks.map((c) => c.bm25Score || 0)
            )
          : 0,

      vectorTopScore:
        chunks.length > 0
          ? Math.max(
              ...chunks.map(
                (c) => c.vectorSimilarity || c.similarityScore || 0
              )
            )
          : 0,

      totalDocsIndexed:
        data.retrievalBreakdown?.totalDocsIndexed || 0,

      totalChunksIndexed:
        data.retrievalBreakdown?.totalChunksIndexed || chunks.length,

      searchMethod:
        data.retrievalBreakdown?.searchMethod ||
        "FAISS + BM25 + RRF",
    },
  };
}


function HighlightedChunkContent({
  content,
  query,
}: {
  content: string;
  query: string;
}) {
  const keywords = useMemo(
    () => extractKeywords(query),
    [query]
  );

  if (keywords.length === 0) {
    return <span>{content}</span>;
  }

  const escaped = keywords
    .map((keyword) =>
      keyword.replace(
        /[.*+?^${}()|[\]\\]/g,
        "\\$&"
      )
    )
    .join("|");

  const regex = new RegExp(
    `(${escaped})`,
    "gi"
  );

  const parts = content.split(regex);

  return (
    <span>
      {parts.map((part, index) => {
        const isMatch = keywords.some(
          (keyword) =>
            keyword.toLowerCase() ===
            part.toLowerCase()
        );

        if (isMatch) {
          return (
            <mark
              key={index}
              className="bg-orange-500/30 text-orange-200 px-1 py-0.5 rounded font-semibold border border-orange-500/40 inline-block my-0.5"
            >
              {part}
            </mark>
          );
        }

        return (
          <span key={index}>
            {part}
          </span>
        );
      })}
    </span>
  );
}


// ============================================================
// SAMPLE QUERIES
// ============================================================

const SAMPLE_QUERIES = [
  "What is Artificial Intelligence and how does it work?",
  "What are the primary factors affecting monsoon patterns in North Goa?",
  "What is the history of Basilica of Bom Jesus in Old Goa?",
  "What spices and ingredients are essential for authentic Goan Fish Curry?",
  "How do I visit Dudhsagar Falls and what is the best season?",
  "What makes Fontainhas Latin Quarter unique in Asia?",
  "How do rockets reach orbit around Mars? (Unrelated Query Test)",
  "What is quantum entanglement in physics? (Unrelated Query Test)",
];


// ============================================================
// APP
// ============================================================

export default function App() {

  // ----------------------------------------------------------
  // QUERY STATE
  // ----------------------------------------------------------

  const [currentQuery, setCurrentQuery] = useState(
    "What is Artificial Intelligence and how does it work?"
  );

  const [typedInput, setTypedInput] =
    useState("");

  const [editableQuery, setEditableQuery] =
    useState(
      "What is Artificial Intelligence and how does it work?"
    );

  const [isEditingInline, setIsEditingInline] =
    useState(false);


  // ----------------------------------------------------------
  // RAG STATE
  // ----------------------------------------------------------

  const [isProcessing, setIsProcessing] =
    useState(false);

  const [lastResponse, setLastResponse] =
    useState<RAGResponse | null>(null);

  const [chunks, setChunks] =
    useState<RetrievedChunk[]>([]);

  const [answer, setAnswer] =
    useState("");

  const [confidence, setConfidence] =
    useState(0);

  const [latencyMs, setLatencyMs] =
    useState(0);

  const [sourceFile, setSourceFile] =
    useState(
      "src/rag/orchestration/orchestrator.py"
    );

  const [indexRef, setIndexRef] =
    useState(
      "FAISS + BM25 Hybrid RRF"
    );


  // ----------------------------------------------------------
  // GUARDRAILS
  // ----------------------------------------------------------

  const [minSimilarity, setMinSimilarity] =
    useState(0);

  const [isGrounded, setIsGrounded] =
    useState(false);

  const [guardrailWarning, setGuardrailWarning] =
    useState<string | null>(null);


  // ----------------------------------------------------------
  // KNOWLEDGE BASE
  // ----------------------------------------------------------

  const [documents, setDocuments] =
    useState<DocumentItem[]>([]);

  const [totalChunks, setTotalChunks] =
    useState(0);

  const [isLoadingDocs, setIsLoadingDocs] =
    useState(false);


  // ----------------------------------------------------------
  // MODALS
  // ----------------------------------------------------------

  const [showKBModal, setShowKBModal] =
    useState(false);

  const [showInspectorModal, setShowInspectorModal] =
    useState(false);

  const [showLatencyModal, setShowLatencyModal] =
    useState(false);


  // ----------------------------------------------------------
  // LATENCY
  // ----------------------------------------------------------

  const [queryHistory, setQueryHistory] =
    useState<LatencyQueryRecord[]>([]);

  const [latencyStats, setLatencyStats] =
    useState<LatencyStats>({
      p50: 0,
      p75: 0,
      p100: 0,
      avg: 0,
      min: 0,
      max: 0,
      total: 0,
      budgetLimitMs: 200,
      underBudgetRatio: 0,
    });


  // ----------------------------------------------------------
  // AUDIO
  // ----------------------------------------------------------

  const [isRecording, setIsRecording] =
    useState(false);

  const [isSpeaking, setIsSpeaking] =
    useState(false);

  const [copiedState, setCopiedState] =
    useState(false);

  const [permissionError, setPermissionError] =
    useState<string | null>(null);

  const [statusMessage, setStatusMessage] =
    useState(
      "Dynamic RAG Engine Ready"
    );

  const [audioVolume, setAudioVolume] =
    useState(0);

  const [waveformLevels, setWaveformLevels] =
    useState([15, 25, 40, 25, 15]);


  // ----------------------------------------------------------
  // LOGS
  // ----------------------------------------------------------

  const [logs, setLogs] =
    useState<PipelineLog[]>([
      {
        timestamp:
          new Date().toISOString(),

        level: "INFO",

        component:
          "src.rag.init",

        message:
          "HHGoa Voice RAG frontend initialized.",

        color:
          "text-orange-400",
      },

      {
        timestamp:
          new Date().toISOString(),

        level: "INFO",

        component:
          "src.rag.guardrails",

        message:
          "Grounding verification threshold: 70%.",

        color:
          "text-emerald-400",
      },

      {
        timestamp:
          new Date().toISOString(),

        level: "INFO",

        component:
          "src.rag.asr",

        message:
          "Web Speech Recognition ready.",

        color:
          "text-white/60",
      },
    ]);


  // ----------------------------------------------------------
  // AUDIO REFS
  // ----------------------------------------------------------

  const recognitionRef =
    useRef<any>(null);

  const audioContextRef =
    useRef<AudioContext | null>(null);

  const mediaStreamRef =
    useRef<MediaStream | null>(null);

  const analyserRef =
    useRef<AnalyserNode | null>(null);

  const animFrameRef =
    useRef<number | null>(null);

  const isRecordingRef =
    useRef(false);

  const liveTranscriptRef =
    useRef("");


  // ==========================================================
  // LOGGING
  // ==========================================================

  const addLog = useCallback(
    (
      level: string,
      component: string,
      message: string,
      color?: string
    ) => {
      const newLog: PipelineLog = {
        timestamp:
          new Date().toISOString(),

        level,
        component,
        message,
        color,
      };

      setLogs((previous) => [
        newLog,
        ...previous.slice(0, 49),
      ]);
    },
    []
  );


  // ==========================================================
  // FETCH DOCUMENTS
  // ==========================================================

  const fetchDocuments = useCallback(
    async () => {
      setIsLoadingDocs(true);

      try {
        const response = await fetch(
          `${API_BASE_URL}/api/documents`
        );

        if (!response.ok) {
          throw new Error(
            `HTTP ${response.status}`
          );
        }

        const data =
          await response.json();

        setDocuments(
          data.documents || []
        );

        setTotalChunks(
          data.totalChunks || 0
        );

      } catch (error) {

        console.warn(
          "Knowledge base endpoint unavailable:",
          error
        );

        addLog(
          "WARN",
          "src.rag.documents",
          "Knowledge-base endpoint is unavailable. RAG querying remains available.",
          "text-amber-400"
        );

      } finally {
        setIsLoadingDocs(false);
      }
    },
    [addLog]
  );


  // ==========================================================
  // INITIAL LOAD
  // ==========================================================

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);


  // ==========================================================
  // MAIN RAG QUERY
  // ==========================================================

  const executeQuery = useCallback(
    async (queryText: string) => {

      if (!queryText.trim()) {
        return;
      }

      const clean =
        queryText.trim();

      setCurrentQuery(clean);
      setEditableQuery(clean);
      setIsProcessing(true);

      setStatusMessage(
        "Executing Hybrid RAG Pipeline..."
      );

      const startTime =
        performance.now();

      addLog(
        "INFO",
        "src.rag.query",
        `Processing query: "${clean}"`,
        "text-orange-400"
      );

      try {

        // ----------------------------------------------------
        // CALL FASTAPI
        // ----------------------------------------------------

        const response =
          await fetch(
            `${API_BASE_URL}/api/rag`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body: JSON.stringify({
                query: clean,
              }),
            }
          );


        if (!response.ok) {

          const errorText =
            await response.text();

          throw new Error(
            `Backend returned ${response.status}: ${errorText}`
          );
        }


        // ----------------------------------------------------
        // READ BACKEND RESPONSE
        // ----------------------------------------------------

        const backendData:
          BackendRAGResponse =
            await response.json();


        // ----------------------------------------------------
        // NORMALIZE RESPONSE
        // ----------------------------------------------------

        const normalized =
          normalizeBackendResponse(
            backendData,
            clean
          );


        setLastResponse(
          normalized
        );

        setChunks(
          normalized.chunks
        );

        setAnswer(
          normalized.answer
        );

        setConfidence(
          normalized.confidence
        );

        setLatencyMs(
          normalized.latencyMs ||
            Math.round(
              performance.now() -
                startTime
            )
        );

        setSourceFile(
          normalized.sourceFile
        );

        setIndexRef(
          normalized.indexRef
        );

        setMinSimilarity(
          normalized.minSimilarity
        );

        setIsGrounded(
          normalized.isGrounded
        );

        setGuardrailWarning(
          normalized.guardrailWarning
        );


        // ----------------------------------------------------
        // TOTAL CHUNKS
        // ----------------------------------------------------

        if (
          normalized.retrievalBreakdown
        ) {
          setTotalChunks(
            normalized
              .retrievalBreakdown
              .totalChunksIndexed
          );
        }


        // ----------------------------------------------------
        // LOGGING
        // ----------------------------------------------------

        addLog(
          "INFO",
          "src.rag.retrieval",
          `Retrieved ${normalized.chunks.length} context chunks.`,
          "text-teal-400"
        );

        addLog(
          "INFO",
          "src.rag.model",
          `Model engine: ${
            normalized.modelEngine ||
            "Backend RAG"
          }`,
          "text-orange-400"
        );

        if (
          normalized.isGrounded
        ) {

          addLog(
            "SUCCESS",
            "src.rag.guardrails",
            `Grounding verified: ${Math.round(
              normalized.minSimilarity * 100
            )}% minimum similarity.`,
            "text-emerald-400"
          );

        } else {

          addLog(
            "WARN",
            "src.rag.guardrails",
            normalized.guardrailWarning ||
              "Low similarity detected.",
            "text-amber-400"
          );
        }


        addLog(
          "SUCCESS",
          "src.rag.monitoring",
          `RAG response completed in ${normalized.latencyMs}ms.`,
          "text-orange-400"
        );


        // ----------------------------------------------------
        // LATENCY HISTORY
        // ----------------------------------------------------

        const elapsed =
          normalized.latencyMs ||
          Math.round(
            performance.now() -
              startTime
          );

        const record:
          LatencyQueryRecord = {
            id:
              `query-${Date.now()}`,

            queryNumber:
              queryHistory.length + 1,

            query: clean,

            latencyMs: elapsed,

            p50Benchmark:
              latencyStats.p50,

            p75Benchmark:
              latencyStats.p75,

            p100Benchmark:
              latencyStats.p100,

            confidence:
              normalized.confidence,

            minSimilarity:
              normalized.minSimilarity,

            isGrounded:
              normalized.isGrounded,

            timestamp:
              new Date().toISOString(),
          };

        setQueryHistory(
          (previous) => [
            ...previous.slice(-49),
            record,
          ]
        );


        setStatusMessage(
          `RAG Complete • ${elapsed}ms`
        );

      } catch (error) {

        console.error(
          "RAG request failed:",
          error
        );

        const elapsed =
          Math.round(
            performance.now() -
              startTime
          );


        setStatusMessage(
          "RAG backend connection failed"
        );


        addLog(
          "ERROR",
          "src.rag.api",
          `Unable to reach FastAPI backend: ${
            error instanceof Error
              ? error.message
              : "Unknown error"
          }`,
          "text-rose-400"
        );


        setAnswer(
          "Unable to connect to the RAG backend. Please make sure the FastAPI server is running on port 8000."
        );

        setChunks([]);

        setConfidence(0);

        setLatencyMs(
          elapsed
        );

        setMinSimilarity(0);

        setIsGrounded(false);

        setGuardrailWarning(
          "Backend connection failed."
        );

      } finally {

        setIsProcessing(false);
      }
    },
    [
      addLog,
      latencyStats,
      queryHistory.length,
    ]
  );


  // ==========================================================
  // INITIAL QUERY
  // ==========================================================

  useEffect(() => {

    executeQuery(
      "What is Artificial Intelligence and how does it work?"
    );

  }, [executeQuery]);


  // ==========================================================
  // DOCUMENT INGEST
  // ==========================================================

  const handleIngestDocument =
    async (doc: {
      title: string;
      category: string;
      source: string;
      content: string;
    }) => {

      try {

        const response =
          await fetch(
            `${API_BASE_URL}/api/documents`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body: JSON.stringify(doc),
            }
          );

        if (!response.ok) {
          return false;
        }

        await fetchDocuments();

        addLog(
          "SUCCESS",
          "src.rag.ingestion",
          `Indexed document: "${doc.title}"`,
          "text-teal-400"
        );

        return true;

      } catch (error) {

        console.error(
          "Document ingestion failed:",
          error
        );

        return false;
      }
    };


  // ==========================================================
  // RESET CORPUS
  // ==========================================================

  const handleResetCorpus =
    async () => {

      try {

        const response =
          await fetch(
            `${API_BASE_URL}/api/documents/reset`,
            {
              method: "POST",
            }
          );

        if (!response.ok) {
          throw new Error(
            `HTTP ${response.status}`
          );
        }

        await fetchDocuments();

        addLog(
          "INFO",
          "src.rag.ingestion",
          "Corpus reset completed.",
          "text-amber-400"
        );

      } catch (error) {

        console.error(
          "Corpus reset failed:",
          error
        );
      }
    };


  // ==========================================================
  // AUDIO CLEANUP
  // ==========================================================

  const stopAudioCapture =
    useCallback(() => {

      if (animFrameRef.current) {

        cancelAnimationFrame(
          animFrameRef.current
        );

        animFrameRef.current =
          null;
      }


      if (mediaStreamRef.current) {

        mediaStreamRef.current
          .getTracks()
          .forEach(
            (track) =>
              track.stop()
          );

        mediaStreamRef.current =
          null;
      }


      if (
        audioContextRef.current &&
        audioContextRef.current
          .state !== "closed"
      ) {

        try {
          audioContextRef.current.close();
        } catch {}

        audioContextRef.current =
          null;
      }


      setAudioVolume(0);

      setWaveformLevels(
        [15, 25, 40, 25, 15]
      );

    }, []);


  // ==========================================================
  // STOP RECORDING
  // ==========================================================

  const stopRecording =
    useCallback(() => {

      setIsRecording(false);

      isRecordingRef.current =
        false;


      if (recognitionRef.current) {

        try {
          recognitionRef.current.stop();
        } catch {}
      }


      stopAudioCapture();


      const finalQuery =
        liveTranscriptRef.current.trim();


      if (finalQuery) {

        executeQuery(
          finalQuery
        );

      } else {

        setStatusMessage(
          "Voice input completed • Ready"
        );
      }

    }, [
      stopAudioCapture,
      executeQuery,
    ]);


  // ==========================================================
  // START RECORDING
  // ==========================================================

  const startRecording =
    async () => {

      setPermissionError(null);

      liveTranscriptRef.current =
        "";


      const SpeechRecognition =
        (window as any)
          .SpeechRecognition ||
        (window as any)
          .webkitSpeechRecognition;


      if (!SpeechRecognition) {

        setPermissionError(
          "Speech recognition is not supported in this browser."
        );

        return;
      }


      try {

        const stream =
          await navigator.mediaDevices
            .getUserMedia({
              audio: true,
            });


        mediaStreamRef.current =
          stream;


        const AudioCtx =
          window.AudioContext ||
          (window as any)
            .webkitAudioContext;


        const audioCtx =
          new AudioCtx();

        audioContextRef.current =
          audioCtx;


        const sourceNode =
          audioCtx.createMediaStreamSource(
            stream
          );


        const analyser =
          audioCtx.createAnalyser();

        analyser.fftSize = 64;

        sourceNode.connect(
          analyser
        );

        analyserRef.current =
          analyser;


        const bufferLength =
          analyser.frequencyBinCount;

        const dataArray =
          new Uint8Array(
            bufferLength
          );


        const updateAudioVisuals =
          () => {

            if (
              !isRecordingRef.current
            ) {
              return;
            }


            analyser.getByteFrequencyData(
              dataArray
            );


            let sum = 0;

            for (
              let i = 0;
              i < bufferLength;
              i++
            ) {
              sum += dataArray[i];
            }


            const average =
              sum / bufferLength;


            setAudioVolume(
              Math.min(
                100,
                Math.round(
                  (average / 128) *
                    100
                )
              )
            );


            const levels = [
              Math.max(
                15,
                Math.min(
                  80,
                  Math.round(
                    (dataArray[2] /
                      255) *
                      80
                  )
                )
              ),

              Math.max(
                20,
                Math.min(
                  95,
                  Math.round(
                    (dataArray[4] /
                      255) *
                      95
                  )
                )
              ),

              Math.max(
                30,
                Math.min(
                  100,
                  Math.round(
                    (dataArray[8] /
                      255) *
                      100
                  )
                )
              ),

              Math.max(
                20,
                Math.min(
                  90,
                  Math.round(
                    (dataArray[12] /
                      255) *
                      90
                  )
                )
              ),

              Math.max(
                15,
                Math.min(
                  75,
                  Math.round(
                    (dataArray[16] /
                      255) *
                      75
                  )
                ),
              ),
            ];


            setWaveformLevels(
              levels
            );


            animFrameRef.current =
              requestAnimationFrame(
                updateAudioVisuals
              );
          };


        const recognition =
          new SpeechRecognition();


        recognition.continuous =
          false;

        recognition.interimResults =
          true;

        recognition.lang =
          "en-IN";


        recognition.onstart = () => {

          setIsRecording(true);

          isRecordingRef.current =
            true;

          setStatusMessage(
            "Listening... Speak your query clearly"
          );

          addLog(
            "INFO",
            "src.rag.asr",
            "Microphone stream active.",
            "text-orange-400"
          );

          updateAudioVisuals();
        };


        recognition.onresult =
          (event: any) => {

            let interim = "";

            for (
              let i =
                event.resultIndex;
              i <
              event.results.length;
              i++
            ) {

              if (
                event.results[i]
                  .isFinal
              ) {

                liveTranscriptRef.current +=
                  event.results[i][0]
                    .transcript;

              } else {

                interim +=
                  event.results[i][0]
                    .transcript;
              }
            }


            const fullLive =
              (
                liveTranscriptRef.current +
                " " +
                interim
              ).trim();


            if (fullLive) {

              setCurrentQuery(
                fullLive
              );

              setEditableQuery(
                fullLive
              );
            }
          };


        recognition.onerror =
          (event: any) => {

            if (
              event.error !==
              "no-speech"
            ) {

              setPermissionError(
                `Speech recognition error: ${event.error}`
              );
            }

            stopRecording();
          };


        recognition.onend = () => {

          if (
            isRecordingRef.current
          ) {
            stopRecording();
          }
        };


        recognitionRef.current =
          recognition;


        recognition.start();

      } catch (error) {

        console.error(
          "Microphone access failed:",
          error
        );

        setPermissionError(
          "Microphone access denied or unavailable."
        );

        setIsRecording(false);

        isRecordingRef.current =
          false;

        stopAudioCapture();
      }
    };


  // ==========================================================
  // TEXT TO SPEECH
  // ==========================================================

  const toggleSpeech =
    () => {

      if (
        !("speechSynthesis" in window)
      ) {

        alert(
          "Text-to-speech is not supported in your browser."
        );

        return;
      }


      if (isSpeaking) {

        window.speechSynthesis.cancel();

        setIsSpeaking(false);

        return;
      }


      window.speechSynthesis.cancel();


      const cleanText =
        answer
          .replace(
            /[0-9]+\.\s*/g,
            ""
          )
          .replace(
            /[*_#]/g,
            ""
          );


      const utterance =
        new SpeechSynthesisUtterance(
          cleanText
        );


      utterance.rate = 1;
      utterance.pitch = 1;


      const voices =
        window.speechSynthesis
          .getVoices();


      const voice =
        voices.find(
          (v) =>
            v.lang.includes(
              "en-IN"
            ) ||
            v.lang.includes(
              "en-GB"
            ) ||
            v.lang.includes(
              "en-US"
            )
        );


      if (voice) {
        utterance.voice =
          voice;
      }


      utterance.onstart =
        () => {
          setIsSpeaking(true);
        };


      utterance.onend =
        () => {
          setIsSpeaking(false);
        };


      utterance.onerror =
        () => {
          setIsSpeaking(false);
        };


      window.speechSynthesis.speak(
        utterance
      );
    };


  // ==========================================================
  // COPY
  // ==========================================================

  const handleCopy =
    async () => {

      try {

        await navigator.clipboard.writeText(
          answer
        );

        setCopiedState(true);

        setTimeout(
          () =>
            setCopiedState(false),
          2000
        );

      } catch (error) {

        console.error(
          "Clipboard error:",
          error
        );
      }
    };


  // ==========================================================
  // EXPORT JSON
  // ==========================================================

  const handleExportJson =
    () => {

      const payload = {
        timestamp:
          new Date().toISOString(),

        query:
          currentQuery,

        answer,

        confidence,

        latencyMs,

        minSimilarity,

        isGrounded,

        guardrailStatus:
          isGrounded
            ? "VERIFIED_GROUNDED"
            : "FLAGGED_LOW_SIMILARITY",

        guardrailWarning,

        modelEngine:
          lastResponse?.modelEngine,

        retrievalBreakdown:
          lastResponse
            ?.retrievalBreakdown,

        chunks,

        systemMetrics:
          latencyStats,
      };


      const blob =
        new Blob(
          [
            JSON.stringify(
              payload,
              null,
              2
            ),
          ],
          {
            type:
              "application/json",
          }
        );


      const url =
        URL.createObjectURL(
          blob
        );


      const anchor =
        document.createElement(
          "a"
        );

      anchor.href = url;

      anchor.download =
        `rag-query-${Date.now()}.json`;

      anchor.click();


      URL.revokeObjectURL(
        url
      );
    };


  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div className="min-h-screen bg-[#0c0d12] text-neutral-100 flex flex-col font-sans selection:bg-orange-500/30 selection:text-orange-200">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="border-b border-neutral-800/80 bg-[#0e1017]/90 backdrop-blur-md sticky top-0 z-40 px-4 lg:px-8 py-3 flex items-center justify-between">

        <div className="flex items-center gap-3">

          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-lg shadow-orange-500/20 text-neutral-950 font-black text-lg">
            R
          </div>

          <div>

            <div className="flex items-center gap-2">

              <h1 className="font-bold text-base text-white tracking-tight">
                RAG in Goa
              </h1>

              <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30 uppercase tracking-wider">
                Dynamic RAG Engine
              </span>

            </div>

            <p className="text-xs text-neutral-400">
              Low-Latency Retrieval-Augmented Generation
            </p>

          </div>

        </div>


        <div className="flex items-center gap-2">

          <button
            onClick={() =>
              setShowKBModal(true)
            }
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-neutral-900 border border-neutral-700 hover:border-teal-500 text-neutral-200 hover:text-teal-300 transition-all"
          >
            <Database className="w-3.5 h-3.5 text-teal-400" />

            <span className="hidden sm:inline">
              Knowledge Corpus
            </span>

            <span className="px-1.5 py-0.2 bg-teal-500/20 text-teal-300 rounded text-[10px] border border-teal-500/30">
              {documents.length} Docs
            </span>
          </button>


          <button
            onClick={() =>
              setShowInspectorModal(true)
            }
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-neutral-900 border border-neutral-700 hover:border-amber-500 text-neutral-200 hover:text-amber-300 transition-all"
          >
            <Cpu className="w-3.5 h-3.5 text-amber-400" />

            <span className="hidden sm:inline">
              Pipeline Telemetry
            </span>
          </button>


          <button
            onClick={() =>
              setShowLatencyModal(true)
            }
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-neutral-900 border border-neutral-700 hover:border-orange-500 text-neutral-200 hover:text-orange-300 transition-all"
          >
            <Activity className="w-3.5 h-3.5 text-orange-400" />

            <span className="hidden sm:inline">
              Latency
            </span>

            <span className="font-mono text-orange-400 font-bold">
              {latencyMs}ms
            </span>
          </button>

        </div>

      </header>


      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">


        {/* ====================================================
            QUERY INPUT
        ==================================================== */}

        <section className="bg-[#12141e] border border-neutral-800 rounded-2xl p-4 sm:p-5 shadow-xl space-y-3">

          <div className="flex items-center justify-between">

            <div className="flex items-center gap-2 text-xs font-semibold text-neutral-300">

              <Search className="w-4 h-4 text-orange-400" />

              <span>
                Ask Any Question
              </span>

            </div>

            <span className="px-2 py-0.5 rounded bg-neutral-900 border border-neutral-700 text-orange-300 font-mono text-[11px]">
              {totalChunks} Chunks
            </span>

          </div>


          <div className="flex items-center gap-2">

            <div className="relative flex-1">

              <input
                type="text"
                value={typedInput}
                onChange={(event) =>
                  setTypedInput(
                    event.target.value
                  )
                }
                onKeyDown={(event) => {

                  if (
                    event.key ===
                      "Enter" &&
                    typedInput.trim()
                  ) {
                    executeQuery(
                      typedInput
                    );
                  }

                }}
                placeholder="Ask a question about Goa or your knowledge base..."
                className="w-full pl-4 pr-10 py-3 rounded-xl bg-neutral-950 border border-neutral-700 text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-orange-500 transition-all"
              />


              {typedInput && (

                <button
                  onClick={() =>
                    setTypedInput("")
                  }
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-neutral-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>

              )}

            </div>


            <button
              onClick={
                isRecording
                  ? stopRecording
                  : startRecording
              }
              disabled={isProcessing}
              className={`p-3 rounded-xl transition-all flex items-center justify-center ${
                isRecording
                  ? "bg-rose-500 text-white animate-pulse"
                  : "bg-neutral-900 border border-neutral-700 text-orange-400"
              }`}
            >
              {isRecording ? (
                <MicOff className="w-5 h-5" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </button>


            <button
              onClick={() =>
                typedInput.trim() &&
                executeQuery(
                  typedInput
                )
              }
              disabled={
                isProcessing ||
                !typedInput.trim()
              }
              className="px-5 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 text-neutral-950 font-bold text-sm flex items-center gap-2 disabled:opacity-40"
            >

              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}

              <span className="hidden sm:inline">
                Ask RAG
              </span>

            </button>

          </div>


          {/* SAMPLE QUERIES */}

          <div className="flex flex-wrap gap-1.5">

            <span className="text-[11px] text-neutral-400 mr-1 flex items-center gap-1">

              <Sparkles className="w-3 h-3 text-orange-400" />

              Samples:

            </span>


            {SAMPLE_QUERIES.map(
              (query, index) => (

                <button
                  key={index}
                  onClick={() => {

                    setTypedInput(
                      query
                    );

                    executeQuery(
                      query
                    );

                  }}
                  className={`text-[11px] px-2.5 py-1 rounded-lg border transition-all ${
                    currentQuery ===
                    query
                      ? "bg-orange-500/20 text-orange-300 border-orange-500/50"
                      : "bg-neutral-900 text-neutral-300 border-neutral-800"
                  }`}
                >
                  {query}
                </button>

              )
            )}

          </div>


          {permissionError && (

            <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">

              <AlertTriangle className="w-4 h-4" />

              {permissionError}

            </div>

          )}

        </section>


        {/* ====================================================
            VOICE WAVEFORM
        ==================================================== */}

        {isRecording && (

          <section className="bg-rose-950/40 border border-rose-800 rounded-2xl p-4 flex items-center justify-between">

            <div className="flex items-center gap-3">

              <div className="w-3 h-3 rounded-full bg-rose-500 animate-ping" />

              <div>

                <div className="text-xs font-bold text-rose-300 uppercase">
                  Listening
                </div>

                <div className="text-sm font-mono text-white">
                  {currentQuery ||
                    "Speak now..."}
                </div>

              </div>

            </div>


            <div className="flex items-end gap-1 h-8">

              {waveformLevels.map(
                (level, index) => (

                  <div
                    key={index}
                    style={{
                      height:
                        `${level}%`,
                    }}
                    className="w-1.5 bg-rose-400 rounded-full"
                  />

                )
              )}

            </div>

          </section>

        )}


        {/* ====================================================
            THREE COLUMNS
        ==================================================== */}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">


          {/* ==================================================
              QUERY CARD
          ================================================== */}

          <div className="bg-[#12141e] border border-neutral-800 rounded-2xl p-5 flex flex-col justify-between shadow-xl">

            <div>

              <div className="flex items-center justify-between pb-3 border-b border-neutral-800">

                <div className="flex items-center gap-2 text-xs font-bold text-neutral-300 uppercase">

                  <Mic className="w-4 h-4 text-orange-400" />

                  1. User Query

                </div>

                <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-neutral-900 text-neutral-400 border border-neutral-800">
                  ASR en-IN
                </span>

              </div>


              <div className="mt-4">

                <div className="bg-neutral-950 p-3.5 rounded-xl border border-neutral-800 min-h-[120px]">

                  {isEditingInline ? (

                    <div className="space-y-2">

                      <textarea
                        value={
                          editableQuery
                        }
                        onChange={(event) =>
                          setEditableQuery(
                            event.target
                              .value
                          )
                        }
                        className="w-full bg-neutral-900 border border-neutral-700 rounded p-2 text-xs text-white"
                        rows={3}
                      />

                      <div className="flex justify-end gap-2">

                        <button
                          onClick={() =>
                            setIsEditingInline(
                              false
                            )
                          }
                          className="px-2 py-1 text-xs text-neutral-400"
                        >
                          Cancel
                        </button>


                        <button
                          onClick={() => {

                            setIsEditingInline(
                              false
                            );

                            executeQuery(
                              editableQuery
                            );

                          }}
                          className="px-3 py-1 bg-orange-500 text-neutral-950 rounded text-xs font-semibold"
                        >
                          Search
                        </button>

                      </div>

                    </div>

                  ) : (

                    <>

                      <p className="text-sm leading-relaxed">
                        "{currentQuery}"
                      </p>

                      <button
                        onClick={() =>
                          setIsEditingInline(
                            true
                          )
                        }
                        className="mt-3 text-[11px] text-orange-400 flex items-center gap-1"
                      >
                        <Edit3 className="w-3 h-3" />
                        Edit query
                      </button>

                    </>

                  )}

                </div>

              </div>

            </div>


            <div className="mt-4 p-3 bg-neutral-950/60 rounded-xl border border-neutral-800 text-xs text-neutral-400 space-y-1 font-mono">

              <div className="flex justify-between">

                <span>
                  Query Tokens:
                </span>

                <span className="text-white">
                  {extractKeywords(
                    currentQuery
                  ).length}
                </span>

              </div>


              <div className="flex justify-between">

                <span>
                  Audio:
                </span>

                <span className="text-emerald-400">
                  {audioVolume}%
                </span>

              </div>


              <div className="flex justify-between">

                <span>
                  Latency:
                </span>

                <span className="text-orange-400">
                  {latencyMs} ms
                </span>

              </div>

            </div>

          </div>


          {/* ==================================================
              RETRIEVED CHUNKS
          ================================================== */}

          <div className="bg-[#12141e] border border-neutral-800 rounded-2xl p-5 flex flex-col shadow-xl">

            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">

              <div className="flex items-center gap-2 text-xs font-bold text-neutral-300 uppercase">

                <Layers className="w-4 h-4 text-teal-400" />

                2. Retrieved Context

                ({chunks.length})

              </div>


              <button
                onClick={() =>
                  setShowInspectorModal(
                    true
                  )
                }
                className="text-[11px] text-teal-400 flex items-center gap-1"
              >
                <Cpu className="w-3 h-3" />
                Inspect
              </button>

            </div>


            <div className="mt-4 space-y-3 max-h-[360px] overflow-y-auto">

              {chunks.length === 0 ? (

                <div className="text-center text-neutral-500 text-xs py-10">

                  {isProcessing
                    ? "Retrieving context..."
                    : "No chunks retrieved."}

                </div>

              ) : (

                chunks.map(
                  (chunk, index) => (

                    <div
                      key={
                        chunk.id ||
                        index
                      }
                      className="p-3.5 rounded-xl bg-neutral-950 border border-neutral-800 text-xs space-y-2"
                    >

                      <div className="flex items-center justify-between">

                        <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-teal-500/20 text-teal-300 border border-teal-500/30">
                          Chunk{" "}
                          {chunk.chunkNumber ||
                            index + 1}
                        </span>


                        <span className="font-mono text-[11px] font-bold text-emerald-400">

                          {Math.round(
                            (chunk.similarityScore ||
                              0) *
                              100
                          )}
                          %

                        </span>

                      </div>


                      <p className="text-neutral-300 leading-relaxed">

                        <HighlightedChunkContent
                          content={
                            chunk.content
                          }
                          query={
                            currentQuery
                          }
                        />

                      </p>


                      <div className="text-[10px] text-neutral-500 font-mono truncate pt-1 border-t border-neutral-900">

                        Source:{" "}
                        {chunk.source}

                      </div>

                    </div>

                  )
                )

              )}

            </div>


            {/* GUARDRAIL */}

            <div
              className={`mt-auto pt-3`}
            >

              <div
                className={`p-3 rounded-xl border flex items-center justify-between text-xs ${
                  isGrounded
                    ? "bg-emerald-950/30 border-emerald-800/60 text-emerald-300"
                    : "bg-amber-950/30 border-amber-800/60 text-amber-300"
                }`}
              >

                <div className="flex items-center gap-2">

                  {isGrounded ? (
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                  )}

                  <span className="font-semibold">

                    {isGrounded
                      ? "Grounded"
                      : "Low Similarity"}

                  </span>

                </div>


                <span className="font-mono font-bold">

                  {Math.round(
                    minSimilarity *
                      100
                  )}
                  %

                </span>

              </div>

            </div>

          </div>


          {/* ==================================================
              ANSWER
          ================================================== */}

          <div className="bg-[#12141e] border border-neutral-800 rounded-2xl p-5 flex flex-col shadow-xl">

            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">

              <div className="flex items-center gap-2 text-xs font-bold text-neutral-300 uppercase">

                <Sparkles className="w-4 h-4 text-orange-400" />

                3. Grounded Answer

              </div>


              <div className="flex items-center gap-1">

                <button
                  onClick={
                    toggleSpeech
                  }
                  className="p-1.5 rounded-lg bg-neutral-900 text-neutral-300"
                >
                  {isSpeaking ? (
                    <VolumeX className="w-3.5 h-3.5" />
                  ) : (
                    <Volume2 className="w-3.5 h-3.5" />
                  )}
                </button>


                <button
                  onClick={
                    handleCopy
                  }
                  className="p-1.5 rounded-lg bg-neutral-900 text-neutral-300"
                >
                  {copiedState ? (
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <Share2 className="w-3.5 h-3.5" />
                  )}
                </button>


                <button
                  onClick={
                    handleExportJson
                  }
                  className="p-1.5 rounded-lg bg-neutral-900 text-neutral-300"
                >
                  <Download className="w-3.5 h-3.5" />
                </button>

              </div>

            </div>


            <div className="mt-4 p-4 rounded-xl bg-neutral-950 border border-neutral-800 text-sm leading-relaxed text-neutral-200 whitespace-pre-line max-h-[300px] overflow-y-auto">

              {isProcessing ? (

                <div className="flex items-center gap-2 text-neutral-400 py-6 justify-center">

                  <Loader2 className="w-4 h-4 animate-spin text-orange-400" />

                  Synthesizing grounded response...

                </div>

              ) : (

                answer ||
                "No response generated yet."

              )}

            </div>


            <div className="mt-auto pt-4 border-t border-neutral-800 space-y-2">

              <div className="flex justify-between text-xs font-mono">

                <span className="text-neutral-400">
                  Confidence
                </span>

                <span className="text-emerald-400 font-bold">

                  {Math.round(
                    confidence * 100
                  )}
                  %

                </span>

              </div>


              <div className="flex justify-between text-xs font-mono">

                <span className="text-neutral-400">
                  Latency
                </span>

                <span className="text-orange-400 font-bold">

                  {latencyMs} ms

                </span>

              </div>


              {lastResponse?.modelEngine && (

                <div className="flex justify-between text-xs font-mono">

                  <span className="text-neutral-400">
                    Model
                  </span>

                  <span className="text-teal-400 truncate max-w-[180px]">

                    {lastResponse.modelEngine}

                  </span>

                </div>

              )}


              {guardrailWarning && (

                <div className="text-[11px] p-2 rounded bg-amber-950/40 border border-amber-800/70 text-amber-300">

                  {guardrailWarning}

                </div>

              )}

            </div>

          </div>

        </div>


        {/* ====================================================
            LOGS
        ==================================================== */}

        <section className="bg-[#12141e] border border-neutral-800 rounded-2xl p-4 shadow-xl">

          <div className="flex items-center justify-between pb-2 mb-2 border-b border-neutral-800">

            <div className="flex items-center gap-2 text-xs font-bold text-neutral-300">

              <Activity className="w-4 h-4 text-orange-400" />

              Real-Time Execution Logs

            </div>


            <span className="text-[11px] text-neutral-500 font-mono">

              {statusMessage}

            </span>

          </div>


          <div className="space-y-1 font-mono text-[11px] max-h-24 overflow-y-auto">

            {logs
              .slice(0, 5)
              .map(
                (log, index) => (

                  <div
                    key={index}
                    className="flex items-center gap-2 text-neutral-400"
                  >

                    <span className="text-neutral-600">

                      [
                      {new Date(
                        log.timestamp
                      ).toLocaleTimeString()}
                      ]

                    </span>


                    <span
                      className={`font-bold ${
                        log.level ===
                        "WARN"
                          ? "text-amber-400"
                          : log.level ===
                            "ERROR"
                          ? "text-rose-400"
                          : log.level ===
                            "SUCCESS"
                          ? "text-emerald-400"
                          : "text-teal-400"
                      }`}
                    >
                      {log.level}
                    </span>


                    <span className="text-neutral-500">
                      {log.component}:
                    </span>


                    <span
                      className={
                        log.color ||
                        "text-neutral-300"
                      }
                    >
                      {log.message}
                    </span>

                  </div>

                )
              )}

          </div>

        </section>

      </main>


      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer className="border-t border-neutral-800/80 bg-[#0e1017] px-6 py-4 text-center text-xs text-neutral-500">

        RAG in Goa • Voice-Enabled Retrieval-Augmented Generation

      </footer>


      {/* ======================================================
          MODALS
      ====================================================== */}

      <KnowledgeBaseModal
        isOpen={
          showKBModal
        }
        onClose={() =>
          setShowKBModal(false)
        }
        documents={
          documents
        }
        totalChunks={
          totalChunks
        }
        onIngestDocument={
          handleIngestDocument
        }
        onResetCorpus={
          handleResetCorpus
        }
        isLoading={
          isLoadingDocs
        }
      />


      <PipelineInspector
        data={
          lastResponse
        }
        isOpen={
          showInspectorModal
        }
        onClose={() =>
          setShowInspectorModal(false)
        }
      />


      <LatencyMetricsModal
        isOpen={
          showLatencyModal
        }
        onClose={() =>
          setShowLatencyModal(false)
        }
        queryHistory={
          queryHistory
        }
        stats={
          latencyStats
        }
      />

    </div>
  );
}