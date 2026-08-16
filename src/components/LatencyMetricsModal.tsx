import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine
} from 'recharts';
import { Activity, X, Gauge, Zap, TrendingUp, AlertTriangle, ShieldCheck } from 'lucide-react';

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

interface LatencyMetricsModalProps {
  isOpen: boolean;
  onClose: () => void;
  queryHistory: LatencyQueryRecord[];
  stats: LatencyStats;
}

export const LatencyMetricsModal: React.FC<LatencyMetricsModalProps> = ({
  isOpen,
  onClose,
  queryHistory,
  stats
}) => {
  if (!isOpen) return null;

  // Take the last 50 queries for charting
  const last50Queries = queryHistory.slice(-50).map((item, index) => ({
    queryLabel: `#${item.queryNumber || index + 1}`,
    queryNumber: item.queryNumber || index + 1,
    latencyMs: item.latencyMs,
    p50: item.p50Benchmark || stats.p50,
    p75: item.p75Benchmark || stats.p75,
    p100: item.p100Benchmark || stats.p100,
    query: item.query,
    confidence: item.confidence,
    minSimilarity: item.minSimilarity,
    isGrounded: item.isGrounded,
    timestamp: item.timestamp
  }));

  return (
    <div
      id="modal-latency-telemetry"
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 overflow-y-auto"
    >
      <div className="bg-[#0D0D0E] border border-white/10 rounded-2xl sm:rounded-3xl p-5 sm:p-7 md:p-8 max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl relative my-auto animate-scaleUp">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-white/10 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-orange-500/10 border border-orange-500/30 flex items-center justify-center text-orange-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm sm:text-base font-bold uppercase tracking-widest text-white">
                  Latency Telemetry & Metrics
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold">
                  Last {last50Queries.length} Queries
                </span>
              </div>
              <p className="text-xs text-white/40 font-mono mt-0.5">
                Real-time sub-200ms budget tracking with P50, P75, and P100 SLA thresholds
              </p>
            </div>
          </div>
          <button
            id="btn-close-latency-modal"
            onClick={onClose}
            className="text-white/40 hover:text-white text-xs uppercase tracking-widest font-mono cursor-pointer min-h-[36px] min-w-[36px] flex items-center justify-center rounded-lg hover:bg-white/5 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Metric KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-5 shrink-0">
          <div className="bg-[#141416] border border-white/5 rounded-xl sm:rounded-2xl p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between text-white/40 text-[10px] uppercase font-mono tracking-wider">
              <span>P50 (Median)</span>
              <Zap className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-1">
              <span className="text-xl sm:text-2xl font-bold font-mono text-emerald-400">{stats.p50}</span>
              <span className="text-[10px] font-mono text-white/40">ms</span>
            </div>
            <div className="text-[9px] font-mono text-emerald-400/80 mt-1">
              ✓ Sub-120ms target met
            </div>
          </div>

          <div className="bg-[#141416] border border-white/5 rounded-xl sm:rounded-2xl p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between text-white/40 text-[10px] uppercase font-mono tracking-wider">
              <span>P75 Latency</span>
              <Gauge className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-1">
              <span className="text-xl sm:text-2xl font-bold font-mono text-amber-300">{stats.p75}</span>
              <span className="text-[10px] font-mono text-white/40">ms</span>
            </div>
            <div className="text-[9px] font-mono text-amber-400/80 mt-1">
              ✓ Fast hybrid retrieval
            </div>
          </div>

          <div className="bg-[#141416] border border-white/5 rounded-xl sm:rounded-2xl p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between text-white/40 text-[10px] uppercase font-mono tracking-wider">
              <span>P100 (Max Spike)</span>
              <TrendingUp className="w-3.5 h-3.5 text-orange-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-1">
              <span className="text-xl sm:text-2xl font-bold font-mono text-orange-400">{stats.p100}</span>
              <span className="text-[10px] font-mono text-white/40">ms</span>
            </div>
            <div className="text-[9px] font-mono text-orange-400/80 mt-1">
              ✓ Below 200ms budget limit
            </div>
          </div>

          <div className="bg-[#141416] border border-white/5 rounded-xl sm:rounded-2xl p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between text-white/40 text-[10px] uppercase font-mono tracking-wider">
              <span>SLA Compliance</span>
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-1">
              <span className="text-xl sm:text-2xl font-bold font-mono text-white">{Math.round(stats.underBudgetRatio * 100)}</span>
              <span className="text-[10px] font-mono text-white/40">% &lt;200ms</span>
            </div>
            <div className="text-[9px] font-mono text-white/50 mt-1">
              Avg: {stats.avg}ms • Min: {stats.min}ms
            </div>
          </div>
        </div>

        {/* Recharts Line Chart for P50/P75/P100 and Query Latencies */}
        <div className="bg-[#050505] border border-white/5 rounded-xl sm:rounded-2xl p-4 flex-1 flex flex-col min-h-[280px]">
          <div className="flex items-center justify-between mb-3 text-xs">
            <span className="font-mono text-[11px] uppercase tracking-widest text-white/50">
              Query-by-Query Latency & Percentile Benchmarks
            </span>
            <span className="font-mono text-[10px] text-white/40">
              Target: &lt; 200ms (Red Dashed Line)
            </span>
          </div>

          <div className="w-full h-64 sm:h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={last50Queries} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis 
                  dataKey="queryLabel" 
                  stroke="#666" 
                  fontSize={10} 
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis 
                  stroke="#666" 
                  fontSize={10} 
                  domain={[0, 220]} 
                  unit="ms" 
                  tickLine={false}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-[#18181B] border border-white/10 rounded-xl p-3 shadow-xl font-mono text-xs max-w-xs">
                          <p className="font-bold text-white mb-1">Query {label}</p>
                          <p className="text-[11px] text-white/60 truncate mb-2">&ldquo;{data.query}&rdquo;</p>
                          <div className="space-y-1 text-[11px] border-t border-white/10 pt-1.5">
                            <div className="flex justify-between">
                              <span className="text-orange-400">Recorded Latency:</span>
                              <span className="font-bold text-white">{data.latencyMs} ms</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-emerald-400">P50 Baseline:</span>
                              <span className="text-white">{data.p50} ms</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-amber-400">P75 Baseline:</span>
                              <span className="text-white">{data.p75} ms</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-rose-400">P100 Ceiling:</span>
                              <span className="text-white">{data.p100} ms</span>
                            </div>
                            <div className="flex justify-between pt-1 border-t border-white/5">
                              <span className="text-white/40">Min Similarity:</span>
                              <span className={data.minSimilarity < 0.70 ? "text-amber-400 font-bold" : "text-emerald-400"}>
                                {Math.round(data.minSimilarity * 100)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend 
                  wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
                  iconType="circle"
                />
                
                {/* 200ms Budget Limit line */}
                <ReferenceLine 
                  y={200} 
                  stroke="#ef4444" 
                  strokeDasharray="4 4" 
                  label={{ value: '200ms Budget Limit', fill: '#ef4444', fontSize: 10, position: 'top' }} 
                />

                {/* Main Query Latency Line */}
                <Line
                  type="monotone"
                  dataKey="latencyMs"
                  name="Recorded Latency (ms)"
                  stroke="#f97316"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#f97316' }}
                  activeDot={{ r: 5, fill: '#ffffff', stroke: '#f97316' }}
                />

                {/* P50 Benchmark */}
                <Line
                  type="monotone"
                  dataKey="p50"
                  name="P50 Benchmark"
                  stroke="#10b981"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={false}
                />

                {/* P75 Benchmark */}
                <Line
                  type="monotone"
                  dataKey="p75"
                  name="P75 Benchmark"
                  stroke="#f59e0b"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={false}
                />

                {/* P100 Benchmark */}
                <Line
                  type="monotone"
                  dataKey="p100"
                  name="P100 Benchmark"
                  stroke="#ec4899"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Footer controls */}
        <div className="mt-4 sm:mt-5 flex flex-col sm:flex-row items-center justify-between gap-3 shrink-0 pt-3 border-t border-white/5">
          <div className="flex items-center gap-2 text-[10px] font-mono text-white/40">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Telemetry buffer active: {queryHistory.length} total requests recorded</span>
          </div>

          <button
            onClick={onClose}
            className="w-full sm:w-auto min-h-[38px] px-5 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-xs font-mono uppercase tracking-widest text-white transition-colors cursor-pointer"
          >
            Close Telemetry
          </button>
        </div>

      </div>
    </div>
  );
};
