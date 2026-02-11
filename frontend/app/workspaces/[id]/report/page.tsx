"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import {
  useReportCards,
  useReportLens,
  useReports,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";
import { workspaceApi, ReportCard, ReportClaim } from "@/lib/api";
import {
  FileSpreadsheet,
  Loader2,
  Download,
  RefreshCw,
  ExternalLink,
  Filter,
  AlertCircle,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

const REPORT_CLASSIFICATION_LABELS: Record<string, string> = {
  good_target: "Good target",
  borderline_watchlist: "Borderline / Watchlist",
  not_good_target: "Not good",
  insufficient_evidence: "Insufficient evidence",
};

function reportClassificationClass(classification?: string | null) {
  if (classification === "good_target") return "badge-success";
  if (classification === "borderline_watchlist") return "badge-warning";
  if (classification === "not_good_target") return "badge-danger";
  return "badge-neutral";
}

function SourcePill({ claim }: { claim: ReportClaim }) {
  if (!claim.source) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 text-xs bg-warning/10 text-warning border border-warning/30">
        Hypothesis
      </span>
    );
  }

  return (
    <a
      href={claim.source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-info/10 text-info border border-info/30 hover:bg-info/20"
      title={claim.source.label}
    >
      {claim.source.label}
      <ExternalLink className="w-3 h-3" />
    </a>
  );
}

function CardSection({ card }: { card: ReportCard }) {
  return (
    <div className="bg-steel-50 border border-steel-200 p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold text-oxford">{card.name}</h3>
          <div className="text-sm text-steel-500 mt-1">
            {card.hq_country || "Unknown country"} • {card.size_bucket}
            {card.size_range_low != null && card.size_range_high != null && card.size_range_low !== card.size_range_high
              ? ` • ${card.size_range_low}-${card.size_range_high} employees`
              : card.size_estimate
              ? ` • ~${card.size_estimate} employees`
              : ""}
          </div>
          {card.website && (
            <a
              href={card.website}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-info hover:underline mt-1"
            >
              {card.website.replace(/^https?:\/\//, "")}
              <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
        <span className={clsx("badge", reportClassificationClass(card.decision_classification))}>
          {REPORT_CLASSIFICATION_LABELS[card.decision_classification || "insufficient_evidence"] || "Insufficient evidence"}
        </span>
      </div>

      {card.reason_highlights && card.reason_highlights.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Decision Rationale</div>
          <div className="space-y-1">
            {card.reason_highlights.slice(0, 5).map((reason, idx) => (
              <div key={`${card.vendor_id}-reason-${idx}`} className="text-sm text-steel-700">
                - {reason}
              </div>
            ))}
          </div>
        </div>
      )}

      {card.known_unknowns && card.known_unknowns.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Known Unknowns</div>
          <div className="flex flex-wrap gap-2">
            {card.known_unknowns.slice(0, 6).map((item, idx) => (
              <span
                key={`${card.vendor_id}-unknown-${idx}`}
                className="px-2 py-0.5 text-xs bg-warning/10 text-warning border border-warning/30"
              >
                {item}
              </span>
            ))}
          </div>
        </div>
      )}

      {card.evidence_quality_summary && Object.keys(card.evidence_quality_summary).length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Evidence Quality</div>
          <div className="text-sm text-steel-700 space-y-1">
            {"freshness_ratio" in card.evidence_quality_summary && (
              <div>
                Freshness ratio:{" "}
                {Math.round(Number(card.evidence_quality_summary.freshness_ratio || 0) * 100)}%
              </div>
            )}
            {"total_evidence" in card.evidence_quality_summary && (
              <div>Total evidence items: {String(card.evidence_quality_summary.total_evidence)}</div>
            )}
          </div>
        </div>
      )}

      <div>
        <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Brick Mapping</div>
        <div className="space-y-2">
          {card.brick_mapping.slice(0, 5).map((claim, idx) => (
            <div key={idx} className="flex items-start justify-between gap-2 text-sm">
              <span className="text-steel-700">{claim.text}</span>
              <SourcePill claim={claim} />
            </div>
          ))}
          {card.brick_mapping.length === 0 && (
            <div className="text-sm text-steel-500">No brick evidence available in this snapshot.</div>
          )}
        </div>
      </div>

      <div>
        <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Customer / Partner Evidence</div>
        <div className="space-y-2">
          {card.customer_partner_evidence.slice(0, 5).map((claim, idx) => (
            <div key={idx} className="flex items-start justify-between gap-2 text-sm">
              <span className="text-steel-700">{claim.text}</span>
              <SourcePill claim={claim} />
            </div>
          ))}
          {card.customer_partner_evidence.length === 0 && (
            <div className="text-sm text-steel-500">No customer/partner evidence available in this snapshot.</div>
          )}
        </div>
      </div>

      <div>
        <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Filing Metrics</div>
        <div className="space-y-2">
          {Object.entries(card.filing_metrics).map(([key, metric]) => (
            <div key={key} className="flex items-start justify-between gap-2 text-sm">
              <span className="text-steel-700">
                {key}: {metric.value}
                {metric.unit ? ` ${metric.unit}` : ""}
                {metric.period ? ` (${metric.period})` : ""}
              </span>
              <a
                href={metric.source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-info/10 text-info border border-info/30 hover:bg-info/20"
              >
                {metric.source.label}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          ))}
          {Object.keys(card.filing_metrics).length === 0 && (
            <div className="text-sm text-steel-500">{card.coverage_note || "No filing metrics captured."}</div>
          )}
        </div>
      </div>

      {card.next_validation_questions.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Next Validation Questions</div>
          <ul className="list-disc pl-5 text-sm text-steel-700 space-y-1">
            {card.next_validation_questions.map((question, idx) => (
              <li key={idx}>{question}</li>
            ))}
          </ul>
        </div>
      )}

      {card.source_pills && card.source_pills.length > 0 && (
        <div>
          <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Sources Used</div>
          <div className="flex flex-wrap gap-2">
            {card.source_pills.map((source, idx) => (
              <a
                key={`${source.url}-${idx}`}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-info/10 text-info border border-info/30 hover:bg-info/20"
                title={source.label}
              >
                {source.label}
                <ExternalLink className="w-3 h-3" />
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ReportPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const [reportName, setReportName] = useState("");
  const [selectedReportId, setSelectedReportId] = useState<number | null>(null);
  const [mode, setMode] = useState<"compete" | "complement">("compete");
  const [sizeFilter, setSizeFilter] = useState<"all" | "sme_in_range" | "unknown" | "outside_sme_range">("all");
  const [exporting, setExporting] = useState(false);

  const { data: reports, isLoading: reportsLoading, refetch: refetchReports } = useReports(workspaceId);

  const reportRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.generateReport(workspaceId, { name: reportName || undefined, include_unknown_size: true }),
    () => {
      refetchReports();
      setReportName("");
    }
  );

  useEffect(() => {
    if (!selectedReportId && reports && reports.length > 0) {
      setSelectedReportId(reports[0].id);
    }
  }, [reports, selectedReportId]);

  const bucketParam = sizeFilter === "all" ? undefined : sizeFilter;
  const { data: cards, isLoading: cardsLoading } = useReportCards(
    workspaceId,
    selectedReportId,
    bucketParam
  );
  const { data: lens, isLoading: lensLoading } = useReportLens(workspaceId, selectedReportId, mode);

  const sortedCards = useMemo(() => {
    if (!cards) return [];
    return [...cards].sort((a, b) => {
      const aScore = mode === "compete" ? a.compete_score : a.complement_score;
      const bScore = mode === "compete" ? b.compete_score : b.complement_score;
      return bScore - aScore;
    });
  }, [cards, mode]);

  const handleExport = async () => {
    if (!selectedReportId) return;
    setExporting(true);
    try {
      const payload = await workspaceApi.exportReport(workspaceId, selectedReportId, "rich_json");
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `report-snapshot-${selectedReportId}-rich.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="space-y-6">
      <StepHeader
        icon={FileSpreadsheet}
        step={4}
        title="Static Report"
        subtitle="Generate immutable snapshot cards with classification-first narratives, reason highlights, and source-backed filing metrics when coverage is reliable."
      />

      <div className="bg-steel-50 border border-steel-200 p-4 space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-[280px]">
            <label className="label">Snapshot name (optional)</label>
            <input
              value={reportName}
              onChange={(e) => setReportName(e.target.value)}
              placeholder="e.g. Q1 EU Wealth SME Radar"
              className="input"
            />
          </div>
          <button
            onClick={reportRunner.run}
            disabled={reportRunner.isRunning}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {reportRunner.isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating... {Math.round(reportRunner.progress * 100)}%
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                Generate Snapshot
              </>
            )}
          </button>

          <div className="min-w-[240px]">
            <label className="label">Snapshot</label>
            <select
              value={selectedReportId ?? ""}
              onChange={(e) => setSelectedReportId(e.target.value ? Number(e.target.value) : null)}
              className="input"
            >
              <option value="">Select snapshot</option>
              {reports?.map((report) => (
                <option key={report.id} value={report.id}>
                  {report.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleExport}
            disabled={!selectedReportId || exporting}
            className="btn-secondary flex items-center gap-2 disabled:opacity-50"
          >
            {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            Export
          </button>
        </div>

        {reportRunner.isRunning && reportRunner.progressMessage && (
          <div className="text-sm text-steel-600 border border-steel-200 bg-white px-3 py-2">
            {reportRunner.progressMessage}
          </div>
        )}

        {reportRunner.jobError && (
          <div className="text-sm text-danger border border-danger/40 bg-danger/10 px-3 py-2">
            {reportRunner.jobError}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-1 bg-steel-50 border border-steel-200 p-1">
          <button
            onClick={() => setMode("compete")}
            className={clsx(
              "px-3 py-1.5 text-sm font-medium",
              mode === "compete" ? "bg-oxford text-white" : "text-steel-600 hover:bg-steel-100"
            )}
          >
            Compete
          </button>
          <button
            onClick={() => setMode("complement")}
            className={clsx(
              "px-3 py-1.5 text-sm font-medium",
              mode === "complement" ? "bg-success text-white" : "text-steel-600 hover:bg-steel-100"
            )}
          >
            Complement
          </button>
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-steel-500" />
          <select
            value={sizeFilter}
            onChange={(e) => setSizeFilter(e.target.value as typeof sizeFilter)}
            className="input"
          >
            <option value="all">All buckets</option>
            <option value="sme_in_range">SME 15-100</option>
            <option value="unknown">Unknown size</option>
            <option value="outside_sme_range">Outside range</option>
          </select>
        </div>
      </div>

      {reportsLoading || cardsLoading || lensLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-oxford" />
        </div>
      ) : !selectedReportId ? (
        <div className="text-center py-12 bg-steel-50 border border-steel-200 text-steel-600">
          Generate or select a snapshot to view report cards.
        </div>
      ) : (
        <>
          {lens && (
            <div className="bg-steel-50 border border-steel-200 p-4 text-sm text-steel-700">
              <div className="font-medium text-oxford mb-2">Lens Summary ({mode})</div>
              <div className="flex flex-wrap gap-4">
                <span>Total companies: {lens.total_count}</span>
                <span>SME in range: {lens.counts_by_bucket?.sme_in_range ?? 0}</span>
                <span>Unknown size: {lens.counts_by_bucket?.unknown ?? 0}</span>
                <span>Outside range: {lens.counts_by_bucket?.outside_sme_range ?? 0}</span>
              </div>
            </div>
          )}

          {sortedCards.length === 0 ? (
            <div className="text-center py-12 bg-steel-50 border border-steel-200">
              <AlertCircle className="w-10 h-10 text-steel-400 mx-auto mb-3" />
              <div className="text-steel-600">No cards found for this filter.</div>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {sortedCards.map((card) => (
                <CardSection key={card.vendor_id} card={card} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
