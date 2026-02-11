"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import {
  useVendors,
  useUpdateVendor,
  useCreateVendor,
  useTopCandidates,
  useGates,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";
import { workspaceApi, Vendor } from "@/lib/api";
import {
  Globe,
  Search,
  Plus,
  Check,
  X,
  ExternalLink,
  Loader2,
  CheckCircle,
  AlertCircle,
  Building2,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

const CLASSIFICATION_LABELS: Record<string, string> = {
  good_target: "Good target",
  borderline_watchlist: "Borderline / Watchlist",
  not_good_target: "Not good",
  insufficient_evidence: "Insufficient evidence",
};

function classificationBadgeClass(classification?: string | null) {
  if (classification === "good_target") return "badge-success";
  if (classification === "borderline_watchlist") return "badge-warning";
  if (classification === "not_good_target") return "badge-danger";
  return "badge-neutral";
}

function reasonCodeChips(vendor: Vendor): string[] {
  const codes = vendor.reason_codes || { positive: [], caution: [], reject: [] };
  return [...codes.positive, ...codes.caution, ...codes.reject].slice(0, 5);
}

function CitedSummary({ vendor }: { vendor: Vendor }) {
  const summary = vendor.citation_summary_v1;
  const fallback = vendor.rationale_summary || vendor.why_relevant[0]?.text || "No rationale generated yet.";
  if (!summary || !summary.sentences?.length || !summary.source_pills?.length) {
    return <div className="text-sm text-steel-600 mb-3 line-clamp-3">{fallback}</div>;
  }

  const pillById = new Map(summary.source_pills.map((pill) => [pill.pill_id, pill]));
  const pillNumberById = new Map(summary.source_pills.map((pill, index) => [pill.pill_id, index + 1]));
  const sentences = summary.sentences
    .map((sentence) => ({
      ...sentence,
      citation_pill_ids: sentence.citation_pill_ids.filter((pillId) => pillById.has(pillId)),
    }))
    .filter((sentence) => sentence.citation_pill_ids.length > 0);

  if (!sentences.length) {
    return <div className="text-sm text-steel-600 mb-3 line-clamp-3">{fallback}</div>;
  }

  return (
    <div className="text-sm text-steel-600 mb-3 space-y-2">
      {sentences.map((sentence) => (
        <p key={`${vendor.id}-${sentence.id}`} className="leading-relaxed">
          {sentence.text}{" "}
          {sentence.citation_pill_ids.map((pillId) => {
            const pill = pillById.get(pillId);
            const number = pillNumberById.get(pillId);
            if (!pill || !number) return null;
            return (
              <a
                key={`${sentence.id}-${pillId}`}
                href={pill.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-1.5 py-0.5 text-xs border border-info/30 bg-info/10 text-info hover:bg-info/20 mr-1"
                title={`${pill.label} • ${pill.source_tier}`}
              >
                [{number}]
              </a>
            );
          })}
        </p>
      ))}
    </div>
  );
}

export default function UniversePage() {
  const params = useParams();
  const workspaceId = Number(params.id);
  const [allowDegradedRun, setAllowDegradedRun] = useState(false);

  const { data: vendors, isLoading, refetch } = useVendors(workspaceId);
  const {
    data: topCandidates,
    error: topCandidatesError,
  } = useTopCandidates(workspaceId, 25, allowDegradedRun);
  const { data: gates } = useGates(workspaceId);
  const updateVendor = useUpdateVendor(workspaceId);
  const createVendor = useCreateVendor(workspaceId);

  const [filter, setFilter] = useState<
    "all" | "good_target" | "borderline_watchlist" | "not_good_target" | "insufficient_evidence"
  >("all");
  const [showAddModal, setShowAddModal] = useState(false);
  const [newVendor, setNewVendor] = useState({ name: "", website: "", hq_country: "" });

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.runDiscovery(workspaceId),
    () => refetch()
  );

  const handleKeep = async (vendor: Vendor) => {
    await updateVendor.mutateAsync({
      vendorId: vendor.id,
      data: { status: "kept" },
    });
  };

  const handleRemove = async (vendor: Vendor) => {
    await updateVendor.mutateAsync({
      vendorId: vendor.id,
      data: { status: "removed" },
    });
  };

  const handleRestore = async (vendor: Vendor) => {
    await updateVendor.mutateAsync({
      vendorId: vendor.id,
      data: { status: "candidate" },
    });
  };

  const handleAddVendor = async () => {
    if (!newVendor.name) return;
    await createVendor.mutateAsync(newVendor);
    setNewVendor({ name: "", website: "", hq_country: "" });
    setShowAddModal(false);
  };

  const nonSolutionVendors = vendors?.filter((v) => (v.entity_type || "company") !== "solution") || [];
  const filteredVendors = nonSolutionVendors.filter((v) => {
    if (filter === "all") return true;
    return (v.decision_classification || "insufficient_evidence") === filter;
  });

  const keptCount = nonSolutionVendors.filter((v) => v.status === "kept" || v.status === "enriched").length || 0;
  const goodCount = nonSolutionVendors.filter((v) => (v.decision_classification || "insufficient_evidence") === "good_target").length || 0;
  const borderlineCount =
    nonSolutionVendors.filter((v) => (v.decision_classification || "insufficient_evidence") === "borderline_watchlist").length || 0;
  const notGoodCount =
    nonSolutionVendors.filter((v) => (v.decision_classification || "insufficient_evidence") === "not_good_target").length || 0;
  const insufficientCount =
    nonSolutionVendors.filter((v) => (v.decision_classification || "insufficient_evidence") === "insufficient_evidence").length || 0;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-8 h-8 animate-spin text-oxford" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <StepHeader
        icon={Globe}
        step={3}
        title="Universe"
        subtitle="Build a curated longlist of potential acquisition targets matching your brick model. Keep strong fits, remove weak fits, and prepare for static report generation."
      />

      {/* Status Banner */}
      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.universe
              ? "bg-success/10 border-success"
              : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.universe ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.universe ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.universe
                ? `${keptCount} vendors kept — you can proceed to Report`
                : gates.missing_items.universe?.join(", ") || "Keep at least 5 vendors to continue"}
            </span>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-oxford">Candidate Universe</h2>
          <p className="text-steel-500">
            {nonSolutionVendors.length || 0} companies discovered • {goodCount} good • {borderlineCount} watchlist • {notGoodCount} not-good
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-secondary flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Vendor
          </button>
          <button
            onClick={jobRunner.run}
            disabled={jobRunner.isRunning}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {jobRunner.isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Discovering... {Math.round(jobRunner.progress * 100)}%
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Run Discovery
              </>
            )}
          </button>
        </div>
      </div>

      {/* Add Vendor Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-oxford/80 flex items-center justify-center z-50">
          <div className="bg-steel-50 p-6 w-full max-w-md shadow-xl border border-steel-200">
            <h3 className="text-lg font-semibold text-oxford mb-4">Add Vendor Manually</h3>
            <div className="space-y-4">
              <div>
                <label className="label">
                  Company Name *
                </label>
                <input
                  type="text"
                  value={newVendor.name}
                  onChange={(e) => setNewVendor({ ...newVendor, name: e.target.value })}
                  className="input"
                />
              </div>
              <div>
                <label className="label">
                  Website
                </label>
                <input
                  type="url"
                  value={newVendor.website}
                  onChange={(e) => setNewVendor({ ...newVendor, website: e.target.value })}
                  placeholder="https://..."
                  className="input"
                />
              </div>
              <div>
                <label className="label">
                  HQ Country
                </label>
                <input
                  type="text"
                  value={newVendor.hq_country}
                  onChange={(e) => setNewVendor({ ...newVendor, hq_country: e.target.value })}
                  placeholder="e.g., UK"
                  className="input"
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAddModal(false)}
                className="flex-1 btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleAddVendor}
                disabled={!newVendor.name || createVendor.isPending}
                className="flex-1 btn-primary disabled:opacity-50"
              >
                Add Vendor
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-1 bg-steel-50 border border-steel-200 p-1 inline-flex flex-wrap">
        {(["all", "good_target", "borderline_watchlist", "not_good_target", "insufficient_evidence"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={clsx(
              "px-4 py-2 text-sm font-medium transition",
              filter === f
                ? "bg-oxford text-white"
                : "text-steel-600 hover:bg-steel-100"
            )}
          >
            {f === "all" && `All (${nonSolutionVendors.length || 0})`}
            {f === "good_target" && `Good (${goodCount})`}
            {f === "borderline_watchlist" && `Watchlist (${borderlineCount})`}
            {f === "not_good_target" && `Not Good (${notGoodCount})`}
            {f === "insufficient_evidence" && `Insufficient (${insufficientCount})`}
          </button>
        ))}
      </div>

      {topCandidatesError && (topCandidatesError as { status?: number; detail?: unknown }).status === 409 && (
        <div className="bg-warning/10 border border-warning p-4">
          <div className="text-sm font-medium text-warning mb-1">High-quality gate blocked latest degraded run</div>
          <div className="text-xs text-steel-600 mb-3">
            {String(((topCandidatesError as { detail?: Record<string, unknown> }).detail || {})["message"] || "Latest run is degraded.")}
          </div>
          <button
            onClick={() => setAllowDegradedRun(true)}
            className="btn-secondary text-xs py-1 px-3"
          >
            Load degraded run anyway
          </button>
        </div>
      )}

      {topCandidates && topCandidates.length > 0 && (
        <div className="bg-steel-50 border border-steel-200 p-4">
          <div className="text-sm font-medium text-oxford mb-2">Top 25 Ranking-Eligible Companies</div>
          <div className="text-xs text-steel-500 mb-3">
            Directory-only and solution-level entities are excluded.
          </div>
          {topCandidates[0]?.run_quality_tier && (
            <div className="text-xs text-steel-500 mb-2">
              Run quality: {topCandidates[0].run_quality_tier}
              {topCandidates[0].degraded_reasons && topCandidates[0].degraded_reasons.length > 0
                ? ` (${topCandidates[0].degraded_reasons.join(", ")})`
                : ""}
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            {topCandidates.slice(0, 10).map((row) => (
              <span
                key={`${row.candidate_entity_id ?? "candidate"}-${row.company_name}`}
                className="badge badge-neutral"
              >
                {row.company_name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Vendor Grid */}
      {filteredVendors && filteredVendors.length === 0 ? (
        <div className="text-center py-16 bg-steel-50 border border-steel-200">
          <Globe className="w-12 h-12 text-steel-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-oxford mb-2">No vendors found</h3>
          <p className="text-steel-500 mb-4">
            Run discovery to find potential acquisition targets
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredVendors?.map((vendor) => (
            <div
              key={vendor.id}
              className={clsx(
                "bg-steel-50 border p-4 transition",
                vendor.decision_classification === "good_target"
                  ? "border-success"
                  : vendor.decision_classification === "not_good_target"
                  ? "border-danger/50"
                  : vendor.decision_classification === "borderline_watchlist"
                  ? "border-warning/60"
                  : "border-steel-200 hover:border-oxford"
              )}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-oxford">{vendor.name}</h3>
                  {(vendor.official_website_url || vendor.website) && (
                    <a
                      href={vendor.official_website_url || vendor.website || undefined}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-info hover:underline flex items-center gap-1"
                    >
                      {new URL((vendor.official_website_url || vendor.website) as string).hostname}
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
                <span
                  className={clsx(
                    "badge",
                    classificationBadgeClass(vendor.decision_classification)
                  )}
                >
                  {CLASSIFICATION_LABELS[vendor.decision_classification || "insufficient_evidence"] || "Insufficient evidence"}
                </span>
              </div>

              <div className="flex items-center gap-2 text-sm text-steel-500 mb-2">
                <Building2 className="w-4 h-4" />
                {vendor.hq_country || "Unknown"} • status: {vendor.status}
              </div>

              {vendor.entity_type && vendor.entity_type !== "company" && (
                <div className="text-xs text-warning mb-2">Entity type: {vendor.entity_type}</div>
              )}

              {vendor.tags_vertical.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {vendor.tags_vertical.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 text-xs bg-steel-100 text-steel-600"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {reasonCodeChips(vendor).length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {reasonCodeChips(vendor).map((code) => (
                    <span
                      key={`${vendor.id}-${code}`}
                      className={clsx(
                        "px-2 py-0.5 text-xs border",
                        code.startsWith("POS-")
                          ? "bg-success/10 text-success border-success/30"
                          : code.startsWith("REJ-")
                          ? "bg-danger/10 text-danger border-danger/30"
                          : "bg-warning/10 text-warning border-warning/30"
                      )}
                    >
                      {code}
                    </span>
                  ))}
                </div>
              )}

              {((vendor.unresolved_contradictions_count || 0) > 0 || vendor.evidence_sufficiency === "insufficient") && (
                <div className="flex flex-wrap gap-2 mb-2">
                  {!(vendor.official_website_url || vendor.website) && (
                    <span className="px-2 py-0.5 text-xs border border-warning/30 bg-warning/10 text-warning">
                      official website unresolved
                    </span>
                  )}
                  {(vendor.unresolved_contradictions_count || 0) > 0 && (
                    <span className="px-2 py-0.5 text-xs border border-danger/30 bg-danger/10 text-danger">
                      {vendor.unresolved_contradictions_count} contradiction(s)
                    </span>
                  )}
                  {vendor.evidence_sufficiency === "insufficient" && (
                    <span className="px-2 py-0.5 text-xs border border-warning/30 bg-warning/10 text-warning">
                      Unknown/missing evidence
                    </span>
                  )}
                </div>
              )}

              {vendor.top_claim?.text && vendor.top_claim?.source_url && (
                <div className="mt-2 p-2 bg-steel-100 border border-steel-200">
                  <div className="text-xs text-steel-500 mb-1">
                    Top claim • {vendor.top_claim.source_tier || "unknown tier"}
                  </div>
                  <div className="text-sm text-steel-700 line-clamp-3">{vendor.top_claim.text}</div>
                </div>
              )}

              <CitedSummary vendor={vendor} />

              <div className="flex items-center justify-between pt-3 border-t border-steel-100">
                <span className="text-xs text-steel-400">
                  {vendor.evidence_count} citations
                </span>
                <div className="flex gap-2">
                  {vendor.status === "removed" ? (
                    <button
                      onClick={() => handleRestore(vendor)}
                      className="btn-secondary text-sm py-1 px-3"
                    >
                      Restore
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={() => handleRemove(vendor)}
                        className="p-1.5 text-steel-400 hover:text-danger hover:bg-danger/10 transition"
                      >
                        <X className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleKeep(vendor)}
                        disabled={vendor.status === "kept" || vendor.status === "enriched"}
                        className={clsx(
                          "p-1.5 transition",
                          vendor.status === "kept" || vendor.status === "enriched"
                            ? "text-success bg-success/10"
                            : "text-steel-400 hover:text-success hover:bg-success/10"
                        )}
                      >
                        <Check className="w-4 h-4" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
