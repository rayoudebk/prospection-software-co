"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import {
  useCompanies,
  useUpdateCompany,
  useCreateCompany,
  useTopCandidates,
  useExpansionBrief,
  useGates,
  useWorkspaceJobs,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";
import { workspaceApi, Company } from "@/lib/api";
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
  Network,
  Tags,
  Sparkles,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import clsx from "clsx";

const CLASSIFICATION_LABELS: Record<string, string> = {
  good_target: "Good target",
  borderline_watchlist: "Borderline / Watchlist",
  not_good_target: "Not good",
  insufficient_evidence: "Insufficient evidence",
};

function prettyLabel(raw: string | null | undefined) {
  if (!raw) return "Unknown";
  return raw
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function classificationBadgeClass(classification?: string | null) {
  if (classification === "good_target") return "badge-success";
  if (classification === "borderline_watchlist") return "badge-warning";
  if (classification === "not_good_target") return "badge-danger";
  return "badge-neutral";
}

function reasonCodeChips(company: Company): string[] {
  const codes = company.reason_codes || { positive: [], caution: [], reject: [] };
  return [...codes.positive, ...codes.caution, ...codes.reject].slice(0, 5);
}

function CitedSummary({ company }: { company: Company }) {
  const summary = company.citation_summary_v1;
  const fallback = company.rationale_summary || company.why_relevant[0]?.text || "No rationale generated yet.";
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
        <p key={`${company.id}-${sentence.id}`} className="leading-relaxed">
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

  const { data: companies, isLoading, refetch } = useCompanies(workspaceId);
  const { data: expansionArtifact } = useExpansionBrief(workspaceId);
  const {
    data: topCandidates,
    error: topCandidatesError,
  } = useTopCandidates(workspaceId, 25, allowDegradedRun);
  const { data: gates } = useGates(workspaceId);
  const { data: discoveryJobs } = useWorkspaceJobs(workspaceId, "discovery_universe");
  const updateCompany = useUpdateCompany(workspaceId);
  const createCompany = useCreateCompany(workspaceId);

  const [filter, setFilter] = useState<
    "all" | "good_target" | "borderline_watchlist" | "not_good_target" | "insufficient_evidence"
  >("all");
  const [showAddModal, setShowAddModal] = useState(false);
  const [newCompany, setNewCompany] = useState({ name: "", website: "", hq_country: "" });

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.runDiscovery(workspaceId),
    () => refetch(),
    (jobId) => workspaceApi.cancelJob(workspaceId, jobId)
  );

  const handleKeep = async (company: Company) => {
    await updateCompany.mutateAsync({
      companyId: company.id,
      data: { status: "kept" },
    });
  };

  const handleRemove = async (company: Company) => {
    await updateCompany.mutateAsync({
      companyId: company.id,
      data: { status: "removed" },
    });
  };

  const handleRestore = async (company: Company) => {
    await updateCompany.mutateAsync({
      companyId: company.id,
      data: { status: "candidate" },
    });
  };

  const handleAddCompany = async () => {
    if (!newCompany.name) return;
    await createCompany.mutateAsync(newCompany);
    setNewCompany({ name: "", website: "", hq_country: "" });
    setShowAddModal(false);
  };

  const nonSolutionCompanies = companies?.filter((company) => (company.entity_type || "company") !== "solution") || [];
  const filteredCompanies = nonSolutionCompanies.filter((company) => {
    if (filter === "all") return true;
    return (company.decision_classification || "insufficient_evidence") === filter;
  });

  const keptCount = nonSolutionCompanies.filter((company) => company.status === "kept" || company.status === "enriched").length || 0;
  const goodCount = nonSolutionCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "good_target").length || 0;
  const borderlineCount =
    nonSolutionCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "borderline_watchlist").length || 0;
  const notGoodCount =
    nonSolutionCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "not_good_target").length || 0;
  const insufficientCount =
    nonSolutionCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "insufficient_evidence").length || 0;
  const latestCompletedDiscoveryJob = discoveryJobs?.find((job) => job.state === "completed") ?? null;
  const expansionBrief = expansionArtifact?.expansion_brief;
  const adjacencyBoxes = expansionBrief?.adjacency_boxes || [];
  const companySeeds = expansionBrief?.company_seeds || [];
  const technologyShiftClaims = expansionBrief?.technology_shift_claims || [];
  const prioritizedLanes = adjacencyBoxes
    .filter((box) => box.status !== "user_removed" && box.status !== "user_deprioritized")
    .slice(0, 4);
  const prioritizedSeeds = companySeeds
    .filter((seed) => seed.status !== "rejected")
    .slice(0, 6);

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
        step={3}
        title="Universe"
        subtitle="Run lane-driven sourcing, inspect why each company fits, and triage the resulting candidate universe with auditable source-backed evidence."
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
                ? `${keptCount} companies kept — you can proceed to Validation`
                : gates.missing_items.universe?.join(", ") || "Keep at least 5 companies to continue"}
            </span>
          </div>
        </div>
      )}

      {!jobRunner.isRunning && <JobRunSummary job={latestCompletedDiscoveryJob} />}

      <section className="border border-steel-200 bg-steel-50 p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-[0.18em] text-steel-400">
              Discovery Inputs
            </div>
            <h3 className="text-lg font-semibold text-oxford mt-1">
              Universe is now driven by canonical adjacency lanes and seed companies
            </h3>
            <p className="text-sm text-steel-600 mt-1 max-w-3xl">
              Discovery reads reviewed `adjacency_boxes` and `company_seeds` from the expansion brief, then falls back only if that canonical layer is missing.
            </p>
          </div>
          {expansionBrief?.fallback_mode ? (
            <span className="inline-flex items-center gap-2 border border-warning/40 bg-warning/10 px-3 py-2 text-xs font-medium text-warning">
              <AlertCircle className="w-4 h-4" />
              Expansion brief is in fallback mode
            </span>
          ) : null}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="border border-steel-200 bg-white p-3">
            <div className="text-xs uppercase tracking-[0.14em] text-steel-400">Adjacency lanes</div>
            <div className="text-2xl font-semibold text-oxford mt-1">{adjacencyBoxes.length}</div>
          </div>
          <div className="border border-steel-200 bg-white p-3">
            <div className="text-xs uppercase tracking-[0.14em] text-steel-400">Company seeds</div>
            <div className="text-2xl font-semibold text-oxford mt-1">{companySeeds.length}</div>
          </div>
          <div className="border border-steel-200 bg-white p-3">
            <div className="text-xs uppercase tracking-[0.14em] text-steel-400">Technology shifts</div>
            <div className="text-2xl font-semibold text-oxford mt-1">{technologyShiftClaims.length}</div>
          </div>
        </div>

        {expansionBrief?.normalization_warning ? (
          <div className="border border-warning/30 bg-warning/10 p-3 text-sm text-warning">
            {expansionBrief.normalization_warning}
          </div>
        ) : null}

        {prioritizedLanes.length ? (
          <div className="grid grid-cols-1 xl:grid-cols-[1.4fr_1fr] gap-4">
            <div className="border border-steel-200 bg-white p-4 space-y-3">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-steel-400">
                <Network className="w-4 h-4" />
                Reviewed Adjacency Lanes
              </div>
              <div className="space-y-3">
                {prioritizedLanes.map((box) => (
                  <div key={box.id} className="border border-steel-200 bg-steel-50 p-3">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold text-oxford">{box.label}</div>
                        <div className="text-xs text-steel-500 mt-1">
                          {prettyLabel(box.adjacency_kind)} · {prettyLabel(box.priority_tier)}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2 text-[11px]">
                        <span className="border border-info/30 bg-info/10 px-2 py-1 text-info">
                          Workflow {prettyLabel(box.criticality.workflow_criticality)}
                        </span>
                        <span className="border border-info/30 bg-info/10 px-2 py-1 text-info">
                          Switching {prettyLabel(box.criticality.switching_cost_intensity)}
                        </span>
                      </div>
                    </div>
                    {box.why_it_matters ? (
                      <p className="text-sm text-steel-700 mt-3">{box.why_it_matters}</p>
                    ) : null}
                    {box.retrieval_query_seeds.length ? (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {box.retrieval_query_seeds.slice(0, 3).map((query) => (
                          <span
                            key={`${box.id}-${query}`}
                            className="border border-steel-200 bg-white px-2 py-1 text-[11px] text-steel-700"
                          >
                            {query}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>

            <div className="border border-steel-200 bg-white p-4 space-y-3">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-steel-400">
                <Tags className="w-4 h-4" />
                Seed Companies
              </div>
              {prioritizedSeeds.length ? (
                <div className="space-y-2">
                  {prioritizedSeeds.map((seed) => (
                    <div key={seed.id} className="border border-steel-200 bg-steel-50 p-3">
                      <div className="text-sm font-semibold text-oxford">{seed.name}</div>
                      <div className="text-xs text-steel-500 mt-1">
                        {prettyLabel(seed.seed_type)}
                        {seed.seed_role ? ` · ${prettyLabel(seed.seed_role)}` : ""}
                      </div>
                      {seed.website ? (
                        <a
                          href={seed.website}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 mt-2 text-xs text-info hover:underline"
                        >
                          {new URL(seed.website).hostname}
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="border border-dashed border-steel-300 bg-steel-50 p-4 text-sm text-steel-500">
                  No canonical seed companies available yet.
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="border border-dashed border-steel-300 bg-white p-4 text-sm text-steel-500">
            Generate and confirm the expansion brief to populate canonical discovery lanes before running Universe.
          </div>
        )}
      </section>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-oxford">Candidate Universe</h2>
          <p className="text-steel-500">
            {nonSolutionCompanies.length || 0} companies surfaced • {goodCount} strong fit • {borderlineCount} watchlist • {notGoodCount} weak fit
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-secondary flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Company
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

      {jobRunner.isRunning && (
        <JobProgressPanel
          job={
            jobRunner.job ?? {
              id: 0,
              workspace_id: workspaceId,
              company_id: null,
              job_type: "discovery_universe",
              state: "queued",
              provider: "gemini_flash",
              progress: jobRunner.progress,
              progress_message: jobRunner.progressMessage ?? null,
              result_json: null,
              error_message: null,
              created_at: new Date().toISOString(),
              started_at: null,
              finished_at: null,
            }
          }
          progress={jobRunner.progress}
          progressMessage={jobRunner.progressMessage}
          isStopping={jobRunner.isStopping}
          onStop={jobRunner.canStop ? jobRunner.stop : undefined}
        />
      )}

      {/* Add Company Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-oxford/80 flex items-center justify-center z-50">
          <div className="bg-steel-50 p-6 w-full max-w-md shadow-xl border border-steel-200">
            <h3 className="text-lg font-semibold text-oxford mb-4">Add Company Manually</h3>
            <div className="space-y-4">
              <div>
                <label className="label">
                  Company Name *
                </label>
                <input
                  type="text"
                  value={newCompany.name}
                  onChange={(e) => setNewCompany({ ...newCompany, name: e.target.value })}
                  className="input"
                />
              </div>
              <div>
                <label className="label">
                  Website
                </label>
                <input
                  type="url"
                  value={newCompany.website}
                  onChange={(e) => setNewCompany({ ...newCompany, website: e.target.value })}
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
                  value={newCompany.hq_country}
                  onChange={(e) => setNewCompany({ ...newCompany, hq_country: e.target.value })}
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
                onClick={handleAddCompany}
                disabled={!newCompany.name || createCompany.isPending}
                className="flex-1 btn-primary disabled:opacity-50"
              >
                Add Company
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
            {f === "all" && `All (${nonSolutionCompanies.length || 0})`}
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
            Directory-only and solution-level entities are excluded from this view.
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

      {/* Company Grid */}
      {filteredCompanies && filteredCompanies.length === 0 ? (
        <div className="text-center py-16 bg-steel-50 border border-steel-200">
          <Globe className="w-12 h-12 text-steel-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-oxford mb-2">No companies found</h3>
          <p className="text-steel-500 mb-4">
            Run discovery to source candidate companies
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredCompanies?.map((company) => (
            <div
              key={company.id}
              className={clsx(
                "bg-steel-50 border p-4 transition",
                company.decision_classification === "good_target"
                  ? "border-success"
                  : company.decision_classification === "not_good_target"
                  ? "border-danger/50"
                  : company.decision_classification === "borderline_watchlist"
                  ? "border-warning/60"
                  : "border-steel-200 hover:border-oxford"
              )}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-oxford">{company.name}</h3>
                  {(company.official_website_url || company.website) && (
                    <a
                      href={company.official_website_url || company.website || undefined}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-info hover:underline flex items-center gap-1"
                    >
                      {new URL((company.official_website_url || company.website) as string).hostname}
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
                <span
                  className={clsx(
                    "badge",
                    classificationBadgeClass(company.decision_classification)
                  )}
                >
                  {CLASSIFICATION_LABELS[company.decision_classification || "insufficient_evidence"] || "Insufficient evidence"}
                </span>
              </div>

              <div className="flex items-center gap-2 text-sm text-steel-500 mb-2">
                <Building2 className="w-4 h-4" />
                {company.hq_country || "Unknown"} • status: {company.status}
              </div>
              {company.entity_type && company.entity_type !== "company" && (
                <div className="text-xs text-warning mb-2">Entity type: {company.entity_type}</div>
              )}

              {reasonCodeChips(company).length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {reasonCodeChips(company).map((code) => (
                    <span
                      key={`${company.id}-${code}`}
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

              {((company.unresolved_contradictions_count || 0) > 0 || company.evidence_sufficiency === "insufficient") && (
                <div className="flex flex-wrap gap-2 mb-2">
                  {!(company.official_website_url || company.website) && (
                    <span className="px-2 py-0.5 text-xs border border-warning/30 bg-warning/10 text-warning">
                      official website unresolved
                    </span>
                  )}
                  {(company.unresolved_contradictions_count || 0) > 0 && (
                    <span className="px-2 py-0.5 text-xs border border-danger/30 bg-danger/10 text-danger">
                      {company.unresolved_contradictions_count} contradiction(s)
                    </span>
                  )}
                  {company.evidence_sufficiency === "insufficient" && (
                    <span className="px-2 py-0.5 text-xs border border-warning/30 bg-warning/10 text-warning">
                      Unknown/missing evidence
                    </span>
                  )}
                </div>
              )}

              {company.top_claim?.text && company.top_claim?.source_url && (
                <div className="mt-2 p-2 bg-steel-100 border border-steel-200">
                  <div className="text-xs text-steel-500 mb-1">
                    Top claim • {company.top_claim.source_tier || "unknown tier"}
                  </div>
                  <div className="text-sm text-steel-700 line-clamp-3">{company.top_claim.text}</div>
                </div>
              )}

              {company.why_fit_bullets && company.why_fit_bullets.length > 0 && (
                <div className="mt-3 space-y-2">
                  <div className="text-xs uppercase tracking-wide text-steel-500">Why It Fits</div>
                  {company.why_fit_bullets.slice(0, 3).map((bullet, index) => (
                    <div key={`${company.id}-fit-${index}`} className="text-sm text-steel-700">
                      - {bullet.text}
                      {bullet.citation_url ? (
                        <a
                          href={bullet.citation_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-2 text-xs text-info hover:underline"
                        >
                          source
                        </a>
                      ) : null}
                    </div>
                  ))}
                </div>
              )}

              {(company.business_model_signal || company.employee_signal) && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {company.business_model_signal ? (
                    <span className="px-2 py-0.5 text-xs border border-steel-200 bg-steel-100 text-steel-700">
                      {company.business_model_signal}
                    </span>
                  ) : null}
                  {company.employee_signal ? (
                    <span className="px-2 py-0.5 text-xs border border-steel-200 bg-steel-100 text-steel-700">
                      {company.employee_signal}
                    </span>
                  ) : null}
                </div>
              )}

              {company.customer_proof && company.customer_proof.length > 0 && (
                <div className="mt-3">
                  <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Customer Proof</div>
                  <div className="space-y-1">
                    {company.customer_proof.slice(0, 3).map((proof, index) => (
                      <div key={`${company.id}-proof-${index}`} className="text-sm text-steel-700">
                        - {proof}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <CitedSummary company={company} />

              {company.open_questions && company.open_questions.length > 0 && (
                <div className="mt-3">
                  <div className="text-xs uppercase tracking-wide text-steel-500 mb-2">Open Questions</div>
                  <div className="flex flex-wrap gap-2">
                    {company.open_questions.slice(0, 3).map((question) => (
                      <span
                        key={`${company.id}-${question}`}
                        className="px-2 py-0.5 text-xs border border-warning/30 bg-warning/10 text-warning"
                      >
                        {question}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between pt-3 border-t border-steel-100">
                <span className="text-xs text-steel-400">
                  {company.evidence_count} citations
                </span>
                <div className="flex gap-2">
                  {company.status === "removed" ? (
                    <button
                      onClick={() => handleRestore(company)}
                      className="btn-secondary text-sm py-1 px-3"
                    >
                      Restore
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={() => handleRemove(company)}
                        className="p-1.5 text-steel-400 hover:text-danger hover:bg-danger/10 transition"
                      >
                        <X className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleKeep(company)}
                        disabled={company.status === "kept" || company.status === "enriched"}
                        className={clsx(
                          "p-1.5 transition",
                          company.status === "kept" || company.status === "enriched"
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
