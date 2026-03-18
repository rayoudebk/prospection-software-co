"use client";

import { useMemo, useState } from "react";
import { useParams } from "next/navigation";
import clsx from "clsx";
import {
  AlertCircle,
  Check,
  CheckCircle,
  Database,
  ExternalLink,
  Globe,
  Loader2,
  Network,
  Plus,
  Search,
  Sparkles,
  X,
} from "lucide-react";

import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import { StepHeader } from "@/components/StepHeader";
import { Company, DiscoveryReadiness, UniverseTopCandidate, workspaceApi } from "@/lib/api";
import {
  useCompanies,
  useCreateCompany,
  useDiscoveryDiagnostics,
  useGates,
  useTopCandidates,
  useUpdateCompany,
  useWorkspaceJobs,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";

const CLASSIFICATION_LABELS: Record<string, string> = {
  good_target: "Good target",
  borderline_watchlist: "Watchlist",
  not_good_target: "Not good",
  insufficient_evidence: "Insufficient evidence",
};

const READINESS_REASON_LABELS: Record<string, string> = {
  db_schema_invalid: "Discovery schema is missing required columns.",
  redis_unavailable: "Redis is unavailable.",
  worker_unavailable: "No Celery worker is responding.",
  retrieval_provider_unavailable: "No retrieval provider key is configured.",
  model_provider_unavailable: "No model provider key is configured.",
  expansion_not_confirmed: "Expansion brief is not confirmed yet.",
  company_profile_missing: "Company profile is missing.",
};

function classificationBadgeClass(classification?: string | null) {
  if (classification === "good_target") return "badge-success";
  if (classification === "borderline_watchlist") return "badge-warning";
  if (classification === "not_good_target") return "badge-danger";
  return "badge-neutral";
}

function statusLabel(status?: string | null) {
  if (status === "kept") return "Kept";
  if (status === "removed") return "Removed";
  if (status === "enriched") return "Enriched";
  return "Candidate";
}

function titleCaseToken(value?: string | null) {
  return String(value || "")
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function hostnameFor(url?: string | null) {
  if (!url) return null;
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

function CitedSummary({ candidate }: { candidate: UniverseTopCandidate }) {
  const summary = candidate.citation_summary_v1;
  const fallback = candidate.rationale_summary || "No company description generated yet.";

  if (!summary || !summary.sentences?.length || !summary.source_pills?.length) {
    return <p className="text-sm leading-6 text-steel-600">{fallback}</p>;
  }

  const pillById = new Map(summary.source_pills.map((pill) => [pill.pill_id, pill]));
  const pillNumberById = new Map(summary.source_pills.map((pill, index) => [pill.pill_id, index + 1]));
  const sentences = summary.sentences
    .map((sentence) => ({
      ...sentence,
      citation_pill_ids: sentence.citation_pill_ids.filter((pillId) => pillById.has(pillId)),
    }))
    .filter((sentence) => sentence.citation_pill_ids.length > 0)
    .slice(0, 2);

  if (!sentences.length) {
    return <p className="text-sm leading-6 text-steel-600">{fallback}</p>;
  }

  return (
    <div className="space-y-2 text-sm leading-6 text-steel-600">
      {sentences.map((sentence) => (
        <p key={`${candidate.company_id ?? candidate.candidate_entity_id ?? candidate.company_name}-${sentence.id}`}>
          {sentence.text}{" "}
          {sentence.citation_pill_ids.slice(0, 2).map((pillId) => {
            const pill = pillById.get(pillId);
            const number = pillNumberById.get(pillId);
            if (!pill || !number) return null;
            return (
              <a
                key={`${sentence.id}-${pillId}`}
                href={pill.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center rounded-full border border-info/30 bg-info/10 px-1.5 py-0.5 text-xs text-info hover:bg-info/20"
                title={pill.label}
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

function SignalChips({
  title,
  items,
  tone = "neutral",
}: {
  title: string;
  items: string[];
  tone?: "neutral" | "info" | "success";
}) {
  if (!items.length) return null;
  const toneClass =
    tone === "info"
      ? "border-info/20 bg-info/10 text-info"
      : tone === "success"
        ? "border-success/20 bg-success/10 text-success"
        : "border-steel-200 bg-steel-50 text-steel-600";

  return (
    <div className="space-y-2">
      <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">{title}</div>
      <div className="flex flex-wrap gap-2">
        {items.map((item) => (
          <span key={`${title}-${item}`} className={clsx("rounded-full border px-2.5 py-1 text-xs", toneClass)}>
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function MetricTile({ label, value, icon: Icon }: { label: string; value: string | number; icon: typeof Globe }) {
  return (
    <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-steel-400">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-oxford">{value}</div>
    </div>
  );
}

function DiscoveryReadinessPanel({ readiness }: { readiness: DiscoveryReadiness }) {
  const blockedReasons = readiness.reasons_blocked.map(
    (reason) => READINESS_REASON_LABELS[reason] || titleCaseToken(reason)
  );

  return (
    <div
      className={clsx(
        "mt-5 rounded-3xl border px-5 py-4",
        readiness.runnable ? "border-success/30 bg-success/10" : "border-warning/30 bg-warning/10"
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            {readiness.runnable ? (
              <CheckCircle className="h-5 w-5 text-success" />
            ) : (
              <AlertCircle className="h-5 w-5 text-warning" />
            )}
            <span className={readiness.runnable ? "font-medium text-success" : "font-medium text-warning-dark"}>
              {readiness.runnable
                ? "Discovery runtime is ready."
                : "Discovery is blocked until the runtime issues below are fixed."}
            </span>
          </div>
          {blockedReasons.length ? (
            <ul className="mt-3 space-y-1 text-sm text-warning-dark/90">
              {blockedReasons.map((reason) => (
                <li key={reason}>• {reason}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-success">
              Expansion is confirmed, provider keys are available, and a worker is reachable.
            </p>
          )}
        </div>

        <div className="grid gap-2 text-xs text-steel-500 sm:grid-cols-2">
          <span className="rounded-full border border-steel-200 bg-white px-3 py-1">
            Mode {titleCaseToken(readiness.execution_mode)}
          </span>
          <span className="rounded-full border border-steel-200 bg-white px-3 py-1">
            Expansion {readiness.expansion_confirmed ? "confirmed" : "pending"}
          </span>
          <span className="rounded-full border border-steel-200 bg-white px-3 py-1">
            Worker {readiness.worker_available ? "ready" : "missing"}
          </span>
          <span className="rounded-full border border-steel-200 bg-white px-3 py-1">
            Retrieval {readiness.execution_mode === "fixture"
              ? "not required"
              : readiness.retrieval_provider_available
                ? readiness.available_retrieval_providers.join(", ")
                : "missing"}
          </span>
          <span className="rounded-full border border-steel-200 bg-white px-3 py-1">
            Models {readiness.execution_mode === "fixture"
              ? "not required"
              : readiness.model_available
                ? readiness.available_model_providers.join(", ")
                : "missing"}
          </span>
        </div>
      </div>
    </div>
  );
}

function UniverseCandidateCard({
  candidate,
  onKeep,
  onRemove,
  onRestore,
}: {
  candidate: UniverseTopCandidate;
  onKeep: () => Promise<void>;
  onRemove: () => Promise<void>;
  onRestore: () => Promise<void>;
}) {
  const companyStatus = candidate.company_status || "candidate";
  const discoveryHosts = candidate.discovery_sources
    .map((source) => hostnameFor(source))
    .filter((host): host is string => Boolean(host))
    .slice(0, 3);
  const registryIdentity = candidate.registry_identity || null;
  const registryMatchConfidence =
    typeof registryIdentity?.match_confidence === "number" && registryIdentity.match_confidence > 0
      ? `${Math.round(registryIdentity.match_confidence * 100)}%`
      : null;
  const actionsDisabled = !candidate.company_id;

  return (
    <article className="rounded-3xl border border-steel-200 bg-white px-5 py-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-lg text-oxford">{candidate.company_name}</h3>
            <span className="rounded-full border border-steel-200 bg-steel-50 px-2 py-0.5 text-xs text-steel-500">
              {statusLabel(companyStatus)}
            </span>
          </div>
          <p className="text-sm text-steel-500">
            {candidate.hq_country || "Unknown country"}
            {candidate.entity_type ? ` · ${titleCaseToken(candidate.entity_type)}` : ""}
          </p>
          {candidate.official_website_url ? (
            <a
              href={candidate.official_website_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-info hover:underline"
            >
              {hostnameFor(candidate.official_website_url) || candidate.official_website_url}
              <ExternalLink className="h-3 w-3" />
            </a>
          ) : null}
        </div>

        <span className={clsx("badge", classificationBadgeClass(candidate.decision_classification))}>
          {CLASSIFICATION_LABELS[candidate.decision_classification || "insufficient_evidence"] ||
            "Insufficient evidence"}
        </span>
      </div>

      <CitedSummary candidate={candidate} />

      {candidate.top_claim?.text && candidate.top_claim?.source_url ? (
        <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">
            Top evidence line
          </div>
          <p className="mt-2 text-sm leading-6 text-steel-700">{candidate.top_claim.text}</p>
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2">
        <SignalChips
          title="Scope Match"
          items={candidate.scope_buckets.map((item) => titleCaseToken(item))}
          tone="info"
        />
        <SignalChips title="Capability Signals" items={candidate.capability_signals.slice(0, 4)} />
        <SignalChips title="Vertical Hints" items={candidate.likely_verticals.slice(0, 4)} tone="success" />
        <SignalChips title="Discovery Sources" items={discoveryHosts} />
      </div>

      {registryIdentity ? (
        <div className="rounded-2xl border border-steel-200 bg-white px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">
            Registry Identity
          </div>
          <div className="mt-2 flex flex-wrap gap-2 text-xs text-steel-600">
            {registryIdentity.country ? (
              <span className="rounded-full border border-steel-200 px-2.5 py-1">
                {registryIdentity.country}
              </span>
            ) : null}
            {registryIdentity.source ? (
              <span className="rounded-full border border-steel-200 px-2.5 py-1">
                {titleCaseToken(registryIdentity.source)}
              </span>
            ) : null}
            {registryIdentity.id ? (
              <span className="rounded-full border border-steel-200 px-2.5 py-1">
                ID {registryIdentity.id}
              </span>
            ) : null}
            {registryMatchConfidence ? (
              <span className="rounded-full border border-success/20 bg-success/10 px-2.5 py-1 text-success">
                Match {registryMatchConfidence}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {candidate.expansion_provenance.length ? (
        <div className="rounded-2xl border border-steel-200 bg-white px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">
            Retrieval Provenance
          </div>
          <div className="mt-2 space-y-2">
            {candidate.expansion_provenance.slice(0, 3).map((item, index) => (
              <div key={`${candidate.company_name}-provenance-${index}`} className="text-sm text-steel-600">
                <span className="font-medium text-steel-700">
                  {[item.provider, item.query_type, item.brick_name].filter(Boolean).join(" · ") || "Discovery query"}
                </span>
                {item.query_text ? <span> · {item.query_text}</span> : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-steel-100 pt-4">
        <div className="flex flex-wrap items-center gap-2 text-xs text-steel-400">
          <span>{candidate.evidence_count} citations</span>
          <span>•</span>
          <span>{candidate.discovery_sources.length} discovery links</span>
          {candidate.first_party_hint_pages_crawled_total ? (
            <>
              <span>•</span>
              <span>{candidate.first_party_hint_pages_crawled_total} hint pages crawled</span>
            </>
          ) : null}
        </div>
        <div className="flex gap-2">
          {companyStatus === "removed" ? (
            <button
              onClick={onRestore}
              disabled={actionsDisabled}
              className="btn-secondary text-sm py-1 px-3 disabled:opacity-50"
            >
              Restore
            </button>
          ) : (
            <>
              <button
                onClick={onRemove}
                disabled={actionsDisabled}
                className="inline-flex items-center gap-1 rounded-full border border-steel-200 px-3 py-1.5 text-sm text-steel-700 transition-colors hover:border-danger/30 hover:bg-danger/10 hover:text-danger disabled:opacity-50"
              >
                <X className="h-4 w-4" />
                Remove
              </button>
              <button
                onClick={onKeep}
                disabled={actionsDisabled || companyStatus === "kept" || companyStatus === "enriched"}
                className={clsx(
                  "inline-flex items-center gap-1 rounded-full border px-3 py-1.5 text-sm transition-colors disabled:opacity-50",
                  companyStatus === "kept" || companyStatus === "enriched"
                    ? "border-success/30 bg-success/10 text-success"
                    : "border-steel-200 text-steel-700 hover:border-success/30 hover:bg-success/10 hover:text-success"
                )}
              >
                <Check className="h-4 w-4" />
                Keep
              </button>
            </>
          )}
        </div>
      </div>

      {actionsDisabled ? (
        <p className="text-xs text-warning-dark">
          This candidate has not been materialized into a company row yet, so keep/remove actions are unavailable.
        </p>
      ) : null}
    </article>
  );
}

function ManualCompanyCard({
  company,
  onRemove,
  onRestore,
}: {
  company: Company;
  onRemove: () => Promise<void>;
  onRestore: () => Promise<void>;
}) {
  return (
    <article className="rounded-3xl border border-steel-200 bg-white px-5 py-5 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <h3 className="text-lg text-oxford">{company.name}</h3>
          <p className="text-sm text-steel-500">
            Manual addition
            {company.hq_country ? ` · ${company.hq_country}` : ""}
          </p>
          {company.website ? (
            <a
              href={company.website}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-info hover:underline"
            >
              {hostnameFor(company.website) || company.website}
              <ExternalLink className="h-3 w-3" />
            </a>
          ) : null}
        </div>
        <span className="rounded-full border border-steel-200 bg-steel-50 px-2 py-1 text-xs text-steel-500">
          {statusLabel(company.status)}
        </span>
      </div>

      <p className="text-sm text-steel-600">
        Manual queue entries stay visible here even before they have discovery-backed screening metadata.
      </p>

      <div className="flex justify-end gap-2 border-t border-steel-100 pt-4">
        {company.status === "removed" ? (
          <button onClick={onRestore} className="btn-secondary text-sm py-1 px-3">
            Restore
          </button>
        ) : (
          <button
            onClick={onRemove}
            className="inline-flex items-center gap-1 rounded-full border border-steel-200 px-3 py-1.5 text-sm text-steel-700 transition-colors hover:border-danger/30 hover:bg-danger/10 hover:text-danger"
          >
            <X className="h-4 w-4" />
            Remove
          </button>
        )}
      </div>
    </article>
  );
}

export default function UniversePage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const [filter, setFilter] = useState<"all" | "kept" | "good_target" | "borderline_watchlist">("all");
  const [showAddModal, setShowAddModal] = useState(false);
  const [allowDegraded, setAllowDegraded] = useState(false);
  const [newCompany, setNewCompany] = useState({ name: "", website: "", hq_country: "" });

  const { data: companies, isLoading: companiesLoading, refetch: refetchCompanies } = useCompanies(workspaceId);
  const {
    data: topCandidates,
    isLoading: topCandidatesLoading,
    error: topCandidatesError,
    refetch: refetchTopCandidates,
  } = useTopCandidates(workspaceId, 60, allowDegraded);
  const { data: diagnostics, refetch: refetchDiagnostics } = useDiscoveryDiagnostics(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const { data: discoveryJobs } = useWorkspaceJobs(workspaceId, "discovery_universe");
  const updateCompany = useUpdateCompany(workspaceId);
  const createCompany = useCreateCompany(workspaceId);

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.runDiscovery(workspaceId),
    () => {
      void refetchTopCandidates();
      void refetchCompanies();
      void refetchDiagnostics();
    },
    (jobId) => workspaceApi.cancelJob(workspaceId, jobId)
  );

  const candidates = useMemo(() => topCandidates ?? [], [topCandidates]);
  const manualCompanies = useMemo(() => (companies || []).filter((company) => company.is_manual), [companies]);

  const filteredCandidates = useMemo(() => {
    if (filter === "all") return candidates;
    if (filter === "kept") {
      return candidates.filter(
        (candidate) => candidate.company_status === "kept" || candidate.company_status === "enriched"
      );
    }
    return candidates.filter(
      (candidate) => (candidate.decision_classification || "insufficient_evidence") === filter
    );
  }, [candidates, filter]);

  const keptCount =
    candidates.filter((candidate) => candidate.company_status === "kept" || candidate.company_status === "enriched")
      .length || 0;
  const goodCount =
    candidates.filter((candidate) => (candidate.decision_classification || "insufficient_evidence") === "good_target")
      .length || 0;
  const watchlistCount =
    candidates.filter(
      (candidate) => (candidate.decision_classification || "insufficient_evidence") === "borderline_watchlist"
    ).length || 0;
  const latestCompletedDiscoveryJob = discoveryJobs?.find((job) => job.state === "completed") ?? null;

  const providerMix = diagnostics?.source_coverage?.external_search?.provider_mix || {};
  const providerMixLabels = Object.entries(providerMix)
    .filter(([, count]) => Number(count) > 0)
    .map(([provider, count]) => `${titleCaseToken(provider)} ${count}`);
  const registryQueriesByCountry = diagnostics?.funnel_metrics?.registry_queries_by_country || {};
  const registryCountryLabels = Object.entries(registryQueriesByCountry)
    .filter(([, count]) => Number(count) > 0)
    .map(([country, count]) => `${country} ${count}`);

  const handleKeep = async (candidate: UniverseTopCandidate) => {
    if (!candidate.company_id) return;
    await updateCompany.mutateAsync({
      companyId: candidate.company_id,
      data: { status: "kept" },
    });
  };

  const handleRemove = async (candidate: UniverseTopCandidate) => {
    if (!candidate.company_id) return;
    await updateCompany.mutateAsync({
      companyId: candidate.company_id,
      data: { status: "removed" },
    });
  };

  const handleRestore = async (candidate: UniverseTopCandidate) => {
    if (!candidate.company_id) return;
    await updateCompany.mutateAsync({
      companyId: candidate.company_id,
      data: { status: "candidate" },
    });
  };

  const handleManualRemove = async (company: Company) => {
    await updateCompany.mutateAsync({
      companyId: company.id,
      data: { status: "removed" },
    });
  };

  const handleManualRestore = async (company: Company) => {
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

  if (companiesLoading || topCandidatesLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-8 w-8 animate-spin text-oxford" />
      </div>
    );
  }

  const degradedError = topCandidatesError as (Error & { status?: number; detail?: unknown }) | null;
  const canShowDegraded = degradedError?.status === 409 && !allowDegraded;
  const discoveryReadiness = gates?.discovery_readiness;

  return (
    <div className="space-y-6">
      <StepHeader
        step={3}
        title="Universe"
        subtitle="Search around the approved graph nodes, resolve identities, and decide which companies deserve deeper validation."
      />

      {gates ? (
        <div
          className={clsx(
            "rounded-3xl border px-5 py-4",
            gates.universe ? "border-success/30 bg-success/10" : "border-warning/30 bg-warning/10"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.universe ? (
              <CheckCircle className="h-5 w-5 text-success" />
            ) : (
              <AlertCircle className="h-5 w-5 text-warning" />
            )}
            <span className={gates.universe ? "font-medium text-success" : "font-medium text-warning-dark"}>
              {gates.universe
                ? `${keptCount} companies kept`
                : "Universe output is still early. Use keep/remove to shape the list."}
            </span>
          </div>
        </div>
      ) : null}

      <section className="rounded-[28px] border border-steel-200 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              Retrieval Overview
            </div>
            <h3 className="mt-2 text-2xl font-semibold text-oxford">Graph-oriented discovery output</h3>
            <p className="mt-2 max-w-3xl text-sm text-steel-500">
              Universe now shows which capabilities, vertical hints, and retrieval paths surfaced each company, plus
              registry identity context before deeper enrichment.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <button onClick={() => setShowAddModal(true)} className="btn-secondary flex items-center gap-2">
              <Plus className="h-4 w-4" />
              Add Company
            </button>
            <button
              onClick={jobRunner.run}
              disabled={jobRunner.isRunning || !discoveryReadiness?.runnable}
              className="btn-primary flex items-center gap-2 disabled:opacity-50"
            >
              {jobRunner.isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Running… {Math.round(jobRunner.progress * 100)}%
                </>
              ) : (
                <>
                  <Search className="h-4 w-4" />
                  Run Discovery
                </>
              )}
            </button>
          </div>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-4">
          <MetricTile label="Surfaced" value={candidates.length} icon={Globe} />
          <MetricTile label="Good Target" value={goodCount} icon={Sparkles} />
          <MetricTile
            label="Registry Mapped"
            value={diagnostics?.funnel_metrics?.registry_identity_mapped_count ?? 0}
            icon={Database}
          />
          <MetricTile
            label="Registry Queries"
            value={diagnostics?.funnel_metrics?.registry_queries_count ?? 0}
            icon={Network}
          />
        </div>

        {providerMixLabels.length || registryCountryLabels.length ? (
          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <SignalChips title="Web Provider Mix" items={providerMixLabels} />
            <SignalChips title="Registry Query Mix" items={registryCountryLabels} tone="info" />
          </div>
        ) : null}

        {discoveryReadiness ? <DiscoveryReadinessPanel readiness={discoveryReadiness} /> : null}
      </section>

      {!jobRunner.isRunning ? <JobRunSummary job={latestCompletedDiscoveryJob} /> : null}

      {jobRunner.isRunning ? (
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
      ) : null}

      {showAddModal ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-oxford/80">
          <div className="w-full max-w-md rounded-3xl border border-steel-200 bg-white p-6 shadow-xl">
            <h3 className="text-lg text-oxford">Add Company Manually</h3>
            <div className="mt-4 space-y-4">
              <div>
                <label className="label">Company Name *</label>
                <input
                  type="text"
                  value={newCompany.name}
                  onChange={(event) => setNewCompany({ ...newCompany, name: event.target.value })}
                  className="input"
                />
              </div>
              <div>
                <label className="label">Website</label>
                <input
                  type="url"
                  value={newCompany.website}
                  onChange={(event) => setNewCompany({ ...newCompany, website: event.target.value })}
                  placeholder="https://company.com"
                  className="input"
                />
              </div>
              <div>
                <label className="label">HQ Country</label>
                <input
                  type="text"
                  value={newCompany.hq_country}
                  onChange={(event) => setNewCompany({ ...newCompany, hq_country: event.target.value })}
                  placeholder="e.g. France"
                  className="input"
                />
              </div>
            </div>
            <div className="mt-6 flex gap-3">
              <button onClick={() => setShowAddModal(false)} className="flex-1 btn-secondary">
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
      ) : null}

      <div className="flex flex-wrap gap-1 rounded-full border border-steel-200 bg-white p-1">
        {([
          { key: "all", label: `All (${candidates.length})` },
          { key: "good_target", label: `Good (${goodCount})` },
          { key: "borderline_watchlist", label: `Watchlist (${watchlistCount})` },
          { key: "kept", label: `Kept (${keptCount})` },
        ] as const).map((item) => (
          <button
            key={item.key}
            onClick={() => setFilter(item.key)}
            className={clsx(
              "rounded-full px-4 py-2 text-sm transition-colors",
              filter === item.key ? "bg-oxford text-white" : "text-steel-600 hover:bg-steel-50"
            )}
          >
            {item.label}
          </button>
        ))}
      </div>

      {canShowDegraded ? (
        <div className="rounded-3xl border border-warning/30 bg-warning/10 px-5 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="font-medium text-warning-dark">Latest discovery run is degraded</div>
              <p className="mt-1 text-sm text-warning-dark/80">
                The page is hiding the last run by default because the quality gate failed. You can still inspect it.
              </p>
            </div>
            <button onClick={() => setAllowDegraded(true)} className="btn-secondary">
              Show Degraded Run
            </button>
          </div>
        </div>
      ) : null}

      {degradedError && !canShowDegraded ? (
        <div className="rounded-3xl border border-danger/20 bg-danger/10 px-5 py-4 text-sm text-danger">
          {degradedError.message}
        </div>
      ) : null}

      {filteredCandidates.length === 0 ? (
        <div className="rounded-[28px] border border-steel-200 bg-white px-6 py-12 text-center">
          <Globe className="mx-auto h-12 w-12 text-steel-300" />
          <h3 className="mt-4 text-lg text-oxford">No surfaced candidates yet</h3>
          <p className="mt-2 text-sm text-steel-500">
            Run discovery to populate the universe from approved capabilities, customers, and adjacent lanes.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {filteredCandidates.map((candidate) => (
            <UniverseCandidateCard
              key={candidate.company_id ?? candidate.candidate_entity_id ?? candidate.company_name}
              candidate={candidate}
              onKeep={() => handleKeep(candidate)}
              onRemove={() => handleRemove(candidate)}
              onRestore={() => handleRestore(candidate)}
            />
          ))}
        </div>
      )}

      {manualCompanies.length ? (
        <section className="space-y-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">Manual Queue</div>
            <h3 className="mt-2 text-xl font-semibold text-oxford">Manual additions</h3>
            <p className="mt-1 text-sm text-steel-500">
              These companies were added directly and may not have discovery provenance yet.
            </p>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {manualCompanies.map((company) => (
              <ManualCompanyCard
                key={company.id}
                company={company}
                onRemove={() => handleManualRemove(company)}
                onRestore={() => handleManualRestore(company)}
              />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
