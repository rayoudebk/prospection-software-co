"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useParams } from "next/navigation";
import clsx from "clsx";
import {
  AlertCircle,
  ArrowRight,
  BadgeCheck,
  Building2,
  CheckCircle,
  ExternalLink,
  Globe,
  Landmark,
  Layers3,
  Loader2,
  MapPinned,
  Network,
  Search,
} from "lucide-react";

import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import { StepHeader } from "@/components/StepHeader";
import {
  DiscoveryReadiness,
  UniverseCandidate,
  UniverseCandidateRegistrySummary,
  workspaceApi,
} from "@/lib/api";
import {
  useDiscoveryDiagnostics,
  useGates,
  useRefreshValidationQueue,
  useUniverseCandidates,
  useWorkspaceJobs,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";

const READINESS_REASON_LABELS: Record<string, string> = {
  db_schema_invalid: "Discovery schema is missing required columns.",
  redis_unavailable: "Redis is unavailable.",
  worker_unavailable: "No Celery worker is responding.",
  retrieval_provider_unavailable: "No retrieval provider key is configured.",
  model_provider_unavailable: "No model provider key is configured.",
  expansion_not_confirmed: "Expansion brief is not confirmed yet.",
  company_profile_missing: "Company profile is missing.",
};

function titleCaseToken(value?: string | null) {
  return String(value || "")
    .replaceAll("::", " / ")
    .split(/[_\s/-]+/)
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

function compactText(value?: string | null) {
  const trimmed = String(value || "").trim();
  return trimmed || null;
}

function brandFirstName(candidate: UniverseCandidate) {
  return compactText(candidate.display_name) || compactText(candidate.legal_name) || "Unknown candidate";
}

function legalSecondaryName(candidate: UniverseCandidate) {
  const brand = brandFirstName(candidate).toLowerCase();
  const legal = compactText(candidate.legal_name);
  if (!legal || legal.toLowerCase() === brand) return null;
  return legal;
}

function registrySummaryEntries(registry?: UniverseCandidateRegistrySummary | null) {
  if (!registry) return [];
  const entries: Array<{ label: string; value: string }> = [];
  const apeCode = compactText(registry.ape_code);
  const naf25Code = compactText(registry.naf25_code);
  const status = compactText(registry.active_status);
  const country = compactText(registry.country) || compactText(registry.country_code);
  const observationCount =
    typeof registry.observation_count === "number" && registry.observation_count > 0
      ? String(registry.observation_count)
      : null;
  const commercialNames = (registry.commercial_names || []).slice(0, 2).join(", ");

  if (apeCode) entries.push({ label: "APE", value: apeCode });
  if (naf25Code) entries.push({ label: "NAF25", value: naf25Code });
  if (status) entries.push({ label: "Status", value: status });
  if (country) entries.push({ label: "Country", value: country });
  if (commercialNames) entries.push({ label: "Brands", value: commercialNames });
  if (observationCount) entries.push({ label: "Observations", value: observationCount });

  return entries.slice(0, 6);
}

function matchedNodeLabels(candidate: UniverseCandidate) {
  const labels = [
    ...(candidate.node_fit_summary?.matched_node_labels || []),
    ...(candidate.lane_labels || []),
    ...(candidate.lane_ids || []),
  ]
    .map((value) => compactText(value))
    .filter((value): value is string => Boolean(value));

  return Array.from(new Set(labels)).slice(0, 8);
}

function nodeFitSummaryText(candidate: UniverseCandidate) {
  const labels = matchedNodeLabels(candidate);
  const coreCount = Number(candidate.node_fit_summary?.core_match_count || 0);
  const adjacentCount = Number(candidate.node_fit_summary?.adjacent_match_count || 0);
  if (!labels.length && !coreCount && !adjacentCount) return null;
  if (labels.length) {
    return `${labels.slice(0, 3).join(", ")}${labels.length > 3 ? ` +${labels.length - 3} more` : ""}`;
  }
  return `Core overlap ${coreCount} • adjacent overlap ${adjacentCount}`;
}

function candidateCountries(candidate: UniverseCandidate) {
  return Array.from(
    new Set(
      [
        ...(candidate.geo_signals || []),
        compactText(candidate.registry_summary?.country),
        compactText(candidate.registry_summary?.country_code),
      ].filter((value): value is string => Boolean(value))
    )
  );
}

function isFranceLinked(candidate: UniverseCandidate) {
  const values = [
    ...candidateCountries(candidate),
    compactText(candidate.official_website_url),
    compactText(candidate.provenance_summary?.primary_market as string | null),
  ]
    .filter((value): value is string => Boolean(value))
    .map((value) => value.toLowerCase());
  return values.some((value) => value.includes("france") || value === "fr" || value.endsWith(".fr"));
}

function MetricTile({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  icon: typeof Globe;
}) {
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
        </div>
      </div>
    </div>
  );
}

function ChipList({
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

function RegistrySummaryCard({ candidate }: { candidate: UniverseCandidate }) {
  const entries = registrySummaryEntries(candidate.registry_summary);
  const summary = compactText(candidate.registry_summary?.summary);
  const registryId = compactText(candidate.registry_id);

  if (!entries.length && !summary && !registryId) return null;

  return (
    <div className="rounded-2xl border border-steel-200 bg-white/70 px-4 py-3">
      <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">
        <Landmark className="h-3.5 w-3.5" />
        Registry
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {registryId ? (
          <div className="rounded-xl border border-steel-100 bg-steel-50 px-3 py-2">
            <div className="text-[11px] uppercase tracking-[0.14em] text-steel-400">Registry ID</div>
            <div className="mt-1 text-sm text-oxford">{registryId}</div>
          </div>
        ) : null}
        {entries.map((entry) => (
          <div key={`${entry.label}-${entry.value}`} className="rounded-xl border border-steel-100 bg-steel-50 px-3 py-2">
            <div className="text-[11px] uppercase tracking-[0.14em] text-steel-400">{entry.label}</div>
            <div className="mt-1 text-sm text-oxford">{entry.value}</div>
          </div>
        ))}
      </div>
      {summary ? <p className="mt-3 text-sm leading-6 text-steel-600">{summary}</p> : null}
    </div>
  );
}

function UniverseCandidateCard({
  candidate,
  onValidate,
  pending,
}: {
  candidate: UniverseCandidate;
  onValidate: () => Promise<void>;
  pending: boolean;
}) {
  const displayName = brandFirstName(candidate);
  const legalName = legalSecondaryName(candidate);
  const countries = candidateCountries(candidate);
  const nodeLabels = matchedNodeLabels(candidate);
  const nodeFitText = nodeFitSummaryText(candidate);
  const discoveryHosts = (Array.isArray(candidate.provenance_summary?.top_source_urls) ? candidate.provenance_summary.top_source_urls : [])
    .map((source) => hostnameFor(source))
    .filter((host): host is string => Boolean(host))
    .slice(0, 5);

  return (
    <article className="space-y-4 rounded-3xl border border-steel-200 bg-white px-5 py-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-lg font-medium text-oxford">{displayName}</h3>
            <span className="rounded-full border border-info/20 bg-info/10 px-2 py-0.5 text-xs text-info">
              {titleCaseToken(candidate.directness)}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-steel-500">
            {countries.length ? <span>• {countries.join(", ")}</span> : null}
          </div>

          {legalName ? (
            <div className="text-sm text-steel-600">
              Legal entity <span className="font-medium text-oxford">{legalName}</span>
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
            {candidate.official_website_url ? (
              <a
                href={candidate.official_website_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-info hover:underline"
              >
                {hostnameFor(candidate.official_website_url) || candidate.official_website_url}
                <ExternalLink className="h-3 w-3" />
              </a>
            ) : null}
            {candidate.registry_id ? <span className="text-steel-500">Registry ID {candidate.registry_id}</span> : null}
          </div>
        </div>

        <div className="text-right">
          <div className="text-[11px] uppercase tracking-[0.18em] text-steel-400">Discovery Score</div>
          <div className="mt-1 text-2xl font-semibold text-oxford">{Math.round(candidate.discovery_score)}</div>
        </div>
      </div>

      <p className="text-sm leading-6 text-steel-600">
        {candidate.short_description ||
          "No short description yet. This candidate is in the registry because multiple discovery paths point to it."}
      </p>

      {nodeFitText ? (
        <div className="rounded-2xl border border-info/20 bg-info/10 px-4 py-3 text-sm text-info">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-info/80">Node Fit</div>
          <div className="mt-2 leading-6">{nodeFitText}</div>
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2">
        <ChipList title="Matched Nodes" items={nodeLabels} tone="info" />
        <ChipList title="Source Families" items={candidate.source_families.map(titleCaseToken)} />
        <ChipList title="Query Families" items={candidate.query_families.map(titleCaseToken)} tone="success" />
        <ChipList title="Discovery Hosts" items={discoveryHosts} />
      </div>

      <RegistrySummaryCard candidate={candidate} />

      <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3 text-sm text-steel-600">
        <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">Discovery provenance</div>
        <div className="mt-2">
          {String(candidate.provenance_summary?.origin_count || 0)} origins •{" "}
          {String(candidate.provenance_summary?.source_family_count || candidate.source_families.length)} source families •{" "}
          {String(candidate.provenance_summary?.query_family_count || candidate.query_families.length)} query families
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-steel-100 pt-4">
        <div className="text-xs text-steel-400">
          Universe is the registry stage. Use Validation for the first homepage and legal-fit pass.
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={onValidate}
            disabled={pending}
            className="btn-secondary px-3 py-1.5 text-sm disabled:opacity-50"
          >
            {pending ? "Queuing..." : "Queue for Validation"}
          </button>
          <Link href={`./validation`} className="btn-secondary flex items-center gap-2 px-3 py-1.5 text-sm">
            Open Validation
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>
    </article>
  );
}

export default function UniversePage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const [directnessFilter, setDirectnessFilter] = useState<"all" | "direct" | "adjacent" | "broad_market">("all");

  const {
    data: topCandidates,
    isLoading: topCandidatesLoading,
    error: topCandidatesError,
    refetch: refetchTopCandidates,
  } = useUniverseCandidates(workspaceId, 90, true);
  const { data: diagnostics, refetch: refetchDiagnostics } = useDiscoveryDiagnostics(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const { data: discoveryJobs } = useWorkspaceJobs(workspaceId, "discovery_universe");
  const refreshValidation = useRefreshValidationQueue(workspaceId);

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.runDiscovery(workspaceId),
    () => {
      void refetchTopCandidates();
      void refetchDiagnostics();
    },
    (jobId) => workspaceApi.cancelJob(workspaceId, jobId)
  );

  const candidates = useMemo(() => topCandidates ?? [], [topCandidates]);
  const filteredCandidates = useMemo(() => {
    if (directnessFilter === "all") return candidates;
    return candidates.filter((candidate) => candidate.directness === directnessFilter);
  }, [candidates, directnessFilter]);

  const latestCompletedDiscoveryJob = discoveryJobs?.find((job) => job.state === "completed") ?? null;
  const discoveryReadiness = gates?.discovery_readiness;
  const providerMix = diagnostics?.source_coverage?.external_search?.provider_mix || {};
  const providerMixLabels = Object.entries(providerMix)
    .filter(([, count]) => Number(count) > 0)
    .map(([provider, count]) => `${titleCaseToken(provider)} ${count}`);
  const directCount = candidates.filter((candidate) => candidate.directness === "direct").length;
  const franceLinkedCount = candidates.filter((candidate) => isFranceLinked(candidate)).length;
  const registryLinkedCount = candidates.filter((candidate) => Boolean(candidate.registry_id || candidate.registry_summary)).length;
  const uniqueSourceFamilies = new Set(candidates.flatMap((candidate) => candidate.source_families)).size;
  const degradedError = topCandidatesError as (Error & { status?: number }) | null;

  if (topCandidatesLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-8 w-8 animate-spin text-oxford" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <StepHeader
        step={3}
        title="Universe"
        subtitle="Build the longlist as a brand-first candidate registry before sending selected companies into Validation."
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
                ? `${candidates.length} candidates are available in the discovery registry.`
                : "Universe is the registry stage. It should stay broad, cheap, and provenance-rich."}
            </span>
          </div>
        </div>
      ) : null}

      <section className="rounded-[28px] border border-steel-200 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              Market Registry
            </div>
            <h3 className="mt-2 text-2xl font-semibold text-oxford">Brand-first longlist around approved nodes</h3>
            <p className="mt-2 max-w-3xl text-sm text-steel-500">
              Universe now acts as the market registry: brand-first rows, legal metadata secondary, and cheap relevance
              only. Validation is where the first real homepage and fit pass happens.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
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
            <Link href={`./validation`} className="btn-secondary flex items-center gap-2">
              Validation
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-5">
          <MetricTile label="Candidates" value={candidates.length} icon={Building2} />
          <MetricTile label="France-linked" value={franceLinkedCount} icon={MapPinned} />
          <MetricTile label="Registry-linked" value={registryLinkedCount} icon={Landmark} />
          <MetricTile label="Direct" value={directCount} icon={BadgeCheck} />
          <MetricTile label="Source Families" value={uniqueSourceFamilies} icon={Layers3} />
        </div>

        {providerMixLabels.length ? (
          <div className="mt-5">
            <ChipList title="Web Provider Mix" items={providerMixLabels} />
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

      {degradedError ? (
        <div className="rounded-3xl border border-warning/30 bg-warning/10 px-5 py-4 text-sm text-warning-dark">
          {degradedError.message || "Universe candidates could not be loaded."}
        </div>
      ) : null}

      <div className="rounded-3xl border border-steel-200 bg-white px-5 py-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-oxford">Universe registry</h2>
            <p className="mt-1 text-sm text-steel-500">
              Browse the longlist by directness, matched nodes, geography, and discovery provenance. No shortlist or
              quality semantics live here.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {[
              { key: "all", label: "All" },
              { key: "direct", label: "Direct" },
              { key: "adjacent", label: "Adjacent" },
              { key: "broad_market", label: "Broad market" },
            ].map((option) => (
              <button
                key={option.key}
                onClick={() => setDirectnessFilter(option.key as typeof directnessFilter)}
                className={clsx(
                  "rounded-full border px-3 py-1.5 text-sm transition-colors",
                  directnessFilter === option.key
                    ? "border-oxford bg-oxford text-white"
                    : "border-steel-200 bg-white text-steel-600 hover:border-info/30 hover:bg-info/10 hover:text-info"
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {!filteredCandidates.length ? (
        <div className="rounded-3xl border border-steel-200 bg-white px-6 py-12 text-center text-sm text-steel-500">
          No candidates match this filter yet.
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          {filteredCandidates.map((candidate) => (
            <UniverseCandidateCard
              key={candidate.candidate_entity_id}
              candidate={candidate}
              pending={refreshValidation.isPending}
              onValidate={() =>
                refreshValidation
                  .mutateAsync({ candidateEntityIds: [candidate.candidate_entity_id] })
                  .then(() => undefined)
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
