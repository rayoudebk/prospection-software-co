"use client";

import { useMemo, useState } from "react";
import { useParams } from "next/navigation";
import clsx from "clsx";
import {
  AlertCircle,
  Check,
  CheckCircle,
  ExternalLink,
  Globe,
  Loader2,
  Plus,
  Search,
  X,
} from "lucide-react";

import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import { StepHeader } from "@/components/StepHeader";
import { Company, workspaceApi } from "@/lib/api";
import {
  useCompanies,
  useCreateCompany,
  useGates,
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

function classificationBadgeClass(classification?: string | null) {
  if (classification === "good_target") return "badge-success";
  if (classification === "borderline_watchlist") return "badge-warning";
  if (classification === "not_good_target") return "badge-danger";
  return "badge-neutral";
}

function CitedSummary({ company }: { company: Company }) {
  const summary = company.citation_summary_v1;
  const fallback =
    company.rationale_summary ||
    company.why_fit_bullets?.[0]?.text ||
    company.why_relevant?.[0]?.text ||
    "No company description generated yet.";

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
        <p key={`${company.id}-${sentence.id}`}>
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

function CompanyCard({
  company,
  onKeep,
  onRemove,
  onRestore,
}: {
  company: Company;
  onKeep: () => Promise<void>;
  onRemove: () => Promise<void>;
  onRestore: () => Promise<void>;
}) {
  return (
    <article className="rounded-3xl border border-steel-200 bg-white px-5 py-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h3 className="text-lg text-oxford">{company.name}</h3>
          <p className="text-sm text-steel-500">
            {company.hq_country || "Unknown country"}
            {company.status ? ` · ${company.status}` : ""}
          </p>
          {(company.official_website_url || company.website) ? (
            <a
              href={company.official_website_url || company.website || undefined}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-info hover:underline"
            >
              {new URL((company.official_website_url || company.website) as string).hostname}
              <ExternalLink className="h-3 w-3" />
            </a>
          ) : null}
        </div>

        <span className={clsx("badge", classificationBadgeClass(company.decision_classification))}>
          {CLASSIFICATION_LABELS[company.decision_classification || "insufficient_evidence"] ||
            "Insufficient evidence"}
        </span>
      </div>

      <CitedSummary company={company} />

      {company.top_claim?.text && company.top_claim?.source_url ? (
        <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">
            Top evidence line
          </div>
          <p className="mt-2 text-sm leading-6 text-steel-700">{company.top_claim.text}</p>
        </div>
      ) : null}

      <div className="flex items-center justify-between border-t border-steel-100 pt-4">
        <span className="text-xs text-steel-400">{company.evidence_count} citations</span>
        <div className="flex gap-2">
          {company.status === "removed" ? (
            <button onClick={onRestore} className="btn-secondary text-sm py-1 px-3">
              Restore
            </button>
          ) : (
            <>
              <button
                onClick={onRemove}
                className="inline-flex items-center gap-1 rounded-full border border-steel-200 px-3 py-1.5 text-sm text-steel-700 transition-colors hover:border-danger/30 hover:bg-danger/10 hover:text-danger"
              >
                <X className="h-4 w-4" />
                Remove
              </button>
              <button
                onClick={onKeep}
                disabled={company.status === "kept" || company.status === "enriched"}
                className={clsx(
                  "inline-flex items-center gap-1 rounded-full border px-3 py-1.5 text-sm transition-colors disabled:opacity-50",
                  company.status === "kept" || company.status === "enriched"
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
    </article>
  );
}

export default function UniversePage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: companies, isLoading, refetch } = useCompanies(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const { data: discoveryJobs } = useWorkspaceJobs(workspaceId, "discovery_universe");
  const updateCompany = useUpdateCompany(workspaceId);
  const createCompany = useCreateCompany(workspaceId);

  const [filter, setFilter] = useState<"all" | "kept" | "good_target" | "borderline_watchlist">("all");
  const [showAddModal, setShowAddModal] = useState(false);
  const [newCompany, setNewCompany] = useState({ name: "", website: "", hq_country: "" });

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.runDiscovery(workspaceId),
    () => refetch(),
    (jobId) => workspaceApi.cancelJob(workspaceId, jobId)
  );

  const nonSolutionCompanies = useMemo(
    () =>
      companies?.filter((company) => (company.entity_type || "company") !== "solution") || [],
    [companies]
  );

  const filteredCompanies = useMemo(() => {
    if (filter === "all") return nonSolutionCompanies;
    if (filter === "kept") {
      return nonSolutionCompanies.filter(
        (company) => company.status === "kept" || company.status === "enriched"
      );
    }
    return nonSolutionCompanies.filter(
      (company) => (company.decision_classification || "insufficient_evidence") === filter
    );
  }, [filter, nonSolutionCompanies]);

  const keptCount =
    nonSolutionCompanies.filter((company) => company.status === "kept" || company.status === "enriched").length || 0;
  const goodCount =
    nonSolutionCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "good_target").length || 0;
  const watchlistCount =
    nonSolutionCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "borderline_watchlist").length || 0;
  const latestCompletedDiscoveryJob =
    discoveryJobs?.find((job) => job.state === "completed") ?? null;

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

  if (isLoading) {
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
        subtitle="Review the companies generated from the validated expansion focus. This is intentionally a simple company list for now, not a full dossier surface."
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
              Company List
            </div>
            <h3 className="mt-2 text-2xl font-semibold text-oxford">
              Basic discovery output
            </h3>
            <p className="mt-2 max-w-3xl text-sm text-steel-500">
              The backend is still maturing here, so the goal is a short evidence-backed company description plus basic keep/remove triage.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setShowAddModal(true)}
              className="btn-secondary flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              Add Company
            </button>
            <button
              onClick={jobRunner.run}
              disabled={jobRunner.isRunning}
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

        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
            <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Generated</div>
            <div className="mt-1 text-2xl font-semibold text-oxford">{nonSolutionCompanies.length}</div>
          </div>
          <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
            <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Good target</div>
            <div className="mt-1 text-2xl font-semibold text-oxford">{goodCount}</div>
          </div>
          <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
            <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Kept</div>
            <div className="mt-1 text-2xl font-semibold text-oxford">{keptCount}</div>
          </div>
        </div>
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
          { key: "all", label: `All (${nonSolutionCompanies.length})` },
          { key: "good_target", label: `Good (${goodCount})` },
          { key: "borderline_watchlist", label: `Watchlist (${watchlistCount})` },
          { key: "kept", label: `Kept (${keptCount})` },
        ] as const).map((item) => (
          <button
            key={item.key}
            onClick={() => setFilter(item.key)}
            className={clsx(
              "rounded-full px-4 py-2 text-sm transition-colors",
              filter === item.key
                ? "bg-oxford text-white"
                : "text-steel-600 hover:bg-steel-50"
            )}
          >
            {item.label}
          </button>
        ))}
      </div>

      {filteredCompanies.length === 0 ? (
        <div className="rounded-[28px] border border-steel-200 bg-white px-6 py-12 text-center">
          <Globe className="mx-auto h-12 w-12 text-steel-300" />
          <h3 className="mt-4 text-lg text-oxford">No companies yet</h3>
          <p className="mt-2 text-sm text-steel-500">
            Run discovery to populate the first universe list.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {filteredCompanies.map((company) => (
            <CompanyCard
              key={company.id}
              company={company}
              onKeep={() => handleKeep(company)}
              onRemove={() => handleRemove(company)}
              onRestore={() => handleRestore(company)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
