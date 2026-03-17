"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import clsx from "clsx";
import {
  AlertCircle,
  Check,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Loader2,
  Plus,
  RefreshCw,
  X,
} from "lucide-react";

import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import { ReportArtifactRenderer } from "@/components/ReportArtifactRenderer";
import { StepHeader } from "@/components/StepHeader";
import { TaxonomyNode, workspaceApi } from "@/lib/api";
import {
  useContextPack,
  useCompanyContextPack,
  useGates,
  useRefreshCompanyContextPack,
  useUpdateContextPack,
  useUpdateCompanyContextPack,
  useWorkspaceJobs,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";

function normalizeUrlInput(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const candidate =
    trimmed.startsWith("http://") || trimmed.startsWith("https://")
      ? trimmed
      : `https://${trimmed}`;
  try {
    const parsed = new URL(candidate);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    const query = parsed.search || "";
    return `${parsed.protocol}//${parsed.host}${parsed.pathname}${query}`;
  } catch {
    return null;
  }
}

function getSourceDisplayUrl(url: string): string {
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.replace(/^www\./, "");
    const path = `${parsed.pathname}${parsed.search}`.replace(/\/$/, "");
    return `${hostname}${path === "" || path === "/" ? "" : path}`;
  } catch {
    return url;
  }
}

function UrlEditor({
  title,
  helper,
  urls,
  newUrl,
  onNewUrlChange,
  onAdd,
  onRemove,
  error,
  placeholder,
}: {
  title: string;
  helper: string;
  urls: string[];
  newUrl: string;
  onNewUrlChange: (value: string) => void;
  onAdd: () => void;
  onRemove: (index: number) => void;
  error: string | null;
  placeholder: string;
}) {
  return (
    <div className="space-y-2.5">
      <label className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
        {title}
      </label>
      <p className="text-sm text-steel-500">{helper}</p>

      {urls.length ? (
        <div className="flex flex-wrap gap-2">
          {urls.map((url, index) => (
            <span
              key={`${url}-${index}`}
              className="inline-flex max-w-full items-center gap-2 rounded-full border border-steel-200 bg-steel-50 px-3 py-1.5 text-xs text-steel-600"
            >
              <span className="truncate font-mono">{getSourceDisplayUrl(url)}</span>
              <button
                type="button"
                onClick={() => onRemove(index)}
                className="text-steel-400 transition-colors hover:text-danger"
                aria-label={`Remove ${title} URL`}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </span>
          ))}
        </div>
      ) : null}

      <div className="flex gap-2">
        <input
          type="url"
          value={newUrl}
          onChange={(event) => onNewUrlChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onAdd();
            }
          }}
          placeholder={placeholder}
          className="input h-10 text-sm"
        />
        <button
          type="button"
          onClick={onAdd}
          className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-steel-200 bg-white text-steel-600 transition-colors hover:border-oxford hover:text-oxford"
          aria-label={`Add ${title} URL`}
        >
          <Plus className="h-4 w-4" />
        </button>
      </div>

      {error ? <p className="text-xs text-danger">{error}</p> : null}
    </div>
  );
}

function NodeGroup({
  title,
  nodes,
}: {
  title: string;
  nodes: TaxonomyNode[];
}) {
  if (!nodes.length) return null;

  return (
    <section className="space-y-3">
      <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
        {title}
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {nodes.slice(0, 6).map((node) => (
          <article key={node.id} className="rounded-3xl border border-steel-200 bg-white px-5 py-4">
            <p className="text-lg leading-7 text-oxford">{node.phrase}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function SourcingBriefPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: profile, isLoading } = useContextPack(workspaceId);
  const { data: companyContext } = useCompanyContextPack(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const { data: contextJobs } = useWorkspaceJobs(workspaceId, "context_pack");
  const updateProfile = useUpdateContextPack(workspaceId);
  const updateCompanyContext = useUpdateCompanyContextPack(workspaceId);
  const refreshCompanyContext = useRefreshCompanyContextPack(workspaceId);
  const isCompanyContextRefreshing =
    refreshCompanyContext.isPending || companyContext?.graph_status === "refreshing";

  const [buyerUrl, setBuyerUrl] = useState("");
  const [referenceUrls, setReferenceUrls] = useState<string[]>([]);
  const [newReferenceUrl, setNewReferenceUrl] = useState("");
  const [referenceUrlError, setReferenceUrlError] = useState<string | null>(null);
  const [evidenceUrls, setEvidenceUrls] = useState<string[]>([]);
  const [newEvidenceUrl, setNewEvidenceUrl] = useState("");
  const [evidenceUrlError, setEvidenceUrlError] = useState<string | null>(null);
  const [isInputPanelCollapsed, setIsInputPanelCollapsed] = useState(false);

  useEffect(() => {
    if (!profile) return;
    setBuyerUrl(profile.buyer_company_url || "");
    setReferenceUrls(profile.comparator_seed_urls || []);
    setEvidenceUrls(profile.supporting_evidence_urls || []);
  }, [profile]);

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.refreshContextPack(workspaceId),
    () => {
      refreshCompanyContext.mutate();
    },
    (jobId) => workspaceApi.cancelJob(workspaceId, jobId)
  );

  const latestCompletedContextJob = useMemo(
    () => contextJobs?.find((job) => job.state === "completed") ?? null,
    [contextJobs]
  );

  const hasGeneratedContextPack = Boolean(
    profile?.context_pack_generated_at ||
      profile?.context_pack_markdown ||
      profile?.context_pack_json
  );

  const hasSourcingArtifact = Boolean(
    companyContext?.sourcing_report || companyContext?.sourcing_brief?.source_summary
  );

  useEffect(() => {
    setIsInputPanelCollapsed(hasSourcingArtifact);
  }, [hasSourcingArtifact]);

  const buyerEvidence = companyContext?.buyer_evidence || null;
  const showBuyerEvidenceWarning = buyerEvidence?.status === "insufficient";
  const crawlButtonLabel = profile?.context_pack_generated_at
    ? "Recrawl & Update Brief"
    : "Generate Sourcing Brief";

  const inputSummary = useMemo(() => {
    const sourceLabel = buyerUrl.trim()
      ? getSourceDisplayUrl(normalizeUrlInput(buyerUrl) || buyerUrl)
      : "Add source company website";
    return [
      sourceLabel,
      `${referenceUrls.length} competitor${referenceUrls.length === 1 ? "" : "s"}`,
      `${evidenceUrls.length} support link${evidenceUrls.length === 1 ? "" : "s"}`,
    ].join(" · ");
  }, [buyerUrl, referenceUrls.length, evidenceUrls.length]);

  const saveProfileInputs = async () => {
    await updateProfile.mutateAsync({
      buyer_company_url: buyerUrl,
      comparator_seed_urls: referenceUrls,
      supporting_evidence_urls: evidenceUrls,
    });
  };

  const confirmBrief = async () => {
    await updateCompanyContext.mutateAsync({ confirmed: true });
  };

  const handleAddReference = () => {
    const normalized = normalizeUrlInput(newReferenceUrl);
    if (!normalized) {
      setReferenceUrlError("Enter a valid URL.");
      return;
    }
    if (referenceUrls.some((url) => url.toLowerCase() === normalized.toLowerCase())) {
      setReferenceUrlError("That competitor URL is already added.");
      return;
    }
    setReferenceUrls([...referenceUrls, normalized]);
    setNewReferenceUrl("");
    setReferenceUrlError(null);
  };

  const handleAddEvidence = () => {
    const normalized = normalizeUrlInput(newEvidenceUrl);
    if (!normalized) {
      setEvidenceUrlError("Enter a valid supporting-evidence URL.");
      return;
    }
    if (evidenceUrls.some((url) => url.toLowerCase() === normalized.toLowerCase())) {
      setEvidenceUrlError("That evidence URL is already added.");
      return;
    }
    setEvidenceUrls([...evidenceUrls, normalized]);
    setNewEvidenceUrl("");
    setEvidenceUrlError(null);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-6 w-6 animate-spin text-oxford" />
      </div>
    );
  }

  const sourcingBrief = companyContext?.sourcing_brief;

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <StepHeader
          step={1}
          title="Source & Brief"
          subtitle="Enter the company website and optional comparators, generate the sourcing brief, then validate the source-grounded market summary."
        />
        {gates ? (
          <div
            className={clsx(
              "mt-1 flex shrink-0 items-center gap-1.5 text-xs",
              gates.context_pack ? "text-success" : "text-warning"
            )}
          >
            {gates.context_pack ? (
              <>
                <CheckCircle className="h-3.5 w-3.5" />
                <span className="font-medium">Validated</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3.5 w-3.5" />
                <span className="font-medium">Needs review</span>
              </>
            )}
          </div>
        ) : null}
      </div>

      <section className="rounded-[28px] border border-steel-200 bg-white px-5 py-5 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              Inputs
            </div>
            <h3 className="mt-2 text-lg font-semibold text-oxford">
              Company website, comparators, and support links
            </h3>
            <p className="mt-1 text-sm text-steel-500">{inputSummary}</p>
          </div>
          <button
            type="button"
            onClick={() => setIsInputPanelCollapsed((current) => !current)}
            className="inline-flex items-center gap-2 rounded-full border border-steel-200 bg-steel-50 px-3 py-1.5 text-sm font-medium text-steel-700 transition-colors hover:border-oxford hover:text-oxford"
            aria-expanded={!isInputPanelCollapsed}
          >
            {isInputPanelCollapsed ? "Expand" : "Collapse"}
            {isInputPanelCollapsed ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronUp className="h-4 w-4" />
            )}
          </button>
        </div>

        {!isInputPanelCollapsed || jobRunner.isRunning ? (
          <div className="mt-5 space-y-5 border-t border-steel-100 pt-5">
            <div className="space-y-2.5">
              <label className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                Source company website
              </label>
              <p className="text-sm text-steel-500">
                The main company URL used for crawl, graph construction, and the sourcing brief.
              </p>
              <input
                type="url"
                value={buyerUrl}
                onChange={(event) => setBuyerUrl(event.target.value)}
                placeholder="https://company.com"
                className="input h-10 text-sm"
              />
            </div>

            <div className="grid gap-5 lg:grid-cols-2">
              <UrlEditor
                title="Competitors"
                helper="Optional comparable company URLs used to sharpen nearby market context."
                urls={referenceUrls}
                newUrl={newReferenceUrl}
                onNewUrlChange={(value) => {
                  setNewReferenceUrl(value);
                  setReferenceUrlError(null);
                }}
                onAdd={handleAddReference}
                onRemove={(index) =>
                  setReferenceUrls(referenceUrls.filter((_, itemIndex) => itemIndex !== index))
                }
                error={referenceUrlError}
                placeholder="https://comparable-company.com"
              />

              <UrlEditor
                title="Support Links"
                helper="Optional product pages, docs, customer stories, or PDFs worth forcing into the crawl."
                urls={evidenceUrls}
                newUrl={newEvidenceUrl}
                onNewUrlChange={(value) => {
                  setNewEvidenceUrl(value);
                  setEvidenceUrlError(null);
                }}
                onAdd={handleAddEvidence}
                onRemove={(index) =>
                  setEvidenceUrls(evidenceUrls.filter((_, itemIndex) => itemIndex !== index))
                }
                error={evidenceUrlError}
                placeholder="https://company.com/customer-story"
              />
            </div>

            <div className="flex flex-wrap gap-2 border-t border-steel-100 pt-4">
              <button
                type="button"
                onClick={async () => {
                  await saveProfileInputs();
                  jobRunner.run();
                }}
                disabled={updateProfile.isPending || jobRunner.isRunning || !buyerUrl.trim()}
                className="btn-primary gap-2 px-4 py-2 text-sm disabled:opacity-50"
              >
                {jobRunner.isRunning ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running… {Math.round(jobRunner.progress * 100)}%
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4" />
                    {crawlButtonLabel}
                  </>
                )}
              </button>

              {hasGeneratedContextPack ? (
                <button
                  type="button"
                  onClick={async () => {
                    await saveProfileInputs();
                    await refreshCompanyContext.mutateAsync();
                  }}
                  disabled={updateProfile.isPending || isCompanyContextRefreshing}
                  className="btn-secondary gap-1.5 px-4 py-2 text-sm disabled:opacity-50"
                >
                  {isCompanyContextRefreshing ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5" />
                  )}
                  Refresh Brief
                </button>
              ) : null}
            </div>

            {jobRunner.isRunning ? (
              <JobProgressPanel
                job={
                  jobRunner.job ?? {
                    id: 0,
                    workspace_id: workspaceId,
                    company_id: null,
                    job_type: "context_pack",
                    state: "queued",
                    provider: "crawler",
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
            ) : (
              <JobRunSummary job={latestCompletedContextJob} />
            )}

            {jobRunner.jobError ? (
              <div className="rounded-2xl border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger">
                {jobRunner.jobError}
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      {showBuyerEvidenceWarning ? (
        <div className="rounded-3xl border border-warning/30 bg-warning/10 px-5 py-4 text-sm text-warning-dark">
          {buyerEvidence?.warning ||
            "Buyer evidence is still thin. Add stronger first-party product, docs, or customer-proof links before validating the brief."}
        </div>
      ) : null}

      <section className="space-y-5 rounded-[28px] border border-steel-200 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              Sourcing Brief
            </div>
            <h3 className="mt-2 font-serif text-2xl text-oxford">
              Small report backed by crawl evidence
            </h3>
            {companyContext?.generated_at ? (
              <p className="mt-1 text-xs text-steel-400">
                Generated {new Date(companyContext.generated_at).toLocaleString()}
              </p>
            ) : null}
          </div>
          {companyContext?.confirmed_at ? (
            <span className="rounded-full border border-success/25 bg-success/10 px-3 py-1 text-xs font-medium text-success">
              Confirmed
            </span>
          ) : null}
        </div>

        {companyContext?.sourcing_report ? (
          <ReportArtifactRenderer artifact={companyContext.sourcing_report} />
        ) : (
          <div className="rounded-2xl border border-dashed border-steel-200 bg-steel-50 px-4 py-5 text-sm text-steel-500">
            Generate a sourcing brief to render the report.
          </div>
        )}

        {sourcingBrief ? (
          <div className="space-y-6 border-t border-steel-100 pt-5">
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
                Major Graph Nodes
              </div>
              <p className="mt-2 text-sm text-steel-500">
                These are the main source-grounded nodes extracted from the company graph. Keep this section simple and use it as a quick pressure test of the brief.
              </p>
            </div>

            <NodeGroup title="Customer Signals" nodes={sourcingBrief.customer_nodes || []} />
            <NodeGroup title="Workflow Signals" nodes={sourcingBrief.workflow_nodes || []} />
            <NodeGroup title="Capability Signals" nodes={sourcingBrief.capability_nodes || []} />
            <NodeGroup
              title="Integration Signals"
              nodes={sourcingBrief.delivery_or_integration_nodes || []}
            />
          </div>
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-steel-100 pt-4">
          <div className="text-sm text-steel-500">
            {companyContext?.confirmed_at
              ? `Brief confirmed ${new Date(companyContext.confirmed_at).toLocaleString()}`
              : "Validate the sourcing brief once the report and major graph nodes look right."}
          </div>
          <button
            type="button"
            onClick={confirmBrief}
            disabled={updateCompanyContext.isPending || !hasSourcingArtifact}
            className="btn-primary gap-2 disabled:opacity-50"
          >
            <Check className="h-4 w-4" />
            Validate Brief
          </button>
        </div>
      </section>
    </div>
  );
}
