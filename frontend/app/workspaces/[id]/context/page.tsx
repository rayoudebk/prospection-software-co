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
  ExternalLink,
  Globe,
  Loader2,
  Plus,
  RefreshCw,
  X,
} from "lucide-react";

import { StepHeader } from "@/components/StepHeader";
import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import { ReportArtifactRenderer } from "@/components/ReportArtifactRenderer";
import {
  ContextPackV2,
  SourceDocument,
  workspaceApi,
} from "@/lib/api";
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

type SourceDrawerItem = SourceDocument & {
  hostname: string;
  displayUrl: string;
  badge: string | null;
};

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

function getSourceHostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
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

function getSourceBadge(label: string): string | null {
  if (label === "Buyer website") return "Buyer";
  if (label.startsWith("Comparator seed:")) return "Comparator";
  if (label.startsWith("Evidence source:")) return "Evidence";
  return null;
}

function SourcesDrawer({
  isOpen,
  onClose,
  sources,
}: {
  isOpen: boolean;
  onClose: () => void;
  sources: SourceDrawerItem[];
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        aria-label="Close sources panel"
        onClick={onClose}
        className="absolute inset-0 bg-oxford/20 backdrop-blur-[2px]"
      />

      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="sources-panel-title"
        className="absolute inset-y-0 right-0 flex w-full max-w-xl flex-col border-l border-white/10 bg-[#111315] text-white shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-white/45">
              Crawl sources
            </p>
            <h4 id="sources-panel-title" className="mt-1 text-2xl font-semibold">
              {sources.length} source{sources.length === 1 ? "" : "s"}
            </h4>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 text-white/70 transition-colors hover:border-white/20 hover:bg-white/5 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4">
          <div className="space-y-3">
            {sources.map((source) => (
              <a
                key={source.id}
                href={source.url || "#"}
                target="_blank"
                rel="noopener noreferrer"
                className="group block rounded-2xl border border-white/10 bg-white/[0.03] p-4 transition-colors hover:border-white/20 hover:bg-white/[0.06]"
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-white/[0.06] text-white/65">
                    <Globe className="h-4 w-4" />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      {source.badge ? (
                        <span className="rounded-full border border-white/10 bg-white/[0.06] px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.18em] text-white/70">
                          {source.badge}
                        </span>
                      ) : null}
                      <span className="text-[11px] uppercase tracking-[0.22em] text-white/45">
                        {source.hostname}
                      </span>
                    </div>

                    <div className="text-base font-semibold leading-snug text-white">
                      {source.name}
                    </div>
                    <div className="mt-1 break-all text-sm text-white/58">
                      {source.displayUrl}
                    </div>
                  </div>

                  <ExternalLink className="mt-1 h-4 w-4 shrink-0 text-white/40 transition-colors group-hover:text-white/75" />
                </div>
              </a>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}

function UrlEditor({
  title,
  helper,
  badgeTone,
  badgeLabel,
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
  badgeTone: string;
  badgeLabel: string;
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
      <label className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
          {title}
        </span>
        <span className={clsx("text-[11px]", badgeTone)}>{badgeLabel}</span>
      </label>

      <p className="text-sm text-steel-500">{helper}</p>

      {urls.length ? (
        <div className="flex flex-wrap gap-2">
          {urls.map((url, index) => (
            <span
              key={`${url}-${index}`}
              className="inline-flex max-w-full items-center gap-2 rounded-full border border-steel-200 bg-steel-50 px-3 py-1.5 text-xs text-steel-600"
            >
              <Globe className="h-3 w-3 shrink-0 text-steel-400" />
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
  const [isSourcesDrawerOpen, setIsSourcesDrawerOpen] = useState(false);
  const [isInputPanelCollapsed, setIsInputPanelCollapsed] = useState(false);

  useEffect(() => {
    if (!profile) return;
    setBuyerUrl(profile.buyer_company_url || "");
    setReferenceUrls(profile.comparator_seed_urls || []);
    setEvidenceUrls(profile.supporting_evidence_urls || []);
  }, [profile]);

  useEffect(() => {
    if (!isSourcesDrawerOpen) return;

    const { overflow } = document.body.style;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsSourcesDrawerOpen(false);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = overflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isSourcesDrawerOpen]);

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
    if (hasSourcingArtifact) {
      setIsInputPanelCollapsed(true);
      return;
    }
    setIsInputPanelCollapsed(false);
  }, [
    hasSourcingArtifact,
    companyContext?.generated_at,
    companyContext?.sourcing_report?.generated_at,
  ]);

  const sourceDrawerItems = useMemo(
    () =>
      (companyContext?.source_documents || []).map((pill) => ({
        ...pill,
        hostname: getSourceHostname(pill.url || ""),
        displayUrl: getSourceDisplayUrl(pill.url || ""),
        badge: getSourceBadge(pill.name),
      })),
    [companyContext?.source_documents]
  );

  const contextPack = (companyContext?.context_pack_v2 ||
    profile?.context_pack_json ||
    null) as ContextPackV2 | null;

  const buyerEvidence = companyContext?.buyer_evidence || null;
  const showBuyerEvidenceWarning = buyerEvidence?.status === "insufficient";
  const crawlButtonLabel = profile?.context_pack_generated_at
    ? "Recrawl and update map"
    : "Generate map from website";
  const outputMetaLine = companyContext?.generated_at
    ? [
        `Last generated ${new Date(companyContext.generated_at).toLocaleString()}`,
        contextPack?.crawl_coverage?.total_pages
          ? `${contextPack.crawl_coverage.total_pages} pages analyzed`
          : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : "Generate a market-map brief from a company website to populate this panel";
  const secondaryEvidenceCount =
    typeof companyContext?.graph_stats?.secondary_evidence_count === "number"
      ? companyContext.graph_stats.secondary_evidence_count
      : null;

  const inputSummary = useMemo(() => {
    const sourceLabel = buyerUrl.trim()
      ? getSourceHostname(normalizeUrlInput(buyerUrl) || buyerUrl)
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

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <StepHeader
          step={1}
          title="Sourcing Brief"
          subtitle="Ground the source-company understanding in primary and secondary evidence, then review the brief artifact that will bound downstream expansion."
        />
        {gates ? (
          <div
            className={clsx(
              "mt-1 flex shrink-0 items-center gap-1.5 text-xs",
              gates.context_pack ? "text-success" : "text-steel-400"
            )}
          >
            {gates.context_pack ? (
              <>
                <CheckCircle className="h-3.5 w-3.5" />
                <span className="font-medium">Ready</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3.5 w-3.5 text-warning" />
                <span className="font-medium text-warning">Incomplete</span>
              </>
            )}
          </div>
        ) : null}
      </div>

      <div className="rounded-[28px] border border-steel-200 bg-white px-5 py-5 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              Crawl setup
            </div>
            <h3 className="mt-2 text-lg font-semibold text-oxford">
              Source company and supporting links
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
            <div className="grid gap-5 lg:grid-cols-2">
              <div className="space-y-2.5">
                <label className="flex items-center justify-between">
                  <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                    Source company website
                  </span>
                  <span className="text-[11px] text-warning">Required</span>
                </label>
                <p className="text-sm text-steel-500">
                  The primary company URL used for the crawl and sourcing brief.
                </p>
                <input
                  type="url"
                  value={buyerUrl}
                  onChange={(event) => setBuyerUrl(event.target.value)}
                  placeholder="https://company.com"
                  className="input h-10 text-sm"
                />
              </div>

              <UrlEditor
                title="Competitor seeds"
                helper="Optional comparable company URLs to help bound nearby companies."
                badgeTone="text-steel-400"
                badgeLabel="Optional"
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
            </div>

            <UrlEditor
              title="Supporting evidence"
              helper="High-value first-party pages, customer stories, docs, PDFs, or integration links that should be included in the crawl."
              badgeTone="text-success"
              badgeLabel="High value"
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

            <div className="flex flex-wrap gap-2 border-t border-steel-100 pt-4">
              <button
                type="button"
                onClick={async () => {
                  await saveProfileInputs();
                  jobRunner.run();
                }}
                disabled={
                  updateProfile.isPending ||
                  jobRunner.isRunning ||
                  !buyerUrl.trim()
                }
                className="btn-primary gap-2 px-4 py-2 text-sm disabled:opacity-50"
              >
                {jobRunner.isRunning ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running... {Math.round(jobRunner.progress * 100)}%
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
                  {companyContext?.graph_status === "refreshing"
                    ? "Refreshing sourcing brief..."
                    : "Re-run reasoning on current crawl"}
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
                progressMessage={
                  jobRunner.progressMessage ||
                  "If this stays queued, the worker is not consuming the crawl queue yet."
                }
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
      </div>

      <div className="space-y-5 rounded-[28px] border border-steel-200 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              System sourcing brief
            </div>
            <h3 className="mt-2 font-serif text-2xl text-oxford">
              Brief-first review
            </h3>
            <p className="mt-1 text-xs text-steel-400">{outputMetaLine}</p>
          </div>

          <div className="flex flex-wrap gap-2 text-xs text-steel-600">
            <span className="rounded-full border border-steel-200 bg-steel-50 px-2.5 py-1">
              Graph status: {companyContext?.graph_status || "not_synced"}
            </span>
            {companyContext?.graph_synced_at ? (
              <span className="rounded-full border border-steel-200 bg-steel-50 px-2.5 py-1">
                Synced {new Date(companyContext.graph_synced_at).toLocaleString()}
              </span>
            ) : null}
            {secondaryEvidenceCount !== null ? (
              <span className="rounded-full border border-steel-200 bg-steel-50 px-2.5 py-1">
                Secondary evidence: {secondaryEvidenceCount}
              </span>
            ) : null}
          </div>
        </div>

        {showBuyerEvidenceWarning ? (
          <div className="rounded-2xl border border-warning/30 bg-warning/10 p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-warning" />
              <div className="min-w-0">
                <div className="text-sm font-medium text-warning">
                  Buyer evidence is still weak
                </div>
                <p className="mt-1 text-sm text-steel-700">
                  {buyerEvidence?.warning ||
                    "Add first-party product pages, customer stories, docs, PDFs, and integration pages before relying on the map."}
                </p>
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-steel-600">
                  <span className="rounded-full border border-warning/25 bg-white px-2.5 py-1">
                    Pages crawled: {buyerEvidence?.metrics.pages_crawled ?? 0}
                  </span>
                  <span className="rounded-full border border-warning/25 bg-white px-2.5 py-1">
                    Content pages: {buyerEvidence?.metrics.content_pages ?? 0}
                  </span>
                  <span className="rounded-full border border-warning/25 bg-white px-2.5 py-1">
                    Signals: {buyerEvidence?.metrics.signal_count ?? 0}
                  </span>
                  <span className="rounded-full border border-warning/25 bg-white px-2.5 py-1">
                    Customer proof: {buyerEvidence?.metrics.customer_evidence_count ?? 0}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {companyContext?.sourcing_report ? (
          <ReportArtifactRenderer
            artifact={companyContext.sourcing_report}
            onRegenerate={() => refreshCompanyContext.mutate()}
          />
        ) : (
          <div className="rounded-2xl border border-dashed border-steel-200 bg-steel-50 px-4 py-5 text-sm text-steel-500">
            Generate a sourcing brief to render the report artifact.
          </div>
        )}

        <div className="rounded-[24px] border border-steel-200 bg-[#fbfcfd] px-5 py-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
                Crawl coverage
              </div>
              <p className="mt-2 text-sm text-steel-500">
                Coverage reflects the full crawl job and evidence pass, separate from the cited sources used in the brief above.
              </p>
            </div>

            {sourceDrawerItems.length ? (
              <button
                type="button"
                onClick={() => setIsSourcesDrawerOpen(true)}
                className="inline-flex items-center gap-2 rounded-full border border-steel-200 bg-white px-3 py-1.5 text-sm font-medium text-steel-700 transition-colors hover:border-oxford hover:text-oxford"
              >
                {sourceDrawerItems.length} crawl source
                {sourceDrawerItems.length === 1 ? "" : "s"}
                <ExternalLink className="h-3.5 w-3.5" />
              </button>
            ) : null}
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl border border-steel-200 bg-white p-4">
              <div className="text-xs text-steel-500">Sites</div>
              <div className="mt-1 text-2xl font-semibold text-oxford">
                {contextPack?.crawl_coverage?.total_sites ?? 0}
              </div>
            </div>
            <div className="rounded-2xl border border-steel-200 bg-white p-4">
              <div className="text-xs text-steel-500">Pages</div>
              <div className="mt-1 text-2xl font-semibold text-oxford">
                {contextPack?.crawl_coverage?.total_pages ?? 0}
              </div>
            </div>
            <div className="rounded-2xl border border-steel-200 bg-white p-4">
              <div className="text-xs text-steel-500">Signal pages</div>
              <div className="mt-1 text-2xl font-semibold text-oxford">
                {contextPack?.crawl_coverage?.pages_with_signals ?? 0}
              </div>
            </div>
            <div className="rounded-2xl border border-steel-200 bg-white p-4">
              <div className="text-xs text-steel-500">Career pages kept</div>
              <div className="mt-1 text-2xl font-semibold text-oxford">
                {contextPack?.crawl_coverage?.career_pages_selected ?? 0}
              </div>
            </div>
          </div>

          <div className="mt-5">
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              Strongest evidence buckets
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {(companyContext?.sourcing_brief?.strongest_evidence_buckets || []).length ? (
                (companyContext?.sourcing_brief?.strongest_evidence_buckets || []).map((bucket) => (
                  <span
                    key={bucket.label}
                    className="rounded-full border border-steel-200 bg-white px-3 py-1 text-xs text-steel-600"
                  >
                    {bucket.label}: {bucket.count}
                  </span>
                ))
              ) : (
                <span className="rounded-full border border-steel-200 bg-white px-3 py-1 text-xs text-steel-500">
                  No evidence buckets summarized yet.
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-steel-100 pt-4">
          <div className="text-sm text-steel-500">
            {companyContext?.confirmed_at
              ? `Brief confirmed ${new Date(companyContext.confirmed_at).toLocaleString()}`
              : "Confirm this brief once the sourcing report looks right for expansion."}
          </div>
          <button
            type="button"
            onClick={confirmBrief}
            disabled={updateCompanyContext.isPending}
            className="btn-primary gap-2 disabled:opacity-50"
          >
            <Check className="h-4 w-4" />
            Confirm brief
          </button>
        </div>
      </div>

      <SourcesDrawer
        isOpen={isSourcesDrawerOpen}
        onClose={() => setIsSourcesDrawerOpen(false)}
        sources={sourceDrawerItems}
      />
    </div>
  );
}
