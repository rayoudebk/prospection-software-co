"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import {
  useApplyThesisAdjustment,
  useContextPack,
  useGates,
  useRefreshThesisPack,
  useThesisPack,
  useUpdateContextPack,
  useUpdateThesisPack,
  useWorkspaceJobs,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";
import { workspaceApi, ThesisClaim } from "@/lib/api";
import {
  AlertCircle,
  Bot,
  Building2,
  Check,
  CheckCircle,
  ExternalLink,
  FileText,
  Globe,
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import ReactMarkdown from "react-markdown";
import clsx from "clsx";

const SECTION_LABELS: Record<string, string> = {
  core_capability: "Core Capabilities",
  adjacent_capability: "Adjacent Capabilities",
  business_model: "Business Model",
  customer_profile: "Customer Profile",
  deployment_model: "Deployment Model",
  size_signal: "Size Signals",
  geography: "Geography",
  include_constraint: "Include Constraints",
  exclude_constraint: "Exclude Constraints",
};

function normalizeUrlInput(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const candidate = trimmed.startsWith("http://") || trimmed.startsWith("https://")
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

function ClaimBadge({ claim }: { claim: ThesisClaim }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className={clsx(
          "px-2 py-0.5 border",
          claim.rendering === "fact"
            ? "border-success/30 bg-success/10 text-success"
            : "border-warning/30 bg-warning/10 text-warning"
        )}
      >
        {claim.rendering === "fact" ? "Fact" : "Hypothesis"}
      </span>
      <span className="px-2 py-0.5 border border-steel-200 bg-white text-steel-600">
        {Math.round((claim.confidence || 0) * 100)}% confidence
      </span>
      <span className="px-2 py-0.5 border border-steel-200 bg-white text-steel-600">
        {claim.user_status}
      </span>
    </div>
  );
}

export default function ThesisPackPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: profile, isLoading } = useContextPack(workspaceId);
  const { data: thesisPack } = useThesisPack(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const { data: contextJobs } = useWorkspaceJobs(workspaceId, "context_pack");
  const updateProfile = useUpdateContextPack(workspaceId);
  const updateThesisPack = useUpdateThesisPack(workspaceId);
  const refreshThesisPack = useRefreshThesisPack(workspaceId);
  const applyThesisAdjustment = useApplyThesisAdjustment(workspaceId);

  const [buyerUrl, setBuyerUrl] = useState("");
  const [entryMode, setEntryMode] = useState<"company" | "thesis">("company");
  const [briefText, setBriefText] = useState("");
  const [referenceUrls, setReferenceUrls] = useState<string[]>([]);
  const [newReferenceUrl, setNewReferenceUrl] = useState("");
  const [referenceUrlError, setReferenceUrlError] = useState<string | null>(null);
  const [evidenceUrls, setEvidenceUrls] = useState<string[]>([]);
  const [newEvidenceUrl, setNewEvidenceUrl] = useState("");
  const [evidenceUrlError, setEvidenceUrlError] = useState<string | null>(null);
  const [editingClaimId, setEditingClaimId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState("");
  const [adjustmentMessage, setAdjustmentMessage] = useState("");
  const [draftSummary, setDraftSummary] = useState("");

  useEffect(() => {
    if (profile) {
      setBuyerUrl(profile.buyer_company_url || "");
      setEntryMode(profile.buyer_company_url ? "company" : profile.buyer_context_summary ? "thesis" : "company");
      setBriefText(profile.buyer_company_url ? "" : profile.buyer_context_summary || "");
      setReferenceUrls(profile.reference_company_urls || []);
      setEvidenceUrls(profile.reference_evidence_urls || []);
    }
  }, [profile]);

  useEffect(() => {
    setDraftSummary(thesisPack?.summary || "");
  }, [thesisPack?.summary]);

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.refreshContextPack(workspaceId),
    () => {
      refreshThesisPack.mutate();
    },
    (jobId) => workspaceApi.cancelJob(workspaceId, jobId)
  );

  const sourcePillById = useMemo(
    () => new Map((thesisPack?.source_pills || []).map((pill) => [pill.id, pill])),
    [thesisPack]
  );

  const groupedClaims = useMemo(() => {
    const groups = new Map<string, ThesisClaim[]>();
    for (const claim of thesisPack?.claims || []) {
      if (claim.user_status === "removed") continue;
      const bucket = groups.get(claim.section) || [];
      bucket.push(claim);
      groups.set(claim.section, bucket);
    }
    return groups;
  }, [thesisPack]);
  const latestCompletedContextJob = useMemo(
    () => contextJobs?.find((job) => job.state === "completed") ?? null,
    [contextJobs]
  );

  const crawlButtonLabel = profile?.context_pack_generated_at
    ? "Recrawl And Update Brief"
    : "Generate Draft From Website";
  const thesisButtonLabel = thesisPack?.generated_at
    ? "Regenerate Draft From Brief"
    : "Generate Draft From Brief";
  const jobStateLabel =
    jobRunner.job?.state === "queued"
      ? "Queued for crawl worker..."
      : `Working... ${Math.round(jobRunner.progress * 100)}%`;

  const saveProfileInputs = async () => {
    const shouldClearThesisBrief =
      entryMode === "company" &&
      !profile?.buyer_company_url &&
      Boolean(briefText.trim());
    await updateProfile.mutateAsync({
      buyer_company_url: entryMode === "company" ? buyerUrl : "",
      buyer_context_summary: entryMode === "thesis" ? briefText : shouldClearThesisBrief ? "" : undefined,
      reference_company_urls: referenceUrls,
      reference_evidence_urls: evidenceUrls,
    });
  };

  const updateClaim = async (claimId: string, patch: Partial<ThesisClaim>) => {
    if (!thesisPack) return;
    const claims = thesisPack.claims.map((claim) =>
      claim.id === claimId ? { ...claim, ...patch } : claim
    );
    await updateThesisPack.mutateAsync({ claims });
  };

  const handleAddReference = () => {
    const normalized = normalizeUrlInput(newReferenceUrl);
    if (!normalized) {
      setReferenceUrlError("Enter a valid URL.");
      return;
    }
    if (referenceUrls.some((url) => url.toLowerCase() === normalized.toLowerCase())) {
      setReferenceUrlError("That comparator URL is already added.");
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

  const handleAdjustmentSubmit = async () => {
    if (!adjustmentMessage.trim()) return;
    await applyThesisAdjustment.mutateAsync({ message: adjustmentMessage.trim() });
    setAdjustmentMessage("");
  };

  const handleSaveSummary = async () => {
    await updateThesisPack.mutateAsync({ summary: draftSummary.trim() || null });
  };

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
        icon={FileText}
        step={1}
        title="Sourcing Brief"
        subtitle="Start from a company website or an investment thesis. Add comparable companies and supporting evidence if you have them. Then generate a structured draft, review it, and correct it before moving into Search Lanes."
      />

      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.context_pack ? "bg-success/10 border-success" : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.context_pack ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.context_pack ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.context_pack
                ? "Sourcing brief ready — review the suggested Search Lanes"
                : gates.missing_items.context_pack?.join(", ") || "Complete the sourcing brief to continue"}
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[420px,minmax(0,1fr)] gap-6">
        <div className="bg-steel-50 border border-steel-200 p-6 space-y-6">
          <div>
            <h2 className="text-lg font-semibold text-oxford flex items-center gap-2 mb-4">
              <Building2 className="w-5 h-5" />
              Sourcing Brief Inputs
            </h2>
            <p className="text-sm text-steel-600 mb-4">
              Choose the entry point that matches your mandate. You can start from a buyer website or from a plain-English investment thesis, then add comparables and evidence to make the draft more precise.
            </p>

            <div className="grid grid-cols-1 gap-3 mb-5">
              <button
                type="button"
                onClick={() => setEntryMode("company")}
                className={clsx(
                  "text-left border p-4 transition",
                  entryMode === "company"
                    ? "border-oxford bg-oxford text-white"
                    : "border-steel-200 bg-white text-oxford"
                )}
              >
                <div className="text-sm font-semibold">Start from a company website</div>
                <div className={clsx("text-xs mt-1", entryMode === "company" ? "text-steel-200" : "text-steel-500")}>
                  Best when you already know the buyer or sourcing platform and want the system to infer product scope from first-party pages.
                </div>
              </button>
              <button
                type="button"
                onClick={() => setEntryMode("thesis")}
                className={clsx(
                  "text-left border p-4 transition",
                  entryMode === "thesis"
                    ? "border-oxford bg-oxford text-white"
                    : "border-steel-200 bg-white text-oxford"
                )}
              >
                <div className="text-sm font-semibold">Start from an investment thesis</div>
                <div className={clsx("text-xs mt-1", entryMode === "thesis" ? "text-steel-200" : "text-steel-500")}>
                  Use this when there is no sourcing company yet and the mandate is defined by sector, customer type, business model, size, or geography.
                </div>
              </button>
            </div>

            <div className="space-y-3 border border-steel-200 bg-white p-4 mb-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-oxford">
                    1. {entryMode === "company" ? "Your company website" : "Investment thesis or sourcing brief"}
                  </div>
                  <p className="text-xs text-steel-500 mt-1">
                    {entryMode === "company"
                      ? "Required in company-led mode. This is the main source used to understand what your platform actually does."
                      : "Required in thesis-led mode. Describe the market, customer, business model, size, and geography you want to source."}
                  </p>
                </div>
                <span className="px-2 py-0.5 text-xs border border-warning/30 bg-warning/10 text-warning">
                  Required
                </span>
              </div>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-oxford">2. Comparable companies</div>
                  <p className="text-xs text-steel-500 mt-1">
                    Optional. Useful when the mandate is broad or the first input is not specific enough on category, product shape, or adjacency.
                  </p>
                </div>
                <span className="px-2 py-0.5 text-xs border border-steel-200 bg-steel-50 text-steel-600">
                  Optional
                </span>
              </div>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-oxford">3. Supporting evidence</div>
                  <p className="text-xs text-steel-500 mt-1">
                    Optional, but high value. Add direct evidence pages, market reports, or other source material that should shape the sourcing brief.
                  </p>
                </div>
                <span className="px-2 py-0.5 text-xs border border-success/30 bg-success/10 text-success">
                  High Value
                </span>
              </div>
            </div>

            {entryMode === "company" ? (
              <>
                <label className="label flex items-center gap-2">
                  Your company website
                  <span className="px-2 py-0.5 text-[11px] border border-warning/30 bg-warning/10 text-warning">
                    Required
                  </span>
                </label>
                <input
                  type="url"
                  value={buyerUrl}
                  onChange={(event) => setBuyerUrl(event.target.value)}
                  placeholder="https://your-company.com"
                  className="input"
                />
                <p className="text-xs text-steel-500 mt-1">
                  Use your own company site. The crawler uses this to infer capabilities, ICP, and evidence-backed constraints.
                </p>
              </>
            ) : (
              <>
                <label className="label flex items-center gap-2">
                  Investment thesis or sourcing brief
                  <span className="px-2 py-0.5 text-[11px] border border-warning/30 bg-warning/10 text-warning">
                    Required
                  </span>
                </label>
                <textarea
                  value={briefText}
                  onChange={(event) => setBriefText(event.target.value)}
                  placeholder="Example: I want to invest in companies in Europe that provide software to healthcare actors such as hospitals and doctors, sold primarily as licensed software rather than SaaS, with less than $10M in revenue and 30-50 employees."
                  className="w-full min-h-[164px] border border-steel-300 px-3 py-2 text-sm bg-white"
                />
                <p className="text-xs text-steel-500 mt-1">
                  Write the mandate in plain English. Include customer type, business model, geography, size, and exclusions if they matter.
                </p>
              </>
            )}
          </div>

          <div>
            <label className="label flex items-center gap-2">
              Comparable companies
              <span className="px-2 py-0.5 text-[11px] border border-steel-200 bg-steel-50 text-steel-600">
                Optional
              </span>
            </label>
            <p className="text-xs text-steel-500 mb-2">
              Add 2-5 close comparables if they help define your category, product shape, or adjacency.
            </p>
            <div className="space-y-2 mb-2">
              {referenceUrls.map((url, index) => (
                <div key={`${url}-${index}`} className="flex items-center gap-2 px-3 py-2 bg-white border border-steel-200">
                  <Globe className="w-4 h-4 text-steel-400" />
                  <span className="flex-1 text-sm truncate">{url}</span>
                  <button
                    onClick={() => setReferenceUrls(referenceUrls.filter((_, itemIndex) => itemIndex !== index))}
                    className="text-steel-400 hover:text-danger"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="url"
                value={newReferenceUrl}
                onChange={(event) => {
                  setNewReferenceUrl(event.target.value);
                  setReferenceUrlError(null);
                }}
                placeholder="https://comparable-company.com"
                className="input text-sm"
              />
              <button onClick={handleAddReference} className="btn-secondary px-3">
                <Plus className="w-4 h-4" />
              </button>
            </div>
            {referenceUrlError && <p className="text-xs text-danger mt-2">{referenceUrlError}</p>}
          </div>

          <div>
            <label className="label flex items-center gap-2">
              Supporting evidence
              <span className="px-2 py-0.5 text-[11px] border border-success/30 bg-success/10 text-success">
                Optional, high value
              </span>
            </label>
            <p className="text-xs text-steel-500 mb-2">
              Add direct evidence such as customer stories, product pages, partner pages, industry directories, or market research reports.
            </p>
            <div className="space-y-2 mb-2">
              {evidenceUrls.map((url, index) => (
                <div key={`${url}-${index}`} className="flex items-center gap-2 px-3 py-2 bg-white border border-steel-200">
                  <Globe className="w-4 h-4 text-steel-400" />
                  <span className="flex-1 text-sm truncate">{url}</span>
                  <button
                    onClick={() => setEvidenceUrls(evidenceUrls.filter((_, itemIndex) => itemIndex !== index))}
                    className="text-steel-400 hover:text-danger"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="url"
                value={newEvidenceUrl}
                onChange={(event) => {
                  setNewEvidenceUrl(event.target.value);
                  setEvidenceUrlError(null);
                }}
                placeholder="https://company.com/customer-story or https://bcg.com/report"
                className="input text-sm"
              />
              <button onClick={handleAddEvidence} className="btn-secondary px-3">
                <Plus className="w-4 h-4" />
              </button>
            </div>
            {evidenceUrlError && <p className="text-xs text-danger mt-2">{evidenceUrlError}</p>}
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={saveProfileInputs}
              disabled={updateProfile.isPending}
              className="btn-secondary disabled:opacity-50"
            >
              {updateProfile.isPending ? "Saving..." : "Save Brief Inputs"}
            </button>
            <button
              onClick={async () => {
                await saveProfileInputs();
                if (entryMode === "company") {
                  jobRunner.run();
                  return;
                }
                await refreshThesisPack.mutateAsync();
              }}
              disabled={
                updateProfile.isPending ||
                (entryMode === "company"
                  ? jobRunner.isRunning || !buyerUrl
                  : refreshThesisPack.isPending || !briefText.trim())
              }
              className="btn-primary flex items-center gap-2 disabled:opacity-50"
            >
              {entryMode === "company" && jobRunner.isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {jobStateLabel}
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  {entryMode === "company" ? crawlButtonLabel : thesisButtonLabel}
                </>
              )}
            </button>
            {entryMode === "company" ? (
              <button
                onClick={async () => {
                  await saveProfileInputs();
                  await refreshThesisPack.mutateAsync();
                }}
                disabled={
                  updateProfile.isPending ||
                  refreshThesisPack.isPending ||
                  (!buyerUrl.trim() && !thesisPack?.summary)
                }
                className="btn-secondary flex items-center gap-2 disabled:opacity-50"
              >
                {refreshThesisPack.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                Regenerate Structured Draft
              </button>
            ) : null}
          </div>

          {entryMode === "company" && jobRunner.isRunning && (
            <JobProgressPanel
              job={jobRunner.job ?? {
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
              }}
              progress={jobRunner.progress}
              progressMessage={
                jobRunner.progressMessage ||
                "If this stays queued, the worker is not consuming the crawl queue yet."
              }
              isStopping={jobRunner.isStopping}
              onStop={jobRunner.canStop ? jobRunner.stop : undefined}
            />
          )}

          {entryMode === "company" && !jobRunner.isRunning && <JobRunSummary job={latestCompletedContextJob} />}

          {(jobRunner.jobError || applyThesisAdjustment.error) && (
            <div className="text-sm text-danger border border-danger/40 bg-danger/10 px-3 py-2">
              {jobRunner.jobError || applyThesisAdjustment.error?.message}
            </div>
          )}
        </div>

        <div className="space-y-6">
          <div className="bg-oxford text-white border border-oxford-dark p-6 space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">Draft Thesis</h2>
                <p className="text-sm text-steel-300">
                  {thesisPack?.generated_at
                    ? `Generated ${new Date(thesisPack.generated_at).toLocaleString()}`
                    : "Generate a draft from the saved company sources."}
                </p>
              </div>
              {profile?.product_pages_found ? (
                <span className="px-2 py-1 text-xs border border-success/40 bg-success/10 text-success">
                  {profile.product_pages_found} product pages found
                </span>
              ) : null}
            </div>

            <div className="border border-oxford-light bg-oxford-dark/40 p-4">
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="text-xs uppercase tracking-wide text-steel-400">Structured Summary</div>
                <button
                  onClick={handleSaveSummary}
                  disabled={updateThesisPack.isPending}
                  className="btn-secondary text-xs disabled:opacity-50"
                >
                  Save Summary
                </button>
              </div>
              <textarea
                value={draftSummary}
                onChange={(event) => setDraftSummary(event.target.value)}
                placeholder="Generate a draft from your sourcing inputs, then tighten the summary here if the framing is off."
                className="w-full min-h-[148px] border border-oxford-light bg-oxford px-3 py-2 text-sm text-steel-100"
              />
              <p className="text-xs text-steel-400 mt-2">
                Edit this if the system interpreted the sourcing brief incorrectly before you confirm Search Lanes.
              </p>
            </div>

            {thesisPack?.source_pills?.length ? (
              <div>
                <div className="text-xs uppercase tracking-wide text-steel-400 mb-2">Source Pills</div>
                <div className="flex flex-wrap gap-2">
                  {thesisPack.source_pills.map((pill) => (
                    <a
                      key={pill.id}
                      href={pill.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 px-2 py-1 text-xs border border-info/30 bg-info/10 text-info hover:bg-info/20"
                    >
                      {pill.label}
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="space-y-4">
              {Array.from(groupedClaims.entries()).map(([section, claims]) => (
                <div key={section} className="border border-oxford-light bg-oxford-dark/40 p-4">
                  <div className="text-xs uppercase tracking-wide text-steel-400 mb-3">
                    {SECTION_LABELS[section] || section}
                  </div>
                  <div className="space-y-3">
                    {claims.map((claim: ThesisClaim) => {
                      const isEditing = editingClaimId === claim.id;
                      return (
                        <div key={claim.id} className="border border-oxford-light/70 bg-oxford/50 p-3 space-y-3">
                          <ClaimBadge claim={claim} />
                          {isEditing ? (
                            <div className="flex gap-2">
                              <input
                                value={editingValue}
                                onChange={(event) => setEditingValue(event.target.value)}
                                className="input flex-1 bg-white text-oxford"
                              />
                              <button
                                onClick={async () => {
                                  await updateClaim(claim.id, { value: editingValue, user_status: "edited" });
                                  setEditingClaimId(null);
                                  setEditingValue("");
                                }}
                                className="btn-secondary"
                              >
                                Save
                              </button>
                            </div>
                          ) : (
                            <p className="text-sm text-steel-100">{claim.value}</p>
                          )}

                          {claim.source_pill_ids.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                              {claim.source_pill_ids.map((pillId: string) => {
                                const pill = sourcePillById.get(pillId);
                                if (!pill) return null;
                                return (
                                  <a
                                    key={`${claim.id}-${pillId}`}
                                    href={pill.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 px-2 py-0.5 text-xs border border-info/30 bg-info/10 text-info hover:bg-info/20"
                                  >
                                    {pill.label}
                                    <ExternalLink className="w-3 h-3" />
                                  </a>
                                );
                              })}
                            </div>
                          )}

                          <div className="flex flex-wrap gap-2">
                            <button
                              onClick={() => updateClaim(claim.id, { user_status: "confirmed" })}
                              disabled={updateThesisPack.isPending}
                              className="btn-secondary flex items-center gap-2 text-sm"
                            >
                              <Check className="w-4 h-4" />
                              Accept
                            </button>
                            <button
                              onClick={() => {
                                setEditingClaimId(claim.id);
                                setEditingValue(claim.value);
                              }}
                              className="btn-secondary flex items-center gap-2 text-sm"
                            >
                              <Pencil className="w-4 h-4" />
                              Edit
                            </button>
                            <button
                              onClick={() => updateClaim(claim.id, { user_status: "removed" })}
                              disabled={updateThesisPack.isPending}
                              className="btn-secondary flex items-center gap-2 text-sm"
                            >
                              <Trash2 className="w-4 h-4" />
                              Remove
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}

              {groupedClaims.size === 0 && (
                <div className="text-center py-10 text-steel-300 border border-oxford-light bg-oxford-dark/40">
                  <Globe className="w-10 h-10 mx-auto mb-3 text-steel-500" />
                  <p>No thesis claims yet.</p>
                  <p className="text-sm text-steel-400 mt-1">Generate the draft from your sourcing inputs first.</p>
                </div>
              )}
            </div>

            <div className="border border-oxford-light bg-oxford-dark/40 p-4">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-steel-400 mb-3">
                <Bot className="w-4 h-4" />
                Correct Or Refine This Draft
              </div>
              <textarea
                value={adjustmentMessage}
                onChange={(event) => setAdjustmentMessage(event.target.value)}
                placeholder={"Examples:\nadd core: healthcare provider software\nadd adjacent: patient engagement workflow\nremove Named customer proof: Example Corp\nbusiness model: license-based software\nexclude: SaaS-first vendors"}
                className="w-full min-h-[132px] border border-oxford-light bg-oxford text-white px-3 py-2 text-sm"
              />
              <div className="flex items-center justify-between gap-3 mt-3">
                <p className="text-xs text-steel-400">
                  Use short corrections. Accepted changes are written back as structured thesis updates.
                </p>
                <button
                  onClick={handleAdjustmentSubmit}
                  disabled={applyThesisAdjustment.isPending || !adjustmentMessage.trim()}
                  className="btn-primary flex items-center gap-2 disabled:opacity-50"
                >
                  {applyThesisAdjustment.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Bot className="w-4 h-4" />}
                  Apply
                </button>
              </div>
            </div>

            {thesisPack?.open_questions?.length ? (
              <div className="border border-oxford-light bg-oxford-dark/40 p-4">
                <div className="text-xs uppercase tracking-wide text-steel-400 mb-3">Open Questions</div>
                <div className="space-y-2">
                  {thesisPack.open_questions.map((question) => (
                    <div key={question} className="text-sm text-steel-100">
                      - {question}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <details className="bg-steel-50 border border-steel-200 p-6">
            <summary className="cursor-pointer text-sm font-medium text-oxford">Raw Crawl Output</summary>
            {profile?.context_pack_markdown ? (
              <div className="prose mt-4 max-h-[560px] overflow-y-auto pr-2">
                <ReactMarkdown>{profile.context_pack_markdown}</ReactMarkdown>
              </div>
            ) : (
              <div className="text-sm text-steel-500 mt-4">No raw context pack available yet.</div>
            )}
          </details>
        </div>
      </div>
    </div>
  );
}
