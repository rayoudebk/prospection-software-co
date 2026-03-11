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
  Check,
  CheckCircle,
  ExternalLink,
  FileText,
  Globe,
  Loader2,
  Plus,
  RefreshCw,
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
      setEntryMode(
        profile.buyer_company_url
          ? "company"
          : profile.manual_brief_text
          ? "thesis"
          : "company"
      );
      setBriefText(
        profile.buyer_company_url ? "" : profile.manual_brief_text || ""
      );
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
    () =>
      new Map(
        (thesisPack?.source_pills || []).map((pill) => [pill.id, pill])
      ),
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

  const hasGeneratedContextPack = Boolean(
    profile?.context_pack_generated_at ||
      profile?.context_pack_markdown ||
      profile?.context_pack_json
  );

  const crawlButtonLabel = profile?.context_pack_generated_at
    ? "Recrawl and update brief"
    : "Generate draft from website";
  const thesisButtonLabel = thesisPack?.generated_at
    ? "Regenerate draft from brief"
    : "Generate draft from brief";
  const jobStateLabel =
    jobRunner.job?.state === "queued"
      ? "Queued..."
      : `Running... ${Math.round(jobRunner.progress * 100)}%`;

  // Merged meta line for the output panel header
  const outputMetaLine = thesisPack?.generated_at
    ? [
        `Last generated ${new Date(thesisPack.generated_at).toLocaleString()}`,
        profile?.product_pages_found
          ? `${profile.product_pages_found} pages crawled`
          : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : "Generate a draft from your inputs to populate this panel";

  const saveProfileInputs = async () => {
    const shouldClearThesisBrief =
      entryMode === "company" &&
      !profile?.buyer_company_url &&
      Boolean(briefText.trim());
    await updateProfile.mutateAsync({
      buyer_company_url: entryMode === "company" ? buyerUrl : "",
      manual_brief_text:
        entryMode === "thesis"
          ? briefText
          : shouldClearThesisBrief
          ? ""
          : undefined,
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
    if (
      referenceUrls.some(
        (url) => url.toLowerCase() === normalized.toLowerCase()
      )
    ) {
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
    if (
      evidenceUrls.some(
        (url) => url.toLowerCase() === normalized.toLowerCase()
      )
    ) {
      setEvidenceUrlError("That evidence URL is already added.");
      return;
    }
    setEvidenceUrls([...evidenceUrls, normalized]);
    setNewEvidenceUrl("");
    setEvidenceUrlError(null);
  };

  const handleAdjustmentSubmit = async () => {
    if (!adjustmentMessage.trim()) return;
    await applyThesisAdjustment.mutateAsync({
      message: adjustmentMessage.trim(),
    });
    setAdjustmentMessage("");
  };

  const handleSaveSummary = async () => {
    await updateThesisPack.mutateAsync({
      summary: draftSummary.trim() || null,
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="w-6 h-6 animate-spin text-oxford" />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">

      {/* Page header */}
      <div className="flex items-start justify-between">
        <StepHeader
          step={1}
          title="Sourcing Brief"
          subtitle="Define your mandate from a company website or investment thesis. Add comparables and evidence, then generate and verify a structured brief before moving to Search Lanes."
        />
        {gates && (
          <div
            className={clsx(
              "flex items-center gap-1.5 text-xs shrink-0 mt-1",
              gates.context_pack ? "text-success" : "text-steel-400"
            )}
          >
            {gates.context_pack ? (
              <>
                <CheckCircle className="w-3.5 h-3.5" />
                <span className="font-medium">Ready</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-warning" />
                <span className="text-warning font-medium">Incomplete</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* ── INPUT CARD ── */}
      <div className="bg-white border border-steel-200 p-6 space-y-5">

        {/* Entry mode toggle */}
        <div>
          <p className="text-[11px] font-medium uppercase tracking-widest text-steel-400 mb-2">
            Entry point
          </p>
          <div className="flex border border-steel-200">
            <button
              type="button"
              onClick={() => setEntryMode("company")}
              className={clsx(
                "flex-1 px-3 py-2 text-sm font-medium transition-colors",
                entryMode === "company"
                  ? "bg-oxford text-white"
                  : "bg-white text-steel-600 hover:text-oxford"
              )}
            >
              Company website
            </button>
            <button
              type="button"
              onClick={() => setEntryMode("thesis")}
              className={clsx(
                "flex-1 px-3 py-2 text-sm font-medium transition-colors border-l border-steel-200",
                entryMode === "thesis"
                  ? "bg-oxford text-white"
                  : "bg-white text-steel-600 hover:text-oxford"
              )}
            >
              Investment thesis
            </button>
          </div>
        </div>

        {/* Main input */}
        {entryMode === "company" ? (
          <div>
            <label className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                Company website
              </span>
              <span className="text-[11px] text-warning">Required</span>
            </label>
            <input
              type="url"
              value={buyerUrl}
              onChange={(e) => setBuyerUrl(e.target.value)}
              placeholder="https://your-company.com"
              className="input"
            />
          </div>
        ) : (
          <div>
            <label className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                Thesis or brief
              </span>
              <span className="text-[11px] text-warning">Required</span>
            </label>
            <textarea
              value={briefText}
              onChange={(e) => setBriefText(e.target.value)}
              placeholder="I want to invest in companies in Europe that provide software to healthcare actors such as hospitals and doctors, sold primarily as licensed software rather than SaaS, with less than $10M in revenue..."
              className="w-full min-h-[148px] border border-steel-200 bg-white px-3 py-2 text-sm text-steel-900 placeholder:text-steel-400 focus:outline-none focus:border-oxford focus:ring-1 focus:ring-oxford/20 transition resize-none"
            />
          </div>
        )}

        {/* Comparable companies */}
        <div>
          <label className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Comparable companies
            </span>
            <span className="text-[11px] text-steel-400">Optional</span>
          </label>
          {referenceUrls.length > 0 && (
            <div className="space-y-1.5 mb-2">
              {referenceUrls.map((url, index) => (
                <div
                  key={`${url}-${index}`}
                  className="flex items-center gap-2 px-2.5 py-1.5 bg-steel-50 border border-steel-200"
                >
                  <Globe className="w-3 h-3 text-steel-400 shrink-0" />
                  <span className="flex-1 text-xs font-mono text-steel-600 truncate">
                    {url}
                  </span>
                  <button
                    onClick={() =>
                      setReferenceUrls(
                        referenceUrls.filter((_, i) => i !== index)
                      )
                    }
                    className="text-steel-400 hover:text-danger transition-colors"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <input
              type="url"
              value={newReferenceUrl}
              onChange={(e) => {
                setNewReferenceUrl(e.target.value);
                setReferenceUrlError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleAddReference();
                }
              }}
              placeholder="https://comparable-company.com"
              className="input text-sm"
            />
            <button
              onClick={handleAddReference}
              className="btn-secondary px-3 shrink-0"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
          {referenceUrlError && (
            <p className="text-xs text-danger mt-1">{referenceUrlError}</p>
          )}
        </div>

        {/* Supporting evidence */}
        <div>
          <label className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Supporting evidence
            </span>
            <span className="text-[11px] text-success">High value</span>
          </label>
          {evidenceUrls.length > 0 && (
            <div className="space-y-1.5 mb-2">
              {evidenceUrls.map((url, index) => (
                <div
                  key={`${url}-${index}`}
                  className="flex items-center gap-2 px-2.5 py-1.5 bg-steel-50 border border-steel-200"
                >
                  <Globe className="w-3 h-3 text-steel-400 shrink-0" />
                  <span className="flex-1 text-xs font-mono text-steel-600 truncate">
                    {url}
                  </span>
                  <button
                    onClick={() =>
                      setEvidenceUrls(
                        evidenceUrls.filter((_, i) => i !== index)
                      )
                    }
                    className="text-steel-400 hover:text-danger transition-colors"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <input
              type="url"
              value={newEvidenceUrl}
              onChange={(e) => {
                setNewEvidenceUrl(e.target.value);
                setEvidenceUrlError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleAddEvidence();
                }
              }}
              placeholder="https://company.com/customer-story"
              className="input text-sm"
            />
            <button
              onClick={handleAddEvidence}
              className="btn-secondary px-3 shrink-0"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
          {evidenceUrlError && (
            <p className="text-xs text-danger mt-1">{evidenceUrlError}</p>
          )}
        </div>

        {/* Actions */}
        <div className="pt-3 border-t border-steel-100 space-y-2">
          {/* Primary: Generate (auto-saves inputs before running) */}
          <button
            onClick={async () => {
              await saveProfileInputs();
              if (entryMode === "company") {
                jobRunner.run();
              } else {
                await refreshThesisPack.mutateAsync();
              }
            }}
            disabled={
              updateProfile.isPending ||
              (entryMode === "company"
                ? jobRunner.isRunning || !buyerUrl
                : refreshThesisPack.isPending || !briefText.trim())
            }
            className="w-full btn-primary gap-2 disabled:opacity-50"
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

          {/* Secondary: Regenerate structured draft (company mode only, post-crawl) */}
          {entryMode === "company" && hasGeneratedContextPack && (
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
              className="w-full btn-secondary gap-1.5 disabled:opacity-50"
            >
              {refreshThesisPack.isPending ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <RefreshCw className="w-3 h-3" />
              )}
              Regenerate structured draft
            </button>
          )}
        </div>

        {/* Job progress panel */}
        {entryMode === "company" && jobRunner.isRunning && (
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
        )}

        {/* Job run summary */}
        {entryMode === "company" && !jobRunner.isRunning && (
          <JobRunSummary job={latestCompletedContextJob} />
        )}

        {/* Error state */}
        {(jobRunner.jobError || applyThesisAdjustment.error) && (
          <div className="text-sm text-danger border border-danger/30 bg-danger/5 px-3 py-2">
            {jobRunner.jobError || applyThesisAdjustment.error?.message}
          </div>
        )}
      </div>

      {/* ── OUTPUT CARD ── */}
      <div className="bg-white border border-steel-200 p-6 space-y-5">

        {/* Output header — timestamp + pages crawled merged into one line */}
        <div>
          <h3 className="font-serif text-xl text-oxford">Draft Sourcing Brief</h3>
          <p className="text-xs text-steel-400 mt-1">{outputMetaLine}</p>
        </div>

        {/* Source pills */}
        {thesisPack?.source_pills?.length ? (
          <div className="flex flex-wrap gap-1.5">
            {thesisPack.source_pills.map((pill) => (
              <a
                key={pill.id}
                href={pill.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] border border-steel-200 bg-steel-50 text-steel-500 hover:border-oxford hover:text-oxford transition-colors"
              >
                {pill.label}
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            ))}
          </div>
        ) : null}

        {/* Editable summary */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Summary
            </span>
            <button
              onClick={handleSaveSummary}
              disabled={updateThesisPack.isPending}
              className="text-xs text-oxford hover:text-oxford-light disabled:opacity-50 transition-colors font-medium"
            >
              {updateThesisPack.isPending ? "Saving..." : "Save"}
            </button>
          </div>
          <textarea
            value={draftSummary}
            onChange={(e) => setDraftSummary(e.target.value)}
            placeholder="Generate a draft to populate the structured summary. Edit here if the system interpreted your mandate incorrectly."
            className="w-full min-h-[100px] bg-steel-50 border border-steel-200 px-3 py-2.5 text-sm text-steel-800 placeholder:text-steel-400 focus:outline-none focus:border-oxford focus:ring-1 focus:ring-oxford/20 transition resize-none leading-relaxed"
          />
        </div>

        {/* Claims by section */}
        <div className="space-y-5">
          {Array.from(groupedClaims.entries()).map(([section, claims]) => (
            <div key={section}>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400 shrink-0">
                  {SECTION_LABELS[section] || section}
                </span>
                <div className="flex-1 border-t border-steel-200" />
              </div>

              <div className="space-y-1.5">
                {claims.map((claim: ThesisClaim) => {
                  const isEditing = editingClaimId === claim.id;
                  const isConfirmed = claim.user_status === "confirmed";
                  const isHypothesis = claim.rendering === "hypothesis";

                  return (
                    <div
                      key={claim.id}
                      className={clsx(
                        "bg-steel-50 border border-steel-200 border-l-2 px-3 py-2.5 group transition-all hover:bg-white hover:shadow-sm",
                        isConfirmed
                          ? "border-l-success"
                          : isHypothesis
                          ? "border-l-warning"
                          : "border-l-oxford"
                      )}
                    >
                      {/* Claim value */}
                      {isEditing ? (
                        <div className="flex gap-2 mb-2">
                          <input
                            value={editingValue}
                            onChange={(e) => setEditingValue(e.target.value)}
                            className="flex-1 border border-steel-200 bg-white px-2 py-1.5 text-sm text-steel-900 font-mono focus:outline-none focus:border-oxford"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === "Escape") setEditingClaimId(null);
                            }}
                          />
                          <button
                            onClick={async () => {
                              await updateClaim(claim.id, {
                                value: editingValue,
                                user_status: "edited",
                              });
                              setEditingClaimId(null);
                              setEditingValue("");
                            }}
                            className="btn-primary text-xs px-3 py-1.5"
                          >
                            Save
                          </button>
                          <button
                            onClick={() => setEditingClaimId(null)}
                            className="btn-secondary text-xs px-2 py-1.5"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ) : (
                        <p className="text-sm font-mono text-steel-800 mb-2 leading-relaxed">
                          {claim.value}
                        </p>
                      )}

                      {/* Meta + inline actions */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-[11px] text-steel-400">
                          <span className={isHypothesis ? "text-warning" : ""}>
                            {isHypothesis ? "Hypothesis" : "Fact"}
                          </span>
                          <span>·</span>
                          <span>{Math.round((claim.confidence || 0) * 100)}%</span>
                          {claim.source_pill_ids.length > 0 && (
                            <>
                              <span>·</span>
                              <div className="flex gap-1">
                                {claim.source_pill_ids
                                  .slice(0, 3)
                                  .map((pillId: string) => {
                                    const pill = sourcePillById.get(pillId);
                                    if (!pill) return null;
                                    return (
                                      <a
                                        key={pillId}
                                        href={pill.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-info hover:underline"
                                        title={pill.label}
                                      >
                                        [src]
                                      </a>
                                    );
                                  })}
                              </div>
                            </>
                          )}
                        </div>

                        {/* Hover actions */}
                        <div className="flex items-center gap-3 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={() =>
                              updateClaim(claim.id, { user_status: "confirmed" })
                            }
                            disabled={updateThesisPack.isPending || isConfirmed}
                            className={clsx(
                              "font-medium transition-colors",
                              isConfirmed
                                ? "text-success cursor-default"
                                : "text-steel-500 hover:text-success"
                            )}
                          >
                            {isConfirmed ? (
                              <span className="flex items-center gap-1">
                                <Check className="w-3 h-3" /> Accepted
                              </span>
                            ) : (
                              "Accept"
                            )}
                          </button>
                          <button
                            onClick={() => {
                              setEditingClaimId(claim.id);
                              setEditingValue(claim.value);
                            }}
                            className="text-steel-400 hover:text-oxford transition-colors"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() =>
                              updateClaim(claim.id, { user_status: "removed" })
                            }
                            disabled={updateThesisPack.isPending}
                            className="text-steel-400 hover:text-danger transition-colors"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          {groupedClaims.size === 0 && (
            <div className="flex flex-col items-center justify-center py-14 border border-dashed border-steel-300 bg-steel-50">
              <FileText className="w-8 h-8 text-steel-300 mb-3" />
              <p className="text-sm text-steel-400">No claims generated yet</p>
              <p className="text-xs text-steel-400 mt-1">
                Generate a draft from your inputs above to start reviewing
              </p>
            </div>
          )}
        </div>

        {/* Correct / refine */}
        <div className="border-t border-steel-200 pt-5">
          <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-widest text-steel-400 mb-2">
            <Bot className="w-3.5 h-3.5" />
            Correct or refine
          </div>
          <textarea
            value={adjustmentMessage}
            onChange={(e) => setAdjustmentMessage(e.target.value)}
            placeholder={"add core: healthcare provider software\nremove Named customer proof: Example Corp\nbusiness model: license-based software\nexclude: SaaS-first vendors"}
            className="w-full min-h-[96px] bg-steel-50 border border-steel-200 px-3 py-2.5 text-sm text-steel-800 placeholder:text-steel-400 focus:outline-none focus:border-oxford focus:ring-1 focus:ring-oxford/20 transition resize-none font-mono leading-relaxed"
          />
          <div className="flex items-center justify-between mt-2">
            <p className="text-xs text-steel-400">
              Use short corrections — changes write back as structured updates
            </p>
            <button
              onClick={handleAdjustmentSubmit}
              disabled={
                applyThesisAdjustment.isPending || !adjustmentMessage.trim()
              }
              className="btn-primary gap-1.5 text-sm px-3 py-1.5 disabled:opacity-50"
            >
              {applyThesisAdjustment.isPending ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : null}
              Apply
            </button>
          </div>
        </div>

        {/* Open questions */}
        {thesisPack?.open_questions?.length ? (
          <div className="border-t border-steel-200 pt-4">
            <div className="text-[11px] font-medium uppercase tracking-widest text-steel-400 mb-2">
              Open questions
            </div>
            <ul className="space-y-1.5">
              {thesisPack.open_questions.map((question) => (
                <li
                  key={question}
                  className="text-sm text-steel-600 flex items-start gap-2"
                >
                  <span className="text-steel-300 shrink-0 mt-0.5 font-serif">—</span>
                  <span>{question}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      {/* Raw crawl output — collapsed */}
      <details className="bg-white border border-steel-200">
        <summary className="cursor-pointer px-4 py-3 text-[11px] font-medium uppercase tracking-widest text-steel-400 hover:text-steel-600 transition-colors select-none">
          Raw crawl output
        </summary>
        {profile?.context_pack_markdown ? (
          <div className="prose prose-sm px-4 pb-4 max-h-[560px] overflow-y-auto border-t border-steel-100 pt-3">
            <ReactMarkdown>{profile.context_pack_markdown}</ReactMarkdown>
          </div>
        ) : (
          <div className="px-4 pb-4 text-sm text-steel-400 border-t border-steel-100 pt-3">
            No raw context pack available yet.
          </div>
        )}
      </details>
    </div>
  );
}
