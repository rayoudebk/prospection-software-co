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
import { workspaceApi, ThesisClaim, ThesisSourcePill } from "@/lib/api";
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

const CLAIM_SECTION_ORDER = [
  "customer_profile",
  "core_capability",
  "adjacent_capability",
  "business_model",
  "deployment_model",
  "size_signal",
  "geography",
  "include_constraint",
  "exclude_constraint",
] as const;

const PILL_EDITOR_SECTIONS = new Set<string>([
  "customer_profile",
  "core_capability",
  "adjacent_capability",
  "business_model",
  "deployment_model",
  "geography",
]);

const PILL_INPUT_PLACEHOLDERS: Record<string, string> = {
  customer_profile: "Add customer profile",
  core_capability: "Add core capability",
  adjacent_capability: "Add adjacent capability",
  business_model: "Add business model",
  deployment_model: "Add deployment model",
  geography: "Add geography",
};

type SourceDrawerItem = ThesisSourcePill & {
  hostname: string;
  displayUrl: string;
  badge: string | null;
};

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
              Sources
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
                href={source.url}
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
                      {source.label}
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

function EditableClaimPillsSection({
  section,
  claims,
  sourcePillById,
  draftValue,
  error,
  isPending,
  onDraftChange,
  onAdd,
  onRemove,
}: {
  section: string;
  claims: ThesisClaim[];
  sourcePillById: Map<string, ThesisSourcePill>;
  draftValue: string;
  error?: string | null;
  isPending: boolean;
  onDraftChange: (value: string) => void;
  onAdd: () => void;
  onRemove: (claimId: string) => void;
}) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400 shrink-0">
          {SECTION_LABELS[section] || section}
        </span>
        <div className="flex-1 border-t border-steel-200" />
      </div>

      <div className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
        <div className="flex flex-wrap gap-2">
          {claims.length > 0 ? (
            claims.map((claim) => {
              const linkedSources = claim.source_pill_ids
                .map((pillId) => sourcePillById.get(pillId))
                .filter((pill): pill is ThesisSourcePill => Boolean(pill));
              const sourceLink = linkedSources[0];
              const sourceLabel =
                linkedSources.length > 1 ? `${linkedSources.length} src` : "src";

              return (
                <div
                  key={claim.id}
                  className={clsx(
                    "inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-2 text-sm leading-tight",
                    claim.user_status === "confirmed"
                      ? "border-success/30 bg-success/10 text-success"
                      : claim.rendering === "hypothesis"
                      ? "border-warning/30 bg-warning/10 text-amber-900"
                      : "border-oxford/15 bg-white text-steel-800"
                  )}
                >
                  <span className="max-w-[28rem] whitespace-normal break-words">
                    {claim.value}
                  </span>
                  {sourceLink ? (
                    <a
                      href={sourceLink.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 text-[11px] font-medium uppercase tracking-[0.16em] text-info hover:underline"
                      title={sourceLink.label}
                    >
                      {sourceLabel}
                    </a>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => onRemove(claim.id)}
                    disabled={isPending}
                    className="shrink-0 text-steel-400 transition-colors hover:text-danger disabled:opacity-50"
                    aria-label={`Remove ${claim.value}`}
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })
          ) : (
            <p className="text-sm text-steel-400">No items added yet.</p>
          )}
        </div>

        <div className="mt-3 flex gap-2">
          <input
            type="text"
            value={draftValue}
            onChange={(e) => onDraftChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                onAdd();
              }
            }}
            placeholder={PILL_INPUT_PLACEHOLDERS[section] || "Add item"}
            className="input text-sm"
          />
          <button
            type="button"
            onClick={onAdd}
            disabled={isPending || !draftValue.trim()}
            className="btn-secondary px-3 shrink-0 disabled:opacity-50"
            aria-label={`Add ${SECTION_LABELS[section] || section}`}
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
        {error ? <p className="mt-1 text-xs text-danger">{error}</p> : null}
      </div>
    </div>
  );
}

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
  const [claimDraftValues, setClaimDraftValues] = useState<Record<string, string>>({});
  const [claimDraftErrors, setClaimDraftErrors] = useState<Record<string, string>>({});
  const [adjustmentMessage, setAdjustmentMessage] = useState("");
  const [draftSummary, setDraftSummary] = useState("");
  const [isSourcesDrawerOpen, setIsSourcesDrawerOpen] = useState(false);

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

  const sourceDrawerItems = useMemo(
    () =>
      (thesisPack?.source_pills || []).map((pill) => ({
        ...pill,
        hostname: getSourceHostname(pill.url),
        displayUrl: getSourceDisplayUrl(pill.url),
        badge: getSourceBadge(pill.label),
      })),
    [thesisPack?.source_pills]
  );

  const buyerEvidence = thesisPack?.buyer_evidence || null;
  const showBuyerEvidenceWarning =
    entryMode === "company" && buyerEvidence?.status === "insufficient";

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

  const displaySections = useMemo(
    () =>
      CLAIM_SECTION_ORDER.filter(
        (section) =>
          groupedClaims.has(section) || PILL_EDITOR_SECTIONS.has(section)
      ).map((section) => [section, groupedClaims.get(section) || []] as const),
    [groupedClaims]
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

  const setClaimDraftValue = (section: string, value: string) => {
    setClaimDraftValues((current) => ({ ...current, [section]: value }));
    setClaimDraftErrors((current) => ({ ...current, [section]: "" }));
  };

  const handleAddSectionClaim = async (section: string) => {
    if (!thesisPack) return;
    const value = (claimDraftValues[section] || "").trim();
    if (!value) return;

    const duplicateExists = thesisPack.claims.some(
      (claim) =>
        claim.section === section &&
        claim.user_status !== "removed" &&
        claim.value.trim().toLowerCase() === value.toLowerCase()
    );
    if (duplicateExists) {
      setClaimDraftErrors((current) => ({
        ...current,
        [section]: "That item is already present.",
      }));
      return;
    }

    await updateThesisPack.mutateAsync({
      claims: [
        ...thesisPack.claims,
        {
          id: `manual_${section}_${Date.now()}`,
          section,
          value,
          rendering: "hypothesis",
          confidence: 0.9,
          source_pill_ids: [],
          user_status: "edited",
        },
      ],
    });

    setClaimDraftValues((current) => ({ ...current, [section]: "" }));
    setClaimDraftErrors((current) => ({ ...current, [section]: "" }));
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
        {sourceDrawerItems.length ? (
          <div className="flex items-center">
            <button
              type="button"
              onClick={() => setIsSourcesDrawerOpen(true)}
              className="inline-flex items-center gap-2 rounded-full border border-steel-200 bg-steel-50 px-3 py-1.5 text-sm font-medium text-steel-700 transition-colors hover:border-oxford hover:text-oxford"
            >
              <span>
                {sourceDrawerItems.length} source
                {sourceDrawerItems.length === 1 ? "" : "s"}
              </span>
              <span className="rounded-full border border-steel-200 bg-white px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.16em] text-steel-500">
                Web links
              </span>
            </button>
          </div>
        ) : null}

        {showBuyerEvidenceWarning ? (
          <div className="rounded-2xl border border-warning/30 bg-warning/10 p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-warning" />
              <div className="min-w-0">
                <div className="text-sm font-medium text-warning">
                  Buyer evidence is too weak for reliable inference
                </div>
                <p className="mt-1 text-sm text-steel-700">
                  {buyerEvidence?.warning ||
                    "Add first-party product pages, PDFs, case studies, or supporting evidence before trusting buyer claims."}
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
                  <span className="rounded-full border border-warning/25 bg-white px-2.5 py-1">
                    Summary chars: {buyerEvidence?.metrics.summary_chars ?? 0}
                  </span>
                  <span className="rounded-full border border-warning/25 bg-white px-2.5 py-1">
                    Score: {buyerEvidence?.score ?? 0}
                  </span>
                </div>
              </div>
            </div>
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
          {displaySections.map(([section, claims]) =>
            PILL_EDITOR_SECTIONS.has(section) ? (
              <EditableClaimPillsSection
                key={section}
                section={section}
                claims={claims}
                sourcePillById={sourcePillById}
                draftValue={claimDraftValues[section] || ""}
                error={claimDraftErrors[section] || null}
                isPending={updateThesisPack.isPending}
                onDraftChange={(value) => setClaimDraftValue(section, value)}
                onAdd={() => handleAddSectionClaim(section)}
                onRemove={(claimId) =>
                  updateClaim(claimId, { user_status: "removed" })
                }
              />
            ) : (
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
            )
          )}

          {displaySections.every(([, claims]) => claims.length === 0) &&
            !thesisPack?.generated_at && (
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

      <SourcesDrawer
        isOpen={isSourcesDrawerOpen}
        onClose={() => setIsSourcesDrawerOpen(false)}
        sources={sourceDrawerItems}
      />
    </div>
  );
}
