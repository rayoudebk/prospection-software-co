"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import clsx from "clsx";
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
  Save,
  X,
} from "lucide-react";

import { StepHeader } from "@/components/StepHeader";
import { JobProgressPanel } from "@/components/JobProgressPanel";
import { JobRunSummary } from "@/components/JobRunSummary";
import {
  ContextPackV2,
  ContextPackEvidenceItem,
  TaxonomyNode,
  ThesisSourcePill,
  workspaceApi,
} from "@/lib/api";
import {
  useContextPack,
  useGates,
  useRefreshThesisPack,
  useThesisPack,
  useUpdateContextPack,
  useUpdateThesisPack,
  useWorkspaceJobs,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";

const LAYER_LABELS: Record<string, string> = {
  customer_archetype: "Customer Archetypes",
  workflow: "Workflow Taxonomy",
  capability: "Capabilities",
  delivery_or_integration: "Delivery & Integration",
};

const LAYER_DESCRIPTIONS: Record<string, string> = {
  customer_archetype:
    "Who the source company appears to sell to, based on first-party language and named proof.",
  workflow:
    "The workflow cluster that bounds the market box and keeps adjacency mapping from drifting into generic industry search.",
  capability:
    "The concrete solution or product phrases that anchor direct competitor and same-product views.",
  delivery_or_integration:
    "How the product is delivered or connected into the stack, such as APIs, documentation, or integration surfaces.",
};

type SourceDrawerItem = ThesisSourcePill & {
  hostname: string;
  displayUrl: string;
  badge: string | null;
};

type TaxonomyDraft = {
  phrase: string;
  aliases: string;
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

function EvidenceLinks({
  evidenceItems,
}: {
  evidenceItems: ContextPackEvidenceItem[];
}) {
  if (!evidenceItems.length) return null;

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {evidenceItems.map((item) => (
        <a
          key={item.id}
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 rounded-full border border-steel-200 bg-white px-2.5 py-1 text-[11px] font-medium text-steel-600 transition-colors hover:border-oxford hover:text-oxford"
        >
          <ExternalLink className="h-3 w-3" />
          <span>{item.page_title || item.text || item.page_type || "Source"}</span>
        </a>
      ))}
    </div>
  );
}

function TaxonomyCard({
  node,
  draft,
  evidenceItems,
  isPending,
  onDraftChange,
  onSave,
  onScopeChange,
}: {
  node: TaxonomyNode;
  draft: TaxonomyDraft;
  evidenceItems: ContextPackEvidenceItem[];
  isPending: boolean;
  onDraftChange: (nextDraft: TaxonomyDraft) => void;
  onSave: () => void;
  onScopeChange: (status: TaxonomyNode["scope_status"]) => void;
}) {
  const scopeClass =
    node.scope_status === "out_of_scope"
      ? "border-warning/40 bg-warning/5"
      : node.scope_status === "removed"
      ? "border-danger/30 bg-danger/5"
      : "border-steel-200 bg-white";

  return (
    <div className={clsx("rounded-2xl border p-4", scopeClass)}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-oxford">{node.phrase}</div>
          <div className="mt-1 text-xs text-steel-500">
            Confidence {Math.round((node.confidence || 0) * 100)}%
          </div>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <button
            type="button"
            onClick={() => onScopeChange("in_scope")}
            disabled={isPending}
            className={clsx(
              "rounded-full px-2.5 py-1 text-[11px] font-medium",
              node.scope_status === "in_scope"
                ? "bg-success/15 text-success"
                : "border border-steel-200 text-steel-500"
            )}
          >
            In scope
          </button>
          <button
            type="button"
            onClick={() => onScopeChange("out_of_scope")}
            disabled={isPending}
            className={clsx(
              "rounded-full px-2.5 py-1 text-[11px] font-medium",
              node.scope_status === "out_of_scope"
                ? "bg-warning/15 text-warning"
                : "border border-steel-200 text-steel-500"
            )}
          >
            Out of scope
          </button>
          <button
            type="button"
            onClick={() => onScopeChange("removed")}
            disabled={isPending}
            className={clsx(
              "rounded-full px-2.5 py-1 text-[11px] font-medium",
              node.scope_status === "removed"
                ? "bg-danger/15 text-danger"
                : "border border-steel-200 text-steel-500"
            )}
          >
            Remove
          </button>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
        <input
          value={draft.phrase}
          onChange={(event) =>
            onDraftChange({ ...draft, phrase: event.target.value })
          }
          className="input text-sm"
          placeholder="Canonical phrase"
        />
        <input
          value={draft.aliases}
          onChange={(event) =>
            onDraftChange({ ...draft, aliases: event.target.value })
          }
          className="input text-sm"
          placeholder="Aliases, comma separated"
        />
        <button
          type="button"
          onClick={onSave}
          disabled={isPending}
          className="btn-secondary gap-2 whitespace-nowrap disabled:opacity-50"
        >
          <Save className="h-4 w-4" />
          Save
        </button>
      </div>

      {draft.aliases.trim() ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {draft.aliases
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean)
            .map((alias) => (
              <span
                key={alias}
                className="rounded-full border border-steel-200 bg-steel-50 px-2.5 py-1 text-[11px] text-steel-600"
              >
                {alias}
              </span>
            ))}
        </div>
      ) : null}

      <EvidenceLinks evidenceItems={evidenceItems.slice(0, 3)} />
    </div>
  );
}

export default function MarketMapBriefPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: profile, isLoading } = useContextPack(workspaceId);
  const { data: thesisPack } = useThesisPack(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const { data: contextJobs } = useWorkspaceJobs(workspaceId, "context_pack");
  const updateProfile = useUpdateContextPack(workspaceId);
  const updateThesisPack = useUpdateThesisPack(workspaceId);
  const refreshThesisPack = useRefreshThesisPack(workspaceId);

  const [buyerUrl, setBuyerUrl] = useState("");
  const [entryMode, setEntryMode] = useState<"company" | "thesis">("company");
  const [briefText, setBriefText] = useState("");
  const [referenceUrls, setReferenceUrls] = useState<string[]>([]);
  const [newReferenceUrl, setNewReferenceUrl] = useState("");
  const [referenceUrlError, setReferenceUrlError] = useState<string | null>(null);
  const [evidenceUrls, setEvidenceUrls] = useState<string[]>([]);
  const [newEvidenceUrl, setNewEvidenceUrl] = useState("");
  const [evidenceUrlError, setEvidenceUrlError] = useState<string | null>(null);
  const [draftSummary, setDraftSummary] = useState("");
  const [taxonomyDrafts, setTaxonomyDrafts] = useState<Record<string, TaxonomyDraft>>({});
  const [isSourcesDrawerOpen, setIsSourcesDrawerOpen] = useState(false);

  useEffect(() => {
    if (!profile) return;
    setBuyerUrl(profile.buyer_company_url || "");
    setEntryMode(
      profile.buyer_company_url
        ? "company"
        : profile.manual_brief_text
        ? "thesis"
        : "company"
    );
    setBriefText(profile.buyer_company_url ? "" : profile.manual_brief_text || "");
    setReferenceUrls(profile.reference_company_urls || []);
    setEvidenceUrls(profile.reference_evidence_urls || []);
  }, [profile]);

  useEffect(() => {
    setDraftSummary(
      thesisPack?.market_map_brief?.source_summary || thesisPack?.summary || ""
    );
  }, [thesisPack?.market_map_brief?.source_summary, thesisPack?.summary]);

  useEffect(() => {
    const nextDrafts: Record<string, TaxonomyDraft> = {};
    for (const node of thesisPack?.taxonomy_nodes || []) {
      nextDrafts[node.id] = {
        phrase: node.phrase || "",
        aliases: (node.aliases || []).join(", "),
      };
    }
    setTaxonomyDrafts(nextDrafts);
  }, [thesisPack?.taxonomy_nodes]);

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

  const latestCompletedContextJob = useMemo(
    () => contextJobs?.find((job) => job.state === "completed") ?? null,
    [contextJobs]
  );

  const hasGeneratedContextPack = Boolean(
    profile?.context_pack_generated_at ||
      profile?.context_pack_markdown ||
      profile?.context_pack_json
  );

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

  const contextPack = (thesisPack?.context_pack_v2 ||
    profile?.context_pack_json ||
    null) as ContextPackV2 | null;

  const evidenceById = useMemo(() => {
    const map = new Map<string, ContextPackEvidenceItem>();
    for (const item of contextPack?.evidence_items || []) {
      if (!item?.id) continue;
      map.set(item.id, item);
    }
    return map;
  }, [contextPack]);

  const taxonomyByLayer = useMemo(() => {
    const grouped = new Map<string, TaxonomyNode[]>();
    for (const node of thesisPack?.taxonomy_nodes || []) {
      if (node.scope_status === "removed") continue;
      const bucket = grouped.get(node.layer) || [];
      bucket.push(node);
      grouped.set(node.layer, bucket);
    }
    return grouped;
  }, [thesisPack?.taxonomy_nodes]);

  const buyerEvidence = thesisPack?.buyer_evidence || null;
  const showBuyerEvidenceWarning =
    entryMode === "company" && buyerEvidence?.status === "insufficient";
  const crawlButtonLabel = profile?.context_pack_generated_at
    ? "Recrawl and update map"
    : "Generate map from website";
  const thesisButtonLabel = thesisPack?.generated_at
    ? "Regenerate map from thesis"
    : "Generate map from thesis";
  const outputMetaLine = thesisPack?.generated_at
    ? [
        `Last generated ${new Date(thesisPack.generated_at).toLocaleString()}`,
        contextPack?.crawl_coverage?.total_pages
          ? `${contextPack.crawl_coverage.total_pages} pages analyzed`
          : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : "Generate a market-map brief from a company website or thesis to populate this panel";

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

  const patchTaxonomyNodes = async (updater: (nodes: TaxonomyNode[]) => TaxonomyNode[]) => {
    if (!thesisPack) return;
    await updateThesisPack.mutateAsync({
      taxonomy_nodes: updater(thesisPack.taxonomy_nodes || []),
    });
  };

  const saveTaxonomyNode = async (nodeId: string) => {
    const draft = taxonomyDrafts[nodeId];
    if (!draft) return;
    await patchTaxonomyNodes((nodes) =>
      nodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              phrase: draft.phrase.trim() || node.phrase,
              aliases: draft.aliases
                .split(",")
                .map((value) => value.trim())
                .filter(Boolean),
            }
          : node
      )
    );
  };

  const setTaxonomyScope = async (
    nodeId: string,
    scopeStatus: TaxonomyNode["scope_status"]
  ) => {
    await patchTaxonomyNodes((nodes) =>
      nodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              scope_status: scopeStatus,
            }
          : node
      )
    );
  };

  const saveSummary = async () => {
    await updateThesisPack.mutateAsync({
      summary: draftSummary.trim() || null,
    });
  };

  const confirmBrief = async () => {
    await updateThesisPack.mutateAsync({ confirmed: true });
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

  const evidenceForIds = (ids: string[] = []) =>
    ids
      .map((id) => evidenceById.get(id))
      .filter((item): item is ContextPackEvidenceItem => Boolean(item))
      .slice(0, 3);

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
          title="Market Map Brief"
          subtitle="Ground the market map in a source-company crawl, confirm the customer/workflow/capability taxonomy, and activate the map lenses you want to carry into discovery."
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

      <div className="space-y-5 border border-steel-200 bg-white p-6">
        <div>
          <p className="mb-2 text-[11px] font-medium uppercase tracking-widest text-steel-400">
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
              Source company website
            </button>
            <button
              type="button"
              onClick={() => setEntryMode("thesis")}
              className={clsx(
                "flex-1 border-l border-steel-200 px-3 py-2 text-sm font-medium transition-colors",
                entryMode === "thesis"
                  ? "bg-oxford text-white"
                  : "bg-white text-steel-600 hover:text-oxford"
              )}
            >
              Investment thesis
            </button>
          </div>
        </div>

        {entryMode === "company" ? (
          <div>
            <label className="mb-1.5 flex items-center justify-between">
              <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                Source company website
              </span>
              <span className="text-[11px] text-warning">Required</span>
            </label>
            <input
              type="url"
              value={buyerUrl}
              onChange={(event) => setBuyerUrl(event.target.value)}
              placeholder="https://company.com"
              className="input"
            />
          </div>
        ) : (
          <div>
            <label className="mb-1.5 flex items-center justify-between">
              <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                Thesis or sourcing note
              </span>
              <span className="text-[11px] text-warning">Required</span>
            </label>
            <textarea
              value={briefText}
              onChange={(event) => setBriefText(event.target.value)}
              placeholder="Asset managers who buy front-office PMS/OMS often also buy adjacent workflows such as voting-rights software, reporting, and compliance tooling..."
              className="min-h-[148px] w-full resize-none border border-steel-200 bg-white px-3 py-2 text-sm text-steel-900 placeholder:text-steel-400 focus:border-oxford focus:outline-none focus:ring-1 focus:ring-oxford/20"
            />
          </div>
        )}

        <div>
          <label className="mb-1.5 flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Comparable companies
            </span>
            <span className="text-[11px] text-steel-400">Optional</span>
          </label>
          {referenceUrls.length ? (
            <div className="mb-2 space-y-1.5">
              {referenceUrls.map((url, index) => (
                <div
                  key={`${url}-${index}`}
                  className="flex items-center gap-2 border border-steel-200 bg-steel-50 px-2.5 py-1.5"
                >
                  <Globe className="h-3 w-3 shrink-0 text-steel-400" />
                  <span className="flex-1 truncate font-mono text-xs text-steel-600">
                    {url}
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      setReferenceUrls(referenceUrls.filter((_, itemIndex) => itemIndex !== index))
                    }
                    className="text-steel-400 transition-colors hover:text-danger"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <div className="flex gap-2">
            <input
              type="url"
              value={newReferenceUrl}
              onChange={(event) => {
                setNewReferenceUrl(event.target.value);
                setReferenceUrlError(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  handleAddReference();
                }
              }}
              placeholder="https://comparable-company.com"
              className="input text-sm"
            />
            <button type="button" onClick={handleAddReference} className="btn-secondary px-3">
              <Plus className="h-4 w-4" />
            </button>
          </div>
          {referenceUrlError ? (
            <p className="mt-1 text-xs text-danger">{referenceUrlError}</p>
          ) : null}
        </div>

        <div>
          <label className="mb-1.5 flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Supporting evidence
            </span>
            <span className="text-[11px] text-success">High value</span>
          </label>
          {evidenceUrls.length ? (
            <div className="mb-2 space-y-1.5">
              {evidenceUrls.map((url, index) => (
                <div
                  key={`${url}-${index}`}
                  className="flex items-center gap-2 border border-steel-200 bg-steel-50 px-2.5 py-1.5"
                >
                  <Globe className="h-3 w-3 shrink-0 text-steel-400" />
                  <span className="flex-1 truncate font-mono text-xs text-steel-600">
                    {url}
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      setEvidenceUrls(evidenceUrls.filter((_, itemIndex) => itemIndex !== index))
                    }
                    className="text-steel-400 transition-colors hover:text-danger"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <div className="flex gap-2">
            <input
              type="url"
              value={newEvidenceUrl}
              onChange={(event) => {
                setNewEvidenceUrl(event.target.value);
                setEvidenceUrlError(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  handleAddEvidence();
                }
              }}
              placeholder="https://company.com/customer-story"
              className="input text-sm"
            />
            <button type="button" onClick={handleAddEvidence} className="btn-secondary px-3">
              <Plus className="h-4 w-4" />
            </button>
          </div>
          {evidenceUrlError ? (
            <p className="mt-1 text-xs text-danger">{evidenceUrlError}</p>
          ) : null}
        </div>

        <div className="space-y-2 border-t border-steel-100 pt-3">
          <button
            type="button"
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
                ? jobRunner.isRunning || !buyerUrl.trim()
                : refreshThesisPack.isPending || !briefText.trim())
            }
            className="btn-primary w-full gap-2 disabled:opacity-50"
          >
            {entryMode === "company" && jobRunner.isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running... {Math.round(jobRunner.progress * 100)}%
              </>
            ) : (
              <>
                <RefreshCw className="h-4 w-4" />
                {entryMode === "company" ? crawlButtonLabel : thesisButtonLabel}
              </>
            )}
          </button>

          {entryMode === "company" && hasGeneratedContextPack ? (
            <button
              type="button"
              onClick={async () => {
                await saveProfileInputs();
                await refreshThesisPack.mutateAsync();
              }}
              disabled={updateProfile.isPending || refreshThesisPack.isPending}
              className="btn-secondary w-full gap-1.5 disabled:opacity-50"
            >
              {refreshThesisPack.isPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <RefreshCw className="h-3 w-3" />
              )}
              Re-run reasoning on current crawl
            </button>
          ) : null}
        </div>

        {entryMode === "company" && jobRunner.isRunning ? (
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
        ) : null}

        {entryMode === "company" && !jobRunner.isRunning ? (
          <JobRunSummary job={latestCompletedContextJob} />
        ) : null}

        {jobRunner.jobError ? (
          <div className="border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger">
            {jobRunner.jobError}
          </div>
        ) : null}
      </div>

      <div className="space-y-5 border border-steel-200 bg-white p-6">
        <div>
          <h3 className="font-serif text-xl text-oxford">System Market Map Brief</h3>
          <p className="mt-1 text-xs text-steel-400">{outputMetaLine}</p>
        </div>

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

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,0.7fr)]">
          <div className="rounded-2xl border border-steel-200 bg-steel-50 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                Source Summary
              </span>
              <button
                type="button"
                onClick={saveSummary}
                disabled={updateThesisPack.isPending}
                className="inline-flex items-center gap-1 text-xs font-medium text-oxford transition-colors hover:text-oxford-light disabled:opacity-50"
              >
                <Save className="h-3.5 w-3.5" />
                {updateThesisPack.isPending ? "Saving..." : "Save"}
              </button>
            </div>
            {thesisPack?.market_map_brief?.reasoning_status &&
            thesisPack.market_map_brief.reasoning_status !== "success" ? (
              <div className="mb-3 rounded-2xl border border-warning/30 bg-warning/5 px-3 py-2 text-sm text-warning-dark">
                <div className="flex items-start gap-2">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <div>
                    <div className="font-medium">
                      {thesisPack.market_map_brief.reasoning_status === "not_applicable"
                        ? "Reasoning not run yet"
                        : "Reasoning degraded"}
                    </div>
                    <div className="mt-1 text-warning-dark/80">
                      {thesisPack.market_map_brief.reasoning_warning ||
                        "Showing a fallback summary from the source evidence graph."}
                    </div>
                  </div>
                </div>
              </div>
            ) : thesisPack?.market_map_brief?.reasoning_provider ? (
              <div className="mb-3 rounded-2xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
                Reasoning succeeded via{" "}
                <span className="font-medium">
                  {thesisPack.market_map_brief.reasoning_provider}
                  {thesisPack.market_map_brief.reasoning_model
                    ? ` · ${thesisPack.market_map_brief.reasoning_model}`
                    : ""}
                </span>
                .
              </div>
            ) : null}
            <textarea
              value={draftSummary}
              onChange={(event) => setDraftSummary(event.target.value)}
              placeholder="Generate a brief to populate this system summary."
              className="min-h-[150px] w-full resize-none border border-steel-200 bg-white px-3 py-2 text-sm text-steel-800 placeholder:text-steel-400 focus:border-oxford focus:outline-none focus:ring-1 focus:ring-oxford/20"
            />
          </div>

          <div className="rounded-2xl border border-steel-200 bg-white p-4">
            <div className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Crawl Coverage
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                <div className="text-xs text-steel-500">Sites</div>
                <div className="mt-1 text-2xl font-semibold text-oxford">
                  {contextPack?.crawl_coverage?.total_sites ?? 0}
                </div>
              </div>
              <div className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                <div className="text-xs text-steel-500">Pages</div>
                <div className="mt-1 text-2xl font-semibold text-oxford">
                  {contextPack?.crawl_coverage?.total_pages ?? 0}
                </div>
              </div>
              <div className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                <div className="text-xs text-steel-500">Signal pages</div>
                <div className="mt-1 text-2xl font-semibold text-oxford">
                  {contextPack?.crawl_coverage?.pages_with_signals ?? 0}
                </div>
              </div>
              <div className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                <div className="text-xs text-steel-500">Career pages kept</div>
                <div className="mt-1 text-2xl font-semibold text-oxford">
                  {contextPack?.crawl_coverage?.career_pages_selected ?? 0}
                </div>
              </div>
            </div>

            <div className="mt-4 text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Strongest Evidence Buckets
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {(thesisPack?.market_map_brief?.strongest_evidence_buckets || []).map((bucket) => (
                <span
                  key={bucket.label}
                  className="rounded-full border border-steel-200 bg-steel-50 px-3 py-1 text-xs text-steel-600"
                >
                  {bucket.label}: {bucket.count}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-5">
          {(["customer_archetype", "workflow", "capability", "delivery_or_integration"] as const).map((layer) => {
            const nodes = taxonomyByLayer.get(layer) || [];

            return (
              <div key={layer}>
                <div className="mb-3 flex items-center gap-3">
                  <div>
                    <div className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
                      {LAYER_LABELS[layer]}
                    </div>
                    <p className="mt-1 text-sm text-steel-500">
                      {LAYER_DESCRIPTIONS[layer]}
                    </p>
                  </div>
                </div>

                {nodes.length ? (
                  <div className="grid gap-4 lg:grid-cols-2">
                    {nodes.map((node) => (
                      <TaxonomyCard
                        key={node.id}
                        node={node}
                        draft={
                          taxonomyDrafts[node.id] || {
                            phrase: node.phrase || "",
                            aliases: (node.aliases || []).join(", "),
                          }
                        }
                        evidenceItems={evidenceForIds(node.evidence_ids)}
                        isPending={updateThesisPack.isPending}
                        onDraftChange={(nextDraft) =>
                          setTaxonomyDrafts((current) => ({
                            ...current,
                            [node.id]: nextDraft,
                          }))
                        }
                        onSave={() => saveTaxonomyNode(node.id)}
                        onScopeChange={(status) => setTaxonomyScope(node.id, status)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-dashed border-steel-200 bg-steel-50 px-4 py-5 text-sm text-steel-500">
                    No {LAYER_LABELS[layer].toLowerCase()} extracted yet.
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <div className="rounded-2xl border border-steel-200 bg-white p-4">
            <div className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Named Customer Proof
            </div>
            <div className="mt-3 space-y-3">
              {(thesisPack?.market_map_brief?.named_customer_proof || []).length ? (
                thesisPack?.market_map_brief?.named_customer_proof.map((item) => (
                  <div key={`${item.name}-${item.source_url || item.evidence_id || ""}`} className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                    <div className="text-sm font-semibold text-oxford">{item.name}</div>
                    {item.context ? (
                      <p className="mt-1 text-sm text-steel-600">{item.context}</p>
                    ) : null}
                    <EvidenceLinks
                      evidenceItems={evidenceForIds(item.evidence_id ? [item.evidence_id] : [])}
                    />
                    {!item.evidence_id && item.source_url ? (
                      <a
                        href={item.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-oxford hover:underline"
                      >
                        <ExternalLink className="h-3 w-3" />
                        Open source
                      </a>
                    ) : null}
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-steel-200 bg-steel-50 px-4 py-5 text-sm text-steel-500">
                  No named-customer proof extracted yet.
                </div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-steel-200 bg-white p-4">
            <div className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Integration / Partner Proof
            </div>
            <div className="mt-3 space-y-3">
              {(thesisPack?.market_map_brief?.integration_partner_proof || []).length ? (
                thesisPack?.market_map_brief?.integration_partner_proof.map((item) => (
                  <div key={`${item.name}-${item.source_url || item.evidence_id || ""}`} className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                    <div className="text-sm font-semibold text-oxford">{item.name}</div>
                    <EvidenceLinks
                      evidenceItems={evidenceForIds(item.evidence_id ? [item.evidence_id] : [])}
                    />
                    {!item.evidence_id && item.source_url ? (
                      <a
                        href={item.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-oxford hover:underline"
                      >
                        <ExternalLink className="h-3 w-3" />
                        Open source
                      </a>
                    ) : null}
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-steel-200 bg-steel-50 px-4 py-5 text-sm text-steel-500">
                  No integration or partner evidence extracted yet.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <div className="rounded-2xl border border-steel-200 bg-white p-4">
            <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-widest text-steel-400">
              <Bot className="h-3.5 w-3.5" />
              Recommended Lenses
            </div>
            <div className="mt-3 space-y-3">
              {(thesisPack?.market_map_brief?.active_lenses || thesisPack?.lens_seeds || []).length ? (
                (thesisPack?.market_map_brief?.active_lenses || thesisPack?.lens_seeds || []).map((lens) => (
                  <div key={lens.id} className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-sm font-semibold text-oxford">{lens.label}</div>
                      <span className="rounded-full border border-steel-200 bg-white px-2.5 py-1 text-[11px] text-steel-500">
                        {Math.round((lens.confidence || 0) * 100)}%
                      </span>
                    </div>
                    {lens.query_phrase ? (
                      <div className="mt-1 text-xs text-steel-500">
                        Seed phrase: {lens.query_phrase}
                      </div>
                    ) : null}
                    <p className="mt-2 text-sm text-steel-600">{lens.rationale}</p>
                    <EvidenceLinks evidenceItems={evidenceForIds(lens.evidence_ids)} />
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-steel-200 bg-steel-50 px-4 py-5 text-sm text-steel-500">
                  No map lenses are active yet. Improve capability, workflow, or customer coverage first.
                </div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-steel-200 bg-white p-4">
            <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-widest text-steel-400">
              <FileText className="h-3.5 w-3.5" />
              Adjacency Hypotheses
            </div>
            <div className="mt-3 space-y-3">
              {(thesisPack?.market_map_brief?.adjacency_hypotheses || []).length ? (
                thesisPack?.market_map_brief?.adjacency_hypotheses.map((item) => (
                  <div key={item.id} className="rounded-2xl border border-steel-200 bg-steel-50 p-3">
                    <div className="text-sm text-steel-700">{item.text}</div>
                    <div className="mt-2 text-xs text-steel-500">
                      Confidence {Math.round((item.confidence || 0) * 100)}%
                    </div>
                    <EvidenceLinks evidenceItems={evidenceForIds(item.evidence_ids)} />
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-steel-200 bg-steel-50 px-4 py-5 text-sm text-steel-500">
                  No adjacency hypotheses generated yet.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <div className="rounded-2xl border border-steel-200 bg-white p-4">
            <div className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Confidence Gaps
            </div>
            <div className="mt-3 space-y-2">
              {(thesisPack?.market_map_brief?.confidence_gaps || []).length ? (
                thesisPack?.market_map_brief?.confidence_gaps.map((gap) => (
                  <div
                    key={gap}
                    className="rounded-2xl border border-warning/25 bg-warning/10 px-3 py-2 text-sm text-steel-700"
                  >
                    {gap}
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-success/20 bg-success/10 px-3 py-2 text-sm text-success">
                  No major confidence gaps flagged by the current artifact set.
                </div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-steel-200 bg-white p-4">
            <div className="text-[11px] font-medium uppercase tracking-widest text-steel-400">
              Open Questions
            </div>
            <div className="mt-3 space-y-2">
              {(thesisPack?.market_map_brief?.open_questions || thesisPack?.open_questions || []).map((question) => (
                <div
                  key={question}
                  className="rounded-2xl border border-steel-200 bg-steel-50 px-3 py-2 text-sm text-steel-700"
                >
                  {question}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-steel-100 pt-4">
          <div className="text-sm text-steel-500">
            {thesisPack?.confirmed_at
              ? `Brief confirmed ${new Date(thesisPack.confirmed_at).toLocaleString()}`
              : "Confirm this brief once the taxonomy and lenses match the market map you want to search."}
          </div>
          <button
            type="button"
            onClick={confirmBrief}
            disabled={updateThesisPack.isPending}
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
