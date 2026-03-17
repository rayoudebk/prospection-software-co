"use client";

import { useMemo } from "react";
import { useParams } from "next/navigation";
import clsx from "clsx";
import {
  AlertCircle,
  Check,
  CheckCircle,
  Loader2,
  MapPinned,
  Network,
  Sparkles,
} from "lucide-react";

import { ReportArtifactRenderer } from "@/components/ReportArtifactRenderer";
import { StepHeader } from "@/components/StepHeader";
import { AdjacencyBox, ScopeReviewItem } from "@/lib/api";
import {
  useConfirmScopeReview,
  useExpansionBrief,
  useGates,
  useGenerateExpansionBrief,
  useScopeReview,
  useUpdateScopeReview,
} from "@/lib/hooks";

function prettyLabel(raw: string | null | undefined): string {
  if (!raw) return "Unknown";
  return raw
    .split("_")
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
}

function statusTone(status: string) {
  if (status === "user_kept" || status === "source_grounded" || status === "corroborated_expansion") {
    return "border-success/25 bg-success/10 text-success";
  }
  if (status === "user_removed") {
    return "border-danger/25 bg-danger/10 text-danger";
  }
  if (status === "user_deprioritized") {
    return "border-warning/25 bg-warning/10 text-warning-dark";
  }
  return "border-steel-200 bg-steel-50 text-steel-700";
}

function ScopeStatusBadge({ status }: { status: string }) {
  return (
    <span className={clsx("rounded-full border px-2.5 py-1 text-xs font-medium", statusTone(status))}>
      {prettyLabel(status)}
    </span>
  );
}

function DecisionButtons({
  currentStatus,
  disabled,
  onSelect,
}: {
  currentStatus: string;
  disabled: boolean;
  onSelect: (status: "user_kept" | "user_deprioritized" | "user_removed") => void;
}) {
  const options: Array<{
    label: string;
    value: "user_kept" | "user_deprioritized" | "user_removed";
    activeClass: string;
  }> = [
    {
      label: "Focus",
      value: "user_kept",
      activeClass: "border-success/30 bg-success/10 text-success",
    },
    {
      label: "Not Now",
      value: "user_deprioritized",
      activeClass: "border-warning/30 bg-warning/10 text-warning-dark",
    },
    {
      label: "Remove",
      value: "user_removed",
      activeClass: "border-danger/30 bg-danger/10 text-danger",
    },
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {options.map((option) => {
        const isActive = currentStatus === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onSelect(option.value)}
            disabled={disabled}
            className={clsx(
              "rounded-full border px-3 py-1.5 text-sm transition-colors disabled:opacity-50",
              isActive
                ? option.activeClass
                : "border-steel-200 bg-white text-steel-700 hover:border-oxford/30 hover:text-oxford"
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function FocusCard({
  title,
  description,
  status,
  metadata,
  disabled,
  onSelect,
}: {
  title: string;
  description?: string | null;
  status: string;
  metadata: string[];
  disabled: boolean;
  onSelect: (status: "user_kept" | "user_deprioritized" | "user_removed") => void;
}) {
  return (
    <article className="rounded-3xl border border-steel-200 bg-white px-5 py-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-2">
          <h4 className="text-lg leading-7 text-oxford">{title}</h4>
          {metadata.length ? (
            <p className="text-sm text-steel-500">{metadata.join(" · ")}</p>
          ) : null}
        </div>
        <ScopeStatusBadge status={status} />
      </div>

      {description ? <p className="text-sm leading-6 text-steel-700">{description}</p> : null}

      <DecisionButtons currentStatus={status} disabled={disabled} onSelect={onSelect} />
    </article>
  );
}

function FocusSection({
  title,
  subtitle,
  items,
  disabled,
  onSelect,
}: {
  title: string;
  subtitle: string;
  items: Array<
    | (ScopeReviewItem & { kind: "scope" })
    | (AdjacencyBox & { kind: "adjacency" })
  >;
  disabled: boolean;
  onSelect: (id: string, status: "user_kept" | "user_deprioritized" | "user_removed") => void;
}) {
  if (!items.length) return null;

  return (
    <section className="space-y-4">
      <div>
        <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
          {title}
        </div>
        <p className="mt-2 text-sm text-steel-500">{subtitle}</p>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {items.map((item) => {
          const metadata =
            item.kind === "adjacency"
              ? [
                  prettyLabel(item.adjacency_kind),
                  prettyLabel(item.priority_tier),
                  `${item.retrieval_query_seeds.length} query seed${item.retrieval_query_seeds.length === 1 ? "" : "s"}`,
                ]
              : [
                  prettyLabel(item.scope_item_type),
                  item.priority_tier ? prettyLabel(item.priority_tier) : null,
                  `${item.evidence_urls.length} evidence link${item.evidence_urls.length === 1 ? "" : "s"}`,
                ].filter(Boolean) as string[];

          return (
            <FocusCard
              key={item.id}
              title={item.label}
              description={item.why_it_matters}
              status={item.status}
              metadata={metadata}
              disabled={disabled}
              onSelect={(status) => onSelect(item.id, status)}
            />
          );
        })}
      </div>
    </section>
  );
}

export default function ExpansionBriefPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: scopeReview, isLoading } = useScopeReview(workspaceId);
  const { data: expansionArtifact } = useExpansionBrief(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const generateExpansionBrief = useGenerateExpansionBrief(workspaceId);
  const updateScopeReview = useUpdateScopeReview(workspaceId);
  const confirmScopeReview = useConfirmScopeReview(workspaceId);

  const isExpansionGenerating =
    generateExpansionBrief.isPending || expansionArtifact?.status === "generating";

  const focusCounts = useMemo(() => {
    const items = [
      ...(scopeReview?.adjacency_boxes || []),
      ...(scopeReview?.named_account_anchors || []),
      ...(scopeReview?.geography_expansions || []),
    ];
    return {
      total: items.length,
      kept: items.filter((item) => item.status === "user_kept").length,
      deprioritized: items.filter((item) => item.status === "user_deprioritized").length,
      removed: items.filter((item) => item.status === "user_removed").length,
    };
  }, [scopeReview]);

  const updateDecision = async (
    id: string,
    status: "user_kept" | "user_deprioritized" | "user_removed"
  ) => {
    await updateScopeReview.mutateAsync({
      decisions: [{ id, status }],
    });
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
        step={2}
        title="Expansion"
        subtitle="Review the expansion brief, then decide which proposed lanes should feed Universe discovery. The source brief already validated the source truth; this tab is only for choosing focus."
      />

      {gates ? (
        <div
          className={clsx(
            "rounded-3xl border px-5 py-4",
            gates.scope_review ? "border-success/30 bg-success/10" : "border-warning/30 bg-warning/10"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.scope_review ? (
              <CheckCircle className="h-5 w-5 text-success" />
            ) : (
              <AlertCircle className="h-5 w-5 text-warning" />
            )}
            <span className={gates.scope_review ? "font-medium text-success" : "font-medium text-warning-dark"}>
              {gates.scope_review
                ? "Expansion focus validated. Universe is unlocked."
                : "Choose focus areas and validate this tab before moving to Universe."}
            </span>
          </div>
        </div>
      ) : null}

      <section className="rounded-[28px] border border-steel-200 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-steel-400">
              Expansion Brief
            </div>
            <h3 className="mt-2 text-2xl font-semibold text-oxford">
              One-hop expansion from the graph + sourcing brief
            </h3>
            <p className="mt-2 max-w-3xl text-sm text-steel-500">
              This report should stay bounded to plausible adjacent lanes, named-account anchors, and geographies worth carrying into discovery.
            </p>
          </div>
          <button
            type="button"
            onClick={() => generateExpansionBrief.mutate()}
            disabled={isExpansionGenerating}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {isExpansionGenerating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {expansionArtifact?.expansion_report ? "Regenerate Expansion Brief" : "Generate Expansion Brief"}
          </button>
        </div>

        {expansionArtifact?.warning ? (
          <div className="mt-4 rounded-2xl border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning-dark">
            {expansionArtifact.warning}
          </div>
        ) : null}

        <div className="mt-5 grid gap-3 md:grid-cols-4">
          <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
            <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Focus items</div>
            <div className="mt-1 text-2xl font-semibold text-oxford">{focusCounts.total}</div>
          </div>
          <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
            <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Focused</div>
            <div className="mt-1 text-2xl font-semibold text-oxford">{focusCounts.kept}</div>
          </div>
          <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
            <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Not now</div>
            <div className="mt-1 text-2xl font-semibold text-oxford">{focusCounts.deprioritized}</div>
          </div>
          <div className="rounded-2xl border border-steel-200 bg-steel-50 px-4 py-3">
            <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Removed</div>
            <div className="mt-1 text-2xl font-semibold text-oxford">{focusCounts.removed}</div>
          </div>
        </div>
      </section>

      {expansionArtifact?.expansion_report ? (
        <ReportArtifactRenderer artifact={expansionArtifact.expansion_report} />
      ) : (
        <div className="rounded-3xl border border-dashed border-steel-300 bg-white px-5 py-5 text-sm text-steel-500">
          Generate the expansion brief to review proposed lanes and start selecting focus areas.
        </div>
      )}

      {scopeReview?.expansion_warning ? (
        <div className="rounded-3xl border border-warning/30 bg-warning/10 px-5 py-4 text-sm text-warning-dark">
          {scopeReview.expansion_warning}
        </div>
      ) : null}

      <FocusSection
        title="Expansion Lanes"
        subtitle="Pick the adjacent lanes that should actively shape discovery."
        items={(scopeReview?.adjacency_boxes || []).map((item) => ({ ...item, kind: "adjacency" as const }))}
        disabled={updateScopeReview.isPending}
        onSelect={updateDecision}
      />

      <FocusSection
        title="Named Account Anchors"
        subtitle="Keep named accounts only when they sharpen the discovery direction."
        items={(scopeReview?.named_account_anchors || []).map((item) => ({ ...item, kind: "scope" as const }))}
        disabled={updateScopeReview.isPending}
        onSelect={updateDecision}
      />

      <FocusSection
        title="Geography Expansions"
        subtitle="Keep only the geography moves that should materially affect the search universe."
        items={(scopeReview?.geography_expansions || []).map((item) => ({ ...item, kind: "scope" as const }))}
        disabled={updateScopeReview.isPending}
        onSelect={updateDecision}
      />

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-[28px] border border-steel-200 bg-white px-6 py-5 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="space-y-1">
          <div className="text-sm font-medium text-oxford">Validate expansion focus</div>
          <p className="text-sm text-steel-500">
            Confirm the kept lanes and anchors once the expansion report looks right for discovery.
          </p>
        </div>
        <button
          type="button"
          onClick={() => confirmScopeReview.mutate()}
          disabled={confirmScopeReview.isPending || !scopeReview}
          className="btn-primary flex items-center gap-2 disabled:opacity-50"
        >
          {confirmScopeReview.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Check className="h-4 w-4" />
          )}
          Validate Expansion Focus
        </button>
      </div>
    </div>
  );
}
