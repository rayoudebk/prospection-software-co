"use client";

import { useParams } from "next/navigation";
import {
  useCompanyContextPack,
  useConfirmScopeReview,
  useExpansionBrief,
  useGenerateExpansionBrief,
  useGates,
  useScopeReview,
} from "@/lib/hooks";
import { AdjacencyBox, ScopeReviewItem } from "@/lib/api";
import {
  ArrowRight,
  AlertCircle,
  Check,
  CheckCircle,
  CircleDashed,
  ExternalLink,
  MapPinned,
  Network,
  Orbit,
  Loader2,
  ShieldCheck,
  Sparkles,
  Tags,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import { ReportArtifactRenderer } from "@/components/ReportArtifactRenderer";
import clsx from "clsx";

const CRITICALITY_FIELDS: Array<{
  key:
    | "market_importance"
    | "operational_centrality"
    | "workflow_criticality"
    | "daily_operator_usage"
    | "switching_cost_intensity";
  label: string;
}> = [
  { key: "market_importance", label: "Market importance" },
  { key: "operational_centrality", label: "Operational centrality" },
  { key: "workflow_criticality", label: "Workflow criticality" },
  { key: "daily_operator_usage", label: "Daily operator usage" },
  { key: "switching_cost_intensity", label: "Switching cost intensity" },
];

function prettyLabel(raw: string | null | undefined): string {
  if (!raw) return "Unknown";
  return raw
    .split("_")
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
}

function confidencePercent(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${Math.round(value * 100)}%`;
}

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "confirmed" || status === "source_grounded" || status === "user_kept"
      ? "border-success/40 bg-success/10 text-success"
      : status === "user_removed" || status === "user_deprioritized"
        ? "border-warning/40 bg-warning/10 text-warning"
        : "border-steel-300 bg-steel-100 text-steel-700";
  return (
    <span className={clsx("px-2 py-1 text-xs border font-medium", tone)}>
      {prettyLabel(status)}
    </span>
  );
}

function ScopeItemCard({ item }: { item: ScopeReviewItem }) {
  const criticality = CRITICALITY_FIELDS.flatMap(({ key, label }) => {
    const value = item[key];
    return value ? [{ key, label, value: prettyLabel(value) }] : [];
  });

  const evidencePreview = item.evidence_urls.slice(0, 2);

  return (
    <article className="border border-steel-200 bg-white p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="text-sm font-semibold text-oxford">{item.label}</h4>
          <p className="text-xs text-steel-500 mt-1">
            {prettyLabel(item.scope_item_type)}
            {item.priority_tier ? ` · ${prettyLabel(item.priority_tier)}` : ""}
          </p>
        </div>
        <StatusBadge status={item.status} />
      </div>

      {item.why_it_matters ? (
        <p className="text-sm text-steel-700 leading-relaxed">{item.why_it_matters}</p>
      ) : null}

      <div className="flex flex-wrap gap-2 text-[11px] text-steel-600">
        <span className="inline-flex items-center gap-1 border border-steel-200 bg-steel-50 px-2 py-1">
          <ShieldCheck className="w-3.5 h-3.5" />
          Confidence {confidencePercent(item.confidence)}
        </span>
        <span className="inline-flex items-center gap-1 border border-steel-200 bg-steel-50 px-2 py-1">
          <Orbit className="w-3.5 h-3.5" />
          Origin {prettyLabel(item.origin)}
        </span>
        <span className="inline-flex items-center gap-1 border border-steel-200 bg-steel-50 px-2 py-1">
          <ExternalLink className="w-3.5 h-3.5" />
          {item.evidence_urls.length} evidence link{item.evidence_urls.length === 1 ? "" : "s"}
        </span>
      </div>

      {criticality.length ? (
        <div className="space-y-2">
          <p className="text-[11px] uppercase tracking-[0.14em] text-steel-400">
            Criticality Signals
          </p>
          <div className="flex flex-wrap gap-2">
            {criticality.map((entry) => (
              <span
                key={`${item.id}-${entry.key}`}
                className="inline-flex items-center border border-info/30 bg-info/10 px-2 py-1 text-[11px] text-info"
              >
                {entry.label}: {entry.value}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {item.source_entity_names.length ? (
        <div className="space-y-2">
          <p className="text-[11px] uppercase tracking-[0.14em] text-steel-400">
            Referenced Entities
          </p>
          <div className="flex flex-wrap gap-2">
            {item.source_entity_names.slice(0, 4).map((entity) => (
              <span
                key={`${item.id}-${entity}`}
                className="inline-flex items-center border border-steel-200 bg-steel-50 px-2 py-1 text-[11px] text-steel-700"
              >
                {entity}
              </span>
            ))}
            {item.source_entity_names.length > 4 ? (
              <span className="inline-flex items-center border border-steel-200 bg-white px-2 py-1 text-[11px] text-steel-500">
                +{item.source_entity_names.length - 4} more
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {evidencePreview.length ? (
        <div className="space-y-2">
          <p className="text-[11px] uppercase tracking-[0.14em] text-steel-400">
            Evidence Preview
          </p>
          <div className="space-y-1.5">
            {evidencePreview.map((url) => (
              <a
                key={`${item.id}-${url}`}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="block truncate text-xs text-info hover:text-info-dark"
                title={url}
              >
                {url}
              </a>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  );
}

function ScopeGroup({
  title,
  subtitle,
  items,
}: {
  title: string;
  subtitle: string;
  items: ScopeReviewItem[];
}) {
  return (
    <section className="border border-steel-200 bg-steel-50 p-4 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-oxford">{title}</h3>
          <p className="text-xs text-steel-500 mt-1">{subtitle}</p>
        </div>
        <span className="text-xs font-medium border border-steel-200 bg-white px-2 py-1 text-steel-700">
          {items.length}
        </span>
      </div>

      {items.length ? (
        <div className="space-y-3">
          {items.map((item) => (
            <ScopeItemCard key={item.id} item={item} />
          ))}
        </div>
      ) : (
        <div className="border border-dashed border-steel-300 bg-white p-4 text-sm text-steel-500">
          No items yet.
        </div>
      )}
    </section>
  );
}

function AdjacencyBoxCard({ box }: { box: AdjacencyBox }) {
  return (
    <article className="border border-steel-200 bg-white p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="text-sm font-semibold text-oxford">{box.label}</h4>
          <p className="text-xs text-steel-500 mt-1">
            {prettyLabel(box.adjacency_kind)} · {prettyLabel(box.priority_tier)}
          </p>
        </div>
        <StatusBadge status={box.status} />
      </div>

      {box.why_it_matters ? (
        <p className="text-sm text-steel-700 leading-relaxed">{box.why_it_matters}</p>
      ) : null}

      <div className="flex flex-wrap gap-2 text-[11px] text-steel-600">
        <span className="inline-flex items-center gap-1 border border-steel-200 bg-steel-50 px-2 py-1">
          <ShieldCheck className="w-3.5 h-3.5" />
          Confidence {confidencePercent(box.confidence)}
        </span>
        <span className="inline-flex items-center gap-1 border border-steel-200 bg-steel-50 px-2 py-1">
          <Tags className="w-3.5 h-3.5" />
          {box.company_seed_ids.length} company seed{box.company_seed_ids.length === 1 ? "" : "s"}
        </span>
        <span className="inline-flex items-center gap-1 border border-steel-200 bg-steel-50 px-2 py-1">
          <Network className="w-3.5 h-3.5" />
          {box.retrieval_query_seeds.length} query seed{box.retrieval_query_seeds.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        <span className="inline-flex items-center border border-info/30 bg-info/10 px-2 py-1 text-[11px] text-info">
          Workflow: {prettyLabel(box.criticality.workflow_criticality)}
        </span>
        <span className="inline-flex items-center border border-info/30 bg-info/10 px-2 py-1 text-[11px] text-info">
          Daily usage: {prettyLabel(box.criticality.daily_operator_usage)}
        </span>
        <span className="inline-flex items-center border border-info/30 bg-info/10 px-2 py-1 text-[11px] text-info">
          Switching: {prettyLabel(box.criticality.switching_cost_intensity)}
        </span>
      </div>
    </article>
  );
}

export default function ScopeReviewPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: scopeReview, isLoading } = useScopeReview(workspaceId);
  const { data: companyContext } = useCompanyContextPack(workspaceId);
  const { data: expansionArtifact } = useExpansionBrief(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const confirmScopeReview = useConfirmScopeReview(workspaceId);
  const generateExpansionBrief = useGenerateExpansionBrief(workspaceId);
  const isExpansionGenerating =
    generateExpansionBrief.isPending || expansionArtifact?.status === "generating";
  const scopeNeedsExpansion = scopeReview?.expansion_status !== "ready";
  const expansionBrief = expansionArtifact?.expansion_brief;
  const adjacencyBoxes = expansionBrief?.adjacency_boxes || [];
  const companySeeds = expansionBrief?.company_seeds || [];
  const technologyShiftClaims = expansionBrief?.technology_shift_claims || [];
  const sourceItemCount =
    (scopeReview?.source_capabilities.length || 0) +
    (scopeReview?.source_customer_segments.length || 0) +
    (scopeReview?.source_workflows.length || 0) +
    (scopeReview?.source_delivery_or_integration.length || 0);
  const expansionItemCount =
    (scopeReview?.adjacency_boxes.length || 0) +
    (scopeReview?.named_account_anchors.length || 0) +
    (scopeReview?.geography_expansions.length || 0);
  const reportGeneratedAt =
    expansionArtifact?.expansion_report?.generated_at || expansionArtifact?.generated_at || null;

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
        title="Expansion Brief"
        subtitle="Review the bounded one-hop expansion artifact before moving into scope review and universe discovery."
      />

      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.scope_review ? "bg-success/10 border-success" : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.scope_review ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.scope_review ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.scope_review
                ? "Scope confirmed — you can proceed to Universe"
                : gates.missing_items.scope_review?.join(", ") || "Confirm scope review to continue"}
            </span>
          </div>
        </div>
      )}

      <section className="border border-steel-200 bg-steel-50 p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-[0.18em] text-steel-400">
              Expansion Artifact
            </div>
            <h3 className="text-lg font-semibold text-oxford mt-1">
              Canonical v3 adjacency intelligence
            </h3>
            <p className="text-sm text-steel-600 mt-1 max-w-3xl">
              This review combines the report artifact with structured adjacency boxes, technology shifts, and seed-company signals for downstream Universe discovery.
            </p>
          </div>
          <button
            onClick={() => generateExpansionBrief.mutate()}
            disabled={isExpansionGenerating}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {isExpansionGenerating ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            {expansionArtifact?.expansion_report ? "Regenerate Expansion Brief" : "Generate Expansion Brief"}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
          <div className="border border-steel-200 bg-white p-3">
            <div className="text-xs uppercase tracking-[0.14em] text-steel-400">Adjacency boxes</div>
            <div className="text-2xl font-semibold text-oxford mt-1">{adjacencyBoxes.length}</div>
          </div>
          <div className="border border-steel-200 bg-white p-3">
            <div className="text-xs uppercase tracking-[0.14em] text-steel-400">Company seeds</div>
            <div className="text-2xl font-semibold text-oxford mt-1">{companySeeds.length}</div>
          </div>
          <div className="border border-steel-200 bg-white p-3">
            <div className="text-xs uppercase tracking-[0.14em] text-steel-400">Tech shifts</div>
            <div className="text-2xl font-semibold text-oxford mt-1">{technologyShiftClaims.length}</div>
          </div>
          <div className="border border-steel-200 bg-white p-3">
            <div className="text-xs uppercase tracking-[0.14em] text-steel-400">Generated</div>
            <div className="text-sm font-medium text-oxford mt-1">
              {reportGeneratedAt ? new Date(reportGeneratedAt).toLocaleString() : "Not generated"}
            </div>
          </div>
        </div>

        {expansionArtifact?.warning ? (
          <p className="text-sm text-warning">{expansionArtifact.warning}</p>
        ) : null}
      </section>

      {adjacencyBoxes.length ? (
        <section className="space-y-3">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-steel-400">
            <Network className="w-4 h-4" />
            Structured Adjacency Boxes
          </div>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {adjacencyBoxes.slice(0, 6).map((box) => (
              <AdjacencyBoxCard key={box.id} box={box} />
            ))}
          </div>
          {adjacencyBoxes.length > 6 ? (
            <p className="text-sm text-steel-500">
              Showing 6 of {adjacencyBoxes.length} adjacency boxes from the canonical expansion brief.
            </p>
          ) : null}
        </section>
      ) : null}

      {expansionArtifact?.expansion_report ? (
        <ReportArtifactRenderer
          artifact={expansionArtifact.expansion_report}
          onRegenerate={() => {
            if (!isExpansionGenerating) generateExpansionBrief.mutate();
          }}
        />
      ) : (
        <div className="border border-dashed border-steel-300 bg-white p-5 text-sm text-steel-600">
          No expansion report available yet. Generate the expansion brief to unlock adjacency review context and downstream scope signals.
        </div>
      )}

      {companyContext?.sourcing_brief?.source_summary && (
        <div className="bg-oxford text-white border border-oxford-dark p-5">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-steel-400 mb-2">
            <Sparkles className="w-4 h-4" />
            Source Brief
          </div>
          <p className="text-sm text-steel-100 leading-relaxed">
            {companyContext.sourcing_brief.source_summary}
          </p>
        </div>
      )}

      {scopeReview && (
        <div className="space-y-4">
          {scopeNeedsExpansion ? (
            <div className="bg-warning/10 border border-warning p-4 text-sm text-warning">
              {scopeReview.expansion_warning
                ? scopeReview.expansion_warning
                : "Adjacent nodes are unavailable until the expansion brief is generated. Generate tab 2 first, then validate scope here."}
            </div>
          ) : null}

          <section className="border border-steel-200 bg-steel-50 p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-[11px] uppercase tracking-[0.18em] text-steel-400">
                  Scope Review Board
                </div>
                <h3 className="text-lg font-semibold text-oxford mt-1">
                  Validate core baseline and expansion-lane criticality
                </h3>
              </div>
              <div className="flex flex-wrap gap-2 text-xs">
                <span className="inline-flex items-center gap-1 border border-steel-200 bg-white px-2 py-1 text-steel-700">
                  <CircleDashed className="w-3.5 h-3.5" />
                  Core items {sourceItemCount}
                </span>
                <span className="inline-flex items-center gap-1 border border-steel-200 bg-white px-2 py-1 text-steel-700">
                  <ArrowRight className="w-3.5 h-3.5" />
                  Expansion items {expansionItemCount}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-steel-400">
                  <Network className="w-4 h-4" />
                  Core Workflow Baseline
                </div>
                <ScopeGroup
                  title="Source Capabilities"
                  subtitle="Core capabilities from source-company evidence and taxonomy."
                  items={scopeReview.source_capabilities}
                />
                <ScopeGroup
                  title="Source Customer Segments"
                  subtitle="Customer segments directly grounded in source-company footprint."
                  items={scopeReview.source_customer_segments}
                />
                <ScopeGroup
                  title="Source Workflows"
                  subtitle="Operational workflows where the source company currently participates."
                  items={scopeReview.source_workflows}
                />
                <ScopeGroup
                  title="Source Delivery / Integration"
                  subtitle="Integration and delivery surfaces tied to source workflows."
                  items={scopeReview.source_delivery_or_integration}
                />
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-steel-400">
                  <MapPinned className="w-4 h-4" />
                  Expansion Lanes
                </div>
                <section className="border border-steel-200 bg-steel-50 p-4 space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-semibold text-oxford">Adjacency Boxes</h3>
                      <p className="text-xs text-steel-500 mt-1">
                        Canonical expansion lanes with workflow criticality and retrieval intent.
                      </p>
                    </div>
                    <span className="text-xs font-medium border border-steel-200 bg-white px-2 py-1 text-steel-700">
                      {scopeReview.adjacency_boxes.length}
                    </span>
                  </div>

                  {scopeReview.adjacency_boxes.length ? (
                    <div className="space-y-3">
                      {scopeReview.adjacency_boxes.map((box) => (
                        <AdjacencyBoxCard key={box.id} box={box} />
                      ))}
                    </div>
                  ) : (
                    <div className="border border-dashed border-steel-300 bg-white p-4 text-sm text-steel-500">
                      No adjacency boxes yet.
                    </div>
                  )}
                </section>
                <ScopeGroup
                  title="Named Account Anchors"
                  subtitle="Specific account anchors surfaced for strategic relevance."
                  items={scopeReview.named_account_anchors}
                />
                <ScopeGroup
                  title="Geography Expansions"
                  subtitle="Geo expansion opportunities with supporting context."
                  items={scopeReview.geography_expansions}
                />
              </div>
            </div>
          </section>

          {!scopeNeedsExpansion ? (
            <div className="border border-info/30 bg-info/5 p-4 text-sm text-info-dark">
              Scope includes expansion lanes and can now feed targeted universe discovery signals.
            </div>
          ) : (
            <div className="border border-warning/30 bg-warning/10 p-4 text-sm text-warning">
              Expansion-derived lanes are incomplete. Generate Expansion Brief first, then confirm scope.
            </div>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => confirmScopeReview.mutate()}
          disabled={confirmScopeReview.isPending || !scopeReview}
          className="btn-primary flex items-center gap-2 disabled:opacity-50"
        >
          {confirmScopeReview.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
          Confirm Scope
        </button>
        <p className="text-sm text-steel-500">
          Confirming scope unlocks Universe discovery with these reviewed lanes.
        </p>
      </div>
    </div>
  );
}
