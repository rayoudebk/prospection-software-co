"use client";

import { useParams } from "next/navigation";
import {
  useCompanyContextPack,
  useConfirmScopeReview,
  useGates,
  useScopeReview,
} from "@/lib/hooks";
import { ScopeReviewItem } from "@/lib/api";
import {
  AlertCircle,
  Check,
  CheckCircle,
  Loader2,
  Sparkles,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "confirmed" || status === "source_grounded" || status === "user_kept"
      ? "border-success/40 bg-success/10 text-success"
      : status === "user_removed" || status === "user_deprioritized"
        ? "border-warning/40 bg-warning/10 text-warning"
        : "border-steel-300 bg-steel-100 text-steel-700";
  return <span className={clsx("px-2 py-1 text-xs border", tone)}>{status}</span>;
}

function ScopeItemSection({
  title,
  items,
}: {
  title: string;
  items: ScopeReviewItem[];
}) {
  if (!items.length) return null;
  return (
    <div className="bg-steel-50 border border-steel-200 p-5 space-y-3">
      <h2 className="text-base font-semibold text-oxford">{title}</h2>
      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.id} className="border border-steel-200 bg-white px-4 py-3 space-y-2">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-medium text-oxford">{item.label}</div>
                <div className="text-xs text-steel-500">
                  {item.scope_item_type}
                  {item.priority_tier ? ` · ${item.priority_tier}` : ""}
                </div>
              </div>
              <StatusBadge status={item.status} />
            </div>
            {item.why_it_matters && (
              <p className="text-sm text-steel-700 leading-relaxed">{item.why_it_matters}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ScopeReviewPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: scopeReview, isLoading } = useScopeReview(workspaceId);
  const { data: companyContext } = useCompanyContextPack(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const confirmScopeReview = useConfirmScopeReview(workspaceId);

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
        title="Scope Review"
        subtitle="Review the source-backed and expansion-backed nodes before universe discovery."
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

      {companyContext?.market_map_brief?.source_summary && (
        <div className="bg-oxford text-white border border-oxford-dark p-5">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-steel-400 mb-2">
            <Sparkles className="w-4 h-4" />
            Source Brief
          </div>
          <p className="text-sm text-steel-100 leading-relaxed">
            {companyContext.market_map_brief.source_summary}
          </p>
        </div>
      )}

      {scopeReview && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <ScopeItemSection title="Source Capabilities" items={scopeReview.source_capabilities} />
          <ScopeItemSection title="Source Customer Segments" items={scopeReview.source_customer_segments} />
          <ScopeItemSection title="Adjacent Capabilities" items={scopeReview.adjacent_capabilities} />
          <ScopeItemSection title="Adjacent Customer Segments" items={scopeReview.adjacent_customer_segments} />
          <ScopeItemSection title="Named Account Anchors" items={scopeReview.named_account_anchors} />
          <ScopeItemSection title="Geography Expansions" items={scopeReview.geography_expansions} />
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => confirmScopeReview.mutate()}
          disabled={confirmScopeReview.isPending || !scopeReview}
          className="btn-primary flex items-center gap-2 disabled:opacity-50"
        >
          {confirmScopeReview.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
          Confirm Scope
        </button>
      </div>
    </div>
  );
}
