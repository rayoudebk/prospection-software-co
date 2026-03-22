"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useParams } from "next/navigation";
import clsx from "clsx";
import { AlertCircle, ArrowRight, CheckCircle, ExternalLink, Loader2 } from "lucide-react";

import { StepHeader } from "@/components/StepHeader";
import { ValidationQueueItem } from "@/lib/api";
import { useGates, useRefreshValidationQueue, useUpdateValidationCandidate, useValidationQueue } from "@/lib/hooks";

const STATUS_LABELS: Record<string, string> = {
  queued_for_validation: "Queued",
  validated_keep: "Keep",
  validated_watchlist: "Watchlist",
  validated_reject: "Reject",
  removed_from_validation: "Removed",
  promoted_to_cards: "Promoted",
};

function hostLabel(url?: string | null) {
  if (!url) return null;
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

function queueGroupLabel(item: ValidationQueueItem) {
  return item.validation_lane_labels[0] || item.validation_lane_ids[0] || "Unscoped";
}

function QueueCard({
  item,
  onStatus,
  pending,
}: {
  item: ValidationQueueItem;
  onStatus: (status: string) => Promise<void>;
  pending: boolean;
}) {
  const actions = [
    { status: "validated_keep", label: "Keep", tone: "success" },
    { status: "validated_watchlist", label: "Watchlist", tone: "warning" },
    { status: "validated_reject", label: "Reject", tone: "danger" },
    { status: "promoted_to_cards", label: "Promote to Cards", tone: "primary" },
  ] as const;

  return (
    <article className="rounded-3xl border border-steel-200 bg-white px-5 py-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-lg text-oxford">{item.company_name}</h3>
            <span className="rounded-full border border-steel-200 bg-steel-50 px-2 py-0.5 text-xs text-steel-500">
              {STATUS_LABELS[item.validation_status] || item.validation_status}
            </span>
            {item.page_classification ? (
              <span className="rounded-full border border-info/20 bg-info/10 px-2 py-0.5 text-xs text-info">
                {item.page_classification.replaceAll("_", " ")}
              </span>
            ) : null}
          </div>
          <div className="text-sm text-steel-500">
            Rank #{item.validation_queue_rank} • {item.hq_country || "Unknown country"} • {item.entity_type} •{" "}
            {item.directness.replaceAll("_", " ")}
          </div>
          {item.official_website_url ? (
            <a
              href={item.official_website_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-info hover:underline"
            >
              {hostLabel(item.official_website_url)}
              <ExternalLink className="h-3 w-3" />
            </a>
          ) : null}
        </div>
        <div className="text-right text-xs text-steel-500 space-y-1">
          <div>Identity {item.identity_confidence || "unknown"}</div>
          <div>Website {item.official_website_confidence || "unknown"}</div>
          <div>{item.vendor_classification || "unclassified"}</div>
          <div>Validation {Math.round(item.validation_score || 0)}</div>
          {item.identity_diagnostics?.has_first_party_evidence ? (
            <div>Homepage signals {item.identity_diagnostics.signals_extracted || 0}</div>
          ) : null}
        </div>
      </div>

      <p className="text-sm leading-6 text-steel-600">
        {item.validated_description || item.short_description || "No validated description yet."}
      </p>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-2">
          <div className="text-[11px] uppercase tracking-[0.18em] text-steel-400">Lanes</div>
          <div className="flex flex-wrap gap-2">
            {(item.validation_lane_labels.length ? item.validation_lane_labels : item.validation_lane_ids).map((lane) => (
              <span key={lane} className="rounded-full border border-info/20 bg-info/10 px-2.5 py-1 text-xs text-info">
                {lane}
              </span>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          <div className="text-[11px] uppercase tracking-[0.18em] text-steel-400">Query Families</div>
          <div className="flex flex-wrap gap-2">
            {item.validation_query_families.slice(0, 4).map((family) => (
              <span key={family} className="rounded-full border border-steel-200 bg-steel-50 px-2.5 py-1 text-xs text-steel-600">
                {family}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-[11px] uppercase tracking-[0.18em] text-steel-400">Provenance</div>
        <div className="text-sm text-steel-600">
          {item.multi_origin_count} origin types •{" "}
          {String(item.provenance_summary?.source_family_count || item.validation_source_families.length)} source
          families • {String(item.provenance_summary?.query_family_count || item.validation_query_families.length)} query
          families
        </div>
      </div>

      <div className="flex flex-wrap justify-between gap-3 border-t border-steel-100 pt-4">
        <div className="text-xs text-steel-400">
          {item.discovery_sources.length} discovery links • {item.multi_origin_count} origin types
        </div>
        <div className="flex flex-wrap gap-2">
          {actions.map((action) => {
            const selected = item.validation_status === action.status;
            return (
              <button
                key={action.status}
                onClick={() => onStatus(action.status)}
                disabled={pending || selected}
                className={clsx(
                  "rounded-full border px-3 py-1.5 text-sm transition-colors disabled:opacity-50",
                  selected
                    ? "border-oxford bg-oxford text-white"
                    : action.tone === "danger"
                      ? "border-steel-200 text-steel-700 hover:border-danger/30 hover:bg-danger/10 hover:text-danger"
                      : action.tone === "success"
                        ? "border-steel-200 text-steel-700 hover:border-success/30 hover:bg-success/10 hover:text-success"
                        : action.tone === "warning"
                          ? "border-steel-200 text-steel-700 hover:border-warning/30 hover:bg-warning/10 hover:text-warning"
                          : "border-steel-200 text-steel-700 hover:border-info/30 hover:bg-info/10 hover:text-info"
                )}
              >
                {action.label}
              </button>
            );
          })}
        </div>
      </div>
    </article>
  );
}

export default function ValidationPage() {
  const params = useParams();
  const workspaceId = Number(params.id);
  const { data: gates } = useGates(workspaceId);
  const { data: queue, isLoading } = useValidationQueue(workspaceId, 48, false, false);
  const updateValidation = useUpdateValidationCandidate(workspaceId);
  const refreshValidation = useRefreshValidationQueue(workspaceId);

  const grouped = useMemo(() => {
    const map = new Map<string, ValidationQueueItem[]>();
    for (const item of queue || []) {
      const key = queueGroupLabel(item);
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(item);
    }
    return Array.from(map.entries());
  }, [queue]);

  const promotedCount = (queue || []).filter((item) => item.promoted_to_cards).length;
  const keepCount = (queue || []).filter((item) => item.validation_status === "validated_keep").length;
  const watchlistCount = (queue || []).filter((item) => item.validation_status === "validated_watchlist").length;

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
        step={4}
        title="Validation"
        subtitle="Review the diversified longlist, qualify vendor identity, and explicitly promote only the shortlist into Cards."
      />

      {gates && (
        <div
          className={clsx(
            "rounded-3xl border px-5 py-4",
            gates.cards ? "border-success/30 bg-success/10" : "border-warning/30 bg-warning/10"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.cards ? (
              <CheckCircle className="h-5 w-5 text-success" />
            ) : (
              <AlertCircle className="h-5 w-5 text-warning" />
            )}
            <span className={gates.cards ? "font-medium text-success" : "font-medium text-warning-dark"}>
              {gates.cards
                ? "Shortlist promoted — Cards can now be generated from the validated set."
                : gates.missing_items.cards?.join(", ") || "Promote at least one validated company into Cards."}
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <div className="rounded-3xl border border-steel-200 bg-white p-5">
          <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Queued</div>
          <div className="mt-2 text-3xl font-semibold text-oxford">{queue?.length || 0}</div>
        </div>
        <div className="rounded-3xl border border-steel-200 bg-white p-5">
          <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Keep</div>
          <div className="mt-2 text-3xl font-semibold text-oxford">{keepCount}</div>
        </div>
        <div className="rounded-3xl border border-steel-200 bg-white p-5">
          <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Watchlist</div>
          <div className="mt-2 text-3xl font-semibold text-oxford">{watchlistCount}</div>
        </div>
        <div className="rounded-3xl border border-steel-200 bg-white p-5">
          <div className="text-xs uppercase tracking-[0.18em] text-steel-400">Promoted to Cards</div>
          <div className="mt-2 text-3xl font-semibold text-oxford">{promotedCount}</div>
        </div>
      </div>

      <div className="rounded-3xl border border-steel-200 bg-white px-5 py-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-oxford">Validation Queue</h2>
            <p className="mt-1 text-sm text-steel-500">
              This queue is sourced from Universe ranking and is the first place where light enrichment and shortlist
              decisions happen.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => refreshValidation.mutate({ topN: 12 })}
              disabled={refreshValidation.isPending}
              className="btn-secondary"
            >
              {refreshValidation.isPending ? "Refreshing..." : "Refresh Top Slice"}
            </button>
            <Link href={`/workspaces/${workspaceId}/report`} className="btn-secondary flex items-center gap-2">
              Cards
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>

      {!queue?.length ? (
        <div className="rounded-3xl border border-steel-200 bg-white px-6 py-12 text-center text-sm text-steel-500">
          No validation queue yet. Run Universe first so the ranking layer can produce candidate slices.
        </div>
      ) : (
        <div className="space-y-8">
          {grouped.map(([group, items]) => (
            <section key={group} className="space-y-4">
              <div>
                <div className="text-[11px] uppercase tracking-[0.22em] text-steel-400">Lane</div>
                <h3 className="mt-2 text-xl font-semibold text-oxford">{group}</h3>
              </div>
              <div className="grid gap-4 xl:grid-cols-2">
                {items.map((item) => (
                  <QueueCard
                    key={item.candidate_entity_id}
                    item={item}
                    pending={updateValidation.isPending}
                    onStatus={(status: string) =>
                      updateValidation.mutateAsync({
                        candidateEntityId: item.candidate_entity_id,
                        status,
                      }).then(() => undefined)
                    }
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
