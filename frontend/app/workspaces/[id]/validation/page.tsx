"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { ClipboardCheck, CheckCircle, AlertCircle, ArrowRight } from "lucide-react";
import clsx from "clsx";
import { StepHeader } from "@/components/StepHeader";
import { useCompanies, useGates } from "@/lib/hooks";

function pickRecommendedCompanies(
  companies: Array<{
    id: number;
    name: string;
    decision_classification?: string | null;
    evidence_sufficiency?: string | null;
    status: string;
  }>
) {
  return companies
    .filter((company) => {
      const classification = company.decision_classification || "insufficient_evidence";
      if (classification === "good_target") return true;
      if (classification === "borderline_watchlist") return true;
      return classification === "insufficient_evidence" && company.status !== "removed";
    })
    .slice(0, 12);
}

export default function ValidationPage() {
  const params = useParams();
  const workspaceId = Number(params.id);
  const { data: companies } = useCompanies(workspaceId);
  const { data: gates } = useGates(workspaceId);

  const activeCompanies = (companies || []).filter((company) => company.status !== "removed");
  const recommendedCompanies = pickRecommendedCompanies(activeCompanies);
  const goodCount = activeCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "good_target").length;
  const watchlistCount = activeCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "borderline_watchlist").length;
  const insufficientCount = activeCompanies.filter((company) => (company.decision_classification || "insufficient_evidence") === "insufficient_evidence").length;
  const enrichedCount = activeCompanies.filter((company) => company.status === "enriched").length;

  return (
    <div className="space-y-6">
      <StepHeader
        icon={ClipboardCheck}
        step={4}
        title="Validation"
        subtitle="Use the universe cards as a 15-20 second scan. Then manually choose which companies deserve simple enrichment before sending only the shortlist into deep cards."
      />

      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.enrichment ? "bg-success/10 border-success" : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.enrichment ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.enrichment ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.enrichment
                ? "Enough companies are enriched — you can proceed to Cards"
                : gates.missing_items.enrichment?.join(", ") || "Manually select companies for simple enrichment before Cards"}
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-white border border-steel-200 p-5">
          <div className="text-xs uppercase tracking-wider text-steel-400 mb-2">Recommended For Simple Enrichment</div>
          <div className="text-3xl font-semibold text-oxford">{recommendedCompanies.length}</div>
          <p className="text-sm text-steel-500 mt-2">
            Good-target, watchlist, and selected insufficient-evidence names should be reviewed here before any deep card work.
          </p>
        </div>
        <div className="bg-white border border-steel-200 p-5">
          <div className="text-xs uppercase tracking-wider text-steel-400 mb-2">Universe Mix</div>
          <div className="space-y-2 text-sm text-steel-600">
            <div>Good target: <span className="font-medium text-oxford">{goodCount}</span></div>
            <div>Watchlist: <span className="font-medium text-oxford">{watchlistCount}</span></div>
            <div>Insufficient evidence: <span className="font-medium text-oxford">{insufficientCount}</span></div>
          </div>
        </div>
        <div className="bg-white border border-steel-200 p-5">
          <div className="text-xs uppercase tracking-wider text-steel-400 mb-2">Deep-Card Ready</div>
          <div className="text-3xl font-semibold text-oxford">{enrichedCount}</div>
          <p className="text-sm text-steel-500 mt-2">
            Only fully enriched companies should move into Cards.
          </p>
        </div>
      </div>

      <div className="bg-white border border-steel-200 p-5 space-y-4">
        <div>
          <h2 className="text-xl font-semibold text-oxford">Manual Validation Sequence</h2>
          <p className="text-steel-500 mt-1">
            Keep the basic scan in Universe. Use this step to decide which names deserve simple enrichment, then promote only the best-documented shortlist into Cards.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border border-steel-200 bg-steel-50 p-4">
            <div className="text-sm font-medium text-oxford mb-2">1. Scan Universe</div>
            <p className="text-sm text-steel-600">
              Review what they do, who buys it, and the rough deal shape in 15-20 seconds.
            </p>
          </div>
          <div className="border border-steel-200 bg-steel-50 p-4">
            <div className="text-sm font-medium text-oxford mb-2">2. Send To Simple Enrichment</div>
            <p className="text-sm text-steel-600">
              Manually advance only the companies where a little more evidence could change the decision.
            </p>
          </div>
          <div className="border border-steel-200 bg-steel-50 p-4">
            <div className="text-sm font-medium text-oxford mb-2">3. Promote Shortlist To Cards</div>
            <p className="text-sm text-steel-600">
              Deep cards should be reserved for the shortlist, not the full longlist.
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white border border-steel-200 p-5">
        <div className="flex items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-xl font-semibold text-oxford">Suggested Review Queue</h2>
            <p className="text-steel-500 mt-1">
              Start with these companies when deciding who to send to simple enrichment.
            </p>
          </div>
          <Link href={`/workspaces/${workspaceId}/report`} className="btn-secondary flex items-center gap-2">
            Cards
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        {recommendedCompanies.length === 0 ? (
          <div className="text-sm text-steel-500">
            No validation queue yet. Return to Universe and keep or review promising companies first.
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {recommendedCompanies.map((company) => (
              <span
                key={company.id}
                className={clsx(
                  "px-3 py-1.5 text-sm border",
                  company.decision_classification === "good_target"
                    ? "bg-success/10 text-success border-success/30"
                    : company.decision_classification === "borderline_watchlist"
                    ? "bg-warning/10 text-warning border-warning/30"
                    : "bg-steel-50 text-steel-700 border-steel-200"
                )}
              >
                {company.name}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
