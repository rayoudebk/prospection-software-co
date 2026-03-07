"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import {
  useSimilarityLens,
  useComplementarityLens,
  useGates,
  useBricks,
} from "@/lib/hooks";
import { LensVendor } from "@/lib/api";
import {
  Eye,
  Loader2,
  AlertCircle,
  ExternalLink,
  Layers,
  Plus,
  Users,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

function VendorLensCard({
  vendor,
  mode,
}: {
  vendor: LensVendor;
  mode: "similarity" | "complementarity";
}) {
  const [showProof, setShowProof] = useState(false);

  const capabilities = mode === "similarity" ? vendor.overlapping_bricks : vendor.added_bricks;

  return (
    <div className="bg-steel-50 border border-steel-200 p-5">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-oxford">{vendor.name}</h3>
          {vendor.website && (
            <a
              href={vendor.website}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-info hover:underline flex items-center gap-1"
            >
              {new URL(vendor.website).hostname}
              <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
        <div className="text-sm text-steel-500">{vendor.evidence_count} citations</div>
      </div>

      {/* Capability overlap */}
      {capabilities.length > 0 && (
        <div className="mb-4">
          <div className="text-xs text-steel-500 mb-2 flex items-center gap-1">
            {mode === "similarity" ? (
              <>
                <Layers className="w-3 h-3" />
                Overlapping Capabilities
              </>
            ) : (
              <>
                <Plus className="w-3 h-3" />
                Adds Capabilities
              </>
            )}
          </div>
          <div className="flex flex-wrap gap-1">
            {capabilities.map((capability, i) => (
              <span
                key={i}
                className={clsx(
                  "px-2 py-1 text-xs",
                  mode === "similarity"
                    ? "bg-info/10 text-info"
                    : "bg-success/10 text-success"
                )}
              >
                {capability}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Customer Overlaps */}
      {vendor.customer_overlaps.length > 0 && (
        <div className="mb-4">
          <div className="text-xs text-steel-500 mb-2 flex items-center gap-1">
            <Users className="w-3 h-3" />
            Shared Customers
          </div>
          <div className="flex flex-wrap gap-1">
            {vendor.customer_overlaps.map((customer, i) => (
              <span
                key={i}
                className="px-2 py-1 text-xs bg-oxford/10 text-oxford"
              >
                {customer}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Proof Bullets */}
      {vendor.proof_bullets.length > 0 && (
        <div>
          <button
            onClick={() => setShowProof(!showProof)}
            className="text-xs text-oxford hover:underline font-medium"
          >
            {showProof ? "Hide" : "Show"} proof ({vendor.proof_bullets.length})
          </button>
          {showProof && (
            <div className="mt-2 space-y-2">
              {vendor.proof_bullets.map((proof, i) => (
                <div
                  key={i}
                  className="text-sm text-steel-600 pl-3 border-l-2 border-oxford"
                >
                  {proof.text}
                  {proof.citation_url && (
                    <a
                      href={proof.citation_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="ml-2 text-info hover:underline inline-flex items-center gap-0.5"
                    >
                      source
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function LensesPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: gates } = useGates(workspaceId);
  const { data: taxonomy } = useBricks(workspaceId);
  const [activeTab, setActiveTab] = useState<"similarity" | "complementarity">("similarity");
  const [selectedBrickIds, setSelectedBrickIds] = useState<string | undefined>(undefined);

  const { data: similarityData, isLoading: loadingSimilarity } = useSimilarityLens(
    workspaceId,
    selectedBrickIds
  );
  const { data: complementarityData, isLoading: loadingComplementarity } =
    useComplementarityLens(workspaceId);

  const isLoading = activeTab === "similarity" ? loadingSimilarity : loadingComplementarity;
  const data = activeTab === "similarity" ? similarityData : complementarityData;

  const priorityBricks = taxonomy?.bricks.filter((b) =>
    taxonomy.priority_brick_ids.includes(b.id)
  );

  if (!gates?.enrichment) {
    return (
      <div className="text-center py-16 bg-white border border-steel-200">
        <AlertCircle className="w-12 h-12 text-warning mx-auto mb-4" />
        <h3 className="text-lg font-medium text-oxford mb-2">Lenses Locked</h3>
        <p className="text-steel-500">
          Enrich at least 5 companies in Map & Enrich to unlock Lenses
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <StepHeader
        icon={Eye}
        step={5}
        title="Lenses"
        subtitle="View your enriched companies through two strategic lenses: Similarity highlights close adjacency around your current capabilities. Complementarity highlights capability expansion paths."
      />

      {/* Tab Toggle */}
      <div className="flex gap-1 bg-steel-50 border border-steel-200 p-1 inline-flex">
        <button
          onClick={() => setActiveTab("similarity")}
          className={clsx(
            "px-4 py-2 font-medium transition",
            activeTab === "similarity"
              ? "bg-oxford text-white"
              : "text-steel-600 hover:bg-steel-100"
          )}
        >
          Similarity Lens
        </button>
        <button
          onClick={() => setActiveTab("complementarity")}
          className={clsx(
            "px-4 py-2 font-medium transition",
            activeTab === "complementarity"
              ? "bg-success text-white"
              : "text-steel-600 hover:bg-steel-100"
          )}
        >
          Complementarity Lens
        </button>
      </div>

      {/* Description */}
      <div
        className={clsx(
          "p-4 border",
          activeTab === "similarity"
            ? "bg-info/10 border-info"
            : "bg-success/10 border-success"
        )}
      >
        <p className={activeTab === "similarity" ? "text-info font-medium" : "text-success font-medium"}>
          {activeTab === "similarity"
            ? "Companies with overlapping capabilities on your priority lane. Useful for close adjacency and bolt-on sourcing."
            : "Companies adding capabilities you do not currently cover. Useful for expansion sourcing."}
        </p>
      </div>

      {/* Capability filter (similarity only) */}
      {activeTab === "similarity" && priorityBricks && priorityBricks.length > 0 && (
        <div>
          <label className="label">
            Filter by Priority Capabilities
          </label>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedBrickIds(undefined)}
              className={clsx(
                "px-3 py-1.5 text-sm border transition",
                !selectedBrickIds
                  ? "bg-oxford border-oxford text-white"
                  : "bg-steel-50 border-steel-300 text-steel-600 hover:border-oxford"
              )}
            >
              All Priority Capabilities
            </button>
            {priorityBricks.map((brick) => (
              <button
                key={brick.id}
                onClick={() => setSelectedBrickIds(brick.id)}
                className={clsx(
                  "px-3 py-1.5 text-sm border transition",
                  selectedBrickIds === brick.id
                    ? "bg-oxford border-oxford text-white"
                    : "bg-steel-50 border-steel-300 text-steel-600 hover:border-oxford"
                )}
              >
                {brick.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-oxford" />
        </div>
      ) : data && data.vendors.length > 0 ? (
        <div>
          <p className="text-sm text-steel-500 mb-4">
            {data.total_count} companies
            {activeTab === "similarity" ? " with overlapping capabilities" : " adding new capabilities"}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.vendors.map((vendor) => (
              <VendorLensCard
                key={vendor.id}
                vendor={vendor}
                mode={activeTab}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="text-center py-16 bg-steel-50 border border-steel-200">
          <Eye className="w-12 h-12 text-steel-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-oxford mb-2">No Results</h3>
          <p className="text-steel-500">
            {activeTab === "similarity"
              ? "No companies found with overlapping capabilities. Try different filters."
              : "No companies found that add new capabilities."}
          </p>
        </div>
      )}
    </div>
  );
}
