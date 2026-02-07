"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import {
  useVendors,
  useRunDiscovery,
  useUpdateVendor,
  useCreateVendor,
  useGates,
  useWorkspaceJobWithPolling,
} from "@/lib/hooks";
import { workspaceApi, Vendor } from "@/lib/api";
import {
  Globe,
  Search,
  Plus,
  Check,
  X,
  ExternalLink,
  Loader2,
  Filter,
  CheckCircle,
  AlertCircle,
  Building2,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

export default function UniversePage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: vendors, isLoading, refetch } = useVendors(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const updateVendor = useUpdateVendor(workspaceId);
  const createVendor = useCreateVendor(workspaceId);

  const [filter, setFilter] = useState<"all" | "candidate" | "kept" | "removed">("all");
  const [showAddModal, setShowAddModal] = useState(false);
  const [newVendor, setNewVendor] = useState({ name: "", website: "", hq_country: "" });

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.runDiscovery(workspaceId),
    () => refetch()
  );

  const handleKeep = async (vendor: Vendor) => {
    await updateVendor.mutateAsync({
      vendorId: vendor.id,
      data: { status: "kept" },
    });
  };

  const handleRemove = async (vendor: Vendor) => {
    await updateVendor.mutateAsync({
      vendorId: vendor.id,
      data: { status: "removed" },
    });
  };

  const handleRestore = async (vendor: Vendor) => {
    await updateVendor.mutateAsync({
      vendorId: vendor.id,
      data: { status: "candidate" },
    });
  };

  const handleAddVendor = async () => {
    if (!newVendor.name) return;
    await createVendor.mutateAsync(newVendor);
    setNewVendor({ name: "", website: "", hq_country: "" });
    setShowAddModal(false);
  };

  const filteredVendors = vendors?.filter((v) => {
    if (filter === "all") return true;
    return v.status === filter;
  });

  const keptCount = vendors?.filter((v) => v.status === "kept" || v.status === "enriched").length || 0;
  const candidateCount = vendors?.filter((v) => v.status === "candidate").length || 0;
  const removedCount = vendors?.filter((v) => v.status === "removed").length || 0;

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
        icon={Globe}
        step={3}
        title="Universe"
        subtitle="Build a curated longlist of potential acquisition targets matching your brick model. Keep strong fits, remove weak fits, and prepare for static report generation."
      />

      {/* Status Banner */}
      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.universe
              ? "bg-success/10 border-success"
              : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.universe ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.universe ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.universe
                ? `${keptCount} vendors kept — you can proceed to Report`
                : gates.missing_items.universe?.join(", ") || "Keep at least 5 vendors to continue"}
            </span>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-oxford">Candidate Universe</h2>
          <p className="text-steel-500">
            {vendors?.length || 0} vendors discovered • {keptCount} kept • {removedCount} removed
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-secondary flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Vendor
          </button>
          <button
            onClick={jobRunner.run}
            disabled={jobRunner.isRunning}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {jobRunner.isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Discovering... {Math.round(jobRunner.progress * 100)}%
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Run Discovery
              </>
            )}
          </button>
        </div>
      </div>

      {/* Add Vendor Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-oxford/80 flex items-center justify-center z-50">
          <div className="bg-steel-50 p-6 w-full max-w-md shadow-xl border border-steel-200">
            <h3 className="text-lg font-semibold text-oxford mb-4">Add Vendor Manually</h3>
            <div className="space-y-4">
              <div>
                <label className="label">
                  Company Name *
                </label>
                <input
                  type="text"
                  value={newVendor.name}
                  onChange={(e) => setNewVendor({ ...newVendor, name: e.target.value })}
                  className="input"
                />
              </div>
              <div>
                <label className="label">
                  Website
                </label>
                <input
                  type="url"
                  value={newVendor.website}
                  onChange={(e) => setNewVendor({ ...newVendor, website: e.target.value })}
                  placeholder="https://..."
                  className="input"
                />
              </div>
              <div>
                <label className="label">
                  HQ Country
                </label>
                <input
                  type="text"
                  value={newVendor.hq_country}
                  onChange={(e) => setNewVendor({ ...newVendor, hq_country: e.target.value })}
                  placeholder="e.g., UK"
                  className="input"
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAddModal(false)}
                className="flex-1 btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleAddVendor}
                disabled={!newVendor.name || createVendor.isPending}
                className="flex-1 btn-primary disabled:opacity-50"
              >
                Add Vendor
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-1 bg-steel-50 border border-steel-200 p-1 inline-flex">
        {(["all", "candidate", "kept", "removed"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={clsx(
              "px-4 py-2 text-sm font-medium transition",
              filter === f
                ? "bg-oxford text-white"
                : "text-steel-600 hover:bg-steel-100"
            )}
          >
            {f === "all" && `All (${vendors?.length || 0})`}
            {f === "candidate" && `To Review (${candidateCount})`}
            {f === "kept" && `Kept (${keptCount})`}
            {f === "removed" && `Removed (${removedCount})`}
          </button>
        ))}
      </div>

      {/* Vendor Grid */}
      {filteredVendors && filteredVendors.length === 0 ? (
        <div className="text-center py-16 bg-steel-50 border border-steel-200">
          <Globe className="w-12 h-12 text-steel-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-oxford mb-2">No vendors found</h3>
          <p className="text-steel-500 mb-4">
            Run discovery to find potential acquisition targets
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredVendors?.map((vendor) => (
            <div
              key={vendor.id}
              className={clsx(
                "bg-steel-50 border p-4 transition",
                vendor.status === "removed"
                  ? "border-steel-200 opacity-60"
                  : vendor.status === "kept" || vendor.status === "enriched"
                  ? "border-success"
                  : "border-steel-200 hover:border-oxford"
              )}
            >
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
                <span
                  className={clsx(
                    "badge",
                    vendor.status === "kept" || vendor.status === "enriched"
                      ? "badge-success"
                      : vendor.status === "removed"
                      ? "badge-neutral"
                      : "badge-warning"
                  )}
                >
                  {vendor.status}
                </span>
              </div>

              <div className="flex items-center gap-2 text-sm text-steel-500 mb-3">
                <Building2 className="w-4 h-4" />
                {vendor.hq_country || "Unknown"}
              </div>

              {vendor.tags_vertical.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {vendor.tags_vertical.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 text-xs bg-steel-100 text-steel-600"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {vendor.why_relevant.length > 0 && (
                <div className="text-sm text-steel-600 mb-3 line-clamp-2">
                  {vendor.why_relevant[0]?.text}
                </div>
              )}

              <div className="flex items-center justify-between pt-3 border-t border-steel-100">
                <span className="text-xs text-steel-400">
                  {vendor.evidence_count} citations
                </span>
                <div className="flex gap-2">
                  {vendor.status === "removed" ? (
                    <button
                      onClick={() => handleRestore(vendor)}
                      className="btn-secondary text-sm py-1 px-3"
                    >
                      Restore
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={() => handleRemove(vendor)}
                        className="p-1.5 text-steel-400 hover:text-danger hover:bg-danger/10 transition"
                      >
                        <X className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleKeep(vendor)}
                        disabled={vendor.status === "kept" || vendor.status === "enriched"}
                        className={clsx(
                          "p-1.5 transition",
                          vendor.status === "kept" || vendor.status === "enriched"
                            ? "text-success bg-success/10"
                            : "text-steel-400 hover:text-success hover:bg-success/10"
                        )}
                      >
                        <Check className="w-4 h-4" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
