"use client";

import { useState, useMemo } from "react";
import { useParams } from "next/navigation";
import {
  useVendors,
  useEnrichVendors,
  useVendorDossier,
  useGates,
  useBricks,
} from "@/lib/hooks";
import { Vendor } from "@/lib/api";
import {
  Grid3X3,
  Loader2,
  CheckCircle,
  AlertCircle,
  Zap,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Users,
  Briefcase,
  Link2,
  Building2,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

function VendorDossierPanel({
  workspaceId,
  vendor,
}: {
  workspaceId: number;
  vendor: Vendor;
}) {
  const [expanded, setExpanded] = useState(false);
  const { data: dossier, isLoading } = useVendorDossier(workspaceId, vendor.id);

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="w-full text-left text-sm text-oxford hover:underline font-medium"
      >
        View dossier
      </button>
    );
  }

  if (isLoading) {
    return (
      <div className="py-4 flex items-center justify-center">
        <Loader2 className="w-5 h-5 animate-spin text-oxford" />
      </div>
    );
  }

  if (!dossier) {
    return (
      <div className="py-4 text-sm text-steel-500 text-center">
        No dossier yet. Enrich this vendor to generate.
      </div>
    );
  }

  return (
    <div className="pt-4 space-y-4">
      <button
        onClick={() => setExpanded(false)}
        className="text-sm text-steel-500 hover:text-oxford flex items-center gap-1"
      >
        <ChevronUp className="w-4 h-4" />
        Collapse
      </button>

      {/* Modules */}
      {dossier.dossier_json.modules && dossier.dossier_json.modules.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-oxford mb-2 flex items-center gap-1">
            <Grid3X3 className="w-4 h-4" />
            Modules ({dossier.dossier_json.modules.length})
          </h4>
          <div className="space-y-2">
            {dossier.dossier_json.modules.map((module, i) => (
              <div key={i} className="text-sm p-2 bg-steel-50 border border-steel-200">
                <div className="font-medium text-oxford">{module.name}</div>
                {module.brick_name && (
                  <div className="text-xs text-info">→ {module.brick_name}</div>
                )}
                {module.description && (
                  <div className="text-xs text-steel-500 mt-1">{module.description}</div>
                )}
                {module.evidence_urls?.[0] && (
                  <a
                    href={module.evidence_urls[0]}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-info hover:underline flex items-center gap-1 mt-1"
                  >
                    Source <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Customers */}
      {dossier.dossier_json.customers && dossier.dossier_json.customers.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-oxford mb-2 flex items-center gap-1">
            <Users className="w-4 h-4" />
            Customers ({dossier.dossier_json.customers.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {dossier.dossier_json.customers.map((customer, i) => (
              <a
                key={i}
                href={customer.evidence_url}
                target="_blank"
                rel="noopener noreferrer"
                className="px-2 py-1 text-xs bg-success/10 text-success hover:bg-success/20"
              >
                {customer.name}
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Hiring */}
      {dossier.dossier_json.hiring?.mix_summary && (
        <div>
          <h4 className="text-sm font-medium text-oxford mb-2 flex items-center gap-1">
            <Briefcase className="w-4 h-4" />
            Team Signals
          </h4>
          <div className="text-sm text-steel-600">
            {dossier.dossier_json.hiring.mix_summary.engineering_heavy && (
              <span className="text-success">Engineering-heavy • </span>
            )}
            {dossier.dossier_json.hiring.mix_summary.team_size_estimate && (
              <span>~{dossier.dossier_json.hiring.mix_summary.team_size_estimate} employees</span>
            )}
          </div>
          {dossier.dossier_json.hiring.mix_summary.notes && (
            <div className="text-xs text-steel-500 mt-1">
              {dossier.dossier_json.hiring.mix_summary.notes}
            </div>
          )}
        </div>
      )}

      {/* Integrations */}
      {dossier.dossier_json.integrations && dossier.dossier_json.integrations.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-oxford mb-2 flex items-center gap-1">
            <Link2 className="w-4 h-4" />
            Integrations ({dossier.dossier_json.integrations.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {dossier.dossier_json.integrations.map((integration, i) => (
              <span
                key={i}
                className="px-2 py-1 text-xs bg-info/10 text-info"
              >
                {integration.name}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="text-xs text-steel-400 pt-2 border-t border-steel-100">
        Version {dossier.version} • Generated{" "}
        {new Date(dossier.created_at).toLocaleDateString()}
      </div>
    </div>
  );
}

export default function MapPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: vendors, isLoading } = useVendors(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const { data: taxonomy } = useBricks(workspaceId);
  const enrichVendors = useEnrichVendors(workspaceId);

  const [selectedVendorIds, setSelectedVendorIds] = useState<Set<number>>(new Set());
  const [filterGeo, setFilterGeo] = useState<string | null>(null);
  const [filterVertical, setFilterVertical] = useState<string | null>(null);

  const keptVendors = useMemo(
    () => vendors?.filter((v) => v.status === "kept" || v.status === "enriched") || [],
    [vendors]
  );

  const enrichedCount = keptVendors.filter((v) => v.status === "enriched").length;

  // Get unique geos and verticals for filters
  const uniqueGeos = useMemo(() => {
    const geos = new Set<string>();
    keptVendors.forEach((v) => v.hq_country && geos.add(v.hq_country));
    return Array.from(geos).sort();
  }, [keptVendors]);

  const uniqueVerticals = useMemo(() => {
    const verticals = new Set<string>();
    keptVendors.forEach((v) => v.tags_vertical.forEach((t) => verticals.add(t)));
    return Array.from(verticals).sort();
  }, [keptVendors]);

  const filteredVendors = useMemo(() => {
    return keptVendors.filter((v) => {
      if (filterGeo && v.hq_country !== filterGeo) return false;
      if (filterVertical && !v.tags_vertical.includes(filterVertical)) return false;
      return true;
    });
  }, [keptVendors, filterGeo, filterVertical]);

  const toggleSelect = (id: number) => {
    const newSelected = new Set(selectedVendorIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedVendorIds(newSelected);
  };

  const selectAll = () => {
    const notEnriched = filteredVendors.filter((v) => v.status !== "enriched");
    setSelectedVendorIds(new Set(notEnriched.map((v) => v.id)));
  };

  const handleEnrich = async () => {
    if (selectedVendorIds.size === 0) return;
    await enrichVendors.mutateAsync({
      vendor_ids: Array.from(selectedVendorIds),
      job_types: ["enrich_full"],
    });
    setSelectedVendorIds(new Set());
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
        icon={Grid3X3}
        step={4}
        title="Map & Enrich"
        subtitle="Filter your kept vendors by geography and vertical, then enrich them to build detailed dossiers with modules, customers, integrations, and hiring signals — all backed by evidence."
      />

      {/* Status Banner */}
      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.enrichment
              ? "bg-success/10 border-success"
              : "bg-warning/10 border-warning"
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
                ? `${enrichedCount} vendors enriched — you can view Lenses`
                : gates.missing_items.enrichment?.join(", ") || "Enrich at least 5 vendors to unlock Lenses"}
            </span>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-oxford">Market Map</h2>
          <p className="text-steel-500">
            {keptVendors.length} kept vendors • {enrichedCount} enriched
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={selectAll}
            className="btn-secondary text-sm"
          >
            Select All Unenriched
          </button>
          <button
            onClick={handleEnrich}
            disabled={selectedVendorIds.size === 0 || enrichVendors.isPending}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {enrichVendors.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Enriching...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                Enrich Selected ({selectedVendorIds.size})
              </>
            )}
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4 flex-wrap">
        <div>
          <label className="label">Geography</label>
          <select
            value={filterGeo || ""}
            onChange={(e) => setFilterGeo(e.target.value || null)}
            className="input"
          >
            <option value="">All countries</option>
            {uniqueGeos.map((geo) => (
              <option key={geo} value={geo}>
                {geo}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Vertical</label>
          <select
            value={filterVertical || ""}
            onChange={(e) => setFilterVertical(e.target.value || null)}
            className="input"
          >
            <option value="">All verticals</option>
            {uniqueVerticals.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Vendor Table */}
      {filteredVendors.length === 0 ? (
        <div className="text-center py-16 bg-steel-50 border border-steel-200">
          <Grid3X3 className="w-12 h-12 text-steel-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-oxford mb-2">No vendors to display</h3>
          <p className="text-steel-500">Keep vendors in Universe to see them here</p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredVendors.map((vendor) => (
            <div
              key={vendor.id}
              className={clsx(
                "bg-steel-50 border p-4 transition",
                selectedVendorIds.has(vendor.id)
                  ? "border-oxford ring-2 ring-oxford/20"
                  : "border-steel-200"
              )}
            >
              <div className="flex items-start gap-4">
                {/* Checkbox */}
                <input
                  type="checkbox"
                  checked={selectedVendorIds.has(vendor.id)}
                  onChange={() => toggleSelect(vendor.id)}
                  disabled={vendor.status === "enriched"}
                  className="mt-1"
                />

                {/* Main Info */}
                <div className="flex-1">
                  <div className="flex items-start justify-between">
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
                    <div className="flex items-center gap-2">
                      <span
                        className={clsx(
                          "badge",
                          vendor.status === "enriched"
                            ? "badge-success"
                            : "badge-warning"
                        )}
                      >
                        {vendor.status === "enriched" ? "Enriched" : "Pending"}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 mt-2 text-sm text-steel-500">
                    <span className="flex items-center gap-1">
                      <Building2 className="w-4 h-4" />
                      {vendor.hq_country || "Unknown"}
                    </span>
                    {vendor.tags_vertical.length > 0 && (
                      <div className="flex gap-1">
                        {vendor.tags_vertical.slice(0, 2).map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-0.5 text-xs bg-steel-100 text-steel-600"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                    <span>{vendor.evidence_count} citations</span>
                  </div>

                  {/* Dossier Panel */}
                  {vendor.status === "enriched" && (
                    <div className="mt-4 pt-4 border-t border-steel-100">
                      <VendorDossierPanel workspaceId={workspaceId} vendor={vendor} />
                    </div>
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
