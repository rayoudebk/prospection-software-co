"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  useConfirmSearchLanes,
  useGates,
  useSearchLanes,
  useThesisPack,
  useUpdateSearchLanes,
} from "@/lib/hooks";
import { SearchLane } from "@/lib/api";
import {
  AlertCircle,
  Check,
  CheckCircle,
  Layers,
  Loader2,
  Save,
  Sparkles,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

function listToTextarea(values: string[]) {
  return (values || []).join("\n");
}

function textareaToList(value: string) {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function LaneEditor({
  lane,
  onChange,
}: {
  lane: SearchLane;
  onChange: (patch: Partial<SearchLane>) => void;
}) {
  return (
    <div className="bg-steel-50 border border-steel-200 p-6 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-steel-500 mb-1">{lane.lane_type} lane</div>
          <h2 className="text-lg font-semibold text-oxford">{lane.title}</h2>
        </div>
        <span
          className={clsx(
            "px-2 py-1 text-xs border",
            lane.status === "confirmed"
              ? "border-success/40 bg-success/10 text-success"
              : "border-warning/40 bg-warning/10 text-warning"
          )}
        >
          {lane.status}
        </span>
      </div>

      <div>
        <label className="label">Title</label>
        <input
          value={lane.title}
          onChange={(event) => onChange({ title: event.target.value })}
          className="input"
        />
      </div>

      <div>
        <label className="label">Intent</label>
        <textarea
          value={lane.intent || ""}
          onChange={(event) => onChange({ intent: event.target.value })}
          className="w-full min-h-[96px] border border-steel-300 px-3 py-2 text-sm bg-white"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="label">Capabilities</label>
          <textarea
            value={listToTextarea(lane.capabilities)}
            onChange={(event) => onChange({ capabilities: textareaToList(event.target.value) })}
            className="w-full min-h-[120px] border border-steel-300 px-3 py-2 text-sm bg-white"
            placeholder="One per line"
          />
        </div>
        <div>
          <label className="label">Customer Tags</label>
          <textarea
            value={listToTextarea(lane.customer_tags)}
            onChange={(event) => onChange({ customer_tags: textareaToList(event.target.value) })}
            className="w-full min-h-[120px] border border-steel-300 px-3 py-2 text-sm bg-white"
            placeholder="One per line"
          />
        </div>
        <div>
          <label className="label">Must-Include Terms</label>
          <textarea
            value={listToTextarea(lane.must_include_terms)}
            onChange={(event) => onChange({ must_include_terms: textareaToList(event.target.value) })}
            className="w-full min-h-[120px] border border-steel-300 px-3 py-2 text-sm bg-white"
            placeholder="One per line"
          />
        </div>
        <div>
          <label className="label">Must-Exclude Terms</label>
          <textarea
            value={listToTextarea(lane.must_exclude_terms)}
            onChange={(event) => onChange({ must_exclude_terms: textareaToList(event.target.value) })}
            className="w-full min-h-[120px] border border-steel-300 px-3 py-2 text-sm bg-white"
            placeholder="One per line"
          />
        </div>
      </div>

      <div>
        <label className="label">Seed URLs</label>
        <textarea
          value={listToTextarea(lane.seed_urls)}
          onChange={(event) => onChange({ seed_urls: textareaToList(event.target.value) })}
          className="w-full min-h-[120px] border border-steel-300 px-3 py-2 text-sm bg-white"
          placeholder="High-confidence comparator URLs, one per line"
        />
      </div>
    </div>
  );
}

export default function SearchLanesPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: searchLanes, isLoading } = useSearchLanes(workspaceId);
  const { data: thesisPack } = useThesisPack(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const updateSearchLanes = useUpdateSearchLanes(workspaceId);
  const confirmSearchLanes = useConfirmSearchLanes(workspaceId);

  const [lanes, setLanes] = useState<SearchLane[]>([]);

  useEffect(() => {
    if (searchLanes?.lanes) {
      setLanes(
        [...searchLanes.lanes].sort((left, right) =>
          left.lane_type === right.lane_type ? 0 : left.lane_type === "core" ? -1 : 1
        )
      );
    }
  }, [searchLanes]);

  const updateLane = (laneType: string, patch: Partial<SearchLane>) => {
    setLanes((current) =>
      current.map((lane) => (lane.lane_type === laneType ? { ...lane, ...patch } : lane))
    );
  };

  const handleSave = async () => {
    await updateSearchLanes.mutateAsync({ lanes });
  };

  const handleConfirm = async () => {
    await updateSearchLanes.mutateAsync({ lanes });
    await confirmSearchLanes.mutateAsync();
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
        icon={Layers}
        step={2}
        title="Search Lanes"
        subtitle="Review the suggested sourcing lanes derived from the brief. Core captures direct-fit companies. Adjacent captures neighboring workflows, extensions, and capability edges worth sourcing separately."
      />

      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.search_lanes ? "bg-success/10 border-success" : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.search_lanes ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.search_lanes ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.search_lanes
                ? "Search lanes confirmed — you can proceed to Universe"
                : gates.missing_items.search_lanes?.join(", ") || "Confirm search lanes to continue"}
            </span>
          </div>
        </div>
      )}

      {thesisPack?.summary && (
        <div className="bg-oxford text-white border border-oxford-dark p-5">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-steel-400 mb-2">
            <Sparkles className="w-4 h-4" />
            Derived From Sourcing Brief
          </div>
          <p className="text-sm text-steel-100 leading-relaxed">{thesisPack.summary}</p>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {lanes.map((lane) => (
          <LaneEditor
            key={lane.lane_type}
            lane={lane}
            onChange={(patch) => updateLane(lane.lane_type, patch)}
          />
        ))}
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleSave}
          disabled={updateSearchLanes.isPending}
          className="btn-secondary flex items-center gap-2 disabled:opacity-50"
        >
          {updateSearchLanes.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          Save Lanes
        </button>
        <button
          onClick={handleConfirm}
          disabled={confirmSearchLanes.isPending || lanes.length === 0}
          className="btn-primary flex items-center gap-2 disabled:opacity-50"
        >
          {confirmSearchLanes.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
          Confirm Lanes
        </button>
      </div>
    </div>
  );
}
