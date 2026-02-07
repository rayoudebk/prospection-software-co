"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import {
  useBricks,
  useUpdateBricks,
  useConfirmBricks,
  useGates,
} from "@/lib/hooks";
import { BrickItem } from "@/lib/api";
import {
  Layers,
  Plus,
  Trash2,
  Star,
  Check,
  Loader2,
  Edit2,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import clsx from "clsx";

const VERTICALS = [
  { id: "asset_manager", label: "Asset Managers" },
  { id: "wealth_manager", label: "Wealth Managers" },
  { id: "bank", label: "Banks" },
  { id: "insurer", label: "Insurers" },
  { id: "hedge_fund", label: "Hedge Funds" },
  { id: "family_office", label: "Family Offices" },
  { id: "pension_fund", label: "Pension Funds" },
  { id: "fund_admin", label: "Fund Administrators" },
];

export default function BricksPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: taxonomy, isLoading } = useBricks(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const updateBricks = useUpdateBricks(workspaceId);
  const confirmBricks = useConfirmBricks(workspaceId);

  const [bricks, setBricks] = useState<BrickItem[]>([]);
  const [priorityIds, setPriorityIds] = useState<string[]>([]);
  const [selectedVerticals, setSelectedVerticals] = useState<string[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [newBrickName, setNewBrickName] = useState("");

  useEffect(() => {
    if (taxonomy) {
      setBricks(taxonomy.bricks);
      setPriorityIds(taxonomy.priority_brick_ids);
      setSelectedVerticals(taxonomy.vertical_focus || []);
    }
  }, [taxonomy]);

  const handleRename = (id: string, newName: string) => {
    setBricks(
      bricks.map((b) => (b.id === id ? { ...b, name: newName } : b))
    );
    setEditingId(null);
  };

  const handleDelete = (id: string) => {
    setBricks(bricks.filter((b) => b.id !== id));
    setPriorityIds(priorityIds.filter((pid) => pid !== id));
  };

  const handleAddBrick = () => {
    if (!newBrickName.trim()) return;
    const newBrick: BrickItem = {
      id: crypto.randomUUID(),
      name: newBrickName.trim(),
    };
    setBricks([...bricks, newBrick]);
    setNewBrickName("");
  };

  const togglePriority = (id: string) => {
    if (priorityIds.includes(id)) {
      setPriorityIds(priorityIds.filter((pid) => pid !== id));
    } else if (priorityIds.length < 10) {
      setPriorityIds([...priorityIds, id]);
    }
  };

  const toggleVertical = (id: string) => {
    if (selectedVerticals.includes(id)) {
      setSelectedVerticals(selectedVerticals.filter((v) => v !== id));
    } else {
      setSelectedVerticals([...selectedVerticals, id]);
    }
  };

  const handleSave = async () => {
    await updateBricks.mutateAsync({
      bricks,
      priority_brick_ids: priorityIds,
      vertical_focus: selectedVerticals,
    });
  };

  const handleConfirm = async () => {
    await handleSave();
    await confirmBricks.mutateAsync();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-8 h-8 animate-spin text-oxford" />
      </div>
    );
  }

  const priorityBricks = bricks.filter((b) => priorityIds.includes(b.id));
  const otherBricks = bricks.filter((b) => !priorityIds.includes(b.id));

  return (
    <div className="space-y-6">
      <StepHeader
        icon={Layers}
        step={2}
        title="Brick Model"
        subtitle="Based on the Context Pack, define which capabilities and customer segments to focus on for company search. The broader the selection, the broader the results — narrow down to find precise matches."
      />

      {/* Status Banner */}
      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.brick_model
              ? "bg-success/10 border-success"
              : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.brick_model ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.brick_model ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.brick_model
                ? "Brick model confirmed — you can proceed to Universe"
                : gates.missing_items.brick_model?.join(", ") || "Confirm taxonomy to continue"}
            </span>
          </div>
        </div>
      )}

      {/* Two Columns: Capabilities and Customer Base */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Capabilities Column */}
        <div className="bg-steel-50 border border-steel-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-oxford flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Capabilities ({bricks.length})
            </h2>
            {taxonomy?.confirmed && (
              <span className="badge-success flex items-center gap-1">
                <Check className="w-3 h-3" />
                Confirmed
              </span>
            )}
          </div>
          <p className="text-sm text-steel-600 mb-4">
            Product, solution, and software capabilities
          </p>

          {/* Add New Brick */}
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={newBrickName}
              onChange={(e) => setNewBrickName(e.target.value)}
              placeholder="Add a new capability..."
              className="input text-sm"
              onKeyDown={(e) => e.key === "Enter" && handleAddBrick()}
            />
            <button
              onClick={handleAddBrick}
              disabled={!newBrickName.trim()}
              className="btn-primary px-3 disabled:opacity-50"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>

          {/* Brick List */}
          <div className="space-y-2 mb-6 max-h-[400px] overflow-y-auto">
            {bricks.length === 0 ? (
              <div className="text-center py-8 text-steel-400">
                <Layers className="w-8 h-8 mx-auto mb-2" />
                <p className="text-sm">No capabilities yet. Add your first one above.</p>
              </div>
            ) : (
              bricks.map((brick) => (
                <div
                  key={brick.id}
                  className="flex items-center gap-2 px-3 py-2 bg-white border border-steel-200 group"
                >
                  <button
                    onClick={() => togglePriority(brick.id)}
                    className={clsx(
                      "transition",
                      priorityIds.includes(brick.id)
                        ? "text-warning"
                        : "text-steel-300 hover:text-warning"
                    )}
                  >
                    <Star
                      className={clsx(
                        "w-4 h-4",
                        priorityIds.includes(brick.id) && "fill-current"
                      )}
                    />
                  </button>

                  {editingId === brick.id ? (
                    <input
                      type="text"
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onBlur={() => handleRename(brick.id, editValue)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleRename(brick.id, editValue);
                        if (e.key === "Escape") setEditingId(null);
                      }}
                      className="flex-1 px-2 py-1 border border-oxford text-sm"
                      autoFocus
                    />
                  ) : (
                    <span className="flex-1 text-sm">{brick.name}</span>
                  )}

                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition">
                    <button
                      onClick={() => {
                        setEditingId(brick.id);
                        setEditValue(brick.name);
                      }}
                      className="text-steel-400 hover:text-oxford"
                    >
                      <Edit2 className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => handleDelete(brick.id)}
                      className="text-steel-400 hover:text-danger"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Customer Base Column */}
        <div className="bg-steel-50 border border-steel-200 p-6">
          <h2 className="text-lg font-semibold text-oxford mb-4">
            Customer Base
          </h2>
          <p className="text-sm text-steel-600 mb-4">
            Target verticals for acquisition opportunities
          </p>
          <div className="flex flex-wrap gap-2">
            {VERTICALS.map((vertical) => {
              const isSelected = selectedVerticals.includes(vertical.id);
              return (
                <button
                  key={vertical.id}
                  onClick={() => toggleVertical(vertical.id)}
                  className={clsx(
                    "px-3 py-1.5 text-sm border transition",
                    isSelected
                      ? "bg-oxford border-oxford text-white"
                      : "bg-white border-steel-300 text-steel-600 hover:border-oxford"
                  )}
                >
                  {vertical.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Priority Bricks Section Below */}
      <div className="bg-oxford text-white border border-oxford-dark p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Star className="w-5 h-5 text-warning" />
          Priority Bricks ({priorityIds.length}/10)
        </h2>
        <p className="text-sm text-steel-300 mb-4">
          Select 3-10 bricks that are most important for your acquisition strategy
        </p>

        {priorityBricks.length === 0 ? (
          <div className="text-center py-8 text-steel-400">
            <Star className="w-8 h-8 mx-auto mb-2" />
            <p className="text-sm">Click the star icon on capabilities to prioritize</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {priorityBricks.map((brick, index) => (
              <div
                key={brick.id}
                className="flex items-center gap-2 px-3 py-2 bg-oxford-light border border-oxford"
              >
                <span className="text-warning font-medium text-sm">
                  {index + 1}.
                </span>
                <span className="flex-1 text-sm">{brick.name}</span>
                <button
                  onClick={() => togglePriority(brick.id)}
                  className="text-warning hover:text-white"
                >
                  <Star className="w-4 h-4 fill-current" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-3 pt-4">
        <button
          onClick={handleSave}
          disabled={updateBricks.isPending}
          className="btn-secondary disabled:opacity-50"
        >
          {updateBricks.isPending ? "Saving..." : "Save Changes"}
        </button>
        <button
          onClick={handleConfirm}
          disabled={
            confirmBricks.isPending ||
            priorityIds.length < 3 ||
            taxonomy?.confirmed
          }
          className="flex-1 btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {confirmBricks.isPending ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : taxonomy?.confirmed ? (
            <>
              <Check className="w-4 h-4" />
              Confirmed
            </>
          ) : (
            <>
              <Check className="w-4 h-4" />
              Confirm Taxonomy & Continue
            </>
          )}
        </button>
      </div>
    </div>
  );
}
