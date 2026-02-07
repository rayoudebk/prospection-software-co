"use client";

import { useState } from "react";
import Link from "next/link";
import { useWorkspaces, useCreateWorkspace, useDeleteWorkspace } from "@/lib/hooks";
import { Plus, Trash2, MapPin, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import clsx from "clsx";

export default function WorkspacesPage() {
  const { data: workspaces, isLoading, error } = useWorkspaces();
  const createWorkspace = useCreateWorkspace();
  const deleteWorkspace = useDeleteWorkspace();

  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newRegion, setNewRegion] = useState("EU+UK");

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;

    await createWorkspace.mutateAsync({
      name: newName.trim(),
      region_scope: newRegion,
    });

    setNewName("");
    setShowCreate(false);
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm("Delete this workspace and all its data?")) {
      await deleteWorkspace.mutateAsync(id);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-steel-100">
        <Loader2 className="w-8 h-8 animate-spin text-oxford" />
      </div>
    );
  }

  if (error) {
    // #region agent log
    const errorMsg = error instanceof Error ? error.message : String(error);
    fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'page.tsx:47',message:'Workspaces error displayed',data:{errorMessage:errorMsg,errorName:error instanceof Error ? error.name : 'Unknown'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    return (
      <div className="flex items-center justify-center min-h-screen bg-steel-200">
        <div className="text-danger font-medium">
          Error loading workspaces: {error instanceof Error ? error.message : String(error)}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-steel-200">
      {/* Header */}
      <div className="bg-oxford text-white">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">M&A Market Maps</h1>
              <p className="text-steel-300 mt-1">
                Create workspaces to research acquisition targets
              </p>
            </div>
            <button
              onClick={() => setShowCreate(true)}
              className="flex items-center gap-2 px-4 py-2 bg-steel-200 text-oxford font-medium hover:bg-steel-300 transition"
            >
              <Plus className="w-4 h-4" />
              New Workspace
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Create Modal */}
        {showCreate && (
          <div className="fixed inset-0 bg-oxford/80 flex items-center justify-center z-50">
            <div className="bg-steel-50 p-6 w-full max-w-md shadow-xl border border-steel-300">
              <h2 className="text-lg font-semibold text-oxford mb-4">Create Workspace</h2>
              <form onSubmit={handleCreate}>
                <div className="mb-4">
                  <label className="label">
                    Workspace Name
                  </label>
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="e.g., Wealth Management Targets Q1"
                    className="input"
                    autoFocus
                  />
                </div>
                <div className="mb-6">
                  <label className="label">
                    Region Scope
                  </label>
                  <select
                    value={newRegion}
                    onChange={(e) => setNewRegion(e.target.value)}
                    className="input"
                  >
                    <option value="EU+UK">EU + UK</option>
                    <option value="US">United States</option>
                    <option value="APAC">Asia Pacific</option>
                    <option value="Global">Global</option>
                  </select>
                </div>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => setShowCreate(false)}
                    className="flex-1 btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={!newName.trim() || createWorkspace.isPending}
                    className="flex-1 btn-primary disabled:opacity-50"
                  >
                    {createWorkspace.isPending ? "Creating..." : "Create"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Workspaces Grid */}
        {workspaces && workspaces.length === 0 ? (
          <div className="text-center py-16 bg-steel-50 border border-steel-200">
            <MapPin className="w-12 h-12 text-steel-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-oxford mb-2">
              No workspaces yet
            </h3>
            <p className="text-steel-500 mb-4">
              Create your first workspace to start researching M&A targets
            </p>
            <button
              onClick={() => setShowCreate(true)}
              className="btn-primary"
            >
              Create Workspace
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {workspaces?.map((workspace) => (
              <Link
                key={workspace.id}
                href={`/workspaces/${workspace.id}/context`}
                className="group bg-steel-50 border border-steel-200 p-5 hover:border-oxford hover:shadow-md transition"
              >
                <div className="flex items-start justify-between mb-3">
                  <h3 className="font-semibold text-oxford group-hover:text-oxford-light">
                    {workspace.name}
                  </h3>
                  <button
                    onClick={(e) => handleDelete(workspace.id, e)}
                    className="p-1 text-steel-400 hover:text-danger opacity-0 group-hover:opacity-100 transition"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div className="flex items-center gap-2 text-sm text-steel-500 mb-4">
                  <MapPin className="w-4 h-4" />
                  {workspace.region_scope}
                </div>

                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-1">
                    {workspace.has_context_pack ? (
                      <CheckCircle className="w-4 h-4 text-success" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-steel-300" />
                    )}
                    <span className={workspace.has_context_pack ? "text-success" : "text-steel-400"}>
                      Context
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {workspace.has_confirmed_taxonomy ? (
                      <CheckCircle className="w-4 h-4 text-success" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-steel-300" />
                    )}
                    <span className={workspace.has_confirmed_taxonomy ? "text-success" : "text-steel-400"}>
                      Bricks
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className={clsx(
                      "font-medium",
                      workspace.vendor_count > 0 ? "text-oxford" : "text-steel-400"
                    )}>
                      {workspace.vendor_count}
                    </span>
                    <span className="text-steel-500">vendors</span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-steel-100 text-xs text-steel-400">
                  Created {new Date(workspace.created_at).toLocaleDateString()}
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
