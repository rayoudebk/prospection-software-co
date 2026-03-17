"use client";

import { useParams, usePathname } from "next/navigation";
import Link from "next/link";
import { useWorkspace, useGates } from "@/lib/hooks";
import {
  ArrowLeft,
  Loader2,
  Lock,
  Check,
} from "lucide-react";
import clsx from "clsx";

const navItems = [
  {
    href: "context",
    label: "Source & Brief",
    step: 1,
    gateKey: null,
  },
  {
    href: "bricks",
    label: "Expansion",
    step: 2,
    gateKey: "context_pack",
  },
  {
    href: "universe",
    label: "Universe",
    step: 3,
    gateKey: "scope_review",
  },
];

export default function WorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const params = useParams();
  const pathname = usePathname();
  const workspaceId = Number(params.id);

  const { data: workspace, isLoading: workspaceLoading } = useWorkspace(workspaceId);
  const { data: gates } = useGates(workspaceId);

  const currentPath = pathname.split("/").pop();

  const completedSteps = gates
    ? [
        gates.context_pack,
        gates.scope_review,
        gates.universe,
      ]
    : [];

  if (workspaceLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-cream">
        <Loader2 className="w-6 h-6 animate-spin text-oxford" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cream">
      {/* Top bar */}
      <div className="bg-oxford text-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center h-12">
            <Link
              href="/"
              className="flex items-center gap-1.5 text-steel-400 hover:text-white mr-6 transition text-sm"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Workspaces</span>
            </Link>
            <div className="flex items-center gap-2 flex-1">
              <span className="text-sm font-medium">
                {workspace?.name || "Loading..."}
              </span>
              {workspace?.region_scope && (
                <span className="text-[11px] text-steel-400 font-normal">
                  · {workspace.region_scope}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Step navigation */}
      <div className="bg-oxford-dark border-b border-oxford">
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex">
            {navItems.map((item, idx) => {
              const isActive = currentPath === item.href;
              const isCompleted = completedSteps[idx];
              const isLocked =
                item.gateKey &&
                gates &&
                !gates[item.gateKey as keyof typeof gates];

              return (
                <Link
                  key={item.href}
                  href={`/workspaces/${workspaceId}/${item.href}`}
                  className={clsx(
                    "relative flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition",
                    isActive
                      ? "text-white bg-oxford"
                      : isLocked
                      ? "text-steel-500 cursor-not-allowed"
                      : "text-steel-300 hover:text-white hover:bg-oxford"
                  )}
                  onClick={(e) => {
                    if (isLocked) e.preventDefault();
                  }}
                >
                  {/* Step badge */}
                  <span
                    className={clsx(
                      "flex items-center justify-center w-5 h-5 text-xs font-bold shrink-0",
                      isCompleted
                        ? "bg-success text-white"
                        : isActive
                        ? "bg-steel-50 text-oxford"
                        : isLocked
                        ? "bg-steel-600 text-steel-400"
                        : "bg-steel-500 text-white"
                    )}
                  >
                    {isCompleted ? <Check className="w-3 h-3" /> : item.step}
                  </span>

                  <span>{item.label}</span>
                  {isLocked && <Lock className="w-3 h-3" />}

                  {/* Active underline */}
                  {isActive && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white" />
                  )}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Page content */}
      <div className="max-w-7xl mx-auto px-6 py-6">{children}</div>
    </div>
  );
}
