"use client";

import { useParams, usePathname } from "next/navigation";
import Link from "next/link";
import { useWorkspace, useGates } from "@/lib/hooks";
import {
  FileText,
  Layers,
  Globe,
  FileSpreadsheet,
  ArrowLeft,
  Loader2,
  Lock,
  Check,
} from "lucide-react";
import clsx from "clsx";

const navItems = [
  {
    href: "context",
    label: "Context Pack",
    description: "Crawl your website & references to ground buyer context",
    icon: FileText,
    step: 1,
    gateKey: null,
  },
  {
    href: "bricks",
    label: "Brick Model",
    description: "Define capabilities and customer segments for target fit",
    icon: Layers,
    step: 2,
    gateKey: "context_pack",
  },
  {
    href: "universe",
    label: "Universe",
    description: "Build and curate the candidate longlist",
    icon: Globe,
    step: 3,
    gateKey: "brick_model",
  },
  {
    href: "report",
    label: "Report",
    description: "Generate static compete/complement cards with source pills",
    icon: FileSpreadsheet,
    step: 4,
    gateKey: "universe",
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

  // Calculate current step and completion
  const currentStepIndex = navItems.findIndex((item) => item.href === currentPath);
  const completedSteps = gates
    ? [
        gates.context_pack,
        gates.brick_model,
        gates.universe,
        gates.universe,
      ]
    : [];

  if (workspaceLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-steel-100">
        <Loader2 className="w-8 h-8 animate-spin text-oxford" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-steel-100">
      {/* Top Bar */}
      <div className="bg-oxford text-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center h-16">
            <Link
              href="/"
              className="flex items-center gap-2 text-steel-300 hover:text-white mr-6 transition"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm">All Workspaces</span>
            </Link>
            <div className="flex-1">
              <h1 className="text-lg font-semibold">
                {workspace?.name || "Loading..."}
              </h1>
              <p className="text-sm text-steel-400">{workspace?.region_scope}</p>
            </div>
            {/* Progress indicator */}
            <div className="hidden md:flex items-center gap-2 text-sm text-steel-300">
              <span>Progress:</span>
              <div className="flex gap-1 ml-2">
                {navItems.map((item, idx) => (
                  <div
                    key={item.href}
                    className={clsx(
                      "w-8 h-1.5",
                      completedSteps[idx]
                        ? "bg-success"
                        : idx === currentStepIndex
                        ? "bg-steel-50"
                        : "bg-oxford-light"
                    )}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Step Navigation */}
      <div className="bg-oxford-dark border-b border-oxford">
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex gap-0">
            {navItems.map((item, idx) => {
              const isActive = currentPath === item.href;
              const isCompleted = completedSteps[idx];
              const isLocked = item.gateKey && gates && !gates[item.gateKey as keyof typeof gates];
              const Icon = item.icon;

              return (
                <Link
                  key={item.href}
                  href={`/workspaces/${workspaceId}/${item.href}`}
                  className={clsx(
                    "relative flex items-start gap-2 px-4 py-3 text-sm font-medium transition",
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
                  {/* Step number badge */}
                  <span
                    className={clsx(
                      "flex items-center justify-center w-5 h-5 text-xs font-bold mt-0.5 shrink-0",
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
                  
                  <div className="flex flex-col">
                    <div className="flex items-center gap-1">
                      <span>{item.label}</span>
                      {isLocked && <Lock className="w-3 h-3" />}
                    </div>
                    <span
                      className={clsx(
                        "text-xs font-normal max-w-[180px] leading-tight hidden lg:block",
                        isActive ? "text-steel-300" : "text-steel-500"
                      )}
                    >
                      {item.description}
                    </span>
                  </div>

                  {/* Active indicator */}
                  {isActive && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white" />
                  )}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-6">{children}</div>
    </div>
  );
}
