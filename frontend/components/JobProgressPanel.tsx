"use client";

import { Job } from "@/lib/api";
import clsx from "clsx";
import { CheckCircle2, Loader2, Square, StopCircle } from "lucide-react";

type StepConfig = {
  label: string;
  start: number;
};

const JOB_BLUEPRINTS: Record<
  string,
  {
    title: string;
    statusLabel: string;
    steps: StepConfig[];
  }
> = {
  context_pack: {
    title: "Building company thesis",
    statusLabel: "Researching company materials...",
    steps: [
      { label: "Prepare source inputs", start: 0.02 },
      { label: "Crawl company and reference sites", start: 0.2 },
      { label: "Extract product and proof pages", start: 0.45 },
      { label: "Draft thesis from evidence", start: 0.8 },
    ],
  },
  discovery_universe: {
    title: "Building candidate universe",
    statusLabel: "Sourcing candidate companies...",
    steps: [
      { label: "Load sourcing inputs", start: 0.05 },
      { label: "Search the candidate universe", start: 0.35 },
      { label: "Expand and normalize companies", start: 0.55 },
      { label: "Score and classify fit", start: 0.75 },
      { label: "Persist the universe", start: 0.92 },
    ],
  },
  generate_report_snapshot: {
    title: "Building card snapshot",
    statusLabel: "Preparing exportable candidate cards...",
    steps: [
      { label: "Prepare snapshot inputs", start: 0.05 },
      { label: "Compile candidate cards", start: 0.35 },
      { label: "Persist immutable snapshot", start: 0.9 },
    ],
  },
};

function resolveBlueprint(jobType?: string) {
  return (
    (jobType ? JOB_BLUEPRINTS[jobType] : null) || {
      title: "Running job",
      statusLabel: "Working...",
      steps: [
        { label: "Queued", start: 0.05 },
        { label: "Processing", start: 0.5 },
        { label: "Finishing", start: 0.9 },
      ],
    }
  );
}

function deriveStepState(job: Job, steps: StepConfig[], index: number) {
  if (job.state === "completed") return "done";
  if (job.state === "failed") return "failed";

  const safeProgress = job.state === "queued" ? Math.max(job.progress || 0, 0.02) : job.progress || 0;
  const current = steps[index];
  const next = steps[index + 1];

  if (next && safeProgress >= next.start) return "done";
  if (safeProgress >= current.start) return "active";
  return "pending";
}

export function JobProgressPanel({
  job,
  progress,
  progressMessage,
  isStopping = false,
  onStop,
}: {
  job: Job;
  progress: number;
  progressMessage?: string | null;
  isStopping?: boolean;
  onStop?: () => void;
}) {
  const blueprint = resolveBlueprint(job.job_type);
  const failedByUser = (job.error_message || "").toLowerCase().includes("stopped by user");
  const safeProgress =
    job.state === "completed"
      ? 1
      : job.state === "queued"
      ? Math.max(progress || 0, 0.02)
      : progress || 0;

  return (
    <div className="border border-oxford/20 bg-oxford text-white p-5 space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">{blueprint.title}</h3>
          <p className="text-sm text-steel-300 mt-1">
            {job.state === "queued"
              ? "Queued for a worker..."
              : job.state === "failed"
              ? progressMessage || job.error_message || "Stopped"
              : progressMessage || blueprint.statusLabel}
          </p>
        </div>
        {onStop ? (
          <button
            onClick={onStop}
            disabled={isStopping}
            className="inline-flex items-center gap-2 px-3 py-2 text-sm border border-white/20 bg-white/5 hover:bg-white/10 disabled:opacity-50"
          >
            {isStopping ? <Loader2 className="w-4 h-4 animate-spin" /> : <StopCircle className="w-4 h-4" />}
            Stop
          </button>
        ) : null}
      </div>

      <div className="space-y-3">
        {blueprint.steps.map((step, index) => {
          const state = deriveStepState(job, blueprint.steps, index);

          return (
            <div key={`${job.job_type}-${step.label}`} className="flex items-start gap-3">
              <div className="pt-0.5">
                {state === "done" ? (
                  <CheckCircle2 className="w-5 h-5 text-success" />
                ) : state === "active" ? (
                  <Loader2 className="w-5 h-5 animate-spin text-white" />
                ) : (
                  <div className="w-5 h-5 rounded-full border border-white/25" />
                )}
              </div>
              <div className={clsx("text-sm leading-relaxed", state === "pending" ? "text-steel-400" : "text-white")}>
                {step.label}
              </div>
            </div>
          );
        })}
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm text-steel-300">
          <span>
            {job.state === "failed"
              ? failedByUser
                ? "Stopped"
                : "Failed"
              : job.state === "completed"
              ? "Complete"
              : blueprint.statusLabel}
          </span>
          <span>{Math.round(safeProgress * 100)}%</span>
        </div>
        <div className="h-2 bg-white/10 overflow-hidden">
          <div className="h-full bg-white transition-all duration-300" style={{ width: `${Math.round(safeProgress * 100)}%` }} />
        </div>
      </div>
    </div>
  );
}
