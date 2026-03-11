"use client";

import { Job } from "@/lib/api";
import clsx from "clsx";
import { CheckCircle2, Loader2, Square } from "lucide-react";

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
    title: "Building sourcing brief",
    statusLabel: "Researching brief inputs...",
    steps: [
      { label: "Prepare source inputs", start: 0.02 },
      { label: "Crawl company and reference sites", start: 0.2 },
      { label: "Extract product and evidence pages", start: 0.45 },
      { label: "Draft sourcing brief from evidence", start: 0.8 },
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

type LiveEvent = {
  message: string;
  timestamp?: string | null;
};

function extractLiveEvents(job: Job): LiveEvent[] {
  const payload = job.result_json;
  if (!payload || typeof payload !== "object") return [];
  const liveEvents = (payload as Record<string, unknown>).live_events;
  if (!Array.isArray(liveEvents)) return [];
  return liveEvents
    .filter((event): event is Record<string, unknown> => !!event && typeof event === "object")
    .map((event) => ({
      message: typeof event.message === "string" ? event.message : "",
      timestamp: typeof event.timestamp === "string" ? event.timestamp : null,
    }))
    .filter((event) => event.message);
}

function extractFirstUrl(message: string): string | null {
  const match = message.match(/https?:\/\/[^\s]+/);
  return match ? match[0] : null;
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
  const liveEvents = extractLiveEvents(job).slice(-5).reverse();
  const safeProgress =
    job.state === "completed"
      ? 1
      : job.state === "queued"
      ? Math.max(progress || 0, 0.02)
      : progress || 0;

  return (
    <div className="border border-oxford/20 bg-oxford text-white p-5 space-y-5">
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

      {liveEvents.length > 0 ? (
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide text-steel-400">Recent Source Activity</div>
          <div className="space-y-2 border border-white/10 bg-white/5 px-3 py-3">
            {liveEvents.map((event, index) => {
              const sourceUrl = extractFirstUrl(event.message);
              return (
                <div key={`${event.timestamp || "event"}-${index}`} className="text-sm text-steel-200 leading-relaxed">
                  {sourceUrl ? (
                    <a href={sourceUrl} target="_blank" rel="noopener noreferrer" className="hover:underline break-all">
                      {event.message}
                    </a>
                  ) : (
                    <span>{event.message}</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

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
        <div className="flex items-center gap-3">
          <div className="h-2 flex-1 bg-white/10 overflow-hidden">
            <div className="h-full bg-white transition-all duration-300" style={{ width: `${Math.round(safeProgress * 100)}%` }} />
          </div>
          {onStop ? (
            <button
              type="button"
              onClick={onStop}
              disabled={isStopping}
              aria-label="Stop job"
              title="Stop"
              className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/15 bg-white/5 text-white hover:bg-white/10 disabled:opacity-50"
            >
              {isStopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-3.5 w-3.5 fill-current" />}
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
