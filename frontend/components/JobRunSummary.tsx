"use client";

import { Job } from "@/lib/api";

function formatDuration(totalSeconds?: number | null): string | null {
  if (typeof totalSeconds !== "number" || !Number.isFinite(totalSeconds) || totalSeconds <= 0) {
    return null;
  }

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes <= 0) return `${seconds}s`;
  return `${minutes}m ${seconds}s`;
}

function deriveDuration(job: Job): string | null {
  const payload = job.result_json && typeof job.result_json === "object" ? job.result_json : null;
  const payloadDuration = payload && typeof payload.duration_seconds === "number" ? payload.duration_seconds : null;
  if (payloadDuration != null) {
    return formatDuration(payloadDuration);
  }

  if (!job.started_at || !job.finished_at) return null;
  const started = new Date(job.started_at).getTime();
  const finished = new Date(job.finished_at).getTime();
  if (!Number.isFinite(started) || !Number.isFinite(finished) || finished <= started) return null;
  return formatDuration(Math.round((finished - started) / 1000));
}

function summaryLabel(jobType: string): string {
  if (jobType === "context_pack") return "Crawl completed";
  if (jobType === "discovery_universe") return "Universe run completed";
  if (jobType === "generate_report_snapshot") return "Cards snapshot completed";
  return "Run completed";
}

function deriveMetrics(job: Job): string[] {
  const payload = job.result_json && typeof job.result_json === "object" ? job.result_json : null;
  if (!payload) return [];

  if (job.job_type === "context_pack") {
    const metrics = [];
    if (typeof payload.sites_crawled === "number") metrics.push(`${payload.sites_crawled} sites`);
    if (typeof payload.pages_crawled === "number") metrics.push(`${payload.pages_crawled} pages`);
    if (typeof payload.product_pages_found === "number") metrics.push(`${payload.product_pages_found} product/proof pages`);
    return metrics;
  }

  if (job.job_type === "discovery_universe") {
    const metrics = [];
    if (typeof payload.candidates_found === "number") metrics.push(`${payload.candidates_found} candidates`);
    if (typeof payload.seed_external_search_count === "number") metrics.push(`${payload.seed_external_search_count} search seeds`);
    if (typeof payload.seed_reference_count === "number") metrics.push(`${payload.seed_reference_count} reference companies`);
    return metrics;
  }

  if (job.job_type === "generate_report_snapshot") {
    const metrics = [];
    if (typeof payload.total_items === "number") metrics.push(`${payload.total_items} cards`);
    if (typeof payload.filing_facts_created === "number") metrics.push(`${payload.filing_facts_created} filing facts`);
    return metrics;
  }

  return [];
}

export function JobRunSummary({ job }: { job: Job | null | undefined }) {
  if (!job || job.state !== "completed") return null;

  const duration = deriveDuration(job);
  const metrics = deriveMetrics(job);
  const detailParts = [duration, ...metrics].filter(Boolean) as string[];

  return (
    <div className="border border-steel-200 bg-steel-50 px-4 py-3 text-sm text-steel-700">
      <span className="font-medium text-oxford">{summaryLabel(job.job_type)}</span>
      {detailParts.length > 0 ? `: ${detailParts.join(" | ")}` : ""}
    </div>
  );
}
