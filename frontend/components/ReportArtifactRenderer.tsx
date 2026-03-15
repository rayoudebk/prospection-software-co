"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import {
  Check,
  Copy,
  ExternalLink,
  MessageSquareMore,
  RefreshCw,
  Share2,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";

import {
  ReportArtifact,
  ReportArtifactBlock,
  ReportArtifactSourcePill,
} from "@/lib/api";

function sourceChipLabel(source: ReportArtifactSourcePill): string {
  return source.label || source.publisher || "Source";
}

function SourceDrawer({
  isOpen,
  title,
  sources,
  onClose,
}: {
  isOpen: boolean;
  title: string;
  sources: ReportArtifactSourcePill[];
  onClose: () => void;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        aria-label="Close sources panel"
        onClick={onClose}
        className="absolute inset-0 bg-oxford/20 backdrop-blur-[2px]"
      />
      <aside
        role="dialog"
        aria-modal="true"
        className="absolute inset-y-0 right-0 flex w-full max-w-xl flex-col border-l border-white/10 bg-[#111315] text-white shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-white/45">Sources</p>
            <h4 className="mt-1 text-xl font-semibold">{title}</h4>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 text-white/70 transition-colors hover:border-white/20 hover:bg-white/5 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
          {sources.map((source) => (
            <a
              key={source.id}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block rounded-2xl border border-white/10 bg-white/[0.03] p-4 transition-colors hover:border-white/20 hover:bg-white/[0.06]"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-base font-semibold text-white">
                    {sourceChipLabel(source)}
                  </div>
                  <div className="mt-1 break-all text-sm text-white/60">{source.url}</div>
                  <div className="mt-2 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.16em] text-white/45">
                    <span>{source.publisher_channel}</span>
                    <span>{source.source_tier}</span>
                    <span>{source.evidence_type}</span>
                  </div>
                </div>
                <ExternalLink className="mt-1 h-4 w-4 shrink-0 text-white/40" />
              </div>
            </a>
          ))}
        </div>
      </aside>
    </div>
  );
}

function InlineCitations({
  pillIds,
  pillById,
  onOpenSources,
}: {
  pillIds: string[];
  pillById: Map<string, ReportArtifactSourcePill>;
  onOpenSources: (pillIds: string[]) => void;
}) {
  const sources = pillIds.map((pillId) => pillById.get(pillId)).filter(Boolean) as ReportArtifactSourcePill[];
  if (!sources.length) return null;
  const [first, ...rest] = sources;

  return (
    <span className="ml-1 inline-flex items-center gap-1 align-middle">
      <a
        href={first.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center rounded-full border border-steel-200 bg-steel-50 px-2 py-0.5 text-[11px] font-medium text-steel-700 transition-colors hover:border-oxford/30 hover:text-oxford"
        title={`${sourceChipLabel(first)} · ${first.publisher_channel}`}
      >
        {sourceChipLabel(first)}
      </a>
      {rest.length ? (
        <button
          type="button"
          onClick={() => onOpenSources(pillIds)}
          className="inline-flex items-center rounded-full border border-steel-200 bg-white px-2 py-0.5 text-[11px] font-medium text-steel-600 transition-colors hover:border-oxford/30 hover:text-oxford"
          title="Show all cited sources"
        >
          +{rest.length}
        </button>
      ) : null}
    </span>
  );
}

function RenderBlock({
  block,
  pillById,
  onOpenSources,
}: {
  block: ReportArtifactBlock;
  pillById: Map<string, ReportArtifactSourcePill>;
  onOpenSources: (pillIds: string[]) => void;
}) {
  if (block.type === "paragraph") {
    return (
      <div className="space-y-3">
        {block.sentences.map((sentence) => (
          <p key={sentence.id} className="text-[15px] leading-7 text-steel-700">
            {sentence.text}
            <InlineCitations
              pillIds={sentence.citation_pill_ids}
              pillById={pillById}
              onOpenSources={onOpenSources}
            />
          </p>
        ))}
      </div>
    );
  }

  if (block.type === "bullet_list") {
    return (
      <ul className="space-y-3">
        {block.items.map((item) => (
          <li key={item.id} className="flex gap-3 text-[15px] leading-7 text-steel-700">
            <span className="mt-[10px] h-1.5 w-1.5 shrink-0 rounded-full bg-steel-400" />
            <span>
              {item.text}
              <InlineCitations
                pillIds={item.citation_pill_ids}
                pillById={pillById}
                onOpenSources={onOpenSources}
              />
            </span>
          </li>
        ))}
      </ul>
    );
  }

  if (block.type === "callout") {
    const toneClass =
      block.tone === "warning"
        ? "border-warning/25 bg-warning/10"
        : block.tone === "success"
          ? "border-success/20 bg-success/10"
          : "border-steel-200 bg-steel-50";
    return (
      <div className={clsx("rounded-2xl border px-4 py-3", toneClass)}>
        {block.title ? (
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-steel-500">
            {block.title}
          </div>
        ) : null}
        <div className="space-y-2">
          {block.sentences.map((sentence) => (
            <p key={sentence.id} className="text-sm leading-6 text-steel-700">
              {sentence.text}
              <InlineCitations
                pillIds={sentence.citation_pill_ids}
                pillById={pillById}
                onOpenSources={onOpenSources}
              />
            </p>
          ))}
        </div>
      </div>
    );
  }

  if (block.type === "key_value") {
    return (
      <div className="grid gap-3">
        {block.items.map((item) => (
          <div key={item.id} className="grid gap-1 border-b border-steel-100 pb-3 last:border-b-0 last:pb-0">
            <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-steel-400">
              {item.key}
            </div>
            <div className="text-sm leading-6 text-steel-700">
              {item.value}
              <InlineCitations
                pillIds={item.citation_pill_ids}
                pillById={pillById}
                onOpenSources={onOpenSources}
              />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return null;
}

export function ReportArtifactRenderer({
  artifact,
  onRegenerate,
}: {
  artifact: ReportArtifact;
  onRegenerate?: () => void;
}) {
  const [selectedFeedback, setSelectedFeedback] = useState<"good" | "bad" | null>(null);
  const [drawerTitle, setDrawerTitle] = useState("Cited Sources");
  const [drawerSources, setDrawerSources] = useState<ReportArtifactSourcePill[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const pillById = useMemo(
    () => new Map((artifact.sources || []).map((source) => [source.id, source])),
    [artifact.sources],
  );

  const handleOpenSources = (pillIds: string[]) => {
    const sources = pillIds.map((pillId) => pillById.get(pillId)).filter(Boolean) as ReportArtifactSourcePill[];
    setDrawerTitle(sources.length > 1 ? "Cited Sources" : sourceChipLabel(sources[0]));
    setDrawerSources(sources);
    setDrawerOpen(true);
  };

  const handleCopy = async () => {
    const body = artifact.sections
      .flatMap((section) =>
        section.blocks.flatMap((block) => {
          if (block.type === "paragraph" || block.type === "callout") {
            return block.sentences.map((sentence) => sentence.text);
          }
          if (block.type === "bullet_list") {
            return block.items.map((item) => `- ${item.text}`);
          }
          if (block.type === "key_value") {
            return block.items.map((item) => `${item.key}: ${item.value}`);
          }
          return [];
        }),
      )
      .join("\n");
    await navigator.clipboard.writeText(`${artifact.title}\n\n${body}`.trim());
  };

  const handleShare = async () => {
    if (navigator.share) {
      await navigator.share({
        title: artifact.title,
        text: artifact.summary || artifact.title,
      });
      return;
    }
    await navigator.clipboard.writeText(window.location.href);
  };

  return (
    <>
      <div className="rounded-[28px] border border-steel-200 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em] text-steel-400">
              {artifact.report_kind.replace(/_/g, " ")}
            </div>
            <h2 className="mt-2 text-2xl font-semibold text-oxford">{artifact.title}</h2>
            {artifact.reasoning_warning ? (
              <p className="mt-3 text-sm text-warning-dark">{artifact.reasoning_warning}</p>
            ) : artifact.reasoning_provider ? (
              <p className="mt-3 text-sm text-steel-500">
                Generated via {artifact.reasoning_provider}
                {artifact.reasoning_model ? ` · ${artifact.reasoning_model}` : ""}
              </p>
            ) : null}
          </div>
          <span
            className={clsx(
              "rounded-full border px-3 py-1 text-xs font-medium uppercase tracking-[0.18em]",
              artifact.status === "ready"
                ? "border-success/25 bg-success/10 text-success"
                : "border-warning/25 bg-warning/10 text-warning-dark",
            )}
          >
            {artifact.status}
          </span>
        </div>

        <div className="mt-6 space-y-8">
          {artifact.sections.map((section) => (
            <section key={section.id} className="space-y-4">
              {section.heading ? (
                <h3 className="text-[11px] font-semibold uppercase tracking-[0.22em] text-steel-400">
                  {section.heading}
                </h3>
              ) : null}
              <div className="space-y-4">
                {section.blocks.map((block, index) => (
                  <RenderBlock
                    key={`${section.id}-${block.type}-${index}`}
                    block={block}
                    pillById={pillById}
                    onOpenSources={handleOpenSources}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="mt-8 border-t border-steel-100 pt-5">
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-steel-400">
            Sources
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {artifact.sources.map((source) => (
              <a
                key={source.id}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-full border border-steel-200 bg-steel-50 px-3 py-1 text-xs font-medium text-steel-700 transition-colors hover:border-oxford/30 hover:text-oxford"
                title={source.url}
              >
                <span>{sourceChipLabel(source)}</span>
                <ExternalLink className="h-3 w-3" />
              </a>
            ))}
          </div>
        </div>

        <div className="mt-8 flex flex-wrap items-center gap-2 border-t border-steel-100 pt-5">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-2 rounded-full border border-steel-200 px-3 py-1.5 text-sm text-steel-700 transition-colors hover:border-oxford/30 hover:text-oxford"
          >
            <Copy className="h-4 w-4" />
            Copy
          </button>
          <button
            type="button"
            onClick={() => setSelectedFeedback("good")}
            className={clsx(
              "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition-colors",
              selectedFeedback === "good"
                ? "border-success/30 bg-success/10 text-success"
                : "border-steel-200 text-steel-700 hover:border-oxford/30 hover:text-oxford",
            )}
          >
            <ThumbsUp className="h-4 w-4" />
            Good
          </button>
          <button
            type="button"
            onClick={() => setSelectedFeedback("bad")}
            className={clsx(
              "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition-colors",
              selectedFeedback === "bad"
                ? "border-warning/30 bg-warning/10 text-warning-dark"
                : "border-steel-200 text-steel-700 hover:border-oxford/30 hover:text-oxford",
            )}
          >
            <ThumbsDown className="h-4 w-4" />
            Bad
          </button>
          <button
            type="button"
            onClick={handleShare}
            className="inline-flex items-center gap-2 rounded-full border border-steel-200 px-3 py-1.5 text-sm text-steel-700 transition-colors hover:border-oxford/30 hover:text-oxford"
          >
            <Share2 className="h-4 w-4" />
            Share
          </button>
          <button
            type="button"
            onClick={onRegenerate}
            disabled={!onRegenerate}
            className="inline-flex items-center gap-2 rounded-full border border-steel-200 px-3 py-1.5 text-sm text-steel-700 transition-colors hover:border-oxford/30 hover:text-oxford disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw className="h-4 w-4" />
            Regenerate
          </button>
          {selectedFeedback ? (
            <span className="ml-auto inline-flex items-center gap-1 text-sm text-steel-500">
              <Check className="h-4 w-4" />
              Feedback captured locally
            </span>
          ) : (
            <span className="ml-auto inline-flex items-center gap-1 text-sm text-steel-500">
              <MessageSquareMore className="h-4 w-4" />
              Report actions
            </span>
          )}
        </div>
      </div>

      <SourceDrawer
        isOpen={drawerOpen}
        title={drawerTitle}
        sources={drawerSources}
        onClose={() => setDrawerOpen(false)}
      />
    </>
  );
}
