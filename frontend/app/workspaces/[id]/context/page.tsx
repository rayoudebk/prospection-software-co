"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  useContextPack,
  useUpdateContextPack,
  useRefreshContextPack,
  useWorkspaceJobWithPolling,
  useGates,
} from "@/lib/hooks";
import { workspaceApi } from "@/lib/api";
import {
  Building2,
  Plus,
  X,
  Globe,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Loader2,
  ExternalLink,
  FileText,
} from "lucide-react";
import { StepHeader } from "@/components/StepHeader";
import ReactMarkdown from "react-markdown";
import clsx from "clsx";

function normalizeUrlInput(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const candidate = trimmed.startsWith("http://") || trimmed.startsWith("https://")
    ? trimmed
    : `https://${trimmed}`;
  try {
    const parsed = new URL(candidate);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    const query = parsed.search || "";
    return `${parsed.protocol}//${parsed.host}${parsed.pathname}${query}`;
  } catch {
    return null;
  }
}

export default function ContextPackPage() {
  const params = useParams();
  const workspaceId = Number(params.id);

  const { data: profile, isLoading } = useContextPack(workspaceId);
  const { data: gates } = useGates(workspaceId);
  const updateProfile = useUpdateContextPack(workspaceId);
  const refreshContextPack = useRefreshContextPack(workspaceId);

  const [buyerUrl, setBuyerUrl] = useState("");
  const [referenceUrls, setReferenceUrls] = useState<string[]>([]);
  const [newReferenceUrl, setNewReferenceUrl] = useState("");
  const [referenceUrlError, setReferenceUrlError] = useState<string | null>(null);
  const [evidenceUrls, setEvidenceUrls] = useState<string[]>([]);
  const [newEvidenceUrl, setNewEvidenceUrl] = useState("");
  const [evidenceUrlError, setEvidenceUrlError] = useState<string | null>(null);

  // Initialize from profile after async hydration.
  useEffect(() => {
    if (profile) {
      setBuyerUrl(profile.buyer_company_url || "");
      setReferenceUrls(profile.reference_vendor_urls || []);
      setEvidenceUrls(profile.reference_evidence_urls || []);
    }
  }, [profile]);

  const jobRunner = useWorkspaceJobWithPolling(
    workspaceId,
    () => workspaceApi.refreshContextPack(workspaceId)
  );

  const handleSave = async () => {
    await updateProfile.mutateAsync({
      buyer_company_url: buyerUrl,
      reference_vendor_urls: referenceUrls,
      reference_evidence_urls: evidenceUrls,
    });
  };

  const handleAddReference = () => {
    const normalized = normalizeUrlInput(newReferenceUrl);
    if (!normalized) {
      setReferenceUrlError("Enter a valid URL.");
      return;
    }
    if (referenceUrls.some((url) => url.toLowerCase() === normalized.toLowerCase())) {
      setReferenceUrlError("That competitor URL is already added.");
      return;
    }
    if (referenceUrls.length >= 3) {
      setReferenceUrlError("You can only add up to 3 reference competitors.");
      return;
    }
    setReferenceUrls([...referenceUrls, normalized]);
    setNewReferenceUrl("");
    setReferenceUrlError(null);
  };

  const handleRemoveReference = (index: number) => {
    setReferenceUrls(referenceUrls.filter((_, i) => i !== index));
  };

  const handleAddEvidence = () => {
    const normalized = normalizeUrlInput(newEvidenceUrl);
    if (!normalized) {
      setEvidenceUrlError("Enter a valid proof-page URL.");
      return;
    }
    if (evidenceUrls.some((url) => url.toLowerCase() === normalized.toLowerCase())) {
      setEvidenceUrlError("That evidence URL is already added.");
      return;
    }
    setEvidenceUrls([...evidenceUrls, normalized]);
    setNewEvidenceUrl("");
    setEvidenceUrlError(null);
  };

  const handleRemoveEvidence = (index: number) => {
    setEvidenceUrls(evidenceUrls.filter((_, i) => i !== index));
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
        icon={FileText}
        step={1}
        title="Context Pack"
        subtitle="Use your website and competitors to understand your product & services landscape as well as your customer base. This builds the foundation for intelligent target discovery."
      />

      {/* Status Banner */}
      {gates && (
        <div
          className={clsx(
            "p-4 border",
            gates.context_pack
              ? "bg-success/10 border-success"
              : "bg-warning/10 border-warning"
          )}
        >
          <div className="flex items-center gap-2">
            {gates.context_pack ? (
              <CheckCircle className="w-5 h-5 text-success" />
            ) : (
              <AlertCircle className="w-5 h-5 text-warning" />
            )}
            <span className={gates.context_pack ? "text-success font-medium" : "text-warning font-medium"}>
              {gates.context_pack
                ? "Context pack ready — you can proceed to Brick Model"
                : gates.missing_items.context_pack?.join(", ") || "Complete the context pack to continue"}
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-steel-50 border border-steel-200 p-6">
          <h2 className="text-lg font-semibold text-oxford mb-4 flex items-center gap-2">
            <Building2 className="w-5 h-5" />
            Company Context
          </h2>

          {/* Buyer URL */}
          <div className="mb-6">
            <label className="label">
              Your Company URL
            </label>
            <input
              type="url"
              value={buyerUrl}
              onChange={(e) => setBuyerUrl(e.target.value)}
              placeholder="https://your-company.com"
              className="input"
            />
            <p className="text-xs text-steel-500 mt-1">
              We&apos;ll crawl this site to understand your capabilities
            </p>
          </div>

          {/* Reference URLs */}
          <div className="mb-6">
            <label className="label">
              Reference Competitors (optional, up to 3)
            </label>
            <div className="space-y-2 mb-2">
              {referenceUrls.map(
                (url, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-2 px-3 py-2 bg-steel-50 border border-steel-200"
                  >
                    <Globe className="w-4 h-4 text-steel-400" />
                    <span className="flex-1 text-sm truncate">{url}</span>
                    <button
                      onClick={() => handleRemoveReference(index)}
                      className="text-steel-400 hover:text-danger"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )
              )}
            </div>
            {referenceUrls.length < 3 && (
              <div className="flex gap-2">
                <input
                  type="url"
                  value={newReferenceUrl}
                  onChange={(e) => {
                    setNewReferenceUrl(e.target.value);
                    setReferenceUrlError(null);
                  }}
                  placeholder="https://reference-competitor.com"
                  className="input text-sm"
                />
                <button
                  onClick={handleAddReference}
                  disabled={!newReferenceUrl}
                  className="btn-secondary px-3 disabled:opacity-50"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>
            )}
            {referenceUrlError && (
              <p className="text-xs text-danger mt-2">{referenceUrlError}</p>
            )}
          </div>

          {/* Evidence URLs */}
          <div className="mb-6">
            <label className="label">
              Evidence Links (proof pages)
            </label>
            <p className="text-xs text-steel-500 mt-1 mb-2">
              Add direct proof pages (case studies, partnership/newsroom links, customer pages) for better depth scoring.
            </p>
            <div className="space-y-2 mb-2">
              {evidenceUrls.map((url, index) => (
                <div
                  key={`${url}-${index}`}
                  className="flex items-center gap-2 px-3 py-2 bg-steel-50 border border-steel-200"
                >
                  <Globe className="w-4 h-4 text-steel-400" />
                  <span className="flex-1 text-sm truncate">{url}</span>
                  <button
                    onClick={() => handleRemoveEvidence(index)}
                    className="text-steel-400 hover:text-danger"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="url"
                value={newEvidenceUrl}
                onChange={(e) => {
                  setNewEvidenceUrl(e.target.value);
                  setEvidenceUrlError(null);
                }}
                placeholder="https://vendor.com/blog/customer-proof"
                className="input text-sm"
              />
              <button
                onClick={handleAddEvidence}
                disabled={!newEvidenceUrl}
                className="btn-secondary px-3 disabled:opacity-50"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
            {evidenceUrlError && (
              <p className="text-xs text-danger mt-2">{evidenceUrlError}</p>
            )}
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={handleSave}
              disabled={updateProfile.isPending}
              className="btn-secondary disabled:opacity-50"
            >
              {updateProfile.isPending ? "Saving..." : "Save Changes"}
            </button>
            <button
              onClick={async () => {
                // Save URLs first, then run the crawl
                await updateProfile.mutateAsync({
                  buyer_company_url: buyerUrl,
                  reference_vendor_urls: referenceUrls,
                  reference_evidence_urls: evidenceUrls,
                });
                jobRunner.run();
              }}
              disabled={jobRunner.isRunning || updateProfile.isPending || !buyerUrl}
              className="flex-1 btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {jobRunner.isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Crawling... {Math.round(jobRunner.progress * 100)}%
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  Generate Context Pack
                </>
              )}
            </button>
          </div>

          {jobRunner.isRunning && jobRunner.progressMessage && (
            <div className="mt-3 p-3 bg-steel-100 border border-steel-200 rounded text-sm text-steel-700">
              <div className="flex items-start gap-2">
                <Loader2 className="w-3 h-3 animate-spin mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium mb-1">Progress:</div>
                  <div className="text-steel-600 whitespace-pre-wrap break-words">{jobRunner.progressMessage}</div>
                </div>
              </div>
            </div>
          )}

          {jobRunner.jobError && (
            <div className="mt-4 p-3 bg-danger/10 border border-danger text-danger text-sm">
              {jobRunner.jobError}
            </div>
          )}
        </div>

        {/* Context Pack Preview */}
        <div className="bg-oxford text-white border border-oxford-dark p-6">
          <h2 className="text-lg font-semibold mb-4">
            Context Pack
            {profile?.context_pack_generated_at && (
              <span className="text-sm font-normal text-steel-300 ml-2">
                Generated{" "}
                {new Date(profile.context_pack_generated_at).toLocaleString()}
              </span>
            )}
          </h2>

          {profile?.product_pages_found !== undefined && profile.product_pages_found > 0 && (
            <div className="mb-4 text-sm text-success flex items-center gap-1">
              <CheckCircle className="w-4 h-4" />
              {profile.product_pages_found} product pages found
            </div>
          )}

          {profile?.context_pack_markdown ? (
            <div className="prose-dark max-h-[600px] overflow-y-auto pr-2">
              <ReactMarkdown
                components={{
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-steel-200 hover:text-white underline inline-flex items-center gap-1"
                    >
                      {children}
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  ),
                }}
              >
                {profile.context_pack_markdown}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="text-center py-12 text-steel-300">
              <Globe className="w-12 h-12 mx-auto mb-4 text-steel-500" />
              <p>No context pack generated yet</p>
              <p className="text-sm text-steel-400">Add your company URL and click Generate</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
