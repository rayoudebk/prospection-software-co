// Use Next.js rewrite proxy in browser, direct URL in Node.js
const API_BASE = typeof window !== 'undefined' ? '/api' : 'http://localhost:8000';

// ============================================================================
// Workspace Types
// ============================================================================

export interface Workspace {
  id: number;
  name: string;
  region_scope: string;
  created_at: string;
  vendor_count: number;
  has_context_pack: boolean;
  has_confirmed_taxonomy: boolean;
}

export interface GeoScope {
  region: string;
  include_countries: string[];
  exclude_countries: string[];
}

export interface CompanyProfile {
  id: number;
  workspace_id: number;
  buyer_company_url: string | null;
  buyer_context_summary: string | null;
  reference_vendor_urls: string[];
  reference_evidence_urls: string[];
  reference_summaries: Record<string, string>;
  geo_scope: GeoScope;
  context_pack_markdown: string | null;
  context_pack_generated_at: string | null;
  product_pages_found: number;
}

export interface CitationSentence {
  id: string;
  text: string;
  citation_pill_ids: string[];
}

export interface CitationSourcePill {
  pill_id: string;
  label: string;
  url: string;
  source_tier: string;
  source_kind: string;
  captured_at?: string | null;
  claim_group: string;
}

export interface CitationSummaryV1 {
  version: "v1" | string;
  sentences: CitationSentence[];
  source_pills: CitationSourcePill[];
}

export interface BrickItem {
  id: string;
  name: string;
  description?: string | null;
}

export interface BrickTaxonomy {
  id: number;
  workspace_id: number;
  bricks: BrickItem[];
  priority_brick_ids: string[];
  vertical_focus: string[];
  version: number;
  confirmed: boolean;
}

export interface WhyRelevant {
  text: string;
  citation_url: string;
}

export interface Vendor {
  id: number;
  workspace_id: number;
  name: string;
  website: string | null;
  official_website_url?: string | null;
  discovery_url?: string | null;
  entity_type?: "company" | "solution" | "service_line" | string | null;
  hq_country: string | null;
  operating_countries: string[];
  tags_vertical: string[];
  tags_custom: string[];
  status: "candidate" | "kept" | "removed" | "enriched";
  why_relevant: WhyRelevant[];
  is_manual: boolean;
  created_at: string;
  evidence_count: number;
  decision_classification?: "good_target" | "borderline_watchlist" | "not_good_target" | "insufficient_evidence" | string | null;
  evidence_sufficiency?: "sufficient" | "insufficient" | "contradictory" | string | null;
  reason_codes?: {
    positive: string[];
    caution: string[];
    reject: string[];
  };
  rationale_summary?: string | null;
  top_claim?: {
    text?: string;
    claim_type?: string;
    source_url?: string;
    source_tier?: string;
    source_kind?: string;
    captured_at?: string;
  };
  citation_summary_v1?: CitationSummaryV1 | null;
  registry_neighbors_with_first_party_website_count?: number;
  registry_neighbors_dropped_missing_official_website_count?: number;
  registry_origin_screening_counts?: Record<string, number>;
  first_party_hint_urls_used_count?: number;
  first_party_hint_pages_crawled_total?: number;
  unresolved_contradictions_count?: number;
}

export interface UniverseTopCandidate {
  vendor_id: number | null;
  candidate_entity_id: number | null;
  company_name: string;
  official_website_url: string | null;
  discovery_sources: string[];
  entity_type: string;
  decision_classification: string;
  evidence_sufficiency: string;
  reason_codes: {
    positive: string[];
    caution: string[];
    reject: string[];
  };
  rationale_summary: string | null;
  top_claim: {
    text?: string;
    claim_type?: string;
    source_url?: string;
    source_tier?: string;
    source_kind?: string;
    captured_at?: string;
  };
  citation_summary_v1?: CitationSummaryV1 | null;
  registry_neighbors_with_first_party_website_count?: number;
  registry_neighbors_dropped_missing_official_website_count?: number;
  registry_origin_screening_counts?: Record<string, number>;
  first_party_hint_urls_used_count?: number;
  first_party_hint_pages_crawled_total?: number;
  missing_claim_groups: string[];
  unresolved_contradictions_count: number;
  ranking_eligible: boolean;
  run_quality_tier?: "high_quality" | "degraded" | string;
  quality_gate_passed?: boolean;
  degraded_reasons?: string[];
}

export interface VendorDossier {
  id: number;
  vendor_id: number;
  dossier_json: {
    modules?: Array<{
      name: string;
      brick_id: string;
      brick_name?: string;
      description?: string;
      evidence_urls: string[];
    }>;
    customers?: Array<{
      name: string;
      context: string;
      evidence_url: string;
    }>;
    hiring?: {
      postings: Array<{
        title: string;
        location: string;
        category: string;
        evidence_url: string;
      }>;
      mix_summary: {
        engineering_heavy?: boolean;
        team_size_estimate?: string;
        notes?: string;
      };
    };
    integrations?: Array<{
      name: string;
      type: string;
      evidence_url: string;
    }>;
  };
  version: number;
  created_at: string;
}

export interface Job {
  id: number;
  workspace_id: number;
  vendor_id: number | null;
  job_type: string;
  state: "queued" | "running" | "polling" | "completed" | "failed";
  provider: string;
  progress: number;
  progress_message: string | null;
  result_json: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Gates {
  context_pack: boolean;
  brick_model: boolean;
  universe: boolean;
  segmentation: boolean;
  enrichment: boolean;
  missing_items: Record<string, string[]>;
}

export interface LensVendor {
  id: number;
  name: string;
  website: string | null;
  overlapping_bricks: string[];
  added_bricks: string[];
  evidence_count: number;
  customer_overlaps: string[];
  proof_bullets: Array<{ text: string; citation_url: string | null }>;
}

export interface LensResponse {
  vendors: LensVendor[];
  total_count: number;
}

export interface SourcePill {
  label: string;
  url: string;
  document_id?: string | null;
  captured_at?: string | null;
}

export interface SourcedValue {
  value: string;
  unit?: string | null;
  period?: string | null;
  confidence: "high" | "medium" | "low" | string;
  source: SourcePill;
}

export interface ReportClaim {
  text: string;
  confidence: "high" | "medium" | "low" | string;
  rendering: "fact" | "hypothesis";
  source?: SourcePill | null;
}

export interface ReportSnapshot {
  id: number;
  workspace_id: number;
  name: string;
  status: string;
  generated_at: string;
  filters_json: Record<string, unknown>;
  coverage_json: Record<string, unknown>;
  item_count: number;
}

export interface ReportCard {
  vendor_id: number;
  name: string;
  website: string | null;
  hq_country: string | null;
  legal_status: string | null;
  size_bucket: "sme_in_range" | "unknown" | "outside_sme_range" | string;
  size_estimate: number | null;
  size_range_low?: number | null;
  size_range_high?: number | null;
  compete_score: number;
  complement_score: number;
  brick_mapping: ReportClaim[];
  customer_partner_evidence: ReportClaim[];
  filing_metrics: Record<string, SourcedValue>;
  source_pills?: SourcePill[];
  coverage_note: string | null;
  next_validation_questions: string[];
  decision_classification?: "good_target" | "borderline_watchlist" | "not_good_target" | "insufficient_evidence" | string | null;
  reason_highlights?: string[];
  evidence_quality_summary?: Record<string, unknown>;
  known_unknowns?: string[];
}

export interface ReportLensItem {
  vendor_id: number;
  name: string;
  website: string | null;
  size_bucket: string;
  score: number;
  lens_breakdown: Record<string, unknown>;
  highlights: ReportClaim[];
}

export interface ReportLens {
  mode: "compete" | "complement" | string;
  items: ReportLensItem[];
  total_count: number;
  counts_by_bucket: Record<string, number>;
}

export interface ReasonCatalogEntry {
  code: string;
  reason_type: "positive" | "caution" | "reject" | string;
  definition: string;
  min_evidence: string;
  user_text: string;
  source_examples: string[];
}

export interface DecisionCatalog {
  version: string;
  codes: ReasonCatalogEntry[];
}

export interface EvidencePolicyResponse {
  workspace_id: number;
  policy: Record<string, unknown>;
}

export interface VendorDecision {
  vendor_id: number;
  workspace_id: number;
  classification: string;
  evidence_sufficiency: string;
  positive_reason_codes: string[];
  caution_reason_codes: string[];
  reject_reason_codes: string[];
  missing_claim_groups: string[];
  unresolved_contradictions_count: number;
  rationale_summary: string | null;
  rationale_markdown: string | null;
  decision_engine_version: string | null;
  gating_passed: boolean;
  generated_at: string;
}

export interface DecisionQualityDiagnostics {
  workspace_id: number;
  totals: number;
  classification_distribution: Record<string, number>;
  evidence_sufficiency_distribution: Record<string, number>;
  contradiction_rate: number;
  evidence_tier_mix: Record<string, number>;
  freshness_compliance_by_group: Record<string, number>;
  keep_to_later_reject_rate: number;
  analyst_override_rate: number;
  ranking_eligible_count: number;
  directory_only_count: number;
  solution_entity_count: number;
  official_website_resolution_rate: number;
  feedback_events_count: number;
  generated_at: string;
}

export interface ClaimsGraphSummary {
  workspace_id: number;
  nodes_count: number;
  edges_count: number;
  edge_evidence_count: number;
  relation_type_distribution: Record<string, number>;
  node_type_distribution: Record<string, number>;
  generated_at: string;
}

export interface WorkspaceFeedback {
  id: number;
  workspace_id: number;
  vendor_id: number | null;
  screening_id: number | null;
  feedback_type: string;
  previous_classification: string | null;
  new_classification: string | null;
  reason_codes: string[];
  comment: string | null;
  metadata: Record<string, unknown>;
  created_by: string | null;
  created_at: string;
}

export interface EvaluationReplayResult {
  workspace_id: number;
  run_id: number;
  metrics: Record<string, unknown>;
  created_at: string;
}

// ============================================================================
// API Helpers
// ============================================================================

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  // #region agent log
  const startTime = Date.now();
  const fullUrl = `${API_BASE}${url}`;
  fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:157',message:'API request starting',data:{url:fullUrl,method:options?.method||'GET'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
  // #endregion
  
  try {
    const res = await fetch(fullUrl, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });
    
    // #region agent log
    const duration = Date.now() - startTime;
    fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:170',message:'API response received',data:{url:fullUrl,status:res.status,statusText:res.statusText,ok:res.ok,duration},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      // #region agent log
      fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:175',message:'API error response',data:{url:fullUrl,status:res.status,errorDetail:error.detail||error,errorText:res.statusText},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
      // #endregion
      const message =
        typeof error?.detail === "string"
          ? error.detail
          : (error?.detail?.message || res.statusText || "API error");
      const err = new Error(message) as Error & { status?: number; detail?: unknown };
      err.status = res.status;
      err.detail = error?.detail ?? error;
      throw err;
    }
    return res.json();
  } catch (err) {
    // #region agent log
    const errorMsg = err instanceof Error ? err.message : String(err);
    const errorName = err instanceof Error ? err.name : 'Unknown';
    fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:182',message:'API fetch exception',data:{url:fullUrl,errorName,errorMessage:errorMsg,isNetworkError:errorName==='TypeError'||errorMsg.includes('fetch')||errorMsg.includes('network')},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    throw err;
  }
}

// ============================================================================
// Workspace API
// ============================================================================

export const workspaceApi = {
  // Workspaces CRUD
  list: () => fetchJSON<Workspace[]>("/workspaces"),

  get: (id: number) => fetchJSON<Workspace>(`/workspaces/${id}`),

  create: (data: { name: string; region_scope?: string }) =>
    fetchJSON<Workspace>("/workspaces", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (id: number, data: { name?: string; region_scope?: string }) =>
    fetchJSON<Workspace>(`/workspaces/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  delete: (id: number) =>
    fetchJSON<{ deleted: boolean }>(`/workspaces/${id}`, {
      method: "DELETE",
    }),

  // Context Pack
  getContextPack: (id: number) =>
    fetchJSON<CompanyProfile>(`/workspaces/${id}/context-pack`),

  updateContextPack: (
    id: number,
    data: {
      buyer_company_url?: string;
      reference_vendor_urls?: string[];
      reference_evidence_urls?: string[];
      geo_scope?: GeoScope;
    }
  ) =>
    fetchJSON<CompanyProfile>(`/workspaces/${id}/context-pack`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  refreshContextPack: (id: number) =>
    fetchJSON<Job>(`/workspaces/${id}/context-pack:refresh`, {
      method: "POST",
    }),

  // Bricks
  getBricks: (id: number) =>
    fetchJSON<BrickTaxonomy>(`/workspaces/${id}/bricks`),

  updateBricks: (
    id: number,
    data: { bricks?: BrickItem[]; priority_brick_ids?: string[]; vertical_focus?: string[] }
  ) =>
    fetchJSON<BrickTaxonomy>(`/workspaces/${id}/bricks`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  confirmBricks: (id: number) =>
    fetchJSON<BrickTaxonomy>(`/workspaces/${id}/bricks:confirm`, {
      method: "POST",
    }),

  // Discovery
  runDiscovery: (id: number) =>
    fetchJSON<Job>(`/workspaces/${id}/discovery:run`, {
      method: "POST",
    }),

  getDiscoveryDiagnostics: (id: number) =>
    fetchJSON<Record<string, unknown>>(`/workspaces/${id}/discovery:diagnostics`),
  getTopCandidates: (id: number, limit = 25, allowDegraded = false) =>
    fetchJSON<UniverseTopCandidate[]>(
      `/workspaces/${id}/universe/top-candidates?limit=${encodeURIComponent(String(limit))}&allow_degraded=${allowDegraded ? "true" : "false"}`
    ),

  // Vendors
  listVendors: (id: number, status?: string) =>
    fetchJSON<Vendor[]>(
      `/workspaces/${id}/vendors${status ? `?status=${status}` : ""}`
    ),

  createVendor: (
    id: number,
    data: {
      name: string;
      website?: string;
      hq_country?: string;
      tags_vertical?: string[];
    }
  ) =>
    fetchJSON<Vendor>(`/workspaces/${id}/vendors`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  updateVendor: (
    workspaceId: number,
    vendorId: number,
    data: {
      name?: string;
      website?: string;
      hq_country?: string;
      operating_countries?: string[];
      tags_vertical?: string[];
      tags_custom?: string[];
      status?: string;
    }
  ) =>
    fetchJSON<Vendor>(`/workspaces/${workspaceId}/vendors/${vendorId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  // Enrichment
  enrichVendors: (
    id: number,
    data: { vendor_ids: number[]; job_types?: string[] }
  ) =>
    fetchJSON<Job[]>(`/workspaces/${id}/vendors:enrich`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  getVendorDossier: (workspaceId: number, vendorId: number) =>
    fetchJSON<VendorDossier | null>(
      `/workspaces/${workspaceId}/vendors/${vendorId}/dossier`
    ),

  // Lenses
  getSimilarityLens: (id: number, brickIds?: string) =>
    fetchJSON<LensResponse>(
      `/workspaces/${id}/lenses/similarity${brickIds ? `?brick_ids=${brickIds}` : ""}`
    ),

  getComplementarityLens: (id: number) =>
    fetchJSON<LensResponse>(`/workspaces/${id}/lenses/complementarity`),

  // Static Reports
  generateReport: (
    id: number,
    data?: { name?: string; include_unknown_size?: boolean; include_outside_sme?: boolean }
  ) =>
    fetchJSON<Job>(`/workspaces/${id}/reports:generate`, {
      method: "POST",
      body: JSON.stringify(data ?? {}),
    }),

  listReports: (id: number) =>
    fetchJSON<ReportSnapshot[]>(`/workspaces/${id}/reports`),

  getReport: (workspaceId: number, reportId: number) =>
    fetchJSON<ReportSnapshot>(`/workspaces/${workspaceId}/reports/${reportId}`),

  listReportCards: (
    workspaceId: number,
    reportId: number,
    sizeBucket?: "sme_in_range" | "unknown" | "outside_sme_range"
  ) =>
    fetchJSON<ReportCard[]>(
      `/workspaces/${workspaceId}/reports/${reportId}/cards${
        sizeBucket ? `?size_bucket=${sizeBucket}` : ""
      }`
    ),

  getReportLens: (
    workspaceId: number,
    reportId: number,
    mode: "compete" | "complement"
  ) =>
    fetchJSON<ReportLens>(
      `/workspaces/${workspaceId}/reports/${reportId}/lenses?mode=${mode}`
    ),

  exportReport: (workspaceId: number, reportId: number, format: "default" | "rich_json" = "default") =>
    fetchJSON<Record<string, unknown>>(
      `/workspaces/${workspaceId}/reports/${reportId}/export?format=${format}`
    ),

  // Gates
  getGates: (id: number) => fetchJSON<Gates>(`/workspaces/${id}/gates`),

  // Decisioning / Policy
  getDecisionCatalog: (workspaceId: number) =>
    fetchJSON<DecisionCatalog>(`/workspaces/${workspaceId}/decision-catalog`),

  getEvidencePolicy: (workspaceId: number) =>
    fetchJSON<EvidencePolicyResponse>(`/workspaces/${workspaceId}/evidence-policy`),

  updateEvidencePolicy: (workspaceId: number, policy: Record<string, unknown>) =>
    fetchJSON<EvidencePolicyResponse>(`/workspaces/${workspaceId}/evidence-policy`, {
      method: "PATCH",
      body: JSON.stringify({ policy }),
    }),

  getVendorDecision: (workspaceId: number, vendorId: number) =>
    fetchJSON<VendorDecision>(`/workspaces/${workspaceId}/vendors/${vendorId}/decision`),

  getDecisionQualityDiagnostics: (workspaceId: number) =>
    fetchJSON<DecisionQualityDiagnostics>(`/workspaces/${workspaceId}/diagnostics/decision-quality`),

  runMonitoring: (
    workspaceId: number,
    data?: {
      max_vendors?: number;
      stale_only?: boolean;
      classifications?: string[];
    }
  ) =>
    fetchJSON<Job>(`/workspaces/${workspaceId}/monitoring:run`, {
      method: "POST",
      body: JSON.stringify(data ?? {}),
    }),

  getClaimsGraph: (workspaceId: number) =>
    fetchJSON<ClaimsGraphSummary>(`/workspaces/${workspaceId}/claims-graph`),

  refreshClaimsGraph: (workspaceId: number) =>
    fetchJSON<{ workspace_id: number; queued: boolean; task_id: string; generated_at: string }>(
      `/workspaces/${workspaceId}/claims-graph:refresh`,
      { method: "POST" }
    ),

  listFeedback: (workspaceId: number, limit = 100) =>
    fetchJSON<WorkspaceFeedback[]>(
      `/workspaces/${workspaceId}/feedback?limit=${encodeURIComponent(String(limit))}`
    ),

  createFeedback: (
    workspaceId: number,
    data: {
      vendor_id?: number;
      screening_id?: number;
      feedback_type?: string;
      previous_classification?: string;
      new_classification?: string;
      reason_codes?: string[];
      comment?: string;
      metadata?: Record<string, unknown>;
      created_by?: string;
    }
  ) =>
    fetchJSON<WorkspaceFeedback>(`/workspaces/${workspaceId}/feedback`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  replayEvaluation: (
    workspaceId: number,
    data: {
      model_version?: string;
      samples: Array<Record<string, unknown>>;
    }
  ) =>
    fetchJSON<EvaluationReplayResult>(`/workspaces/${workspaceId}/evaluations/replay`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  listEvaluations: (workspaceId: number, limit = 20) =>
    fetchJSON<
      Array<{
        id: number;
        run_type: string;
        status: string;
        model_version: string | null;
        metrics: Record<string, unknown>;
        created_at: string | null;
      }>
    >(`/workspaces/${workspaceId}/evaluations?limit=${encodeURIComponent(String(limit))}`),

  // Jobs
  listJobs: (id: number, jobType?: string, state?: string) => {
    const params = new URLSearchParams();
    if (jobType) params.set("job_type", jobType);
    if (state) params.set("state", state);
    const query = params.toString();
    return fetchJSON<Job[]>(`/workspaces/${id}/jobs${query ? `?${query}` : ""}`);
  },

  getJob: (workspaceId: number, jobId: number) =>
    fetchJSON<Job>(`/workspaces/${workspaceId}/jobs/${jobId}`),
};
