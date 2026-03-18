const LOCAL_API_URL = "http://localhost:8000";

function resolveApiBase() {
  const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.API_URL;
  if (configuredApiUrl) {
    return configuredApiUrl.replace(/\/$/, "");
  }

  if (typeof window !== "undefined") {
    return "/api";
  }

  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}/api`;
  }

  return LOCAL_API_URL;
}

const API_BASE = resolveApiBase();

// ============================================================================
// Workspace Types
// ============================================================================

export interface Workspace {
  id: number;
  name: string;
  region_scope: string;
  created_at: string;
  company_count: number;
  has_context_pack: boolean;
  has_confirmed_scope_review: boolean;
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
  comparator_seed_urls: string[];
  supporting_evidence_urls: string[];
  comparator_seed_summaries: Record<string, string>;
  geo_scope: GeoScope;
  context_pack_markdown: string | null;
  context_pack_generated_at: string | null;
  product_pages_found: number;
  context_pack_json?: ContextPackV2 | Record<string, unknown> | null;
}

export interface ContextPackEvidenceItem {
  id: string;
  kind: string;
  text: string;
  snippet?: string | null;
  url: string;
  page_type?: string | null;
  page_title?: string | null;
  captured_at?: string | null;
  confidence?: number | null;
}

export interface ContextPackNamedCustomer {
  name: string;
  source_url?: string | null;
  context?: string | null;
  evidence_type?: string | null;
  evidence_id?: string | null;
}

export interface ContextPackIntegration {
  name: string;
  source_url?: string | null;
  evidence_id?: string | null;
}

export interface ContextPackSelectedPage {
  url?: string | null;
  title?: string | null;
  page_type?: string | null;
  headings?: string[];
  has_signals?: boolean;
  has_customer_evidence?: boolean;
}

export interface ContextPackSiteV2 {
  url?: string | null;
  company_name?: string | null;
  website?: string | null;
  summary?: string | null;
  selected_pages?: ContextPackSelectedPage[];
  evidence_items?: ContextPackEvidenceItem[];
  named_customers?: ContextPackNamedCustomer[];
  integrations?: ContextPackIntegration[];
  partners?: ContextPackIntegration[];
  extracted_raw_phrases?: string[];
  crawl_coverage?: Record<string, unknown>;
}

export interface ContextPackV2 {
  version?: string;
  generated_at?: string | null;
  urls_crawled?: string[];
  sites?: ContextPackSiteV2[];
  evidence_items?: ContextPackEvidenceItem[];
  named_customers?: ContextPackNamedCustomer[];
  integrations?: ContextPackIntegration[];
  partners?: ContextPackIntegration[];
  extracted_raw_phrases?: string[];
  crawl_coverage?: {
    total_sites?: number;
    total_pages?: number;
    page_type_counts?: Record<string, number>;
    pages_with_signals?: number;
    pages_with_customer_evidence?: number;
    career_pages_selected?: number;
  };
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

export interface ReportArtifactSourcePill {
  id: string;
  label: string;
  url: string;
  publisher?: string | null;
  publisher_channel: string;
  publisher_type?: string | null;
  source_tier: string;
  source_kind: string;
  evidence_type: string;
  claim_scope?: string | null;
  published_at?: string | null;
  captured_at?: string | null;
}

export interface ReportArtifactSentence {
  id: string;
  text: string;
  citation_pill_ids: string[];
}

export interface ReportArtifactParagraphBlock {
  type: "paragraph";
  sentences: ReportArtifactSentence[];
}

export interface ReportArtifactBulletListBlock {
  type: "bullet_list";
  items: ReportArtifactSentence[];
}

export interface ReportArtifactCalloutBlock {
  type: "callout";
  tone: "info" | "warning" | "success" | "neutral" | string;
  title?: string | null;
  sentences: ReportArtifactSentence[];
}

export interface ReportArtifactKeyValueItem {
  id: string;
  key: string;
  value: string;
  citation_pill_ids: string[];
}

export interface ReportArtifactKeyValueBlock {
  type: "key_value";
  items: ReportArtifactKeyValueItem[];
}

export type ReportArtifactBlock =
  | ReportArtifactParagraphBlock
  | ReportArtifactBulletListBlock
  | ReportArtifactCalloutBlock
  | ReportArtifactKeyValueBlock;

export interface ReportArtifactSection {
  id: string;
  heading?: string | null;
  blocks: ReportArtifactBlock[];
}

export interface ReportArtifact {
  artifact_type: "report_artifact";
  report_kind: "sourcing_brief" | "expansion_brief" | string;
  version: "v1" | string;
  status: "ready" | "degraded" | "pending" | "failed" | string;
  generated_at?: string | null;
  confirmed_at?: string | null;
  reasoning_status?: "success" | "degraded" | "not_run" | "not_applicable" | string;
  reasoning_warning?: string | null;
  reasoning_provider?: string | null;
  reasoning_model?: string | null;
  title: string;
  summary?: string | null;
  sections: ReportArtifactSection[];
  sources: ReportArtifactSourcePill[];
  footer_actions: Array<"copy" | "good" | "bad" | "share" | "regenerate" | string>;
}

export interface BuyerEvidenceDiagnostics {
  mode: string;
  status: string;
  score: number;
  used_for_inference: boolean;
  warning?: string | null;
  metrics: {
    pages_crawled: number;
    content_pages: number;
    signal_count: number;
    customer_evidence_count: number;
    summary_chars: number;
  };
}

export interface TaxonomyNode {
  id: string;
  layer:
    | "customer_archetype"
    | "workflow"
    | "capability"
    | "delivery_or_integration"
    | string;
  phrase: string;
  aliases: string[];
  confidence: number;
  evidence_ids: string[];
  scope_status: "in_scope" | "out_of_scope" | "removed" | string;
}

export interface TaxonomyEdge {
  from_node_id: string;
  to_node_id: string;
  relation_type: string;
  evidence_ids: string[];
}

export interface LensSeed {
  id: string;
  lens_type:
    | "same_customer_different_product"
    | "same_product_different_customer"
    | "different_product_different_customer_within_market_box"
    | string;
  label: string;
  query_phrase?: string | null;
  rationale: string;
  supporting_node_ids: string[];
  evidence_ids: string[];
  confidence: number;
}

export interface SourcingBrief {
  source_company?: {
    name?: string | null;
    website?: string | null;
  };
  source_summary?: string | null;
  reasoning_status?: "success" | "degraded" | "not_run" | "not_applicable" | string;
  reasoning_warning?: string | null;
  reasoning_provider?: string | null;
  reasoning_model?: string | null;
  customer_nodes: TaxonomyNode[];
  workflow_nodes: TaxonomyNode[];
  capability_nodes: TaxonomyNode[];
  delivery_or_integration_nodes: TaxonomyNode[];
  named_customer_proof: ContextPackNamedCustomer[];
  partner_integration_proof: ContextPackIntegration[];
  secondary_evidence_proof: Array<{
    id: string;
    publisher_channel: string;
    publisher_type?: string | null;
    claim_scope?: string | null;
    subject_company?: string | null;
    claim_type: string;
    publisher?: string | null;
    published_at?: string | null;
    claim_text: string;
    evidence_snippet?: string | null;
    url: string;
    title?: string | null;
    entity_mentions: string[];
    supports_node_ids: string[];
    supports_edge_ids: string[];
    confidence: number;
    freshness?: string | null;
    evidence_tier?: string | null;
  }>;
  customer_partner_corroboration: Array<{
    id: string;
    publisher_channel: string;
    publisher_type?: string | null;
    claim_scope?: string | null;
    subject_company?: string | null;
    claim_type: string;
    publisher?: string | null;
    published_at?: string | null;
    claim_text: string;
    evidence_snippet?: string | null;
    url: string;
    title?: string | null;
    entity_mentions: string[];
    supports_node_ids: string[];
    supports_edge_ids: string[];
    confidence: number;
    freshness?: string | null;
    evidence_tier?: string | null;
  }>;
  directory_category_context: Array<{
    id: string;
    publisher_channel: string;
    publisher_type?: string | null;
    claim_scope?: string | null;
    subject_company?: string | null;
    claim_type: string;
    publisher?: string | null;
    published_at?: string | null;
    claim_text: string;
    evidence_snippet?: string | null;
    url: string;
    title?: string | null;
    entity_mentions: string[];
    supports_node_ids: string[];
    supports_edge_ids: string[];
    confidence: number;
    freshness?: string | null;
    evidence_tier?: string | null;
  }>;
  other_secondary_context: Array<{
    id: string;
    publisher_channel: string;
    publisher_type?: string | null;
    claim_scope?: string | null;
    subject_company?: string | null;
    claim_type: string;
    publisher?: string | null;
    published_at?: string | null;
    claim_text: string;
    evidence_snippet?: string | null;
    url: string;
    title?: string | null;
    entity_mentions: string[];
    supports_node_ids: string[];
    supports_edge_ids: string[];
    confidence: number;
    freshness?: string | null;
    evidence_tier?: string | null;
  }>;
  active_lenses: LensSeed[];
  adjacency_hypotheses: Array<{
    id: string;
    text: string;
    supporting_node_ids: string[];
    evidence_ids: string[];
    confidence: number;
  }>;
  strongest_evidence_buckets: Array<{
    label: string;
    count: number;
  }>;
  confidence_gaps: string[];
  open_questions: string[];
  unknowns_not_publicly_resolvable: string[];
  crawl_coverage?: ContextPackV2["crawl_coverage"];
  confirmed_at?: string | null;
}

export interface SourceDocument {
  id: string;
  name: string;
  url?: string | null;
  publisher?: string | null;
  snippet?: string | null;
  publisher_channel: string;
  publisher_type?: string | null;
  claim_scope?: string | null;
  subject_company?: string | null;
  evidence_tier: string;
  evidence_type: string;
}

export interface CompanyContextGraph {
  graph_ref: string;
  workspace_id: number;
  generated_at?: string | null;
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
  source_documents?: SourceDocument[];
  secondary_evidence_proof?: SourcingBrief["secondary_evidence_proof"];
  graph_stats?: Record<string, unknown>;
}

export interface ExpansionInput {
  name: string;
  website: string;
  source_summary?: string | null;
  taxonomy_nodes?: TaxonomyNode[];
  taxonomy_edges?: TaxonomyEdge[];
  named_customer_proof?: ContextPackNamedCustomer[];
  partner_integration_proof?: ContextPackIntegration[];
  crawl_coverage?: Record<string, unknown>;
}

export interface CompanyContextPack {
  id: number;
  workspace_id: number;
  company_context_graph_ref?: string | null;
  company_context_graph_namespace?: string | null;
  graph_status: string;
  graph_warning?: string | null;
  graph_synced_at?: string | null;
  graph_stats: Record<string, unknown>;
  company_context_graph?: CompanyContextGraph | null;
  deep_research_handoff?: Record<string, unknown> | null;
  buyer_evidence?: BuyerEvidenceDiagnostics | null;
  context_pack_v2?: ContextPackV2 | null;
  source_documents: SourceDocument[];
  expansion_inputs: ExpansionInput[];
  taxonomy_nodes: TaxonomyNode[];
  taxonomy_edges: TaxonomyEdge[];
  lens_seeds: LensSeed[];
  sourcing_brief?: SourcingBrief | null;
  sourcing_report?: ReportArtifact | null;
  generated_at: string | null;
  confirmed_at: string | null;
}

export interface ExpansionArtifact {
  id: number;
  workspace_id: number;
  status: string;
  warning?: string | null;
  generated_at?: string | null;
  confirmed_at?: string | null;
  deep_research_handoff?: Record<string, unknown> | null;
  expansion_inputs: ExpansionInput[];
  expansion_brief?: ExpansionBrief | null;
  expansion_report?: ReportArtifact | null;
}

export interface ExpansionBriefItem {
  id: string;
  label: string;
  expansion_type: string;
  status: string;
  confidence: number;
  why_it_matters?: string | null;
  evidence_urls: string[];
  supporting_node_ids: string[];
  source_entity_names: string[];
  market_importance: string;
  operational_centrality: string;
  workflow_criticality: string;
  daily_operator_usage: string;
  switching_cost_intensity: string;
  priority_tier: string;
}

export interface ModelAttemptTrace {
  stage?: string | null;
  provider?: string | null;
  model?: string | null;
  latency_ms: number;
  status?: string | null;
  retry_count: number;
  error_class?: string | null;
  error_message?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
}

export interface ExpansionEvidence {
  url: string;
  title?: string | null;
  publisher?: string | null;
  publisher_type?: string | null;
  language?: string | null;
  evidence_tier?: string | null;
  source_entity_name?: string | null;
  claim_text?: string | null;
  supports: string[];
}

export interface ExpansionEmergingSignal {
  label: string;
  theme_type: string;
  confidence: number;
  why_it_matters?: string | null;
}

export interface ExpansionSourceFit {
  shared_buyers: string[];
  shared_workflows: string[];
  shared_data_objects: string[];
  shared_integrations: string[];
  rationale?: string | null;
}

export interface ExpansionCriticality {
  market_importance: string;
  operational_centrality: string;
  workflow_criticality: string;
  daily_operator_usage: string;
  switching_cost_intensity: string;
  strategic_value_hypothesis?: string | null;
  replicability: string;
  market_density: string;
  adjacency_confidence: number;
  switching_cost_confidence: number;
  trend_confidence: number;
}

export interface ExpansionWorkflowAnatomy {
  primary_operators: string[];
  primary_triggers: string[];
  core_actions: string[];
  systems_touched: string[];
  frequency: string;
  failure_cost?: string | null;
  management_value?: string | null;
}

export interface AdjacencyBox {
  id: string;
  label: string;
  canonical_concept_key: string;
  adjacency_kind: string;
  status: string;
  confidence: number;
  why_it_matters?: string | null;
  source_fit: ExpansionSourceFit;
  criticality: ExpansionCriticality;
  workflow_anatomy: ExpansionWorkflowAnatomy;
  supporting_node_ids: string[];
  related_source_node_ids: string[];
  likely_customer_segments: string[];
  likely_workflows: string[];
  evidence: ExpansionEvidence[];
  emerging_signals: ExpansionEmergingSignal[];
  company_seed_ids: string[];
  retrieval_query_seeds: string[];
  original_language_aliases: string[];
  language_specific_query_seeds: Array<Record<string, unknown>>;
  priority_tier: string;
}

export interface CompanySeed {
  id: string;
  name: string;
  website?: string | null;
  seed_type: string;
  seed_role?: string | null;
  status: string;
  confidence: number;
  why_relevant?: string | null;
  fit_to_adjacency_box_ids: string[];
  evidence: ExpansionEvidence[];
}

export interface TechnologyShiftClaim {
  id: string;
  label: string;
  status: string;
  confidence: number;
  why_it_matters?: string | null;
  affected_adjacency_box_ids: string[];
  company_seed_ids: string[];
  evidence: ExpansionEvidence[];
}

export interface ExpansionBrief {
  version?: string;
  reasoning_status?: "success" | "degraded" | "not_run" | "not_applicable" | string;
  reasoning_warning?: string | null;
  fallback_mode?: boolean;
  research_report_markdown?: string | null;
  research_attempts?: ModelAttemptTrace[];
  normalization_status?: string;
  normalization_warning?: string | null;
  normalization_provider?: string | null;
  normalization_model?: string | null;
  normalization_attempts?: ModelAttemptTrace[];
  reasoning_provider?: string | null;
  reasoning_model?: string | null;
  confirmed_at?: string | null;
  adjacency_boxes?: AdjacencyBox[];
  company_seeds?: CompanySeed[];
  technology_shift_claims?: TechnologyShiftClaim[];
  confidence_gaps?: string[];
  open_questions?: string[];
  named_account_anchors: ExpansionBriefItem[];
  geography_expansions: ExpansionBriefItem[];
}

export interface ScopeReviewItem {
  id: string;
  label: string;
  scope_item_type: string;
  origin: string;
  status: string;
  confidence: number;
  evidence_ids: string[];
  evidence_urls: string[];
  supporting_node_ids: string[];
  source_entity_names: string[];
  why_it_matters?: string | null;
  priority_tier?: string | null;
  market_importance?: string | null;
  operational_centrality?: string | null;
  workflow_criticality?: string | null;
  daily_operator_usage?: string | null;
  switching_cost_intensity?: string | null;
}

export interface ScopeReview {
  workspace_id: number;
  workspace_geo_scope: Record<string, unknown>;
  confirmed_at?: string | null;
  expansion_status: string;
  expansion_warning?: string | null;
  expansion_generated_at?: string | null;
  source_capabilities: ScopeReviewItem[];
  source_customer_segments: ScopeReviewItem[];
  source_workflows: ScopeReviewItem[];
  source_delivery_or_integration: ScopeReviewItem[];
  adjacency_boxes: AdjacencyBox[];
  named_account_anchors: ScopeReviewItem[];
  geography_expansions: ScopeReviewItem[];
}

export interface WhyRelevant {
  text: string;
  citation_url: string;
}

export interface Company {
  id: number;
  workspace_id: number;
  name: string;
  website: string | null;
  official_website_url?: string | null;
  discovery_url?: string | null;
  entity_type?: "company" | "solution" | "service_line" | string | null;
  hq_country: string | null;
  operating_countries: string[];
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
  why_fit_bullets?: Array<{ text: string; citation_url?: string | null }>;
  business_model_signal?: string | null;
  customer_proof?: string[];
  employee_signal?: string | null;
  open_questions?: string[];
}

export interface UniverseTopCandidate {
  company_id: number | null;
  candidate_entity_id: number | null;
  company_name: string;
  company_status?: "candidate" | "kept" | "removed" | "enriched" | string | null;
  official_website_url: string | null;
  hq_country?: string | null;
  evidence_count: number;
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
  capability_signals: string[];
  likely_verticals: string[];
  scope_buckets: string[];
  origin_types: string[];
  registry_identity?: {
    id?: string | null;
    country?: string | null;
    source?: string | null;
    match_confidence?: number | null;
    matched_query?: string | null;
    status?: string | null;
  } | null;
  expansion_provenance: Array<{
    query_id?: string | null;
    query_type?: string | null;
    query_text?: string | null;
    provider?: string | null;
    brick_name?: string | null;
    scope_bucket?: string | null;
    rank?: number | null;
  }>;
  missing_claim_groups: string[];
  unresolved_contradictions_count: number;
  ranking_eligible: boolean;
  run_quality_tier?: "high_quality" | "degraded" | string;
  quality_gate_passed?: boolean;
  quality_audit_passed?: boolean;
  degraded_reasons?: string[];
}

export interface DiscoveryDiagnostics {
  workspace_id: number;
  screening_run_id?: string | null;
  screening_totals?: {
    screenings: number;
    kept: number;
    review: number;
    rejected: number;
  };
  source_coverage?: {
    external_search?: {
      provider_mix?: Record<string, number>;
      brick_yield?: Record<string, number>;
      query_plan_summary?: {
        scope_buckets?: string[];
      };
    };
  };
  funnel_metrics?: {
    seed_directory_count?: number;
    seed_reference_count?: number;
    seed_llm_count?: number;
    final_universe_count?: number;
    registry_identity_candidates_count?: number;
    registry_identity_mapped_count?: number;
    registry_queries_count?: number;
    registry_neighbors_kept_count?: number;
    registry_queries_by_country?: Record<string, number>;
    first_party_crawl_success_count?: number;
    first_party_crawl_pages_total?: number;
  };
}

export interface CompanyDossier {
  id: number;
  company_id: number;
  dossier_json: {
    workflow?: Array<{
      text: string;
      evidence_url?: string | null;
    }>;
    customer?: Array<{
      text: string;
      evidence_url?: string | null;
    }>;
    business_model?: Array<{
      text: string;
      evidence_url?: string | null;
    }>;
    ownership?: Array<{
      text: string;
      evidence_url?: string | null;
    }>;
    transaction_feasibility?: Array<{
      text: string;
      evidence_url?: string | null;
    }>;
    kpis?: Record<
      string,
      | {
          value: string;
          unit?: string | null;
          period?: string | null;
          confidence?: string | null;
          evidence_url?: string | null;
        }
      | null
    >;
    modules?: Array<{
      name: string;
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
  company_id: number | null;
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

export interface DiscoveryReadiness {
  runnable: boolean;
  execution_mode: "live" | "fixture" | string;
  expansion_confirmed: boolean;
  db_schema_ok: boolean;
  redis_available: boolean;
  worker_available: boolean;
  retrieval_provider_available: boolean;
  model_available: boolean;
  reasons_blocked: string[];
  available_retrieval_providers: string[];
  available_model_providers: string[];
  schema_missing_columns: string[];
}

export interface Gates {
  context_pack: boolean;
  scope_review: boolean;
  universe: boolean;
  segmentation: boolean;
  enrichment: boolean;
  missing_items: Record<string, string[]>;
  discovery_readiness: DiscoveryReadiness;
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
  company_id: number;
  name: string;
  website: string | null;
  hq_country: string | null;
  legal_status: string | null;
  size_bucket: "sme_in_range" | "unknown" | "outside_sme_range" | string;
  size_estimate: number | null;
  size_range_low?: number | null;
  size_range_high?: number | null;
  fit_score: number;
  evidence_score: number;
  workflow_profile: ReportClaim[];
  customer_profile: ReportClaim[];
  business_model_profile: ReportClaim[];
  ownership_profile: ReportClaim[];
  transaction_profile: ReportClaim[];
  filing_metrics: Record<string, SourcedValue>;
  source_pills?: SourcePill[];
  coverage_note: string | null;
  next_validation_questions: string[];
  decision_classification?: "good_target" | "borderline_watchlist" | "not_good_target" | "insufficient_evidence" | string | null;
  reason_highlights?: string[];
  evidence_quality_summary?: Record<string, unknown>;
  known_unknowns?: string[];
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

export interface CompanyDecision {
  company_id: number;
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
  company_id: number | null;
  company_screening_id: number | null;
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
  const fullUrl = `${API_BASE}${url}`;
  const method = options?.method?.toUpperCase() || "GET";
  
  try {
    const res = await fetch(fullUrl, {
      ...options,
      cache: method === "GET" ? "no-store" : options?.cache,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });
    
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
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
      buyer_company_url?: string | null;
      comparator_seed_urls?: string[];
      supporting_evidence_urls?: string[];
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

  // Company Context
  getCompanyContext: (id: number) =>
    fetchJSON<CompanyContextPack>(`/workspaces/${id}/company-context`),

  updateCompanyContext: (
    id: number,
    data: {
      source_summary?: string | null;
      taxonomy_nodes?: TaxonomyNode[];
      confirmed?: boolean;
    }
  ) =>
    fetchJSON<CompanyContextPack>(`/workspaces/${id}/company-context`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  refreshCompanyContext: (id: number) =>
    fetchJSON<CompanyContextPack>(`/workspaces/${id}/company-context:refresh`, {
      method: "POST",
    }),
  getExpansionBrief: (id: number) =>
    fetchJSON<ExpansionArtifact>(`/workspaces/${id}/expansion-brief`),
  generateExpansionBrief: (id: number) =>
    fetchJSON<ExpansionArtifact>(`/workspaces/${id}/expansion-brief:generate`, {
      method: "POST",
    }),
  // Scope Review
  getScopeReview: (id: number) =>
    fetchJSON<ScopeReview>(`/workspaces/${id}/scope-review`),

  updateScopeReview: (
    id: number,
    data: {
      decisions: Array<{ id: string; status: string }>;
    }
  ) =>
    fetchJSON<ScopeReview>(`/workspaces/${id}/scope-review`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  confirmScopeReview: (id: number) =>
    fetchJSON<ScopeReview>(`/workspaces/${id}/scope-review:confirm`, {
      method: "POST",
    }),

  // Discovery
  runDiscovery: (id: number) =>
    fetchJSON<Job>(`/workspaces/${id}/discovery:run`, {
      method: "POST",
    }),

  getDiscoveryDiagnostics: (id: number, includeQualityAudit = true) =>
    fetchJSON<DiscoveryDiagnostics>(
      `/workspaces/${id}/discovery:diagnostics?include_quality_audit=${includeQualityAudit ? "true" : "false"}`
    ),
  getTopCandidates: (id: number, limit = 25, allowDegraded = false) =>
    fetchJSON<UniverseTopCandidate[]>(
      `/workspaces/${id}/universe/top-candidates?limit=${encodeURIComponent(String(limit))}&allow_degraded=${allowDegraded ? "true" : "false"}`
    ),

  // Companies
  listCompanies: (id: number, status?: string) =>
    fetchJSON<Company[]>(
      `/workspaces/${id}/companies${status ? `?status=${status}` : ""}`
    ),

  createCompany: (
    id: number,
    data: {
      name: string;
      website?: string;
      hq_country?: string;
    }
  ) =>
    fetchJSON<Company>(`/workspaces/${id}/companies`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  updateCompany: (
    workspaceId: number,
    companyId: number,
    data: {
      name?: string;
      website?: string;
      hq_country?: string;
      operating_countries?: string[];
      tags_custom?: string[];
      status?: string;
    }
  ) =>
    fetchJSON<Company>(`/workspaces/${workspaceId}/companies/${companyId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  // Enrichment
  enrichCompanies: (
    id: number,
    data: { company_ids: number[]; job_types?: string[] }
  ) =>
    fetchJSON<Job[]>(`/workspaces/${id}/companies:enrich`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  getCompanyDossier: (workspaceId: number, companyId: number) =>
    fetchJSON<CompanyDossier | null>(
      `/workspaces/${workspaceId}/companies/${companyId}/dossier`
    ),

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

  getCompanyDecision: (workspaceId: number, companyId: number) =>
    fetchJSON<CompanyDecision>(`/workspaces/${workspaceId}/companies/${companyId}/decision`),

  getDecisionQualityDiagnostics: (workspaceId: number) =>
    fetchJSON<DecisionQualityDiagnostics>(`/workspaces/${workspaceId}/diagnostics/decision-quality`),

  runMonitoring: (
    workspaceId: number,
    data?: {
      max_companies?: number;
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
      company_id?: number;
      company_screening_id?: number;
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

  cancelJob: (workspaceId: number, jobId: number) =>
    fetchJSON<Job>(`/workspaces/${workspaceId}/jobs/${jobId}:cancel`, {
      method: "POST",
    }),
};
