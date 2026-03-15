"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  workspaceApi,
  Job,
  GeoScope,
  TaxonomyNode,
} from "./api";
import { useEffect, useRef, useState } from "react";

// ============================================================================
// Workspace Hooks
// ============================================================================

export function useWorkspaces() {
  return useQuery({
    queryKey: ["workspaces"],
    queryFn: () => workspaceApi.list(),
  });
}

export function useWorkspace(id: number) {
  return useQuery({
    queryKey: ["workspace", id],
    queryFn: () => workspaceApi.get(id),
    enabled: !!id,
  });
}

export function useCreateWorkspace() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: workspaceApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}

export function useDeleteWorkspace() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: workspaceApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}

// Context Pack
export function useContextPack(workspaceId: number) {
  return useQuery({
    queryKey: ["context-pack", workspaceId],
    queryFn: () => workspaceApi.getContextPack(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useUpdateContextPack(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      buyer_company_url?: string | null;
      comparator_seed_urls?: string[];
      supporting_evidence_urls?: string[];
      geo_scope?: GeoScope;
    }) => workspaceApi.updateContextPack(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["context-pack", workspaceId] });
    },
  });
}

export function useRefreshContextPack(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => workspaceApi.refreshContextPack(workspaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["context-pack", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
    },
  });
}

// Company Context
export function useCompanyContextPack(workspaceId: number) {
  return useQuery({
    queryKey: ["company-context", workspaceId],
    queryFn: () => workspaceApi.getCompanyContext(workspaceId),
    enabled: !!workspaceId,
    refetchInterval: (query) => {
      const status = (query.state.data as { graph_status?: string } | undefined)?.graph_status;
      return status === "refreshing" ? 3000 : false;
    },
  });
}

export function useUpdateCompanyContextPack(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      source_summary?: string | null;
      taxonomy_nodes?: TaxonomyNode[];
      confirmed?: boolean;
    }) => workspaceApi.updateCompanyContext(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["company-context", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["scope-review", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
    },
  });
}

export function useRefreshCompanyContextPack(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => workspaceApi.refreshCompanyContext(workspaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["company-context", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["scope-review", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
    },
  });
}

// Scope Review
export function useScopeReview(workspaceId: number) {
  return useQuery({
    queryKey: ["scope-review", workspaceId],
    queryFn: () => workspaceApi.getScopeReview(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useUpdateScopeReview(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { decisions: Array<{ id: string; status: string }> }) =>
      workspaceApi.updateScopeReview(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["scope-review", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
    },
  });
}

export function useConfirmScopeReview(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => workspaceApi.confirmScopeReview(workspaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["scope-review", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["company-context", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
    },
  });
}

// Discovery & Companies
export function useRunDiscovery(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => workspaceApi.runDiscovery(workspaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["companies", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
    },
  });
}

export function useCompanies(workspaceId: number, status?: string) {
  return useQuery({
    queryKey: ["companies", workspaceId, status],
    queryFn: () => workspaceApi.listCompanies(workspaceId, status),
    enabled: !!workspaceId,
  });
}

export function useTopCandidates(workspaceId: number, limit = 25, allowDegraded = false) {
  return useQuery({
    queryKey: ["top-candidates", workspaceId, limit, allowDegraded],
    queryFn: () => workspaceApi.getTopCandidates(workspaceId, limit, allowDegraded),
    enabled: !!workspaceId,
  });
}

export function useCreateCompany(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      name: string;
      website?: string;
      hq_country?: string;
    }) => workspaceApi.createCompany(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["companies", workspaceId] });
    },
  });
}

export function useUpdateCompany(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      companyId,
      data,
    }: {
      companyId: number;
      data: {
        name?: string;
        website?: string;
        hq_country?: string;
        operating_countries?: string[];
        tags_custom?: string[];
        status?: string;
      };
    }) => workspaceApi.updateCompany(workspaceId, companyId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["companies", workspaceId] });
    },
  });
}

// Enrichment
export function useEnrichCompanies(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { company_ids: number[]; job_types?: string[] }) =>
      workspaceApi.enrichCompanies(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["companies", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
    },
  });
}

export function useCompanyDossier(workspaceId: number, companyId: number) {
  return useQuery({
    queryKey: ["company-dossier", companyId],
    queryFn: () => workspaceApi.getCompanyDossier(workspaceId, companyId),
    enabled: !!workspaceId && !!companyId,
  });
}

// Static Reports
export function useGenerateReport(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data?: { name?: string; include_unknown_size?: boolean }) =>
      workspaceApi.generateReport(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["reports", workspaceId] });
    },
  });
}

export function useReports(workspaceId: number) {
  return useQuery({
    queryKey: ["reports", workspaceId],
    queryFn: () => workspaceApi.listReports(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useReport(workspaceId: number, reportId: number | null) {
  return useQuery({
    queryKey: ["report", workspaceId, reportId],
    queryFn: () => workspaceApi.getReport(workspaceId, reportId!),
    enabled: !!workspaceId && !!reportId,
  });
}

export function useReportCards(
  workspaceId: number,
  reportId: number | null,
  sizeBucket?: "sme_in_range" | "unknown" | "outside_sme_range"
) {
  return useQuery({
    queryKey: ["report-cards", workspaceId, reportId, sizeBucket],
    queryFn: () => workspaceApi.listReportCards(workspaceId, reportId!, sizeBucket),
    enabled: !!workspaceId && !!reportId,
  });
}

// Gates
export function useGates(workspaceId: number) {
  return useQuery({
    queryKey: ["gates", workspaceId],
    queryFn: () => workspaceApi.getGates(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useDecisionCatalog(workspaceId: number) {
  return useQuery({
    queryKey: ["decision-catalog", workspaceId],
    queryFn: () => workspaceApi.getDecisionCatalog(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useEvidencePolicy(workspaceId: number) {
  return useQuery({
    queryKey: ["evidence-policy", workspaceId],
    queryFn: () => workspaceApi.getEvidencePolicy(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useUpdateEvidencePolicy(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (policy: Record<string, unknown>) =>
      workspaceApi.updateEvidencePolicy(workspaceId, policy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["evidence-policy", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
    },
  });
}

export function useCompanyDecision(workspaceId: number, companyId: number | null) {
  return useQuery({
    queryKey: ["company-decision", workspaceId, companyId],
    queryFn: () => workspaceApi.getCompanyDecision(workspaceId, companyId!),
    enabled: !!workspaceId && !!companyId,
  });
}

export function useDecisionQualityDiagnostics(workspaceId: number) {
  return useQuery({
    queryKey: ["decision-quality", workspaceId],
    queryFn: () => workspaceApi.getDecisionQualityDiagnostics(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useRunMonitoring(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data?: { max_companies?: number; stale_only?: boolean; classifications?: string[] }) =>
      workspaceApi.runMonitoring(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["decision-quality", workspaceId] });
    },
  });
}

export function useClaimsGraph(workspaceId: number) {
  return useQuery({
    queryKey: ["claims-graph", workspaceId],
    queryFn: () => workspaceApi.getClaimsGraph(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useRefreshClaimsGraph(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => workspaceApi.refreshClaimsGraph(workspaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["claims-graph", workspaceId] });
    },
  });
}

export function useWorkspaceFeedback(workspaceId: number, limit = 100) {
  return useQuery({
    queryKey: ["workspace-feedback", workspaceId, limit],
    queryFn: () => workspaceApi.listFeedback(workspaceId, limit),
    enabled: !!workspaceId,
  });
}

export function useCreateWorkspaceFeedback(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      company_id?: number;
      company_screening_id?: number;
      feedback_type?: string;
      previous_classification?: string;
      new_classification?: string;
      reason_codes?: string[];
      comment?: string;
      metadata?: Record<string, unknown>;
      created_by?: string;
    }) => workspaceApi.createFeedback(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspace-feedback", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["companies", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["decision-quality", workspaceId] });
    },
  });
}

export function useReplayEvaluation(workspaceId: number) {
  return useMutation({
    mutationFn: (data: { model_version?: string; samples: Array<Record<string, unknown>> }) =>
      workspaceApi.replayEvaluation(workspaceId, data),
  });
}

export function useEvaluations(workspaceId: number, limit = 20) {
  return useQuery({
    queryKey: ["evaluations", workspaceId, limit],
    queryFn: () => workspaceApi.listEvaluations(workspaceId, limit),
    enabled: !!workspaceId,
  });
}

// Workspace Jobs
export function useWorkspaceJobs(workspaceId: number, jobType?: string, state?: string) {
  return useQuery({
    queryKey: ["workspace-jobs", workspaceId, jobType, state],
    queryFn: () => workspaceApi.listJobs(workspaceId, jobType, state),
    enabled: !!workspaceId,
    refetchInterval: (query) => {
      const jobs = query.state.data;
      if (jobs?.some((j) => j.state === "queued" || j.state === "running")) {
        return 2000;
      }
      return false;
    },
  });
}

export function useWorkspaceJob(workspaceId: number, jobId: number | null) {
  return useQuery({
    queryKey: ["workspace-job", jobId],
    queryFn: () => workspaceApi.getJob(workspaceId, jobId!),
    enabled: !!workspaceId && !!jobId,
    refetchInterval: (query) => {
      const job = query.state.data;
      if (job && (job.state === "queued" || job.state === "running")) {
        return 2000;
      }
      return false;
    },
  });
}

// Combined hook for workspace job with polling
export function useWorkspaceJobWithPolling(
  workspaceId: number,
  runMutation: () => Promise<Job>,
  onComplete?: () => void,
  cancelMutation?: (jobId: number) => Promise<Job>
) {
  const [jobId, setJobId] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const handledTerminalJobId = useRef<number | null>(null);
  const queryClient = useQueryClient();

  const jobQuery = useWorkspaceJob(workspaceId, jobId);

  useEffect(() => {
    if (!jobId) return;

    if (jobQuery.data?.state === "completed" || jobQuery.data?.state === "failed") {
      if (handledTerminalJobId.current === jobId) return;

      handledTerminalJobId.current = jobId;
      setIsRunning(false);
      if (jobQuery.data?.state === "completed") {
        queryClient.invalidateQueries({ queryKey: ["context-pack", workspaceId] });
        queryClient.invalidateQueries({ queryKey: ["company-context", workspaceId] });
        queryClient.invalidateQueries({ queryKey: ["scope-review", workspaceId] });
        queryClient.invalidateQueries({ queryKey: ["companies", workspaceId] });
        queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
        onComplete?.();
      }
    }
  }, [jobId, jobQuery.data?.state, onComplete, queryClient, workspaceId]);

  const run = async () => {
    setIsRunning(true);
    handledTerminalJobId.current = null;
    try {
      const result = await runMutation();
      setJobId(result.id);
    } catch (error) {
      setIsRunning(false);
      throw error;
    }
  };

  const stop = async () => {
    if (!jobId || !cancelMutation) return;
    setIsStopping(true);
    try {
      await cancelMutation(jobId);
      await jobQuery.refetch();
    } finally {
      setIsStopping(false);
      setIsRunning(false);
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspace-job", jobId] });
    }
  };

  return {
    run,
    stop,
    isRunning,
    isStopping,
    canStop:
      !!cancelMutation &&
      !!jobId &&
      (jobQuery.data?.state === "queued" ||
        jobQuery.data?.state === "running" ||
        jobQuery.data?.state === "polling" ||
        isRunning),
    job: jobQuery.data,
    jobError: jobQuery.data?.error_message,
    progress: jobQuery.data?.progress ?? 0,
    progressMessage: jobQuery.data?.progress_message,
    reset: () => {
      setJobId(null);
      setIsRunning(false);
    },
  };
}
