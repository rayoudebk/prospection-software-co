"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  workspaceApi,
  Workspace,
  CompanyProfile,
  BrickTaxonomy,
  BrickItem,
  Vendor,
  VendorDossier,
  Job,
  Gates,
  LensResponse,
  GeoScope,
  ReportSnapshot,
  ReportCard,
  ReportLens,
} from "./api";
import { useEffect, useState } from "react";

// ============================================================================
// Workspace Hooks
// ============================================================================

export function useWorkspaces() {
  return useQuery({
    queryKey: ["workspaces"],
    queryFn: async () => {
      // #region agent log
      try {
        fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'hooks.ts:24',message:'useWorkspaces queryFn called',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
        const result = await workspaceApi.list();
        fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'hooks.ts:27',message:'useWorkspaces success',data:{resultCount:result?.length||0},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
        return result;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        fetch('http://127.0.0.1:7243/ingest/b9aef1f8-fb7e-4cf9-8f8f-eaa32841ddf0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'hooks.ts:31',message:'useWorkspaces error caught',data:{errorMessage:errorMsg,errorName:err instanceof Error ? err.name : 'Unknown'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
        throw err;
      }
      // #endregion
    },
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
      buyer_company_url?: string;
      reference_vendor_urls?: string[];
      geo_scope?: GeoScope;
      vertical_focus?: string[];
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

// Bricks
export function useBricks(workspaceId: number) {
  return useQuery({
    queryKey: ["bricks", workspaceId],
    queryFn: () => workspaceApi.getBricks(workspaceId),
    enabled: !!workspaceId,
  });
}

export function useUpdateBricks(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { bricks?: BrickItem[]; priority_brick_ids?: string[]; vertical_focus?: string[] }) =>
      workspaceApi.updateBricks(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["bricks", workspaceId] });
    },
  });
}

export function useConfirmBricks(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => workspaceApi.confirmBricks(workspaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["bricks", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
    },
  });
}

// Discovery & Vendors
export function useRunDiscovery(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => workspaceApi.runDiscovery(workspaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["vendors", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
    },
  });
}

export function useVendors(workspaceId: number, status?: string) {
  return useQuery({
    queryKey: ["vendors", workspaceId, status],
    queryFn: () => workspaceApi.listVendors(workspaceId, status),
    enabled: !!workspaceId,
  });
}

export function useCreateVendor(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      name: string;
      website?: string;
      hq_country?: string;
      tags_vertical?: string[];
    }) => workspaceApi.createVendor(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["vendors", workspaceId] });
    },
  });
}

export function useUpdateVendor(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      vendorId,
      data,
    }: {
      vendorId: number;
      data: {
        name?: string;
        website?: string;
        hq_country?: string;
        operating_countries?: string[];
        tags_vertical?: string[];
        tags_custom?: string[];
        status?: string;
      };
    }) => workspaceApi.updateVendor(workspaceId, vendorId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["vendors", workspaceId] });
    },
  });
}

// Enrichment
export function useEnrichVendors(workspaceId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { vendor_ids: number[]; job_types?: string[] }) =>
      workspaceApi.enrichVendors(workspaceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["vendors", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspace-jobs", workspaceId] });
    },
  });
}

export function useVendorDossier(workspaceId: number, vendorId: number) {
  return useQuery({
    queryKey: ["vendor-dossier", vendorId],
    queryFn: () => workspaceApi.getVendorDossier(workspaceId, vendorId),
    enabled: !!workspaceId && !!vendorId,
  });
}

// Lenses
export function useSimilarityLens(workspaceId: number, brickIds?: string) {
  return useQuery({
    queryKey: ["similarity-lens", workspaceId, brickIds],
    queryFn: () => workspaceApi.getSimilarityLens(workspaceId, brickIds),
    enabled: !!workspaceId,
  });
}

export function useComplementarityLens(workspaceId: number) {
  return useQuery({
    queryKey: ["complementarity-lens", workspaceId],
    queryFn: () => workspaceApi.getComplementarityLens(workspaceId),
    enabled: !!workspaceId,
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

export function useReportLens(
  workspaceId: number,
  reportId: number | null,
  mode: "compete" | "complement"
) {
  return useQuery({
    queryKey: ["report-lens", workspaceId, reportId, mode],
    queryFn: () => workspaceApi.getReportLens(workspaceId, reportId!, mode),
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
  onComplete?: () => void
) {
  const [jobId, setJobId] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const queryClient = useQueryClient();

  const jobQuery = useWorkspaceJob(workspaceId, jobId);

  useEffect(() => {
    if (jobQuery.data?.state === "completed" || jobQuery.data?.state === "failed") {
      setIsRunning(false);
      if (jobQuery.data?.state === "completed") {
        queryClient.invalidateQueries({ queryKey: ["context-pack", workspaceId] });
        queryClient.invalidateQueries({ queryKey: ["vendors", workspaceId] });
        queryClient.invalidateQueries({ queryKey: ["gates", workspaceId] });
        onComplete?.();
      }
    }
  }, [jobQuery.data?.state, onComplete, queryClient, workspaceId]);

  const run = async () => {
    setIsRunning(true);
    try {
      const result = await runMutation();
      setJobId(result.id);
    } catch (error) {
      setIsRunning(false);
      throw error;
    }
  };

  return {
    run,
    isRunning,
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
