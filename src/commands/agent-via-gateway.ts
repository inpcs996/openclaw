import fs from "node:fs";
import { resolveSendableOutboundReplyParts } from "openclaw/plugin-sdk/reply-payload";
import { listAgentIds } from "../agents/agent-scope.js";
import { formatCliCommand } from "../cli/command-format.js";
import type { CliDeps } from "../cli/deps.js";
import { withProgress } from "../cli/progress.js";
import { loadConfig } from "../config/config.js";
import { callGateway, randomIdempotencyKey } from "../gateway/call.js";
import { normalizeAgentId } from "../routing/session-key.js";
import { type RuntimeEnv, writeRuntimeJson } from "../runtime.js";
import {
  GATEWAY_CLIENT_MODES,
  GATEWAY_CLIENT_NAMES,
  normalizeMessageChannel,
} from "../utils/message-channel.js";
import { agentCommand } from "./agent.js";
import { resolveSessionKeyForRequest } from "./agent/session.js";

type AgentGatewayResult = {
  payloads?: Array<{
    text?: string;
    mediaUrl?: string | null;
    mediaUrls?: string[];
  }>;
  meta?: unknown;
};

type GatewayAgentResponse = {
  runId?: string;
  status?: string;
  summary?: string;
  result?: AgentGatewayResult;
};

const NO_GATEWAY_TIMEOUT_MS = 2_147_000_000;

export type AgentCliOpts = {
  message: string;
  agent?: string;
  to?: string;
  sessionId?: string;
  sessionKey?: string;
  thinking?: string;
  verbose?: string;
  json?: boolean;
  timeout?: string;
  deliver?: boolean;
  channel?: string;
  replyTo?: string;
  replyChannel?: string;
  replyAccount?: string;
  bestEffortDeliver?: boolean;
  lane?: string;
  runId?: string;
  extraSystemPrompt?: string;
  local?: boolean;
};

function parseTimeoutSeconds(opts: { cfg: ReturnType<typeof loadConfig>; timeout?: string }) {
  const raw =
    opts.timeout !== undefined
      ? Number.parseInt(String(opts.timeout), 10)
      : (opts.cfg.agents?.defaults?.timeoutSeconds ?? 600);
  if (Number.isNaN(raw) || raw < 0) {
    throw new Error("--timeout must be a non-negative integer (seconds; 0 means no timeout)");
  }
  return raw;
}

function formatPayloadForLog(payload: {
  text?: string;
  mediaUrls?: string[];
  mediaUrl?: string | null;
}) {
  const parts = resolveSendableOutboundReplyParts({
    text: payload.text,
    mediaUrls: payload.mediaUrls,
    mediaUrl: typeof payload.mediaUrl === "string" ? payload.mediaUrl : undefined,
  });
  const lines: string[] = [];
  if (parts.text) {
    lines.push(parts.text.trimEnd());
  }
  for (const url of parts.mediaUrls) {
    lines.push(`MEDIA:${url}`);
  }
  return lines.join("\n").trimEnd();
}

function hasMeaningfulJsonPayloads(payloads: AgentGatewayResult["payloads"]) {
  if (!Array.isArray(payloads) || payloads.length === 0) {
    return false;
  }
  return payloads.some((payload) => {
    const text = typeof payload?.text === "string" ? payload.text.trim() : "";
    const mediaUrls = Array.isArray(payload?.mediaUrls) ? payload.mediaUrls : [];
    const mediaUrl = typeof payload?.mediaUrl === "string" ? payload.mediaUrl.trim() : "";
    if (mediaUrls.length > 0 || mediaUrl) {
      return true;
    }
    if (!text) {
      return false;
    }
    if (text.startsWith("⚠️") || text.startsWith("📖") || text.startsWith("<think>")) {
      return false;
    }
    return true;
  });
}

type TranscriptCandidate = {
  updatedAt: number;
  transcriptPath: string;
};

function resolveTranscriptCandidatesForAgentSession(sessionKey?: string): TranscriptCandidate[] {
  const match = /^agent:([^:]+):/.exec(String(sessionKey ?? ""));
  const home = process.env.HOME?.trim();
  if (!match || !home) {
    return [];
  }

  const sessionsDir = `${home}/.openclaw/agents/${match[1]}/sessions`;
  const registryPath = `${sessionsDir}/sessions.json`;
  if (!fs.existsSync(registryPath)) {
    return [];
  }

  let registry: Record<string, { sessionId?: string; updatedAt?: number }> | null = null;
  try {
    registry = JSON.parse(fs.readFileSync(registryPath, "utf8")) as Record<
      string,
      { sessionId?: string; updatedAt?: number }
    >;
  } catch {
    return [];
  }

  if (!registry || typeof registry !== "object") {
    return [];
  }

  const keys = [String(sessionKey ?? ""), String(sessionKey ?? "").toLowerCase()].filter(
    (value, index, array) => Boolean(value) && array.indexOf(value) === index,
  );

  const candidates: TranscriptCandidate[] = [];
  for (const key of keys) {
    const meta = registry[key];
    const sessionId = typeof meta?.sessionId === "string" ? meta.sessionId : "";
    if (!sessionId) {
      continue;
    }
    const transcriptPath = `${sessionsDir}/${sessionId}.jsonl`;
    if (!fs.existsSync(transcriptPath)) {
      continue;
    }
    candidates.push({
      updatedAt: Number.isFinite(meta?.updatedAt) ? meta.updatedAt! : 0,
      transcriptPath,
    });
  }
  return candidates.sort((left, right) => right.updatedAt - left.updatedAt);
}

function scoreTranscriptRecoveryText(text: string) {
  if (!text) {
    return -1;
  }
  let score = Math.min(text.length, 400);
  if (text.includes("<final>")) {
    score += 10_000;
  }
  if (/\b[A-Z_]+_RESULT\b/.test(text)) {
    score += 5_000;
  }
  if (text.startsWith("⚠️") || text.startsWith("📖")) {
    score -= 2_000;
  }
  if (text.startsWith("<think>")) {
    score -= 5_000;
  }
  return score;
}

function extractBestRecoveryTextFromTranscript(transcriptPath: string) {
  let raw = "";
  try {
    raw = fs.readFileSync(transcriptPath, "utf8");
  } catch {
    return null;
  }

  let best: { text: string; score: number } | null = null;
  for (const line of raw.split(/\r?\n/).reverse()) {
    if (!line.trim()) {
      continue;
    }
    let entry: {
      type?: string;
      message?: { role?: string; content?: Array<{ text?: string }> };
    } | null = null;
    try {
      entry = JSON.parse(line) as {
        type?: string;
        message?: { role?: string; content?: Array<{ text?: string }> };
      };
    } catch {
      continue;
    }
    if (entry?.type !== "message" || entry.message?.role !== "assistant") {
      continue;
    }
    const content = Array.isArray(entry.message?.content) ? entry.message.content : [];
    for (const item of [...content].reverse()) {
      const text = typeof item?.text === "string" ? item.text.trim() : "";
      const score = scoreTranscriptRecoveryText(text);
      if (score < 0) {
        continue;
      }
      if (!best || score > best.score) {
        best = { text, score };
      }
    }
  }

  return best;
}

function reconcileJsonResponseFromTranscript(
  response: GatewayAgentResponse,
  sessionKey?: string,
): GatewayAgentResponse & {
  reconciledFromTranscript?: boolean;
  transcriptPath?: string;
  reconciliationReason?: string;
} {
  const payloads = response?.result?.payloads ?? [];
  if (hasMeaningfulJsonPayloads(payloads)) {
    return response;
  }

  let best:
    | ({ text: string; score: number } & { transcriptPath: string; updatedAt: number })
    | null = null;
  for (const candidate of resolveTranscriptCandidatesForAgentSession(sessionKey)) {
    const recovered = extractBestRecoveryTextFromTranscript(candidate.transcriptPath);
    if (!recovered?.text) {
      continue;
    }
    const scored = { ...recovered, ...candidate };
    if (
      !best ||
      scored.score > best.score ||
      (scored.score === best.score && scored.updatedAt > best.updatedAt)
    ) {
      best = scored;
    }
  }

  if (!best) {
    return response;
  }

  return {
    ...response,
    reconciledFromTranscript: true,
    transcriptPath: best.transcriptPath,
    reconciliationReason: "json payloads incomplete; recovered best assistant text from transcript",
    result: {
      ...(response.result ?? {}),
      payloads: [...payloads, { text: best.text, mediaUrl: null }],
    },
  };
}

export async function agentViaGatewayCommand(opts: AgentCliOpts, runtime: RuntimeEnv) {
  const body = (opts.message ?? "").trim();
  if (!body) {
    throw new Error("Message (--message) is required");
  }
  if (!opts.to && !opts.sessionId && !opts.sessionKey && !opts.agent) {
    throw new Error("Pass --to <E.164>, --session-id, --session-key, or --agent to choose a session");
  }

  const cfg = loadConfig();
  const agentIdRaw = opts.agent?.trim();
  const agentId = agentIdRaw ? normalizeAgentId(agentIdRaw) : undefined;
  if (agentId) {
    const knownAgents = listAgentIds(cfg);
    if (!knownAgents.includes(agentId)) {
      throw new Error(
        `Unknown agent id "${agentIdRaw}". Use "${formatCliCommand("openclaw agents list")}" to see configured agents.`,
      );
    }
  }
  const timeoutSeconds = parseTimeoutSeconds({ cfg, timeout: opts.timeout });
  const gatewayTimeoutMs =
    timeoutSeconds === 0
      ? NO_GATEWAY_TIMEOUT_MS // no timeout (timer-safe max)
      : Math.max(10_000, (timeoutSeconds + 30) * 1000);

  const sessionKey = resolveSessionKeyForRequest({
    cfg,
    agentId,
    to: opts.to,
    sessionId: opts.sessionId,
    sessionKey: opts.sessionKey,
  }).sessionKey;

  const channel = normalizeMessageChannel(opts.channel);
  const idempotencyKey = opts.runId?.trim() || randomIdempotencyKey();

  const response = await withProgress(
    {
      label: "Waiting for agent reply…",
      indeterminate: true,
      enabled: opts.json !== true,
    },
    async () =>
      await callGateway<GatewayAgentResponse>({
        method: "agent",
        params: {
          message: body,
          agentId,
          to: opts.to,
          replyTo: opts.replyTo,
          sessionId: opts.sessionId,
          sessionKey,
          thinking: opts.thinking,
          deliver: Boolean(opts.deliver),
          channel,
          replyChannel: opts.replyChannel,
          replyAccountId: opts.replyAccount,
          bestEffortDeliver: opts.bestEffortDeliver,
          timeout: timeoutSeconds,
          lane: opts.lane,
          extraSystemPrompt: opts.extraSystemPrompt,
          idempotencyKey,
        },
        expectFinal: true,
        timeoutMs: gatewayTimeoutMs,
        clientName: GATEWAY_CLIENT_NAMES.CLI,
        mode: GATEWAY_CLIENT_MODES.CLI,
      }),
  );

  if (opts.json) {
    const jsonResponse = reconcileJsonResponseFromTranscript(response, sessionKey);
    writeRuntimeJson(runtime, jsonResponse);
    return jsonResponse;
  }

  const result = response?.result;
  const payloads = result?.payloads ?? [];

  if (payloads.length === 0) {
    runtime.log(response?.summary ? String(response.summary) : "No reply from agent.");
    return response;
  }

  for (const payload of payloads) {
    const out = formatPayloadForLog(payload);
    if (out) {
      runtime.log(out);
    }
  }

  return response;
}

export async function agentCliCommand(opts: AgentCliOpts, runtime: RuntimeEnv, deps?: CliDeps) {
  const localOpts = {
    ...opts,
    agentId: opts.agent,
    replyAccountId: opts.replyAccount,
  };
  if (opts.local === true) {
    return await agentCommand(localOpts, runtime, deps);
  }

  try {
    return await agentViaGatewayCommand(opts, runtime);
  } catch (err) {
    runtime.error?.(`Gateway agent failed; falling back to embedded: ${String(err)}`);
    return await agentCommand(localOpts, runtime, deps);
  }
}
