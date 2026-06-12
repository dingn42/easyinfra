/**
 * 连续批处理模拟器的纯逻辑层：
 * 固定种子生成泊松到达的请求流，然后分别按 static / continuous 两种调度
 * 推演出每个请求的时间段（等待 / prefill / decode / 占位空转）。
 * 全部确定性计算 —— UI 只负责回放。
 *
 * 简化模型（在 LAB footer 里向读者说明）：
 * - decode 每 token 固定 T_ITER（忽略 batch 大小对单步时延的影响）
 * - prefill 速率固定 PREFILL_TPS token/ms
 * - static batching 的批内 prefill 按最长 prompt 对齐（padding）
 */

export const T_ITER = 30 // ms / decode token（TPOT）
export const PREFILL_TPS = 8 // prefill 吞吐：token / ms
export const UTIL_DT = 200 // GPU 利用率采样间隔 ms
export const N_REQUESTS = 26

export type Mode = 'static' | 'continuous'

export interface ReqSpec {
  id: number
  arrival: number // ms
  promptLen: number
  outLen: number
}

export interface ReqSched {
  spec: ReqSpec
  /** 服务开始（prefill 起点） */
  start: number
  /** prefill 结束 = 开始 decode */
  prefillEnd: number
  /** 最后一个 token 吐出时刻（用户视角完成） */
  emitEnd: number
  /** 槽位真正释放时刻（static 下 = 整批结束，可能 > emitEnd） */
  release: number
}

export interface Schedule {
  mode: Mode
  slots: number
  reqs: ReqSched[]
  /** 最后一个槽位释放时刻 */
  duration: number
  /** 每 UTIL_DT 采样一次的 GPU 槽位利用率 0..1 */
  util: number[]
}

/** mulberry32 —— 固定种子伪随机 */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 生成请求流：到达 ~ Poisson(rate 个/秒)，prompt/输出长度在区间内均匀分布 */
export function genWorkload(rate: number, promptMax: number, outMax: number, n = N_REQUESTS): ReqSpec[] {
  const rnd = mulberry32(42)
  const reqs: ReqSpec[] = []
  let t = 0
  for (let i = 0; i < n; i++) {
    t += (-Math.log(1 - rnd()) / rate) * 1000
    const promptLen = Math.round(64 + rnd() * (promptMax - 64))
    const outLen = Math.round(16 + rnd() * (outMax - 16))
    reqs.push({ id: i, arrival: t, promptLen, outLen })
  }
  return reqs
}

function prefillDur(r: ReqSpec): number {
  return r.promptLen / PREFILL_TPS
}

/** 连续批处理：槽位一空，等待队列里最早的请求立刻进场（K 服务台 FIFO） */
function schedContinuous(reqs: ReqSpec[], slots: number): ReqSched[] {
  const freeAt = new Array<number>(slots).fill(0)
  const out: ReqSched[] = []
  for (const r of reqs) {
    let idx = 0
    for (let i = 1; i < slots; i++) if (freeAt[i] < freeAt[idx]) idx = i
    const start = Math.max(r.arrival, freeAt[idx])
    const pEnd = start + prefillDur(r)
    const eEnd = pEnd + r.outLen * T_ITER
    freeAt[idx] = eEnd
    out.push({ spec: r, start, prefillEnd: pEnd, emitEnd: eEnd, release: eEnd })
  }
  return out
}

/** 静态批处理：GPU 空闲时把已到达的请求（≤slots 个）整批拉进来，整批跑完才换下一批 */
function schedStatic(reqs: ReqSpec[], slots: number): ReqSched[] {
  const out: ReqSched[] = []
  let gpuFree = 0
  let i = 0
  while (i < reqs.length) {
    const t0 = Math.max(gpuFree, reqs[i].arrival)
    const batch: ReqSpec[] = []
    let j = i
    while (j < reqs.length && reqs[j].arrival <= t0 && batch.length < slots) {
      batch.push(reqs[j])
      j++
    }
    i = j
    let maxPrefill = 0
    let maxOut = 0
    for (const r of batch) {
      maxPrefill = Math.max(maxPrefill, prefillDur(r))
      maxOut = Math.max(maxOut, r.outLen)
    }
    const pEnd = t0 + maxPrefill // 批内 padding：所有人等最长 prompt
    const batchEnd = pEnd + maxOut * T_ITER
    for (const r of batch) {
      out.push({
        spec: r,
        start: t0,
        prefillEnd: pEnd,
        emitEnd: pEnd + r.outLen * T_ITER,
        release: batchEnd,
      })
    }
    gpuFree = batchEnd
  }
  return out
}

export function simulate(reqs: ReqSpec[], slots: number, mode: Mode): Schedule {
  const scheds = mode === 'continuous' ? schedContinuous(reqs, slots) : schedStatic(reqs, slots)
  let duration = 0
  for (const s of scheds) duration = Math.max(duration, s.release)
  // 利用率采样：某时刻“在干活”= 正在 prefill 或还在吐 token（占着槽却已吐完 = 空转）
  const n = Math.max(1, Math.ceil(duration / UTIL_DT))
  const util = new Array<number>(n)
  for (let k = 0; k < n; k++) {
    const t = k * UTIL_DT
    let busy = 0
    for (const s of scheds) if (t >= s.start && t < s.emitEnd) busy++
    util[k] = Math.min(1, busy / slots)
  }
  return { mode, slots, reqs: scheds, duration, util }
}

/** 请求 r 截至 t 已吐出的 token 数（第 k 个 token 在 prefillEnd + k*T_ITER 时刻吐出） */
export function tokensAt(s: ReqSched, t: number): number {
  if (t < s.prefillEnd) return 0
  return Math.min(s.spec.outLen, Math.floor((t - s.prefillEnd) / T_ITER))
}

export interface Stats {
  throughput: number // tok/s（截至 t 的累计 decode 吞吐）
  avgTtft: number | null // ms
  p95Latency: number | null // ms
  gpuUtil: number // 0..1（截至 t 的平均）
  done: number // 已完成请求数
}

/** 回放到 t 时刻的运行统计 */
export function statsAt(sched: Schedule, t: number): Stats {
  let tokens = 0
  const ttfts: number[] = []
  const lats: number[] = []
  for (const s of sched.reqs) {
    tokens += tokensAt(s, t)
    const firstTok = s.prefillEnd + T_ITER
    if (firstTok <= t) ttfts.push(firstTok - s.spec.arrival)
    if (s.emitEnd <= t) lats.push(s.emitEnd - s.spec.arrival)
  }
  const kMax = Math.min(sched.util.length, Math.max(1, Math.ceil(t / UTIL_DT)))
  let u = 0
  for (let k = 0; k < kMax; k++) u += sched.util[k]
  u /= kMax
  let p95: number | null = null
  if (lats.length > 0) {
    lats.sort((a, b) => a - b)
    p95 = lats[Math.min(lats.length - 1, Math.ceil(lats.length * 0.95) - 1)]
  }
  return {
    throughput: t > 0 ? tokens / (t / 1000) : 0,
    avgTtft: ttfts.length > 0 ? ttfts.reduce((a, b) => a + b, 0) / ttfts.length : null,
    p95Latency: p95,
    gpuUtil: u,
    done: lats.length,
  }
}
