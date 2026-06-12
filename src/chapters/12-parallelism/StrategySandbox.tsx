import { useMemo, useState } from 'react'
import { Btn, Segmented, Slider, Stat, Widget } from '@/components/ui'

type ModelKey = '7B' | '70B' | '405B' | '1T'
type Mode = 'train' | 'infer'

interface ModelSpec {
  params: number // 参数量
  layers: number
  hidden: number
  /** 单条 4K 序列、micro-batch=1、flash-attn 后的全模型激活（GB），训练态；1F1B 在飞 ≈P 抵消 PP 切分 */
  actGB: number
  /** 32 并发 × 4K 上下文的全模型 KV cache（GB），GQA 后 */
  kvGB: number
}

const MODELS: Record<ModelKey, ModelSpec> = {
  '7B': { params: 7e9, layers: 32, hidden: 4096, actGB: 18, kvGB: 16 },
  '70B': { params: 70e9, layers: 80, hidden: 8192, actGB: 91, kvGB: 40 },
  '405B': { params: 405e9, layers: 126, hidden: 16384, actGB: 287, kvGB: 64 },
  '1T': { params: 1e12, layers: 128, hidden: 25600, actGB: 456, kvGB: 160 },
}

const OVERHEAD_GB = 3 // CUDA context + NCCL 缓冲 + 碎片
const STAGE_COLORS = ['var(--color-cyan)', 'var(--color-amber)', 'var(--color-violet)', 'var(--color-volt)', 'var(--color-rose)']

const gb = (bytes: number) => bytes / 1e9

interface Preset {
  label: string
  model: ModelKey
  mem: number
  gpus: number
  mode: Mode
  tp: number
  pp: number
  zero: number
}

const PRESETS: Preset[] = [
  { label: 'LLaMA-70B 训练 · 64×A100', model: '70B', mem: 80, gpus: 64, mode: 'train', tp: 8, pp: 2, zero: 1 },
  { label: '405B 训练 · 1024×H100', model: '405B', mem: 80, gpus: 1024, mode: 'train', tp: 8, pp: 16, zero: 1 },
  { label: '70B 推理 · TP8', model: '70B', mem: 80, gpus: 8, mode: 'infer', tp: 8, pp: 1, zero: 0 },
]

/** LAB: 并行策略沙盘 —— TP×PP×DP×ZeRO 组合下的每卡显存账与通信画像 */
export function StrategySandboxLab() {
  const [model, setModel] = useState<ModelKey>('70B')
  const [mem, setMem] = useState(80)
  const [gpus, setGpus] = useState(64)
  const [mode, setMode] = useState<Mode>('train')
  const [tp, setTp] = useState(8)
  const [pp, setPp] = useState(2)
  const [zero, setZero] = useState(1)

  const applyPreset = (pr: Preset) => {
    setModel(pr.model)
    setMem(pr.mem)
    setGpus(pr.gpus)
    setMode(pr.mode)
    setTp(pr.tp)
    setPp(pr.pp)
    setZero(pr.zero)
  }
  const reset = () => applyPreset(PRESETS[0])

  const spec = MODELS[model]
  const dpRaw = gpus / (tp * pp)
  const dpInt = Number.isInteger(dpRaw) && dpRaw >= 1
  const ppOk = pp <= spec.layers
  const valid = dpInt && ppOk
  const dp = valid ? dpRaw : 0

  // ── 每卡显存账（GB）──
  const memo = useMemo(() => {
    if (!valid) return null
    const shard = tp * pp
    const zW = zero >= 3 ? dp : 1
    const zG = zero >= 2 ? dp : 1
    const zO = zero >= 1 ? dp : 1
    if (mode === 'train') {
      const weights = gb(2 * spec.params) / shard / zW
      const grads = gb(2 * spec.params) / shard / zG
      const optim = gb(12 * spec.params) / shard / zO
      const act = spec.actGB / tp
      const segs = [
        { name: '权重 (BF16)', v: weights, color: 'var(--color-cyan)' },
        { name: '梯度 (BF16)', v: grads, color: 'var(--color-amber)' },
        { name: '优化器态 (FP32×3)', v: optim, color: 'var(--color-violet)' },
        { name: '激活 (4K序列)', v: act, color: 'var(--color-volt)' },
        { name: '框架开销', v: OVERHEAD_GB, color: 'var(--color-line2)' },
      ]
      return { segs, total: segs.reduce((a, s) => a + s.v, 0) }
    }
    const weights = gb(2 * spec.params) / shard
    const kv = spec.kvGB / shard
    const segs = [
      { name: '权重 (BF16)', v: weights, color: 'var(--color-cyan)' },
      { name: 'KV cache (32并发×4K)', v: kv, color: 'var(--color-volt)' },
      { name: '框架开销', v: OVERHEAD_GB, color: 'var(--color-line2)' },
    ]
    return { segs, total: segs.reduce((a, s) => a + s.v, 0) }
  }, [valid, tp, pp, dp, zero, mode, spec])

  const fits = memo != null && memo.total <= mem

  // ── GPU 网格几何 ──
  const grid = useMemo(() => {
    if (!valid) return null
    const repPerRow = Math.max(1, Math.min(dp, Math.round(64 / tp)))
    const repRows = Math.ceil(dp / repPerRow)
    const cs = dp * tp * pp > 512 ? 7 : dp * tp * pp > 128 ? 10 : 14
    const repW = tp * (cs + 1) + 5
    const repH = pp * (cs + 1) + 5
    const gap = 8
    const W = Math.max(360, repPerRow * (repW + gap) + 4)
    const H = repRows * (repH + gap) + 16
    return { repPerRow, repRows, cs, repW, repH, gap, W, H }
  }, [valid, dp, tp, pp])

  const zeroDisabled = mode === 'infer'

  return (
    <Widget
      index={3}
      title="并行策略沙盘"
      subtitle="给定模型和集群，亲手切出一个能装下的并行方案"
      onReset={reset}
      wide
      footer={
        <>
          估算口径：训练 = BF16 权重 2B + 梯度 2B + FP32 master/m/v 12B 每参数，激活按 4K 序列、micro-batch=1、flash-attention、1F1B 驻留近似（只被 TP 摊薄）；
          推理 = BF16 权重 + 32 并发 × 4K 上下文 KV（GQA）。ZeRO 只作用于 DP 维 —— DP=1 时调 ZeRO 没有任何效果。真实系统还有碎片与重计算等变量，此处取「约」。
        </>
      }
    >
      {/* ── 输入区 ── */}
      <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-3">
        <div>
          <div className="microlabel mb-1">模型</div>
          <Segmented<ModelKey>
            options={(['7B', '70B', '405B', '1T'] as ModelKey[]).map((k) => ({ value: k, label: k }))}
            value={model}
            onChange={setModel}
          />
        </div>
        <div>
          <div className="microlabel mb-1">单卡显存</div>
          <Segmented<number>
            options={[
              { value: 24, label: '24G' },
              { value: 80, label: '80G' },
              { value: 141, label: '141G' },
            ]}
            value={mem}
            onChange={setMem}
          />
        </div>
        <div>
          <div className="microlabel mb-1">GPU 数</div>
          <Segmented<number>
            options={[8, 64, 512, 1024].map((v) => ({ value: v, label: String(v) }))}
            value={gpus}
            onChange={setGpus}
          />
        </div>
        <div>
          <div className="microlabel mb-1">模式</div>
          <Segmented<Mode>
            options={[
              { value: 'train', label: '训练' },
              { value: 'infer', label: '推理' },
            ]}
            value={mode}
            onChange={(v) => {
              setMode(v)
              if (v === 'infer') setZero(0)
            }}
          />
        </div>
      </div>

      <div className="mb-4 grid gap-4 sm:grid-cols-3">
        <Slider label="张量并行 TP" value={tp} min={1} max={8} onChange={setTp} unit="路" />
        <Slider label="流水并行 PP" value={pp} min={1} max={16} onChange={setPp} unit="段" />
        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="font-mono text-[11px] uppercase tracking-wider text-ink2">数据并行 DP（自动）</span>
            <span className={`font-mono text-xs ${valid ? 'text-volt' : 'text-rose'}`}>
              {dpInt ? `${dpRaw} 路` : '非整数'}
            </span>
          </div>
          <div className="rounded-md border border-line bg-bg2 px-3 py-1.5 font-mono text-xs text-ink2">
            {gpus} ÷ (TP{tp} × PP{pp}) = {dpInt ? dpRaw : dpRaw.toFixed(2)}
          </div>
        </div>
      </div>

      <div className="mb-5 flex flex-wrap items-center gap-x-6 gap-y-3">
        <div>
          <div className="microlabel mb-1">ZeRO STAGE（作用于 DP 维）</div>
          <Segmented<number>
            options={[
              { value: 0, label: 'off' },
              { value: 1, label: 'Z1 优化器' },
              { value: 2, label: 'Z2 +梯度' },
              { value: 3, label: 'Z3 +权重' },
            ]}
            value={zeroDisabled ? 0 : zero}
            onChange={(v) => {
              if (!zeroDisabled) setZero(v)
            }}
          />
          {zeroDisabled && <div className="mt-1 text-[11px] text-ink3">推理没有梯度和优化器态，ZeRO 不适用</div>}
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          {PRESETS.map((pr) => (
            <Btn key={pr.label} variant="ghost" size="sm" onClick={() => applyPreset(pr)}>
              {pr.label}
            </Btn>
          ))}
        </div>
      </div>

      {/* ── 非法组合提示 ── */}
      {!valid && (
        <div className="my-6 rounded-md border border-rose/50 bg-rose/[0.07] px-4 py-3 text-[13.5px] leading-relaxed text-ink">
          <span className="microlabel mr-2 text-rose">✗ 非法组合</span>
          {!dpInt && (
            <>
              {gpus} 张卡不能被 TP×PP = {tp * pp} 整除（得 {dpRaw.toFixed(2)}），DP 副本无法成形。调整 TP/PP 使乘积整除 GPU 数。
            </>
          )}
          {dpInt && !ppOk && (
            <>
              PP = {pp} 段超过了 {model} 的 {spec.layers} 层 —— 没法给每段都分到层。
            </>
          )}
        </div>
      )}

      {valid && memo && grid && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
          {/* ── GPU 网格 ── */}
          <div>
            <div className="microlabel mb-2">
              集群拓扑 · {gpus} GPU = TP{tp} × PP{pp} × DP{dp}
            </div>
            <svg viewBox={`0 0 ${grid.W} ${grid.H}`} className="w-full select-none" role="img" aria-label="GPU 并行拓扑网格">
              {Array.from({ length: dp }, (_, r) => {
                const rx = (r % grid.repPerRow) * (grid.repW + grid.gap) + 2
                const ry = Math.floor(r / grid.repPerRow) * (grid.repH + grid.gap) + 2
                return (
                  <g key={r} transform={`translate(${rx}, ${ry})`}>
                    <rect width={grid.repW} height={grid.repH} rx={4} fill="none" stroke="var(--color-line)" />
                    {Array.from({ length: pp }, (_, s) => (
                      <g key={s}>
                        {/* 一行 = 一个 PP stage 的 TP 组（同色 = NVLink 域） */}
                        {Array.from({ length: tp }, (_, t) => (
                          <rect
                            key={t}
                            x={3 + t * (grid.cs + 1)}
                            y={3 + s * (grid.cs + 1)}
                            width={grid.cs}
                            height={grid.cs}
                            rx={2}
                            fill={STAGE_COLORS[s % STAGE_COLORS.length]}
                            opacity={0.32 + 0.4 * (pp > 1 ? 1 - s / (pp * 2) : 0.5)}
                          />
                        ))}
                        {tp > 1 && (
                          <rect
                            x={2.5}
                            y={2.5}
                            width={tp * (grid.cs + 1)}
                            height={grid.cs + 1}
                            transform={`translate(0, ${s * (grid.cs + 1)})`}
                            rx={2.5}
                            fill="none"
                            stroke="var(--color-line2)"
                            strokeWidth={0.75}
                          />
                        )}
                      </g>
                    ))}
                    {dp <= 8 && (
                      <text x={grid.repW / 2} y={grid.repH + 11} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
                        副本 {r}
                      </text>
                    )}
                  </g>
                )
              })}
            </svg>
            <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11.5px] text-ink3">
              <span>■ 同色横排 = 一个 TP 组（NVLink 域内）</span>
              <span>行 = PP stage（颜色区分）</span>
              <span>外框 = 一个 DP 副本（完整模型）</span>
            </div>

            {/* ── 通信强度 ── */}
            <div className="mt-5 space-y-2.5">
              <div className="microlabel">通信画像</div>
              {[
                {
                  name: 'TP',
                  on: tp > 1,
                  w: 0.95,
                  desc: tp > 1 ? '每层前向+反向共 4 次 AllReduce，在关键路径上 —— 必须 NVLink 域内' : '关闭（TP=1）',
                },
                {
                  name: 'PP',
                  on: pp > 1,
                  w: 0.45,
                  desc: pp > 1 ? '每个 micro-batch 边界一次 P2P 激活传输，量小、可与计算重叠' : '关闭（PP=1）',
                },
                {
                  name: 'DP',
                  on: dp > 1,
                  w: zero >= 3 ? 0.4 : 0.25,
                  desc:
                    dp > 1
                      ? mode === 'infer'
                        ? '推理副本间零通信，只在负载均衡器层面分流'
                        : zero >= 3
                          ? '每层 AllGather 权重 + 每 step ReduceScatter 梯度（ZeRO-3 通信 ×1.5）'
                          : '每 step 一次梯度 AllReduce，可与反向重叠'
                      : '关闭（DP=1）',
                },
              ].map((c) => (
                <div key={c.name} className="flex items-center gap-3">
                  <span className="w-7 font-mono text-xs text-ink2">{c.name}</span>
                  <div className="h-2 w-28 shrink-0 overflow-hidden rounded-full bg-bg2">
                    <div
                      className="h-full rounded-full bg-amber transition-all duration-300"
                      style={{ width: `${c.on ? c.w * 100 : 0}%`, opacity: c.on ? 0.9 : 0 }}
                    />
                  </div>
                  <span className="text-[11.5px] leading-snug text-ink3">{c.desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── 每卡显存账 ── */}
          <div>
            <div className="microlabel mb-2">每卡显存估算</div>
            <div className="mb-3 flex items-baseline gap-3">
              <span className={`font-mono text-2xl tabular-nums ${fits ? 'text-volt' : 'text-rose'}`}>{memo.total.toFixed(1)}</span>
              <span className="text-xs text-ink3">/ {mem} GB</span>
              <span className={`ml-auto font-mono text-sm ${fits ? 'text-volt' : 'text-rose'}`}>{fits ? '✓ 装下了' : '✗ OOM'}</span>
            </div>
            {/* 堆叠条 + 红线 */}
            <svg viewBox="0 0 300 46" className="w-full select-none" role="img" aria-label="每卡显存堆叠条">
              {(() => {
                const scale = 292 / Math.max(memo.total, mem * 1.08)
                let x = 4
                const parts = memo.segs.map((s, i) => {
                  const w = s.v * scale
                  const el = <rect key={i} x={x} y={8} width={Math.max(0.5, w - 0.8)} height={22} fill={s.color} opacity={0.8} />
                  x += w
                  return el
                })
                return (
                  <>
                    <rect x={4} y={8} width={292} height={22} rx={3} fill="var(--color-bg2)" />
                    {parts}
                    <line x1={4 + mem * scale} y1={2} x2={4 + mem * scale} y2={40} stroke="var(--color-rose)" strokeWidth={1.5} strokeDasharray="4 3" />
                    <text x={Math.min(288, 4 + mem * scale)} y={45} textAnchor="end" fontSize={9} fontFamily="var(--font-mono, monospace)" fill="var(--color-rose)">
                      {mem}G 红线
                    </text>
                  </>
                )
              })()}
            </svg>
            <div className="mt-2 space-y-1.5">
              {memo.segs.map((s) => (
                <div key={s.name} className="flex items-center gap-2 text-[12px]">
                  <span className="size-2.5 shrink-0 rounded-sm" style={{ background: s.color, opacity: 0.85 }} />
                  <span className="text-ink2">{s.name}</span>
                  <span className="ml-auto font-mono tabular-nums text-ink">{s.v < 0.1 ? '<0.1' : s.v.toFixed(1)} GB</span>
                </div>
              ))}
            </div>
            <div className="mt-4 flex gap-6">
              <Stat label="DP 副本" value={dp} unit="路" tone="cyan" size="sm" />
              <Stat label="每副本卡数" value={tp * pp} unit="卡" tone="ink" size="sm" />
              <Stat label="ZeRO" value={zeroDisabled || zero === 0 ? 'off' : `Z${zero}`} tone={zero > 0 && !zeroDisabled ? 'volt' : 'ink'} size="sm" />
            </div>
            {dp === 1 && zero > 0 && !zeroDisabled && (
              <div className="mt-3 rounded-md border border-amber/40 bg-amber/[0.06] px-3 py-2 text-[12px] leading-relaxed text-ink2">
                DP=1 时 ZeRO 没有可切分的副本维度，等同于 off。
              </div>
            )}
          </div>
        </div>
      )}
    </Widget>
  )
}
