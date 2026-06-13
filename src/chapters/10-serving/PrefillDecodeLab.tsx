import { useCallback, useState } from 'react'
import { PlayBar, Slider, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { useT } from '@/lib/i18n'
import { clamp } from '@/lib/format'

/** ── LAB 01：Prefill / Decode 对比 ──
 * 上面板：整段 prompt 的 token 并排同时流过模型（一次大矩阵乘，GPU 满载）。
 * 下面板：decode 一次只送一个 token，GPU 大块时间闲着。
 * 右侧：迷你 roofline 标出两阶段位置 + TTFT/TPOT 估算。
 */

// 估算用参数：Llama-70B FP16 @ H100 SXM
const PARAMS = 70e9
const PEAK_FLOPS = 989e12 // H100 BF16 dense
const HBM_BW = 3.35e12 // B/s
const PREFILL_MFU = 0.62 // 实测量级
const WEIGHT_BYTES = 2 * PARAMS // FP16

const PREFILL_PERIOD = 2600 // ms 一个动画循环
const DECODE_TOK_MS = 900 // 每个 decode token 的动画间隔
const DECODE_N = 7 // 一个循环吐几个 token
const TRAVEL = 0.24 // token 穿越模型占间隔的比例（其余时间 = 闲置）

function LaneTokens({ n, cols, x, y, cell, color }: { n: number; cols: number; x: number; y: number; cell: number; color: string }) {
  const items = []
  for (let i = 0; i < n; i++) {
    const cx = x + (i % cols) * (cell + 2)
    const cy = y + Math.floor(i / cols) * (cell + 2)
    items.push(<rect key={i} x={cx} y={cy} width={cell} height={cell} rx={1.5} className={color} />)
  }
  return <g>{items}</g>
}

export function PrefillDecodeLab() {
  const tr = useT()
  const reduced = useReducedMotion()
  const [playing, setPlaying] = useState(!reduced)
  const [speed, setSpeed] = useState(1)
  const [promptLen, setPromptLen] = useState(1024)
  // t: 动画时钟（ms），reduced motion 时固定在一个有代表性的相位
  const [t, setT] = useState(reduced ? PREFILL_PERIOD * 0.3 : 0)

  // reduced motion 时不自动播放（初始 playing=false），但用户主动点播放仍可看
  useRafLoop((dt) => setT((p) => p + dt * speed), playing)

  const reset = useCallback(() => {
    setT(reduced ? PREFILL_PERIOD * 0.3 : 0)
    setPlaying(!reduced)
    setPromptLen(1024)
    setSpeed(1)
  }, [reduced])

  // ── prefill 相位：0..1，前 55% 是 token 块整体穿越模型，其余是出 token + 间歇 ──
  const pPhase = (t % PREFILL_PERIOD) / PREFILL_PERIOD
  const pSweep = clamp(pPhase / 0.55, 0, 1) // 0..1 块的位置
  const pBusy = pPhase < 0.55

  // ── decode 相位：一次循环 DECODE_N 个 token，每个 token 只有 TRAVEL 比例在“干活” ──
  const dCycle = t % (DECODE_TOK_MS * DECODE_N)
  const dIdx = Math.floor(dCycle / DECODE_TOK_MS) // 当前第几个 token
  const dSub = (dCycle % DECODE_TOK_MS) / DECODE_TOK_MS
  const dBusy = dSub < TRAVEL
  const dSweep = clamp(dSub / TRAVEL, 0, 1)

  // ── 读数 ──
  const ttftMs = (2 * PARAMS * promptLen) / (PEAK_FLOPS * PREFILL_MFU) * 1000
  const tpotMs = (WEIGHT_BYTES / HBM_BW) * 1000
  // decode 的 FLOPs 利用率：算 2P FLOPs 花了 t_mem 的时间
  const decodeMfu = (2 * PARAMS) / (WEIGHT_BYTES / HBM_BW) / PEAK_FLOPS

  // 几何（viewBox 560 x 252，两条泳道）
  const laneY = [16, 142]
  const modelX = 218
  const modelW = 130
  const outX = 392

  // prefill token 块当前 x（从输入区扫到模型另一侧）
  const blockX = 32 + pSweep * (modelX + modelW - 60 - 32)

  return (
    <Widget
      index={1}
      title={tr('Prefill / Decode side by side', 'Prefill / Decode 对比')}
      subtitle={tr('One model, two completely different lives', '同一个模型，两段完全不同的人生')}
      onReset={reset}
      footer={
        <>
          {tr(
            <>
              Numbers estimated for <span className="font-mono">Llama-70B FP16 @ H100</span> (989 TFLOPS / 3.35
              TB/s), batch=1. Prefill packs the whole prompt into one big matmul that saturates compute; decode
              reads all 140 GB of weights from HBM on every single step yet computes only 140 GFLOPs for one
              token — utilization differs by well over an order of magnitude.
            </>,
            <>
              读数按 <span className="font-mono">Llama-70B FP16 @ H100</span>（989 TFLOPS / 3.35 TB/s）估算，batch=1。
              prefill 把整段 prompt 拼成一次大矩阵乘，算力吃满；decode 每一步都要把 140 GB 权重从 HBM
              完整读一遍，却只为一个 token 算 140 GFLOPs —— 利用率差出一个数量级不止。
            </>,
          )}
        </>
      }
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_232px]">
        {/* 左：两条泳道 */}
        <div className="min-w-0">
          <svg viewBox="0 0 560 252" className="w-full" role="img" aria-label={tr('Token flow comparison of the prefill and decode stages', 'prefill 与 decode 两阶段的 token 流动对比')}>
            {/* ───────── PREFILL 泳道 ───────── */}
            <g className="text-ink3">
              <text x={4} y={laneY[0] - 4} className="fill-current font-mono" fontSize={9} letterSpacing={1}>
                {tr('PREFILL · WHOLE PROMPT IN PARALLEL', 'PREFILL · 整段 PROMPT 并行')}
              </text>
            </g>
            <rect x={2} y={laneY[0]} width={556} height={92} rx={6} className="fill-bg2 stroke-line" strokeWidth={1} />
            {/* 输入 token 堆（prompt） */}
            <LaneTokens n={24} cols={4} x={20} y={laneY[0] + 14} cell={10} color="fill-cyan/35" />
            <text x={20} y={laneY[0] + 84} className="fill-current font-mono text-ink3" fontSize={8}>
              {tr('prompt', 'prompt')} × {promptLen}
            </text>
            {/* 模型块 */}
            <rect
              x={modelX} y={laneY[0] + 10} width={modelW} height={64} rx={5}
              className={pBusy ? 'fill-volt/15 stroke-volt' : 'fill-panel2 stroke-line2'}
              strokeWidth={1.2}
            />
            <text x={modelX + modelW / 2} y={laneY[0] + 38} textAnchor="middle" className={`fill-current font-mono ${pBusy ? 'text-volt' : 'text-ink3'}`} fontSize={10}>
              MODEL
            </text>
            <text x={modelX + modelW / 2} y={laneY[0] + 52} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={8}>
              {tr('one big matmul', '一次大矩阵乘')}
            </text>
            {/* 整块 token 同时穿越 */}
            <g opacity={pBusy ? 1 : 0}>
              <LaneTokens n={16} cols={2} x={blockX} y={laneY[0] + 20} cell={9} color="fill-cyan" />
            </g>
            {/* 输出：首 token */}
            <rect x={outX + 24} y={laneY[0] + 32} width={14} height={14} rx={2} className={pBusy ? 'fill-panel2' : 'fill-volt'} />
            <text x={outX + 31} y={laneY[0] + 62} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={8}>
              {tr('first token', '首个 token')}
            </text>
            <text x={outX + 31} y={laneY[0] + 74} textAnchor="middle" className={`fill-current font-mono ${pBusy ? 'text-ink3' : 'text-volt'}`} fontSize={8}>
              → TTFT
            </text>
            {/* prefill GPU 利用率条 */}
            <g>
              <text x={470} y={laneY[0] + 22} className="fill-current font-mono text-ink3" fontSize={8}>GPU</text>
              <rect x={492} y={laneY[0] + 14} width={54} height={10} rx={2} className="fill-bg stroke-line" strokeWidth={1} />
              <rect x={492} y={laneY[0] + 14} width={54 * (pBusy ? PREFILL_MFU : 0.04)} height={10} rx={2} className="fill-volt" style={{ transition: 'width 120ms linear' }} />
            </g>

            {/* ───────── DECODE 泳道 ───────── */}
            <g className="text-ink3">
              <text x={4} y={laneY[1] - 4} className="fill-current font-mono" fontSize={9} letterSpacing={1}>
                {tr('DECODE · ONE TOKEN AT A TIME, SERIAL', 'DECODE · 一次一个 TOKEN 串行')}
              </text>
            </g>
            <rect x={2} y={laneY[1]} width={556} height={92} rx={6} className="fill-bg2 stroke-line" strokeWidth={1} />
            {/* 输入：上一个 token */}
            <rect x={28} y={laneY[1] + 32} width={11} height={11} rx={1.5} className="fill-cyan/50" />
            <text x={34} y={laneY[1] + 62} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={8}>
              {tr('prev token', '上一个 token')}
            </text>
            {/* 模型块（偶尔闪一下） */}
            <rect
              x={modelX} y={laneY[1] + 10} width={modelW} height={64} rx={5}
              className={dBusy ? 'fill-volt/15 stroke-volt' : 'fill-panel2 stroke-line2'}
              strokeWidth={1.2}
            />
            <text x={modelX + modelW / 2} y={laneY[1] + 38} textAnchor="middle" className={`fill-current font-mono ${dBusy ? 'text-volt' : 'text-ink3'}`} fontSize={10}>
              MODEL
            </text>
            <text x={modelX + modelW / 2} y={laneY[1] + 52} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={8}>
              {dBusy ? tr('reading all weights…', '读全部权重…') : tr('idle (awaiting next step)', '空闲（等下一步）')}
            </text>
            {/* 单个 token 穿越 */}
            <g opacity={dBusy ? 1 : 0}>
              <rect x={40 + dSweep * (modelX + modelW - 60 - 40)} y={laneY[1] + 36} width={10} height={10} rx={1.5} className="fill-cyan" />
            </g>
            {/* 输出 token 逐个累积 */}
            {Array.from({ length: DECODE_N }, (_, i) => (
              <rect
                key={i}
                x={outX + 8 + i * 16}
                y={laneY[1] + 34}
                width={11}
                height={11}
                rx={1.5}
                className={i < dIdx || (i === dIdx && !dBusy) ? 'fill-volt' : 'fill-panel2'}
              />
            ))}
            <text x={outX + 8} y={laneY[1] + 62} className="fill-current font-mono text-ink3" fontSize={8}>
              {tr('token gap → TPOT', 'token 间隔 → TPOT')}
            </text>
            {/* decode GPU 利用率条 */}
            <g>
              <text x={470} y={laneY[1] + 22} className="fill-current font-mono text-ink3" fontSize={8}>GPU</text>
              <rect x={492} y={laneY[1] + 14} width={54} height={10} rx={2} className="fill-bg stroke-line" strokeWidth={1} />
              <rect x={492} y={laneY[1] + 14} width={54 * (dBusy ? 0.3 : 0.03)} height={10} rx={2} className="fill-amber" style={{ transition: 'width 120ms linear' }} />
            </g>
          </svg>

          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-3">
            <PlayBar playing={playing} onToggle={() => setPlaying((p) => !p)} onReset={reset} speed={speed} onSpeed={setSpeed} />
            <Slider
              className="w-full max-w-[260px]"
              label={tr('PROMPT LENGTH', 'PROMPT 长度')}
              value={promptLen}
              min={128}
              max={8192}
              step={128}
              onChange={setPromptLen}
              unit="tok"
            />
          </div>
        </div>

        {/* 右：迷你 roofline + 读数 */}
        <div className="flex flex-col gap-4">
          <div className="rounded-md border border-line bg-bg2 p-3">
            <div className="microlabel mb-2">{tr('THE TWO STAGES ON THE ROOFLINE', '两阶段在 ROOFLINE 上的位置')}</div>
            <svg viewBox="0 0 200 130" className="w-full" role="img" aria-label={tr('On the roofline, prefill sits near the compute ceiling while decode is mired on the bandwidth slope', 'roofline 上 prefill 接近算力屋顶，decode 深陷带宽斜坡')}>
              {/* 轴 */}
              <line x1={24} y1={108} x2={192} y2={108} className="stroke-line2" strokeWidth={1} />
              <line x1={24} y1={108} x2={24} y2={10} className="stroke-line2" strokeWidth={1} />
              <text x={108} y={124} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={8}>
                {tr('Arithmetic intensity FLOPs/Byte (log)', '算术强度 FLOPs/Byte (log)')}
              </text>
              <text x={14} y={60} textAnchor="middle" transform="rotate(-90 14 60)" className="fill-current font-mono text-ink3" fontSize={8}>
                FLOPs (log)
              </text>
              {/* 屋顶：斜坡 + 平台，ridge ≈ 295 */}
              <path d="M 26 104 L 122 26 L 190 26" fill="none" className="stroke-ink3" strokeWidth={1.5} />
              <line x1={122} y1={26} x2={122} y2={108} className="stroke-line2" strokeWidth={1} strokeDasharray="3 3" />
              <text x={122} y={118} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={7}>
                ridge≈295
              </text>
              {/* decode 点：AI≈1，贴在斜坡底部 */}
              <circle cx={38} cy={94} r={4} className="fill-amber" />
              <text x={46} y={90} className="fill-current font-mono text-amber" fontSize={8}>
                decode (AI≈1)
              </text>
              <text x={46} y={100} className="fill-current text-ink3" fontSize={7}>
                memory-bound
              </text>
              {/* prefill 点：AI≈promptLen，靠近平台 */}
              <circle cx={clamp(26 + Math.log2(promptLen) * 11, 26, 178)} cy={promptLen >= 512 ? 32 : 40} r={4} className="fill-volt" />
              <text x={clamp(26 + Math.log2(promptLen) * 11, 26, 178) - 4} y={promptLen >= 512 ? 20 : 28} className="fill-current font-mono text-volt" fontSize={8}>
                prefill
              </text>
            </svg>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-3">
            <Stat label={tr('PREFILL UTILIZATION', 'PREFILL 利用率')} value={Math.round(PREFILL_MFU * 100)} unit="%" tone="volt" />
            <Stat label={tr('DECODE UTILIZATION', 'DECODE 利用率')} value={(decodeMfu * 100).toFixed(1)} unit="%" tone="amber" />
            <Stat label="TTFT ≈" value={ttftMs < 1000 ? Math.round(ttftMs) : (ttftMs / 1000).toFixed(2)} unit={ttftMs < 1000 ? 'ms' : 's'} tone="cyan" size="sm" />
            <Stat label="TPOT ≈" value={Math.round(tpotMs)} unit="ms" tone="cyan" size="sm" />
          </div>
        </div>
      </div>
    </Widget>
  )
}
