import { useMemo, useState } from 'react'
import { MathTex, Segmented, Slider, Stat, Widget } from '@/components/ui'
import { pct } from '@/lib/format'
import { useT } from '@/lib/i18n'

type Sched = 'gpipe' | '1f1b'
interface Task {
  s: number
  m: number
  kind: 'F' | 'B'
  start: number
  end: number
}

const TF = 1 // 前向耗时（单位时间）
const TB = 2 // 反向耗时 ≈ 2× 前向

/** 离散事件模拟：每个 stage 按固定任务序列执行，受上/下游依赖约束 */
function simulate(P: number, M: number, sched: Sched): { tasks: Task[]; total: number } {
  // 1. 每个 stage 的任务序列（kind + micro-batch 序号）
  const seqs: { kind: 'F' | 'B'; m: number }[][] = []
  for (let s = 0; s < P; s++) {
    const seq: { kind: 'F' | 'B'; m: number }[] = []
    if (sched === 'gpipe') {
      for (let m = 0; m < M; m++) seq.push({ kind: 'F', m })
      for (let m = 0; m < M; m++) seq.push({ kind: 'B', m })
    } else {
      const w = Math.min(P - s, M) // 暖机前向数：越靠前的 stage 越多
      let f = 0
      let b = 0
      for (; f < w; f++) seq.push({ kind: 'F', m: f })
      while (f < M) {
        seq.push({ kind: 'B', m: b++ })
        seq.push({ kind: 'F', m: f++ })
      }
      while (b < M) seq.push({ kind: 'B', m: b++ })
    }
    seqs.push(seq)
  }

  // 2. 依赖驱动调度：F(s,m) 依赖 F(s-1,m)；B(s,m) 依赖 F(s,m) 与 B(s+1,m)
  const fEnd: number[][] = Array.from({ length: P }, () => Array<number>(M).fill(-1))
  const bEnd: number[][] = Array.from({ length: P }, () => Array<number>(M).fill(-1))
  const ptr = Array<number>(P).fill(0)
  const freeAt = Array<number>(P).fill(0)
  const tasks: Task[] = []
  let remaining = 2 * P * M
  let guard = 0
  while (remaining > 0 && guard++ < 10000) {
    let progress = false
    for (let s = 0; s < P; s++) {
      while (ptr[s] < seqs[s].length) {
        const t = seqs[s][ptr[s]]
        let dep = 0
        if (t.kind === 'F') {
          if (s > 0) {
            if (fEnd[s - 1][t.m] < 0) break
            dep = fEnd[s - 1][t.m]
          }
        } else {
          if (fEnd[s][t.m] < 0) break
          dep = fEnd[s][t.m]
          if (s < P - 1) {
            if (bEnd[s + 1][t.m] < 0) break
            dep = Math.max(dep, bEnd[s + 1][t.m])
          }
        }
        const start = Math.max(freeAt[s], dep)
        const end = start + (t.kind === 'F' ? TF : TB)
        if (t.kind === 'F') fEnd[s][t.m] = end
        else bEnd[s][t.m] = end
        freeAt[s] = end
        tasks.push({ s, m: t.m, kind: t.kind, start, end })
        ptr[s]++
        remaining--
        progress = true
      }
    }
    if (!progress) break
  }
  let total = 0
  for (const t of tasks) total = Math.max(total, t.end)
  return { tasks, total }
}

/** LAB: 流水线气泡 —— GPipe vs 1F1B 甘特图 + 气泡率读数 */
export function PipelineBubbleLab() {
  const t = useT()
  const [p, setP] = useState(4)
  const [m, setM] = useState(8)
  const [sched, setSched] = useState<Sched>('gpipe')

  const { tasks, total } = useMemo(() => simulate(p, m, sched), [p, m, sched])
  const ideal = m * (TF + TB)
  const bubble = (p - 1) / (m + p - 1)
  const bubbleSim = total > 0 ? 1 - ideal / total : 0
  // 峰值激活驻留（stage 0 同时在飞的 micro-batch 数）
  const inflight = sched === 'gpipe' ? m : Math.min(p, m)

  const reset = () => {
    setP(4)
    setM(8)
    setSched('gpipe')
  }

  // ── 甘特图几何 ──
  const W = 720
  const rowH = 24
  const rowGap = 5
  const x0 = 42
  const axisH = 22
  const H = p * (rowH + rowGap) + axisH
  const sx = (W - x0 - 6) / total
  const showLabel = sx * TF > 13 // 格子够宽才标 micro-batch 序号

  return (
    <Widget
      index={2}
      title={t('Pipeline Bubble', '流水线气泡')}
      subtitle={t('The more micro-batches, the thinner the fill/drain idle time is spread', 'micro-batch 越多，启动/排空的空转被摊得越薄')}
      onReset={reset}
      footer={t(
        <>
          <span className="text-cyan">Cyan = forward</span>, <span className="text-amber">amber = backward</span> (timed at 2× forward), gray ground = bubble (that stage idling).
          Switch to 1F1B and watch total time — identical to GPipe, the bubble ratio is unchanged; what changes is "peak activation residency": backward starts as early as possible, so the micro-batches each stage holds at once drop from M to ≤P.
        </>,
        <>
          <span className="text-cyan">青色 = 前向</span>，<span className="text-amber">琥珀 = 反向</span>（耗时取 2× 前向），灰底 = 气泡（该 stage 在空转）。
          切到 1F1B 看总时长 —— 和 GPipe 一模一样，气泡率没变；变的是「峰值激活驻留」：反向尽早开跑，每个 stage 同时挂着的 micro-batch 从 M 个降到 ≤P 个。
        </>,
      )}
    >
      <div className="mb-4 grid gap-4 sm:grid-cols-3">
        <Slider label={t('Pipeline stages P', '流水段数 P')} value={p} min={2} max={8} onChange={setP} unit="stage" />
        <Slider label={t('Micro-batch count M', 'micro-batch 数 M')} value={m} min={1} max={32} onChange={setM} unit={t('', '个')} />
        <div className="flex items-end">
          <Segmented<Sched>
            options={[
              { value: 'gpipe', label: 'GPipe' },
              { value: '1f1b', label: '1F1B' },
            ]}
            value={sched}
            onChange={setSched}
            block
          />
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img" aria-label={t('Pipeline parallelism Gantt chart', '流水线并行甘特图')}>
        {/* 行底（气泡背景） */}
        {Array.from({ length: p }, (_, s) => (
          <g key={s}>
            <text
              x={x0 - 8}
              y={s * (rowH + rowGap) + rowH / 2}
              textAnchor="end"
              dominantBaseline="central"
              fontSize={11}
              fontFamily="var(--font-mono, monospace)"
              fill="var(--color-ink3)"
            >
              S{s}
            </text>
            <rect x={x0} y={s * (rowH + rowGap)} width={total * sx} height={rowH} rx={3} fill="var(--color-bg2)" />
          </g>
        ))}
        {/* 任务块 */}
        {tasks.map((t, i) => (
          <g key={i}>
            <rect
              x={x0 + t.start * sx + 0.5}
              y={t.s * (rowH + rowGap) + 1}
              width={(t.end - t.start) * sx - 1}
              height={rowH - 2}
              rx={2.5}
              fill={t.kind === 'F' ? 'var(--color-cyan)' : 'var(--color-amber)'}
              opacity={0.38 + 0.55 * (m > 1 ? t.m / (m - 1) : 1)}
            />
            {showLabel && (
              <text
                x={x0 + ((t.start + t.end) / 2) * sx}
                y={t.s * (rowH + rowGap) + rowH / 2}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={9}
                fontFamily="var(--font-mono, monospace)"
                fill="var(--color-bg)"
              >
                {t.m}
              </text>
            )}
          </g>
        ))}
        {/* 时间轴 */}
        <line x1={x0} y1={p * (rowH + rowGap) + 4} x2={x0 + total * sx} y2={p * (rowH + rowGap) + 4} stroke="var(--color-line2)" />
        <text x={x0} y={H - 4} fontSize={10} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
          0
        </text>
        <text x={x0 + total * sx} y={H - 4} textAnchor="end" fontSize={10} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
          t = {total}
        </text>
      </svg>

      <div className="mt-4 flex flex-wrap items-end gap-x-8 gap-y-4">
        <Stat label={t('Total time', '总时长')} value={total} unit={t('units', '单位')} tone="ink" />
        <Stat label={t('Ideal time M(tf+tb)', '理想时长 M(tf+tb)')} value={ideal} unit={t('units', '单位')} tone="cyan" />
        <Stat label={t('Bubble ratio (measured)', '气泡率（实测）')} value={pct(bubbleSim, 1)} tone={bubbleSim > 0.25 ? 'rose' : 'volt'} />
        <Stat label={t('Peak activation residency / stage', '峰值激活驻留 / stage')} value={inflight} unit={t('', '份')} tone={sched === '1f1b' ? 'volt' : 'amber'} />
        <div className="text-[13px] text-ink2">
          <MathTex tex={`\\text{bubble}=\\frac{P-1}{M+P-1}=\\frac{${p - 1}}{${m + p - 1}}\\approx ${(bubble * 100).toFixed(1)}\\%`} />
        </div>
      </div>
    </Widget>
  )
}
