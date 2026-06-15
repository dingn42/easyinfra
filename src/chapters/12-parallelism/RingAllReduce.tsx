import { useRef, useState } from 'react'
import { MathTex, PlayBar, Slider, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { useT } from '@/lib/i18n'

/**
 * LAB: Ring AllReduce 步进动画。
 * N 卡成环，每卡数据切 N 块：reduce-scatter N-1 步（累加块沿环流动），
 * all-gather N-1 步（成品块回流）。展示「每卡通信量 2(N-1)/N·D 与 N 无关地有界」。
 */
export function RingAllReduceLab() {
  const tr = useT()
  const [n, setN] = useState(4)
  const [step, setStep] = useState(0) // 已完成的步数，0..2(n-1)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const reduced = useReducedMotion()
  const accRef = useRef(0)

  const total = 2 * (n - 1)
  const STEP_MS = 1100

  useRafLoop((dt) => {
    accRef.current += dt * speed
    if (accRef.current >= STEP_MS) {
      accRef.current = 0
      setStep((s) => {
        if (s >= total) {
          setPlaying(false)
          return s
        }
        if (s + 1 >= total) setPlaying(false)
        return s + 1
      })
    }
  }, playing)

  const changeN = (v: number) => {
    setN(v)
    setStep(0)
    setPlaying(false)
    accRef.current = 0
  }
  const reset = () => {
    setStep(0)
    setPlaying(false)
    accRef.current = 0
  }
  const stepOnce = () => setStep((s) => Math.min(total, s + 1))

  // ── 推导当前状态 ──
  const t = Math.min(step, n - 1) // 已完成的 reduce-scatter 步数
  const u = Math.max(0, step - (n - 1)) // 已完成的 all-gather 步数
  const inRS = step > 0 && step <= n - 1
  const inAG = step > n - 1 && step <= total

  /** 单元格状态：chunk c 在 GPU i 上的展示 */
  const cellState = (i: number, c: number): { kind: 'own' | 'accum' | 'full'; count: number } => {
    const fc = (c + n - 1) % n // reduce-scatter 结束时持有完整 chunk c 的 GPU
    const dist = (i - fc + n) % n
    if (step >= n - 1 && dist <= u) return { kind: 'full', count: n }
    if (step < n - 1 && i === (c + t) % n) return { kind: 'accum', count: t + 1 }
    return { kind: 'own', count: 1 }
  }

  /** 第 step 步里 GPU i 发往 i+1 的 chunk 序号（用于边上的「飞行包」标注） */
  const flyingChunk = (i: number): number => {
    if (inRS) return (((i - step + 1) % n) + n) % n
    if (inAG) {
      const uu = step - (n - 1)
      return (((i - uu + 2) % n) + n) % n
    }
    return -1
  }

  // ── 几何 ──
  const W = 600
  const H = 470
  const cx = W / 2
  const cy = H / 2 + 6
  const r = 168
  const cs = n <= 4 ? 19 : n <= 6 ? 15 : 12 // chunk 单元格边长
  const nodeW = n * (cs + 2) + 10
  const nodeH = cs + 30
  const pos = (i: number) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / n
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) }
  }

  const phaseLabel =
    step === 0
      ? tr('Initial', '初始')
      : step <= n - 1
        ? `reduce-scatter ${step}/${n - 1}`
        : step < total
          ? `all-gather ${step - (n - 1)}/${n - 1}`
          : tr('Done ✓', '完成 ✓')
  const phaseTone = step === 0 ? 'ink' : step <= n - 1 ? 'amber' : step < total ? 'cyan' : 'volt'
  const coeff = (2 * (n - 1)) / n

  const fillFor = (k: 'own' | 'accum' | 'full') =>
    k === 'full' ? 'var(--color-volt)' : k === 'accum' ? 'var(--color-amber)' : 'var(--color-cyan)'
  const opFor = (k: 'own' | 'accum' | 'full') => (k === 'full' ? 0.85 : k === 'accum' ? 0.8 : 0.18)

  return (
    <Widget
      index={1}
      title={tr('Ring AllReduce Animation', 'Ring AllReduce 动画')}
      subtitle={tr('N-1 reduce-scatter steps + N-1 all-gather steps', 'N-1 步 reduce-scatter + N-1 步 all-gather')}
      onReset={reset}
      footer={tr(
        <>
          Each GPU splits its gradient into N chunks. <span className="text-amber">Amber</span> is the partial sum being accumulated around the ring (reduce-scatter);
          <span className="text-volt"> green</span> is the finished chunk, fully summed and now flowing back to be broadcast (all-gather). At every instant all N edges transmit at once: bandwidth is fully saturated, and each edge moves only D/N per step.
        </>,
        <>
          每张卡把自己的梯度切成 N 块。<span className="text-amber">琥珀色</span>是正沿环累加的部分和（reduce-scatter），
          <span className="text-volt"> 荧光绿</span>是已完成全量求和、正在回流广播的成品块（all-gather）。任意时刻所有 N 条边都在同时传输：带宽被完全用满，每条边每步只传 D/N。
        </>,
      )}
    >
      <div className="mb-4 grid gap-4 sm:grid-cols-[1fr_auto]">
        <Slider label={tr('GPU count N', 'GPU 数 N')} value={n} min={2} max={8} onChange={changeN} unit={tr('GPUs', '卡')} />
        <div className="flex items-end">
          <PlayBar
            playing={playing}
            onToggle={() => {
              if (step >= total) reset()
              setPlaying((p) => !p)
            }}
            onStep={stepOnce}
            onReset={reset}
            speed={speed}
            onSpeed={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_180px]">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img" aria-label={tr('Ring AllReduce ring-topology animation', 'Ring AllReduce 环形拓扑动画')}>
          {/* 环上的边（弧线 + 方向箭头 + 飞行块） */}
          {Array.from({ length: n }, (_, i) => {
            const a = pos(i)
            const b = pos((i + 1) % n)
            const mx = (a.x + b.x) / 2
            const my = (a.y + b.y) / 2
            // 控制点沿行进方向的右侧（朝环外）推出去，n=2 时两条边也不会重叠
            const dx = b.x - a.x
            const dy = b.y - a.y
            const dl = Math.hypot(dx, dy) || 1
            const px = mx + (dy / dl) * 56
            const py = my - (dx / dl) * 56
            const active = step > 0 && step <= total && (inRS || inAG)
            const fc = flyingChunk(i)
            // 贝塞尔 t=0.66 处的点与切向，放方向箭头（避开节点框与弧顶标注）
            const tt = 0.66
            const bx = (1 - tt) ** 2 * a.x + 2 * (1 - tt) * tt * px + tt ** 2 * b.x
            const by = (1 - tt) ** 2 * a.y + 2 * (1 - tt) * tt * py + tt ** 2 * b.y
            const tx = 2 * (1 - tt) * (px - a.x) + 2 * tt * (b.x - px)
            const ty = 2 * (1 - tt) * (py - a.y) + 2 * tt * (b.y - py)
            const ang = (Math.atan2(ty, tx) * 180) / Math.PI
            const edgeColor = active ? (inRS ? 'var(--color-amber)' : 'var(--color-cyan)') : 'var(--color-line2)'
            return (
              <g key={i}>
                <path
                  d={`M ${a.x} ${a.y} Q ${px} ${py} ${b.x} ${b.y}`}
                  fill="none"
                  stroke={edgeColor}
                  strokeWidth={active ? 1.6 : 1}
                  strokeDasharray={active ? 'none' : '3 4'}
                  opacity={active ? 0.85 : 0.6}
                />
                {/* 方向箭头 */}
                <polygon
                  points="0,0 -7,3.5 -7,-3.5"
                  transform={`translate(${bx}, ${by}) rotate(${ang})`}
                  fill={edgeColor}
                  opacity={active ? 0.9 : 0.55}
                />
                {/* 飞行块标注：弧顶处的 chunk 序号 */}
                <g transform={`translate(${(a.x + 2 * px + b.x) / 4}, ${(a.y + 2 * py + b.y) / 4})`}>
                  <circle r={active ? 9.5 : 0} fill="var(--color-bg)" opacity={0.9} />
                  {active && fc >= 0 && (
                    <text
                      textAnchor="middle"
                      dominantBaseline="central"
                      fontSize={10}
                      fontFamily="var(--font-mono, monospace)"
                      fill={inRS ? 'var(--color-amber)' : 'var(--color-cyan)'}
                    >
                      {fc}
                    </text>
                  )}
                </g>
              </g>
            )
          })}

          {/* GPU 节点 */}
          {Array.from({ length: n }, (_, i) => {
            const p = pos(i)
            const sending = step > 0 && step <= total
            return (
              <g key={i} transform={`translate(${p.x - nodeW / 2}, ${p.y - nodeH / 2})`}>
                <rect
                  width={nodeW}
                  height={nodeH}
                  rx={6}
                  fill="var(--color-panel)"
                  stroke={sending ? 'var(--color-line2)' : 'var(--color-line)'}
                />
                <text x={8} y={15} fontSize={10.5} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
                  GPU {i}
                </text>
                {Array.from({ length: n }, (_, c) => {
                  const st = cellState(i, c)
                  return (
                    <g key={c} transform={`translate(${5 + c * (cs + 2)}, ${nodeH - cs - 5})`}>
                      <rect
                        width={cs}
                        height={cs}
                        rx={2.5}
                        fill={fillFor(st.kind)}
                        opacity={opFor(st.kind)}
                        stroke={st.kind === 'own' ? 'var(--color-line2)' : 'transparent'}
                        strokeWidth={0.75}
                        style={reduced ? undefined : { transition: 'opacity 280ms, fill 280ms' }}
                      />
                      {cs >= 14 && (
                        <text
                          x={cs / 2}
                          y={cs / 2}
                          textAnchor="middle"
                          dominantBaseline="central"
                          fontSize={9.5}
                          fontFamily="var(--font-mono, monospace)"
                          fill={st.kind === 'own' ? 'var(--color-ink3)' : 'var(--color-bg)'}
                        >
                          {st.count}
                        </text>
                      )}
                    </g>
                  )
                })}
              </g>
            )
          })}

          {/* 图例：英文标签更宽，位置/字号随语言调整以免重叠 */}
          <g transform={`translate(14, ${H - 18})`} fontSize={tr(9.5, 10.5)} fontFamily="var(--font-mono, monospace)">
            <rect width={10} height={10} y={-9} rx={2} fill="var(--color-cyan)" opacity={0.18} stroke="var(--color-line2)" strokeWidth={0.75} />
            <text x={15} fill="var(--color-ink3)">
              {tr('local chunk (1)', '本地块(计数1)')}
            </text>
            <rect width={10} height={10} x={tr(125, 118)} y={-9} rx={2} fill="var(--color-amber)" opacity={0.8} />
            <text x={tr(140, 133)} fill="var(--color-ink3)">
              {tr('partial sum', '部分和(累加中)')}
            </text>
            <rect width={10} height={10} x={tr(255, 238)} y={-9} rx={2} fill="var(--color-volt)" opacity={0.85} />
            <text x={tr(270, 253)} fill="var(--color-ink3)">
              {tr('full sum (N ready)', '完整和(N 份已齐)')}
            </text>
          </g>
        </svg>

        <div className="flex flex-row flex-wrap gap-5 lg:flex-col">
          <Stat label={tr('Phase', '阶段')} value={phaseLabel} tone={phaseTone as 'ink' | 'amber' | 'cyan' | 'volt'} size="sm" />
          <Stat label={tr('Step', '步数')} value={`${step} / ${total}`} tone="ink" />
          <Stat label={tr('Total steps 2(N-1)', '总步数 2(N-1)')} value={total} unit={tr('steps', '步')} tone="cyan" />
          <Stat label={tr('Per-GPU traffic', '每卡通信量')} value={coeff.toFixed(2)} unit="× D" tone="volt" />
          <div className="text-[12px] leading-relaxed text-ink3">
            <MathTex tex={`2\\cdot\\tfrac{${n}-1}{${n}}D = ${coeff.toFixed(2)}D`} />
          </div>
        </div>
      </div>
    </Widget>
  )
}
