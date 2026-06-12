import { useEffect, useMemo, useRef, useState } from 'react'
import { PlayBar, Segmented, Slider, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { fmtFlops, fmtInt } from '@/lib/format'

/** 每个 attention score 单元 ≈ 2·d_head FLOPs（一次 q·k 点积） */
const D_HEAD = 128

/**
 * LAB 01 有无 Cache 对比：
 * 左 = 无 cache，每步把整个因果三角矩阵全部重算（旧行染 rose = 冗余重算）；
 * 右 = 有 cache，每步只追加一行（旧行 amber = 静静躺在显存里）。
 * 下方 FLOPs 曲线：每步（二次 vs 线性）/ 累计（三次 vs 二次）。
 */
export default function CacheCompareLab() {
  const reduced = useReducedMotion()
  const [S, setS] = useState(32)
  const [t, setT] = useState(reduced ? 16 : 1) // 已生成 token 数（当前步）
  const [playing, setPlaying] = useState(!reduced)
  const [speed, setSpeed] = useState(1)
  const [mode, setMode] = useState<'step' | 'cum'>('step')
  const acc = useRef(0)

  useRafLoop((dt) => {
    acc.current += dt * speed
    const interval = 200
    while (acc.current >= interval) {
      acc.current -= interval
      setT((p) => (p >= S ? p : p + 1))
    }
  }, playing)

  // 播到尾自动停
  useEffect(() => {
    if (playing && t >= S) setPlaying(false)
  }, [playing, t, S])

  const data = useMemo(() => {
    const noStep: number[] = []
    const cacheStep: number[] = []
    const noCum: number[] = []
    const cacheCum: number[] = []
    let a = 0
    let b = 0
    for (let k = 1; k <= S; k++) {
      const ns = ((k * (k + 1)) / 2) * 2 * D_HEAD // 整个三角重算
      const cs = k * 2 * D_HEAD // 只算新一行
      a += ns
      b += cs
      noStep.push(ns)
      cacheStep.push(cs)
      noCum.push(a)
      cacheCum.push(b)
    }
    return { noStep, cacheStep, noCum, cacheCum }
  }, [S])

  const idx = Math.min(t, S) - 1
  const noNow = data.noStep[idx]
  const cacheNow = data.cacheStep[idx]
  const noCumNow = data.noCum[idx]
  const cacheCumNow = data.cacheCum[idx]

  const reset = () => {
    acc.current = 0
    setPlaying(false)
    setT(1)
  }

  /* ── 矩阵网格 ── */
  const P = 12 // 格距
  const cellGrid = (cached: boolean) => {
    const cells = []
    for (let i = 0; i < S; i++) {
      for (let j = 0; j <= i; j++) {
        const x = j * P + 1
        const y = i * P + 1
        if (i >= t) {
          // 还没生成到这一行：仅画轮廓
          cells.push(
            <rect key={`${i}-${j}`} x={x} y={y} width={P - 2} height={P - 2} rx={1.5}
              fill="none" stroke="var(--color-line)" strokeWidth={1} />,
          )
        } else if (i === t - 1) {
          // 本步新算的一行：volt
          cells.push(
            <rect key={`${i}-${j}`} x={x} y={y} width={P - 2} height={P - 2} rx={1.5}
              fill="var(--color-volt)" fillOpacity={0.85} />,
          )
        } else if (cached) {
          // 缓存命中：amber，静静躺着
          cells.push(
            <rect key={`${i}-${j}`} x={x} y={y} width={P - 2} height={P - 2} rx={1.5}
              fill="var(--color-amber)" fillOpacity={0.28} />,
          )
        } else {
          // 冗余重算：rose
          cells.push(
            <rect key={`${i}-${j}`} x={x} y={y} width={P - 2} height={P - 2} rx={1.5}
              fill="var(--color-rose)" fillOpacity={0.62} />,
          )
        }
      }
    }
    return cells
  }

  const vb = S * P + 2
  const noCells = (t * (t + 1)) / 2
  const cacheCells = t

  /* ── FLOPs 曲线 ── */
  const W = 340
  const H = 130
  const padL = 8
  const padB = 14
  const plotW = W - padL - 6
  const plotH = H - padB - 8
  const noArr = mode === 'step' ? data.noStep : data.noCum
  const cacheArr = mode === 'step' ? data.cacheStep : data.cacheCum
  const maxV = noArr[S - 1]
  const px = (k: number) => padL + (S === 1 ? 0 : (k / (S - 1)) * plotW)
  const py = (v: number) => 8 + plotH - (v / maxV) * plotH
  const toPts = (arr: number[]) => arr.map((v, k) => `${px(k).toFixed(1)},${py(v).toFixed(1)}`).join(' ')

  return (
    <Widget
      index={1}
      title="有无 Cache 对比"
      subtitle="同一段生成，两种算法的计算量"
      onReset={reset}
      footer={
        <>
          左侧每一步都把整个 score 三角重新点亮 —— 旧行的<span className="text-rose">玫红</span>就是白白烧掉的算力；
          右侧旧行只是<span className="text-amber">琥珀色</span>地躺在显存里等着被读。每步代价 O(t²·d) vs O(t·d)，
          整个序列生成下来累计就是 O(S³·d) vs O(S²·d)。
        </>
      }
    >
      <div className="mb-4 flex flex-wrap items-end gap-x-6 gap-y-3">
        <PlayBar playing={playing} onToggle={() => {
          if (!playing && t >= S) setT(1)
          setPlaying(!playing)
        }} onStep={() => setT((p) => Math.min(S, p + 1))} onReset={reset} speed={speed} onSpeed={setSpeed} />
        <Slider className="w-44" label="目标长度 S" value={S} min={8} max={48} step={4}
          onChange={(v) => { setS(v); setT((p) => Math.min(p, v)) }} unit="token" />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="microlabel text-rose">无 KV CACHE</span>
            <span className="font-mono text-[11px] tabular-nums text-ink3">
              本步重算 <span className="text-rose">{fmtInt(noCells)}</span> 单元
            </span>
          </div>
          <svg viewBox={`0 0 ${vb} ${vb}`} className="w-full rounded-md border border-line bg-bg2">
            {cellGrid(false)}
          </svg>
        </div>
        <div>
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="microlabel text-volt">有 KV CACHE</span>
            <span className="font-mono text-[11px] tabular-nums text-ink3">
              本步只算 <span className="text-volt">{fmtInt(cacheCells)}</span> 单元
            </span>
          </div>
          <svg viewBox={`0 0 ${vb} ${vb}`} className="w-full rounded-md border border-line bg-bg2">
            {cellGrid(true)}
          </svg>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-[1fr_auto]">
        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <span className="microlabel">FLOPS 曲线（d_head = 128）</span>
            <Segmented
              options={[
                { value: 'step', label: '每步' },
                { value: 'cum', label: '累计' },
              ]}
              value={mode}
              onChange={setMode}
            />
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-md border border-line bg-bg2">
            {/* 网格基线 */}
            <line x1={padL} y1={8 + plotH} x2={W - 6} y2={8 + plotH} stroke="var(--color-line2)" strokeWidth={1} />
            <polyline points={toPts(noArr)} fill="none" stroke="var(--color-rose)" strokeWidth={1.8} />
            <polyline points={toPts(cacheArr)} fill="none" stroke="var(--color-volt)" strokeWidth={1.8} />
            {/* 当前步标记 */}
            <line x1={px(idx)} y1={8} x2={px(idx)} y2={8 + plotH} stroke="var(--color-line2)" strokeWidth={1} strokeDasharray="3 3" />
            <circle cx={px(idx)} cy={py(noArr[idx])} r={3.2} fill="var(--color-rose)" />
            <circle cx={px(idx)} cy={py(cacheArr[idx])} r={3.2} fill="var(--color-volt)" />
            <text x={padL} y={H - 3} fontSize={8.5} className="font-mono" fill="var(--color-ink3)">t=1</text>
            <text x={W - 6} y={H - 3} fontSize={8.5} textAnchor="end" className="font-mono" fill="var(--color-ink3)">t={S}</text>
            <text x={W - 6} y={14} fontSize={8.5} textAnchor="end" className="font-mono" fill="var(--color-rose)">
              {mode === 'step' ? '每步 ~t²（二次）' : '累计 ~S³'}
            </text>
            <text x={W - 6} y={26} fontSize={8.5} textAnchor="end" className="font-mono" fill="var(--color-volt)">
              {mode === 'step' ? '每步 ~t（线性）' : '累计 ~S²'}
            </text>
          </svg>
        </div>
        <div className="grid grid-cols-2 content-start gap-x-8 gap-y-3 lg:grid-cols-1">
          <Stat label={`当前步 t = ${t}`} value={t} unit={`/ ${S}`} size="sm" />
          <Stat label="本步 FLOPS（无 / 有）" size="sm" tone="rose"
            value={<>{fmtFlops(noNow)} <span className="text-ink3">/</span> <span className="text-volt">{fmtFlops(cacheNow)}</span></>} />
          <Stat label="累计 FLOPS（无 / 有）" size="sm" tone="rose"
            value={<>{fmtFlops(noCumNow)} <span className="text-ink3">/</span> <span className="text-volt">{fmtFlops(cacheCumNow)}</span></>} />
          <Stat label="累计节省" value={`${(noCumNow / cacheCumNow).toFixed(1)}×`} size="sm" tone="volt" />
        </div>
      </div>
    </Widget>
  )
}
