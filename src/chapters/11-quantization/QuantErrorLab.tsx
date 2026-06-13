import { useMemo, useState } from 'react'
import { Segmented, Slider, Stat, Widget } from '@/components/ui'
import { clamp, pct } from '@/lib/format'
import { useT } from '@/lib/i18n'

/* ────────── 确定性数据生成（固定种子） ────────── */

const N = 4096

function mulberry32(seed: number) {
  let t0 = seed
  return () => {
    let t = (t0 += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 高斯主体（seed 42）+ 固定位置的 outlier（seed 1337），全程确定性。 */
function genWeights(sigma: number, outPct: number, outMag: number): Float64Array {
  const rand = mulberry32(42)
  const arr = new Float64Array(N)
  for (let i = 0; i < N; i += 2) {
    const r = Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-12)))
    const th = 2 * Math.PI * rand()
    arr[i] = r * Math.cos(th) * sigma
    if (i + 1 < N) arr[i + 1] = r * Math.sin(th) * sigma
  }
  const nOut = Math.round((N * outPct) / 100)
  const r2 = mulberry32(1337)
  const used = new Set<number>()
  for (let j = 0; j < nOut; j++) {
    let idx = Math.floor(r2() * N)
    while (used.has(idx)) idx = (idx + 1) % N
    used.add(idx)
    const sign = r2() < 0.5 ? -1 : 1
    arr[idx] = sign * sigma * outMag * (0.85 + 0.3 * r2())
  }
  return arr
}

/* ────────── 量化方案 ────────── */

type Scheme = 'int8-absmax' | 'int8-p999' | 'int4-absmax' | 'int4-g128'

const SCHEMES: { value: Scheme; label: string }[] = [
  { value: 'int8-absmax', label: 'INT8 absmax' },
  { value: 'int8-p999', label: 'INT8 p99.9' },
  { value: 'int4-absmax', label: 'INT4 absmax' },
  { value: 'int4-g128', label: 'INT4 g=128' },
]

interface QuantResult {
  mse: number
  snrDb: number
  clipped: number
  scales: number[]
  qmax: number
}

function quantize(x: Float64Array, scheme: Scheme): QuantResult {
  const qmax = scheme.startsWith('int8') ? 127 : 7
  const scales: number[] = []
  let sumE = 0
  let sumX = 0
  let nClip = 0
  const run = (start: number, end: number, s: number) => {
    scales.push(s)
    for (let i = start; i < end; i++) {
      const q = Math.round(x[i] / s)
      if (q > qmax || q < -qmax) nClip++
      const qc = clamp(q, -qmax, qmax)
      const err = x[i] - qc * s
      sumE += err * err
      sumX += x[i] * x[i]
    }
  }
  if (scheme === 'int4-g128') {
    for (let g = 0; g < N; g += 128) {
      let am = 0
      for (let i = g; i < g + 128; i++) am = Math.max(am, Math.abs(x[i]))
      run(g, g + 128, (am || 1) / qmax)
    }
  } else {
    let s: number
    if (scheme === 'int8-p999') {
      const abs = Array.from(x, Math.abs).sort((a, b) => a - b)
      s = (abs[Math.floor(0.999 * (N - 1))] || 1) / qmax
    } else {
      let am = 0
      for (let i = 0; i < N; i++) am = Math.max(am, Math.abs(x[i]))
      s = (am || 1) / qmax
    }
    run(0, N, s)
  }
  return {
    mse: sumE / N,
    snrDb: sumE === 0 ? Infinity : 10 * Math.log10(sumX / sumE),
    clipped: nClip / N,
    scales,
    qmax,
  }
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]
}

/* ────────── 组件 ────────── */

const DEFAULTS = { sigma: 1.0, outPct: 0.5, outMag: 12 }

export function QuantErrorLab() {
  const t = useT()
  const [sigma, setSigma] = useState(DEFAULTS.sigma)
  const [outPct, setOutPct] = useState(DEFAULTS.outPct)
  const [outMag, setOutMag] = useState(DEFAULTS.outMag)
  const [scheme, setScheme] = useState<Scheme>('int8-absmax')

  const data = useMemo(() => genWeights(sigma, outPct, outMag), [sigma, outPct, outMag])
  const res = useMemo(() => quantize(data, scheme), [data, scheme])
  const allSnr = useMemo(
    () => SCHEMES.map((s) => ({ ...s, snr: quantize(data, s.value).snrDb })),
    [data],
  )

  const reset = () => {
    setSigma(DEFAULTS.sigma)
    setOutPct(DEFAULTS.outPct)
    setOutMag(DEFAULTS.outMag)
    setScheme('int8-absmax')
  }

  /* ── 直方图 ── */
  const BINS = 101
  const hist = useMemo(() => {
    let am = 0
    for (let i = 0; i < N; i++) am = Math.max(am, Math.abs(data[i]))
    const R = Math.max(am * 1.04, sigma * 4.5)
    const counts = new Array<number>(BINS).fill(0)
    for (let i = 0; i < N; i++) {
      const b = clamp(Math.floor(((data[i] + R) / (2 * R)) * BINS), 0, BINS - 1)
      counts[b]++
    }
    return { R, counts, maxC: Math.max(...counts) }
  }, [data, sigma])

  const W = 720
  const PL = 14
  const PR = 706
  const PT = 16
  const PB = 196
  const xOf = (v: number) => PL + ((v + hist.R) / (2 * hist.R)) * (PR - PL)

  const bodyEdge = 4 * sigma
  const isGroup = scheme === 'int4-g128'
  const medScale = median(res.scales)
  const maxScale = Math.max(...res.scales)

  /** 单一 scale 方案的格点；group 方案画中位组（volt）+ 被 outlier 撑宽的组（rose） */
  const gridLevels = useMemo(() => {
    const levels: { v: number; wasted: boolean }[] = []
    const s = isGroup ? medScale : res.scales[0]
    for (let q = -res.qmax; q <= res.qmax; q++) {
      const v = q * s
      if (Math.abs(v) <= hist.R) levels.push({ v, wasted: Math.abs(v) > bodyEdge })
    }
    return levels
  }, [res, hist.R, bodyEdge, isGroup, medScale])

  const outlierGroupLevels = useMemo(() => {
    if (!isGroup || maxScale < medScale * 1.3) return []
    const levels: number[] = []
    for (let q = -res.qmax; q <= res.qmax; q++) {
      const v = q * maxScale
      if (Math.abs(v) <= hist.R) levels.push(v)
    }
    return levels
  }, [isGroup, maxScale, medScale, res.qmax, hist.R])

  const wastedRatio = gridLevels.length ? gridLevels.filter((l) => l.wasted).length / gridLevels.length : 0

  const binW = (PR - PL) / BINS

  return (
    <Widget
      index={2}
      title={t('Quantization Error Playground', '量化误差实验场')}
      subtitle={t('How a single outlier wrecks a whole tensor', '一个 outlier 如何毁掉整个 tensor 的精度')}
      onReset={reset}
      footer={t(
        <>
          The histogram (<span className="text-cyan">cyan</span>, √-scaled height) is 4096 "weights";
          the vertical lines are quantization grid points:
          <span className="text-volt"> volt = grid points inside the ±4σ body that actually do work</span>,
          <span className="text-rose"> rose = points stretched out by the outlier with almost no values to land on</span>.
          In g=128 mode the grid of the group containing the outlier is drawn separately (dashed rose) —
          the damage is locked inside its 128 numbers. The data uses a fixed seed, so results are reproducible.
        </>,
        <>
          直方图（<span className="text-cyan">cyan</span>，√ 高度刻度）是 4096 个「权重」；竖线是量化格点：
          <span className="text-volt"> volt = 落在主体 ±4σ 内、真正干活的格点</span>，
          <span className="text-rose"> rose = 被 outlier 撑出去、几乎没值可接的格点</span>。
          g=128 模式下另画出含 outlier 的那一组的格点（玫瑰色）—— 灾难被锁在 128 个数里。数据固定种子，结果可复现。
        </>,
      )}
    >
      {/* 控件 */}
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-3">
        <Slider label={t('BODY σ', '主体 σ')} value={sigma} min={0.2} max={3} step={0.05} onChange={setSigma} fmt={(v) => v.toFixed(2)} />
        <Slider label={t('OUTLIER FRACTION', 'OUTLIER 比例')} value={outPct} min={0} max={2} step={0.05} onChange={setOutPct} fmt={(v) => v.toFixed(2)} unit="%" />
        <Slider label={t('OUTLIER MAGNITUDE', 'OUTLIER 幅度')} value={outMag} min={2} max={40} step={1} onChange={setOutMag} fmt={(v) => `×${v}`} unit="σ" />
      </div>
      <div className="mt-3">
        <Segmented options={SCHEMES} value={scheme} onChange={setScheme} block />
      </div>

      {/* 直方图 + 格点 */}
      <svg viewBox={`0 0 ${W} 224`} className="mt-4 w-full">
        {/* 被浪费的格点区底色 */}
        {!isGroup && wastedRatio > 0 && (
          <>
            <rect x={PL} y={PT} width={Math.max(0, xOf(-bodyEdge) - PL)} height={PB - PT} fill="var(--color-rose)" opacity={0.06} />
            <rect x={xOf(bodyEdge)} y={PT} width={Math.max(0, PR - xOf(bodyEdge))} height={PB - PT} fill="var(--color-rose)" opacity={0.06} />
          </>
        )}
        {/* 量化格点 */}
        {gridLevels.map((l, i) => (
          <line
            key={i}
            x1={xOf(l.v)}
            y1={PT}
            x2={xOf(l.v)}
            y2={PB}
            stroke={l.wasted ? 'var(--color-rose)' : 'var(--color-volt)'}
            strokeWidth={res.qmax > 7 ? 0.7 : 1.2}
            opacity={l.wasted ? 0.55 : 0.5}
          />
        ))}
        {outlierGroupLevels.map((v, i) => (
          <line key={`og${i}`} x1={xOf(v)} y1={PT} x2={xOf(v)} y2={PB} stroke="var(--color-rose)" strokeWidth={1.2} strokeDasharray="3 3" opacity={0.7} />
        ))}
        {/* 直方图 */}
        {hist.counts.map((c, i) =>
          c === 0 ? null : (
            <rect
              key={i}
              x={PL + i * binW + 0.3}
              y={PB - Math.sqrt(c / hist.maxC) * (PB - PT)}
              width={binW - 0.6}
              height={Math.sqrt(c / hist.maxC) * (PB - PT)}
              fill="var(--color-cyan)"
              opacity={0.5}
            />
          ),
        )}
        {/* 轴与刻度 */}
        <g className="text-ink3">
          <line x1={PL} y1={PB} x2={PR} y2={PB} stroke="currentColor" strokeWidth={1} opacity={0.7} />
          {[-hist.R, -bodyEdge, 0, bodyEdge, hist.R].map((v, i) => (
            <g key={i}>
              <line x1={xOf(v)} y1={PB} x2={xOf(v)} y2={PB + 5} stroke="currentColor" strokeWidth={1} />
              <text x={xOf(v)} y={PB + 18} fontSize={10} fontFamily="var(--font-mono)" fill="currentColor" textAnchor="middle">
                {Math.abs(Math.abs(v) - bodyEdge) < 1e-9 ? `${v < 0 ? '−' : '+'}4σ` : v.toFixed(1)}
              </text>
            </g>
          ))}
        </g>
      </svg>

      {/* 读数 */}
      <div className="mt-4 flex flex-wrap gap-x-8 gap-y-3">
        <Stat label="MSE" value={res.mse === 0 ? '0' : res.mse.toExponential(2)} tone="cyan" />
        <Stat
          label="SNR"
          value={Number.isFinite(res.snrDb) ? res.snrDb.toFixed(1) : '∞'}
          unit="dB"
          tone={res.snrDb > 20 ? 'volt' : res.snrDb > 10 ? 'amber' : 'rose'}
        />
        <Stat label={t('Clipped fraction', '被裁剪比例')} value={pct(res.clipped, 2)} tone={res.clipped > 0 ? 'rose' : 'ink'} />
        {isGroup ? (
          <Stat label={t('Cross-group scale spread', '组间 scale 极差')} value={`×${(maxScale / medScale).toFixed(1)}`} tone="amber" />
        ) : (
          <Stat label={t('Grid points outside body', '主体外的格点')} value={pct(wastedRatio, 0)} tone={wastedRatio > 0.3 ? 'rose' : 'ink'} />
        )}
      </div>

      {/* 四方案 SNR 对比 */}
      <div className="mt-5 space-y-1.5">
        <div className="microlabel mb-2">{t('SNR across all four schemes on the same data', '同一份数据下四种方案的 SNR 对比')}</div>
        {allSnr.map((s) => {
          const active = s.value === scheme
          return (
            <button key={s.value} onClick={() => setScheme(s.value)} className="group flex w-full items-center gap-2 text-left">
              <span className={`w-[92px] shrink-0 font-mono text-[10.5px] ${active ? 'text-volt' : 'text-ink3 group-hover:text-ink2'}`}>
                {s.label}
              </span>
              <span className="h-2 flex-1 overflow-hidden rounded-sm bg-bg2">
                <span
                  className="block h-full rounded-sm transition-all duration-300"
                  style={{
                    width: pct(clamp(s.snr, 0, 48) / 48, 1),
                    background: active ? 'var(--color-volt)' : 'var(--color-line2)',
                  }}
                />
              </span>
              <span className={`w-16 shrink-0 text-right font-mono text-[11px] tabular-nums ${active ? 'text-volt' : 'text-ink3'}`}>
                {Number.isFinite(s.snr) ? s.snr.toFixed(1) : '∞'} dB
              </span>
            </button>
          )
        })}
      </div>
    </Widget>
  )
}
