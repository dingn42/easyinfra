import { useMemo, useState } from 'react'
import { Segmented, Slider, Stat, Widget } from '@/components/ui'
import { useT } from '@/lib/i18n'

/** LAB 01 合并访存实验台：warp 访问模式 → 32B 段事务 */

const LANES = 32
const W = 720
const LANE_Y = 36
const STRIP_Y = 140
const STRIP_H = 34
const VH = 200

type Mode = 'seq' | 'stride' | 'random'

/** 固定种子伪随机（mulberry32），保证每次打开页面随机模式长一个样 */
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 随机模式：32 个 lane 在 256 个 float（1 KB / 32 段）里乱抓 */
const RANDOM_FLOATS: number[] = (() => {
  const rnd = mulberry32(0x04c0a1e5)
  return Array.from({ length: LANES }, () => Math.floor(rnd() * 256))
})()

const DEFAULT_MODE: Mode = 'seq'
const DEFAULT_STRIDE = 2

export function CoalescingLab() {
  const t = useT()
  const [mode, setMode] = useState<Mode>(DEFAULT_MODE)
  const [stride, setStride] = useState(DEFAULT_STRIDE)

  const { floats, totalSegs, segSet } = useMemo(() => {
    const fs = Array.from({ length: LANES }, (_, i) =>
      mode === 'seq' ? i : mode === 'stride' ? i * stride : RANDOM_FLOATS[i],
    )
    const set = new Set(fs.map((f) => Math.floor((f * 4) / 32)))
    const maxSeg = fs.reduce((m, f) => Math.max(m, Math.floor((f * 4) / 32)), 0)
    return { floats: fs, totalSegs: mode === 'random' ? 32 : maxSeg + 1, segSet: set }
  }, [mode, stride])

  const txn = segSet.size
  const totalBytes = totalSegs * 32
  const util = 128 / (txn * 32)
  const segW = W / totalSegs
  const cellW = Math.max(W / (totalSegs * 8), 1.6)
  const laneX = (i: number) => ((i + 0.5) * W) / LANES
  const cellX = (f: number) => ((f * 4) / totalBytes) * W
  const tone = txn <= 4 ? 'volt' : txn <= 8 ? 'amber' : 'rose'

  const reset = () => {
    setMode(DEFAULT_MODE)
    setStride(DEFAULT_STRIDE)
  }

  return (
    <Widget
      index={1}
      title={t('Coalescing lab', '合并访存实验台')}
      subtitle={t('How a warp access pattern decides the 32B transaction count', 'warp 访问模式如何决定 32B 事务数')}
      wide
      onReset={reset}
      footer={
        <>
          {t(
            <>
              Sequential -&gt; 4 txn / 100%; stride=2 -&gt; 8 txn / 50%; stride&ge;8 -&gt; 32 txn / 12.5%, the
              floor (each 32B sector holds only 4B anyone wants). The AoS field access in the next section is
              this same stride pattern out in the wild, with stride = struct size.
            </>,
            <>
              连续 → 4 txn / 100%；stride=2 → 8 txn / 50%；stride≥8 → 32 txn / 12.5% 见底（每个 32B
              段里只有 4B 有人要）。下一节的 AoS 取字段就是这套 stride 在真实代码里的样子，stride = struct 大小。
            </>,
          )}
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <Segmented<Mode>
          options={[
            { value: 'seq', label: t('Sequential', '连续') },
            { value: 'stride', label: t('Strided', '跨步') },
            { value: 'random', label: t('Random', '随机') },
          ]}
          value={mode}
          onChange={setMode}
        />
        <Slider
          className="w-56 max-w-full"
          label="STRIDE"
          value={stride}
          min={1}
          max={32}
          onChange={setStride}
          disabled={mode !== 'stride'}
          unit={t('elems', '元素')}
        />
        <span className="font-mono text-[11px] text-ink3">
          {mode === 'seq' && t('a[tid] — textbook coalescing', 'a[tid] —— 教科书式合并')}
          {mode === 'stride' &&
            t(`a[tid * ${stride}] — one float every ${stride}`, `a[tid * ${stride}] —— 每隔 ${stride} 个 float 取一个`)}
          {mode === 'random' && t('a[idx[tid]] — fixed-seed scattered gather', 'a[idx[tid]] —— 固定种子的乱序 gather')}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${VH}`} className="mt-4 w-full font-mono">
        <g className="text-ink3" fontSize={10} fill="currentColor">
          <text x={0} y={12}>
            {t('WARP: 32 LANES, 4 B (float) per request', 'WARP：32 LANE，每个请求 4 B（float）')}
          </text>
          <text x={0} y={STRIP_Y - 10}>
            {t(
              'GLOBAL MEMORY (each cell = one 32 B sector; amber = transaction fired)',
              'GLOBAL MEMORY（每格 = 一个 32 B 段；amber = 被触发的事务）',
            )}
          </text>
          <text x={0} y={STRIP_Y + STRIP_H + 16}>
            0 B
          </text>
          <text x={W} y={STRIP_Y + STRIP_H + 16} textAnchor="end">
            {totalBytes} B
          </text>
        </g>

        {/* lane → 地址连线 */}
        <g>
          {floats.map((f, i) => (
            <line
              key={i}
              x1={laneX(i)}
              y1={LANE_Y + 8}
              x2={cellX(f) + cellW / 2}
              y2={STRIP_Y}
              stroke="var(--color-cyan)"
              strokeOpacity={0.42}
              strokeWidth={1}
            />
          ))}
        </g>

        {/* 32B 段条带 */}
        <g>
          {Array.from({ length: totalSegs }, (_, s) => {
            const hit = segSet.has(s)
            return (
              <rect
                key={s}
                x={s * segW}
                y={STRIP_Y}
                width={segW}
                height={STRIP_H}
                fill={hit ? 'var(--color-amber)' : 'none'}
                fillOpacity={hit ? 0.16 : 0}
                stroke={hit ? 'var(--color-amber)' : 'var(--color-line)'}
                strokeOpacity={hit ? 0.6 : 1}
              />
            )
          })}
        </g>

        {/* 被读取的 float（cyan = 读的数据） */}
        <g>
          {floats.map((f, i) => (
            <rect
              key={i}
              x={cellX(f)}
              y={STRIP_Y + 1.5}
              width={cellW}
              height={STRIP_H - 3}
              fill="var(--color-cyan)"
              fillOpacity={0.9}
            />
          ))}
        </g>

        {/* lane 圆点 */}
        <g>
          {Array.from({ length: LANES }, (_, i) => (
            <g key={i}>
              <circle cx={laneX(i)} cy={LANE_Y} r={5} fill="var(--color-volt)" fillOpacity={0.85} />
              {i % 8 === 0 && (
                <text
                  x={laneX(i)}
                  y={LANE_Y - 11}
                  textAnchor="middle"
                  fontSize={9}
                  fill="currentColor"
                  className="text-ink3"
                >
                  {i}
                </text>
              )}
            </g>
          ))}
          <text
            x={laneX(31)}
            y={LANE_Y - 11}
            textAnchor="middle"
            fontSize={9}
            fill="currentColor"
            className="text-ink3"
          >
            31
          </text>
        </g>
      </svg>

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label={t('Bytes requested', '请求字节')} value={128} unit="B" tone="cyan" />
        <Stat label={t('Transactions', '事务数')} value={txn} unit="txn" tone={tone} size="lg" />
        <Stat label={t('Bytes moved', '实际搬运')} value={txn * 32} unit="B" tone="amber" />
        <Stat label={t('Bandwidth utilization', '带宽利用率')} value={`${Math.round(util * 100)}%`} tone={tone} size="lg" />
      </div>
    </Widget>
  )
}
