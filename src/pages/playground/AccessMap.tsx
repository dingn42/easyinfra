import { useMemo, useRef, useState } from 'react'
import { Slider } from '@/components/ui'
import type { AccessEvent, SimStats } from '../../lib/cudasim/types'

const WARP_WINDOW = 2 // 一次最多显示 2 个 warp（64 条 lane），可用滑杆平移窗口
const COLS_MAX = 64 // 步数多时按 step 聚合降采样

const C_READ = '#59d8ea' // global read → cyan
const C_WRITE = '#b8f53d' // global write → volt
const C_SHARED = '#ffb454' // shared → amber

interface Cell {
  gr: number
  gw: number
  sh: number
  total: number
  first: AccessEvent
}

function cellColor(c: Cell): string {
  if (c.gw >= c.gr && c.gw >= c.sh) return C_WRITE
  if (c.sh >= c.gr) return C_SHARED
  return C_READ
}

/** 访问轨迹热图：x = 逻辑步，y = lane（按 warp 分隔） */
export function AccessMap({
  accesses,
  stats,
  truncated,
}: {
  accesses: AccessEvent[]
  stats: SimStats | null
  truncated: boolean
}) {
  const [warpStart, setWarpStart] = useState(0)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [tip, setTip] = useState<{ x: number; y: number; lines: string[] } | null>(null)

  const warpCount = stats?.warps ?? accesses.reduce((m, a) => Math.max(m, a.warpId + 1), 1)
  const shownWarps = Math.min(WARP_WINDOW, warpCount)
  const start = Math.min(warpStart, Math.max(0, warpCount - shownWarps))

  const { cells, cols, bucket, maxStep, rows } = useMemo(() => {
    const maxStep = Math.max(1, stats?.maxSteps ?? accesses.reduce((m, a) => Math.max(m, a.step), 1))
    const bucket = Math.max(1, Math.ceil(maxStep / COLS_MAX))
    const cols = Math.ceil(maxStep / bucket)
    const rows = shownWarps * 32
    const cells = new Map<number, Cell>()
    for (const a of accesses) {
      const w = a.warpId - start
      if (w < 0 || w >= shownWarps) continue
      const col = Math.min(cols - 1, Math.floor((a.step - 1) / bucket))
      const row = w * 32 + a.laneId
      const key = row * cols + col
      let c = cells.get(key)
      if (!c) cells.set(key, (c = { gr: 0, gw: 0, sh: 0, total: 0, first: a }))
      if (a.space === 'shared') c.sh++
      else if (a.kind === 'read') c.gr++
      else c.gw++
      c.total++
    }
    return { cells, cols, bucket, maxStep, rows }
  }, [accesses, stats, start, shownWarps])

  if (accesses.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-14 text-center">
        <span className="microlabel">NO TRACE</span>
        <p className="text-[13px] text-ink3">这次运行没有产生内存访问事件。</p>
      </div>
    )
  }

  // 几何：viewBox 自适应；渲染尺寸用缩放因子限制，避免步数极少时拉成细长条
  const labelW = 40
  const topH = 20
  const cellW = Math.max(7, Math.min(36, Math.floor(560 / cols)))
  const cellH = rows <= 32 ? 10 : 7
  const vbW = labelW + cols * cellW + 6
  const vbH = topH + rows * cellH + 6
  const scale = Math.min(620 / vbW, 520 / vbH, 1.9)
  const tickEvery = Math.max(1, Math.ceil(cols / 8))

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget
    const rect = svg.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * vbW
    const py = ((e.clientY - rect.top) / rect.height) * vbH
    const col = Math.floor((px - labelW) / cellW)
    const row = Math.floor((py - topH) / cellH)
    if (col < 0 || col >= cols || row < 0 || row >= rows) return setTip(null)
    const c = cells.get(row * cols + col)
    if (!c) return setTip(null)
    const warp = start + Math.floor(row / 32)
    const lane = row % 32
    const s0 = col * bucket + 1
    const s1 = Math.min(maxStep, (col + 1) * bucket)
    const head = `W${warp} · lane ${lane} · step ${bucket > 1 ? `${s0}–${s1}` : s0}`
    const lines =
      c.total === 1
        ? [
            head,
            `${c.first.buffer}[${c.first.index}] ${c.first.space === 'shared' ? 'shared' : 'global'} ${
              c.first.kind === 'read' ? 'read' : 'write'
            } @L${c.first.line}`,
          ]
        : [head, `${c.gr} 读 / ${c.gw} 写 global · ${c.sh} shared（如 ${c.first.buffer}[${c.first.index}]@L${c.first.line}）`]
    const wrapRect = wrapRef.current?.getBoundingClientRect()
    setTip({
      x: e.clientX - (wrapRect?.left ?? rect.left) + 14,
      y: e.clientY - (wrapRect?.top ?? rect.top) + 14,
      lines,
    })
  }

  return (
    <div>
      {/* 图例 + 控制 */}
      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[11px] text-ink2">
        <span className="flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-[2px]" style={{ background: C_READ }} /> GLOBAL READ
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-[2px]" style={{ background: C_WRITE }} /> GLOBAL WRITE
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-[2px]" style={{ background: C_SHARED }} /> SHARED R/W
        </span>
        {bucket > 1 && <span className="text-ink3">每格聚合 {bucket} 步</span>}
        {truncated && <span className="text-amber">⚠ 轨迹被截断（事件过多）</span>}
      </div>
      {warpCount > WARP_WINDOW && (
        <div className="mb-3 max-w-[320px]">
          <Slider
            label="WARP 窗口"
            value={start}
            min={0}
            max={warpCount - shownWarps}
            onChange={setWarpStart}
            fmt={(v) => `W${v}–W${v + shownWarps - 1}`}
            unit={` / ${warpCount} warps`}
          />
        </div>
      )}

      <div ref={wrapRef} className="relative" onMouseLeave={() => setTip(null)}>
        <svg
          viewBox={`0 0 ${vbW} ${vbH}`}
          className="w-full rounded-md border border-line bg-bg2"
          style={{ maxWidth: Math.max(220, vbW * scale) }}
          onMouseMove={onMove}
        >
          {/* step 轴刻度 */}
          {Array.from({ length: Math.ceil(cols / tickEvery) }, (_, i) => i * tickEvery).map((c) => (
            <text
              key={c}
              x={labelW + c * cellW + 1}
              y={topH - 7}
              fontSize={7.5}
              fill="currentColor"
              className="font-mono text-ink3"
            >
              {c * bucket + 1}
            </text>
          ))}
          <text x={vbW - 4} y={topH - 7} fontSize={7.5} textAnchor="end" fill="currentColor" className="font-mono text-ink3">
            STEP →
          </text>

          {/* warp 分隔线与标签 */}
          {Array.from({ length: shownWarps }, (_, w) => (
            <g key={w}>
              <text
                x={labelW - 6}
                y={topH + w * 32 * cellH + 9}
                fontSize={8}
                textAnchor="end"
                fill="currentColor"
                className="font-mono text-ink3"
              >
                W{start + w}
              </text>
              {w > 0 && (
                <line
                  x1={labelW - 2}
                  x2={vbW - 4}
                  y1={topH + w * 32 * cellH}
                  y2={topH + w * 32 * cellH}
                  stroke="var(--color-line2)"
                  strokeWidth={0.8}
                  strokeDasharray="3 3"
                />
              )}
            </g>
          ))}

          {/* 色块 */}
          {Array.from(cells.entries()).map(([key, c]) => {
            const row = Math.floor(key / cols)
            const col = key % cols
            return (
              <rect
                key={key}
                x={labelW + col * cellW + 0.6}
                y={topH + row * cellH + 0.5}
                width={cellW - 1.2}
                height={Math.max(1, cellH - 1)}
                rx={1}
                fill={cellColor(c)}
                opacity={0.92}
              />
            )
          })}
        </svg>

        {tip && (
          <div
            className="pointer-events-none absolute z-10 rounded-md border border-line2 bg-bg2/95 px-2.5 py-1.5 font-mono text-[11px] leading-relaxed shadow-lg"
            style={{ left: tip.x, top: tip.y, maxWidth: 320 }}
          >
            <div className="text-ink3">{tip.lines[0]}</div>
            <div className="text-ink">{tip.lines[1]}</div>
          </div>
        )}
      </div>

      <p className="mt-2 text-[11.5px] leading-relaxed text-ink3">
        每行一条 lane（虚线分隔 warp），横轴是逻辑步。竖直成列的色块 = 同 warp 同拍访问 ——
        合并访存与 lockstep 的直观形态；错开的色块 = 分支分化把线程拖出了同拍。
      </p>
    </div>
  )
}
