import { useT } from '@/lib/i18n'

/**
 * 芯片 die 面积示意图（SEC1 静态图与 LAB02 共用的渲染器）。
 * 把 die 自上而下分成三个色带：控制逻辑（violet）/ 缓存（cyan）/ ALU 阵列（volt 小格）。
 * 纯函数布局 + 无状态 SVG，外部传入百分比即可。
 */

const W = 320
const H = 236
const PAD = 14
const BAND_GAP = 5

interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export interface DieLayout {
  W: number
  H: number
  ctrl: Rect
  cache: Rect
  alu: Rect
  /** 缓存条带（横向 slab） */
  slabs: Rect[]
  /** ALU 单元格 */
  cells: Rect[]
}

/** 根据三个百分比（总和应为 100）计算 die 布局；cols 控制 ALU 格子的粒度 */
export function layoutDie(ctrl: number, cache: number, alu: number, cols: number): DieLayout {
  const ix = PAD
  const iy = PAD
  const iw = W - PAD * 2
  const usable = H - PAD * 2 - BAND_GAP * 2
  const hC = (usable * ctrl) / 100
  const hK = (usable * cache) / 100
  const hA = usable - hC - hK
  const yC = iy
  const yK = iy + hC + BAND_GAP
  const yA = yK + hK + BAND_GAP

  // 缓存：若干横向 slab
  const slabs: Rect[] = []
  if (hK > 1.5) {
    const n = Math.max(1, Math.floor((hK + 4) / 22))
    const sh = (hK - (n - 1) * 4) / n
    for (let i = 0; i < n; i++) {
      slabs.push({ x: ix, y: yK + i * (sh + 4), w: iw, h: sh })
    }
  }

  // ALU：小格阵列，数量随面积变化
  const cells: Rect[] = []
  if (hA > 4) {
    const cgap = 4
    const cw = (iw - (cols - 1) * cgap) / cols
    let ch = Math.min(18, cw * 0.62)
    let rows = Math.floor((hA + cgap) / (ch + cgap))
    if (rows < 1) {
      rows = 1
      ch = hA
    }
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        cells.push({ x: ix + c * (cw + cgap), y: yA + r * (ch + cgap), w: cw, h: ch })
      }
    }
  }

  return {
    W,
    H,
    ctrl: { x: ix, y: yC, w: iw, h: hC },
    cache: { x: ix, y: yK, w: iw, h: hK },
    alu: { x: ix, y: yA, w: iw, h: hA },
    slabs,
    cells,
  }
}

/** ALU 单元数（LAB02 的读数与 SVG 保持一致） */
export function aluCellCount(ctrl: number, cache: number, alu: number, cols: number): number {
  return layoutDie(ctrl, cache, alu, cols).cells.length
}

export function ChipDie({
  ctrl,
  cache,
  alu,
  cols = 12,
  className = '',
}: {
  ctrl: number
  cache: number
  alu: number
  /** ALU 格子列数：4 = CPU 式大核，12 = GPU 式小核阵列 */
  cols?: number
  className?: string
}) {
  const t = useT()
  const L = layoutDie(ctrl, cache, alu, cols)
  const violet = 'var(--color-violet)'
  const cyan = 'var(--color-cyan)'
  const volt = 'var(--color-volt)'

  return (
    <svg
      viewBox={`0 0 ${L.W} ${L.H}`}
      className={`w-full ${className}`}
      role="img"
      aria-label={t('Schematic of how die area is split across the chip', '芯片 die 面积分配示意图')}
    >
      {/* die 基板 */}
      <rect x="3" y="3" width={L.W - 6} height={L.H - 6} rx="10" fill="var(--color-bg2)" stroke="var(--color-line2)" />

      {/* 控制逻辑 */}
      {L.ctrl.h > 1.5 && (
        <g>
          <rect
            x={L.ctrl.x}
            y={L.ctrl.y}
            width={L.ctrl.w}
            height={L.ctrl.h}
            rx="3"
            fill={violet}
            opacity="0.18"
          />
          <rect
            x={L.ctrl.x}
            y={L.ctrl.y}
            width={L.ctrl.w}
            height={L.ctrl.h}
            rx="3"
            fill="none"
            stroke={violet}
            strokeOpacity="0.6"
          />
          {L.ctrl.h >= 18 && (
            <text
              x={L.ctrl.x + L.ctrl.w / 2}
              y={L.ctrl.y + L.ctrl.h / 2}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="10.5"
              className="font-mono"
              fill={violet}
            >
              CONTROL {Math.round(ctrl)}%
            </text>
          )}
        </g>
      )}

      {/* 缓存 slab */}
      {L.slabs.map((s, i) => (
        <g key={i}>
          <rect x={s.x} y={s.y} width={s.w} height={s.h} rx="3" fill={cyan} opacity="0.16" />
          <rect x={s.x} y={s.y} width={s.w} height={s.h} rx="3" fill="none" stroke={cyan} strokeOpacity="0.5" />
        </g>
      ))}
      {L.cache.h >= 18 && (
        <text
          x={L.cache.x + L.cache.w / 2}
          y={L.cache.y + L.cache.h / 2}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize="10.5"
          className="font-mono"
          fill={cyan}
        >
          CACHE {Math.round(cache)}%
        </text>
      )}

      {/* ALU 阵列 */}
      {L.cells.map((c, i) => (
        <rect
          key={i}
          x={c.x}
          y={c.y}
          width={c.w}
          height={c.h}
          rx="2"
          fill={volt}
          opacity="0.32"
          stroke={volt}
          strokeOpacity="0.55"
          strokeWidth="0.75"
        />
      ))}
    </svg>
  )
}

/** 色块图例（SVG 下方的说明行） */
export function DieLegend({ ctrl, cache, alu }: { ctrl: number; cache: number; alu: number }) {
  const t = useT()
  const items = [
    { color: 'var(--color-violet)', label: `${t('Control', '控制逻辑')} ${Math.round(ctrl)}%` },
    { color: 'var(--color-cyan)', label: `${t('Cache', '缓存')} ${Math.round(cache)}%` },
    { color: 'var(--color-volt)', label: `ALU ${Math.round(alu)}%` },
  ]
  return (
    <div className="mt-2 flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5 font-mono text-[11px] tabular-nums text-ink2">
          <span className="size-2 rounded-[2px]" style={{ background: it.color, opacity: 0.7 }} />
          {it.label}
        </span>
      ))}
    </div>
  )
}
