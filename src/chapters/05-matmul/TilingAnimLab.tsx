import { useRef, useState, type CSSProperties } from 'react'
import { PlayBar, Stat, Widget } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { fmtInt } from '@/lib/format'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'

/**
 * LAB 02 Tiling 动画：N=12、T=4 的小棋盘。
 * 步进序列：选中 C 瓦片 → 第 k 段的 A/B 子块飞入 shared → C 瓦片 16 格逐个累加 → 下一段。
 * 右侧双计数器：naive 全局读取数 vs tiled 全局读取数，同步累计。
 */

const N = 12
const T = 4
const NT = N / T // 每个维度 3 个瓦片
const SEGS = NT // K 维分 3 段
const STEPS_PER_SEG = 1 + T * T // 1 步装载 + 16 步逐格累加
const TOTAL = NT * NT * SEGS * STEPS_PER_SEG // 459
const BASE_RATE = 14 // 1× 时每秒步数

// ── SVG 几何 ──
const CELL = 15
const MAT = N * CELL // 180
const GAP = 26
const X0 = MAT + GAP // B/C 的左边界 206
const Y0 = MAT + GAP // A/C 的上边界 206
const VB_W = X0 + MAT + 2
const VB_H = Y0 + MAT + 2
// shared 小棋盘（位于左上空白区）
const AS_X = 14
const BS_X = 104
const SH_Y = 92

interface Phase {
  done: boolean
  tile: number // 0..8，当前 C 瓦片（行优先）
  bi: number // 瓦片行
  bj: number // 瓦片列
  k: number // 0..2，当前 K 段
  within: number // 0=装载步；1..16=已点亮的累加格数
  loadsDone: number // 已完成的装载次数（每次 2*T*T 个 float）
  cellSegsDone: number // 已完成的「格×段」累加数（每个对应 naive 的 2T 次读取）
}

function derive(step: number): Phase {
  if (step >= TOTAL) {
    return { done: true, tile: NT * NT - 1, bi: NT - 1, bj: NT - 1, k: SEGS - 1, within: T * T, loadsDone: NT * NT * SEGS, cellSegsDone: NT * NT * SEGS * T * T }
  }
  const segIdx = Math.floor(step / STEPS_PER_SEG)
  const within = step % STEPS_PER_SEG
  const tile = Math.floor(segIdx / SEGS)
  const k = segIdx % SEGS
  return {
    done: false,
    tile,
    bi: Math.floor(tile / NT),
    bj: tile % NT,
    k,
    within,
    loadsDone: segIdx + (within >= 1 ? 1 : 0),
    cellSegsDone: segIdx * T * T + (within >= 1 ? within : 0),
  }
}

export function TilingAnimLab() {
  const t = useT()
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const acc = useRef(0)
  const reduced = useReducedMotion()

  useRafLoop((dt) => {
    acc.current += (dt / 1000) * BASE_RATE * speed
    if (acc.current >= 1) {
      const inc = Math.floor(acc.current)
      acc.current -= inc
      setStep((s) => {
        const n = Math.min(TOTAL, s + inc)
        if (n >= TOTAL) setPlaying(false)
        return n
      })
    }
  }, playing)

  const p = derive(step)
  const tiledReads = p.loadsDone * 2 * T * T // 每段装载 32 个 float
  const naiveReads = p.cellSegsDone * 2 * T // 每格每段的 4 次 FMA 在 naive 下要读 8 个 float
  const ratio = tiledReads > 0 ? naiveReads / tiledReads : 0

  const onStep = () => {
    setStep((s) => {
      if (s >= TOTAL) return TOTAL
      const within = s % STEPS_PER_SEG
      if (within === 0) return Math.min(TOTAL, s + T * T) // 装载 → 本段累加完成
      if (within < T * T) return Math.min(TOTAL, s - within + T * T) // 补完本段累加
      return Math.min(TOTAL, s + 1) // → 下一段装载
    })
  }
  const onReset = () => {
    setStep(0)
    setPlaying(false)
    acc.current = 0
  }

  // C 矩阵某格已累加的段数（0..3）
  const cSegs = (r: number, c: number): number => {
    const ti = Math.floor(r / T) * NT + Math.floor(c / T)
    if (p.done || ti < p.tile) return SEGS
    if (ti > p.tile) return 0
    const ci = (r % T) * T + (c % T)
    return p.k + (p.within >= 1 && ci < p.within ? 1 : 0)
  }

  const loading = !p.done && p.within === 0
  const curCell = !p.done && p.within >= 1 ? p.within - 1 : -1 // 当前正点亮的格（瓦片内索引）
  const curR = curCell >= 0 ? Math.floor(curCell / T) : -1
  const curC = curCell >= 0 ? curCell % T : -1

  // 飞入动画的位移（从 A/B 中的源子块 → shared 棋盘）
  const asDx = p.k * T * CELL - AS_X
  const asDy = Y0 + p.bi * T * CELL - SH_Y
  const bsDx = X0 + p.bj * T * CELL - BS_X
  const bsDy = p.k * T * CELL - SH_Y

  const flyStyle = (dx: number, dy: number): CSSProperties =>
    ({ '--fx': `${dx}px`, '--fy': `${dy}px` }) as CSSProperties

  return (
    <Widget
      index={2}
      title={t('Tiling Animation', 'Tiling 动画')}
      subtitle={t(
        `A small N=${N}, T=${T} board: watch one block reuse shared memory`,
        `N=${N}、T=${T} 的小棋盘：看一个 block 如何复用 shared memory`,
      )}
      wide
      onReset={onReset}
      footer={t(
        <>
          Each segment loads only 2×T²=32 floats from global memory, yet the tile's 16 output cells do 4 multiply-adds each. For the same
          work, naive would read 16×8=128 floats. The gap between the two counters converges to{' '}
          <span className="font-mono text-volt">T = {T}×</span>: tiling's payoff is proportional to the tile edge length.
        </>,
        <>
          每段装载只从全局内存读 2×T²=32 个 float，瓦片内 16 个输出格却各做 4 次乘加。
          同样的活，naive 要读 16×8=128 个 float。两个计数器的差距最终收敛到{' '}
          <span className="font-mono text-volt">T = {T}×</span>：tiling 的收益与瓦片边长成正比。
        </>,
      )}
    >
      {!reduced && (
        <style>{`
          @keyframes mmfly {
            from { transform: translate(var(--fx), var(--fy)); opacity: .3; }
            to { transform: translate(0, 0); opacity: 1; }
          }
          .mm-fly { animation: mmfly .45s cubic-bezier(.22, .8, .3, 1) both; }
        `}</style>
      )}
      <div className="flex flex-col gap-5 md:flex-row">
        <div className="min-w-0 flex-1">
          <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={t('Matrix-multiply tiling animation board', '矩阵乘法 tiling 动画棋盘')}>
            {/* ── 左上：状态与 shared memory ── */}
            <text x={AS_X} y={18} fontSize={10} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
              {p.done
                ? t('all 9 tiles done', '全部 9 个瓦片完成')
                : t(`C tile (${p.bi},${p.bj}) · segment ${p.k + 1}/${SEGS}`, `C 瓦片 (${p.bi},${p.bj}) · 第 ${p.k + 1}/${SEGS} 段`)}
            </text>
            <text
              x={AS_X}
              y={34}
              fontSize={10}
              fontFamily="var(--font-mono, monospace)"
              fill={p.done ? 'var(--color-volt)' : loading ? 'var(--color-cyan)' : 'var(--color-amber)'}
            >
              {p.done
                ? 'DONE'
                : loading
                  ? t('load: A, B sub-blocks → shared', '装载：A、B 子块 → shared')
                  : t(`accumulate: reuse shared (${p.within}/16 cells)`, `累加：复用 shared（${p.within}/16 格）`)}
            </text>
            <text x={AS_X} y={SH_Y - 10} fontSize={9} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
              SHARED MEMORY
            </text>

            {/* shared As / Bs 棋盘外框 */}
            {[AS_X, BS_X].map((x, i) => (
              <g key={i}>
                <rect
                  x={x - 2}
                  y={SH_Y - 2}
                  width={T * CELL + 3}
                  height={T * CELL + 3}
                  fill="none"
                  stroke="var(--color-line2)"
                  strokeDasharray="3 2"
                  rx={2}
                />
                <text
                  x={x}
                  y={SH_Y + T * CELL + 13}
                  fontSize={9}
                  fontFamily="var(--font-mono, monospace)"
                  fill={i === 0 ? 'var(--color-cyan)' : 'var(--color-amber)'}
                >
                  {i === 0 ? `As[${T}][${T}]` : `Bs[${T}][${T}]`}
                </text>
              </g>
            ))}

            {/* shared 内容：装载完成后填充（带飞入动画） */}
            {!p.done && p.within >= 1 && (
              <>
                <g key={`as-${p.tile}-${p.k}`} className={reduced ? '' : 'mm-fly'} style={flyStyle(asDx, asDy)}>
                  {Array.from({ length: T * T }, (_, i) => {
                    const r = Math.floor(i / T)
                    const c = i % T
                    const hot = curR === r // 当前累加格正在读 As 的这一行
                    return (
                      <rect
                        key={i}
                        x={AS_X + c * CELL}
                        y={SH_Y + r * CELL}
                        width={CELL - 2}
                        height={CELL - 2}
                        rx={1.5}
                        fill="var(--color-cyan)"
                        fillOpacity={hot ? 0.9 : 0.45}
                      />
                    )
                  })}
                </g>
                <g key={`bs-${p.tile}-${p.k}`} className={reduced ? '' : 'mm-fly'} style={flyStyle(bsDx, bsDy)}>
                  {Array.from({ length: T * T }, (_, i) => {
                    const r = Math.floor(i / T)
                    const c = i % T
                    const hot = curC === c // 当前累加格正在读 Bs 的这一列
                    return (
                      <rect
                        key={i}
                        x={BS_X + c * CELL}
                        y={SH_Y + r * CELL}
                        width={CELL - 2}
                        height={CELL - 2}
                        rx={1.5}
                        fill="var(--color-amber)"
                        fillOpacity={hot ? 0.9 : 0.45}
                      />
                    )
                  })}
                </g>
              </>
            )}

            {/* ── B（右上）── */}
            <text x={X0 + MAT - 14} y={12} fontSize={10} fontFamily="var(--font-mono, monospace)" fill="var(--color-amber)">
              B
            </text>
            {Array.from({ length: N * N }, (_, i) => {
              const r = Math.floor(i / N)
              const c = i % N
              const active = !p.done && r >= p.k * T && r < (p.k + 1) * T && c >= p.bj * T && c < (p.bj + 1) * T
              return (
                <rect
                  key={i}
                  x={X0 + c * CELL}
                  y={r * CELL}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={1.5}
                  fill="var(--color-amber)"
                  fillOpacity={active ? (loading ? 0.9 : 0.5) : 0.1}
                />
              )
            })}
            {!p.done && (
              <rect
                x={X0 + p.bj * T * CELL - 1.5}
                y={p.k * T * CELL - 1.5}
                width={T * CELL + 1}
                height={T * CELL + 1}
                fill="none"
                stroke="var(--color-amber)"
                strokeWidth={1.2}
                rx={2}
              />
            )}

            {/* ── A（左下）── */}
            <text x={4} y={Y0 + 12} fontSize={10} fontFamily="var(--font-mono, monospace)" fill="var(--color-cyan)">
              A
            </text>
            {Array.from({ length: N * N }, (_, i) => {
              const r = Math.floor(i / N)
              const c = i % N
              const active = !p.done && r >= p.bi * T && r < (p.bi + 1) * T && c >= p.k * T && c < (p.k + 1) * T
              return (
                <rect
                  key={i}
                  x={c * CELL}
                  y={Y0 + r * CELL}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={1.5}
                  fill="var(--color-cyan)"
                  fillOpacity={active ? (loading ? 0.9 : 0.5) : 0.1}
                />
              )
            })}
            {!p.done && (
              <rect
                x={p.k * T * CELL - 1.5}
                y={Y0 + p.bi * T * CELL - 1.5}
                width={T * CELL + 1}
                height={T * CELL + 1}
                fill="none"
                stroke="var(--color-cyan)"
                strokeWidth={1.2}
                rx={2}
              />
            )}

            {/* ── C（右下）── */}
            <text
              x={X0 + MAT - 14}
              y={Y0 + 12}
              fontSize={10}
              fontFamily="var(--font-mono, monospace)"
              fill="var(--color-volt)"
            >
              C
            </text>
            {Array.from({ length: N * N }, (_, i) => {
              const r = Math.floor(i / N)
              const c = i % N
              const segs = cSegs(r, c)
              const isCur =
                !p.done &&
                Math.floor(r / T) === p.bi &&
                Math.floor(c / T) === p.bj &&
                (r % T) * T + (c % T) === curCell
              return (
                <rect
                  key={i}
                  x={X0 + c * CELL}
                  y={Y0 + r * CELL}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={1.5}
                  fill={segs > 0 ? 'var(--color-volt)' : 'var(--color-line)'}
                  fillOpacity={isCur ? 1 : segs > 0 ? 0.18 + (segs / SEGS) * 0.55 : 0.5}
                />
              )
            })}
            {!p.done && (
              <rect
                x={X0 + p.bj * T * CELL - 2}
                y={Y0 + p.bi * T * CELL - 2}
                width={T * CELL + 2}
                height={T * CELL + 2}
                fill="none"
                stroke="var(--color-volt)"
                strokeWidth={1.5}
                rx={2}
              />
            )}
          </svg>
        </div>

        {/* ── 右侧双计数器 ── */}
        <div className="flex w-full shrink-0 flex-row flex-wrap items-start gap-x-6 gap-y-4 md:w-52 md:flex-col">
          <Stat label={t('naive global reads', 'naive 全局读取')} value={fmtInt(naiveReads)} unit="floats" tone="rose" />
          <Stat label={t('tiled global reads', 'tiled 全局读取')} value={fmtInt(tiledReads)} unit="floats" tone="volt" />
          <Stat label={t('live gap', '实时差距')} value={`${ratio.toFixed(1)}×`} tone="cyan" />
          <div className="text-[12px] leading-relaxed text-ink3">
            {t(
              <>At the same compute progress, the HBM reads each kernel needs. Theoretical gap = T = {T}×.</>,
              <>同样的计算进度，两种 kernel 各自需要的 HBM 读取量。理论差距 = T = {T}×。</>,
            )}
          </div>
        </div>
      </div>

      <div className="mt-4">
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (step >= TOTAL) {
              setStep(0)
              acc.current = 0
              setPlaying(true)
            } else {
              setPlaying((v) => !v)
            }
          }}
          onStep={onStep}
          onReset={onReset}
          speed={speed}
          onSpeed={setSpeed}
          extra={
            <span className="ml-auto font-mono text-[11px] tabular-nums text-ink3">
              STEP {fmtInt(step)}/{fmtInt(TOTAL)}
            </span>
          }
        />
      </div>
    </Widget>
  )
}
