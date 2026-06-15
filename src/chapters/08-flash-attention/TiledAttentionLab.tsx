import { useMemo, useRef, useState } from 'react'
import { PlayBar, Stat, Toggle, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { fmtBytes } from '@/lib/format'
import { useT } from '@/lib/i18n'

/** ── LAB 02: 分块注意力动画 ──
 * 8×8 块的 score 矩阵「影子」（虚线 = 从未物化到 HBM）。
 * 外层遍历 Q 块行，内层扫 K/V 块；右侧 SRAM 面板展示驻留块与 (m,l,O) 状态。
 * 底部双累计条：naive O(S²) vs flash O(S·d) 的 HBM 流量。
 */

const NB = 8 // 每边块数
const S = 8192 // 序列长度
const D = 128 // head dim
const BYTES = 2 // fp16
const BLK = S / NB // 块边长 1024
const BLK_BYTES = BLK * D * BYTES // 一个 Q/K/V/O 块的字节数
// naive：写 S、softmax 读 S 写 P、PV 读 P —— 4 次 S² 往返 + Q/K/V/O 本体
const NAIVE_TOTAL = 4 * S * S * BYTES + 4 * S * D * BYTES
const STEP_MS = 600

interface Cell {
  i: number
  j: number
}

export function TiledAttentionLab() {
  const t = useT()
  const [causal, setCausal] = useState(false)
  const [sIdx, setSIdx] = useState(11) // 默认停在中途：打开页面即有信息量
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const accRef = useRef(0)
  const reduced = useReducedMotion()

  /* ── 步序列 ── */
  const steps = useMemo(() => {
    const arr: Cell[] = []
    for (let i = 0; i < NB; i++)
      for (let j = 0; j < NB; j++) {
        if (causal && j > i) continue
        arr.push({ i, j })
      }
    return arr
  }, [causal])
  const total = steps.length
  const lastJ = (i: number) => (causal ? i : NB - 1)

  // 每个格子的处理顺序（-1 = causal 跳过）
  const orderOf = useMemo(() => {
    const m = Array.from({ length: NB }, () => Array<number>(NB).fill(-1))
    steps.forEach((c, k) => (m[c.i][c.j] = k))
    return m
  }, [steps])

  /* ── 流量模型 ── */
  const flashAfter = (n: number) => {
    let b = 0
    for (let k = 0; k < n; k++) {
      const { i, j } = steps[k]
      b += 2 * BLK_BYTES // 读 K_j + V_j
      if (j === 0) b += BLK_BYTES // 行首：读入 Q_i
      if (j === lastJ(i)) b += BLK_BYTES // 行尾：写回 O_i
    }
    return b
  }
  const flashCum = flashAfter(sIdx)
  const flashTotal = useMemo(() => flashAfter(total), [steps]) // eslint-disable-line react-hooks/exhaustive-deps
  const naiveCum = (sIdx / total) * NAIVE_TOTAL
  const ratio = flashCum > 0 ? naiveCum / flashCum : NAIVE_TOTAL / flashTotal

  /* ── 播放控制 ── */
  const done = sIdx >= total
  const active: Cell | null = done ? null : steps[sIdx]
  const advance = () =>
    setSIdx((p) => {
      const n = Math.min(total, p + 1)
      if (n >= total) setPlaying(false)
      return n
    })
  useRafLoop((dt) => {
    accRef.current += dt * speed
    if (accRef.current >= STEP_MS) {
      accRef.current = 0
      advance()
    }
  }, playing)

  const rewind = () => {
    setPlaying(false)
    setSIdx(0)
    accRef.current = 0
  }
  const toggleCausal = (v: boolean) => {
    setCausal(v)
    rewind()
  }

  /* ── SRAM 面板状态 ── */
  // 当前（或刚完成的）行
  const curRow = active ? active.i : total > 0 ? steps[Math.max(0, sIdx - 1)].i : 0
  const rowLen = lastJ(curRow) + 1
  // 当前行已合并的 K/V 块数
  const mergedInRow = active ? active.j : done && total > 0 ? rowLen : 0
  const rowFrac = rowLen > 0 ? mergedInRow / rowLen : 0
  const rowDone = done || (active != null && active.j === 0 && sIdx > 0 && steps[sIdx - 1].i !== active.i)

  /* ── SVG 几何 ── */
  const CELL = 40
  const GAP = 3
  const ML = 30 // 左标签
  const MT = 22 // 顶标签
  const W = ML + NB * (CELL + GAP)
  const H = MT + NB * (CELL + GAP)

  return (
    <Widget
      index={2}
      title={t('Tiled Attention Animation', '分块注意力动画')}
      subtitle={t('S=8192 · d=128 · 8×8 blocks · fp16 · single head', 'S=8192 · d=128 · 8×8 块 · fp16 · 单头')}
      wide
      onReset={rewind}
      footer={t(
        <>
          The dashed cells are the "phantom" of the score matrix: each S<sub>ij</sub> block lives only in
          SRAM, gets merged into (m, l, O) on the spot, and is never written back to HBM. The traffic model
          counts 4 round-trips of S² for naive, and for flash it counts K/V re-read once per Q row (the cost
          of the outer Q loop, but O(S·d)-scale). Real implementations use much smaller blocks (Br/Bc ≈
          64–128, limited by SRAM capacity); we slice into 8 blocks here so it stays legible. With 32 heads,
          both sides' numbers scale by another ×32, and the ratio is unchanged.
        </>,
        <>
          虚线格子是 score 矩阵的「影子」：每个 S<sub>ij</sub> 小块只活在 SRAM，
          算完就地合并进 (m, l, O)，从未写回 HBM。流量模型：naive 按 4 次 S² 往返计，
          flash 按 K/V 每个 Q 行重读一遍计（这正是外层 Q 循环的代价，但它是 O(S·d) 级的）。
          真实实现的块要小得多（Br/Bc ≈ 64~128，受 SRAM 容量限制），这里切成 8 块是为了看得清。
          32 个头的话，两边数字都再 ×32，差距比例不变。
        </>,
      )}
    >
      <PlayBar
        playing={playing}
        onToggle={() => {
          if (done) {
            setSIdx(0)
            accRef.current = 0
          }
          setPlaying((p) => !p)
        }}
        onStep={() => !done && advance()}
        onReset={rewind}
        speed={speed}
        onSpeed={setSpeed}
        extra={
          <div className="ml-auto flex items-center gap-4">
            <Toggle label="causal mask" checked={causal} onChange={toggleCausal} />
            <span className="font-mono text-[11px] tabular-nums text-ink3">
              {t('BLOCK', '块')} {Math.min(sIdx, total)}/{total}
            </span>
          </div>
        }
      />

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_250px]">
        {/* ── 左：score 矩阵影子网格 ── */}
        <div>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full max-w-[440px]"
            role="img"
            aria-label={t('Tiled attention scan grid', '分块注意力扫描网格')}
          >
            {/* 顶部 K/V 块标签 */}
            {Array.from({ length: NB }, (_, j) => (
              <text
                key={`kt${j}`}
                x={ML + j * (CELL + GAP) + CELL / 2}
                y={MT - 8}
                textAnchor="middle"
                fontSize={9}
                className="font-mono text-ink3"
                fill="currentColor"
              >
                K{j}
              </text>
            ))}
            {/* 左侧 Q 块标签 */}
            {Array.from({ length: NB }, (_, i) => {
              const rowFinished = orderOf[i][lastJ(i)] >= 0 && orderOf[i][lastJ(i)] < sIdx
              return (
                <text
                  key={`qt${i}`}
                  x={ML - 8}
                  y={MT + i * (CELL + GAP) + CELL / 2 + 3}
                  textAnchor="end"
                  fontSize={9}
                  className={`font-mono ${rowFinished ? 'text-volt' : active && active.i === i ? 'text-ink' : 'text-ink3'}`}
                  fill="currentColor"
                >
                  Q{i}
                </text>
              )
            })}
            {/* 格子 */}
            {Array.from({ length: NB }, (_, i) =>
              Array.from({ length: NB }, (_, j) => {
                const x = ML + j * (CELL + GAP)
                const y = MT + i * (CELL + GAP)
                const skipped = causal && j > i
                const ord = orderOf[i][j]
                const isDone = ord >= 0 && ord < sIdx
                const isActive = active != null && active.i === i && active.j === j
                return (
                  <g key={`${i}-${j}`}>
                    <rect
                      x={x}
                      y={y}
                      width={CELL}
                      height={CELL}
                      rx={4}
                      fill={
                        skipped
                          ? 'var(--color-bg2)'
                          : isActive
                            ? 'var(--color-volt)'
                            : isDone
                              ? 'var(--color-volt)'
                              : 'transparent'
                      }
                      fillOpacity={skipped ? 0.65 : isActive ? 0.3 : isDone ? 0.09 : 0}
                      stroke={isActive ? 'var(--color-volt)' : skipped ? 'var(--color-line)' : 'var(--color-line2)'}
                      strokeWidth={isActive ? 1.5 : 1}
                      strokeDasharray={skipped || isActive ? undefined : '4 3'}
                    />
                    {skipped ? (
                      <text
                        x={x + CELL / 2}
                        y={y + CELL / 2 + 3}
                        textAnchor="middle"
                        fontSize={8}
                        className="font-mono text-ink3"
                        fill="currentColor"
                        opacity={0.5}
                      >
                        {t('skip', '跳过')}
                      </text>
                    ) : (
                      <text
                        x={x + CELL / 2}
                        y={y + CELL / 2 + 3}
                        textAnchor="middle"
                        fontSize={8.5}
                        className={`font-mono ${isActive ? 'text-volt' : isDone ? 'text-ink2' : 'text-ink3'}`}
                        fill="currentColor"
                        opacity={isActive ? 1 : isDone ? 0.9 : 0.45}
                      >
                        {isActive ? `S${i}${j}` : isDone ? '✓' : `${i},${j}`}
                      </text>
                    )}
                  </g>
                )
              }),
            )}
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-ink3">
            <span>
              <span className="text-volt">■</span> {t('active block (in SRAM)', '当前块（SRAM 内）')}
            </span>
            <span>{t('✓ merged', '✓ 已合并')}</span>
            <span>{t('┄ dashed = never materialized to HBM', '┄ 虚线 = 从未物化到 HBM')}</span>
            {causal && <span>{t('dim = causal skip', '暗格 = causal 跳过')}</span>}
          </div>
        </div>

        {/* ── 右：SRAM 面板 ── */}
        <div className="rounded-md border border-line bg-bg2/50 p-3">
          <div className="microlabel mb-2.5 text-amber">SRAM · ON-CHIP</div>
          <div className="mb-3 flex flex-wrap gap-1.5">
            <span className="rounded border border-volt/50 bg-volt/10 px-2 py-0.5 font-mono text-[11px] text-volt">
              Q{active ? active.i : done ? '·' : 0}
            </span>
            <span className="rounded border border-cyan/50 bg-cyan/10 px-2 py-0.5 font-mono text-[11px] text-cyan">
              K{active ? active.j : done ? '·' : 0}
            </span>
            <span className="rounded border border-cyan/50 bg-cyan/10 px-2 py-0.5 font-mono text-[11px] text-cyan">
              V{active ? active.j : done ? '·' : 0}
            </span>
            <span className="self-center font-mono text-[10px] text-ink3">{fmtBytes(BLK_BYTES)}{t('/block', '/块')}</span>
          </div>

          {/* m, l 状态条 */}
          <div className="mb-1 flex items-baseline justify-between font-mono text-[10px]">
            <span className="text-ink2">{t('m, l (row state)', 'm, l（行状态）')}</span>
            <span className="tabular-nums text-violet">
              {done ? t('all rows ✓', '全部行 ✓') : t(`merged ${mergedInRow}/${rowLen} blocks`, `已并 ${mergedInRow}/${rowLen} 块`)}
            </span>
          </div>
          <div className="mb-3 h-2 overflow-hidden rounded-sm bg-bg2">
            <div
              className="h-full rounded-sm bg-violet/70 transition-all duration-300"
              style={{ width: `${(done ? 1 : rowFrac) * 100}%` }}
            />
          </div>

          {/* O 累计条 */}
          <div className="mb-1 flex items-baseline justify-between font-mono text-[10px]">
            <span className="text-ink2">{t('O accumulator (un-normalized)', 'O 累计（未归一化）')}</span>
            <span className="tabular-nums text-amber">{done ? t('written back ✓', '已写回 ✓') : t(`Q${curRow} row`, `Q${curRow} 行`)}</span>
          </div>
          <div className="relative mb-3 h-2 overflow-hidden rounded-sm bg-bg2">
            <div
              className="h-full rounded-sm bg-amber/70 transition-all duration-300"
              style={{ width: `${(done ? 1 : rowFrac) * 100}%` }}
            />
            {/* 更新闪动 */}
            {!reduced && !done && sIdx > 0 && (
              <div key={sIdx} className="ei-fa-flash absolute inset-0 bg-amber/60" />
            )}
          </div>

          <div className="space-y-0.5 font-mono text-[10.5px] leading-relaxed text-ink3">
            {active ? (
              <>
                <div>
                  S{active.i}{active.j} = Q{active.i}·K{active.j}
                  <sup>T</sup> <span className="text-ink3">{t(`(${BLK}×${BLK}, SRAM only)`, `（${BLK}×${BLK}，仅 SRAM）`)}</span>
                </div>
                <div>
                  {t(`local softmax → merge online into (m, l, O${active.i})`, `局部 softmax → 在线并入 (m, l, O${active.i})`)}
                </div>
                {active.j === lastJ(active.i) && (
                  <div className="text-amber">{t(`end of row: O${active.i}/l written back to HBM`, `行尾：O${active.i}/l 写回 HBM`)}</div>
                )}
                {rowDone && <div className="text-volt">{t(`new row: load Q${active.i}, reset m, l, O`, `新行开始：载入 Q${active.i}，重置 m,l,O`)}</div>}
              </>
            ) : done ? (
              <div className="text-volt">{t(`✓ all ${total} blocks processed, O written back`, `✓ 全部 ${total} 块处理完毕，O 已写回`)}</div>
            ) : (
              <div>{t('press play to start the scan', '按播放开始扫描')}</div>
            )}
          </div>
        </div>
      </div>

      {/* ── 底部：HBM 流量双累计条 ── */}
      <div className="mt-5 space-y-2.5">
        <div className="microlabel">{t('HBM READ/WRITE CUMULATIVE (SINGLE HEAD)', 'HBM 读写累计（单头）')}</div>
        <div className="flex items-center gap-3">
          <span className="w-12 shrink-0 font-mono text-[11px] text-ink3">naive</span>
          <div className="h-3.5 flex-1 overflow-hidden rounded-sm border border-line bg-bg2">
            <div
              className="h-full bg-rose/60 transition-all duration-300"
              style={{ width: `${Math.min(100, (naiveCum / NAIVE_TOTAL) * 100)}%` }}
            />
          </div>
          <span className="w-20 shrink-0 text-right font-mono text-[11px] tabular-nums text-rose">
            {fmtBytes(naiveCum)}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="w-12 shrink-0 font-mono text-[11px] text-ink3">flash</span>
          <div className="h-3.5 flex-1 overflow-hidden rounded-sm border border-line bg-bg2">
            <div
              className="h-full bg-volt/70 transition-all duration-300"
              style={{
                width: `${Math.min(100, Math.max(flashCum > 0 ? 0.6 : 0, (flashCum / NAIVE_TOTAL) * 100))}%`,
              }}
            />
          </div>
          <span className="w-20 shrink-0 text-right font-mono text-[11px] tabular-nums text-volt">
            {fmtBytes(flashCum)}
          </span>
        </div>
        <div className="flex flex-wrap items-end gap-x-8 gap-y-2 pt-1">
          <Stat label={t('naive total traffic', 'naive 总流量')} value={fmtBytes(NAIVE_TOTAL)} tone="rose" size="sm" />
          <Stat label={t('flash total traffic', 'flash 总流量')} value={fmtBytes(flashTotal)} tone="volt" size="sm" />
          <Stat label={t('traffic ratio', '流量比')} value={`×${ratio.toFixed(1)}`} tone="volt" size="md" />
          {causal && (
            <span className="pb-1 font-mono text-[11px] text-ink3">
              {t('causal: flash saves about another half; naive still computes the full table', 'causal：flash 流量再省约一半，naive 照算全表')}
            </span>
          )}
        </div>
      </div>

      {/* O 条更新闪动的 keyframes（仅本组件作用域内的类名） */}
      <style>{`
        .ei-fa-flash { animation: ei-fa-flash-kf 0.45s ease-out forwards; }
        @keyframes ei-fa-flash-kf { from { opacity: 1; } to { opacity: 0; } }
      `}</style>
    </Widget>
  )
}
