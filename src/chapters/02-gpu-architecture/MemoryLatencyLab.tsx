import { useRef, useState } from 'react'
import { Btn, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { clamp, lerp } from '@/lib/format'
import { useLocale, useT, pick, type Loc } from '@/lib/i18n'
import { C, rgba } from '@/lib/palette'

/* ───────────────── LAB 03 显存层级延迟实验 ─────────────────
 * 点击某一级「取数」，数据方块沿 核心 → 该级 → 核心 的路径飞行，
 * 周期计数器滚动累计；右侧柱状图记录各级累计耗时，可连点累计平均。
 */

interface Level {
  name: Loc
  en: string
  cycles: number
  cap: string
  /** SVG 中目标点 */
  tx: number
  ty: number
  tone: 'volt' | 'cyan' | 'violet' | 'amber'
}

const LEVELS: Level[] = [
  { name: { en: 'Registers', zh: '寄存器' }, en: 'REG', cycles: 1, cap: '256 KB/SM', tx: 130, ty: 208, tone: 'volt' },
  { name: { en: 'Shared/L1', zh: 'Shared/L1' }, en: 'SMEM', cycles: 30, cap: '228 KB/SM', tx: 330, ty: 150, tone: 'cyan' },
  { name: { en: 'L2 cache', zh: 'L2 缓存' }, en: 'L2', cycles: 200, cap: '50 MB', tx: 492, ty: 150, tone: 'violet' },
  { name: { en: 'HBM3 memory', zh: 'HBM3 显存' }, en: 'HBM', cycles: 500, cap: '80 GB', tx: 648, ty: 150, tone: 'amber' },
]

/** 数据出发点：SM 内的执行单元 */
const CORE = { x: 130, y: 104 }

const TONE_TEXT = { volt: 'text-volt', cyan: 'text-cyan', violet: 'text-violet', amber: 'text-amber' } as const
const TONE_FILL = { volt: 'fill-volt', cyan: 'fill-cyan', violet: 'fill-violet', amber: 'fill-amber' } as const
const TONE_BG = { volt: 'bg-volt', cyan: 'bg-cyan', violet: 'bg-violet', amber: 'bg-amber' } as const
const TONE_STROKE = { volt: 'stroke-volt', cyan: 'stroke-cyan', violet: 'stroke-violet', amber: 'stroke-amber' } as const

interface LabState {
  counts: number[]
  cum: number[] // 各级累计周期
  total: number // 总累计周期
  fetches: number
  queue: number[]
  cur: { idx: number; t: number } | null
}

/** 默认就有一轮「每级各取一次」的记录，开页即可对比 */
function seedState(): LabState {
  return {
    counts: [1, 1, 1, 1],
    cum: LEVELS.map((l) => l.cycles),
    total: LEVELS.reduce((s, l) => s + l.cycles, 0),
    fetches: 4,
    queue: [],
    cur: null,
  }
}

function flightMs(idx: number) {
  return 260 + LEVELS[idx].cycles * 1.5
}

const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2)

export function MemoryLatencyLab() {
  const t = useT()
  const { lang } = useLocale()
  const reduced = useReducedMotion()
  const stRef = useRef<LabState>(seedState())
  const [, setTick] = useState(0)
  const bump = () => setTick((t) => t + 1)

  const finalize = (idx: number) => {
    const st = stRef.current
    st.counts[idx]++
    st.cum[idx] += LEVELS[idx].cycles
    st.total += LEVELS[idx].cycles
    st.fetches++
  }

  const doFetch = (idx: number) => {
    const st = stRef.current
    if (reduced) {
      finalize(idx) // 减少动画偏好：直接记账，不做飞行动画
      bump()
      return
    }
    if (st.cur == null) st.cur = { idx, t: 0 }
    else if (st.queue.length < 12) st.queue.push(idx)
    bump()
  }

  useRafLoop((dt) => {
    const st = stRef.current
    if (!st.cur) return
    st.cur.t += dt / flightMs(st.cur.idx)
    if (st.cur.t >= 1) {
      finalize(st.cur.idx)
      const next = st.queue.shift()
      st.cur = next == null ? null : { idx: next, t: 0 }
    }
    bump()
  }, stRef.current.cur != null && !reduced)

  const onReset = () => {
    stRef.current = seedState()
    bump()
  }

  const st = stRef.current
  const cur = st.cur
  // 飞行中的数据方块位置：去程 0~0.5，回程 0.5~1
  let dot: { x: number; y: number } | null = null
  let partial = 0
  if (cur) {
    const lv = LEVELS[cur.idx]
    const t = clamp(cur.t, 0, 1)
    const p = t < 0.5 ? ease(t * 2) : ease((t - 0.5) * 2)
    dot =
      t < 0.5
        ? { x: lerp(CORE.x, lv.tx, p), y: lerp(CORE.y, lv.ty, p) }
        : { x: lerp(lv.tx, CORE.x, p), y: lerp(lv.ty, CORE.y, p) }
    partial = Math.round(lv.cycles * t)
  }
  const counterValue = st.total + partial
  const maxCum = Math.max(...st.cum, 1)

  return (
    <Widget
      index={3}
      title={t('Memory hierarchy latency lab', '显存层级延迟实验')}
      subtitle={t('Click "fetch" to see how many cycles one round trip burns', '点「取数」看一次往返要烧掉多少周期')}
      onReset={onReset}
      footer={t(
        <>
          The same datum costs 500× more from HBM than from a register. Everything in chapter 4,
          coalesced access and shared-memory tiling alike, is at bottom a fight against this one chart.
        </>,
        <>
          同一份数据，放在寄存器和放在 HBM，取一次差 500 倍。第 4 章的全部内容，
          合并访存也好、shared memory 分块也好，说到底都是在跟这张图搏斗。
        </>,
      )}
    >
      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="min-w-0 flex-1">
          <div className="overflow-x-auto">
            <svg viewBox="0 0 720 260" className="w-full min-w-[540px] select-none" role="img" aria-label={t('Memory hierarchy fetch-path diagram', '存储层级取数路径图')}>
              {/* SM 框 */}
              <rect x={24} y={56} width={212} height={186} rx={6} className="fill-panel stroke-line2" />
              <text x={34} y={74} fontSize={11} className="fill-ink font-mono">{t('SM (on-chip)', 'SM（片上）')}</text>
              {/* 执行单元 */}
              <rect x={48} y={84} width={164} height={40} rx={4} className="fill-volt/10 stroke-volt/50" />
              <text x={130} y={108} fontSize={10.5} textAnchor="middle" className="fill-volt font-mono">{t('Execution units (cores)', '执行单元（核心）')}</text>
              {/* 寄存器（SM 内部目标） */}
              <rect
                x={48} y={188} width={164} height={40} rx={4}
                className={`fill-bg2 ${cur && cur.idx === 0 ? 'stroke-volt' : 'stroke-line2'}`}
              />
              <text x={130} y={205} fontSize={10.5} textAnchor="middle" className="fill-ink2 font-mono">{t('Register file', '寄存器堆')}</text>
              <text x={130} y={220} fontSize={9} textAnchor="middle" className="fill-ink3 font-mono">{t('≈1 cycle · 256 KB/SM', '≈1 周期 · 256 KB/SM')}</text>

              {/* 片上/片外分界 */}
              <line x1={566} x2={566} y1={40} y2={250} className="stroke-line" strokeDasharray="4 4" />
              <text x={560} y={36} fontSize={9} textAnchor="end" className="fill-ink3 font-mono">{t('on-chip ←', '片上 ←')}</text>
              <text x={572} y={36} fontSize={9} className="fill-ink3 font-mono">{t('→ off-chip', '→ 片外')}</text>

              {/* 片外/片上各级目标盒 */}
              {LEVELS.slice(1).map((lv, i) => {
                const idx = i + 1
                const x = lv.tx - 62
                const active = cur && cur.idx === idx
                return (
                  <g key={lv.en}>
                    <rect
                      x={x} y={112} width={124} height={76} rx={5}
                      className={`fill-bg2 ${active ? TONE_STROKE[lv.tone] : 'stroke-line2'}`}
                    />
                    <text x={lv.tx} y={138} fontSize={11} textAnchor="middle" className={`${TONE_FILL[lv.tone]} font-mono`}>
                      {pick(lv.name, lang)}
                    </text>
                    <text x={lv.tx} y={156} fontSize={9} textAnchor="middle" className="fill-ink3 font-mono">{lv.cap}</text>
                    <text x={lv.tx} y={172} fontSize={9} textAnchor="middle" className="fill-ink2 font-mono">{t(`≈${lv.cycles} cyc`, `≈${lv.cycles} 周期`)}</text>
                  </g>
                )
              })}

              {/* 路径虚线 */}
              <line x1={130} y1={124} x2={130} y2={188} className="stroke-line2" strokeDasharray="3 3" />
              {LEVELS.slice(1).map((lv, i) => (
                <line
                  key={i}
                  x1={212} y1={104}
                  x2={lv.tx - 62} y2={lv.ty}
                  className="stroke-line2"
                  strokeDasharray="3 3"
                />
              ))}

              {/* 飞行中的数据方块 */}
              {dot && cur && (
                <g>
                  <rect
                    x={dot.x - 7} y={dot.y - 7} width={14} height={14} rx={3}
                    className={TONE_FILL[LEVELS[cur.idx].tone]}
                    style={{ filter: `drop-shadow(0 0 6px ${rgba(C[LEVELS[cur.idx].tone], 0.55)})` }}
                  />
                  <text x={dot.x} y={dot.y - 12} fontSize={9} textAnchor="middle" className="fill-ink font-mono">
                    {cur.t < 0.5 ? t('request →', '请求 →') : t('← data', '← 数据')}
                  </text>
                </g>
              )}
            </svg>
          </div>

          {/* 取数按钮 */}
          <div className="mt-3 flex flex-wrap gap-2">
            {LEVELS.map((lv, i) => (
              <Btn key={lv.en} variant={i === 0 ? 'solid' : 'ghost'} onClick={() => doFetch(i)}>
                <span className={TONE_TEXT[lv.tone]}>▶</span>{' '}
                {t(`Fetch from ${pick(lv.name, 'en')}`, `从${pick(lv.name, 'zh')}取数`)}
              </Btn>
            ))}
            {st.queue.length > 0 && (
              <span className="self-center font-mono text-[11px] text-ink3">{t(`queued +${st.queue.length}`, `队列中 +${st.queue.length}`)}</span>
            )}
          </div>
        </div>

        {/* 右侧：计数器 + 柱状图 */}
        <div className="w-full shrink-0 rounded-md border border-line bg-bg2/60 p-4 lg:w-64">
          <div className="flex items-end justify-between gap-3">
            <Stat label={t('Cumulative cycles', '累计周期')} value={counterValue.toLocaleString('en-US')} unit="cyc" tone="volt" size="lg" />
            <Stat label={t('Fetches', '取数次数')} value={st.fetches} tone="ink" size="sm" />
          </div>
          <div className="mt-1 font-mono text-[11px] text-ink2">
            {t('avg', '平均')} <span className="text-cyan">{(st.total / Math.max(1, st.fetches)).toFixed(1)}</span> {t('cyc/fetch', '周期/次')}
          </div>

          <div className="microlabel mb-2 mt-4">{t('Cumulative cost per level', '各级累计耗时')}</div>
          <div className="space-y-2.5">
            {LEVELS.map((lv, i) => (
              <div key={lv.en}>
                <div className="mb-0.5 flex items-baseline justify-between font-mono text-[10.5px]">
                  <span className={TONE_TEXT[lv.tone]}>
                    {pick(lv.name, lang)} <span className="text-ink3">×{st.counts[i]}</span>
                  </span>
                  <span className="tabular-nums text-ink2">{st.cum[i].toLocaleString('en-US')} cyc</span>
                </div>
                <div className="h-2 overflow-hidden rounded-sm bg-bg">
                  <div
                    className={`h-full rounded-sm ${TONE_BG[lv.tone]} transition-[width] duration-300`}
                    style={{ width: `${Math.max(1.5, (st.cum[i] / maxCum) * 100)}%`, opacity: 0.75 }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-ink3">
            {t(
              'Click repeatedly and the fetches queue up and fly one by one, cycles piling on. Try 5 hits on HBM, then 5 on registers.',
              '连续点击会排队依次飞行，周期持续累计。试试连点 5 次 HBM，再连点 5 次寄存器。',
            )}
          </p>
        </div>
      </div>
    </Widget>
  )
}
