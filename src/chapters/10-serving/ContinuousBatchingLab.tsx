import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { PlayBar, Segmented, Slider, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { useT } from '@/lib/i18n'
import { clamp, fmtInt, pct } from '@/lib/format'
import { genWorkload, simulate, statsAt, UTIL_DT, type Mode } from './sim'

/** ── LAB 02：连续批处理模拟器（本章主菜） ──
 * 同一条固定种子的请求流，分别用 static / continuous 调度回放。
 * 横向滚动时间轴：每请求一行色带（等待=灰、prefill=cyan、decode=volt、
 * static 下完成但占位=斜纹）；顶部 GPU 利用率随播放实时绘制。
 */

const PX_PER_MS = 0.05 // 50px / 秒
const TIME_SCALE = 3 // 1 real ms = 3 sim ms（再乘 speed）
const ROW_H = 13
const STRIP_H = 44 // 顶部利用率条带高
const AXIS_H = 22

const DEFAULTS = { rate: 2, promptMax: 1024, outMax: 256, slots: 8 }

export function ContinuousBatchingLab() {
  const tr = useT()
  const reduced = useReducedMotion()
  const [mode, setMode] = useState<Mode>('static')
  const [rate, setRate] = useState(DEFAULTS.rate)
  const [promptMax, setPromptMax] = useState(DEFAULTS.promptMax)
  const [outMax, setOutMax] = useState(DEFAULTS.outMax)
  const [slots, setSlots] = useState(DEFAULTS.slots)
  const [playing, setPlaying] = useState(!reduced)
  const [speed, setSpeed] = useState(1)
  const [simT, setSimT] = useState(0)
  const scrollRef = useRef<HTMLDivElement>(null)

  const workload = useMemo(() => genWorkload(rate, promptMax, outMax), [rate, promptMax, outMax])
  const sched = useMemo(() => simulate(workload, slots, mode), [workload, slots, mode])
  const duration = sched.duration

  // 工作负载参数变化 → 从头回放；切 mode 故意保留播放头，方便同一时刻 A/B 对比
  useEffect(() => {
    setSimT(reduced ? Number.MAX_SAFE_INTEGER : 0)
    if (!reduced) setPlaying(true)
  }, [rate, promptMax, outMax, slots, reduced])

  useRafLoop(
    (dt) => setSimT((p) => Math.min(duration, p + dt * speed * TIME_SCALE)),
    playing,
  )

  const t = clamp(simT, 0, duration)

  // 播完自动停
  useEffect(() => {
    if (playing && simT >= duration) setPlaying(false)
  }, [playing, simT, duration])
  const stats = useMemo(() => statsAt(sched, t), [sched, t])
  const nowUtil = sched.util[Math.min(sched.util.length - 1, Math.floor(t / UTIL_DT))] ?? 0

  const reset = useCallback(() => {
    setSimT(0)
    setPlaying(true)
    setSpeed(1)
  }, [])

  const fullReset = useCallback(() => {
    setMode('static')
    setRate(DEFAULTS.rate)
    setPromptMax(DEFAULTS.promptMax)
    setOutMax(DEFAULTS.outMax)
    setSlots(DEFAULTS.slots)
    setSimT(0)
    setPlaying(!reduced)
    setSpeed(1)
  }, [reduced])

  // 自动跟随播放头滚动
  const phX = t * PX_PER_MS
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !playing) return
    if (phX > el.scrollLeft + el.clientWidth * 0.72 || phX < el.scrollLeft) {
      el.scrollLeft = Math.max(0, phX - el.clientWidth * 0.45)
    }
  }, [phX, playing])

  // ── 几何 ──
  const n = sched.reqs.length
  const totalW = Math.max(640, duration * PX_PER_MS + 80)
  const rowsTop = STRIP_H + 14
  const totalH = rowsTop + n * ROW_H + AXIS_H
  const X = (ms: number) => ms * PX_PER_MS

  // 利用率面积图路径（整条画好，靠 clipPath 只露出播放头左侧）
  const utilPath = useMemo(() => {
    if (sched.util.length === 0) return ''
    const pts = sched.util.map((u, k) => `L${X(k * UTIL_DT).toFixed(1)},${(STRIP_H - u * (STRIP_H - 6)).toFixed(1)}`)
    return `M0,${STRIP_H} ${pts.join(' ')} L${X(duration).toFixed(1)},${STRIP_H} Z`
  }, [sched, duration])

  const ticks = useMemo(() => {
    const step = duration > 90_000 ? 10_000 : 5_000
    const arr: number[] = []
    for (let ms = 0; ms <= duration; ms += step) arr.push(ms)
    return arr
  }, [duration])

  const fmtMs = (ms: number | null) => (ms == null ? '—' : ms < 1000 ? `${Math.round(ms)}` : (ms / 1000).toFixed(1))
  const msUnit = (ms: number | null) => (ms == null ? '' : ms < 1000 ? 'ms' : 's')

  return (
    <Widget
      index={2}
      title={tr('Continuous batching simulator', '连续批处理模拟器')}
      subtitle={tr('One request stream, two scheduling fates', '同一条请求流，两种调度的命运')}
      wide
      onReset={fullReset}
      footer={
        <>
          {tr(
            <>
              Bands: <span className="text-ink3">gray = queue wait</span> ·{' '}
              <span className="text-cyan">cyan = prefill</span> · <span className="text-volt">volt = decode</span>{' '}
              · hatch = finished but still holding a slot (static only). The request stream uses a fixed seed —
              the workload is identical when you switch schedulers, so the readouts are directly comparable.
              Simplifying assumptions: TPOT fixed at 30 ms/token, prefill at 8 token/ms, ignoring the effect of
              batch size on per-step latency.
            </>,
            <>
              色带：<span className="text-ink3">灰=排队等待</span> · <span className="text-cyan">cyan=prefill</span> ·{' '}
              <span className="text-volt">volt=decode</span> · 斜纹=已完成却占着槽位（仅 static）。
              请求流固定种子 —— 切换调度模式时负载完全相同，读数可直接对比。
              简化假设：TPOT 固定 30ms/token、prefill 8 token/ms，忽略 batch 大小对单步时延的影响。
            </>,
          )}
        </>
      }
    >
      {/* 控制区 */}
      <div className="flex flex-wrap items-end gap-x-6 gap-y-4">
        <div>
          <div className="microlabel mb-1.5">{tr('SCHEDULING MODE', '调度模式')}</div>
          <Segmented
            options={[
              { value: 'static', label: 'STATIC' },
              { value: 'continuous', label: 'CONTINUOUS' },
            ]}
            value={mode}
            onChange={setMode}
          />
        </div>
        <Slider className="w-[150px]" label={tr('ARRIVAL RATE λ', '到达速率 λ')} value={rate} min={0.5} max={6} step={0.5} onChange={setRate} unit="req/s" />
        <Slider className="w-[150px]" label={tr('PROMPT RANGE', 'PROMPT 区间')} value={promptMax} min={256} max={2048} step={128} onChange={setPromptMax} fmt={(v) => `64–${v}`} unit="tok" />
        <Slider className="w-[150px]" label={tr('OUTPUT RANGE', '输出区间')} value={outMax} min={64} max={512} step={32} onChange={setOutMax} fmt={(v) => `16–${v}`} unit="tok" />
        <Slider className="w-[150px]" label={tr('CONCURRENT SLOTS', '并发槽位')} value={slots} min={4} max={32} step={2} onChange={setSlots} unit={tr('slots', '槽')} />
      </div>

      {/* 读数面板 */}
      <div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 rounded-md border border-line bg-bg2 px-4 py-3 sm:grid-cols-5">
        <Stat label={tr('THROUGHPUT', '吞吐')} value={fmtInt(stats.throughput)} unit="tok/s" tone="volt" />
        <Stat label={tr('AVG TTFT', '平均 TTFT')} value={fmtMs(stats.avgTtft)} unit={msUnit(stats.avgTtft)} tone="cyan" />
        <Stat label={tr('P95 LATENCY', 'P95 延迟')} value={fmtMs(stats.p95Latency)} unit={msUnit(stats.p95Latency)} tone="amber" />
        <Stat label={tr('GPU UTILIZATION', 'GPU 利用率')} value={Math.round(stats.gpuUtil * 100)} unit="%" tone={stats.gpuUtil > 0.7 ? 'volt' : 'ink'} />
        <Stat label={tr('DONE', '完成')} value={`${stats.done}/${n}`} tone="ink" />
      </div>

      {/* 瞬时利用率小条 */}
      <div className="mt-3 flex items-center gap-3">
        <span className="microlabel shrink-0">{tr('GPU NOW', 'GPU 瞬时')}</span>
        <div className="h-[8px] flex-1 overflow-hidden rounded-sm border border-line bg-bg">
          <div className="h-full bg-volt transition-[width] duration-100" style={{ width: `${nowUtil * 100}%` }} />
        </div>
        <span className="w-10 text-right font-mono text-[11px] tabular-nums text-volt">{pct(nowUtil)}</span>
      </div>

      {/* 时间轴 */}
      <div ref={scrollRef} className="mt-3 overflow-x-auto rounded-md border border-line bg-bg">
        <svg width={totalW} height={totalH} className="block" role="img" aria-label={tr('Request timeline: each row is one request with wait, prefill, and decode bands', '请求时间线：每行一个请求的等待、prefill、decode 色带')}>
          <defs>
            <clipPath id="cb-playclip">
              <rect x={0} y={0} width={phX} height={totalH} />
            </clipPath>
            <pattern id="cb-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" className="fill-panel2" />
              <line x1="0" y1="0" x2="0" y2="6" className="stroke-line2" strokeWidth="2" />
            </pattern>
          </defs>

          {/* 顶部 GPU 利用率条带 */}
          <text x={4} y={11} className="fill-current font-mono text-ink3" fontSize={8} letterSpacing={1}>
            GPU UTIL
          </text>
          <line x1={0} y1={STRIP_H} x2={totalW} y2={STRIP_H} className="stroke-line" strokeWidth={1} />
          <g clipPath="url(#cb-playclip)">
            <path d={utilPath} className="fill-volt/20 stroke-volt" strokeWidth={1} />
          </g>

          {/* 时间刻度 */}
          {ticks.map((ms) => (
            <g key={ms}>
              <line x1={X(ms)} y1={rowsTop} x2={X(ms)} y2={rowsTop + n * ROW_H} className="stroke-line" strokeWidth={1} />
              <text x={X(ms) + 3} y={totalH - 8} className="fill-current font-mono text-ink3" fontSize={8}>
                {ms / 1000}s
              </text>
            </g>
          ))}

          {/* 到达标记（不裁剪，提示未来的请求什么时候来） */}
          {sched.reqs.map((s, i) => (
            <path
              key={s.spec.id}
              d={`M${X(s.spec.arrival)},${rowsTop + i * ROW_H + 2} l4,4 l-4,4 z`}
              className="fill-ink3/60"
            />
          ))}

          {/* 请求色带（裁剪到播放头） */}
          <g clipPath="url(#cb-playclip)">
            {sched.reqs.map((s, i) => {
              const y = rowsTop + i * ROW_H + 2
              const h = ROW_H - 4
              return (
                <g key={s.spec.id}>
                  <title>{tr(
                    `REQ ${s.spec.id} · prompt ${s.spec.promptLen} tok · output ${s.spec.outLen} tok`,
                    `REQ ${s.spec.id} · prompt ${s.spec.promptLen} tok · 输出 ${s.spec.outLen} tok`,
                  )}</title>
                  {/* 等待 */}
                  {s.start > s.spec.arrival && (
                    <rect x={X(s.spec.arrival)} y={y} width={X(s.start - s.spec.arrival)} height={h} className="fill-ink3/25" rx={1} />
                  )}
                  {/* prefill */}
                  <rect x={X(s.start)} y={y} width={Math.max(1.5, X(s.prefillEnd - s.start))} height={h} className="fill-cyan/80" rx={1} />
                  {/* decode */}
                  <rect x={X(s.prefillEnd)} y={y} width={X(s.emitEnd - s.prefillEnd)} height={h} className="fill-volt/80" rx={1} />
                  {/* static：完成但占位 */}
                  {s.release > s.emitEnd && (
                    <rect x={X(s.emitEnd)} y={y} width={X(s.release - s.emitEnd)} height={h} fill="url(#cb-hatch)" rx={1} />
                  )}
                </g>
              )
            })}
          </g>

          {/* 播放头 */}
          <line x1={phX} y1={0} x2={phX} y2={totalH - AXIS_H} className="stroke-volt" strokeWidth={1.2} />
          <text x={phX + 4} y={STRIP_H + 11} className="fill-current font-mono text-volt" fontSize={9}>
            t = {(t / 1000).toFixed(1)}s
          </text>
        </svg>
      </div>

      <div className="mt-4">
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (!playing && simT >= duration) setSimT(0)
            setPlaying((p) => !p)
          }}
          onStep={() => setSimT((p) => clamp(p + 1000, 0, duration))}
          onReset={reset}
          speed={speed}
          onSpeed={setSpeed}
        />
      </div>
    </Widget>
  )
}
