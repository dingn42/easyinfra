import { useMemo, useRef, useState } from 'react'
import { Btn, PlayBar, Slider, Stat, Widget } from '@/components/ui'
import { useRafLoop } from '@/lib/hooks'
import { useT } from '@/lib/i18n'

/** ── LAB 01: Online Softmax 实验 ──
 * 左面板 = 两遍法（pass1 扫 max → pass2 累加 exp），右面板 = 单遍 online。
 * 共享一个步进时钟：两遍法要 16 步，online 8 步就完成 —— 差距本身就是演示。
 */

const N = 8
const TOTAL = N * 2
const DEFAULT_X = [2, -1, 3.5, 0.5, -2, 4, 1, -0.5]
const STEP_MS = 750

/** 固定种子伪随机（mulberry32），「随机」按钮可复现 */
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface TraceStep {
  mPrev: number
  lPrev: number
  m: number
  l: number
  alpha: number
}

/** 在线 softmax 的逐步轨迹：处理第 i 个元素后的 (m, l) 与修正因子 α */
function onlineTrace(xs: number[]): TraceStep[] {
  const out: TraceStep[] = []
  let m = -Infinity
  let l = 0
  for (const x of xs) {
    const mNew = Math.max(m, x)
    const alpha = m === -Infinity ? 1 : Math.exp(m - mNew)
    const lNew = l * alpha + Math.exp(x - mNew)
    out.push({ mPrev: m, lPrev: l, m: mNew, l: lNew, alpha })
    m = mNew
    l = lNew
  }
  return out
}

const fmtM = (v: number) => (v === -Infinity ? '−∞' : v.toFixed(2))

export function OnlineSoftmaxLab() {
  const t = useT()
  const [xs, setXs] = useState<number[]>(DEFAULT_X)
  const [step, setStep] = useState(0) // 已完成的微步数 0..16
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const seedRef = useRef(1)
  const accRef = useRef(0)

  /* ── 数值计算（即时，与动画无关） ── */
  const trace = useMemo(() => onlineTrace(xs), [xs])
  const fullMax = useMemo(() => Math.max(...xs), [xs])
  const twoPass = useMemo(() => {
    const den = xs.reduce((a, x) => a + Math.exp(x - fullMax), 0)
    return xs.map((x) => Math.exp(x - fullMax) / den)
  }, [xs, fullMax])
  const online = useMemo(() => {
    const last = trace[N - 1]
    return xs.map((x) => Math.exp(x - last.m) / last.l)
  }, [xs, trace])
  const maxErr = useMemo(
    () => Math.max(...xs.map((_, i) => Math.abs(twoPass[i] - online[i]))),
    [xs, twoPass, online],
  )

  /* ── 步进控制 ── */
  const advance = () =>
    setStep((p) => {
      const n = Math.min(TOTAL, p + 1)
      if (n >= TOTAL) setPlaying(false)
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
    setStep(0)
    accRef.current = 0
  }
  const setX = (i: number, v: number) => {
    setXs((p) => p.map((x, k) => (k === i ? v : x)))
    rewind()
  }
  const randomize = () => {
    const rng = mulberry32(seedRef.current++)
    setXs(Array.from({ length: N }, () => Math.round((rng() * 10 - 5) * 2) / 2))
    rewind()
  }
  const resetAll = () => {
    seedRef.current = 1
    setXs(DEFAULT_X)
    rewind()
  }

  /* ── 派生显示状态 ── */
  // 左：两遍法
  const p1n = Math.min(step, N) // pass1 已处理个数
  const p2n = Math.max(0, step - N) // pass2 已处理个数
  const p1Cursor = step < N ? step : -1
  const p2Cursor = step >= N && step < TOTAL ? step - N : -1
  let runMax = -Infinity
  let runMaxIdx = -1
  for (let i = 0; i < p1n; i++) if (xs[i] > runMax) (runMax = xs[i]), (runMaxIdx = i)
  let runSum = 0
  for (let i = 0; i < p2n; i++) runSum += Math.exp(xs[i] - fullMax)

  // 右：online
  const on = Math.min(step, N)
  const cur = on > 0 ? trace[on - 1] : null
  const onCursor = on < N ? on : -1
  // 刚发生 max 更新 → 前缀格子（0..on-2）闪 amber，表示乘了修正因子 α
  const prefixScaled = cur != null && cur.alpha < 1 && on >= 2

  const exact = maxErr < 1e-12

  return (
    <Widget
      index={1}
      title={t('Online Softmax Lab', 'Online Softmax 实验')}
      subtitle={t('Two-pass vs single-pass online merge · same inputs', '两遍法 vs 单遍在线合并 · 同一组输入')}
      onReset={resetAll}
      footer={t(
        <>
          Try dragging X[6] to its max: the new max appears only near the end of the scan, and the right side
          shows a cascade of amber flashes — the entire accumulated l is multiplied by α=e<sup>m−m′</sup> to
          retroactively change basis. Compare the two methods digit by digit; the max error always sits at the
          floating-point ulp level.
        </>,
        <>
          试着把 X[6] 拉到最大：扫描快结束时才出现新 max，右侧会看到一连串 amber 闪烁 ——
          已累计的 l 全部乘 α=e<sup>m−m′</sup> 追溯换底。两法结果逐位对比，最大误差始终在浮点 ulp 量级。
        </>,
      )}
    >
      {/* 控件行 */}
      <div className="mb-4 grid grid-cols-2 items-end gap-x-4 gap-y-3 sm:grid-cols-4">
        {[1, 4, 6].map((i) => (
          <Slider
            key={i}
            label={`X[${i}]`}
            value={xs[i]}
            min={-5}
            max={5}
            step={0.5}
            onChange={(v) => setX(i, v)}
            fmt={(v) => v.toFixed(1)}
          />
        ))}
        <Btn variant="ghost" onClick={randomize} title={t('Fixed seed, reproducible', '固定种子，可复现')}>
          {t('⚄ Randomize', '⚄ 随机一组')}
        </Btn>
      </div>

      <PlayBar
        playing={playing}
        onToggle={() => {
          if (step >= TOTAL) {
            setStep(0)
            accRef.current = 0
          }
          setPlaying((p) => !p)
        }}
        onStep={() => step < TOTAL && advance()}
        onReset={rewind}
        speed={speed}
        onSpeed={setSpeed}
        extra={
          <span className="ml-auto font-mono text-[11px] tabular-nums text-ink3">
            STEP {step}/{TOTAL}
          </span>
        }
      />

      {/* 双面板 */}
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        {/* ── 左：两遍法 ── */}
        <div className="rounded-md border border-line bg-bg2/50 p-3">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="microlabel text-cyan">{t('TWO-PASS', '两遍法 TWO-PASS')}</span>
            <span className="font-mono text-[11px] text-ink3">{t('2 full scans', '2 次完整扫描')}</span>
          </div>
          <div className="grid grid-cols-8 gap-1">
            {xs.map((x, i) => {
              const isCursor = i === p1Cursor || i === p2Cursor
              let cls = 'border-line bg-bg2/40 opacity-55'
              if (isCursor) cls = 'border-volt bg-volt/10 ring-1 ring-volt/50 opacity-100'
              else if (i < p2n) cls = 'border-cyan/40 bg-cyan/5 opacity-100'
              else if (i < p1n) cls = 'border-line2 bg-bg2 opacity-100'
              return (
                <div key={i} className={`rounded border py-1 text-center transition-all duration-200 ${cls}`}>
                  <div
                    className={`font-mono text-[11px] tabular-nums ${
                      i === runMaxIdx && p1n > 0 ? 'font-semibold text-cyan' : 'text-text'
                    }`}
                  >
                    {x.toFixed(1)}
                  </div>
                  <div className="mt-0.5 font-mono text-[9px] tabular-nums text-ink3">
                    {i < p2n ? Math.exp(x - fullMax).toFixed(3) : '·'}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="mt-3 space-y-1 font-mono text-[11px] leading-relaxed text-ink2">
            {step < N && (
              <>
                <div>
                  <span className="text-volt">PASS 1</span> {t('scan the row for max', '扫描整行找 max')}
                </div>
                <div>
                  {t('current m = ', '当前 m = ')}
                  <span className="text-cyan">{p1n > 0 ? runMax.toFixed(2) : '−∞'}</span>
                  {runMaxIdx >= 0 && <span className="text-ink3">{t(` (from X[${runMaxIdx}])`, `（来自 X[${runMaxIdx}]）`)}</span>}
                </div>
              </>
            )}
            {step >= N && step < TOTAL && (
              <>
                <div>
                  <span className="text-ink3">PASS 1 ✓ m = {fullMax.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-volt">PASS 2</span> {t('accumulate', '累加')} Σ e<sup>x−m</sup> ={' '}
                  <span className="text-cyan">{runSum.toFixed(3)}</span>
                </div>
              </>
            )}
            {step >= TOTAL && (
              <div>
                <span className="text-volt">{t('✓ done', '✓ 完成')}</span>: m = {fullMax.toFixed(2)}, Σ = {runSum.toFixed(3)}
                <span className="text-ink3">{t(' (2 passes = 16 steps)', '（扫了 2 遍 = 16 步）')}</span>
              </div>
            )}
          </div>
        </div>

        {/* ── 右：online 单遍 ── */}
        <div className="rounded-md border border-line bg-bg2/50 p-3">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="microlabel text-volt">{t('ONLINE single-pass', 'ONLINE 单遍')}</span>
            <span className="font-mono text-[11px] text-ink3">{t('fix (m, l) as you go', '边走边修 (m, l)')}</span>
          </div>
          <div className="grid grid-cols-8 gap-1">
            {xs.map((x, i) => {
              const isCursor = i === onCursor && step < N
              const flashed = prefixScaled && i < on - 1
              let cls = 'border-line bg-bg2/40 opacity-55'
              if (isCursor) cls = 'border-volt bg-volt/10 ring-1 ring-volt/50 opacity-100'
              else if (flashed) cls = 'border-amber/60 bg-amber/15 opacity-100'
              else if (i < on) cls = 'border-line2 bg-bg2 opacity-100'
              return (
                <div key={i} className={`rounded border py-1 text-center transition-all duration-200 ${cls}`}>
                  <div className="font-mono text-[11px] tabular-nums text-text">{x.toFixed(1)}</div>
                  <div
                    className={`mt-0.5 font-mono text-[9px] tabular-nums ${flashed ? 'text-amber' : 'text-ink3'}`}
                  >
                    {cur && i < on ? (Math.exp(x - cur.m) / cur.l).toFixed(3) : '·'}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="mt-3 space-y-1 font-mono text-[11px] leading-relaxed text-ink2">
            {cur == null && (
              <div>
                {t('init: m = ', '初始化：m = ')}<span className="text-cyan">−∞</span>, l = <span className="text-cyan">0</span>
              </div>
            )}
            {cur != null && (
              <>
                <div>
                  m′ = max({fmtM(cur.mPrev)}, {xs[on - 1].toFixed(2)}) ={' '}
                  <span className="text-cyan">{cur.m.toFixed(2)}</span>
                </div>
                <div>
                  α = e<sup>m−m′</sup> ={' '}
                  {cur.mPrev === -Infinity ? (
                    <span className="text-ink3">{t('— (first element)', '—（首个元素）')}</span>
                  ) : (
                    <span className={cur.alpha < 1 ? 'text-amber' : 'text-ink2'}>
                      e<sup>{(cur.mPrev - cur.m).toFixed(2)}</sup> = {cur.alpha.toFixed(3)}
                    </span>
                  )}
                </div>
                <div>
                  l′ = {cur.lPrev.toFixed(3)}×{cur.alpha.toFixed(3)} + e
                  <sup>{(xs[on - 1] - cur.m).toFixed(2)}</sup> ={' '}
                  <span className="text-cyan">{cur.l.toFixed(3)}</span>
                </div>
              </>
            )}
            {on >= N && (
              <div>
                <span className="text-volt">{t('✓ done: just 1 pass (8 steps)', '✓ 完成：只扫 1 遍（8 步）')}</span>
                {step < TOTAL && <span className="text-ink3">{t('…left still on pass 2', '…左边还在第二遍')}</span>}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 底部：最终结果并排 + 误差 */}
      <div className="mt-4 grid items-end gap-4 sm:grid-cols-[minmax(0,1fr)_auto]">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse font-mono text-[10.5px] tabular-nums">
            <tbody>
              <tr className="text-ink3">
                <td className="pr-2 text-left">softmax</td>
                {xs.map((_, i) => (
                  <td key={i} className="px-1 text-center">
                    [{i}]
                  </td>
                ))}
              </tr>
              <tr className="text-cyan">
                <td className="pr-2 text-left text-ink3">{t('two-pass', '两遍')}</td>
                {twoPass.map((p, i) => (
                  <td key={i} className="px-1 text-center">
                    {p.toFixed(3)}
                  </td>
                ))}
              </tr>
              <tr className="text-volt">
                <td className="pr-2 text-left text-ink3">online</td>
                {online.map((p, i) => (
                  <td key={i} className="px-1 text-center">
                    {p.toFixed(3)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
        <div className="flex items-end gap-2">
          <Stat
            label={t('max error', '最大误差')}
            value={maxErr === 0 ? '0' : maxErr.toExponential(1)}
            tone={exact ? 'volt' : 'amber'}
            size="md"
          />
          {exact && <span className="pb-4 font-mono text-sm text-volt">≈0 ✓</span>}
        </div>
      </div>
    </Widget>
  )
}
