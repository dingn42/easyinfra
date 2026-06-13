import { useMemo, useState } from 'react'
import { Btn, MathTex, Segmented, Stat, Widget } from '@/components/ui'
import { clamp } from '@/lib/format'
import { useT, useLocale, pick, type Loc } from '@/lib/i18n'

/* ────────── 浮点格式定义与位级编解码 ────────── */

type FmtId = 'fp32' | 'fp16' | 'bf16' | 'e4m3' | 'e5m2'

interface FloatFmt {
  id: FmtId
  name: string
  /** 指数位数 */
  E: number
  /** 尾数位数 */
  M: number
  bias: number
  /** FP8-E4M3 特例：指数全 1 仍可为正规数，仅 S.1111.111 是 NaN，没有 Inf */
  e4m3: boolean
}

const FORMATS: FloatFmt[] = [
  { id: 'fp32', name: 'FP32', E: 8, M: 23, bias: 127, e4m3: false },
  { id: 'fp16', name: 'FP16', E: 5, M: 10, bias: 15, e4m3: false },
  { id: 'bf16', name: 'BF16', E: 8, M: 7, bias: 127, e4m3: false },
  { id: 'e4m3', name: 'FP8-E4M3', E: 4, M: 3, bias: 7, e4m3: true },
  { id: 'e5m2', name: 'FP8-E5M2', E: 5, M: 2, bias: 15, e4m3: false },
]

const fmtById = (id: FmtId): FloatFmt => FORMATS.find((f) => f.id === id) as FloatFmt

type Kind = 'zero' | 'subnormal' | 'normal' | 'inf' | 'nan'

interface Decoded {
  kind: Kind
  value: number
  s: number
  e: number
  m: number
}

function decodeBits(bits: number[], f: FloatFmt): Decoded {
  const s = bits[0]
  let e = 0
  for (let i = 0; i < f.E; i++) e = e * 2 + bits[1 + i]
  let m = 0
  for (let i = 0; i < f.M; i++) m = m * 2 + bits[1 + f.E + i]
  const maxE = (1 << f.E) - 1
  const sign = s ? -1 : 1
  if (f.e4m3) {
    if (e === maxE && m === (1 << f.M) - 1) return { kind: 'nan', value: NaN, s, e, m }
  } else if (e === maxE) {
    if (m === 0) return { kind: 'inf', value: sign * Infinity, s, e, m }
    return { kind: 'nan', value: NaN, s, e, m }
  }
  if (e === 0) {
    if (m === 0) return { kind: 'zero', value: sign * 0, s, e, m }
    return { kind: 'subnormal', value: sign * (m / 2 ** f.M) * 2 ** (1 - f.bias), s, e, m }
  }
  return { kind: 'normal', value: sign * (1 + m / 2 ** f.M) * 2 ** (e - f.bias), s, e, m }
}

function bitsFromFields(f: FloatFmt, s: number, e: number, m: number): number[] {
  const bits: number[] = new Array(1 + f.E + f.M).fill(0)
  bits[0] = s
  for (let i = f.E - 1; i >= 0; i--) {
    bits[1 + i] = e & 1
    e >>= 1
  }
  for (let i = f.M - 1; i >= 0; i--) {
    bits[1 + f.E + i] = m & 1
    m >>= 1
  }
  return bits
}

function nanBits(f: FloatFmt): number[] {
  const maxE = (1 << f.E) - 1
  return bitsFromFields(f, 0, maxE, f.e4m3 ? (1 << f.M) - 1 : 1 << (f.M - 1))
}

function maxBits(f: FloatFmt, s = 0): number[] {
  const maxE = (1 << f.E) - 1
  if (f.e4m3) return bitsFromFields(f, s, maxE, (1 << f.M) - 2)
  return bitsFromFields(f, s, maxE - 1, (1 << f.M) - 1)
}

/** 把任意实数（round-to-nearest）编码进格式 f。E4M3 溢出时饱和到最大值（无 Inf）。 */
function encode(x: number, f: FloatFmt): number[] {
  if (Number.isNaN(x)) return nanBits(f)
  const s = x < 0 || Object.is(x, -0) ? 1 : 0
  const maxE = (1 << f.E) - 1
  if (!Number.isFinite(x)) {
    if (f.e4m3) return maxBits(f, s)
    return bitsFromFields(f, s, maxE, 0)
  }
  const a = Math.abs(x)
  if (a === 0) return bitsFromFields(f, s, 0, 0)
  let e = Math.floor(Math.log2(a))
  while (a / 2 ** e >= 2) e++
  while (a / 2 ** e < 1) e--
  const minNormalE = 1 - f.bias
  if (e < minNormalE) {
    // 次正规区：固定指数 1-bias，无隐含 1
    const q = Math.round(a / 2 ** (minNormalE - f.M))
    if (q === 0) return bitsFromFields(f, s, 0, 0)
    if (q >= 1 << f.M) return bitsFromFields(f, s, 1, 0)
    return bitsFromFields(f, s, 0, q)
  }
  let m = Math.round((a / 2 ** e - 1) * (1 << f.M))
  if (m === 1 << f.M) {
    m = 0
    e++
  }
  const eField = e + f.bias
  const maxNormalField = f.e4m3 ? maxE : maxE - 1
  if (eField > maxNormalField) {
    if (f.e4m3) return maxBits(f, s)
    return bitsFromFields(f, s, maxE, 0) // ±Inf
  }
  if (f.e4m3 && eField === maxE && m === (1 << f.M) - 1) m = (1 << f.M) - 2 // 避开 NaN 槽位
  return bitsFromFields(f, s, eField, m)
}

/* ────────── 显示辅助 ────────── */

function fmtVal(v: number): string {
  if (Number.isNaN(v)) return 'NaN'
  if (!Number.isFinite(v)) return v > 0 ? '+∞' : '−∞'
  if (v === 0) return Object.is(v, -0) ? '-0' : '0'
  const a = Math.abs(v)
  if (a >= 1e7 || a < 1e-5) return v.toExponential(4)
  const s = String(v)
  return s.length > 16 ? v.toPrecision(10) : s
}

function texNum(v: number): string {
  if (Number.isNaN(v)) return '\\mathrm{NaN}'
  if (!Number.isFinite(v)) return (v < 0 ? '-' : '') + '\\infty'
  const a = Math.abs(v)
  if (a !== 0 && (a >= 1e5 || a < 1e-4)) {
    const [mm, ee] = v.toExponential(3).split('e')
    return `${mm}\\times10^{${Number(ee)}}`
  }
  let s = String(v)
  if (s.length > 12) s = v.toPrecision(8)
  return s
}

function formulaTex(dec: Decoded, f: FloatFmt, L: typeof FORMULA_L.en): string {
  const denom = `2^{${f.M}}`
  switch (dec.kind) {
    case 'normal':
      return `(-1)^{${dec.s}}\\times\\Bigl(1+\\tfrac{${dec.m}}{${denom}}\\Bigr)\\times 2^{\\,${dec.e}-${f.bias}} = ${texNum(dec.value)}`
    case 'subnormal':
      return `(-1)^{${dec.s}}\\times\\tfrac{${dec.m}}{${denom}}\\times 2^{\\,1-${f.bias}} = ${texNum(dec.value)}\\;(\\text{${L.noImplicit}})`
    case 'zero':
      return `e=0,\\ m=0\\ \\Rightarrow\\ ${dec.s ? '-0' : '+0'}`
    case 'inf':
      return `e=\\text{${L.allOnes}},\\ m=0\\ \\Rightarrow\\ ${dec.s ? '-' : '+'}\\infty`
    case 'nan':
      return f.e4m3 ? `\\texttt{S.1111.111}\\ \\Rightarrow\\ \\mathrm{NaN}` : `e=\\text{${L.allOnes}},\\ m\\neq 0\\ \\Rightarrow\\ \\mathrm{NaN}`
  }
}

/** formulaTex 里需要本地化的少量文字（嵌在 KaTeX \text{} 中） */
const FORMULA_L: Loc<{ noImplicit: string; allOnes: string }> = {
  en: { noImplicit: 'no implicit 1', allOnes: 'all 1s' },
  zh: { noImplicit: '无隐含 1', allOnes: '全1' },
}

const KIND_INFO: Record<Kind, { label: string; desc: Loc; cls: string; tone: 'volt' | 'amber' | 'ink' | 'rose' }> = {
  normal: { label: 'NORMAL', desc: { en: 'normal', zh: '正规数' }, cls: 'border-volt/50 text-volt', tone: 'volt' },
  subnormal: { label: 'SUBNORMAL', desc: { en: 'subnormal', zh: '次正规数' }, cls: 'border-amber/50 text-amber', tone: 'amber' },
  zero: { label: 'ZERO', desc: { en: 'zero', zh: '零' }, cls: 'border-line2 text-ink2', tone: 'ink' },
  inf: { label: 'INF', desc: { en: 'infinity', zh: '无穷' }, cls: 'border-rose/50 text-rose', tone: 'rose' },
  nan: { label: 'NAN', desc: { en: 'not-a-number', zh: '非数' }, cls: 'border-rose/50 text-rose', tone: 'rose' },
}

const TONES = {
  sign: {
    on: 'border-rose/70 bg-rose/20 text-rose',
    off: 'border-rose/25 bg-rose/[0.04] text-ink3',
  },
  exp: {
    on: 'border-cyan/70 bg-cyan/20 text-cyan',
    off: 'border-cyan/25 bg-cyan/[0.04] text-ink3',
  },
  man: {
    on: 'border-volt/70 bg-volt/20 text-volt',
    off: 'border-volt/25 bg-volt/[0.04] text-ink3',
  },
} as const

/* ────────── 全景数轴几何 ────────── */

const AXIS_MIN = -152
const AXIS_MAX = 134
const PANO_W = 720
const PAD_L = 86
const PAD_R = 12
const X = (t: number) => PAD_L + ((t - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * (PANO_W - PAD_L - PAD_R)

function rangeOf(f: FloatFmt) {
  const lo = Math.log2(2 ** (1 - f.bias - f.M)) // 最小次正规
  const mid = Math.log2(2 ** (1 - f.bias)) // 最小正规
  const hi = Math.log2(decodeBits(maxBits(f), f).value) // 最大值
  return { lo, mid, hi }
}

/* ────────── 组件 ────────── */

export function FloatBitLab() {
  const t = useT()
  const { lang } = useLocale()
  const [fmtId, setFmtId] = useState<FmtId>('fp16')
  const f = fmtById(fmtId)
  const [bits, setBits] = useState<number[]>(() => encode(1, fmtById('fp16')))

  const dec = useMemo(() => decodeBits(bits, f), [bits, f])
  const kindInfo = KIND_INFO[dec.kind]

  const switchFmt = (id: FmtId) => {
    const nf = fmtById(id)
    const cur = decodeBits(bits, f)
    setFmtId(id)
    setBits(cur.kind === 'nan' ? nanBits(nf) : encode(cur.value, nf))
  }
  const toggleBit = (i: number) => setBits((b) => b.map((x, j) => (j === i ? 1 - x : x)))
  const reset = () => {
    setFmtId('fp16')
    setBits(encode(1, fmtById('fp16')))
  }

  const presets: { id: string; label: string; make: () => number[] }[] = [
    { id: 'one', label: '1.0', make: () => encode(1, f) },
    { id: 'tenth', label: '0.1', make: () => encode(0.1, f) },
    { id: 'max', label: t('max value', '最大值'), make: () => maxBits(f) },
    { id: 'minnorm', label: t('min normal', '最小正规值'), make: () => bitsFromFields(f, 0, 1, 0) },
    { id: 'nan', label: 'NaN', make: () => nanBits(f) },
  ]

  const total = 1 + f.E + f.M
  const bitCls = total > 16 ? 'h-8 w-[19px] text-[11px]' : 'h-9 w-7 text-[13px]'

  /* ── 放大视图：当前值所在的一格 ── */
  const zoom = useMemo(() => {
    const count = 1 << f.M
    let isSub = false
    let k = 0
    if (dec.kind === 'normal') k = dec.e - f.bias
    else if (dec.kind === 'subnormal' || dec.kind === 'zero') isSub = true
    const lo = isSub ? 0 : 2 ** k
    const hi = isSub ? 2 ** (1 - f.bias) : 2 ** (k + 1)
    const step = (hi - lo) / count
    const stride = Math.max(1, count / 1024)
    const hasCursor = dec.kind === 'normal' || dec.kind === 'subnormal' || dec.kind === 'zero'
    return { count, lo, hi, step, stride, hasCursor, curIdx: dec.m }
  }, [dec, f])

  const zoomTicks: number[] = []
  for (let i = 0; i < zoom.count; i += zoom.stride) zoomTicks.push(i)

  const ZW = 720
  const ZPAD = 18
  const zx = (i: number) => ZPAD + (i / zoom.count) * (ZW - 2 * ZPAD)

  return (
    <Widget
      index={1}
      title={t('Floating-Point Bit Dissector', '浮点位拆解器')}
      subtitle={t('Click a bit, flip a bit, watch the value move', '点一位，翻一位，看值怎么变')}
      onReset={reset}
      wide
      footer={t(
        <>
          Bit colors: <span className="text-rose">sign</span> / <span className="text-cyan">exponent</span> /{' '}
          <span className="text-volt">mantissa</span>. Switching format re-encodes the current value with
          round-to-nearest — try hitting "0.1" in FP32, then flip to BF16 and watch where the value drifts
          once the lower mantissa bits get chopped off.
        </>,
        <>
          位颜色：<span className="text-rose">sign 符号</span> / <span className="text-cyan">exponent 指数</span> /{' '}
          <span className="text-volt">mantissa 尾数</span>。切换格式时会把当前值四舍五入重编码 ——
          试试在 FP32 下按「0.1」再切到 BF16，看尾数被砍掉后值偏到哪去了。
        </>,
      )}
    >
      {/* 控件行 */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          options={FORMATS.map((x) => ({ value: x.id, label: x.name }))}
          value={fmtId}
          onChange={switchFmt}
        />
        <div className="flex flex-wrap gap-1.5">
          {presets.map((p) => (
            <Btn key={p.id} size="sm" variant="ghost" onClick={() => setBits(p.make())}>
              {p.label}
            </Btn>
          ))}
        </div>
      </div>

      {/* 位条 */}
      <div className="mt-4 flex flex-wrap items-center gap-[3px]">
        {bits.map((b, i) => {
          const region = i === 0 ? 'sign' : i <= f.E ? 'exp' : 'man'
          const firstOfGroup = i === 1 || i === 1 + f.E
          const name =
            region === 'sign'
              ? t('sign bit', '符号位')
              : region === 'exp'
                ? t(`exponent bit ${i}`, `指数位 ${i}`)
                : t(`mantissa bit ${i - f.E}`, `尾数位 ${i - f.E}`)
          return (
            <button
              key={i}
              onClick={() => toggleBit(i)}
              title={t(`${name} · click to flip`, `${name} · 点击翻转`)}
              className={`${bitCls} ${firstOfGroup ? 'ml-2' : ''} rounded border font-mono transition-colors hover:brightness-125 ${TONES[region][b ? 'on' : 'off']}`}
            >
              {b}
            </button>
          )
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px]">
        <span className="text-rose">SIGN s={dec.s}</span>
        <span className="text-cyan">
          EXP e={dec.e}
          {dec.kind === 'normal' ? t(` (e−bias = ${dec.e - f.bias})`, `（e−bias = ${dec.e - f.bias}）`) : ''}
        </span>
        <span className="text-volt">
          MANTISSA m={dec.m}/{1 << f.M}
        </span>
      </div>

      {/* 读数 */}
      <div className="mt-4 grid gap-x-6 gap-y-2 rounded-md border border-line bg-bg2/60 px-4 py-3 sm:grid-cols-[auto_1fr] sm:items-center">
        <div className="flex items-center gap-3">
          <Stat label={t('decimal value', '十进制值')} value={fmtVal(dec.value)} tone={kindInfo.tone} size="md" />
          <span className={`rounded border px-2 py-0.5 font-mono text-[10px] tracking-wider ${kindInfo.cls}`}>
            {kindInfo.label} · {pick(kindInfo.desc, lang)}
          </span>
        </div>
        <div className="overflow-x-auto [&>div]:my-1 [&>div]:text-left sm:[&>div]:text-center">
          <MathTex block tex={formulaTex(dec, f, pick(FORMULA_L, lang))} />
        </div>
      </div>

      {/* 全景：动态范围对比 */}
      <div className="mt-5">
        <div className="microlabel mb-2">{t('Panorama · representable range per format (log₂ axis, dot = current value)', '全景 · 各格式可表示范围（log₂ 数轴，点 = 当前值）')}</div>
        <svg viewBox={`0 0 ${PANO_W} 196`} className="w-full">
          {FORMATS.map((ff, i) => {
            const r = rangeOf(ff)
            const y = 18 + i * 30
            const active = ff.id === fmtId
            return (
              <g key={ff.id} className={active ? 'text-volt' : 'text-ink3'}>
                <text x={4} y={y + 4} fontSize={11} fontFamily="var(--font-mono)" fill="currentColor">
                  {ff.name}
                </text>
                {/* 次正规段（淡） */}
                <rect
                  x={X(r.lo)}
                  y={y - 3}
                  width={Math.max(2, X(r.mid) - X(r.lo))}
                  height={6}
                  rx={2}
                  fill="currentColor"
                  opacity={active ? 0.3 : 0.12}
                />
                {/* 正规段 */}
                <rect
                  x={X(r.mid)}
                  y={y - 4}
                  width={Math.max(2, X(r.hi) - X(r.mid))}
                  height={8}
                  rx={2}
                  fill="currentColor"
                  opacity={active ? 0.85 : 0.3}
                />
                {active && (
                  <text
                    x={Math.min(X(r.hi), PANO_W - 6)}
                    y={y - 9}
                    fontSize={10}
                    fontFamily="var(--font-mono)"
                    fill="currentColor"
                    textAnchor="end"
                  >
                    max {fmtVal(decodeBits(maxBits(ff), ff).value)}
                  </text>
                )}
                {active && Number.isFinite(dec.value) && dec.value !== 0 && (
                  <circle
                    cx={X(clamp(Math.log2(Math.abs(dec.value)), AXIS_MIN, AXIS_MAX))}
                    cy={y}
                    r={4.5}
                    fill="currentColor"
                    stroke="var(--color-bg)"
                    strokeWidth={1.5}
                  />
                )}
              </g>
            )
          })}
          {/* 坐标轴 */}
          <g className="text-ink3">
            <line x1={PAD_L} y1={170} x2={PANO_W - PAD_R} y2={170} stroke="currentColor" strokeWidth={1} opacity={0.6} />
            {[-128, -64, -16, 0, 16, 64, 128].map((t) => (
              <g key={t}>
                <line x1={X(t)} y1={12} x2={X(t)} y2={170} stroke="currentColor" strokeWidth={0.5} opacity={0.18} />
                <line x1={X(t)} y1={170} x2={X(t)} y2={175} stroke="currentColor" strokeWidth={1} opacity={0.7} />
                <text x={X(t)} y={188} fontSize={10} fontFamily="var(--font-mono)" fill="currentColor" textAnchor="middle">
                  2^{t}
                </text>
              </g>
            ))}
          </g>
        </svg>
      </div>

      {/* 放大：一格内的格点 */}
      <div className="mt-4">
        <div className="microlabel mb-2">{t('Zoom · the bucket the current value sits in (between two adjacent powers of 2)', '放大 · 当前值所在的一格（相邻两个 2 的幂之间）')}</div>
        <svg viewBox={`0 0 ${ZW} 84`} className="w-full">
          <line x1={ZPAD} y1={48} x2={ZW - ZPAD} y2={48} stroke="currentColor" strokeWidth={1} className="text-ink3" opacity={0.5} />
          {/* 边界 */}
          {[0, zoom.count].map((i) => (
            <line key={i} x1={zx(i)} y1={12} x2={zx(i)} y2={56} stroke="currentColor" strokeWidth={1.5} className="text-ink2" />
          ))}
          {/* 可表示值格点 */}
          {zoomTicks.map((i) => (
            <line key={i} x1={zx(i)} y1={22} x2={zx(i)} y2={48} stroke="currentColor" strokeWidth={1} className="text-volt" opacity={0.55} />
          ))}
          {/* 当前值 */}
          {zoom.hasCursor && (
            <line x1={zx(zoom.curIdx)} y1={10} x2={zx(zoom.curIdx)} y2={56} stroke="currentColor" strokeWidth={2.5} className="text-rose" />
          )}
          <text x={ZPAD} y={74} fontSize={10.5} fontFamily="var(--font-mono)" fill="currentColor" className="text-ink2" textAnchor="start">
            {fmtVal(zoom.lo)}
          </text>
          <text x={ZW - ZPAD} y={74} fontSize={10.5} fontFamily="var(--font-mono)" fill="currentColor" className="text-ink2" textAnchor="end">
            {fmtVal(zoom.hi)}
          </text>
        </svg>
        <p className="mt-1 text-[12.5px] leading-relaxed text-ink3">
          {t('This bucket holds ', '这一格内共有 ')}
          <span className="font-mono text-volt">{(1 << f.M).toLocaleString()}</span>
          {t(' representable values, spaced ', ' 个可表示值，相邻间隔 ')}
          <span className="font-mono text-volt">{fmtVal(zoom.step)}</span>
          {t(' apart', '')}
          {zoom.stride > 1
            ? t(` (one line drawn per ${zoom.stride.toLocaleString()} values)`, `（图中每 ${zoom.stride.toLocaleString()} 个画 1 个）`)
            : t(' (every value drawn)', '（已逐个画出）')}
          {t(
            '. Same 16 bits: FP16 packs 1024 values per bucket but only reaches ~65k; BF16 fits 128 per bucket yet stretches to 3.4×10³⁸ — that is the whole trade-off.',
            '。同样 16 位：FP16 每格 1024 个值但只覆盖到 6.5 万；BF16 每格 128 个值却覆盖到 3.4×10³⁸ —— 取舍就在这。',
          )}
        </p>
      </div>
    </Widget>
  )
}
