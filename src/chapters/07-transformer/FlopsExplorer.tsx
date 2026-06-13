import { useState } from 'react'
import { Widget, Slider, Segmented, Btn, Stat } from '@/components/ui'
import { fmtBytes, fmtFlops, fmtInt, fmtSI, pct } from '@/lib/format'
import { pick, useLocale, useT, type Loc } from '@/lib/i18n'
import { calcDecodeFlops, calcParams, calcPrefillFlops, type ModelCfg } from './model'

/* ── LAB 02：FLOPs 巡览器 ─────────────────────────────────────────
   拖滑杆配置一个 Transformer，实时读出参数量 / 单 token FLOPs /
   prefill 总 FLOPs，堆叠条展示 QKV / score·AV / O proj / MLP 占比。 */

interface Preset {
  name: string
  d: number
  L: number
  ratio: number
  kvRatio: number
  mlpMats: number
  V: number
  official: string
}

const PRESETS: Preset[] = [
  { name: 'LLaMA-7B', d: 4096, L: 32, ratio: 2.6875, kvRatio: 1, mlpMats: 3, V: 32000, official: '6.74B' },
  { name: 'LLaMA-2 70B', d: 8192, L: 80, ratio: 3.5, kvRatio: 0.125, mlpMats: 3, V: 32000, official: '69.0B' },
  { name: 'GPT-3 175B', d: 12288, L: 96, ratio: 4, kvRatio: 1, mlpMats: 2, V: 50257, official: '175B' },
]

const DEFAULT = PRESETS[0]

const SEG_LABELS: Loc[] = [
  { en: 'QKV projection', zh: 'QKV 投影' },
  { en: 'score · AV', zh: 'score · AV' },
  { en: 'O projection', zh: 'O 投影' },
  { en: 'MLP', zh: 'MLP' },
]

export function FlopsExplorer() {
  const t = useT()
  const { lang } = useLocale()
  const [d, setD] = useState<number>(DEFAULT.d)
  const [L, setL] = useState<number>(DEFAULT.L)
  const [ratio, setRatio] = useState<number>(DEFAULT.ratio)
  const [kvRatio, setKvRatio] = useState<number>(DEFAULT.kvRatio)
  const [mlpMats, setMlpMats] = useState<number>(DEFAULT.mlpMats)
  const [V, setV] = useState<number>(DEFAULT.V)
  const [log2S, setLog2S] = useState(11) // 2048
  const [log2B, setLog2B] = useState(0) // 1

  const dff = Math.round(ratio * d)
  const S = 2 ** log2S
  const B = 2 ** log2B
  const cfg: ModelCfg = { d, L, dff, V, kvRatio, mlpMats }

  const p = calcParams(cfg)
  const f = calcDecodeFlops(cfg, S)
  const prefill = calcPrefillFlops(cfg, S, B)

  const segs = [
    { label: SEG_LABELS[0], v: f.qkv, color: 'var(--color-cyan)' },
    { label: SEG_LABELS[1], v: f.score, color: 'var(--color-amber)' },
    { label: SEG_LABELS[2], v: f.oproj, color: 'var(--color-violet)' },
    { label: SEG_LABELS[3], v: f.mlp, color: 'var(--color-volt)' },
  ]
  const segSum = segs.reduce((a, s) => a + s.v, 0)

  const applyPreset = (pr: Preset) => {
    setD(pr.d)
    setL(pr.L)
    setRatio(pr.ratio)
    setKvRatio(pr.kvRatio)
    setMlpMats(pr.mlpMats)
    setV(pr.V)
  }

  const reset = () => {
    applyPreset(DEFAULT)
    setLog2S(11)
    setLog2B(0)
  }

  const matched = PRESETS.find(
    (pr) => pr.d === d && pr.L === L && pr.ratio === ratio && pr.kvRatio === kvRatio && pr.mlpMats === mlpMats && pr.V === V,
  )

  return (
    <Widget
      index={2}
      title={t('FLOPs explorer', 'FLOPs 巡览器')}
      subtitle={t('A live ledger of parameter count and per-token compute', '参数量与每 token 计算量的实时账本')}
      onReset={reset}
      footer={t(
        <>
          Try pulling SEQ LEN from 2K to 32K: the linear layers&apos; FLOPs don&apos;t move, but the{' '}
          <span className="text-amber">score·AV</span> segment grows ever wider — attention is the only term that
          gets pricier with context length. Then switch to the GPT-3 preset and feel the force of d².
        </>,
        <>
          试着把 SEQ LEN 从 2K 拉到 32K：线性层的 FLOPs 一动不动，<span className="text-amber">score·AV</span> 那一段
          却越来越宽——attention 是唯一随上下文长度涨价的项。再换到 GPT-3 预设，感受 d² 的威力。
        </>,
      )}
    >
      {/* 预设 */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="microlabel mr-1">{t('preset', '预设')}</span>
        {PRESETS.map((pr) => (
          <Btn key={pr.name} size="sm" variant={matched?.name === pr.name ? 'solid' : 'ghost'} onClick={() => applyPreset(pr)}>
            {pr.name}
          </Btn>
        ))}
        {matched && (
          <span className="font-mono text-[11px] text-volt">
            {t(`✓ computed ${fmtSI(p.total, 2)} ≈ official ${matched.official}`, `✓ 计算值 ${fmtSI(p.total, 2)} ≈ 官方 ${matched.official}`)}
          </span>
        )}
      </div>

      {/* 滑杆 */}
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
        <Slider label="d_model" value={d} min={1024} max={12288} step={256} onChange={setD} fmt={fmtInt} />
        <Slider label={t('layers L', '层数 L')} value={L} min={8} max={96} step={4} onChange={setL} />
        <Slider
          label="d_ff / d"
          value={ratio}
          min={2.5}
          max={4}
          step={0.0625}
          onChange={setRatio}
          fmt={(v) => v.toFixed(2)}
          unit={`→ d_ff=${fmtInt(dff)}`}
        />
        <Slider label="seq len S" value={log2S} min={7} max={15} step={1} onChange={setLog2S} fmt={(v) => fmtSI(2 ** v, 0)} unit="token" />
        <Slider label="batch B" value={log2B} min={0} max={6} step={1} onChange={setLog2B} fmt={(v) => String(2 ** v)} unit="seq" />
        <div className="flex flex-wrap items-end gap-x-5 gap-y-2 pb-0.5">
          <div>
            <div className="microlabel mb-1.5">KV heads</div>
            <Segmented<number>
              options={[
                { value: 1, label: 'MHA' },
                { value: 0.25, label: 'GQA ¼' },
                { value: 0.125, label: 'GQA ⅛' },
              ]}
              value={kvRatio}
              onChange={setKvRatio}
            />
          </div>
          <div>
            <div className="microlabel mb-1.5">{t('MLP structure', 'MLP 结构')}</div>
            <Segmented<number>
              options={[
                { value: 3, label: 'SwiGLU ×3' },
                { value: 2, label: 'GELU ×2' },
              ]}
              value={mlpMats}
              onChange={setMlpMats}
            />
          </div>
        </div>
      </div>

      {/* 读数 */}
      <div className="mt-5 grid grid-cols-2 gap-4 border-t border-line pt-4 lg:grid-cols-4">
        <Stat label={t('total params', '总参数')} value={fmtSI(p.total, 2)} tone="volt" size="lg" />
        <Stat label={t('weight memory · FP16', '权重显存 · FP16')} value={fmtBytes(p.total * 2)} tone="amber" />
        <Stat label={t('decode · forward / token', 'decode · 单 token 前向')} value={fmtFlops(f.total)} tone="cyan" />
        <Stat label={t(`prefill total · B=${B}`, `prefill 总量 · B=${B}`)} value={fmtFlops(prefill)} tone="ink" />
      </div>

      {/* 堆叠条：单 token FLOPs 去向 */}
      <div className="mt-5">
        <div className="mb-2 flex items-baseline justify-between">
          <span className="microlabel">{t(`where per-token FLOPs go (decode @ S=${fmtSI(S, 0)})`, `单 token FLOPs 去向（decode @ S=${fmtSI(S, 0)}）`)}</span>
          <span className="font-mono text-[11px] text-ink3">{t(`attention (score·AV) = ${pct(f.score / segSum, 1)}`, `attention（score·AV）占 ${pct(f.score / segSum, 1)}`)}</span>
        </div>
        <div className="flex h-7 w-full overflow-hidden rounded-md border border-line bg-bg2">
          {segs.map((s) => (
            <div
              key={s.label.en}
              title={`${pick(s.label, lang)}: ${fmtFlops(s.v)} (${pct(s.v / segSum, 1)})`}
              className="h-full transition-[width] duration-200"
              style={{
                width: `${(s.v / segSum) * 100}%`,
                background: `color-mix(in srgb, ${s.color} 55%, transparent)`,
                boxShadow: `inset -1px 0 0 var(--color-line)`,
              }}
            />
          ))}
        </div>
        <div className="mt-2.5 grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
          {segs.map((s) => (
            <div key={s.label.en} className="flex items-baseline gap-2 font-mono text-[12px]">
              <span className="inline-block size-2 shrink-0 translate-y-px rounded-[2px]" style={{ background: s.color }} />
              <span className="text-ink2">{pick(s.label, lang)}</span>
              <span className="ml-auto tabular-nums text-ink">{fmtFlops(s.v)}</span>
              <span className="w-14 text-right tabular-nums text-ink3">{pct(s.v / segSum, 1)}</span>
            </div>
          ))}
        </div>
      </div>
    </Widget>
  )
}
