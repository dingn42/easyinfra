import { useState } from 'react'
import { Segmented, Stat, Widget } from '@/components/ui'
import { useT, useLocale, pick, type Loc } from '@/lib/i18n'

/* ────────── 模型 / 格式参数 ────────── */

const MODELS = [
  { id: '7b', label: '7B', params: 7e9 },
  { id: '13b', label: '13B', params: 13e9 },
  { id: '70b', label: '70B', params: 70e9 },
] as const
type ModelId = (typeof MODELS)[number]['id']

const FMTS = [
  { id: 'fp16', label: 'FP16', bytes: 2 },
  { id: 'int8', label: 'INT8', bytes: 1 },
  { id: 'int4', label: 'INT4', bytes: 0.5 },
] as const
type WFmtId = (typeof FMTS)[number]['id']

/** 简化口径：80G 卡 HBM 带宽 ~2 TB/s；decode 每 token 把全部权重读一遍 */
const BW_GBPS = 2000
const SCALE_MAX = 150 // 条形图满刻度 GB

const CARDS: { gb: number; label: Loc }[] = [
  { gb: 24, label: { en: '24G (4090)', zh: '24G（4090）' } },
  { gb: 80, label: { en: '80G (A100/H100)', zh: '80G（A100/H100）' } },
]

export function MemoryCalcLab() {
  const t = useT()
  const { lang } = useLocale()
  const [modelId, setModelId] = useState<ModelId>('70b')
  const [fmtId, setFmtId] = useState<WFmtId>('fp16')

  const model = MODELS.find((m) => m.id === modelId) ?? MODELS[2]
  const fmt = FMTS.find((f) => f.id === fmtId) ?? FMTS[0]

  const gbOf = (bytes: number) => (model.params * bytes) / 1e9
  const weightGB = gbOf(fmt.bytes)
  const speedup = 2 / fmt.bytes
  const tokS = BW_GBPS / weightGB

  const reset = () => {
    setModelId('70b')
    setFmtId('fp16')
  }

  /* ── SVG 条形图几何 ── */
  const W = 720
  const BAR_X = 70
  const BAR_R = 708
  const xOf = (gb: number) => BAR_X + (Math.min(gb, SCALE_MAX) / SCALE_MAX) * (BAR_R - BAR_X)
  const rowY = (i: number) => 38 + i * 34

  return (
    <Widget
      index={3}
      title={t('VRAM Payoff Calculator', '显存收益计算器')}
      subtitle={t('One model, three byte widths', '同一个模型，三种字节数')}
      onReset={reset}
      footer={t(
        <>
          Ground rules: decode, batch=1, every token streams the full weight set out of HBM once;
          the theoretical ceiling assumes ~2 TB/s on an 80G card and ignores KV cache, activations
          and comms, so real numbers run a bit lower. Note that in W4A16 the multiply is still done
          in FP16 — weights are <em>stored</em> as INT4, then dequantized on the fly inside the
          kernel back to FP16 before hitting the Tensor Cores.
        </>,
        <>
          估算口径：decode、batch=1、每生成一个 token 都要把全部权重从 HBM 读一遍，按 80G 卡 ~2 TB/s
          带宽算理论上限；忽略 KV cache、激活与通信开销，实际值会低一些。注意 W4A16 方案里乘法仍然用
          FP16 做 —— 权重以 INT4 存放，kernel 内现场反量化（dequantize）回 FP16 再进 Tensor Core。
        </>,
      )}
    >
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2.5">
          <span className="microlabel">{t('Model', '模型')}</span>
          <Segmented options={MODELS.map((m) => ({ value: m.id, label: m.label }))} value={modelId} onChange={setModelId} />
        </div>
        <div className="flex items-center gap-2.5">
          <span className="microlabel">{t('Weight format', '权重格式')}</span>
          <Segmented options={FMTS.map((f) => ({ value: f.id, label: f.label }))} value={fmtId} onChange={setFmtId} />
        </div>
      </div>

      {/* 条形图：三种格式并排，点击切换 */}
      <svg viewBox={`0 0 ${W} 148`} className="mt-4 w-full">
        {/* 容量标尺线 */}
        {CARDS.map((c) => (
          <g key={c.gb} className="text-amber">
            <line x1={xOf(c.gb)} y1={20} x2={xOf(c.gb)} y2={124} stroke="currentColor" strokeWidth={1} strokeDasharray="4 3" opacity={0.55} />
            <text x={xOf(c.gb)} y={13} fontSize={10} fontFamily="var(--font-mono)" fill="currentColor" textAnchor="middle" opacity={0.9}>
              {pick(c.label, lang)}
            </text>
          </g>
        ))}
        {FMTS.map((f, i) => {
          const gb = gbOf(f.bytes)
          const active = f.id === fmtId
          const y = rowY(i)
          const barW = Math.max(2, xOf(gb) - BAR_X)
          const over = gb > SCALE_MAX
          return (
            <g key={f.id} onClick={() => setFmtId(f.id)} style={{ cursor: 'pointer' }} className={active ? 'text-volt' : 'text-ink3'}>
              {/* 整行命中区 */}
              <rect x={0} y={y - 13} width={W} height={30} fill="transparent" />
              <text x={4} y={y + 4} fontSize={11.5} fontFamily="var(--font-mono)" fill="currentColor">
                {f.label}
              </text>
              <rect x={BAR_X} y={y - 8} width={BAR_R - BAR_X} height={16} rx={3} fill="currentColor" opacity={0.07} />
              <rect x={BAR_X} y={y - 8} width={barW} height={16} rx={3} fill="currentColor" opacity={active ? 0.85 : 0.3} />
              {over && (
                <text x={BAR_R - 4} y={y + 4} fontSize={10} fontFamily="var(--font-mono)" fill="currentColor" textAnchor="end">
                  »
                </text>
              )}
              <text
                x={Math.min(BAR_X + barW + 8, BAR_R - 30)}
                y={y + 4}
                fontSize={11}
                fontFamily="var(--font-mono)"
                fill="currentColor"
                opacity={active ? 1 : 0.7}
              >
                {gb % 1 === 0 ? gb : gb.toFixed(1)} GB
              </text>
            </g>
          )
        })}
      </svg>

      {/* 读数 */}
      <div className="mt-2 flex flex-wrap items-end gap-x-8 gap-y-3">
        <Stat label={t('Weight size', '权重大小')} value={weightGB % 1 === 0 ? weightGB : weightGB.toFixed(1)} unit="GB" tone="amber" size="lg" />
        {CARDS.map((c) => {
          const fit = weightGB <= c.gb
          return (
            <Stat
              key={c.gb}
              label={t(`Fits in ${c.gb}G card`, `装进 ${c.gb}G 卡`)}
              value={
                <span className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-sm ${fit ? 'border-volt/50 text-volt' : 'border-rose/50 text-rose'}`}>
                  {fit ? t('✓ Fits', '✓ 放得下') : t('✗ Too big', '✗ 放不下')}
                </span>
              }
              tone={fit ? 'volt' : 'rose'}
            />
          )
        })}
        <Stat label={t('Theoretical decode speedup vs FP16', '相对 FP16 理论 decode 加速')} value={`${speedup % 1 === 0 ? speedup : speedup.toFixed(1)}×`} tone="volt" />
        <Stat label={t('Estimated decode speed', '预估 decode 速度')} value={`~${tokS >= 100 ? Math.round(tokS) : tokS.toFixed(1)}`} unit="tok/s" tone="cyan" />
      </div>
    </Widget>
  )
}
