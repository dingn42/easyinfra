import { useState } from 'react'
import { Widget, Slider, Stat } from '@/components/ui'
import { fmtBytes, fmtSI } from '@/lib/format'
import { useT } from '@/lib/i18n'
import { LLAMA7B, LLAMA7B_PARAMS } from './model'

/* ── LAB 03：Attention 显存增长 ──────────────────────────────────
   naive attention 的单层 S×S score 矩阵 vs 模型权重 vs KV cache，
   对数轴柱状图 + 80GB HBM 上限线。S 拉到 32K 时 score 反超权重。 */

const WEIGHT_BYTES = LLAMA7B_PARAMS * 2 // 7B · fp16
const HBM_CEIL = 80 * 2 ** 30 // A100/H100 80GB

/* 对数轴范围（log2 字节） */
const LOG_LO = 21 // 2 MB
const LOG_HI = 48 // 256 TB

const TICKS = [2 ** 25, 2 ** 30, 2 ** 35, 2 ** 40, 2 ** 45]

const VB_W = 720
const VB_H = 200
const PLOT_X = 10
const PLOT_W = VB_W - 20

const xOf = (bytes: number) =>
  PLOT_X + (Math.min(Math.max(Math.log2(Math.max(bytes, 1)), LOG_LO), LOG_HI) - LOG_LO) / (LOG_HI - LOG_LO) * PLOT_W

interface Bar {
  label: string
  bytes: number
  color: string
  note: string
}

export function AttnMemoryLab() {
  const t = useT()
  const [log2S, setLog2S] = useState(13) // 8192
  const [heads, setHeads] = useState<number>(LLAMA7B.nHeads)
  const [log2B, setLog2B] = useState(0)

  const S = 2 ** log2S
  const B = 2 ** log2B

  // 单层瞬时 score 矩阵：B · h · S² · 2 字节（fp16）
  const scoreBytes = B * heads * S * S * 2
  // KV cache（全 32 层常驻）：B · S · L · 2(K,V) · h·128 · 2 字节
  const kvBytes = B * S * LLAMA7B.L * 2 * (heads * LLAMA7B.headDim) * 2

  const bars: Bar[] = [
    { label: t('score matrix (per-layer, temporary)', 'score 矩阵（单层临时）'), bytes: scoreBytes, color: 'var(--color-rose)', note: 'B·h·S²·2B' },
    { label: t('model weights (7B · FP16)', '模型权重（7B · FP16）'), bytes: WEIGHT_BYTES, color: 'var(--color-cyan)', note: t('fixed', '固定') },
    { label: t('KV cache (resident across 32 layers)', 'KV cache（32 层常驻）'), bytes: kvBytes, color: 'var(--color-amber)', note: 'B·S·L·2·d·2B' },
  ]

  const overWeights = scoreBytes > WEIGHT_BYTES
  const overCeil = scoreBytes > HBM_CEIL

  const reset = () => {
    setLog2S(13)
    setHeads(LLAMA7B.nHeads)
    setLog2B(0)
  }

  const ceilX = xOf(HBM_CEIL)

  return (
    <Widget
      index={3}
      title={t('Attention memory growth', 'Attention 显存增长')}
      subtitle={t('the S×S invoice of a naive implementation · log axis', 'naive 实现的 S×S 临时账单 · 对数轴')}
      onReset={reset}
      footer={t(
        <>
          Pull SEQ LEN to 32K and a single layer&apos;s temporary score matrix already needs{' '}
          <span className="text-rose">64 GB</span>, nearly devouring the whole 80GB card, all for an intermediate result
          thrown away the instant it&apos;s computed. So how do you avoid materializing this matrix? That question is the
          entire motivation for FlashAttention in the next chapter.
        </>,
        <>
          把 SEQ LEN 拉到 32K，单层临时的 score 矩阵就要 <span className="text-rose">64 GB</span>，
          几乎吃光整张 80GB 卡，而它只是个算完就丢的中间结果。那怎么才能不物化这个矩阵？这个问题就是下一章
          FlashAttention 的全部动机。
        </>,
      )}
    >
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-3">
        <Slider label="seq len S" value={log2S} min={9} max={17} step={1} onChange={setLog2S} fmt={(v) => fmtSI(2 ** v, 0)} unit="token" />
        <Slider label="heads h" value={heads} min={8} max={64} step={8} onChange={setHeads} />
        <Slider label="batch B" value={log2B} min={0} max={6} step={1} onChange={setLog2B} fmt={(v) => String(2 ** v)} unit="seq" />
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4 border-t border-line pt-4 sm:grid-cols-3">
        <Stat label={t('score matrix', 'score 矩阵')} value={fmtBytes(scoreBytes)} tone={overCeil ? 'rose' : overWeights ? 'amber' : 'ink'} size="lg" />
        <Stat label={t('KV cache', 'KV cache')} value={fmtBytes(kvBytes)} tone="amber" />
        <Stat label={t('score ÷ weights', 'score ÷ 权重')} value={`${(scoreBytes / WEIGHT_BYTES).toFixed(scoreBytes / WEIGHT_BYTES < 10 ? 2 : 0)}×`} tone={overWeights ? 'rose' : 'cyan'} />
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="mt-4 w-full" role="img" aria-label={t('log-axis comparison of three memory footprints', '三种显存占用的对数轴对比')}>
        {/* 轴刻度 */}
        {TICKS.map((v) => (
          <g key={v}>
            <line x1={xOf(v)} y1={14} x2={xOf(v)} y2={VB_H - 22} stroke="var(--color-line)" strokeWidth={1} />
            <text x={xOf(v)} y={VB_H - 8} textAnchor="middle" fontSize={10} fill="currentColor" className="font-mono text-ink3">
              {fmtBytes(v, 0)}
            </text>
          </g>
        ))}
        {/* 80GB HBM 上限 */}
        <line x1={ceilX} y1={8} x2={ceilX} y2={VB_H - 22} stroke="var(--color-rose)" strokeWidth={1.2} strokeDasharray="5 4" opacity={0.85} />
        <text x={ceilX + 5} y={16} fontSize={10.5} fill="currentColor" className="font-mono text-rose">
          80GB HBM
        </text>

        {/* 柱 */}
        {bars.map((b, i) => {
          const y = 28 + i * 50
          const w = Math.max(xOf(b.bytes) - PLOT_X, 2)
          const clipped = b.bytes > 2 ** LOG_HI
          return (
            <g key={b.label}>
              <text x={PLOT_X} y={y - 5} fontSize={11.5} fill="currentColor" className="text-ink2">
                {b.label}
                <tspan fill="currentColor" className="font-mono text-ink3" fontSize={9.5}>
                  {'  '}{b.note}
                </tspan>
              </text>
              <rect x={PLOT_X} y={y} width={w} height={22} rx={3} fill={`color-mix(in srgb, ${b.color} 45%, transparent)`} stroke={b.color} strokeWidth={1} />
              <text
                x={Math.min(PLOT_X + w + 6, VB_W - 4)}
                y={y + 15}
                fontSize={11.5}
                textAnchor={PLOT_X + w + 6 > VB_W - 70 ? 'end' : 'start'}
                fill={b.color}
                className="font-mono"
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {fmtBytes(b.bytes)}
                {clipped ? ' ≫' : ''}
              </text>
            </g>
          )
        })}
      </svg>

      {(overWeights || overCeil) && (
        <div className={`mt-1 font-mono text-[12px] ${overCeil ? 'text-rose' : 'text-amber'}`}>
          {overCeil
            ? t(
                '▲ The per-layer temporary score matrix already exceeds the full 80GB HBM — naive attention simply cannot run at this sequence length.',
                '▲ 单层临时 score 矩阵已超过整卡 80GB HBM——naive attention 在这个序列长度下根本跑不起来。',
              )
            : t(
                '▲ The temporary score matrix already exceeds all the model weights — paying more memory than the model itself for an intermediate result that gets thrown away.',
                '▲ 临时 score 矩阵已经超过全部模型权重——为一个算完就丢的中间结果付出比模型本身还贵的显存。',
              )}
        </div>
      )}
    </Widget>
  )
}
