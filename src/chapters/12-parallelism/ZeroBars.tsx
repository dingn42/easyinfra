import { useState } from 'react'
import { Figure, Slider } from '@/components/ui'
import { useT } from '@/lib/i18n'

/**
 * SEC2 小图：以 7B 模型为例，纯 DP vs ZeRO-1/2/3 的每卡显存（不含激活）。
 * 滑杆调 DP 卡数 N，柱状图随 stage 依次下降。
 */
export function ZeroBars() {
  const t = useT()
  const [exp, setExp] = useState(3) // N = 2^exp
  const n = 2 ** exp

  // 7B 模型，混合精度：权重 14GB + 梯度 14GB + 优化器态 84GB（FP32 master + m + v）
  const W = 14
  const G = 14
  const O = 84
  const bars = [
    { name: t('Plain DP', '纯 DP'), w: W, g: G, o: O },
    { name: 'ZeRO-1', w: W, g: G, o: O / n },
    { name: 'ZeRO-2', w: W, g: G / n, o: O / n },
    { name: 'ZeRO-3', w: W / n, g: G / n, o: O / n },
  ]

  const VW = 640
  const VH = 270
  const x0 = 46
  const plotW = VW - x0 - 10
  const plotH = VH - 50
  const maxY = 120
  const sy = plotH / maxY
  const bw = 64
  const slot = plotW / bars.length

  return (
    <Figure
      caption={t(
        `For a 7B model (mixed-precision training, activations excluded): the per-GPU three-piece memory at DP=${n} GPUs. Each ZeRO tier shards one more class of state into 1/N.`,
        `以 7B 模型（混合精度训练，不含激活）为例：DP=${n} 张卡时每卡的「三件套」显存。ZeRO 每升一档，又一类状态被切成 1/N。`,
      )}
    >
      <div className="mb-3 max-w-[280px]">
        <Slider label={t('DP GPU count N', 'DP 卡数 N')} value={exp} min={1} max={6} onChange={setExp} fmt={(v) => String(2 ** v)} unit={t('GPUs', '卡')} />
      </div>
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full select-none" role="img" aria-label={t('ZeRO per-GPU memory bar chart across stages', 'ZeRO 各 stage 每卡显存柱状图')}>
        {/* y 轴刻度 */}
        {[0, 40, 80, 120].map((v) => (
          <g key={v}>
            <line x1={x0} y1={20 + plotH - v * sy} x2={VW - 10} y2={20 + plotH - v * sy} stroke="var(--color-line)" strokeDasharray="2 4" />
            <text x={x0 - 6} y={20 + plotH - v * sy} textAnchor="end" dominantBaseline="central" fontSize={10} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
              {v}G
            </text>
          </g>
        ))}
        {/* 80G 红线 */}
        <line x1={x0} y1={20 + plotH - 80 * sy} x2={VW - 10} y2={20 + plotH - 80 * sy} stroke="var(--color-rose)" strokeWidth={1.25} strokeDasharray="5 3" />
        <text x={VW - 12} y={20 + plotH - 80 * sy - 5} textAnchor="end" fontSize={9.5} fontFamily="var(--font-mono, monospace)" fill="var(--color-rose)">
          A100 80G
        </text>
        {/* 柱 */}
        {bars.map((b, i) => {
          const cx = x0 + slot * i + slot / 2
          const total = b.w + b.g + b.o
          let y = 20 + plotH
          const seg = (v: number, color: string, key: string) => {
            const h = v * sy
            y -= h
            return <rect key={key} x={cx - bw / 2} y={y} width={bw} height={Math.max(0.5, h - 0.6)} fill={color} opacity={0.8} style={{ transition: 'all 300ms' }} />
          }
          return (
            <g key={b.name}>
              {seg(b.o, 'var(--color-violet)', 'o')}
              {seg(b.g, 'var(--color-amber)', 'g')}
              {seg(b.w, 'var(--color-cyan)', 'w')}
              <text x={cx} y={y - 7} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono, monospace)" fill={total <= 80 ? 'var(--color-volt)' : 'var(--color-ink2)'}>
                {total.toFixed(total < 10 ? 1 : 0)}G
              </text>
              <text x={cx} y={20 + plotH + 16} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink2)">
                {b.name}
              </text>
            </g>
          )
        })}
        {/* 图例 */}
        <g transform={`translate(${x0}, ${VH - 8})`} fontSize={10} fontFamily="var(--font-mono, monospace)" fill="var(--color-ink3)">
          <rect width={9} height={9} y={-8} fill="var(--color-cyan)" opacity={0.8} />
          <text x={13}>{t('weights 14G', '权重 14G')}</text>
          <rect x={t(110, 86)} width={9} height={9} y={-8} fill="var(--color-amber)" opacity={0.8} />
          <text x={t(123, 99)}>{t('gradients 14G', '梯度 14G')}</text>
          <rect x={t(240, 172)} width={9} height={9} y={-8} fill="var(--color-violet)" opacity={0.8} />
          <text x={t(253, 185)}>{t('optimizer 84G', '优化器态 84G')}</text>
        </g>
      </svg>
    </Figure>
  )
}
