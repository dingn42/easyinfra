import { useMemo, useState } from 'react'
import { Btn, Figure } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { mulberry32 } from './sim'
import { pct } from '@/lib/format'

/** ── SEC3 插图：静态 batching 的木桶效应 ──
 * 8 个请求一批，输出长度参差不齐；最长的拖住全员，
 * 先完成的槽位灰掉空转，直到整批结束才能换人。
 */

const N = 8

export function StaticBatchTimeline() {
  const t = useT()
  const [seed, setSeed] = useState(7)

  const lens = useMemo(() => {
    const rnd = mulberry32(seed)
    return Array.from({ length: N }, () => Math.round(40 + rnd() * 360))
  }, [seed])

  const maxLen = Math.max(...lens)
  const sum = lens.reduce((a, b) => a + b, 0)
  const waste = 1 - sum / (N * maxLen)

  // 几何
  const W = 640
  const labelW = 76
  const prefillW = 26
  const plotW = W - labelW - prefillW - 60
  const rowH = 22
  const H = N * rowH + 56
  const xOf = (tok: number) => labelW + prefillW + (tok / maxLen) * plotW

  return (
    <Figure
      caption={t(
        <>
          Static batching: the whole batch enters and exits together. The longest output in this batch is{' '}
          {maxLen} tokens; the gray hatching = finished slots trapped in the batch and idling, accounting for{' '}
          <span className="font-mono text-rose">{pct(waste)}</span> of the batch&apos;s GPU slot-time. Click
          &quot;Resample&quot; to draw a new batch.
        </>,
        <>
          静态批处理：整批同进同出。本批最长输出 {maxLen} token，
          灰色斜纹 = 已完成却被困在批里的空转槽位，占整批 GPU 槽位时间的{' '}
          <span className="font-mono text-rose">{pct(waste)}</span>。点「换一批」重新抽样。
        </>,
      )}
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="microlabel">{t('8 requests · one static batch', '8 个请求 · 一个静态批次')}</span>
        <Btn size="sm" variant="ghost" onClick={() => setSeed((s) => s + 1)}>{t('⟳ Resample', '⟳ 换一批')}</Btn>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={t('Static batching timeline: the longest request holds up the whole batch', '静态批处理时间线：最长请求拖住全批')}>
        <defs>
          <pattern id="sb-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" className="fill-panel2" />
            <line x1="0" y1="0" x2="0" y2="6" className="stroke-line2" strokeWidth="2" />
          </pattern>
        </defs>
        {lens.map((len, i) => {
          const y = 14 + i * rowH
          const isMax = len === maxLen
          return (
            <g key={i}>
              <text x={0} y={y + 11} className="fill-current font-mono text-ink3" fontSize={9}>
                REQ {i}
              </text>
              {/* prefill */}
              <rect x={labelW} y={y} width={prefillW} height={rowH - 7} rx={2} className="fill-cyan/60" />
              {/* decode */}
              <rect x={labelW + prefillW} y={y} width={(len / maxLen) * plotW} height={rowH - 7} rx={2} className={isMax ? 'fill-volt' : 'fill-volt/55'} />
              {/* 完成后空转 */}
              {!isMax && (
                <rect x={xOf(len)} y={y} width={(1 - len / maxLen) * plotW} height={rowH - 7} rx={2} fill="url(#sb-hatch)" opacity={0.8} />
              )}
              <text x={xOf(len) + (isMax ? -4 : 4)} y={y + 11} textAnchor={isMax ? 'end' : 'start'} className={`fill-current font-mono ${isMax ? 'text-bg' : 'text-ink3'}`} fontSize={8}>
                {isMax ? t(`${len} tok ← longest stave`, `${len} tok ← 木桶最长板`) : `✓ ${len}`}
              </text>
            </g>
          )
        })}
        {/* 批次结束线 */}
        <line x1={labelW + prefillW + plotW} y1={8} x2={labelW + prefillW + plotW} y2={14 + N * rowH - 4} className="stroke-rose" strokeWidth={1.2} strokeDasharray="4 3" />
        <text x={labelW + prefillW + plotW} y={14 + N * rowH + 12} textAnchor="end" className="fill-current font-mono text-rose" fontSize={9}>
          {t('batch ends before new requests can enter →', '整批结束，新请求才能进来 →')}
        </text>
        {/* 图例 */}
        <g transform={`translate(${labelW}, ${14 + N * rowH + 6})`} className="text-ink3">
          <rect x={0} y={0} width={10} height={10} rx={2} className="fill-cyan/60" />
          <text x={14} y={9} className="fill-current font-mono" fontSize={8}>prefill</text>
          <rect x={62} y={0} width={10} height={10} rx={2} className="fill-volt/55" />
          <text x={76} y={9} className="fill-current font-mono" fontSize={8}>decode</text>
          <rect x={126} y={0} width={10} height={10} rx={2} fill="url(#sb-hatch)" />
          <text x={140} y={9} className="fill-current font-mono" fontSize={8}>{t('finished but holding a slot (idle)', '完成但占着槽位（空转）')}</text>
        </g>
      </svg>
    </Figure>
  )
}
