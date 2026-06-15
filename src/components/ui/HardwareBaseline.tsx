import type { ReactNode } from 'react'
import { GPUS, type GpuId } from '@/lib/hardware'
import { pick, useLocale, useT } from '@/lib/i18n'

/**
 * 章首"参考硬件"声明条。让每章明确它的数字假设的是哪块卡，统一全站基线。
 * 用法：<HardwareBaseline ids={['a100']} />  或多卡 ids={['a100','h100']} + 可选 note。
 */
export function HardwareBaseline({ ids, note }: { ids: GpuId[]; note?: ReactNode }) {
  const t = useT()
  const { lang } = useLocale()
  const cards = ids.map((id) => GPUS[id])

  return (
    <div className="my-6 overflow-hidden rounded-lg border border-line bg-bg2/60">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line px-4 py-2">
        <span className="microlabel text-volt">⌬ {t('REFERENCE GPU', '参考硬件')}</span>
        <span className="text-[13px] font-medium text-ink">{cards.map((c) => c.name).join(' · ')}</span>
        <span className="ml-auto text-[11.5px] text-ink3">
          {note ?? t('numbers below assume this card unless a figure says otherwise', '除非另有标注，本章数字均以此卡为准')}
        </span>
      </div>
      <div className="flex flex-col divide-y divide-line">
        {cards.map((c) => (
          <div key={c.id} className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-4 py-2.5">
            {cards.length > 1 && (
              <span className="font-mono text-[11px] font-medium tracking-wider text-ink2">{c.short}</span>
            )}
            {c.specs.map((s) => (
              <span key={s.label.en} className="inline-flex items-baseline gap-1.5">
                <span className="font-mono text-[10.5px] uppercase tracking-wider text-ink3">{pick(s.label, lang)}</span>
                <span className="font-mono text-[12px] tabular-nums text-ink">{s.value}</span>
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
