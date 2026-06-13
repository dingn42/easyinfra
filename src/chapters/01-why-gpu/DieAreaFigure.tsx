import { Figure } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { ChipDie, DieLegend } from './ChipDie'

/**
 * SEC1 静态插图：CPU 与 GPU 的 die 面积分配对比。
 * 数字与 LAB02 的两个预设一致，正文里会回收这个呼应。
 */
export function DieAreaFigure() {
  const t = useT()
  return (
    <Figure
      caption={t(
        <>
          Where the transistor budget goes (illustrative proportions, not a real floorplan): the CPU spends the
          bulk on cache and control logic so its few cores never stall; the GPU spends the bulk on ALUs, trading
          quantity for throughput. In the lab in SEC 04 you can drag out both of these layouts yourself.
        </>,
        <>
          晶体管预算的去向（示意比例，非实测版图）：CPU 把大头花在缓存与控制逻辑上，让少数几个核心不等待；
          GPU 把大头花在 ALU 上，用数量换吞吐。SEC 04 的实验里你可以亲手拖出这两张图。
        </>,
      )}
    >
      <div className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
        <div>
          <div className="mb-2 text-center font-mono text-xs tracking-wider text-ink2">
            CPU<span className="ml-2 text-ink3">{t('latency machine', '延迟机器')}</span>
          </div>
          <ChipDie ctrl={30} cache={50} alu={20} cols={4} />
          <DieLegend ctrl={30} cache={50} alu={20} />
        </div>
        <div>
          <div className="mb-2 text-center font-mono text-xs tracking-wider text-ink2">
            GPU<span className="ml-2 text-ink3">{t('throughput machine', '吞吐机器')}</span>
          </div>
          <ChipDie ctrl={8} cache={12} alu={80} cols={12} />
          <DieLegend ctrl={8} cache={12} alu={80} />
        </div>
      </div>
    </Figure>
  )
}
