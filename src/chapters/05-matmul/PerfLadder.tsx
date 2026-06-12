import { Figure } from '@/components/ui'

/** SEC 05 静态图：优化阶梯，各级实现相对 cuBLAS 的大致百分比 */

const LEVELS: { name: string; pct: number; tone: string; bar: string }[] = [
  { name: 'naive', pct: 2, tone: 'text-rose', bar: 'bg-rose/60' },
  { name: '合并访存', pct: 8, tone: 'text-ink2', bar: 'bg-amber/45' },
  { name: 'shared tiling', pct: 40, tone: 'text-ink2', bar: 'bg-amber/60' },
  { name: '寄存器 tiling', pct: 70, tone: 'text-ink2', bar: 'bg-cyan/55' },
  { name: '向量化+双缓冲', pct: 85, tone: 'text-ink2', bar: 'bg-cyan/75' },
  { name: 'cuBLAS / Tensor Core', pct: 100, tone: 'text-volt', bar: 'bg-volt/70' },
]

export function PerfLadder() {
  return (
    <Figure caption="SGEMM 优化阶梯：各级 kernel 相对 cuBLAS 的大致性能（FP32、大矩阵、数据综合自 Simon Boehm 等公开复现实验，量级仅供直觉参考）">
      <div className="space-y-2">
        {LEVELS.map((l) => (
          <div key={l.name} className="flex items-center gap-2.5">
            <span className={`w-32 shrink-0 text-right font-mono text-[11px] sm:w-40 ${l.tone}`}>{l.name}</span>
            <div className="relative h-6 flex-1 overflow-hidden rounded-sm bg-bg2">
              <div className={`absolute inset-y-0 left-0 rounded-sm ${l.bar}`} style={{ width: `${l.pct}%` }} />
            </div>
            <span className={`w-12 shrink-0 font-mono text-[11px] tabular-nums ${l.tone}`}>{l.pct}%</span>
          </div>
        ))}
      </div>
    </Figure>
  )
}
