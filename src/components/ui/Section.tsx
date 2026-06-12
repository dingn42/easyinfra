import type { ReactNode } from 'react'
import { useInView } from '@/lib/hooks'

/**
 * 章节内的一节。自动渲染 "SEC 01" 微标签 + 标题，进入视口时上浮显现。
 */
export function Section({
  index,
  title,
  id,
  lead,
  children,
}: {
  /** 节序号，1 → "SEC 01" */
  index: number
  title: string
  /** 锚点 id（可选） */
  id?: string
  /** 标题下的一句导语（可选） */
  lead?: ReactNode
  children: ReactNode
}) {
  const [ref, inView] = useInView<HTMLElement>()
  return (
    <section ref={ref} id={id} className={`reveal mt-16 first:mt-10 ${inView ? 'is-in' : ''}`}>
      <div className="mb-5 flex items-baseline gap-3 border-b border-line pb-3">
        <span className="microlabel text-volt">
          SEC {String(index).padStart(2, '0')}
        </span>
        <h2 className="font-display text-[22px] font-semibold text-ink">{title}</h2>
      </div>
      {lead && <p className="-mt-1 mb-5 text-[15px] leading-relaxed text-ink2">{lead}</p>}
      {children}
    </section>
  )
}
