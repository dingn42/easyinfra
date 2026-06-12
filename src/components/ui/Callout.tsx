import type { ReactNode } from 'react'

const STYLES = {
  insight: { tone: 'var(--color-volt)', label: 'INSIGHT', zh: '直觉' },
  note: { tone: 'var(--color-cyan)', label: 'NOTE', zh: '注' },
  warn: { tone: 'var(--color-amber)', label: 'CAUTION', zh: '当心' },
  deep: { tone: 'var(--color-violet)', label: 'DEEP DIVE', zh: '深入一点' },
} as const

export type CalloutType = keyof typeof STYLES

/** 高亮文本块：insight（核心直觉）/ note / warn / deep（选读延伸） */
export function Callout({
  type = 'note',
  title,
  children,
}: {
  type?: CalloutType
  title?: string
  children: ReactNode
}) {
  const s = STYLES[type]
  return (
    <aside
      className="my-6 rounded-r-lg border-l-2 py-3.5 pl-5 pr-5"
      style={{
        borderColor: s.tone,
        background: `color-mix(in srgb, ${s.tone} 5%, transparent)`,
      }}
    >
      <div className="mb-1.5 flex items-baseline gap-2.5">
        <span className="microlabel" style={{ color: s.tone }}>
          ⌬ {s.label}
        </span>
        <span className="text-[13px] font-medium text-ink">{title ?? s.zh}</span>
      </div>
      <div className="text-[14.5px] leading-[1.95] text-text [&>p]:mb-2 [&>p:last-child]:mb-0">{children}</div>
    </aside>
  )
}
