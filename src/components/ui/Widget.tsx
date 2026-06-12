import type { ReactNode } from 'react'

/**
 * 交互实验容器（"LAB"）。统一的仪器风格外框：标题栏 + 四角括号 + 可选重置。
 * wide=true 时在大屏上向两侧溢出排版栏。
 */
export function Widget({
  index,
  title,
  subtitle,
  onReset,
  footer,
  wide,
  children,
}: {
  /** 实验序号，1 → "LAB 01" */
  index: number
  title: string
  subtitle?: string
  onReset?: () => void
  /** 底部说明条（可选） */
  footer?: ReactNode
  wide?: boolean
  children: ReactNode
}) {
  return (
    <div className={`group relative my-8 ${wide ? 'lg:-mx-16' : ''}`}>
      {/* 四角括号 */}
      {(['top-0 left-0 border-t border-l', 'top-0 right-0 border-t border-r', 'bottom-0 left-0 border-b border-l', 'bottom-0 right-0 border-b border-r'] as const).map(
        (pos) => (
          <span
            key={pos}
            aria-hidden
            className={`pointer-events-none absolute size-3 border-line2 transition-colors duration-300 group-hover:border-volt/70 ${pos}`}
            style={{ margin: '-5px' }}
          />
        ),
      )}
      <div className="panel overflow-hidden">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line bg-panel2/60 px-4 py-2.5">
          <span className="microlabel text-volt">⌬ LAB {String(index).padStart(2, '0')}</span>
          <span className="text-sm font-medium text-ink">{title}</span>
          {subtitle && <span className="text-xs text-ink3">{subtitle}</span>}
          {onReset && (
            <button
              onClick={onReset}
              className="ml-auto font-mono text-[11px] tracking-wider text-ink3 transition-colors hover:text-volt"
              title="重置"
            >
              ↺ RESET
            </button>
          )}
        </div>
        <div className="p-4 sm:p-5">{children}</div>
        {footer && (
          <div className="border-t border-line bg-bg2/50 px-4 py-2.5 text-[12.5px] leading-relaxed text-ink2">
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}
