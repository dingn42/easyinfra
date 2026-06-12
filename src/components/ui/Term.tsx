import type { ReactNode } from 'react'

/** 术语：点状下划线 + 悬停释义气泡 */
export function Term({ t, children }: { t: ReactNode; children: ReactNode }) {
  return (
    <span className="group/term relative cursor-help border-b border-dotted border-ink3 text-ink">
      {t}
      <span
        className="pointer-events-none absolute bottom-full left-1/2 z-40 mb-2 w-64 -translate-x-1/2 rounded-md border border-line2 bg-panel2 px-3.5 py-2.5 text-left text-[12.5px] font-normal leading-[1.8] text-text opacity-0 shadow-xl shadow-black/40 transition-opacity duration-150 group-hover/term:opacity-100"
      >
        {children}
      </span>
    </span>
  )
}
