import type { ReactNode } from 'react'

/** 插图容器 + 编号说明 */
export function Figure({ caption, children }: { caption?: ReactNode; children: ReactNode }) {
  return (
    <figure className="my-6">
      <div className="panel overflow-hidden p-4 sm:p-5">{children}</div>
      {caption && (
        <figcaption className="mt-2 text-center text-[12.5px] leading-relaxed text-ink3">{caption}</figcaption>
      )}
    </figure>
  )
}
