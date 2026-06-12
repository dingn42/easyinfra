import { useMemo } from 'react'
import katex from 'katex'

/** KaTeX 公式。block=true 渲染为独立展示行。 */
export function MathTex({ tex, block }: { tex: string; block?: boolean }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: !!block,
        throwOnError: false,
        strict: false,
      }),
    [tex, block],
  )
  if (block) {
    return (
      <div
        className="my-5 overflow-x-auto py-1 text-center text-[15px] text-ink"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }
  return <span className="text-ink" dangerouslySetInnerHTML={{ __html: html }} />
}
