import { useMemo, useState } from 'react'
import { highlightLines, type Lang } from '@/lib/highlight'

/**
 * 静态代码块（自带语法高亮、行号、行高亮、复制按钮）。
 * 可编辑场景请用 CodeMirror（@uiw/react-codemirror）。
 */
export function CodeBlock({
  code,
  lang = 'cuda',
  title,
  highlight,
  showLines,
}: {
  code: string
  lang?: Lang
  /** 标题栏文件名，如 "vector_add.cu"；省略则不渲染标题栏 */
  title?: string
  /** 需要高亮的行号（1-based） */
  highlight?: number[]
  /** 是否显示行号；默认 >4 行时显示 */
  showLines?: boolean
}) {
  const lines = useMemo(() => highlightLines(code, lang), [code, lang])
  const showNums = showLines ?? lines.length > 4
  const hl = useMemo(() => new Set(highlight ?? []), [highlight])
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="panel my-5 overflow-hidden">
      <div className="flex items-center gap-2 border-b border-line bg-panel2/60 px-4 py-2">
        <span className="size-2 rounded-full bg-line2" />
        {title ? (
          <span className="font-mono text-xs text-ink2">{title}</span>
        ) : (
          <span className="microlabel">CODE</span>
        )}
        <span className="ml-auto font-mono text-[10px] uppercase tracking-widest text-ink3">{lang}</span>
        <button
          onClick={copy}
          className="font-mono text-[11px] text-ink3 transition-colors hover:text-volt"
          title="复制代码"
        >
          {copied ? '✓ COPIED' : 'COPY'}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-[1.75] text-text">
        <code>
          {lines.map((tokens, i) => (
            <div
              key={i}
              className={`-mx-4 px-4 ${hl.has(i + 1) ? 'bg-volt/[0.07] shadow-[inset_2px_0_0_var(--color-volt)]' : ''}`}
            >
              {showNums && (
                <span className="mr-4 inline-block w-5 select-none text-right text-ink3/70">{i + 1}</span>
              )}
              {tokens.length > 0 ? tokens : ' '}
            </div>
          ))}
        </code>
      </pre>
    </div>
  )
}
