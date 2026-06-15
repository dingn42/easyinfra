import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { CHAPTERS } from '@/lib/chapters'
import { pick, useLocale } from '@/lib/i18n'

/**
 * 行内跨章引用链接。把正文里的"Chapter 6 / 第 6 章"变成可点跳转。
 * 用法：<ChapterLink n={6} />  →  渲染 "Chapter 6"（en）/ "第 6 章"（zh），点击跳到该章。
 * 可选 label 覆盖显示文字（如 "Chapter 6's Roofline" 里只链 "Chapter 6"）。
 * 在 .doc 容器内会自动继承链接样式（cyan 点状下划线）；title 悬浮显示章节标题。
 */
export function ChapterLink({ n, label }: { n: number; label?: ReactNode }) {
  const { lang } = useLocale()
  const ch = CHAPTERS.find((c) => c.num === n)
  const text = label ?? (lang === 'zh' ? `第 ${n} 章` : `Chapter ${n}`)
  if (!ch) return <>{text}</>
  return (
    <Link
      to={`/learn/${ch.id}`}
      title={pick(ch.title, lang)}
      className="font-medium text-cyan underline decoration-dotted underline-offset-2 transition-colors hover:decoration-solid"
    >
      {text}
    </Link>
  )
}
