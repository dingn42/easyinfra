import { useT } from '@/lib/i18n'

/** 章节内容占位（构建期间显示） */
export function ChapterStub() {
  const t = useT()
  return (
    <div className="panel bg-dots my-10 flex flex-col items-center gap-3 px-6 py-20 text-center">
      <span className="microlabel animate-pulse text-volt">⌬ COMPILING…</span>
      <p className="text-sm text-ink2">{t('This chapter is compiling — back in a moment.', '本章内容正在编译，马上回来。')}</p>
    </div>
  )
}
