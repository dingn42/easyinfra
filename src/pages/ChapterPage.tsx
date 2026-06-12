import { Suspense, useEffect } from 'react'
import { Link, Navigate, useParams } from 'react-router-dom'
import { getChapter, getPart, prevNext } from '@/lib/chapters'
import { markVisited } from '@/lib/progress'
import { CHAPTER_COMPONENTS } from '@/chapters'

function ChapterLoading() {
  return (
    <div className="flex flex-col items-center gap-3 py-24">
      <span className="microlabel animate-pulse text-volt">⌬ LOADING MODULE…</span>
    </div>
  )
}

export default function ChapterPage() {
  const { id = '' } = useParams()
  const meta = getChapter(id)

  useEffect(() => {
    if (meta) {
      markVisited(meta.id)
      document.title = `${String(meta.num).padStart(2, '0')} ${meta.title} · EasyInfra`
    }
    return () => {
      document.title = 'EasyInfra · 从一个线程到一座 GPU 集群'
    }
  }, [meta])

  if (!meta) return <Navigate to="/" replace />
  const part = getPart(meta.part)
  const { prev, next } = prevNext(meta.id)
  const Cmp = CHAPTER_COMPONENTS[meta.id]

  return (
    <div>
      {/* 章头 */}
      <header className="bg-dots border-b border-line">
        <div className="mx-auto max-w-[820px] px-6 pb-10 pt-14 lg:px-8">
          <div className="microlabel mb-4">
            PART {String(meta.part).padStart(2, '0')} — {part.titleEn}
            <span className="mx-2 text-line2">/</span>
            CH {String(meta.num).padStart(2, '0')}
          </div>
          <h1 className="font-display text-3xl font-bold leading-tight text-ink sm:text-4xl">{meta.title}</h1>
          <p className="mt-3 text-[16px] text-ink2">{meta.tagline}</p>
          <div className="mt-6 flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] tracking-wider text-ink3">
            <span>◷ ~{meta.minutes} MIN</span>
            <span className="text-volt/80">⌬ {meta.labs.length} 个交互实验</span>
            <span className="hidden sm:inline">{meta.titleEn}</span>
          </div>
        </div>
      </header>

      {/* 正文 */}
      <div className="mx-auto max-w-[820px] px-6 pb-16 lg:px-8">
        <article className="doc">
          <Suspense fallback={<ChapterLoading />}>
            <Cmp />
          </Suspense>
        </article>

        {/* 上一章 / 下一章 */}
        <nav className="mt-20 grid gap-3 border-t border-line pt-8 sm:grid-cols-2">
          {prev ? (
            <Link
              to={`/learn/${prev.id}`}
              className="panel group px-5 py-4 transition-colors hover:border-line2"
            >
              <div className="microlabel mb-1.5">← PREV · CH {String(prev.num).padStart(2, '0')}</div>
              <div className="text-[15px] font-medium text-ink2 transition-colors group-hover:text-ink">
                {prev.title}
              </div>
            </Link>
          ) : (
            <Link to="/" className="panel group px-5 py-4 transition-colors hover:border-line2">
              <div className="microlabel mb-1.5">← HOME</div>
              <div className="text-[15px] font-medium text-ink2 transition-colors group-hover:text-ink">课程总览</div>
            </Link>
          )}
          {next ? (
            <Link
              to={`/learn/${next.id}`}
              className="panel group px-5 py-4 text-right transition-all hover:border-volt/50 hover:shadow-[0_0_24px_rgba(184,245,61,0.08)]"
            >
              <div className="microlabel mb-1.5 text-volt/80">NEXT · CH {String(next.num).padStart(2, '0')} →</div>
              <div className="text-[15px] font-medium text-ink transition-colors">{next.title}</div>
            </Link>
          ) : (
            <Link
              to="/playground"
              className="panel group px-5 py-4 text-right transition-all hover:border-volt/50 hover:shadow-[0_0_24px_rgba(184,245,61,0.08)]"
            >
              <div className="microlabel mb-1.5 text-volt/80">GRADUATE →</div>
              <div className="text-[15px] font-medium text-ink">去 Playground 写你自己的 kernel</div>
            </Link>
          )}
        </nav>
      </div>
    </div>
  )
}
