import { useEffect, useMemo, useState } from 'react'
import { GLOSSARY, type GlossaryTerm } from '@/lib/glossary'
import { pick, useLocale, useT } from '@/lib/i18n'
import { ChapterLink } from '@/components/ui'

function matches(term: GlossaryTerm, q: string): boolean {
  if (!q) return true
  const hay = (
    term.term.en + ' ' + term.term.zh + ' ' + (term.abbr ?? '') + ' ' + term.def.en + ' ' + term.def.zh
  ).toLowerCase()
  return hay.includes(q.toLowerCase())
}

export default function GlossaryPage() {
  const t = useT()
  const { lang } = useLocale()
  const [q, setQ] = useState('')

  useEffect(() => {
    document.title = t('Glossary · EasyInfra', '术语表 · EasyInfra')
    return () => {
      document.title =
        lang === 'zh' ? 'EasyInfra · 从一个线程到一座 GPU 集群' : 'EasyInfra · From One Thread to a GPU Cluster'
    }
  }, [t, lang])

  const groups = useMemo(
    () =>
      GLOSSARY.map((g) => ({ ...g, terms: g.terms.filter((term) => matches(term, q)) })).filter(
        (g) => g.terms.length > 0,
      ),
    [q],
  )
  const total = useMemo(() => GLOSSARY.reduce((n, g) => n + g.terms.length, 0), [])
  const shown = groups.reduce((n, g) => n + g.terms.length, 0)

  return (
    <div>
      {/* 页头 */}
      <header className="bg-dots border-b border-line">
        <div className="mx-auto max-w-[880px] px-6 pb-9 pt-14 lg:px-8">
          <div className="microlabel mb-4">REFERENCE · GLOSSARY</div>
          <h1 className="font-display text-3xl font-bold leading-tight text-ink sm:text-4xl">
            {t('Glossary', '术语表')}
          </h1>
          <p className="mt-3 max-w-[620px] text-[15px] leading-relaxed text-ink2">
            {t(
              'Every key term in the course, in one place — each links back to the chapter where it’s introduced. Search across both languages.',
              '课程里的每个关键术语集中在一处 —— 每条都链回引入它的章节。支持中英文搜索。',
            )}
          </p>

          {/* 搜索 */}
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <div className="relative w-full max-w-[420px]">
              <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 font-mono text-sm text-ink3">
                ⌕
              </span>
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder={t('Search terms… (e.g. warp, occupancy, KV)', '搜索术语…（如 warp、占用率、KV）')}
                className="w-full rounded-md border border-line2 bg-panel py-2 pl-9 pr-3 text-[14px] text-ink outline-none transition-colors placeholder:text-ink3 focus:border-volt/60"
              />
            </div>
            <span className="font-mono text-[11px] tracking-wider text-ink3">
              {q ? `${shown}/${total}` : total} {t('TERMS', '条术语')}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[880px] px-6 py-10 lg:px-8">
        {/* 分组快速跳转 */}
        {!q && (
          <div className="mb-8 flex flex-wrap gap-2">
            {GLOSSARY.map((g) => (
              <a
                key={g.title.en}
                href={`#${g.title.en.replace(/[^a-z]+/gi, '-').toLowerCase()}`}
                className="rounded-full border border-line bg-panel px-3 py-1 font-mono text-[11px] tracking-wider text-ink2 transition-colors hover:border-volt/50 hover:text-ink"
              >
                {pick(g.title, lang)}
              </a>
            ))}
          </div>
        )}

        {groups.length === 0 && (
          <div className="panel bg-dots flex flex-col items-center gap-2 px-6 py-16 text-center">
            <span className="microlabel text-ink3">∅ NO MATCH</span>
            <p className="text-sm text-ink2">{t('No term matches that search.', '没有匹配的术语。')}</p>
          </div>
        )}

        <div className="space-y-12">
          {groups.map((g) => (
            <section key={g.title.en} id={g.title.en.replace(/[^a-z]+/gi, '-').toLowerCase()} className="scroll-mt-6">
              <div className="mb-4 flex items-baseline gap-3 border-b border-line pb-2">
                <h2 className="font-display text-lg font-semibold text-ink">{pick(g.title, lang)}</h2>
                <span className="font-mono text-[11px] text-ink3">{g.terms.length}</span>
              </div>
              <dl className="grid gap-3 sm:grid-cols-2">
                {g.terms.map((term) => (
                  <div
                    key={term.slug}
                    id={term.slug}
                    className="panel scroll-mt-6 px-4 py-3.5 transition-colors hover:border-line2"
                  >
                    <dt className="mb-1 flex flex-wrap items-baseline gap-2">
                      <span className="text-[14.5px] font-semibold text-ink">{pick(term.term, lang)}</span>
                      {term.abbr && (
                        <span className="rounded border border-line bg-bg2 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-ink3">
                          {term.abbr}
                        </span>
                      )}
                      <span className="ml-auto font-mono text-[10.5px] tracking-wider text-ink3">
                        <ChapterLink n={term.chapter} />
                      </span>
                    </dt>
                    <dd className="text-[13.5px] leading-[1.8] text-text">{pick(term.def, lang)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </div>
    </div>
  )
}
