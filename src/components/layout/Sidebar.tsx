import { Link, NavLink, useLocation } from 'react-router-dom'
import { CHAPTERS, PARTS } from '@/lib/chapters'
import { useVisited } from '@/lib/progress'
import { pick, useLocale, useT } from '@/lib/i18n'
import { LanguageToggle } from '@/components/ui'

export function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const visited = useVisited()
  const { pathname } = useLocation()
  const { lang } = useLocale()
  const t = useT()

  return (
    <div className="flex h-full flex-col">
      {/* 品牌 */}
      <div className="flex items-center justify-between gap-2 border-b border-line px-5 py-[18px]">
        <Link to="/" onClick={onNavigate} className="block">
          <div className="font-display text-lg font-bold tracking-wide text-ink">
            EASY<span className="text-volt">INFRA</span>
          </div>
          <div className="microlabel mt-1">GPU · CUDA · LLM SYSTEMS</div>
        </Link>
        <LanguageToggle />
      </div>

      {/* Playground + 术语表 入口 */}
      <div className="space-y-2 px-4 pt-4">
        <NavLink
          to="/playground"
          onClick={onNavigate}
          className={({ isActive }) =>
            `flex items-center gap-2.5 rounded-md border px-3.5 py-2.5 transition-all ${
              isActive
                ? 'border-volt/55 bg-volt/8'
                : 'border-line2 bg-panel2/60 hover:border-volt/40 hover:bg-volt/5'
            }`
          }
        >
          <span className="font-mono text-sm text-volt">▶</span>
          <span>
            <span className="block font-mono text-xs font-medium tracking-wider text-ink">CUDA PLAYGROUND</span>
            <span className="block text-[11px] text-ink3">{t('A GPU simulator in your browser', '浏览器里的 GPU 模拟器')}</span>
          </span>
        </NavLink>
        <NavLink
          to="/glossary"
          onClick={onNavigate}
          className={({ isActive }) =>
            `flex items-center gap-2.5 rounded-md border px-3.5 py-2 transition-all ${
              isActive
                ? 'border-cyan/55 bg-cyan/8'
                : 'border-line bg-panel2/40 hover:border-cyan/40 hover:bg-cyan/5'
            }`
          }
        >
          <span className="font-mono text-sm text-cyan">⌗</span>
          <span>
            <span className="block font-mono text-xs font-medium tracking-wider text-ink">{t('GLOSSARY', '术语表')}</span>
            <span className="block text-[11px] text-ink3">{t('Every key term, one place', '关键术语速查')}</span>
          </span>
        </NavLink>
      </div>

      {/* 目录 */}
      <nav className="flex-1 overflow-y-auto px-4 py-4">
        {PARTS.map((part) => (
          <div key={part.num} className="mb-5">
            <div className="microlabel mb-1.5 px-1.5">
              {String(part.num).padStart(2, '0')} / {part.titleEn}
            </div>
            <ul>
              {CHAPTERS.filter((c) => c.part === part.num).map((c) => {
                const active = pathname === `/learn/${c.id}`
                const seen = visited.has(c.id)
                return (
                  <li key={c.id}>
                    <Link
                      to={`/learn/${c.id}`}
                      onClick={onNavigate}
                      className={`flex items-baseline gap-2.5 rounded px-1.5 py-[7px] text-[13.5px] leading-snug transition-colors ${
                        active
                          ? 'bg-panel2 text-ink shadow-[inset_2px_0_0_var(--color-volt)]'
                          : 'text-ink2 hover:bg-panel2/70 hover:text-ink'
                      }`}
                    >
                      <span className={`w-5 shrink-0 font-mono text-[11px] ${seen && !active ? 'text-volt' : 'text-ink3'}`}>
                        {seen && !active ? '✓' : String(c.num).padStart(2, '0')}
                      </span>
                      {pick(c.title, lang)}
                    </Link>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* 进度 */}
      <div className="border-t border-line px-5 py-4">
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="microlabel">{t('PROGRESS', '学习进度')}</span>
          <span className="font-mono text-xs text-ink2">
            {visited.size}<span className="text-ink3">/{CHAPTERS.length}</span>
          </span>
        </div>
        <div className="h-1 overflow-hidden rounded-full bg-line">
          <div
            className="h-full rounded-full bg-volt transition-all duration-500"
            style={{ width: `${(visited.size / CHAPTERS.length) * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}
