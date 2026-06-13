import { useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { SidebarContent } from './Sidebar'
import { LanguageToggle } from '@/components/ui'
import { useT } from '@/lib/i18n'

export function AppShell() {
  const [drawer, setDrawer] = useState(false)
  const { pathname } = useLocation()
  const t = useT()

  // 路由变化：滚回顶部 & 关闭抽屉
  useEffect(() => {
    window.scrollTo(0, 0)
    setDrawer(false)
  }, [pathname])

  return (
    <div className="min-h-dvh">
      {/* 桌面侧栏 */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[268px] border-r border-line bg-bg2/70 backdrop-blur lg:block">
        <SidebarContent />
      </aside>

      {/* 移动端顶栏 */}
      <header className="sticky top-0 z-30 flex items-center justify-between gap-2 border-b border-line bg-bg/85 px-4 py-3 backdrop-blur lg:hidden">
        <Link to="/" className="font-display text-base font-bold text-ink">
          EASY<span className="text-volt">INFRA</span>
        </Link>
        <div className="flex items-center gap-2">
          <LanguageToggle />
          <button
            onClick={() => setDrawer(true)}
            className="rounded border border-line2 px-3 py-1.5 font-mono text-xs text-ink2"
            aria-label={t('Open menu', '打开目录')}
          >
            ☰ {t('Menu', '目录')}
          </button>
        </div>
      </header>

      {/* 移动端抽屉 */}
      {drawer && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-ink/30 backdrop-blur-sm" onClick={() => setDrawer(false)} />
          <div className="absolute inset-y-0 left-0 w-[290px] border-r border-line bg-bg2 lift">
            <SidebarContent onNavigate={() => setDrawer(false)} />
          </div>
        </div>
      )}

      <div className="lg:pl-[268px]">
        <main className="min-h-dvh">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
