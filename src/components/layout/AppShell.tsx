import { useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { SidebarContent } from './Sidebar'

export function AppShell() {
  const [drawer, setDrawer] = useState(false)
  const { pathname } = useLocation()

  // 路由变化：滚回顶部 & 关闭抽屉
  useEffect(() => {
    window.scrollTo(0, 0)
    setDrawer(false)
  }, [pathname])

  return (
    <div className="min-h-dvh">
      {/* 桌面侧栏 */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[268px] border-r border-line bg-bg2/80 backdrop-blur lg:block">
        <SidebarContent />
      </aside>

      {/* 移动端顶栏 */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-bg/90 px-4 py-3 backdrop-blur lg:hidden">
        <Link to="/" className="font-display text-base font-bold text-ink">
          EASY<span className="text-volt">INFRA</span>
        </Link>
        <button
          onClick={() => setDrawer(true)}
          className="rounded border border-line2 px-3 py-1.5 font-mono text-xs text-ink2"
          aria-label="打开目录"
        >
          ☰ 目录
        </button>
      </header>

      {/* 移动端抽屉 */}
      {drawer && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setDrawer(false)} />
          <div className="absolute inset-y-0 left-0 w-[290px] border-r border-line bg-bg2">
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
