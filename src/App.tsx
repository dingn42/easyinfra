import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import Home from '@/pages/Home'
import ChapterPage from '@/pages/ChapterPage'

const PlaygroundPage = lazy(() => import('@/pages/PlaygroundPage'))

function PageLoading() {
  return (
    <div className="flex flex-col items-center gap-3 py-32">
      <span className="microlabel animate-pulse text-volt">⌬ LOADING…</span>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Home />} />
        <Route path="/learn/:id" element={<ChapterPage />} />
        <Route
          path="/playground"
          element={
            <Suspense fallback={<PageLoading />}>
              <PlaygroundPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
