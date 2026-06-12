import { useEffect, useState } from 'react'

/** 章节访问进度（localStorage 持久化 + 自定义事件同步） */

const KEY = 'ei-visited-v1'
const EVT = 'ei:visited'

export function getVisited(): Set<string> {
  try {
    const raw = localStorage.getItem(KEY)
    return new Set(raw ? (JSON.parse(raw) as string[]) : [])
  } catch {
    return new Set()
  }
}

export function markVisited(id: string) {
  const set = getVisited()
  if (set.has(id)) return
  set.add(id)
  try {
    localStorage.setItem(KEY, JSON.stringify([...set]))
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new Event(EVT))
}

export function useVisited(): Set<string> {
  const [visited, setVisited] = useState<Set<string>>(getVisited)
  useEffect(() => {
    const f = () => setVisited(getVisited())
    window.addEventListener(EVT, f)
    window.addEventListener('storage', f)
    return () => {
      window.removeEventListener(EVT, f)
      window.removeEventListener('storage', f)
    }
  }, [])
  return visited
}
