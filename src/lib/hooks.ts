import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

/** 元素进入视口时返回 true（一次性，用于 reveal 动画） */
export function useInView<T extends HTMLElement = HTMLDivElement>(
  options?: IntersectionObserverInit,
): [RefObject<T>, boolean] {
  const ref = useRef<T>(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || inView) return
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setInView(true)
          obs.disconnect()
        }
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.05, ...options },
    )
    obs.observe(el)
    return () => obs.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView])
  return [ref, inView]
}

/** localStorage 持久化 state（JSON 序列化，异常安全） */
export function useLocalStorage<T>(key: string, initial: T): [T, (v: T | ((p: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      return raw == null ? initial : (JSON.parse(raw) as T)
    } catch {
      return initial
    }
  })
  const set = useCallback(
    (v: T | ((p: T) => T)) => {
      setValue((prev) => {
        const next = typeof v === 'function' ? (v as (p: T) => T)(prev) : v
        try {
          localStorage.setItem(key, JSON.stringify(next))
        } catch {
          /* ignore */
        }
        return next
      })
    },
    [key],
  )
  return [value, set]
}

/**
 * requestAnimationFrame 循环。
 * cb(dtMs, elapsedMs)；running=false 时暂停。dt 已被钳制在 [0,100ms]，防止后台标签页跳变。
 */
export function useRafLoop(cb: (dt: number, t: number) => void, running: boolean) {
  const cbRef = useRef(cb)
  cbRef.current = cb
  useEffect(() => {
    if (!running) return
    let raf = 0
    let last = performance.now()
    const start = last
    const tick = (now: number) => {
      const dt = Math.min(100, now - last)
      last = now
      cbRef.current(dt, now - start)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [running])
}

/** setInterval 封装；delay 为 null 时暂停 */
export function useInterval(cb: () => void, delay: number | null) {
  const cbRef = useRef(cb)
  cbRef.current = cb
  useEffect(() => {
    if (delay == null) return
    const id = setInterval(() => cbRef.current(), delay)
    return () => clearInterval(id)
  }, [delay])
}

/** 用户是否偏好减少动画 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    const mq = matchMedia('(prefers-reduced-motion: reduce)')
    const f = () => setReduced(mq.matches)
    mq.addEventListener('change', f)
    return () => mq.removeEventListener('change', f)
  }, [])
  return reduced
}

/** ResizeObserver 测量元素尺寸 —— 响应式 SVG/Canvas 组件必备 */
export function useMeasure<T extends HTMLElement = HTMLDivElement>(): [
  RefObject<T>,
  { width: number; height: number },
] {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect
      if (r) setSize({ width: r.width, height: r.height })
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return [ref, size]
}
