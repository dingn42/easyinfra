import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'

/**
 * 双语系统（英文默认，可切中文）。
 * Bilingual layer — English by default, Chinese optional.
 *
 * 用法 / Usage:
 *   const t = useT()
 *   <p>{t('English copy', '中文文案')}</p>           // 字符串或任意 ReactNode 均可
 *   const { lang, setLang } = useLocale()
 *
 * 数据（如 chapters.ts）用 Loc<T> 结构 + pick() 解析：
 *   pick(meta.title, lang)
 */

export type Lang = 'en' | 'zh'

/** 一份双语内容 */
export interface Loc<T = string> {
  en: T
  zh: T
}

const STORAGE_KEY = 'ei-lang-v1'

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'en' || saved === 'zh') return saved
  } catch {
    /* ignore */
  }
  return 'en'
}

interface LocaleValue {
  lang: Lang
  setLang: (l: Lang) => void
  toggle: () => void
}

const LocaleContext = createContext<LocaleValue | null>(null)

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang)

  const setLang = useCallback((l: Lang) => {
    setLangState(l)
    try {
      localStorage.setItem(STORAGE_KEY, l)
    } catch {
      /* ignore */
    }
  }, [])

  const toggle = useCallback(() => setLang(lang === 'en' ? 'zh' : 'en'), [lang, setLang])

  useEffect(() => {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en'
  }, [lang])

  return <LocaleContext.Provider value={{ lang, setLang, toggle }}>{children}</LocaleContext.Provider>
}

export function useLocale(): LocaleValue {
  const ctx = useContext(LocaleContext)
  if (!ctx) throw new Error('useLocale must be used within <LocaleProvider>')
  return ctx
}

/**
 * 返回一个 picker：t(en, zh) → 当前语言对应的值。
 * 因为只是二选一返回，所以同样适用于 ReactNode（含内联组件的富文本）。
 */
export function useT(): <T>(en: T, zh: T) => T {
  const { lang } = useLocale()
  return function t<T>(en: T, zh: T): T {
    return lang === 'zh' ? zh : en
  }
}

/** 从 Loc<T> 取出当前语言的值（非组件场景，如排序/标题计算） */
export function pick<T>(loc: Loc<T>, lang: Lang): T {
  return lang === 'zh' ? loc.zh : loc.en
}
