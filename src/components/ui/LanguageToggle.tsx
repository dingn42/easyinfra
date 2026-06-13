import { useLocale } from '@/lib/i18n'

/** EN / 中 语言切换。两段式分段按钮，等宽，融入仪器风格。 */
export function LanguageToggle({ className = '' }: { className?: string }) {
  const { lang, setLang } = useLocale()
  return (
    <div
      className={`inline-flex items-center rounded-md border border-line bg-panel p-0.5 ${className}`}
      role="group"
      aria-label="Language"
    >
      <button
        onClick={() => setLang('en')}
        aria-pressed={lang === 'en'}
        className={`rounded px-2 py-[3px] font-mono text-[11px] tracking-wider transition-colors ${
          lang === 'en' ? 'bg-volt/12 text-volt shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--color-volt)_30%,transparent)]' : 'text-ink3 hover:text-ink2'
        }`}
      >
        EN
      </button>
      <button
        onClick={() => setLang('zh')}
        aria-pressed={lang === 'zh'}
        className={`rounded px-2 py-[3px] font-mono text-[11px] tracking-wider transition-colors ${
          lang === 'zh' ? 'bg-volt/12 text-volt shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--color-volt)_30%,transparent)]' : 'text-ink3 hover:text-ink2'
        }`}
      >
        中
      </button>
    </div>
  )
}
