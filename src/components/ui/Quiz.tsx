import { useState, type ReactNode } from 'react'
import { useT } from '@/lib/i18n'

export interface QuizOption {
  text: ReactNode
  correct?: boolean
  /** 选择该项后展示的解释 */
  explain?: ReactNode
}

/** 单选小测。选择后即时反馈并展示解释。 */
export function Quiz({ question, options }: { question: ReactNode; options: QuizOption[] }) {
  const [picked, setPicked] = useState<number | null>(null)
  const t = useT()
  const done = picked != null

  return (
    <div className="panel my-8 overflow-hidden">
      <div className="flex items-center gap-3 border-b border-line bg-panel2/60 px-4 py-2.5">
        <span className="microlabel text-cyan">? QUIZ</span>
        <span className="text-[13px] text-ink3">{t('Check your intuition', '检验一下直觉')}</span>
      </div>
      <div className="p-4 sm:p-5">
        <p className="mb-4 text-[15px] font-medium leading-relaxed text-ink">{question}</p>
        <div className="space-y-2">
          {options.map((o, i) => {
            let cls = 'border-line bg-bg2 text-text hover:border-line2 hover:bg-panel2'
            if (done) {
              if (o.correct) cls = 'border-volt/60 bg-volt/[0.08] text-ink'
              else if (i === picked) cls = 'border-rose/60 bg-rose/[0.08] text-ink'
              else cls = 'border-line bg-bg2 text-ink3'
            }
            return (
              <button
                key={i}
                disabled={done}
                onClick={() => setPicked(i)}
                className={`flex w-full items-start gap-3 rounded-md border px-3.5 py-2.5 text-left text-[14px] leading-relaxed transition-colors disabled:cursor-default ${cls}`}
              >
                <span className="mt-px font-mono text-xs text-ink3">{String.fromCharCode(65 + i)}</span>
                <span className="flex-1">{o.text}</span>
                {done && o.correct && <span className="font-mono text-xs text-volt">✓</span>}
                {done && i === picked && !o.correct && <span className="font-mono text-xs text-rose">✗</span>}
              </button>
            )
          })}
        </div>
        {done && (
          <div className="mt-4 rounded-md border border-line bg-bg2 px-4 py-3 text-[13.5px] leading-[1.85] text-text">
            <span className={`mr-2 font-mono text-xs ${options[picked].correct ? 'text-volt' : 'text-rose'}`}>
              {options[picked].correct ? t('✓ Correct', '✓ 正确') : t('✗ Not quite', '✗ 不对')}
            </span>
            {options[picked].explain ?? options.find((o) => o.correct)?.explain}
            <button
              onClick={() => setPicked(null)}
              className="ml-3 font-mono text-[11px] tracking-wider text-ink3 underline decoration-dotted underline-offset-4 hover:text-ink2"
            >
              {t('Try again', '再试一次')}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
