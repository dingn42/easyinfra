import { Btn } from '@/components/ui'
import { CHALLENGES, type ChallengeDef } from './examples'

export function Challenges({
  activeId,
  passed,
  onLoad,
}: {
  activeId: string | null
  passed: Record<string, boolean>
  onLoad: (ch: ChallengeDef) => void
}) {
  return (
    <section className="mt-12">
      <div className="microlabel mb-2">CHALLENGES</div>
      <h2 className="font-display text-xl font-semibold text-ink">挑战：修好这三个 kernel</h2>
      <p className="mt-1.5 max-w-[640px] text-[13.5px] leading-relaxed text-ink2">
        每个挑战载入一个有 bug（或低效）的 kernel。改完按 RUN，模拟器会用同样的配置跑一份参考实现，逐元素比对输出
        —— 通过即点亮 <span className="font-mono text-[12px] text-volt">✓ PASSED</span>。
      </p>
      <div className="mt-5 grid gap-4 md:grid-cols-3">
        {CHALLENGES.map((ch) => {
          const isActive = activeId === ch.id
          const isPassed = !!passed[ch.id]
          return (
            <div
              key={ch.id}
              className={`panel flex flex-col px-5 py-4 transition-all ${
                isActive ? 'border-volt/50 shadow-[0_0_24px_rgba(184,245,61,0.08)]' : ''
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="microlabel text-violet">CHALLENGE {String(ch.num).padStart(2, '0')}</span>
                {isPassed && (
                  <span className="ml-auto rounded border border-volt/50 bg-volt/15 px-1.5 py-0.5 font-mono text-[10px] tracking-widest text-volt">
                    ✓ PASSED
                  </span>
                )}
                {isActive && !isPassed && (
                  <span className="ml-auto rounded border border-line2 px-1.5 py-0.5 font-mono text-[10px] tracking-widest text-ink2">
                    LOADED
                  </span>
                )}
              </div>
              <div className="mt-2 text-[15px] font-medium text-ink">{ch.title}</div>
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink2">{ch.desc}</p>
              <p className="mt-2.5 border-l-2 border-volt/50 pl-2.5 text-[12px] leading-relaxed text-ink3">
                <span className="text-ink2">通关条件：</span>
                {ch.goal}
              </p>
              <div className="mt-auto pt-4">
                <Btn variant={isActive ? 'ghost' : 'solid'} onClick={() => onLoad(ch)}>
                  {isActive ? '↻ 重新载入' : '⌬ 载入挑战'}
                </Btn>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
