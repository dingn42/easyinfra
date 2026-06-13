import { useT } from '@/lib/i18n'

const SHOW_MAX = 64

function fmtVal(v: number): string {
  if (Number.isInteger(v)) return String(v)
  const a = Math.abs(v)
  if (a >= 1000) return v.toFixed(0)
  if (a < 0.005) return v.toExponential(1)
  return v.toFixed(2)
}

/** 每个缓冲区一张表（前 64 元素网格），与初值不同的单元 volt 高亮 */
export function BuffersView({
  buffers,
  initial,
  order,
}: {
  buffers: Record<string, number[]> | null
  initial: Record<string, number[]> | null
  order: string[]
}) {
  const t = useT()
  if (!buffers) {
    return (
      <div className="flex flex-col items-center gap-2 py-14 text-center">
        <span className="microlabel text-rose">NO OUTPUT</span>
        <p className="text-[13px] text-ink3">
          {t('The run did not complete — no buffer contents to show.', '运行未完成，没有可展示的缓冲区内容。')}
        </p>
      </div>
    )
  }
  const names = order.filter((n) => buffers[n] != null)
  for (const n of Object.keys(buffers)) if (!names.includes(n)) names.push(n)

  return (
    <div className="space-y-5">
      {names.map((name) => {
        const data = buffers[name]
        const init = initial?.[name]
        const shown = data.slice(0, SHOW_MAX)
        const changed = init ? data.reduce((c, v, i) => (v !== init[i] ? c + 1 : c), 0) : 0
        return (
          <div key={name}>
            <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
              <span className="font-mono text-[13px] text-cyan">{name}</span>
              <span className="font-mono text-[11px] tabular-nums text-ink3">len {data.length}</span>
              {init && (
                <span className={`font-mono text-[11px] tabular-nums ${changed > 0 ? 'text-volt' : 'text-ink3'}`}>
                  {changed > 0
                    ? t(`◆ ${changed} element${changed === 1 ? '' : 's'} changed`, `◆ ${changed} 个元素被改变`)
                    : t('unchanged from init', '与初值一致')}
                </span>
              )}
            </div>
            <div className="grid grid-cols-4 gap-px overflow-hidden rounded-md border border-line bg-line/60 sm:grid-cols-8">
              {shown.map((v, i) => {
                const diff = init != null && v !== init[i]
                return (
                  <div
                    key={i}
                    title={
                      init && diff
                        ? `${name}[${i}]: ${fmtVal(init[i])} → ${fmtVal(v)}`
                        : `${name}[${i}] = ${fmtVal(v)}`
                    }
                    className={`px-1.5 py-1 text-center ${diff ? 'bg-volt/10' : 'bg-panel'}`}
                  >
                    <div className="font-mono text-[9px] leading-tight text-ink3/80">{i}</div>
                    <div
                      className={`truncate font-mono text-[11px] leading-tight tabular-nums ${
                        diff ? 'text-volt' : 'text-text'
                      }`}
                    >
                      {fmtVal(v)}
                    </div>
                  </div>
                )
              })}
            </div>
            {data.length > SHOW_MAX && (
              <div className="mt-1 font-mono text-[10.5px] text-ink3">
                {t(`… showing first ${SHOW_MAX} of ${data.length} elements`, `… 仅显示前 ${SHOW_MAX} / ${data.length} 个元素`)}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
