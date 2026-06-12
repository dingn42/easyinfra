import { Slider } from '@/components/ui'
import type { BufferDecl, KernelParam } from '../../lib/cudasim/types'

/** 页面内可编辑的缓冲区行（init 多一个 'custom'：示例给的显式数组） */
export interface BufferRow {
  name: string
  length: number
  init: 'zero' | 'iota' | 'random' | 'custom'
  custom?: number[]
}

export function rowsToDecls(rows: BufferRow[]): BufferDecl[] {
  return rows.map((r) => ({
    name: r.name,
    length: r.length,
    init: r.init === 'custom' ? (r.custom ?? []) : r.init,
  }))
}

export function declsToRows(decls: BufferDecl[]): BufferRow[] {
  return decls.map((d) => {
    const init = d.init ?? 'zero'
    return Array.isArray(init)
      ? { name: d.name, length: d.length, init: 'custom' as const, custom: init }
      : { name: d.name, length: d.length, init }
  })
}

const INIT_LABEL: Record<BufferRow['init'], string> = {
  zero: 'zero · 全 0',
  iota: 'iota · 0,1,2…',
  random: 'random · 伪随机',
  custom: 'custom · 自定义',
}

export function ConfigPanel({
  gridX,
  blockX,
  onGrid,
  onBlock,
  scalarParams,
  scalars,
  onScalar,
  rows,
  onRows,
  disabled,
}: {
  gridX: number
  blockX: number
  onGrid: (v: number) => void
  onBlock: (v: number) => void
  /** 来自最近一次成功 compile 的标量参数 */
  scalarParams: KernelParam[]
  scalars: Record<string, string>
  onScalar: (name: string, v: string) => void
  rows: BufferRow[]
  onRows: (rows: BufferRow[]) => void
  disabled?: boolean
}) {
  const total = gridX * blockX
  const overLimit = blockX > 1024 || total > 65536

  const setRow = (i: number, patch: Partial<BufferRow>) => {
    const next = rows.slice()
    next[i] = { ...next[i], ...patch }
    onRows(next)
  }

  return (
    <div className="panel px-4 py-4">
      <div className="microlabel mb-3">LAUNCH CONFIG</div>

      {/* grid / block */}
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
        <Slider label="GRID.X" value={gridX} min={1} max={64} onChange={onGrid} disabled={disabled} unit="blocks" />
        <Slider label="BLOCK.X" value={blockX} min={1} max={1024} onChange={onBlock} disabled={disabled} unit="threads" />
      </div>
      <div className={`mt-1.5 font-mono text-[11px] tabular-nums ${overLimit ? 'text-rose' : 'text-ink3'}`}>
        TOTAL = {gridX} × {blockX} = {total.toLocaleString('en-US')} threads
        {overLimit && '  ⚠ 超出上限（block ≤ 1024，总线程 ≤ 65536）'}
      </div>

      {/* 标量参数（从 kernel.info.params 自动生成） */}
      {scalarParams.length > 0 && (
        <div className="mt-4 border-t border-line pt-3.5">
          <div className="microlabel mb-2.5">SCALAR ARGS</div>
          <div className="flex flex-wrap gap-x-5 gap-y-2.5">
            {scalarParams.map((p) => {
              const raw = scalars[p.name] ?? ''
              const bad = raw.trim() === '' || Number.isNaN(Number(raw))
              return (
                <label key={p.name} className="flex items-center gap-2">
                  <span className="font-mono text-xs text-ink2">
                    <span className="text-violet">{p.type}</span> {p.name} =
                  </span>
                  <input
                    type="text"
                    inputMode="numeric"
                    value={raw}
                    disabled={disabled}
                    onChange={(e) => onScalar(p.name, e.target.value)}
                    className={`w-24 rounded border bg-bg2 px-2 py-1 font-mono text-xs tabular-nums text-ink outline-none transition-colors focus:border-volt/60 disabled:opacity-40 ${
                      bad ? 'border-rose/60' : 'border-line2'
                    }`}
                  />
                </label>
              )
            })}
          </div>
        </div>
      )}

      {/* buffers 表 */}
      {rows.length > 0 && (
        <div className="mt-4 border-t border-line pt-3.5">
          <div className="microlabel mb-2.5">DEVICE BUFFERS</div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse font-mono text-xs">
              <thead>
                <tr className="text-left text-[10.5px] uppercase tracking-wider text-ink3">
                  <th className="pb-1.5 pr-4 font-normal">NAME</th>
                  <th className="pb-1.5 pr-4 font-normal">LENGTH</th>
                  <th className="pb-1.5 font-normal">INIT</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.name} className="border-t border-line/60">
                    <td className="py-1.5 pr-4 text-cyan">{r.name}</td>
                    <td className="py-1.5 pr-4">
                      <input
                        type="number"
                        min={1}
                        max={65536}
                        value={r.length}
                        disabled={disabled}
                        onChange={(e) => {
                          const v = Math.max(1, Math.min(65536, Math.trunc(Number(e.target.value) || 1)))
                          setRow(i, { length: v })
                        }}
                        className="w-24 rounded border border-line2 bg-bg2 px-2 py-1 tabular-nums text-ink outline-none transition-colors focus:border-volt/60 disabled:opacity-40"
                      />
                    </td>
                    <td className="py-1.5">
                      <select
                        value={r.init}
                        disabled={disabled}
                        onChange={(e) => setRow(i, { init: e.target.value as BufferRow['init'] })}
                        className="rounded border border-line2 bg-bg2 px-2 py-1 text-xs text-ink outline-none transition-colors focus:border-volt/60 disabled:opacity-40"
                      >
                        {(['zero', 'iota', 'random'] as const).map((o) => (
                          <option key={o} value={o}>
                            {INIT_LABEL[o]}
                          </option>
                        ))}
                        {r.custom && <option value="custom">{INIT_LABEL.custom}</option>}
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-ink3">
            缓冲区按<span className="text-ink2">同名</span>绑定到 kernel 的指针参数；改源码参数名，表会自动跟随。
          </p>
        </div>
      )}
    </div>
  )
}
