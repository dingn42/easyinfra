import type { ReactNode } from 'react'
import { useT } from '@/lib/i18n'

/** ── 基础控件族：Slider / Segmented / Toggle / Btn / Stat / PlayBar ── */

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  fmt = (v) => String(v),
  unit = '',
  disabled,
  className = '',
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  /** 数值显示格式化 */
  fmt?: (v: number) => string
  unit?: string
  disabled?: boolean
  className?: string
}) {
  return (
    <label className={`block ${className}`}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="font-mono text-[11px] uppercase tracking-wider text-ink2">{label}</span>
        <span className="font-mono text-xs text-volt">
          {fmt(value)}
          {unit && <span className="ml-0.5 text-ink3">{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        className="ei-range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  block,
}: {
  options: { value: T; label: ReactNode }[]
  value: T
  onChange: (v: T) => void
  /** 占满整行 */
  block?: boolean
}) {
  return (
    <div className={`${block ? 'flex w-full' : 'inline-flex'} rounded-md border border-line bg-bg2 p-0.5`}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          onClick={() => onChange(o.value)}
          className={`${block ? 'flex-1' : ''} rounded px-2.5 py-1 font-mono text-xs transition-colors ${
            o.value === value ? 'bg-panel2 text-volt shadow-[inset_0_0_0_1px_var(--color-line2)]' : 'text-ink3 hover:text-ink2'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: ReactNode
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="group/t inline-flex items-center gap-2.5"
      role="switch"
      aria-checked={checked}
    >
      <span
        className={`relative h-[18px] w-[34px] rounded border transition-colors ${
          checked ? 'border-volt/60 bg-volt/15' : 'border-line2 bg-bg2'
        }`}
      >
        <span
          className={`absolute top-[2px] size-[12px] rounded-sm transition-all ${
            checked ? 'left-[18px] bg-volt shadow-[0_0_6px_rgba(184,245,61,0.6)]' : 'left-[2px] bg-ink3'
          }`}
        />
      </span>
      <span className={`text-xs transition-colors ${checked ? 'text-ink' : 'text-ink2'}`}>{label}</span>
    </button>
  )
}

export function Btn({
  children,
  onClick,
  variant = 'solid',
  size = 'md',
  disabled,
  className = '',
  title,
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'solid' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
  disabled?: boolean
  className?: string
  title?: string
}) {
  const base =
    'inline-flex items-center justify-center gap-1.5 rounded-md font-mono transition-all disabled:opacity-40 disabled:cursor-not-allowed'
  const sizes = { sm: 'px-2.5 py-1 text-[11px]', md: 'px-3.5 py-1.5 text-xs' }
  const variants = {
    solid:
      'bg-volt/15 text-volt border border-volt/40 hover:bg-volt/25 hover:shadow-[0_0_16px_rgba(184,245,61,0.25)]',
    ghost: 'border border-line2 text-ink2 hover:text-ink hover:border-ink3',
    danger: 'border border-rose/40 text-rose bg-rose/10 hover:bg-rose/20',
  }
  return (
    <button onClick={onClick} disabled={disabled} title={title} className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}>
      {children}
    </button>
  )
}

/** 数据读出：大号等宽数值 + 小标签 */
export function Stat({
  label,
  value,
  unit,
  tone = 'ink',
  size = 'md',
}: {
  label: string
  value: ReactNode
  unit?: string
  tone?: 'ink' | 'volt' | 'cyan' | 'amber' | 'rose' | 'violet'
  size?: 'sm' | 'md' | 'lg'
}) {
  const colors = {
    ink: 'text-ink',
    volt: 'text-volt',
    cyan: 'text-cyan',
    amber: 'text-amber',
    rose: 'text-rose',
    violet: 'text-violet',
  }
  const sizes = { sm: 'text-base', md: 'text-xl', lg: 'text-3xl' }
  return (
    <div>
      <div className={`font-mono font-medium tabular-nums ${sizes[size]} ${colors[tone]}`}>
        {value}
        {unit && <span className="ml-1 text-[0.6em] font-normal text-ink3">{unit}</span>}
      </div>
      <div className="microlabel mt-0.5">{label}</div>
    </div>
  )
}

/** 动画播放控制条：播放/暂停 + 单步 + 重置 + 速度 */
export function PlayBar({
  playing,
  onToggle,
  onStep,
  onReset,
  speed,
  onSpeed,
  extra,
}: {
  playing: boolean
  onToggle: () => void
  onStep?: () => void
  onReset?: () => void
  speed?: number
  onSpeed?: (s: number) => void
  /** 追加的自定义控件 */
  extra?: ReactNode
}) {
  const t = useT()
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Btn onClick={onToggle} className="min-w-[88px]">
        {playing ? t('❚❚ Pause', '❚❚ 暂停') : t('▶ Play', '▶ 播放')}
      </Btn>
      {onStep && (
        <Btn variant="ghost" onClick={onStep} disabled={playing} title={t('Step', '单步执行')}>
          {t('⇥ Step', '⇥ 单步')}
        </Btn>
      )}
      {onReset && (
        <Btn variant="ghost" onClick={onReset} title={t('Reset', '重置')}>
          ↺
        </Btn>
      )}
      {speed != null && onSpeed && (
        <Segmented
          options={[
            { value: 0.5, label: '0.5×' },
            { value: 1, label: '1×' },
            { value: 2, label: '2×' },
            { value: 4, label: '4×' },
          ]}
          value={speed}
          onChange={onSpeed}
        />
      )}
      {extra}
    </div>
  )
}
