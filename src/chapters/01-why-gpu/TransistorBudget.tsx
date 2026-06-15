import { useState } from 'react'
import { Btn, Slider, Stat, Widget } from '@/components/ui'
import { clamp } from '@/lib/format'
import { useT } from '@/lib/i18n'
import { ChipDie, DieLegend, aluCellCount } from './ChipDie'

/**
 * LAB 02「晶体管预算分配器」：
 * 三个滑杆把 die 面积分给 控制逻辑 / 缓存 / ALU（总和恒为 100%，
 * 拖动一个时其余两项按当前比例缩放），SVG 芯片图实时重绘，
 * 读数给出单线程性能分（收益递减）与吞吐分（正比 ALU）。
 */

type Key = 'ctrl' | 'cache' | 'alu'
interface Alloc {
  ctrl: number
  cache: number
  alu: number
}

const KEYS: Key[] = ['ctrl', 'cache', 'alu']
const CPU_PRESET: Alloc = { ctrl: 30, cache: 50, alu: 20 }
const GPU_PRESET: Alloc = { ctrl: 8, cache: 12, alu: 80 }
const ALU_COLS = 12

/** 把 key 设为 v，其余两项按当前比例分掉剩余额度 */
function rebalance(prev: Alloc, key: Key, raw: number): Alloc {
  const v = clamp(raw, 0, 100)
  const [a, b] = KEYS.filter((k) => k !== key) as [Key, Key]
  const rest = 100 - v
  const sum = prev[a] + prev[b]
  const next: Alloc = { ...prev, [key]: v }
  if (sum <= 1e-6) {
    next[a] = rest / 2
    next[b] = rest / 2
  } else {
    next[a] = (prev[a] / sum) * rest
    next[b] = (prev[b] / sum) * rest
  }
  return next
}

/** 单线程性能分：控制 + 缓存的收益递减（1 - e^(-x/40)），满分 100 */
function singleThreadScore(al: Alloc): number {
  return Math.round(100 * (1 - Math.exp(-(al.ctrl + al.cache) / 40)))
}

/** 吞吐分：正比于 ALU 占比，满分 100 */
function throughputScore(al: Alloc): number {
  return Math.round(al.alu)
}

const SLIDER_META: { key: Key; en: string; zh: string }[] = [
  { key: 'ctrl', en: 'Control logic (branch prediction / OoO)', zh: '控制逻辑（分支预测 / 乱序）' },
  { key: 'cache', en: 'Cache (L1 / L2 / L3)', zh: '缓存（L1 / L2 / L3）' },
  { key: 'alu', en: 'ALU (the part that actually does arithmetic)', zh: 'ALU（真正做算术的部分）' },
]

export function TransistorBudget({ index }: { index: number }) {
  const t = useT()
  const [alloc, setAlloc] = useState<Alloc>(CPU_PRESET)

  // 展示用整数，保证三项之和恰为 100 且不出现负数
  const dCtrl = Math.round(alloc.ctrl)
  let dCache = Math.round(alloc.cache)
  let dAlu = 100 - dCtrl - dCache
  if (dAlu < 0) {
    dCache += dAlu
    dAlu = 0
  }
  const display: Alloc = { ctrl: dCtrl, cache: dCache, alu: dAlu }

  const single = singleThreadScore(alloc)
  const thr = throughputScore(alloc)
  const cells = aluCellCount(alloc.ctrl, alloc.cache, alloc.alu, ALU_COLS)

  const setPart = (key: Key) => (v: number) => setAlloc((p) => rebalance(p, key, v))

  return (
    <Widget
      index={index}
      title={t('Transistor Budget Allocator', '晶体管预算分配器')}
      subtitle={t('100% of the die area, three ways to spend it', '100% 的 die 面积，三种花法')}
      onReset={() => setAlloc(CPU_PRESET)}
      wide
      footer={t(
        <>
          A toy model with unitless scores: the single-thread score applies diminishing returns to the
          control + cache share (echoing Pollack&apos;s rule, where doubling a core&apos;s complexity buys only
          about √2 more single-thread performance), while the throughput score is strictly proportional to ALU
          count. Real silicon is messier, of course, but the shape is real: one side with diminishing returns, the
          other growing linearly.
        </>,
        <>
          卡通模型，分数无量纲：单线程分按控制 + 缓存占比做收益递减（呼应 Pollack 法则，核心复杂度翻倍，
          单线程性能只涨约 √2 倍）；吞吐分严格正比 ALU 数量。真实芯片当然更复杂，但「一边收益递减、
          一边线性增长」这个结构是真的。
        </>,
      )}
    >
      <div className="grid items-start gap-x-7 gap-y-5 md:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <div>
          <ChipDie ctrl={alloc.ctrl} cache={alloc.cache} alu={alloc.alu} cols={ALU_COLS} />
          <DieLegend ctrl={display.ctrl} cache={display.cache} alu={display.alu} />
        </div>

        <div className="space-y-4">
          <div className="space-y-3">
            {SLIDER_META.map(({ key, en, zh }) => (
              <Slider
                key={key}
                label={t(en, zh)}
                value={display[key]}
                min={0}
                max={100}
                onChange={setPart(key)}
                fmt={(v) => `${v}`}
                unit="%"
              />
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="microlabel">PRESET</span>
            <Btn variant="ghost" size="sm" onClick={() => setAlloc(CPU_PRESET)}>
              {t('Typical CPU', '典型 CPU')}
            </Btn>
            <Btn variant="ghost" size="sm" onClick={() => setAlloc(GPU_PRESET)}>
              {t('Typical GPU', '典型 GPU')}
            </Btn>
          </div>

          <div className="flex flex-wrap items-end gap-x-8 gap-y-3 border-t border-line pt-4">
            <Stat label={t('Single-thread score', '单线程性能分')} value={single} unit="/100" tone="cyan" />
            <Stat label={t('Throughput score', '吞吐分')} value={thr} unit="/100" tone="volt" />
            <Stat label={t('ALU units', 'ALU 单元')} value={cells} unit={t('', '个')} size="sm" />
          </div>
        </div>
      </div>
    </Widget>
  )
}
