import { useState } from 'react'
import { Segmented, Slider, Stat, Widget } from '@/components/ui'
import { fmtBytes } from '@/lib/format'

/** LAB 01 访存计数器：不同 tiling 策略下，C=A×B 要从 HBM 搬多少数据 */

type Strategy = 'naive' | 't16' | 't32' | 'ideal'

const HBM_BW = 1.9e12 // A100 HBM2e ~1.9 TB/s
const PEAK_FLOPS = 312e12 // A100 BF16 Tensor Core 312 TFLOPS

const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: 'naive', label: 'naive' },
  { value: 't16', label: 'tiled T=16' },
  { value: 't32', label: 'tiled T=32' },
  { value: 'ideal', label: '理论下限' },
]

/** 全局读取的 float 数 */
function globalReadFloats(N: number, s: Strategy): number {
  const full = 2 * N ** 3
  if (s === 'naive') return full
  if (s === 't16') return full / 16
  if (s === 't32') return full / 32
  return 2 * N * N // 理论下限：A、B 各读一遍
}

/** 全局访存总字节（读 + 写 C 一遍） */
function trafficBytes(N: number, s: Strategy): number {
  return (globalReadFloats(N, s) + N * N) * 4
}

function fmtMs(sec: number): string {
  const ms = sec * 1000
  if (ms >= 100) return ms.toFixed(0)
  if (ms >= 1) return ms.toFixed(1)
  return ms.toFixed(2)
}

export function MemCounterLab() {
  const [exp, setExp] = useState(12) // N = 2^exp, 默认 4096
  const [strategy, setStrategy] = useState<Strategy>('naive')

  const N = 2 ** exp
  const traffic = trafficBytes(N, strategy)
  const reuse = (2 * N ** 3) / globalReadFloats(N, strategy)
  const memTime = traffic / HBM_BW
  const computeTime = (2 * N ** 3) / PEAK_FLOPS
  const memBound = memTime > computeTime

  // 横向条形图：对数刻度（线性刻度下 naive 与下限差 3 个数量级，其余条会缩成 0）
  const logs = STRATEGIES.map((s) => Math.log10(trafficBytes(N, s.value)))
  const lo = Math.min(...logs) - 0.8
  const hi = Math.max(...logs)

  return (
    <Widget
      index={1}
      title="访存计数器"
      subtitle="同一个 C=A×B，不同策略要从 HBM 搬多少数据"
      onReset={() => {
        setExp(12)
        setStrategy('naive')
      }}
      footer={
        <>
          注意两件事：① N=4096 时 naive 要搬约 512 GB，而 A、B、C 本身加起来只有 192 MB ——
          同一份数据被反复从 HBM 拉了上千次；② 即使 T=32，访存时间仍远大于计算时间，
          光靠 shared memory tiling 到不了 compute-bound，这就是后面寄存器 tiling 的出场理由。
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Slider
          label="矩阵规模 N（对数滑杆）"
          value={exp}
          min={8}
          max={13}
          step={1}
          onChange={setExp}
          fmt={(v) => String(2 ** v)}
        />
        <div className="flex items-end">
          <Segmented options={STRATEGIES} value={strategy} onChange={setStrategy} block />
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="全局访存量" value={fmtBytes(traffic)} tone="amber" />
        <Stat label="复用倍数" value={`${reuse >= 100 ? reuse.toFixed(0) : reuse.toFixed(1)}×`} tone="volt" />
        <Stat label="访存时间(估)" value={fmtMs(memTime)} unit="ms" tone={memBound ? 'amber' : 'ink'} />
        <Stat label="计算时间(估)" value={fmtMs(computeTime)} unit="ms" tone={memBound ? 'ink' : 'cyan'} />
      </div>

      <div className="mt-4 flex items-center gap-2">
        <span className="microlabel">瓶颈判定</span>
        <span
          className={`rounded border px-2 py-0.5 font-mono text-[11px] tracking-wider ${
            memBound ? 'border-amber/50 bg-amber/10 text-amber' : 'border-volt/50 bg-volt/10 text-volt'
          }`}
        >
          {memBound ? 'MEMORY-BOUND' : 'COMPUTE-BOUND'}
        </span>
        <span className="text-[12px] text-ink3">
          {memBound ? '搬数据的时间盖过了算的时间' : '访存已喂得上计算，算力成为瓶颈'}
        </span>
      </div>

      <div className="mt-5">
        <div className="microlabel mb-2">四种策略的全局访存量（对数刻度，点击切换）</div>
        <div className="space-y-1.5">
          {STRATEGIES.map((s, i) => {
            const active = s.value === strategy
            const w = ((logs[i] - lo) / (hi - lo)) * 100
            return (
              <button
                key={s.value}
                onClick={() => setStrategy(s.value)}
                className="group/bar flex w-full items-center gap-2 text-left"
              >
                <span
                  className={`w-24 shrink-0 font-mono text-[11px] ${active ? 'text-volt' : 'text-ink3 group-hover/bar:text-ink2'}`}
                >
                  {s.label}
                </span>
                <span className="relative h-5 flex-1 overflow-hidden rounded-sm bg-bg2">
                  <span
                    className={`absolute inset-y-0 left-0 rounded-sm transition-all duration-300 ${
                      active ? 'bg-volt/70' : 'bg-amber/30 group-hover/bar:bg-amber/45'
                    }`}
                    style={{ width: `${w}%` }}
                  />
                </span>
                <span
                  className={`w-20 shrink-0 text-right font-mono text-[11px] tabular-nums ${active ? 'text-volt' : 'text-ink2'}`}
                >
                  {fmtBytes(trafficBytes(N, s.value), 0)}
                </span>
              </button>
            )
          })}
        </div>
      </div>
    </Widget>
  )
}
