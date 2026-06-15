import { useEffect, useState } from 'react'
import { PlayBar, Slider, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { fmtSI } from '@/lib/format'
import { useT } from '@/lib/i18n'

/**
 * LAB 01「吞吐量赛道」：
 * CPU（8 个强核，每任务 1 µs）vs GPU（2048 个弱核，每任务慢 k 倍 + 固定启动开销）
 * 处理 N 个彼此独立的任务，赛跑动画。
 */

const CPU_CORES = 8
const GPU_CORES = 2048
const TASK_US = 1 // CPU 单核处理一个任务的耗时（µs）
const LAUNCH_US = 20 // GPU kernel 启动开销（µs）
const RACE_MS = 6000 // 1× 速度下整场比赛的动画时长

const DEF_EXP = 14 // 2^14 = 16384 个任务
const DEF_SLOW = 8

function fmtTime(us: number): string {
  if (us < 1000) return `${us >= 100 ? Math.round(us) : Math.round(us * 10) / 10} µs`
  if (us < 1e6) return `${(us / 1000).toFixed(1)} ms`
  return `${(us / 1e6).toFixed(2)} s`
}

function fmtN(n: number): string {
  return n >= 100000 ? fmtSI(n, 1) : n.toLocaleString('en-US')
}

export function ThroughputRace({ index }: { index: number }) {
  const tx = useT()
  const [exp, setExp] = useState(DEF_EXP)
  const [slow, setSlow] = useState(DEF_SLOW)
  const [t, setT] = useState(0) // 模拟时钟（µs）
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const reduced = useReducedMotion()

  const n = 2 ** exp
  const cpuTime = (n / CPU_CORES) * TASK_US
  const waves = Math.ceil(n / GPU_CORES) // GPU 需要的"波次"数
  const gpuTime = LAUNCH_US + waves * slow * TASK_US
  const total = Math.max(cpuTime, gpuTime)

  const tt = Math.min(t, total)
  const cpuDone = Math.min(n, Math.floor((tt / TASK_US) * CPU_CORES))
  const gpuDone = tt <= LAUNCH_US ? 0 : Math.min(n, Math.floor((tt - LAUNCH_US) / (slow * TASK_US)) * GPU_CORES)
  const finished = tt >= total
  const speedup = cpuTime / gpuTime
  const gpuWins = gpuTime < cpuTime

  useRafLoop((dt) => {
    setT((p) => Math.min(total, p + (dt * speed * total) / RACE_MS))
  }, playing && !finished)

  // 跑到终点自动停表
  useEffect(() => {
    if (finished && playing) setPlaying(false)
  }, [finished, playing])

  const restart = () => {
    setT(0)
    setPlaying(false)
  }
  const changeExp = (v: number) => {
    setExp(v)
    restart()
  }
  const changeSlow = (v: number) => {
    setSlow(v)
    restart()
  }
  const resetAll = () => {
    setExp(DEF_EXP)
    setSlow(DEF_SLOW)
    setSpeed(1)
    restart()
  }
  const toggle = () => {
    if (reduced) {
      // 偏好减少动画：直接跳到比赛结果
      setT(finished ? 0 : total)
      setPlaying(false)
      return
    }
    if (finished) {
      setT(0)
      setPlaying(true)
      return
    }
    setPlaying((p) => !p)
  }
  const step = () => setT((p) => Math.min(total, p + total / 32))

  const tracks = [
    {
      name: 'CPU',
      sub: tx(
        `${CPU_CORES} strong cores · ${TASK_US} µs / task`,
        `${CPU_CORES} 个强核 · 每任务 ${TASK_US} µs`,
      ),
      color: 'var(--color-cyan)',
      done: cpuDone,
      time: cpuTime,
    },
    {
      name: 'GPU',
      sub: tx(
        `${fmtN(GPU_CORES)} weak cores · ${slow} µs / task · ${LAUNCH_US} µs launch`,
        `${fmtN(GPU_CORES)} 个弱核 · 每任务 ${slow} µs · 启动 ${LAUNCH_US} µs`,
      ),
      color: 'var(--color-volt)',
      done: gpuDone,
      time: gpuTime,
    },
  ]

  return (
    <Widget
      index={index}
      title={tx('Throughput Race', '吞吐量赛道')}
      subtitle={tx('8 fast cores vs. 2048 slow cores', '8 个快核 vs 2048 个慢核')}
      onReset={resetAll}
      wide
      footer={tx(
        <>
          As long as the tasks are independent, more cores always win. But once the program has a serial part, the
          speedup hits a ceiling. That&apos;s{' '}
          <span className="text-ink">Amdahl&apos;s Law</span>: with a serial fraction s, the speedup stays ≤ 1/s no
          matter how many cores you pile on. The GPU&apos;s 20 µs launch overhead is the other half of why the CPU
          wins when there are too few tasks.
        </>,
        <>
          只要任务彼此独立，核多就是正义；可一旦程序有串行部分，加速比就有了天花板，这就是{' '}
          <span className="text-ink">Amdahl 定律</span>：串行占比 s 时，无论堆多少核，加速比都 ≤ 1/s。GPU 那
          20 µs 启动开销，正是「任务太少时 CPU 反而赢」的另一半原因。
        </>,
      )}
    >
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
        <Slider
          label={tx('Task count N (log steps)', '任务数 N（log 步进）')}
          value={exp}
          min={6}
          max={20}
          onChange={changeExp}
          fmt={(e) => fmtN(2 ** e)}
          unit={tx('', '个')}
        />
        <Slider
          label={tx('GPU per-core slowdown', 'GPU 单核慢倍数')}
          value={slow}
          min={4}
          max={32}
          onChange={changeSlow}
          fmt={(v) => `${v}×`}
        />
      </div>

      <div className="mt-5 space-y-4">
        {tracks.map((tr) => {
          const frac = n > 0 ? tr.done / n : 0
          const trackFinished = tr.done >= n
          return (
            <div key={tr.name}>
              <div className="mb-1 flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                <span className="font-mono text-xs font-semibold" style={{ color: tr.color }}>
                  {tr.name}
                </span>
                <span className="text-[11px] text-ink3">{tr.sub}</span>
                <span className="ml-auto font-mono text-[11px] tabular-nums text-ink2">
                  {fmtN(tr.done)} / {fmtN(n)}
                </span>
              </div>
              <div className="relative h-6 overflow-hidden rounded border border-line bg-bg2">
                <div
                  className="absolute inset-y-0 left-0"
                  style={{ width: `${frac * 100}%`, background: tr.color, opacity: 0.3 }}
                />
                <div
                  className="absolute inset-y-0 w-px"
                  style={{ left: `${frac * 100}%`, background: tr.color, opacity: frac > 0 && frac < 1 ? 0.9 : 0 }}
                />
                {trackFinished && (
                  <span
                    className="absolute right-2 top-1/2 -translate-y-1/2 font-mono text-[11px]"
                    style={{ color: tr.color }}
                  >
                    ✓ FINISH @ {fmtTime(tr.time)}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-5">
        <PlayBar
          playing={playing}
          onToggle={toggle}
          onStep={step}
          onReset={restart}
          speed={speed}
          onSpeed={setSpeed}
        />
      </div>

      <div className="mt-5 flex flex-wrap items-end gap-x-8 gap-y-4 border-t border-line pt-4">
        <Stat label={tx('CPU finish time', 'CPU 完成时间')} value={fmtTime(cpuTime)} tone="cyan" />
        <Stat label={tx('GPU finish time', 'GPU 完成时间')} value={fmtTime(gpuTime)} tone="volt" />
        <Stat
          label={tx('GPU speedup', 'GPU 加速比')}
          value={`${speedup >= 10 ? speedup.toFixed(0) : speedup.toFixed(speedup >= 1 ? 1 : 2)}×`}
          tone={gpuWins ? 'volt' : 'rose'}
        />
        <Stat label={tx('Sim clock', '模拟时钟')} value={fmtTime(tt)} size="sm" />
        <div className="microlabel" style={{ color: gpuWins ? 'var(--color-volt)' : 'var(--color-cyan)' }}>
          {tx('Predicted winner: ', '预测胜者：')}
          {gpuWins ? 'GPU' : 'CPU'}
          {!gpuWins && speedup > 0.99 && speedup < 1.01 ? tx(' (near tie)', '（几乎平手）') : ''}
        </div>
      </div>
    </Widget>
  )
}
