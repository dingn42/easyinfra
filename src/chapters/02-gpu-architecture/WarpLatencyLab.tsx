import { useEffect, useRef, useState } from 'react'
import { PlayBar, Slider, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { pct } from '@/lib/format'
import { useT } from '@/lib/i18n'

/* ───────────────────── LAB 02 Warp 延迟隐藏 ─────────────────────
 * 模拟一个 SM 分区的 warp 调度器：每周期从就绪 warp 里挑一个发射。
 * 每个 warp 的指令流：C 条计算指令 → 1 条访存指令 → 等待 L 个周期。
 * 色带：volt=正在执行（被发射），amber=等访存，cyan=就绪排队。
 */

const WINDOW = 360 // 时间轴可见周期数

/** 单周期状态：0=就绪排队 1=执行 2=等访存 */
type CellState = 0 | 1 | 2

interface Sim {
  cycle: number
  issued: number
  rr: number // round-robin 指针
  wait: number[] // 每 warp 剩余等待周期
  instr: number[] // 每 warp 当前迭代里已发射的计算指令数
  hist: CellState[][] // 每 warp 的状态历史（仅保留最近 WINDOW 周期）
}

function makeSim(numWarps: number): Sim {
  return {
    cycle: 0,
    issued: 0,
    rr: 0,
    wait: Array(numWarps).fill(0),
    instr: Array(numWarps).fill(0),
    hist: Array.from({ length: numWarps }, () => []),
  }
}

function stepSim(sim: Sim, latency: number, computePerMem: number) {
  const n = sim.wait.length
  // 1. 等待中的 warp 倒计时
  for (let i = 0; i < n; i++) if (sim.wait[i] > 0) sim.wait[i]--
  // 2. 调度器：round-robin 找一个就绪 warp 发射
  let pick = -1
  for (let k = 1; k <= n; k++) {
    const idx = (sim.rr + k) % n
    if (sim.wait[idx] === 0) {
      pick = idx
      break
    }
  }
  if (pick >= 0) {
    sim.rr = pick
    sim.issued++
    if (sim.instr[pick] < computePerMem) {
      sim.instr[pick]++ // 计算指令：下周期继续就绪
    } else {
      sim.instr[pick] = 0 // 访存指令：发射后进入长等待
      sim.wait[pick] = latency
    }
  }
  // 3. 记录本周期每个 warp 的状态
  for (let i = 0; i < n; i++) {
    const s: CellState = i === pick ? 1 : sim.wait[i] > 0 ? 2 : 0
    const row = sim.hist[i]
    row.push(s)
    if (row.length > WINDOW) row.shift()
  }
  sim.cycle++
}

function runCycles(sim: Sim, latency: number, computePerMem: number, count: number) {
  for (let i = 0; i < count; i++) stepSim(sim, latency, computePerMem)
}

const CHART_X0 = 34
const CHART_W = 720 - CHART_X0 - 4
const ROW_H = 13
const ROW_GAP = 4

export function WarpLatencyLab() {
  const t = useT()
  const reduced = useReducedMotion()
  const [numWarps, setNumWarps] = useState(4)
  const [latency, setLatency] = useState(200)
  const [computePerMem, setComputePerMem] = useState(2)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [, setTick] = useState(0)
  const bump = () => setTick((t) => t + 1)

  const simRef = useRef<Sim>(makeSim(4))
  const accRef = useRef(0)

  // 参数变化（含首次挂载）：重置并预跑一段，保证图表立即有信息量
  useEffect(() => {
    simRef.current = makeSim(numWarps)
    runCycles(simRef.current, latency, computePerMem, WINDOW)
    accRef.current = 0
    bump()
  }, [numWarps, latency, computePerMem])

  useRafLoop((dt) => {
    accRef.current += (dt / 1000) * 90 * speed // 90 周期/秒 × 速度
    const n = Math.floor(accRef.current)
    if (n > 0) {
      accRef.current -= n
      runCycles(simRef.current, latency, computePerMem, Math.min(n, 300))
      bump()
    }
  }, playing && !reduced)

  const onToggle = () => {
    if (reduced) {
      // 减少动画偏好：一次性推进，不做连续动画
      runCycles(simRef.current, latency, computePerMem, WINDOW)
      bump()
      return
    }
    setPlaying((p) => !p)
  }
  const onStep = () => {
    runCycles(simRef.current, latency, computePerMem, 10)
    bump()
  }
  const onReset = () => {
    setPlaying(false)
    simRef.current = makeSim(numWarps)
    accRef.current = 0
    bump()
  }

  const sim = simRef.current
  const util = sim.cycle > 0 ? sim.issued / sim.cycle : 0
  // 稳态理论值：每个 warp 每轮贡献 C+1 个发射周期，每轮总长 C+1+L
  const theory = Math.min(1, (numWarps * (computePerMem + 1)) / (computePerMem + 1 + latency))
  const readyNow = sim.wait.filter((w) => w === 0).length

  // 将历史 RLE 成色块
  const cw = CHART_W / WINDOW
  const colors = ['fill-cyan/45', 'fill-volt', 'fill-amber/55'] as const
  const svgH = 20 + numWarps * (ROW_H + ROW_GAP)

  return (
    <Widget
      index={2}
      title={t('Warp latency hiding', 'Warp 延迟隐藏')}
      subtitle={t('One SM partition · 1 scheduler · 1 warp instruction issued per cycle', '一个 SM 分区 · 1 调度器 · 每周期发射 1 条 warp 指令')}
      onReset={onReset}
      footer={t(
        <>
          Drop warps to 1: the whole pipeline stalls during memory latency (long blank stretches).
          Crank it to 16 warps with high compute density and the ALU barely catches its breath — this is
          exactly why <strong className="text-ink">occupancy</strong> matters, covered in detail in chapter 5.
        </>,
        <>
          把 warp 数拉到 1：访存延迟期间整条流水线空转（大段空白）。拉满 16 个 warp、提高计算密度，
          ALU 几乎没有喘息 —— 这就是 <strong className="text-ink">occupancy（占用率）</strong>重要的原因，第 5 章会细讲。
        </>,
      )}
    >
      <div className="mb-4 grid gap-x-6 gap-y-3 sm:grid-cols-3">
        <Slider label={t('Resident warps', '驻留 warp 数')} value={numWarps} min={1} max={16} onChange={setNumWarps} unit="" />
        <Slider label={t('Memory latency', '访存延迟')} value={latency} min={100} max={400} step={10} onChange={setLatency} unit={t('cyc', '周期')} />
        <Slider
          label={t('Compute instr. between memory ops', '每条访存之间的计算指令')}
          value={computePerMem}
          min={1}
          max={8}
          onChange={setComputePerMem}
          unit=""
        />
      </div>

      <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <PlayBar playing={playing} onToggle={onToggle} onStep={onStep} onReset={onReset} speed={speed} onSpeed={setSpeed} />
        <div className="flex items-end gap-6">
          <Stat label={t('ALU utilization', 'ALU 利用率')} value={pct(util, 1)} tone={util > 0.8 ? 'volt' : util > 0.3 ? 'cyan' : 'amber'} size="lg" />
          <Stat label={t('Steady-state theory', '稳态理论值')} value={pct(theory, 1)} tone="ink" size="sm" />
          <Stat label={t('Cycles simulated', '已模拟周期')} value={sim.cycle.toLocaleString('en-US')} tone="ink" size="sm" />
          <Stat label={t('Ready now', '当前就绪')} value={readyNow} unit="warp" tone="cyan" size="sm" />
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 720 ${svgH}`}
          className="w-full min-w-[540px] select-none"
          role="img"
          aria-label={t('Warp execution timeline', 'warp 执行时间轴')}
        >
          <text x={CHART_X0} y={11} fontSize={9} className="fill-ink3 font-mono">
            {t(`time → (last ${Math.min(sim.cycle, WINDOW)} cycles)`, `时间 →（最近 ${Math.min(sim.cycle, WINDOW)} 个周期）`)}
          </text>
          {sim.hist.map((row, wi) => {
            const y = 20 + wi * (ROW_H + ROW_GAP)
            const rects: { x0: number; x1: number; s: CellState }[] = []
            let start = 0
            for (let i = 1; i <= row.length; i++) {
              if (i === row.length || row[i] !== row[start]) {
                rects.push({ x0: start, x1: i, s: row[start] })
                start = i
              }
            }
            return (
              <g key={wi}>
                <text x={2} y={y + ROW_H - 3} fontSize={9} className="fill-ink3 font-mono">
                  W{wi}
                </text>
                <rect x={CHART_X0} y={y} width={CHART_W} height={ROW_H} className="fill-bg2" rx={2} />
                {rects.map((r, ri) => (
                  <rect
                    key={ri}
                    x={CHART_X0 + r.x0 * cw}
                    y={y}
                    width={Math.max(0.5, (r.x1 - r.x0) * cw)}
                    height={ROW_H}
                    className={colors[r.s]}
                  />
                ))}
              </g>
            )
          })}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[10.5px] text-ink2">
        <span><span className="mr-1.5 inline-block size-2.5 rounded-[2px] bg-volt align-[-1px]" />{t('executing (issued by scheduler)', '执行（被调度器发射）')}</span>
        <span><span className="mr-1.5 inline-block size-2.5 rounded-[2px] bg-amber/60 align-[-1px]" />{t('waiting on memory', '等待访存返回')}</span>
        <span><span className="mr-1.5 inline-block size-2.5 rounded-[2px] bg-cyan/50 align-[-1px]" />{t('ready, queued to issue', '就绪、排队待发射')}</span>
      </div>
    </Widget>
  )
}
