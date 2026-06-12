import { useState } from 'react'
import { Btn, PlayBar, Widget } from '@/components/ui'
import { useInterval, useReducedMotion } from '@/lib/hooks'

/** LAB 02 —— vecAdd 执行模拟器：N=24、blockSize=8 的玩具例，逐 block 重放执行过程 */

const N = 24
const B = 8
const NBLK = 3
// A 取 e 的数字、B 取 π 的数字 —— 一眼能核对 C = A + B
const ARR_A = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3, 6, 0, 2, 8]
const ARR_B = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4]

const CODE = [
  '__global__ void add(float* A, float* B, float* C, int n) {',
  '    int i = blockIdx.x * blockDim.x + threadIdx.x;',
  '    if (i < n) {',
  '        C[i] = A[i] + B[i];',
  '    }',
  '}',
]

// 每个 block 走 4 个子相位：算索引 → 边界检查 → 读 A/B → 写 C
const PHASES = 4
const TOTAL = NBLK * PHASES
const IDENTITY = [0, 1, 2]
const BASE_MS = 850

function randomOrder(prev: number[]): number[] {
  const perms = [
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
  ]
  const pool = perms.filter((p) => p.join() !== prev.join())
  return pool[Math.floor(Math.random() * pool.length)]
}

export function VecAddSim() {
  const [step, setStep] = useState(0) // 0..TOTAL；TOTAL = 全部完成
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [order, setOrder] = useState<number[]>(IDENTITY)
  const reduced = useReducedMotion()

  const done = step >= TOTAL
  const k = Math.min(Math.floor(step / PHASES), NBLK - 1) // 第几个执行槽
  const phase = done ? -1 : step % PHASES
  const curBlk = done ? -1 : order[k]
  const lo = curBlk * B
  const hi = curBlk * B + B - 1

  // C[i] 是否已写入：它所属 block 的写相位是否已经走完
  const cFilled = (i: number) => {
    const pos = order.indexOf(Math.floor(i / B))
    return step > pos * PHASES + 3
  }

  useInterval(
    () => {
      if (step >= TOTAL - 1) {
        setStep(TOTAL)
        setPlaying(false)
      } else {
        setStep(step + 1)
      }
    },
    playing && !done ? BASE_MS / speed : null,
  )

  const onToggle = () => {
    if (!playing && done) setStep(0)
    setPlaying(!playing)
  }
  const onStep = () => setStep(done ? 0 : step + 1)
  const reset = () => {
    setStep(0)
    setPlaying(false)
    setOrder(IDENTITY)
  }
  const shuffle = () => {
    setOrder(randomOrder(order))
    setStep(0)
    setPlaying(true)
  }

  const curLine = phase === 0 ? 2 : phase === 1 ? 3 : phase >= 2 ? 4 : -1
  const statusText = done
    ? `✓ ${N} 个元素全部算完 —— 无论 block 按什么顺序执行，C 的结果一模一样`
    : phase === 0
      ? `block ${curBlk} 的 8 个线程同时算出各自的 i = ${curBlk}×8 + (0…7) → ${lo}…${hi}`
      : phase === 1
        ? `8 个线程同时检查 i < 24 → 全部通过`
        : phase === 2
          ? `同时读取 A[${lo}…${hi}] 与 B[${lo}…${hi}]`
          : `同时写入 C[${lo}…${hi}] —— 一步完成 8 个加法`

  /* ── SVG 布局 ── */
  const cw = 29
  const gp = 2
  const ch = 27
  const x0 = 26
  const yA = 34
  const yB = yA + ch + 9
  const yC = yB + ch + 26
  const cx = (i: number) => x0 + i * (cw + gp)
  const vbW = x0 + N * (cw + gp) - gp + 6
  const vbH = yC + ch + 22
  const pulse = reduced ? '' : 'animate-pulse'

  const cell = (i: number, row: 'A' | 'B' | 'C') => {
    const inBlk = !done && Math.floor(i / B) === curBlk
    const reading = inBlk && phase === 2 && row !== 'C'
    const writing = inBlk && phase === 3 && row === 'C'
    const filled = row === 'C' ? cFilled(i) || writing : true
    const y = row === 'A' ? yA : row === 'B' ? yB : yC
    const v = row === 'A' ? ARR_A[i] : row === 'B' ? ARR_B[i] : ARR_A[i] + ARR_B[i]
    const fill = reading
      ? 'var(--color-cyan)'
      : writing
        ? 'var(--color-volt)'
        : row === 'C' && filled
          ? 'var(--color-volt)'
          : 'var(--color-ink3)'
    const fillOp = reading ? 0.32 : writing ? 0.42 : row === 'C' && filled ? 0.13 : 0.07
    return (
      <g key={`${row}${i}`} className={reading || writing ? pulse : ''}>
        <rect
          x={cx(i)}
          y={y}
          width={cw}
          height={ch}
          rx={3}
          fill={fill}
          fillOpacity={fillOp}
          stroke={reading ? 'var(--color-cyan)' : writing ? 'var(--color-volt)' : 'var(--color-line)'}
          strokeWidth={reading || writing ? 1.4 : 1}
        />
        <text
          x={cx(i) + cw / 2}
          y={y + ch / 2 + 4}
          textAnchor="middle"
          fontSize={11.5}
          fontFamily="var(--font-mono)"
          fill="currentColor"
          className={
            row === 'C' ? (filled ? (writing ? 'text-ink' : 'text-volt') : 'text-ink3') : reading ? 'text-ink' : 'text-ink2'
          }
        >
          {row === 'C' && !filled ? '·' : v}
        </text>
      </g>
    )
  }

  return (
    <Widget
      index={2}
      title="vecAdd 执行模拟器"
      subtitle="N=24，blockSize=8 → 3 个 block"
      onReset={reset}
      footer={
        <>
          点「⇄ 乱序调度」让硬件以随机顺序执行 3 个 block，再对比 C 的结果 ——{' '}
          <span className="text-ink">一模一样</span>。CUDA 不保证 block 的执行顺序，正因为每个 block 互相独立，
          硬件才能把它们随意扔给任何空闲的 SM。写代码时永远不要假设 block 间的先后关系。
        </>
      }
    >
      {/* 迷你代码面板 */}
      <div className="overflow-hidden rounded-md border border-line bg-bg2">
        <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-1.5 font-mono text-[11px]">
          <span className="text-ink3">launch:</span>
          <span className="text-cyan">
            add&lt;&lt;&lt;3, 8&gt;&gt;&gt;(A, B, C, 24)
          </span>
          <span className="ml-auto text-ink3">
            STEP <span className="text-ink tabular-nums">{Math.min(step, TOTAL)}</span>/{TOTAL}
          </span>
        </div>
        <div className="overflow-x-auto py-1.5 font-mono text-[12px] leading-[1.85]">
          {CODE.map((ln, j) => {
            const active = curLine === j + 1
            return (
              <div
                key={j}
                className={`flex whitespace-pre px-3 ${active ? 'bg-volt/[0.08] shadow-[inset_2px_0_0_var(--color-volt)]' : ''}`}
              >
                <span className="mr-3 w-3 select-none text-right text-ink3/70">{j + 1}</span>
                <span className={active ? 'text-ink' : 'text-ink2'}>{ln}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* 状态行 */}
      <div className={`mt-3 font-mono text-[12px] leading-relaxed ${done ? 'text-volt' : 'text-ink2'}`}>
        {statusText}
      </div>

      {/* A / B / C 数组格子条 */}
      <div className="mt-3 overflow-hidden rounded-md border border-line bg-bg2 p-2">
        <svg viewBox={`0 0 ${vbW} ${vbH}`} className="w-full">
          {/* 当前 block 的执行光带 */}
          {!done && (
            <rect
              x={cx(lo) - 3}
              y={yA - 16}
              width={B * (cw + gp) - gp + 6}
              height={yC + ch - yA + 20}
              rx={5}
              fill="var(--color-volt)"
              fillOpacity={0.05}
              stroke="var(--color-volt)"
              strokeOpacity={0.35}
              strokeDasharray="5 4"
            />
          )}
          {/* block 分界与标签 */}
          {[0, 1, 2].map((b) => (
            <text
              key={b}
              x={cx(b * B) + (B * (cw + gp) - gp) / 2}
              y={yA - 21}
              textAnchor="middle"
              fontSize={10.5}
              fontFamily="var(--font-mono)"
              fill="currentColor"
              className={!done && b === curBlk ? 'text-volt' : 'text-ink3'}
            >
              block {b}
              {!done && b === curBlk ? ' ◀ 执行中' : ''}
            </text>
          ))}
          {[1, 2].map((b) => (
            <line
              key={b}
              x1={cx(b * B) - gp / 2 - 1}
              y1={yA - 6}
              x2={cx(b * B) - gp / 2 - 1}
              y2={yC + ch + 4}
              stroke="var(--color-line2)"
              strokeDasharray="3 4"
            />
          ))}
          {/* 行标签 */}
          <text x={6} y={yA + ch / 2 + 4} fontSize={11} fontFamily="var(--font-mono)" fill="currentColor" className="text-cyan">
            A
          </text>
          <text x={6} y={yB + ch / 2 + 4} fontSize={11} fontFamily="var(--font-mono)" fill="currentColor" className="text-cyan">
            B
          </text>
          <text x={6} y={yC + ch / 2 + 4} fontSize={11} fontFamily="var(--font-mono)" fill="currentColor" className="text-volt">
            C
          </text>
          {/* A+B → C 提示 */}
          <text
            x={x0 - 4}
            y={yB + ch + 17}
            fontSize={9.5}
            fontFamily="var(--font-mono)"
            fill="currentColor"
            className="text-ink3"
          >
            ↓ C[i] = A[i] + B[i]
          </text>
          {Array.from({ length: N }, (_, i) => cell(i, 'A'))}
          {Array.from({ length: N }, (_, i) => cell(i, 'B'))}
          {Array.from({ length: N }, (_, i) => cell(i, 'C'))}
          {/* 索引刻度 */}
          {Array.from({ length: N }, (_, i) => (
            <text
              key={i}
              x={cx(i) + cw / 2}
              y={yC + ch + 14}
              textAnchor="middle"
              fontSize={8}
              fontFamily="var(--font-mono)"
              fill="currentColor"
              className="text-ink3"
              opacity={0.75}
            >
              {i}
            </text>
          ))}
        </svg>
      </div>

      {/* 控制条 */}
      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
        <PlayBar
          playing={playing}
          onToggle={onToggle}
          onStep={onStep}
          onReset={reset}
          speed={speed}
          onSpeed={setSpeed}
          extra={
            <Btn variant="ghost" onClick={shuffle} title="随机打乱 block 执行顺序并重放">
              ⇄ 乱序调度
            </Btn>
          }
        />
        <span className="font-mono text-[11px] text-ink3">
          执行顺序:{' '}
          <span className={order.join() === IDENTITY.join() ? 'text-ink2' : 'text-amber'}>
            block {order.join(' → block ')}
          </span>
        </span>
      </div>
    </Widget>
  )
}
