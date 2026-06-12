import { useRef, useState } from 'react'
import { PlayBar, Segmented, Stat, Widget } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { pct } from '@/lib/format'

/* ───────────────────────── 常量与伪随机请求流 ───────────────────────── */

const COLS = 16
const ROWS = 12
const NB = COLS * ROWS // 192 个 block
const BT = 16 // 每 block 容纳 16 token
const CAP = NB * BT // 3072 个 token 槽位
const SEED = 20240613

const PALETTE = [
  '#59d8ea', '#b8f53d', '#ffb454', '#a18aff', '#ff5c7a',
  '#34d399', '#f0abfc', '#facc15', '#93c5fd', '#fb923c',
]

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface Arrival {
  id: number
  tick: number
  prompt: number
  maxLen: number // 用户申报的 max_tokens 上限
  actual: number // 实际会生成到的长度（提前结束）
  color: string
}

function genArrivals(seed: number, n: number): Arrival[] {
  const rnd = mulberry32(seed)
  let tick = 0
  const out: Arrival[] = []
  for (let i = 0; i < n; i++) {
    tick += 2 + Math.floor(rnd() * 9)
    const prompt = 8 + Math.floor(rnd() * 112)
    const budget = 96 + Math.floor(rnd() * 416)
    const maxLen = prompt + budget
    const actual = prompt + Math.max(8, Math.floor(budget * (0.1 + rnd() * 0.75)))
    out.push({ id: i, tick, prompt, maxLen, actual, color: PALETTE[i % PALETTE.length] })
  }
  return out
}

const ARRIVALS = genArrivals(SEED, 1500)

/* ───────────────────────── 分配器模拟 ───────────────────────── */

interface Req {
  id: number
  color: string
  prompt: number
  maxLen: number
  actual: number
  cur: number // 当前已存 token 数
  base: number // contig：起始物理块
  nblocks: number // contig：预留块数
  blocks: number[] // paged：逻辑块 → 物理块
}

interface Alloc {
  contig: boolean
  owner: number[] // 每个物理块的归属请求 id，-1 = 空闲
  reqs: Map<number, Req>
  queue: Arrival[]
  nextArrival: number
  finished: number
  preempted: number
}

function newAlloc(contig: boolean): Alloc {
  return { contig, owner: new Array<number>(NB).fill(-1), reqs: new Map(), queue: [], nextArrival: 0, finished: 0, preempted: 0 }
}

/** paged 准入水位线：留一排块的余量给在跑请求的增长，模拟 vLLM 的 watermark */
const WATERMARK = 16

/** 找 need 个连续空闲块（first-fit），找不到返回 -1 */
function findRun(owner: number[], need: number): number {
  let run = 0
  for (let i = 0; i < NB; i++) {
    run = owner[i] === -1 ? run + 1 : 0
    if (run >= need) return i - need + 1
  }
  return -1
}

function freeBlocks(owner: number[]): number[] {
  const out: number[] = []
  for (let i = 0; i < NB; i++) if (owner[i] === -1) out.push(i)
  return out
}

function mkReq(a: Arrival): Req {
  return { id: a.id, color: a.color, prompt: a.prompt, maxLen: a.maxLen, actual: a.actual, cur: a.prompt, base: -1, nblocks: 0, blocks: [] }
}

function tickAlloc(a: Alloc, tick: number) {
  // 1) 已有请求各 decode 一个 token；到达 actual 即完成并释放
  for (const r of Array.from(a.reqs.values())) {
    if (!a.reqs.has(r.id)) continue // 可能已被本 tick 内的抢占赶回队列
    if (r.cur < r.actual) {
      if (a.contig) {
        r.cur++
      } else if (r.cur < r.blocks.length * BT) {
        r.cur++
      } else {
        // 需要新块：空闲池随便哪一块都行
        let free = freeBlocks(a.owner)
        if (free.length === 0) {
          // 池子全满：抢占最晚到达的其他请求（vLLM 的 recompute 式抢占），防止集体停摆
          let victim: Req | null = null
          for (const o of a.reqs.values()) {
            if (o.id !== r.id && (victim === null || o.id > victim.id)) victim = o
          }
          if (victim) {
            for (const b of victim.blocks) a.owner[b] = -1
            a.reqs.delete(victim.id)
            a.preempted++
            a.queue.unshift({
              id: victim.id, tick: 0, prompt: victim.prompt,
              maxLen: victim.maxLen, actual: victim.actual, color: victim.color,
            })
            free = freeBlocks(a.owner)
          }
        }
        if (free.length > 0) {
          const b = free[0]
          a.owner[b] = r.id
          r.blocks.push(b)
          r.cur++
        }
      }
    }
    if (r.cur >= r.actual) {
      if (a.contig) {
        for (let i = 0; i < r.nblocks; i++) a.owner[r.base + i] = -1
      } else {
        for (const b of r.blocks) a.owner[b] = -1
      }
      a.reqs.delete(r.id)
      a.finished++
    }
  }
  // 2) 新请求到达，进入等待队列
  while (a.nextArrival < ARRIVALS.length && ARRIVALS[a.nextArrival].tick <= tick) {
    a.queue.push(ARRIVALS[a.nextArrival++])
  }
  // 3) FIFO 准入
  while (a.queue.length > 0) {
    const arr = a.queue[0]
    if (a.contig) {
      // 连续预分配：按 max_len 一次性圈下连续区间
      const need = Math.ceil(arr.maxLen / BT)
      const base = findRun(a.owner, need)
      if (base < 0) break
      for (let i = 0; i < need; i++) a.owner[base + i] = arr.id
      const r = mkReq(arr)
      r.base = base
      r.nblocks = need
      a.reqs.set(arr.id, r)
    } else {
      // 分页：只为 prompt 申请块，之后按需追加（留出水位线余量，避免立刻又触发抢占）
      const need = Math.ceil(arr.prompt / BT)
      const free = freeBlocks(a.owner)
      if (free.length < need + WATERMARK) break
      const r = mkReq(arr)
      r.blocks = free.slice(0, need)
      for (const b of r.blocks) a.owner[b] = arr.id
      a.reqs.set(arr.id, r)
    }
    a.queue.shift()
  }
}

function allocStats(a: Alloc) {
  let used = 0
  let alloc = 0
  for (const r of a.reqs.values()) {
    used += r.cur
    alloc += (a.contig ? r.nblocks : r.blocks.length) * BT
  }
  return {
    util: used / CAP,
    waste: (alloc - used) / CAP,
    n: a.reqs.size,
    queue: a.queue.length,
    finished: a.finished,
    preempted: a.preempted,
  }
}

interface Sim {
  tick: number
  contig: Alloc
  paged: Alloc
}

function initSim(warmup: number): Sim {
  const s: Sim = { tick: 0, contig: newAlloc(true), paged: newAlloc(false) }
  for (let i = 0; i < warmup; i++) {
    s.tick++
    tickAlloc(s.contig, s.tick)
    tickAlloc(s.paged, s.tick)
  }
  return s
}

/* ───────────────────────── 组件 ───────────────────────── */

const TICK_MS = 110
const WARMUP = 320

/** LAB 03 PagedAttention 模拟器（wide）：同一请求流喂给两种分配器 */
export default function PagedLab() {
  const reduced = useReducedMotion()
  const [view, setView] = useState<'contig' | 'paged'>('paged')
  const [playing, setPlaying] = useState(!reduced)
  const [speed, setSpeed] = useState(1)
  const [selected, setSelected] = useState<number | null>(null)
  const [, setFrame] = useState(0)
  const sim = useRef<Sim | null>(null)
  if (sim.current === null) sim.current = initSim(WARMUP)
  const acc = useRef(0)

  const advance = (n: number) => {
    const s = sim.current!
    for (let i = 0; i < n; i++) {
      s.tick++
      tickAlloc(s.contig, s.tick)
      tickAlloc(s.paged, s.tick)
    }
    setFrame((f) => f + 1)
  }

  useRafLoop((dt) => {
    acc.current += dt * speed
    let n = 0
    while (acc.current >= TICK_MS && n < 8) {
      acc.current -= TICK_MS
      n++
    }
    if (acc.current >= TICK_MS) acc.current = 0
    if (n > 0) advance(n)
  }, playing)

  const reset = () => {
    sim.current = initSim(WARMUP)
    acc.current = 0
    setSelected(null)
    setFrame((f) => f + 1)
  }

  const s = sim.current
  const shown = view === 'contig' ? s.contig : s.paged
  const cStats = allocStats(s.contig)
  const pStats = allocStats(s.paged)
  const selReq = selected != null ? shown.reqs.get(selected) : undefined

  /* ── 显存池网格 ── */
  const P = 22
  const SZ = 20
  const gridW = COLS * P + 2
  const gridH = ROWS * P + 2

  const blocks = []
  for (let b = 0; b < NB; b++) {
    const x = (b % COLS) * P + 1
    const y = Math.floor(b / COLS) * P + 1
    const id = shown.owner[b]
    if (id === -1) {
      blocks.push(
        <rect key={b} x={x} y={y} width={SZ} height={SZ} rx={2.5}
          fill="var(--color-bg2)" stroke="var(--color-line)" strokeWidth={1} />,
      )
      continue
    }
    const r = shown.reqs.get(id)
    if (!r) continue
    // 该物理块在请求中的逻辑序号
    const logical = shown.contig ? b - r.base : r.blocks.indexOf(b)
    const tokensHere = Math.max(0, Math.min(BT, r.cur - logical * BT))
    const fillH = (tokensHere / BT) * SZ
    const isSel = selected === id
    const dim = selected != null && !isSel
    blocks.push(
      <g key={b} opacity={dim ? 0.22 : 1} onClick={() => setSelected(isSel ? null : id)} style={{ cursor: 'pointer' }}>
        <rect x={x} y={y} width={SZ} height={SZ} rx={2.5}
          fill={r.color} fillOpacity={0.1}
          stroke={isSel ? 'var(--color-volt)' : r.color}
          strokeOpacity={isSel ? 1 : 0.55} strokeWidth={isSel ? 1.8 : 1} />
        {tokensHere > 0 && (
          <rect x={x} y={y + SZ - fillH} width={SZ} height={fillH} rx={1.5}
            fill={r.color} fillOpacity={0.5} pointerEvents="none" />
        )}
        {tokensHere < BT && (
          <rect x={x} y={y} width={SZ} height={SZ - fillH} rx={1.5}
            fill="url(#ei-kv-hatch)" pointerEvents="none" />
        )}
      </g>,
    )
  }

  /* ── block table（选中请求的逻辑块 → 物理块映射） ── */
  let table = null
  if (selReq) {
    const phys = shown.contig
      ? Array.from({ length: selReq.nblocks }, (_, i) => selReq.base + i)
      : selReq.blocks
    const MAXR = 14
    const rows = Math.min(phys.length, MAXR)
    const rowH = 19
    const tH = Math.max(rows * rowH + 16, 120)
    const lines = []
    for (let i = 0; i < rows; i++) {
      const yL = 10 + i * rowH
      const yR = 8 + (phys[i] / NB) * (tH - 16)
      lines.push(
        <g key={i}>
          <line x1={56} y1={yL + 5} x2={138} y2={yR + 5} stroke={selReq.color} strokeOpacity={0.55} strokeWidth={1.2} />
          <rect x={8} y={yL - 4} width={48} height={17} rx={3} fill="var(--color-bg2)" stroke="var(--color-line2)" />
          <text x={32} y={yL + 8} fontSize={9.5} textAnchor="middle" className="font-mono" fill="var(--color-ink2)">L{i}</text>
          <rect x={138} y={yR - 4} width={56} height={17} rx={3} fill="var(--color-bg2)" stroke={selReq.color} strokeOpacity={0.6} />
          <text x={166} y={yR + 8} fontSize={9.5} textAnchor="middle" className="font-mono" fill="var(--color-ink)">P{phys[i]}</text>
        </g>,
      )
    }
    table = (
      <div className="rounded-md border border-line bg-bg2/60 p-3">
        <div className="microlabel mb-1.5" style={{ color: selReq.color }}>
          REQ #{selReq.id} BLOCK TABLE
        </div>
        <div className="mb-2 font-mono text-[10.5px] leading-relaxed tabular-nums text-ink3">
          prompt {selReq.prompt} · 已生成 {selReq.cur}/{selReq.actual} tok
          {shown.contig
            ? <> · 预留 {selReq.nblocks} 块（max_len {selReq.maxLen}）</>
            : <> · 持有 {selReq.blocks.length} 块，按需追加</>}
        </div>
        <svg viewBox={`0 0 202 ${tH}`} className="w-full">
          <text x={8} y={2} fontSize={8} className="font-mono" fill="var(--color-ink3)" dominantBaseline="hanging">逻辑块</text>
          <text x={194} y={2} fontSize={8} textAnchor="end" className="font-mono" fill="var(--color-ink3)" dominantBaseline="hanging">物理块（按池内位置）</text>
          {lines}
        </svg>
        {phys.length > MAXR && (
          <div className="mt-1 font-mono text-[10px] text-ink3">… 还有 {phys.length - MAXR} 个块未画出</div>
        )}
        <div className="mt-2 text-[11.5px] leading-relaxed text-ink3">
          {shown.contig
            ? '物理上连续：寻址 = 基址 + 偏移，根本不需要表 —— 代价是整段区间被一次性锁死。'
            : '物理块东一块西一块，靠这张表把逻辑位置翻译成物理地址 —— 和操作系统的页表一模一样。'}
        </div>
      </div>
    )
  }

  const statCard = (label: string, st: ReturnType<typeof allocStats>, key: 'contig' | 'paged') => (
    <button
      onClick={() => { setView(key); setSelected(null) }}
      className={`flex-1 rounded-md border px-3.5 py-2.5 text-left transition-colors ${
        view === key ? 'border-volt/50 bg-volt/[0.05]' : 'border-line bg-bg2/40 hover:border-line2'
      }`}
    >
      <div className={`microlabel mb-2 ${view === key ? 'text-volt' : ''}`}>{label}</div>
      <div className="flex flex-wrap gap-x-5 gap-y-1.5">
        <Stat label="并发" value={st.n} size="sm" tone={key === 'paged' ? 'volt' : 'ink'} />
        <Stat label="利用率" value={pct(st.util)} size="sm" tone="cyan" />
        <Stat label="浪费" value={pct(st.waste)} size="sm" tone={st.waste > 0.2 ? 'rose' : 'ink'} />
        <Stat label="排队" value={st.queue} size="sm" />
        <Stat label="已完成" value={st.finished} size="sm" />
        {key === 'paged' && <Stat label="抢占" value={st.preempted} size="sm" />}
      </div>
    </button>
  )

  return (
    <Widget
      index={3}
      title="PagedAttention 模拟器"
      subtitle="同一条请求流 · 两种显存分配器"
      onReset={reset}
      wide
      footer={
        <>
          斜线 = 已分配却没有 token 的浪费。真实 vLLM 还有两件本模拟没画的武器：相同 prompt 前缀的请求共享同一批物理块
          （prefix sharing），写时复制（copy-on-write）保证分叉后才各存一份 —— 相同前缀永远只占一份显存。
        </>
      }
    >
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-3">
        <PlayBar playing={playing} onToggle={() => setPlaying(!playing)} onStep={() => advance(1)}
          onReset={reset} speed={speed} onSpeed={setSpeed}
          extra={
            <Segmented
              options={[
                { value: 'contig', label: '连续预分配' },
                { value: 'paged', label: 'PAGED' },
              ]}
              value={view}
              onChange={(v) => { setView(v); setSelected(null) }}
            />
          }
        />
        <span className="ml-auto font-mono text-[11px] tabular-nums text-ink3">tick {s.tick}</span>
      </div>

      <div className="mb-4 flex flex-col gap-3 sm:flex-row">
        {statCard('连续预分配（按 MAX_LEN 圈地）', cStats, 'contig')}
        {statCard('PAGEDATTENTION（按需给块）', pStats, 'paged')}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px]">
        <div>
          <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
            <span className="microlabel">
              显存池 · {NB} BLOCK × {BT} TOKEN（{view === 'contig' ? '连续预分配' : 'PAGED'}视图）
            </span>
            <span className="font-mono text-[10.5px] text-ink3">点击色块查看该请求的 block table</span>
          </div>
          <svg viewBox={`0 0 ${gridW} ${gridH}`} className="w-full rounded-md border border-line bg-bg">
            <defs>
              <pattern id="ei-kv-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="5" stroke="var(--color-rose)" strokeOpacity="0.4" strokeWidth="1.4" />
              </pattern>
            </defs>
            <rect x={0} y={0} width={gridW} height={gridH} fill="transparent" onClick={() => setSelected(null)} />
            {blocks}
          </svg>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10.5px] text-ink3">
            <span><span className="mr-1 inline-block size-2 rounded-sm border border-line bg-bg2 align-middle" />空闲</span>
            <span><span className="mr-1 inline-block size-2 rounded-sm bg-cyan/50 align-middle" />已存 KV token</span>
            <span><span className="mr-1 inline-block size-2 rounded-sm align-middle" style={{ background: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,92,122,0.5) 2px, rgba(255,92,122,0.5) 3px)' }} />已分配未使用（浪费）</span>
          </div>
        </div>
        <div>
          {table ?? (
            <div className="flex h-full min-h-[140px] items-center justify-center rounded-md border border-dashed border-line p-4 text-center text-[12px] leading-relaxed text-ink3">
              点击左侧任意请求的色块，<br />查看它的逻辑块 → 物理块映射
            </div>
          )}
        </div>
      </div>
    </Widget>
  )
}
