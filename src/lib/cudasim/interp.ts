/**
 * CUDA C 子集 —— 执行器
 *
 * 执行模型：
 *  - 逐 block 模拟；block 内每个线程是一个 generator 协程。
 *  - 每条语句开头 / 每次循环回边 yield 一个 'step'（逻辑步），调度器据此把
 *    block 内所有存活线程按"逻辑步"同拍推进（lockstep）。
 *  - 每次内存访问 yield 'mem'，每个分支条件 yield 'branch'，__syncthreads yield 'barrier'。
 *  - 聚合规则（见 types.ts 注释）：
 *      globalTransactions：同 warp 同 step 的全局访问按 32 字节段（每元素 4 字节，
 *        即每 8 个元素一段）聚合，事务数 = 触及的不同段数。
 *      bankConflicts：同 warp 同 step 的 shared 访问按 bank = index % 32 聚合，
 *        每组 conflict = max(同 bank 不同地址数) - 1，累计。
 *      divergentBranches：同 warp 同 step 同分支点的条件取值不一致 → +1。
 */
import type {
  AccessEvent,
  BufferDecl,
  Dim3,
  LaunchConfig,
  RunOptions,
  SimOutcome,
  SimStats,
  ThreadCoord,
} from './types'
import type { Expr, IndexExpr, KernelAst, ScalarType, Stmt, VarExpr } from './parser'

// ---------- 内部事件 ----------
type Ev =
  | { t: 'step'; line: number }
  | { t: 'mem'; space: 'global' | 'shared'; buffer: string; index: number; kind: 'read' | 'write'; line: number }
  | { t: 'branch'; site: number; value: boolean }
  | { t: 'barrier'; line: number }

/** 核函数内部抛出的运行时错误（由调度器补充线程坐标） */
class KError extends Error {
  readonly line?: number
  constructor(msg: string, line?: number) {
    super(msg)
    this.name = 'KError'
    this.line = line
  }
}

/** 带线程坐标的运行时错误（调度层） */
class ThreadError extends Error {
  readonly line?: number
  readonly coord: ThreadCoord
  constructor(msg: string, coord: ThreadCoord, line?: number) {
    super(msg)
    this.name = 'ThreadError'
    this.coord = coord
    this.line = line
  }
}

interface ArrayBuf {
  data: number[]
  elem: ScalarType
  len: number
}

interface RunCtx {
  kernel: KernelAst
  grid: Dim3
  block: Dim3
  global: Map<string, ArrayBuf>
  shared: Map<string, ArrayBuf> // 每个 block 重新分配
}

interface TCtx {
  coord: ThreadCoord
  scopes: Map<string, number>[]
}

export function fmtThread(c: ThreadCoord): string {
  return `block(${c.block.x},${c.block.y},${c.block.z}) thread(${c.thread.x},${c.thread.y},${c.thread.z})`
}

// ---------- 变量读写 ----------
function lookupVar(T: TCtx, name: string): number {
  for (let i = T.scopes.length - 1; i >= 0; i--) {
    const m = T.scopes[i]
    const v = m.get(name)
    if (v !== undefined) return v
  }
  throw new KError(`内部错误：变量 '${name}' 不存在`)
}

function setVar(T: TCtx, name: string, v: number): void {
  for (let i = T.scopes.length - 1; i >= 0; i--) {
    const m = T.scopes[i]
    if (m.has(name)) {
      m.set(name, v)
      return
    }
  }
  throw new KError(`内部错误：变量 '${name}' 不存在`)
}

function getArr(R: RunCtx, e: IndexExpr): ArrayBuf {
  const arr = (e.space === 'shared' ? R.shared : R.global).get(e.name)
  if (!arr) throw new KError(`内部错误：缓冲区 '${e.name}' 不存在`, e.line)
  return arr
}

function checkBounds(arr: ArrayBuf, name: string, idx: number, T: TCtx, line: number): void {
  if (!Number.isInteger(idx) || idx < 0 || idx >= arr.len) {
    throw new KError(`数组越界：${name}[${idx}]（长度 ${arr.len}），线程 ${fmtThread(T.coord)}`, line)
  }
}

// ---------- 表达式求值（generator：内存访问 yield 事件） ----------
function* ev(e: Expr, T: TCtx, R: RunCtx): Generator<Ev, number, void> {
  switch (e.k) {
    case 'num':
      return e.v
    case 'var':
      if (e.name === 'warpSize') return 32
      return lookupVar(T, e.name)
    case 'dim': {
      const d: Dim3 =
        e.base === 'threadIdx'
          ? T.coord.thread
          : e.base === 'blockIdx'
            ? T.coord.block
            : e.base === 'blockDim'
              ? R.block
              : R.grid
      return d[e.field]
    }
    case 'index': {
      const arr = getArr(R, e)
      const idx = yield* ev(e.idx, T, R)
      checkBounds(arr, e.name, idx, T, e.line)
      yield { t: 'mem', space: e.space, buffer: e.name, index: idx, kind: 'read', line: e.line }
      return arr.data[idx]
    }
    case 'un': {
      const v = yield* ev(e.e, T, R)
      return e.op === '-' ? -v : v === 0 ? 1 : 0
    }
    case 'bin': {
      const l = yield* ev(e.l, T, R)
      const r = yield* ev(e.r, T, R)
      switch (e.op) {
        case '+':
          return l + r
        case '-':
          return l - r
        case '*':
          return l * r
        case '/':
          if (!e.f) {
            // 两侧都是 int → C 语义截断除法
            if (r === 0) throw new KError('整数除法除数为 0', e.line)
            return Math.trunc(l / r)
          }
          return l / r
        case '%':
          if (r === 0) throw new KError('整数取模除数为 0', e.line)
          return l % r // JS % 与 C 的截断取余语义一致
        case '==':
          return l === r ? 1 : 0
        case '!=':
          return l !== r ? 1 : 0
        case '<':
          return l < r ? 1 : 0
        case '<=':
          return l <= r ? 1 : 0
        case '>':
          return l > r ? 1 : 0
        case '>=':
          return l >= r ? 1 : 0
      }
      break
    }
    case 'logic': {
      const l = yield* ev(e.l, T, R)
      if (e.op === '&&') {
        if (l === 0) return 0
        const r = yield* ev(e.r, T, R)
        return r !== 0 ? 1 : 0
      }
      if (l !== 0) return 1
      const r = yield* ev(e.r, T, R)
      return r !== 0 ? 1 : 0
    }
    case 'cond': {
      const c = yield* ev(e.c, T, R)
      yield { t: 'branch', site: e.site, value: c !== 0 }
      return c !== 0 ? yield* ev(e.t, T, R) : yield* ev(e.e, T, R)
    }
    case 'call': {
      const a: number[] = []
      for (const arg of e.args) a.push(yield* ev(arg, T, R))
      switch (e.name) {
        case 'min':
          return Math.min(a[0], a[1])
        case 'max':
          return Math.max(a[0], a[1])
        case 'abs':
        case 'fabsf':
          return Math.abs(a[0])
        case 'sqrtf':
          return Math.sqrt(a[0])
        case 'expf':
          return Math.exp(a[0])
        case 'logf':
          return Math.log(a[0])
        case 'powf':
          return Math.pow(a[0], a[1])
        case 'floorf':
          return Math.floor(a[0])
        case 'sinf':
          return Math.sin(a[0])
        case 'cosf':
          return Math.cos(a[0])
        default:
          throw new KError(`内部错误：未知函数 '${e.name}'`, e.line)
      }
    }
  }
  /* istanbul ignore next */
  throw new KError('内部错误：未知表达式节点')
}

// ---------- 语句执行 ----------
// 控制流结果：0 正常 / 1 break / 2 continue / 3 return
type Flow = 0 | 1 | 2 | 3

function* exBlock(body: Stmt[], T: TCtx, R: RunCtx): Generator<Ev, Flow, void> {
  T.scopes.push(new Map())
  let fl: Flow = 0
  for (const s of body) {
    fl = yield* ex(s, T, R)
    if (fl !== 0) break
  }
  T.scopes.pop()
  return fl
}

function* ex(s: Stmt, T: TCtx, R: RunCtx): Generator<Ev, Flow, void> {
  yield { t: 'step', line: s.line }
  switch (s.k) {
    case 'decl': {
      const cur = T.scopes[T.scopes.length - 1]
      for (const d of s.decls) {
        let v = d.init ? yield* ev(d.init, T, R) : 0
        if (s.type === 'int') v = Math.trunc(v)
        cur.set(d.name, v)
      }
      return 0
    }
    case 'assign': {
      const rhs = yield* ev(s.value, T, R)
      let v: number
      if (s.op === '=') {
        v = rhs
      } else {
        let cur: number
        if (s.target.k === 'var') {
          cur = lookupVar(T, s.target.name)
        } else {
          const arr = getArr(R, s.target)
          const idx = yield* ev(s.target.idx, T, R)
          checkBounds(arr, s.target.name, idx, T, s.target.line)
          yield { t: 'mem', space: s.target.space, buffer: s.target.name, index: idx, kind: 'read', line: s.line }
          cur = arr.data[idx]
          // 复合赋值的写入复用同一下标
          v = applyCompound(cur, rhs, s.op, s)
          if (!s.target.f) v = Math.trunc(v)
          yield { t: 'mem', space: s.target.space, buffer: s.target.name, index: idx, kind: 'write', line: s.line }
          arr.data[idx] = v
          return 0
        }
        v = applyCompound(cur, rhs, s.op, s)
      }
      if (!s.target.f) v = Math.trunc(v)
      if (s.target.k === 'var') {
        setVar(T, s.target.name, v)
      } else {
        const arr = getArr(R, s.target)
        const idx = yield* ev(s.target.idx, T, R)
        checkBounds(arr, s.target.name, idx, T, s.target.line)
        yield { t: 'mem', space: s.target.space, buffer: s.target.name, index: idx, kind: 'write', line: s.line }
        arr.data[idx] = v
      }
      return 0
    }
    case 'incdec': {
      const v = lookupVar(T, s.name) + (s.op === '++' ? 1 : -1)
      setVar(T, s.name, v)
      return 0
    }
    case 'if': {
      const c = yield* ev(s.c, T, R)
      yield { t: 'branch', site: s.site, value: c !== 0 }
      if (c !== 0) return yield* exBlock(s.t, T, R)
      if (s.e) return yield* exBlock(s.e, T, R)
      return 0
    }
    case 'while': {
      for (;;) {
        const c = yield* ev(s.c, T, R)
        yield { t: 'branch', site: s.site, value: c !== 0 }
        if (c === 0) return 0
        const fl = yield* exBlock(s.body, T, R)
        if (fl === 1) return 0
        if (fl === 3) return 3
        yield { t: 'step', line: s.line } // 循环回边
      }
    }
    case 'for': {
      T.scopes.push(new Map())
      if (s.init) yield* ex(s.init, T, R)
      for (;;) {
        if (s.c) {
          const c = yield* ev(s.c, T, R)
          yield { t: 'branch', site: s.site, value: c !== 0 }
          if (c === 0) break
        }
        const fl = yield* exBlock(s.body, T, R)
        if (fl === 1) break
        if (fl === 3) {
          T.scopes.pop()
          return 3
        }
        if (s.update) yield* ex(s.update, T, R)
        yield { t: 'step', line: s.line } // 循环回边
      }
      T.scopes.pop()
      return 0
    }
    case 'blk':
      return yield* exBlock(s.body, T, R)
    case 'break':
      return 1
    case 'continue':
      return 2
    case 'return':
      return 3
    case 'sync':
      yield { t: 'barrier', line: s.line }
      return 0
  }
}

function applyCompound(cur: number, rhs: number, op: '+=' | '-=' | '*=' | '/=', s: AssignLike): number {
  switch (op) {
    case '+=':
      return cur + rhs
    case '-=':
      return cur - rhs
    case '*=':
      return cur * rhs
    case '/=':
      if (rhs === 0 && !s.target.f && !s.value.f) throw new KError('整数除法除数为 0', s.line)
      return cur / rhs
  }
}
interface AssignLike {
  target: VarExpr | IndexExpr
  value: Expr
  line: number
}

function* threadMain(T: TCtx, R: RunCtx): Generator<Ev, void, void> {
  T.scopes.push(new Map())
  for (const s of R.kernel.body) {
    const fl = yield* ex(s, T, R)
    if (fl === 3) return
  }
}

// ---------- 固定种子 LCG（seed = 42，可复现） ----------
function makeLcg(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
}

// ============================================================
// run：调度全部 block / 线程，聚合统计
// ============================================================
export function runKernel(
  kernel: KernelAst,
  config: LaunchConfig,
  bufferDecls: BufferDecl[],
  opts?: RunOptions,
): SimOutcome {
  const maxStepsPerThread = opts?.maxStepsPerThread ?? 100_000
  const record = opts?.recordAccesses ?? true
  const maxTrace = opts?.maxTraceEvents ?? 200_000

  const stats: SimStats = {
    threads: 0,
    warps: 0,
    maxSteps: 0,
    globalReads: 0,
    globalWrites: 0,
    globalTransactions: 0,
    sharedReads: 0,
    sharedWrites: 0,
    bankConflicts: 0,
    syncBarriers: 0,
    divergentBranches: 0,
  }
  const accesses: AccessEvent[] = []
  let truncated = false

  const fail = (message: string, line?: number, thread?: ThreadCoord): SimOutcome => ({
    ok: false,
    error: { kind: 'runtime', message, line, thread },
    stats,
    accesses,
  })

  // ---- 启动配置校验 ----
  const { grid, block } = config
  const dims = [grid.x, grid.y, grid.z, block.x, block.y, block.z]
  if (dims.some((d) => !Number.isInteger(d) || d < 1)) {
    return fail('启动配置非法：grid/block 各维度必须是 ≥1 的整数')
  }
  const blockThreads = block.x * block.y * block.z
  if (blockThreads > 1024) return fail(`block 线程数 ${blockThreads} 超过上限 1024`)
  const gridBlocks = grid.x * grid.y * grid.z
  const totalThreads = blockThreads * gridBlocks
  if (totalThreads > 65536) return fail(`grid 总线程数 ${totalThreads} 超过上限 65536`)

  // ---- 缓冲区初始化 ----
  const elemOfParam = new Map<string, ScalarType>()
  for (const p of kernel.params) if (p.isPointer) elemOfParam.set(p.name, p.elem)
  const lcg = makeLcg(42)
  const global = new Map<string, ArrayBuf>()
  for (const d of bufferDecls) {
    if (global.has(d.name)) return fail(`缓冲区 '${d.name}' 重复声明`)
    if (!Number.isInteger(d.length) || d.length < 0) return fail(`缓冲区 '${d.name}' 长度非法：${d.length}`)
    const elem = elemOfParam.get(d.name) ?? 'float'
    const data = new Array<number>(d.length)
    const init = d.init ?? 'zero'
    if (init === 'zero') data.fill(0)
    else if (init === 'iota') for (let i = 0; i < d.length; i++) data[i] = i
    else if (init === 'random') for (let i = 0; i < d.length; i++) data[i] = lcg()
    else for (let i = 0; i < d.length; i++) data[i] = init[i] ?? 0
    if (elem === 'int') for (let i = 0; i < d.length; i++) data[i] = Math.trunc(data[i])
    global.set(d.name, { data, elem, len: d.length })
  }
  for (const p of kernel.params) {
    if (p.isPointer && !global.has(p.name)) {
      return fail(`指针参数 '${p.name}' 缺少同名缓冲区（请在 buffers 中声明 { name: '${p.name}', ... }）`)
    }
  }
  const scalars = new Map<string, number>()
  for (const p of kernel.params) {
    if (p.isPointer) continue
    const v = opts?.scalarArgs?.[p.name]
    if (v === undefined) return fail(`标量参数 '${p.name}' 未提供（请通过 opts.scalarArgs 传入）`)
    scalars.set(p.name, p.elem === 'int' ? Math.trunc(v) : v)
  }

  const warpsPerBlock = Math.ceil(blockThreads / 32)
  stats.threads = totalThreads
  stats.warps = warpsPerBlock * gridBlocks

  // ---- warp 级同拍聚合器（按 block 结算）----
  const globalAgg = new Map<string, Set<number>>() // key → 32B 段集合
  const sharedAgg = new Map<string, Map<number, Set<number>>>() // key → bank → 地址集合
  const branchAgg = new Map<string, number>() // key → bit1: 见过 true, bit2: 见过 false
  const finalizeBlock = () => {
    for (const segs of globalAgg.values()) stats.globalTransactions += segs.size
    for (const banks of sharedAgg.values()) {
      let m = 1
      for (const set of banks.values()) if (set.size > m) m = set.size
      stats.bankConflicts += m - 1
    }
    for (const bits of branchAgg.values()) if (bits === 3) stats.divergentBranches++
    globalAgg.clear()
    sharedAgg.clear()
    branchAgg.clear()
  }

  interface Th {
    gen: Generator<Ev, void, void>
    step: number
    done: boolean
    waiting: boolean
    coord: ThreadCoord
    warpId: number
    laneId: number
    line: number
  }

  /** 推进一个线程到下一个逻辑步 / 屏障 / 结束；中途处理 mem & branch 事件 */
  const advanceOne = (th: Th): void => {
    for (;;) {
      let r: IteratorResult<Ev, void>
      try {
        r = th.gen.next()
      } catch (err) {
        if (err instanceof KError) throw new ThreadError(err.message, th.coord, err.line ?? th.line)
        throw err
      }
      if (r.done) {
        th.done = true
        if (th.step > stats.maxSteps) stats.maxSteps = th.step
        return
      }
      const e = r.value
      if (e.t === 'step') {
        th.step++
        th.line = e.line
        if (th.step > maxStepsPerThread) {
          throw new ThreadError(
            `线程逻辑步数超过 ${maxStepsPerThread}，疑似死循环，线程 ${fmtThread(th.coord)}`,
            th.coord,
            e.line,
          )
        }
        if (th.step > stats.maxSteps) stats.maxSteps = th.step
        return
      }
      if (e.t === 'mem') {
        if (e.space === 'global') {
          if (e.kind === 'read') stats.globalReads++
          else stats.globalWrites++
          const key = `${th.warpId}|${th.step}|${e.kind}|${e.buffer}`
          let segs = globalAgg.get(key)
          if (!segs) globalAgg.set(key, (segs = new Set()))
          segs.add(e.index >> 3) // 4 字节/元素 → 每 8 元素一个 32 字节段
        } else {
          if (e.kind === 'read') stats.sharedReads++
          else stats.sharedWrites++
          const key = `${th.warpId}|${th.step}|${e.kind}|${e.buffer}`
          let banks = sharedAgg.get(key)
          if (!banks) sharedAgg.set(key, (banks = new Map()))
          const bank = e.index % 32
          let set = banks.get(bank)
          if (!set) banks.set(bank, (set = new Set()))
          set.add(e.index)
        }
        if (record) {
          if (accesses.length < maxTrace) {
            accesses.push({
              step: th.step,
              warpId: th.warpId,
              laneId: th.laneId,
              thread: th.coord,
              space: e.space,
              buffer: e.buffer,
              index: e.index,
              kind: e.kind,
              line: e.line,
            })
          } else {
            truncated = true
          }
        }
        continue
      }
      if (e.t === 'branch') {
        const key = `${th.warpId}|${e.site}|${th.step}`
        branchAgg.set(key, (branchAgg.get(key) ?? 0) | (e.value ? 1 : 2))
        continue
      }
      // barrier
      th.waiting = true
      th.line = e.line
      return
    }
  }

  // ---- 逐 block 调度 ----
  for (let bz = 0; bz < grid.z; bz++) {
    for (let by = 0; by < grid.y; by++) {
      for (let bx = 0; bx < grid.x; bx++) {
        const blockLin = bx + by * grid.x + bz * grid.x * grid.y
        const shared = new Map<string, ArrayBuf>()
        for (const sa of kernel.shared) {
          shared.set(sa.name, { data: new Array<number>(sa.length).fill(0), elem: sa.elem, len: sa.length })
        }
        const R: RunCtx = { kernel, grid, block, global, shared }
        const threads: Th[] = []
        for (let tz = 0; tz < block.z; tz++) {
          for (let ty = 0; ty < block.y; ty++) {
            for (let tx = 0; tx < block.x; tx++) {
              const lane = tx + ty * block.x + tz * block.x * block.y
              const coord: ThreadCoord = { block: { x: bx, y: by, z: bz }, thread: { x: tx, y: ty, z: tz } }
              const T: TCtx = { coord, scopes: [new Map(scalars)] }
              threads.push({
                gen: threadMain(T, R),
                step: 0,
                done: false,
                waiting: false,
                coord,
                warpId: blockLin * warpsPerBlock + (lane >> 5),
                laneId: lane & 31,
                line: 0,
              })
            }
          }
        }

        try {
          for (;;) {
            let advanced = 0
            for (const th of threads) {
              if (th.done || th.waiting) continue
              advanced++
              advanceOne(th)
            }
            if (advanced === 0) {
              const waiting = threads.filter((t) => t.waiting)
              if (waiting.length === 0) break // 全部结束
              if (threads.some((t) => t.done)) {
                const w = waiting[0]
                throw new ThreadError(
                  `__syncthreads 屏障分化：部分线程提前退出，其余线程（如 ${fmtThread(w.coord)}）在屏障处永远等待`,
                  w.coord,
                  w.line,
                )
              }
              for (const t of waiting) t.waiting = false
              stats.syncBarriers++
            }
          }
        } catch (err) {
          finalizeBlock()
          if (err instanceof ThreadError) return fail(err.message, err.line, err.coord)
          throw err
        }
        finalizeBlock()
      }
    }
  }

  const buffersOut: Record<string, number[]> = {}
  for (const [name, b] of global) buffersOut[name] = b.data
  return { ok: true, buffers: buffersOut, stats, accesses, traceTruncated: truncated }
}
