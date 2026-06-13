import { useEffect, useMemo, useRef, useState } from 'react'
import { Segmented } from '@/components/ui'
import { useLocale, useT, pick } from '@/lib/i18n'
import { compile, run } from '@/lib/cudasim'
import type {
  AccessEvent,
  BufferDecl,
  CompileError,
  KernelInfo,
  LaunchConfig,
  RuntimeErrorInfo,
  SimStats,
} from '@/lib/cudasim'
import { useLocalStorage } from '@/lib/hooks'
import { CudaEditor } from './playground/CudaEditor'
import { SyntaxPanel } from './playground/SyntaxPanel'
import { ConfigPanel, declsToRows, rowsToDecls, type BufferRow } from './playground/ConfigPanel'
import { BuffersView } from './playground/BuffersView'
import { AccessMap } from './playground/AccessMap'
import { StatsView } from './playground/StatsView'
import { Challenges } from './playground/Challenges'
import {
  CHALLENGES,
  EXAMPLES,
  materializeInitial,
  type ChallengeDef,
  type ExampleDef,
} from './playground/examples'

/* ──────────────────────── 类型 ──────────────────────── */

interface RunView {
  ok: boolean
  buffers: Record<string, number[]> | null
  initial: Record<string, number[]> | null
  stats: SimStats | null
  accesses: AccessEvent[]
  truncated: boolean
  order: string[]
  ms: number
}

interface RunArgs {
  source: string
  gridX: number
  blockX: number
  rows: BufferRow[]
  scalars: Record<string, string>
  challenge: ChallengeDef | null
}

type Verdict = { pass: boolean; msg: string }

/** 当前语言的二选一 picker（与 useT 同形，但可在渲染外调用） */
type Tr = <T>(en: T, zh: T) => T

/* ──────────────────────── 挑战判定 ──────────────────────── */

function judgeChallenge(
  ch: ChallengeDef,
  cfg: LaunchConfig,
  decls: BufferDecl[],
  scalarArgs: Record<string, number>,
  userBuffers: Record<string, number[]>,
  userStats: SimStats,
  t: Tr,
): Verdict {
  const rc = compile(ch.referenceSource)
  if (!rc.ok)
    return { pass: false, msg: t('Reference implementation failed to compile (this should not happen).', '参考实现编译失败（这不应该发生）。') }
  const rr = run(rc.kernel, cfg, decls, { scalarArgs, recordAccesses: false })
  if (!rr.ok) {
    return {
      pass: false,
      msg: t(
        `The reference implementation can't run under this config either (${rr.error.message}). Hit "Reload" to restore the original config and try again.`,
        `参考实现在当前配置下也无法运行（${rr.error.message}）。点「重新载入」还原配置后再试。`,
      ),
    }
  }
  for (const name of ch.compareBuffers) {
    const a = userBuffers[name]
    const b = rr.buffers[name]
    if (!a || !b || a.length !== b.length) {
      return {
        pass: false,
        msg: t(
          `Output buffer ${name} is missing, or its length differs from the reference.`,
          `输出缓冲区 ${name} 缺失或长度与参考不一致。`,
        ),
      }
    }
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > 1e-6) {
        return {
          pass: false,
          msg: t(
            `${name}[${i}] = ${+a[i].toFixed(4)}, expected ${+b[i].toFixed(4)} — the output isn't right yet, keep thinking.`,
            `${name}[${i}] = ${+a[i].toFixed(4)}，期望 ${+b[i].toFixed(4)} —— 输出还不对，再想想。`,
          ),
        }
      }
    }
  }
  if (ch.maxGlobalTransactions != null && userStats.globalTransactions > ch.maxGlobalTransactions) {
    return {
      pass: false,
      msg: t(
        `Output is correct, but GLOBAL TXN = ${userStats.globalTransactions} still exceeds the limit of ${ch.maxGlobalTransactions} — the access isn't coalesced enough.`,
        `输出正确，但 GLOBAL TXN = ${userStats.globalTransactions}，仍超过上限 ${ch.maxGlobalTransactions} —— 访存还不够合并。`,
      ),
    }
  }
  return {
    pass: true,
    msg: ch.maxGlobalTransactions != null
      ? t(
          'Output matches the reference element-for-element, and the transaction count meets the target.',
          '输出与参考实现逐元素一致，且事务数达标。',
        )
      : t('Output matches the reference implementation element-for-element.', '输出与参考实现逐元素一致。'),
  }
}

/* ──────────────────────── 页面 ──────────────────────── */

const FIRST = EXAMPLES[0]

export default function PlaygroundPage() {
  const t = useT()
  const { lang } = useLocale()
  const [active, setActive] = useState<{ kind: 'example' | 'challenge'; id: string }>({
    kind: 'example',
    id: FIRST.id,
  })
  const [source, setSource] = useState(FIRST.source)
  const [gridX, setGridX] = useState(FIRST.grid)
  const [blockX, setBlockX] = useState(FIRST.block)
  const [bufRows, setBufRows] = useState<BufferRow[]>(() => declsToRows(FIRST.buffers))
  const [scalars, setScalars] = useState<Record<string, string>>(() =>
    Object.fromEntries(Object.entries(FIRST.scalars).map(([k, v]) => [k, String(v)])),
  )

  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<RunView | null>(null)
  const [compileErr, setCompileErr] = useState<CompileError | null>(null)
  const [runtimeErr, setRuntimeErr] = useState<RuntimeErrorInfo | null>(null)
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [tab, setTab] = useState<'buffers' | 'access' | 'stats'>('buffers')
  const [passed, setPassed] = useLocalStorage<Record<string, boolean>>('ei-pg-passed', {})

  useEffect(() => {
    document.title = t('Playground · CUDA Simulator · EasyInfra', 'Playground · CUDA 模拟器 · EasyInfra')
    return () => {
      document.title = t('EasyInfra · From One Thread to a GPU Cluster', 'EasyInfra · 从一个线程到一座 GPU 集群')
    }
  }, [t])

  /* —— 实时编译：驱动标量输入框与 buffers 表自动跟随源码 —— */
  const liveInfo = useMemo<KernelInfo | null>(() => {
    const c = compile(source)
    return c.ok ? c.kernel.info : null
  }, [source])
  const lastInfoRef = useRef<KernelInfo | null>(null)
  if (liveInfo) lastInfoRef.current = liveInfo
  const info = liveInfo ?? lastInfoRef.current

  useEffect(() => {
    if (!liveInfo) return
    const ptr = liveInfo.params.filter((p) => p.isPointer)
    setBufRows((prev) => {
      const next = ptr.map(
        (p) => prev.find((r) => r.name === p.name) ?? { name: p.name, length: 256, init: 'zero' as const },
      )
      return next.length === prev.length && next.every((r, i) => r === prev[i]) ? prev : next
    })
    const sc = liveInfo.params.filter((p) => !p.isPointer)
    setScalars((prev) => {
      if (sc.length === Object.keys(prev).length && sc.every((p) => p.name in prev)) return prev
      return Object.fromEntries(sc.map((p) => [p.name, prev[p.name] ?? '0']))
    })
  }, [liveInfo])

  /* —— 执行 —— */
  const exec = (args: RunArgs) => {
    setRunning(true)
    setCompileErr(null)
    setRuntimeErr(null)
    setVerdict(null)
    window.setTimeout(() => {
      const t0 = performance.now()
      try {
        const c = compile(args.source)
        if (!c.ok) {
          setCompileErr(c.error)
          setResult(null)
          return
        }
        const decls = rowsToDecls(args.rows)
        const scalarArgs: Record<string, number> = {}
        for (const p of c.kernel.info.params) {
          if (!p.isPointer) {
            const n = Number(args.scalars[p.name])
            scalarArgs[p.name] = Number.isFinite(n) ? n : 0
          }
        }
        const cfg: LaunchConfig = {
          grid: { x: args.gridX, y: 1, z: 1 },
          block: { x: args.blockX, y: 1, z: 1 },
        }
        const r = run(c.kernel, cfg, decls, { scalarArgs, maxTraceEvents: 120_000 })
        const initial = materializeInitial(decls, c.kernel.info.params)
        const order = decls.map((d) => d.name)
        const ms = performance.now() - t0
        if (!r.ok) {
          setRuntimeErr(r.error)
          setResult({
            ok: false,
            buffers: null,
            initial,
            stats: r.stats ?? null,
            accesses: r.accesses ?? [],
            truncated: false,
            order,
            ms,
          })
          if (args.challenge)
            setVerdict({ pass: false, msg: t(`Runtime error: ${r.error.message}`, `运行时错误：${r.error.message}`) })
          return
        }
        setResult({
          ok: true,
          buffers: r.buffers,
          initial,
          stats: r.stats,
          accesses: r.accesses,
          truncated: r.traceTruncated,
          order,
          ms,
        })
        if (args.challenge) {
          const v = judgeChallenge(args.challenge, cfg, decls, scalarArgs, r.buffers, r.stats, t)
          setVerdict(v)
          if (v.pass) {
            const id = args.challenge.id
            setPassed((prev) => ({ ...prev, [id]: true }))
          }
        }
      } catch (err) {
        setRuntimeErr({
          kind: 'runtime',
          message: t(
            `Simulator internal error: ${err instanceof Error ? err.message : String(err)}`,
            `模拟器内部异常：${err instanceof Error ? err.message : String(err)}`,
          ),
        })
        setResult(null)
      } finally {
        setRunning(false)
      }
    }, 30)
  }

  /* —— 载入示例 / 挑战（切换即载入并自动运行一次） —— */
  const loadItem = (kind: 'example' | 'challenge', def: ExampleDef | ChallengeDef) => {
    setActive({ kind, id: def.id })
    setSource(def.source)
    setGridX(def.grid)
    setBlockX(def.block)
    const rows = declsToRows(def.buffers)
    setBufRows(rows)
    const sc = Object.fromEntries(Object.entries(def.scalars).map(([k, v]) => [k, String(v)]))
    setScalars(sc)
    exec({
      source: def.source,
      gridX: def.grid,
      blockX: def.block,
      rows,
      scalars: sc,
      challenge: kind === 'challenge' ? (def as ChallengeDef) : null,
    })
  }

  /* —— 首屏自动运行 vecAdd —— */
  const didInit = useRef(false)
  useEffect(() => {
    if (didInit.current) return
    didInit.current = true
    exec({
      source: FIRST.source,
      gridX: FIRST.grid,
      blockX: FIRST.block,
      rows: declsToRows(FIRST.buffers),
      scalars: Object.fromEntries(Object.entries(FIRST.scalars).map(([k, v]) => [k, String(v)])),
      challenge: null,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const activeExample = active.kind === 'example' ? EXAMPLES.find((e) => e.id === active.id) : undefined
  const activeChallenge = active.kind === 'challenge' ? CHALLENGES.find((c) => c.id === active.id) : undefined
  const scalarParams = info?.params.filter((p) => !p.isPointer) ?? []
  const overLimit = blockX > 1024 || gridX * blockX > 65536

  const runNow = () =>
    exec({ source, gridX, blockX, rows: bufRows, scalars, challenge: activeChallenge ?? null })

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-12 lg:px-8">
      {/* ───── 页头 ───── */}
      <header>
        <div className="microlabel mb-3 flex items-center gap-2">
          <span className="inline-block size-1.5 rounded-full bg-volt" />
          PLAYGROUND · IN-BROWSER SIMULATOR
        </div>
        <h1 className="font-display text-3xl font-bold text-ink sm:text-4xl">
          {t('CUDA Simulator', 'CUDA 模拟器')}
        </h1>
        <p className="mt-3 max-w-[680px] text-[15px] leading-relaxed text-ink2">
          {t(
            'Write a kernel in the browser, set your grid / block, and hit RUN — the simulator executes your code thread by thread, logs every memory access, and tells you how well it coalesced, whether banks collided, and whether the warp split apart. Error messages pin down the exact thread coordinate.',
            '在浏览器里写 kernel、配置 grid / block、按 RUN —— 模拟器逐线程执行你的代码，记录每一次内存访问，告诉你访存合并得怎么样、bank 撞没撞、warp 有没有分家。错误信息精确到线程坐标。',
          )}
        </p>
      </header>

      <div className="mt-6">
        <SyntaxPanel />
      </div>

      {/* ───── 示例选择 ───── */}
      <section className="mt-8">
        <div className="microlabel mb-3">EXAMPLES</div>
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6">
          {EXAMPLES.map((ex) => {
            const isActive = active.kind === 'example' && active.id === ex.id
            return (
              <button
                key={ex.id}
                onClick={() => loadItem('example', ex)}
                className={`panel group px-3 py-2.5 text-left transition-all hover:border-volt/40 ${
                  isActive ? 'border-volt/60 bg-volt/[0.06] shadow-[0_0_20px_rgba(28,138,63,0.1)]' : ''
                }`}
              >
                <div className={`font-mono text-[12.5px] ${isActive ? 'text-volt' : 'text-ink group-hover:text-volt'}`}>
                  {pick(ex.title, lang)}
                </div>
                <div className="microlabel mt-1 !text-[9.5px]">{pick(ex.badge, lang)}</div>
                <div className="mt-1.5 hidden text-[11px] leading-snug text-ink3 sm:block">{pick(ex.blurb, lang)}</div>
              </button>
            )
          })}
        </div>
        {(activeExample || activeChallenge) && (
          <div className="mt-3 flex gap-2.5 rounded-md border border-cyan/25 bg-cyan/[0.05] px-3.5 py-2.5">
            <span className="shrink-0 font-mono text-[11px] tracking-widest text-cyan">{t('⌖ OBSERVE', '⌖ 观察')}</span>
            <p className="text-[12.5px] leading-relaxed text-ink2">
              {activeExample
                ? pick(activeExample.hint, lang)
                : `${pick(activeChallenge!.desc, lang)} ${pick(activeChallenge!.goal, lang)}`}
            </p>
          </div>
        )}
      </section>

      {/* ───── 双栏：编辑器+配置 | 结果 ───── */}
      <div className="mt-6 grid items-start gap-6 lg:grid-cols-[minmax(0,29fr)_minmax(0,27fr)]">
        {/* —— 左：编辑器 + 配置 + RUN —— */}
        <div className="min-w-0 space-y-4">
          <div>
            <div className="mb-2 flex items-baseline gap-3">
              <span className="microlabel">KERNEL SOURCE</span>
              {activeChallenge && (
                <span className="font-mono text-[11px] tracking-wider text-violet">
                  ⌬ CHALLENGE {String(activeChallenge.num).padStart(2, '0')} · {pick(activeChallenge.title, lang)}
                </span>
              )}
              {info && (
                <span className="ml-auto font-mono text-[11px] text-ink3">
                  {liveInfo ? '✓' : '…'} {info.name}()
                </span>
              )}
            </div>
            <CudaEditor
              value={source}
              onChange={(v) => {
                setSource(v)
                setCompileErr(null)
                setRuntimeErr(null)
              }}
              errorLine={compileErr?.line ?? runtimeErr?.line ?? null}
            />
          </div>

          {compileErr && (
            <div className="flex items-baseline gap-3 rounded-md border border-rose/40 bg-rose/10 px-3.5 py-2.5">
              <span className="shrink-0 font-mono text-[11px] tracking-widest text-rose">✗ COMPILE</span>
              <p className="font-mono text-[12.5px] leading-relaxed text-rose">
                L{compileErr.line}:{compileErr.col} — {compileErr.message}
              </p>
            </div>
          )}

          {runtimeErr && (
            <div className="rounded-md border border-rose/40 bg-rose/10 px-3.5 py-3">
              <div className="flex items-center gap-3">
                <span className="font-mono text-[11px] tracking-widest text-rose">✗ RUNTIME ERROR</span>
                {runtimeErr.line != null && (
                  <span className="font-mono text-[11px] text-rose/80">@ L{runtimeErr.line}</span>
                )}
              </div>
              <p className="mt-1.5 font-mono text-[12.5px] leading-relaxed text-rose">{runtimeErr.message}</p>
              {runtimeErr.thread && (
                <p className="mt-1 font-mono text-[11.5px] text-rose/80">
                  {t('offending thread: ', '触发线程：')}block({runtimeErr.thread.block.x},{runtimeErr.thread.block.y},
                  {runtimeErr.thread.block.z}) thread({runtimeErr.thread.thread.x},{runtimeErr.thread.thread.y},
                  {runtimeErr.thread.thread.z})
                </p>
              )}
            </div>
          )}

          <ConfigPanel
            gridX={gridX}
            blockX={blockX}
            onGrid={setGridX}
            onBlock={setBlockX}
            scalarParams={scalarParams}
            scalars={scalars}
            onScalar={(name, v) => setScalars((prev) => ({ ...prev, [name]: v }))}
            rows={bufRows}
            onRows={setBufRows}
            disabled={running}
          />

          <button
            onClick={runNow}
            disabled={running || overLimit}
            className="w-full rounded-md border border-volt/50 bg-volt/15 px-6 py-3.5 font-mono text-sm font-semibold tracking-[0.2em] text-volt transition-all hover:bg-volt/25 hover:shadow-[0_0_28px_rgba(28,138,63,0.25)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? '⌬ RUNNING…' : '▶ RUN'}
          </button>

          {verdict && (
            <div
              className={`flex items-start gap-3 rounded-md border px-3.5 py-3 ${
                verdict.pass ? 'border-volt/50 bg-volt/10' : 'border-amber/40 bg-amber/10'
              }`}
            >
              <span
                className={`shrink-0 rounded border px-1.5 py-0.5 font-mono text-[10px] tracking-widest ${
                  verdict.pass ? 'border-volt/50 text-volt' : 'border-amber/50 text-amber'
                }`}
              >
                {verdict.pass ? '✓ PASSED' : '✗ NOT YET'}
              </span>
              <p className={`text-[12.5px] leading-relaxed ${verdict.pass ? 'text-ink2' : 'text-ink2'}`}>
                {verdict.msg}
              </p>
            </div>
          )}
        </div>

        {/* —— 右：结果 —— */}
        <div className="panel min-w-0 overflow-hidden">
          <div className="flex flex-wrap items-center gap-3 border-b border-line bg-panel2/60 px-4 py-2.5">
            <span className="microlabel text-volt">⌬ RESULTS</span>
            <Segmented
              options={[
                { value: 'buffers' as const, label: 'BUFFERS' },
                { value: 'access' as const, label: 'ACCESS MAP' },
                { value: 'stats' as const, label: 'STATS' },
              ]}
              value={tab}
              onChange={setTab}
            />
            {result && (
              <span className="ml-auto font-mono text-[10.5px] tabular-nums text-ink3">
                <span className={result.ok ? 'text-volt' : 'text-rose'}>●</span> {result.ms.toFixed(1)} ms
              </span>
            )}
          </div>
          <div className="p-4 sm:p-5">
            {!result && !compileErr && (
              <div className="flex flex-col items-center gap-2 py-16 text-center">
                <span className="microlabel animate-pulse text-volt">⌬ WARMING UP…</span>
              </div>
            )}
            {!result && compileErr && (
              <div className="flex flex-col items-center gap-2 py-16 text-center">
                <span className="microlabel text-rose">COMPILE FAILED</span>
                <p className="text-[13px] text-ink3">
                  {t('Fix the compile error on the left, then try again.', '修好左边的编译错误，再来一次。')}
                </p>
              </div>
            )}
            {result && tab === 'buffers' && (
              <BuffersView buffers={result.buffers} initial={result.initial} order={result.order} />
            )}
            {result && tab === 'access' && (
              <AccessMap accesses={result.accesses} stats={result.stats} truncated={result.truncated} />
            )}
            {result && tab === 'stats' && <StatsView stats={result.stats} />}
          </div>
        </div>
      </div>

      {/* ───── 挑战 ───── */}
      <Challenges
        activeId={activeChallenge?.id ?? null}
        passed={passed}
        onLoad={(ch) => loadItem('challenge', ch)}
      />
    </div>
  )
}
