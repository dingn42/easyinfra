import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { CHAPTERS, PARTS } from '@/lib/chapters'
import { useReducedMotion } from '@/lib/hooks'
import { useVisited } from '@/lib/progress'
import { pick, useLocale, useT } from '@/lib/i18n'
import { C, rgba } from '@/lib/palette'

/** hero 背景：GPU die 网格，随机 SM 单元脉冲点亮 */
function DieGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const reduced = useReducedMotion()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const CELL = 22
    const GAP = 4
    let raf = 0
    let cells: { x: number; y: number; heat: number; hue: 'volt' | 'cyan' }[] = []
    let cols = 0
    let rows = 0

    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const { clientWidth: w, clientHeight: h } = canvas
      canvas.width = w * dpr
      canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      cols = Math.ceil(w / (CELL + GAP))
      rows = Math.ceil(h / (CELL + GAP))
      cells = []
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          cells.push({ x: c, y: r, heat: 0, hue: Math.random() < 0.82 ? 'volt' : 'cyan' })
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    const draw = () => {
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      ctx.clearRect(0, 0, w, h)
      // 点亮几个随机单元
      if (!reduced) {
        for (let k = 0; k < 3; k++) {
          const cell = cells[Math.floor(Math.random() * cells.length)]
          if (cell && Math.random() < 0.5) cell.heat = Math.min(1, cell.heat + 0.9)
        }
      }
      for (const cell of cells) {
        const px = cell.x * (CELL + GAP)
        const py = cell.y * (CELL + GAP)
        // 底格（极淡）
        ctx.strokeStyle = rgba('#2a3550', 0.05)
        ctx.strokeRect(px + 0.5, py + 0.5, CELL, CELL)
        if (cell.heat > 0.01) {
          const a = cell.heat
          const col = cell.hue === 'volt' ? C.volt : C.cyan
          ctx.fillStyle = rgba(col, 0.12 * a)
          ctx.fillRect(px, py, CELL, CELL)
          ctx.strokeStyle = rgba(col, 0.5 * a)
          ctx.strokeRect(px + 0.5, py + 0.5, CELL, CELL)
          cell.heat *= 0.962
        }
      }
      raf = requestAnimationFrame(draw)
    }
    if (reduced) {
      // 静态：随机点亮一批
      for (const c of cells) if (Math.random() < 0.06) c.heat = 0.7
      draw()
      cancelAnimationFrame(raf)
    } else {
      raf = requestAnimationFrame(draw)
    }
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [reduced])

  return <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-hidden />
}

export default function Home() {
  const visited = useVisited()
  const { lang } = useLocale()
  const t = useT()
  const continueCh = CHAPTERS.find((c) => !visited.has(c.id)) ?? CHAPTERS[0]

  useEffect(() => {
    document.title =
      lang === 'zh' ? 'EasyInfra · 从一个线程到一座 GPU 集群' : 'EasyInfra · From One Thread to a GPU Cluster'
  }, [lang])

  const methods = [
    {
      n: '01',
      t: t('Visual first', '可视化优先'),
      d: t(
        'Every abstract idea comes with a diagram you can poke at. How warps get scheduled, how accesses coalesce, how a KV cache gets paged. You see it happen instead of memorizing it.',
        '每个抽象概念都配一张能拨弄的图：warp 怎么调度、访存怎么合并、KV Cache 怎么分页。你是看它发生，而不是背下来。',
      ),
    },
    {
      n: '02',
      t: t('“Run” CUDA in the browser', '在浏览器里"跑" CUDA'),
      d: t(
        'A built-in CUDA simulator: write a kernel, configure the grid/block, single-step it, and watch every memory access. Zero setup, and errors pinpoint the exact thread.',
        '内置 CUDA 模拟器：写 kernel、配置 grid/block、单步执行、观察每一次内存访问。零环境配置，错误信息精确到线程坐标。',
      ),
    },
    {
      n: '03',
      t: t('Straight to production systems', '直达生产系统'),
      d: t(
        'The course doesn’t stop at toy kernels. FlashAttention’s online softmax, vLLM’s PagedAttention, Megatron’s tensor parallelism: these are the techniques serving frontier models in production today.',
        '课程不会停在玩具 kernel。FlashAttention 的 online softmax、vLLM 的 PagedAttention、Megatron 的张量并行，都是当下生产环境里支撑前沿模型的真实手段。',
      ),
    },
  ]

  return (
    <div>
      {/* ───────────── HERO ───────────── */}
      <section className="relative overflow-hidden border-b border-line">
        {/* full-width DieGrid strip across the very top — the page's "status display" */}
        <div className="relative h-[96px] overflow-hidden border-b border-line bg-bg2 sm:h-[120px]">
          <DieGrid />
          {/* dissolve the cells toward the bottom and the right edge */}
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_bottom,transparent_25%,color-mix(in_srgb,var(--color-bg)_88%,transparent)_100%)]" />
          <div className="pointer-events-none absolute inset-y-0 right-0 w-40 bg-[linear-gradient(to_right,transparent,var(--color-bg2))]" />
          <div className="microlabel absolute left-6 top-3.5 flex items-center gap-1.5 text-ink3/85 lg:left-8">
            <span className="inline-block size-1 animate-pulse rounded-full bg-volt" />
            SM ARRAY · LIVE
          </div>
        </div>

        <div className="bg-dots relative">
          <div className="relative mx-auto max-w-[1000px] px-6 pb-0 pt-9 lg:px-8 lg:pt-10">
            <div className="microlabel flex items-center gap-2">
              <span>INTERACTIVE COURSE</span>
              <span className="text-line2">/</span>
              <span>GPU · CUDA · LLM INFRA</span>
            </div>

            <h1 className="mt-3.5 max-w-[760px] font-display text-[28px] font-bold leading-[1.15] text-ink sm:text-[36px]">
              {t(
                <>
                  From one thread, to a <span className="text-volt">GPU cluster</span>.
                </>,
                <>
                  从一个线程，到一座 <span className="text-volt">GPU 集群</span>。
                </>,
              )}
            </h1>

            <p className="mt-4 max-w-[600px] text-[15px] leading-[1.75] text-ink2">
              {t(
                'Take a GPU apart, write CUDA, and build an inference system, all in your browser. The simulator actually runs your code, so you can work out how a frontier large language model is served without installing a thing.',
                '在浏览器里拆开 GPU、写 CUDA、搭一套推理系统。模拟器会真的跑你的代码，不用装任何环境，就能搞懂前沿大模型是怎么跑起来的。',
              )}
            </p>

            {/* instrument bar: CTAs + stat readout under a 1px rule */}
            <div className="mt-7 flex flex-col gap-4 border-t border-line py-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-wrap items-center gap-3">
                <Link
                  to={`/learn/${continueCh.id}`}
                  className="rounded-md border border-volt/45 bg-volt/10 px-5 py-2.5 font-mono text-sm font-medium tracking-wide text-volt transition-all hover:bg-volt/18 hover:glow-volt"
                >
                  {visited.size > 0 ? t('Continue learning', '继续学习') : t('Start chapter one', '开始第一章')} →
                </Link>
                <Link
                  to="/playground"
                  className="rounded-md border border-line2 bg-panel px-5 py-2.5 font-mono text-sm tracking-wide text-ink2 transition-colors hover:border-ink3 hover:text-ink"
                >
                  ▶ {t('Open the CUDA simulator', '打开 CUDA 模拟器')}
                </Link>
              </div>
              <dl className="flex flex-wrap items-baseline gap-x-7 gap-y-2 font-mono text-xs tracking-wider text-ink3">
                <div className="flex items-baseline gap-1.5">
                  <dt className="text-[15px] tabular-nums text-ink">12</dt>
                  <dd>CHAPTERS</dd>
                </div>
                <span className="hidden h-3 w-px bg-line2 sm:inline-block" aria-hidden />
                <div className="flex items-baseline gap-1.5">
                  <dt className="text-[15px] tabular-nums text-ink">30+</dt>
                  <dd>LABS</dd>
                </div>
                <span className="hidden h-3 w-px bg-line2 sm:inline-block" aria-hidden />
                <div className="flex items-baseline gap-1.5">
                  <dt className="text-[15px] tabular-nums text-volt">0</dt>
                  <dd>SETUP</dd>
                </div>
              </dl>
            </div>
          </div>
        </div>
      </section>

      {/* ───────────── 课程地图 ───────────── */}
      <section className="mx-auto max-w-[980px] px-6 py-16 lg:px-8">
        <div className="microlabel mb-2">CURRICULUM</div>
        <h2 className="font-display text-2xl font-semibold text-ink">{t('Course map', '课程地图')}</h2>
        <p className="mt-2 max-w-[640px] text-[14.5px] leading-relaxed text-ink2">
          {t(
            'Three parts, each building on the last. You start by building intuition for the hardware, move on to writing CUDA by hand, and end at the inference systems that run today’s largest models in production. Every chapter comes with experiments you can actually drive.',
            '三个部分层层递进。你先建立对硬件的直觉，再亲手写 CUDA，最后落到支撑今天最大模型的生产推理系统。每章都配了能上手拨弄的实验。',
          )}
        </p>

        <div className="mt-10 space-y-10">
          {PARTS.map((part) => (
            <div key={part.num}>
              <div className="mb-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <span className="font-display text-4xl font-bold text-line2">
                  {String(part.num).padStart(2, '0')}
                </span>
                <div>
                  <div className="font-display text-lg font-semibold text-ink">{pick(part.title, lang)}</div>
                  <div className="microlabel mt-0.5">{part.titleEn}</div>
                </div>
                <p className="w-full text-[13.5px] text-ink3 sm:ml-auto sm:w-auto sm:max-w-[340px] sm:text-right">
                  {pick(part.blurb, lang)}
                </p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {CHAPTERS.filter((c) => c.part === part.num).map((c) => (
                  <Link
                    key={c.id}
                    to={`/learn/${c.id}`}
                    className="panel group relative overflow-hidden px-5 py-4 transition-all hover:border-volt/40 hover:glow-volt"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="font-mono text-[11px] tracking-wider text-ink3">
                        CH {String(c.num).padStart(2, '0')}
                        {visited.has(c.id) && <span className="ml-2 text-volt">✓</span>}
                      </span>
                      <span className="font-mono text-[10px] uppercase tracking-widest text-ink3/80">
                        {c.minutes} min · {c.labs.length} labs
                      </span>
                    </div>
                    <div className="mt-2 text-[15.5px] font-medium text-ink transition-colors group-hover:text-volt">
                      {pick(c.title, lang)}
                    </div>
                    <div className="mt-1 text-[13px] leading-relaxed text-ink2">{pick(c.tagline, lang)}</div>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {c.labs.map((w) => (
                        <span
                          key={w.en}
                          className="rounded border border-line bg-bg2 px-1.5 py-0.5 font-mono text-[10.5px] text-ink3"
                        >
                          ⌬ {pick(w, lang)}
                        </span>
                      ))}
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ───────────── 学习方式 ───────────── */}
      <section className="border-t border-line bg-bg2/50">
        <div className="mx-auto max-w-[980px] px-6 py-16 lg:px-8">
          <div className="microlabel mb-2">METHOD</div>
          <h2 className="font-display text-2xl font-semibold text-ink">
            {t('How this course works', '这门课怎么学')}
          </h2>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {methods.map((x) => (
              <div key={x.n} className="panel px-5 py-5">
                <div className="font-display text-2xl font-bold text-volt">{x.n}</div>
                <div className="mt-2 text-[15px] font-medium text-ink">{x.t}</div>
                <p className="mt-2 text-[13.5px] leading-[1.85] text-ink2">{x.d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ───────────── 页脚 ───────────── */}
      <footer className="border-t border-line">
        <div className="mx-auto max-w-[980px] px-6 py-10 lg:px-8">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div>
              <div className="font-display text-base font-bold text-ink">
                EASY<span className="text-volt">INFRA</span>
              </div>
              <p className="mt-2 max-w-[440px] text-[12.5px] leading-relaxed text-ink3">
                {t(
                  'Everything runs in your browser. The course leans on the references below; read them when you want to go deeper.',
                  '全部内容在浏览器里运行。课程参考了下面这些经典资料，想钻得更深时值得一读。',
                )}
              </p>
            </div>
            <div className="font-mono text-[12px] leading-[2.2] text-ink3">
              <a className="block transition-colors hover:text-cyan" href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/" target="_blank" rel="noreferrer">↗ CUDA C++ Programming Guide</a>
              <a className="block transition-colors hover:text-cyan" href="https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-0" target="_blank" rel="noreferrer">↗ Programming Massively Parallel Processors</a>
              <a className="block transition-colors hover:text-cyan" href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noreferrer">↗ FlashAttention (Dao et al.)</a>
              <a className="block transition-colors hover:text-cyan" href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">↗ vLLM / PagedAttention</a>
              <a className="block transition-colors hover:text-cyan" href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noreferrer">↗ Megatron-LM</a>
            </div>
          </div>
          <div className="microlabel mt-8 border-t border-line pt-5">
            © 2026 EASYINFRA · BUILT FOR CURIOUS ENGINEERS
          </div>
        </div>
      </footer>
    </div>
  )
}
