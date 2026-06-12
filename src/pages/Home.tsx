import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { CHAPTERS, PARTS } from '@/lib/chapters'
import { useReducedMotion } from '@/lib/hooks'
import { useVisited } from '@/lib/progress'

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
        // 底格
        ctx.strokeStyle = 'rgba(150,170,210,0.055)'
        ctx.strokeRect(px + 0.5, py + 0.5, CELL, CELL)
        if (cell.heat > 0.01) {
          const a = cell.heat
          ctx.fillStyle =
            cell.hue === 'volt' ? `rgba(184,245,61,${0.16 * a})` : `rgba(89,216,234,${0.16 * a})`
          ctx.fillRect(px, py, CELL, CELL)
          ctx.strokeStyle =
            cell.hue === 'volt' ? `rgba(184,245,61,${0.5 * a})` : `rgba(89,216,234,${0.5 * a})`
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
  const continueCh = CHAPTERS.find((c) => !visited.has(c.id)) ?? CHAPTERS[0]

  return (
    <div>
      {/* ───────────── HERO ───────────── */}
      <section className="relative overflow-hidden border-b border-line">
        <DieGrid />
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_0%,var(--color-bg)_78%)]" />
        <div className="relative mx-auto max-w-[980px] px-6 pb-20 pt-24 lg:px-8 lg:pt-32">
          <div className="microlabel mb-6 flex items-center gap-2">
            <span className="inline-block size-1.5 animate-pulse rounded-full bg-volt" />
            INTERACTIVE COURSE · GPU / CUDA / LLM INFRA
          </div>
          <h1 className="font-display text-[40px] font-bold leading-[1.15] text-ink sm:text-[56px]">
            从一个线程，
            <br />
            到一座 <span className="text-glow text-volt">GPU 集群</span>。
          </h1>
          <p className="mt-6 max-w-[560px] text-[16px] leading-[1.9] text-ink2">
            在浏览器里拆开 GPU、写 CUDA、搭推理系统。12 章内容、30+
            个可交互实验、一个能跑代码的模拟器 —— 不装任何环境，把现代大模型基础设施的底层逻辑一次讲透。
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-3">
            <Link
              to={`/learn/${continueCh.id}`}
              className="rounded-md border border-volt/50 bg-volt/15 px-6 py-3 font-mono text-sm font-medium tracking-wide text-volt transition-all hover:bg-volt/25 hover:shadow-[0_0_28px_rgba(184,245,61,0.3)]"
            >
              {visited.size > 0 ? '继续学习' : '开始第一章'} →
            </Link>
            <Link
              to="/playground"
              className="rounded-md border border-line2 px-6 py-3 font-mono text-sm tracking-wide text-ink2 transition-colors hover:border-ink3 hover:text-ink"
            >
              ▶ 打开 CUDA 模拟器
            </Link>
          </div>
          <div className="mt-14 flex flex-wrap gap-x-10 gap-y-3 font-mono text-xs tracking-wider text-ink3">
            <span><span className="text-ink">12</span> CHAPTERS</span>
            <span><span className="text-ink">30+</span> INTERACTIVE LABS</span>
            <span><span className="text-ink">0</span> SETUP REQUIRED</span>
            <span><span className="text-volt">∞</span> CURIOSITY</span>
          </div>
        </div>
      </section>

      {/* ───────────── 课程地图 ───────────── */}
      <section className="mx-auto max-w-[980px] px-6 py-16 lg:px-8">
        <div className="microlabel mb-2">CURRICULUM</div>
        <h2 className="font-display text-2xl font-semibold text-ink">课程地图</h2>
        <p className="mt-2 max-w-[620px] text-[14.5px] leading-relaxed text-ink2">
          三个部分层层递进：先建立对硬件的直觉，再亲手写 CUDA，最后抵达生产级大模型推理系统。
          每章都配有可以动手拨弄的实验。
        </p>

        <div className="mt-10 space-y-10">
          {PARTS.map((part) => (
            <div key={part.num}>
              <div className="mb-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <span className="font-display text-4xl font-bold text-line2">
                  {String(part.num).padStart(2, '0')}
                </span>
                <div>
                  <div className="font-display text-lg font-semibold text-ink">{part.title}</div>
                  <div className="microlabel mt-0.5">{part.titleEn}</div>
                </div>
                <p className="w-full text-[13.5px] text-ink3 sm:ml-auto sm:w-auto sm:max-w-[340px] sm:text-right">
                  {part.blurb}
                </p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {CHAPTERS.filter((c) => c.part === part.num).map((c) => (
                  <Link
                    key={c.id}
                    to={`/learn/${c.id}`}
                    className="panel group relative overflow-hidden px-5 py-4 transition-all hover:border-volt/40 hover:shadow-[0_0_28px_rgba(184,245,61,0.07)]"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="font-mono text-[11px] tracking-wider text-ink3">
                        CH {String(c.num).padStart(2, '0')}
                        {visited.has(c.id) && <span className="ml-2 text-volt">✓</span>}
                      </span>
                      <span className="font-mono text-[10px] uppercase tracking-widest text-ink3/70">
                        {c.minutes} min · {c.labs.length} labs
                      </span>
                    </div>
                    <div className="mt-2 text-[15.5px] font-medium text-ink transition-colors group-hover:text-volt">
                      {c.title}
                    </div>
                    <div className="mt-1 text-[13px] leading-relaxed text-ink2">{c.tagline}</div>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {c.labs.map((w) => (
                        <span
                          key={w}
                          className="rounded border border-line bg-bg2 px-1.5 py-0.5 font-mono text-[10.5px] text-ink3"
                        >
                          ⌬ {w}
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
      <section className="border-t border-line bg-bg2/40">
        <div className="mx-auto max-w-[980px] px-6 py-16 lg:px-8">
          <div className="microlabel mb-2">METHOD</div>
          <h2 className="font-display text-2xl font-semibold text-ink">这门课怎么学</h2>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {[
              {
                n: '01',
                t: '可视化优先',
                d: '每个抽象概念都有一张能拨弄的图。warp 怎么调度、访存怎么合并、KV Cache 怎么分页 —— 亲眼看见，而不是死记硬背。',
              },
              {
                n: '02',
                t: '在浏览器里"跑" CUDA',
                d: '内置 CUDA 模拟器：写 kernel、配置 grid/block、单步执行、观察每一次内存访问。零环境配置，错误信息精确到线程坐标。',
              },
              {
                n: '03',
                t: '直达生产系统',
                d: '课程的终点不是玩具：FlashAttention 的 online softmax、vLLM 的 PagedAttention、Megatron 的张量并行 —— 都是工业界正在跑的思想。',
              },
            ].map((x) => (
              <div key={x.n} className="panel px-5 py-5">
                <div className="font-display text-2xl font-bold text-volt/80">{x.n}</div>
                <div className="mt-2 text-[15px] font-medium text-ink">{x.t}</div>
                <p className="mt-2 text-[13.5px] leading-[1.9] text-ink2">{x.d}</p>
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
              <p className="mt-2 max-w-[420px] text-[12.5px] leading-relaxed text-ink3">
                全部内容在浏览器中运行。课程深受以下经典资料启发，强烈推荐延伸阅读。
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
