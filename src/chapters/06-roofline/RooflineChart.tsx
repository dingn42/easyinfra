import { useRef, useState } from 'react'
import { Segmented, Stat, Widget } from '@/components/ui'
import { pick, useLocale, useT, type Loc } from '@/lib/i18n'
import { clamp } from '@/lib/format'
import { HARDWARE, attainable, fmtAI, hwById, ridgeOf, type Hardware } from './data'

/** LAB 02 Roofline 交互图：log-log 屋顶线 + 预置 kernel 点 + 可拖拽的「你的 kernel」 */

const VW = 760
const VH = 470
const ML = 62
const MR = 20
const MT = 30
const MB = 50
const LX0 = Math.log10(0.1)
const LX1 = Math.log10(1000)
const LY0 = Math.log10(0.1)
const LY1 = Math.log10(2000)

const px = (ai: number) => ML + ((Math.log10(ai) - LX0) / (LX1 - LX0)) * (VW - ML - MR)
const py = (tf: number) => VH - MB - ((Math.log10(tf) - LY0) / (LY1 - LY0)) * (VH - MT - MB)
const invx = (x: number) => 10 ** (LX0 + ((x - ML) / (VW - ML - MR)) * (LX1 - LX0))
const invy = (y: number) => 10 ** (LY0 + ((VH - MB - y) / (VH - MT - MB)) * (LY1 - LY0))

const TICKS = [0.1, 1, 10, 100, 1000]

interface KernelPt {
  id: string
  /** 显示名（双语） */
  name: Loc
  ai: number
  /** 典型实测性能占屋顶的比例 */
  frac: number
  blurb: Loc
  /** 标签锚点：1=右侧（默认），-1=左侧 */
  side?: -1 | 1
}

const KERNELS: KernelPt[] = [
  {
    id: 'vecadd',
    name: { en: 'vecAdd', zh: 'vecAdd' },
    ai: 0.25,
    frac: 0.85,
    blurb: {
      en: 'Elementwise op: every byte moved earns only a fraction of a FLOP. No block-size tweak helps — it stays glued to the bandwidth slope, the textbook pure-bandwidth business.',
      zh: '逐元素操作：每搬一个字节只摊到零点几个 FLOP。不管怎么调 block size，它都贴死在带宽斜线上——典型的纯带宽生意。',
    },
  },
  {
    id: 'gemv',
    name: { en: 'GEMV', zh: 'GEMV' },
    ai: 0.5,
    frac: 0.8,
    blurb: {
      en: 'Matrix × vector: each matrix element is read from memory, used in just 2 ops, then discarded — zero reuse. A batch=1 LLM linear layer is essentially a row of GEMVs.',
      zh: '矩阵×向量：矩阵的每个元素从显存读进来只参与 2 次运算就被丢掉，零复用。batch=1 的 LLM 线性层本质上就是一排 GEMV。',
    },
  },
  {
    id: 'attn',
    name: { en: 'attn-decode', zh: 'attn-decode' },
    ai: 1.0,
    frac: 0.6,
    blurb: {
      en: 'In decode, one query sweeps the entire KV cache: each K/V element is used only once or twice, AI ≈ 1–2. This is the root of LLM inference being memory-bound — the star of the next three chapters.',
      zh: '解码阶段一条 query 扫整个 KV cache：每个 K/V 元素只用一两次，AI ≈ 1~2。这就是 LLM 推理 memory-bound 的根源，后面三章的主角。',
    },
  },
  {
    id: 'sgemm',
    name: { en: 'small GEMM 64³', zh: '小 GEMM 64³' },
    ai: 21,
    frac: 0.5,
    blurb: {
      en: 'The matrix is too small, reuse is limited: a BF16 square has AI ≈ n/3, and n=64 gives only 21 FLOP/B — an order of magnitude short of the A100 ridge (164). Bandwidth chokes it before it even starts.',
      zh: '矩阵太小，复用有限：BF16 方阵的 AI ≈ n/3，n=64 只有 21 FLOP/B，离 A100 的 ridge（164）差一个数量级。还没起跑就被带宽掐住。',
    },
  },
  {
    id: 'lgemm',
    name: { en: 'large GEMM 2048³', zh: '大 GEMM 2048³' },
    ai: 680,
    frac: 0.85,
    side: -1,
    blurb: {
      en: 'At n=2048, AI ≈ n/3 ≈ 680, far past any card’s ridge, planted firmly on the compute plateau — the Tensor Cores are finally fed. Matrices have to be big enough for compute to matter.',
      zh: 'n=2048 时 AI ≈ n/3 ≈ 680，远超任何卡的 ridge，稳稳压在算力平台上——Tensor Core 终于吃饱。矩阵要够大，算力才有意义。',
    },
  },
]

const DEF_USER = { ai: 2, tf: 1.2 }

function roofPath(hw: Hardware) {
  const ridge = ridgeOf(hw.tensor, hw.bw)
  return { ridge, slopeStart: { x: px(0.1), y: py(0.1 * hw.bw) }, knee: { x: px(ridge), y: py(hw.tensor) } }
}

export function RooflineChart() {
  const t = useT()
  const { lang } = useLocale()
  const [hwId, setHwId] = useState<string>('a100')
  const [user, setUser] = useState(DEF_USER)
  const [hovered, setHovered] = useState<string | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const dragRef = useRef(false)

  const hw = hwById(hwId)
  const { ridge, slopeStart, knee } = roofPath(hw)
  const ridge32 = ridgeOf(hw.fp32, hw.bw)

  const reset = () => {
    setHwId('a100')
    setUser(DEF_USER)
    setHovered(null)
  }

  /** 像素坐标 → (AI, TFLOPS)，并钳制在屋顶之下 */
  const toPoint = (e: React.PointerEvent): { ai: number; tf: number } | null => {
    const svg = svgRef.current
    if (!svg) return null
    const r = svg.getBoundingClientRect()
    if (r.width === 0) return null
    const x = ((e.clientX - r.left) / r.width) * VW
    const y = ((e.clientY - r.top) / r.height) * VH
    const ai = clamp(invx(x), 0.1, 1000)
    const tfRaw = clamp(invy(y), 0.1, 2000)
    return { ai, tf: Math.min(tfRaw, attainable(ai, hw.bw, hw.tensor)) }
  }

  const startDrag = (e: React.PointerEvent) => {
    e.preventDefault()
    dragRef.current = true
    svgRef.current?.setPointerCapture(e.pointerId)
    const p = toPoint(e)
    if (p) setUser(p)
  }
  const moveDrag = (e: React.PointerEvent) => {
    if (!dragRef.current) return
    const p = toPoint(e)
    if (p) setUser(p)
  }
  const endDrag = () => {
    dragRef.current = false
  }

  // 切换硬件后把用户点钳到新屋顶下
  const uRoof = attainable(user.ai, hw.bw, hw.tensor)
  const uTf = Math.min(user.tf, uRoof)
  const gap = uRoof / uTf
  const memSide = user.ai < ridge
  const nearRoof = gap <= 1.15

  const bwAngle =
    (Math.atan2(py(2 * hw.bw) - py(0.2 * hw.bw), px(2) - px(0.2)) * 180) / Math.PI
  const bwMid = { x: px(Math.sqrt(0.2 * 2)), y: py(Math.sqrt(0.2 * 2) * hw.bw) }

  const hoveredK = KERNELS.find((k) => k.id === hovered) ?? null

  const advice = memSide
    ? nearRoof
      ? t(
          'Already running flat-out against the bandwidth slope. Two roads to faster: right — raise AI (kernel fusion, tiling reuse, quantization to cut bytes); or swap in a card with higher bandwidth. Note: more compute or a higher clock is pointless in this region.',
          '已经贴着带宽斜线跑满了。想更快只有两条路：向右——提高 AI（kernel 融合、tiling 复用、量化减字节）；或者换带宽更高的卡。注意：多堆算力、拉频率在这个区域毫无意义。',
        )
      : t(
          'In the memory-bound region, but not even saturating bandwidth: first check whether accesses are coalesced, the L2 hit rate, and whether the kernel is so small that launch overhead dominates — step one is pushing the point straight up onto the slope.',
          '在 memory-bound 区，但连带宽都没吃满：先查访存是否合并（coalesced）、L2 命中率、kernel 是否太小启动开销占大头——第一步是把点垂直推到斜线上。',
        )
    : nearRoof
      ? t(
          'Hugging the compute plateau — this kernel has wrung the hardware dry. To go faster you can only swap in a stronger card, or compute less at the algorithm level (sparsity, approximation).',
          '贴着算力平台——这个 kernel 已经把硬件吃干净了。再想快只能换更强的卡，或者从算法上少算（稀疏化、近似）。',
        )
      : t(
          'In the compute-bound region but far below the plateau: this is a utilization problem. Check whether the Tensor Cores are really working (instructions, data layout correct), whether occupancy is enough, and what the warps are stalling on.',
          '在 compute-bound 区却离平台很远：这是利用率问题。查 Tensor Core 是否真的在干活（指令、数据布局对不对）、occupancy 够不够、warp 在 stall 什么。',
        )

  return (
    <Widget
      index={2}
      title={t('Interactive Roofline', 'Roofline 交互图')}
      subtitle={t('Locate any kernel’s bottleneck on one plot', '一张图定位任何 kernel 的瓶颈')}
      onReset={reset}
      wide
      footer={t(
        <>
          The cyan points are preset kernels (hover/tap for notes); the{' '}
          <span className="text-volt">green point</span> is "your kernel" — drag it anywhere on the plot (mouse or
          touch), and the diagnosis below updates its situation and headroom in real time.
        </>,
        <>
          青色点是预置 kernel（悬停/点按看说明）；<span className="text-volt">绿色点</span>
          是「你的 kernel」，在图里任意拖动（鼠标或触摸），下方实时判定它的处境与优化空间。
        </>,
      )}
    >
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <Segmented
          options={HARDWARE.map((h) => ({ value: h.id, label: h.name }))}
          value={hwId}
          onChange={setHwId}
        />
        <span className="font-mono text-[11px] text-ink3">
          {hw.mem} {hw.bw} TB/s · BF16 {hw.tensor} TFLOPS · ridge ≈ {ridge.toFixed(0)} FLOP/B
        </span>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${VW} ${VH}`}
        className="w-full touch-none select-none"
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {/* 区域底色 */}
        <rect x={ML} y={MT} width={Math.max(0, px(ridge) - ML)} height={VH - MT - MB} fill="var(--color-amber)" opacity={0.04} />
        <rect x={px(ridge)} y={MT} width={Math.max(0, VW - MR - px(ridge))} height={VH - MT - MB} fill="var(--color-volt)" opacity={0.03} />

        {/* 网格 + 刻度 */}
        {TICKS.map((t) => (
          <g key={`x${t}`}>
            <line x1={px(t)} y1={MT} x2={px(t)} y2={VH - MB} stroke="var(--color-line)" strokeWidth={1} />
            <text x={px(t)} y={VH - MB + 18} textAnchor="middle" fontSize={12} fill="currentColor" className="font-mono text-ink3">
              {t}
            </text>
          </g>
        ))}
        {TICKS.map((t) => (
          <g key={`y${t}`}>
            <line x1={ML} y1={py(t)} x2={VW - MR} y2={py(t)} stroke="var(--color-line)" strokeWidth={1} />
            <text x={ML - 8} y={py(t) + 4} textAnchor="end" fontSize={12} fill="currentColor" className="font-mono text-ink3">
              {t}
            </text>
          </g>
        ))}
        <text x={(ML + VW - MR) / 2} y={VH - 8} textAnchor="middle" fontSize={12.5} fill="currentColor" className="font-mono text-ink2">
          {t('Arithmetic intensity AI (FLOP / Byte) →', '算术强度 AI（FLOP / Byte）→')}
        </text>
        <text
          x={16}
          y={(MT + VH - MB) / 2}
          textAnchor="middle"
          fontSize={12.5}
          fill="currentColor"
          className="font-mono text-ink2"
          transform={`rotate(-90 16 ${(MT + VH - MB) / 2})`}
        >
          {t('Attainable performance (TFLOPS) →', '可达性能（TFLOPS）→')}
        </text>

        {/* 区域标签 */}
        <text x={ML + 10} y={MT + 16} fontSize={11} fill="currentColor" className="font-mono text-amber" opacity={0.8}>
          ◀ MEMORY-BOUND
        </text>
        <text x={VW - MR - 10} y={MT + 16} textAnchor="end" fontSize={11} fill="currentColor" className="font-mono text-volt" opacity={0.8}>
          COMPUTE-BOUND ▶
        </text>

        {/* ridge 竖线 */}
        <line x1={px(ridge)} y1={knee.y} x2={px(ridge)} y2={VH - MB} stroke="var(--color-line2)" strokeWidth={1} strokeDasharray="2 4" />
        <text x={px(ridge)} y={VH - MB - 8} textAnchor="middle" fontSize={11} fill="currentColor" className="font-mono text-ink3">
          ridge {ridge.toFixed(0)}
        </text>

        {/* FP32 屋顶（虚线） */}
        <line x1={px(ridge32)} y1={py(hw.fp32)} x2={VW - MR} y2={py(hw.fp32)} stroke="var(--color-ink3)" strokeWidth={1.2} strokeDasharray="5 5" opacity={0.7} />
        <text
          x={px(Math.sqrt(ridge32 * 1000))}
          y={py(hw.fp32) - 7}
          textAnchor="middle"
          fontSize={11}
          fill="currentColor"
          className="font-mono text-ink3"
        >
          {hw.fp32} TFLOPS · FP32 CUDA Core
        </text>

        {/* 带宽斜线（memory 屋顶） */}
        <line x1={slopeStart.x} y1={slopeStart.y} x2={knee.x} y2={knee.y} stroke="var(--color-amber)" strokeWidth={2} />
        <text
          x={bwMid.x}
          y={bwMid.y - 10}
          textAnchor="middle"
          fontSize={11.5}
          fill="currentColor"
          className="font-mono text-amber"
          transform={`rotate(${bwAngle} ${bwMid.x} ${bwMid.y - 10})`}
        >
          {hw.mem} {hw.bw} TB/s
        </text>

        {/* 算力平台（compute 屋顶） */}
        <line x1={knee.x} y1={knee.y} x2={VW - MR} y2={knee.y} stroke="var(--color-volt)" strokeWidth={2} />
        <text
          x={px(Math.sqrt(ridge * 1000))}
          y={knee.y - 8}
          textAnchor="middle"
          fontSize={11.5}
          fill="currentColor"
          className="font-mono text-volt"
        >
          {hw.tensor} TFLOPS · BF16 Tensor Core
        </text>
        <circle cx={knee.x} cy={knee.y} r={3} fill="var(--color-volt)" />

        {/* 拖拽热区 */}
        <rect
          x={ML}
          y={MT}
          width={VW - ML - MR}
          height={VH - MT - MB}
          fill="transparent"
          className="cursor-crosshair"
          onPointerDown={startDrag}
        />

        {/* 预置 kernel 点 */}
        {KERNELS.map((k) => {
          const roof = attainable(k.ai, hw.bw, hw.tensor)
          const cx = px(k.ai)
          const cy = py(roof * k.frac)
          const on = hovered === k.id
          return (
            <g
              key={k.id}
              onPointerEnter={() => setHovered(k.id)}
              onPointerLeave={() => setHovered((h) => (h === k.id ? null : h))}
              onPointerDown={(e) => {
                e.stopPropagation()
                setHovered(k.id)
              }}
              className="cursor-pointer"
            >
              <circle cx={cx} cy={cy} r={14} fill="transparent" />
              {on && <circle cx={cx} cy={cy} r={9} fill="none" stroke="var(--color-cyan)" strokeWidth={1.5} opacity={0.7} />}
              <circle cx={cx} cy={cy} r={5} fill="var(--color-cyan)" stroke="var(--color-bg)" strokeWidth={1.5} />
              <text
                x={cx + (k.side === -1 ? -10 : 10)}
                y={cy - 8}
                textAnchor={k.side === -1 ? 'end' : 'start'}
                fontSize={11.5}
                fill="currentColor"
                className={`font-mono ${on ? 'text-cyan' : 'text-ink3'}`}
              >
                {pick(k.name, lang)}
              </text>
            </g>
          )
        })}

        {/* 你的 kernel（可拖拽） */}
        <g onPointerDown={startDrag} className="cursor-grab active:cursor-grabbing">
          <line x1={px(user.ai)} y1={py(uTf)} x2={px(user.ai)} y2={py(uRoof)} stroke="var(--color-volt)" strokeWidth={1} strokeDasharray="3 3" opacity={0.55} />
          <circle cx={px(user.ai)} cy={py(uRoof)} r={3} fill="none" stroke="var(--color-volt)" strokeWidth={1} opacity={0.6} />
          <circle cx={px(user.ai)} cy={py(uTf)} r={16} fill="transparent" />
          <circle cx={px(user.ai)} cy={py(uTf)} r={10} fill="none" stroke="var(--color-volt)" strokeWidth={1} opacity={0.5} />
          <circle cx={px(user.ai)} cy={py(uTf)} r={6} fill="var(--color-volt)" stroke="var(--color-bg)" strokeWidth={1.5} />
          <text
            x={px(user.ai) + (px(user.ai) > VW - 120 ? -14 : 14)}
            y={py(uTf) + 14}
            textAnchor={px(user.ai) > VW - 120 ? 'end' : 'start'}
            fontSize={11.5}
            fill="currentColor"
            className="font-mono text-volt"
          >
            {t('your kernel', '你的 kernel')}
          </text>
        </g>
      </svg>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1.3fr]">
        {/* 预置点说明卡 */}
        <div className="rounded-md border border-line bg-bg2 px-4 py-3.5 text-[13px] leading-[1.9] text-ink2">
          {hoveredK ? (
            <>
              <div className="mb-1 flex items-baseline gap-3">
                <span className="font-mono text-sm text-cyan">{pick(hoveredK.name, lang)}</span>
                <span className="font-mono text-[11px] text-ink3">
                  AI ≈ {fmtAI(hoveredK.ai)} FLOP/B · {t('roof', '屋顶')}{' '}
                  {attainable(hoveredK.ai, hw.bw, hw.tensor).toFixed(attainable(hoveredK.ai, hw.bw, hw.tensor) < 10 ? 2 : 0)} TFLOPS @ {hw.name}
                </span>
              </div>
              {pick(hoveredK.blurb, lang)}
            </>
          ) : (
            <>
              <span className="microlabel mb-1 block">{t('PRESET KERNELS', '预置 KERNEL')}</span>
              {t(
                'Hover or tap the cyan points to see why each class of kernel lands where it does. Notice they nearly all cluster to the left of the ridge — among everyday kernels, compute-bound is the rare species.',
                '悬停或点按图中的青色点，看每类 kernel 为什么落在那里。注意它们几乎都挤在 ridge 左边——日常 kernel 里 compute-bound 反而是稀有物种。',
              )}
            </>
          )}
        </div>

        {/* 你的 kernel 判定 */}
        <div className="rounded-md border border-volt/30 bg-volt/[0.04] px-4 py-3.5">
          <div className="mb-2 flex flex-wrap items-end gap-x-6 gap-y-2">
            <Stat label={t('Arithmetic intensity', '算术强度')} value={fmtAI(user.ai)} unit="FLOP/B" tone="volt" size="sm" />
            <Stat label={t('Measured performance', '实测性能')} value={uTf < 10 ? uTf.toFixed(2) : uTf.toFixed(0)} unit="TFLOPS" tone="ink" size="sm" />
            <Stat label={t('Roof here', '此处屋顶')} value={uRoof < 10 ? uRoof.toFixed(2) : uRoof.toFixed(0)} unit="TFLOPS" tone={memSide ? 'amber' : 'volt'} size="sm" />
            <Stat
              label={t('Gap to roof = headroom', '距屋顶 = 优化空间')}
              value={`×${gap.toFixed(1)}`}
              tone={gap >= 3 ? 'rose' : gap >= 1.5 ? 'amber' : 'volt'}
              size="sm"
            />
          </div>
          <p className="text-[13px] leading-[1.9] text-ink2">
            <span className={`mr-2 font-mono text-[11px] tracking-wider ${memSide ? 'text-amber' : 'text-volt'}`}>
              {memSide ? '◀ MEMORY-BOUND' : 'COMPUTE-BOUND ▶'}
            </span>
            {t('Roof', '屋顶')} = {memSide ? `AI × ${hw.bw} TB/s` : t(`peak ${hw.tensor} TFLOPS`, `峰值 ${hw.tensor} TFLOPS`)}. {advice}
          </p>
        </div>
      </div>
    </Widget>
  )
}
