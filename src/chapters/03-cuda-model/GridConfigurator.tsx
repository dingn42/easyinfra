import { memo, useMemo, useState, type MouseEvent } from 'react'
import { MathTex, Slider, Stat, Widget } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { fmtInt, pct } from '@/lib/format'

/** LAB 01 —— Grid/Block 配置器：拖 N 和 blockDim，看 grid 怎么铺、哪些线程被浪费 */

// log 间隔的 N 取值（含 1000 这种「不整除」的现实值）
const N_CHOICES = [16, 24, 32, 48, 64, 100, 128, 200, 256, 384, 512, 768, 1000, 1024, 1536, 2048, 3072, 4096]
const DEFAULT_N_IDX = N_CHOICES.indexOf(1000)
const DEFAULT_B = 256
const MAX_SHOW = 18 // 最多渲染的 block 数（再多就折叠中间部分）
const VBW = 880

interface Slot {
  kind: 'block' | 'ellipsis'
  b: number // block 编号（ellipsis 时为 -1）
  x: number
  y: number
}

interface Layout {
  s: number // 线程格边长
  g: number // 格间距
  pad: number
  blockW: number
  blockH: number
  labelH: number
  width: number
  height: number
  slots: Slot[]
  hidden: number // 被折叠的 block 数
}

function computeLayout(grid: number, B: number): Layout {
  const warps = Math.ceil(B / 32)
  let s = B <= 64 ? 10.5 : B <= 128 ? 9 : B <= 256 ? 8 : B <= 512 ? 7 : 6
  if (grid > 8) s = Math.min(s, 7.5)
  if (grid > 16) s = Math.min(s, 6.5)
  const g = s >= 9 ? 1.4 : 1
  const pad = 4
  const blockW = 32 * s + 31 * g + pad * 2
  const blockH = warps * s + (warps - 1) * g + pad * 2
  const labelH = 15
  const gx = 12
  const gy = 10

  // 要显示哪些 block：太多时显示前 MAX_SHOW-1 个 + 折叠标记 + 最后一个
  let ids: number[]
  let hidden = 0
  if (grid <= MAX_SHOW) {
    ids = Array.from({ length: grid }, (_, i) => i)
  } else {
    ids = Array.from({ length: MAX_SHOW - 1 }, (_, i) => i)
    ids.push(grid - 1)
    hidden = grid - MAX_SHOW
  }

  const perRow = Math.max(1, Math.floor((VBW + gx) / (blockW + gx)))
  const slotCount = ids.length + (hidden > 0 ? 1 : 0)
  const rows = Math.ceil(slotCount / perRow)
  const rowW = Math.min(slotCount, perRow) * (blockW + gx) - gx
  const xOff = Math.max(0, (VBW - rowW) / 2)

  const slots: Slot[] = []
  let k = 0
  const push = (kind: Slot['kind'], b: number) => {
    const row = Math.floor(k / perRow)
    const col = k % perRow
    slots.push({ kind, b, x: xOff + col * (blockW + gx), y: row * (blockH + labelH + gy) + labelH })
    k++
  }
  for (let j = 0; j < ids.length; j++) {
    if (hidden > 0 && j === ids.length - 1) push('ellipsis', -1)
    push('block', ids[j])
  }

  return {
    s,
    g,
    pad,
    blockW,
    blockH,
    labelH,
    width: VBW,
    height: rows * (blockH + labelH + gy) - gy + 4,
    slots,
    hidden,
  }
}

/** 线程格层：与选中态无关，memo 掉避免点选时重渲染几千个 rect */
const CellsLayer = memo(function CellsLayer({ N, B, lay }: { N: number; B: number; lay: Layout }) {
  const t = useT()
  return (
    <>
      {lay.slots.map((slot) =>
        slot.kind === 'ellipsis' ? (
          <g key="ellipsis">
            <rect
              x={slot.x}
              y={slot.y}
              width={lay.blockW}
              height={lay.blockH}
              rx={4}
              fill="none"
              stroke="var(--color-line)"
              strokeDasharray="4 4"
            />
            <text
              x={slot.x + lay.blockW / 2}
              y={slot.y + lay.blockH / 2 + 4}
              textAnchor="middle"
              fontSize={11}
              fontFamily="var(--font-mono)"
              fill="currentColor"
              className="text-ink3"
            >
              {t(`⋯ ${lay.hidden} more identical blocks omitted ⋯`, `⋯ 省略 ${lay.hidden} 个相同的 block ⋯`)}
            </text>
          </g>
        ) : (
          <g key={slot.b}>
            <text
              x={slot.x + 1}
              y={slot.y - 4}
              fontSize={10}
              fontFamily="var(--font-mono)"
              fill="currentColor"
              className="text-ink3"
            >
              block {slot.b}
            </text>
            <rect
              x={slot.x - 1.5}
              y={slot.y - 1.5}
              width={lay.blockW + 3}
              height={lay.blockH + 3}
              rx={4}
              fill="none"
              stroke="var(--color-line2)"
            />
            {Array.from({ length: B }, (_, t) => {
              const i = slot.b * B + t
              const warp = t >> 5
              const lane = t & 31
              const oob = i >= N
              return (
                <rect
                  key={t}
                  data-i={i}
                  x={slot.x + lay.pad + lane * (lay.s + lay.g)}
                  y={slot.y + lay.pad + warp * (lay.s + lay.g)}
                  width={lay.s}
                  height={lay.s}
                  rx={1}
                  className="cursor-pointer"
                  fill={oob ? 'var(--color-rose)' : 'var(--color-cyan)'}
                  fillOpacity={oob ? 0.4 : warp % 2 === 0 ? 0.22 : 0.07}
                  stroke={oob ? 'var(--color-rose)' : 'var(--color-line)'}
                  strokeOpacity={oob ? 0.6 : 1}
                  strokeWidth={0.75}
                />
              )
            })}
          </g>
        ),
      )}
    </>
  )
})

export function GridConfigurator() {
  const t = useT()
  const [nIdx, setNIdx] = useState(DEFAULT_N_IDX)
  const [blockDim, setBlockDim] = useState(DEFAULT_B)
  const [selected, setSelected] = useState(1023) // 默认选最后一个线程：刚好演示越界

  const N = N_CHOICES[nIdx]
  const grid = Math.ceil(N / blockDim)
  const total = grid * blockDim
  const waste = total - N
  const sel = Math.min(selected, total - 1) // N/B 变化后钳制
  const selB = Math.floor(sel / blockDim)
  const selT = sel % blockDim
  const selOob = sel >= N

  const lay = useMemo(() => computeLayout(grid, blockDim), [grid, blockDim])

  // 选中线程的格子位置（可能在被折叠区域里 → 不画 overlay）
  const selPos = useMemo(() => {
    const slot = lay.slots.find((sl) => sl.kind === 'block' && sl.b === selB)
    if (!slot) return null
    const warp = selT >> 5
    const lane = selT & 31
    return {
      x: slot.x + lay.pad + lane * (lay.s + lay.g),
      y: slot.y + lay.pad + warp * (lay.s + lay.g),
    }
  }, [lay, selB, selT])

  const onSvgClick = (e: MouseEvent<SVGSVGElement>) => {
    const el = e.target as SVGElement
    const di = el.getAttribute('data-i')
    if (di != null) setSelected(Number(di))
  }

  const reset = () => {
    setNIdx(DEFAULT_N_IDX)
    setBlockDim(DEFAULT_B)
    setSelected(1023)
  }

  return (
    <Widget
      index={1}
      title={t('Grid/Block configurator', 'Grid/Block 配置器')}
      subtitle={t('Drag N and blockDim, watch how the grid tiles the data', '拖动 N 与 blockDim，看 grid 怎么铺满数据')}
      wide
      onReset={reset}
      footer={t(
        <>
          Try this: pull N up to <span className="font-mono text-ink">1000</span> with blockDim set to{' '}
          <span className="font-mono text-ink">256</span> — the last block has 24 threads with "nothing to do" (rose
          cells). Then set N to <span className="font-mono text-ink">16</span> and max out blockDim at{' '}
          <span className="font-mono text-ink">1024</span> to see just how brutal the waste rate gets. Click any
          thread cell to see its index derivation.
        </>,
        <>
          试试：把 N 拉到 <span className="font-mono text-ink">1000</span>、blockDim 设为{' '}
          <span className="font-mono text-ink">256</span> —— 最后一个 block 有 24 个线程「无活可干」（rose 色）。再把 N 调到{' '}
          <span className="font-mono text-ink">16</span>、blockDim 拉满 <span className="font-mono text-ink">1024</span>，看看浪费率有多惨烈。
          点击任意线程格可查看它的索引推导。
        </>,
      )}
    >
      {/* 控件 */}
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
        <Slider
          label={t('N · array element count (log scale)', 'N · 数组元素数（log 刻度）')}
          value={nIdx}
          min={0}
          max={N_CHOICES.length - 1}
          step={1}
          onChange={(v) => setNIdx(Math.round(v))}
          fmt={(v) => fmtInt(N_CHOICES[Math.round(v)])}
        />
        <Slider
          label={t('blockDim.x · threads per block', 'blockDim.x · 每 block 线程数')}
          value={blockDim}
          min={32}
          max={1024}
          step={32}
          onChange={setBlockDim}
          fmt={fmtInt}
        />
      </div>

      {/* 代入式 */}
      <MathTex
        block
        tex={`\\mathrm{gridDim} \\;=\\; \\left\\lceil \\frac{N}{\\mathrm{blockDim}} \\right\\rceil \\;=\\; \\left\\lceil \\frac{${N}}{${blockDim}} \\right\\rceil \\;=\\; ${grid}`}
      />

      {/* 读数 */}
      <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label={t('GRIDDIM (blocks)', 'GRIDDIM（block 数）')} value={fmtInt(grid)} tone="volt" />
        <Stat label={t('Threads launched', '总启动线程')} value={fmtInt(total)} tone="cyan" />
        <Stat label={t('Working threads', '干活线程')} value={fmtInt(N)} />
        <Stat
          label={t('Wasted threads', '浪费线程')}
          value={
            <>
              {fmtInt(waste)}
              <span className="ml-1.5 text-[0.62em] font-normal">({pct(waste / total, 1)})</span>
            </>
          }
          tone={waste > 0 ? 'rose' : 'ink'}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_252px]">
        {/* block 阵列 */}
        <div className="overflow-hidden rounded-md border border-line bg-bg2 p-2">
          <svg viewBox={`0 0 ${lay.width} ${lay.height}`} className="w-full" onClick={onSvgClick}>
            <CellsLayer N={N} B={blockDim} lay={lay} />
            {selPos && (
              <rect
                x={selPos.x - 1}
                y={selPos.y - 1}
                width={lay.s + 2}
                height={lay.s + 2}
                rx={1.5}
                fill="var(--color-volt)"
                fillOpacity={0.45}
                stroke="var(--color-volt)"
                strokeWidth={1.5}
                pointerEvents="none"
              />
            )}
          </svg>
          {/* 图例 */}
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 px-1 font-mono text-[10.5px] text-ink3">
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-[2px] bg-cyan/25 ring-1 ring-line" /> {t('even warp', '偶数 warp')}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-[2px] bg-cyan/10 ring-1 ring-line" /> {t('odd warp', '奇数 warp')}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-[2px] bg-rose/40 ring-1 ring-rose/50" />{' '}
              {t('out-of-bounds thread (i ≥ N)', '越界线程（i ≥ N）')}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-[2px] bg-volt/50 ring-1 ring-volt" /> {t('selected', '选中')}
            </span>
          </div>
        </div>

        {/* 选中线程面板 */}
        <div className="rounded-md border border-line bg-bg2 p-3.5">
          <div className="microlabel mb-2.5 text-volt">SELECTED THREAD</div>
          <div className="space-y-1 font-mono text-[12px] leading-[1.9] text-ink2">
            <div>
              blockIdx.x&nbsp; = <span className="text-cyan">{selB}</span>
            </div>
            <div>
              threadIdx.x = <span className="text-amber">{selT}</span>
            </div>
            <div className="border-t border-line pt-1.5 text-ink3">i = blockIdx.x × blockDim.x + threadIdx.x</div>
            <div>
              &nbsp;&nbsp;= <span className="text-cyan">{selB}</span> × {blockDim} + <span className="text-amber">{selT}</span>
            </div>
            <div>
              &nbsp;&nbsp;= <span className="text-volt">{fmtInt(sel)}</span>
            </div>
            <div className="border-t border-line pt-1.5">
              if (i &lt; n)：{fmtInt(sel)} &lt; {fmtInt(N)} →{' '}
              {selOob ? <span className="text-rose">false</span> : <span className="text-volt">true</span>}
            </div>
            {selOob ? (
              <div className="text-rose">{t('✗ returns early, this thread spins idle', '✗ 直接返回，这个线程空转')}</div>
            ) : (
              <div className="text-volt">{t('✓ runs C[i] = A[i] + B[i]', '✓ 执行 C[i] = A[i] + B[i]')}</div>
            )}
          </div>
          {!selPos && (
            <div className="mt-2 text-[11px] leading-relaxed text-ink3">
              {t(
                '(this thread’s block is folded and not shown, but the math is unchanged)',
                '（该线程所在 block 被折叠未显示，但计算照常）',
              )}
            </div>
          )}
        </div>
      </div>
    </Widget>
  )
}
