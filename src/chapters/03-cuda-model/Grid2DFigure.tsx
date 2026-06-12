/** SEC6 静态图：2D grid × 2D block 铺满一张「图像」，定位一个像素 */
export function Grid2DFigure() {
  // 12×8 的图像，4×4 的 block → grid(3, 2)
  const W = 12
  const H = 8
  const BX = 4
  const BY = 4
  const cell = 27
  const gap = 1.5
  const x0 = 56
  const y0 = 44
  // 选中：blockIdx=(1,1)，threadIdx=(2,1) → col = 1*4+2 = 6, row = 1*4+1 = 5
  const SB = { x: 1, y: 1 }
  const ST = { x: 2, y: 1 }
  const selCol = SB.x * BX + ST.x
  const selRow = SB.y * BY + ST.y
  const px = (c: number) => x0 + c * (cell + gap)
  const py = (r: number) => y0 + r * (cell + gap)

  const infoX = 470

  return (
    <svg viewBox="0 0 850 310" className="w-full" role="img" aria-label="2D grid 与 2D block 覆盖一张图像，row/col 索引计算">
      {/* 轴标签 */}
      <text x={x0 + (W * (cell + gap)) / 2} y={20} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="currentColor" className="text-ink3">
        x（col）→ blockIdx.x / threadIdx.x
      </text>
      <text
        x={20}
        y={y0 + (H * (cell + gap)) / 2}
        textAnchor="middle"
        fontSize={11}
        fontFamily="var(--font-mono)"
        fill="currentColor"
        className="text-ink3"
        transform={`rotate(-90 20 ${y0 + (H * (cell + gap)) / 2})`}
      >
        y（row）↓
      </text>

      {/* 像素格 */}
      {Array.from({ length: H }, (_, r) =>
        Array.from({ length: W }, (_, c) => {
          const inSelBlock = Math.floor(c / BX) === SB.x && Math.floor(r / BY) === SB.y
          const sel = c === selCol && r === selRow
          return (
            <rect
              key={`${r}-${c}`}
              x={px(c)}
              y={py(r)}
              width={cell}
              height={cell}
              rx={2}
              fill={sel ? 'var(--color-volt)' : inSelBlock ? 'var(--color-cyan)' : 'var(--color-ink3)'}
              fillOpacity={sel ? 0.5 : inSelBlock ? 0.14 : 0.06}
              stroke={sel ? 'var(--color-volt)' : 'var(--color-line)'}
              strokeWidth={sel ? 1.5 : 1}
            />
          )
        }),
      )}

      {/* block 边界（粗线） */}
      {Array.from({ length: W / BX + 1 }, (_, k) => (
        <line
          key={`v${k}`}
          x1={px(k * BX) - gap / 2}
          y1={y0 - 2}
          x2={px(k * BX) - gap / 2}
          y2={py(H) + 2 - gap}
          stroke="var(--color-line2)"
          strokeWidth={2}
        />
      ))}
      {Array.from({ length: H / BY + 1 }, (_, k) => (
        <line
          key={`h${k}`}
          x1={x0 - 2}
          y1={py(k * BY) - gap / 2}
          x2={px(W) + 2 - gap}
          y2={py(k * BY) - gap / 2}
          stroke="var(--color-line2)"
          strokeWidth={2}
        />
      ))}

      {/* block 坐标标签 */}
      {Array.from({ length: H / BY }, (_, by) =>
        Array.from({ length: W / BX }, (_, bx) => {
          const isSel = bx === SB.x && by === SB.y
          return (
            <text
              key={`b${bx}-${by}`}
              x={px(bx * BX) + 4}
              y={py(by * BY) + 13}
              fontSize={9}
              fontFamily="var(--font-mono)"
              fill="currentColor"
              className={isSel ? 'text-cyan' : 'text-ink3'}
              opacity={isSel ? 1 : 0.8}
            >
              ({bx},{by})
            </text>
          )
        }),
      )}

      {/* 选中 block 外框 */}
      <rect
        x={px(SB.x * BX) - gap / 2 - 1}
        y={py(SB.y * BY) - gap / 2 - 1}
        width={BX * (cell + gap) + 2}
        height={BY * (cell + gap) + 2}
        rx={3}
        fill="none"
        stroke="var(--color-cyan)"
        strokeWidth={1.8}
      />

      {/* 右侧推导 */}
      <g fontFamily="var(--font-mono)" fontSize={12.5}>
        <text x={infoX} y={62} fill="currentColor" className="text-ink3" fontSize={10.5}>
          dim3 block(4, 4); dim3 grid(3, 2);
        </text>
        <text x={infoX} y={102} fill="currentColor" className="text-cyan">
          blockIdx = (1, 1)
        </text>
        <text x={infoX} y={126} fill="currentColor" className="text-amber">
          threadIdx = (2, 1)
        </text>
        <text x={infoX} y={168} fill="currentColor" className="text-ink2">
          col = blockIdx.x·4 + threadIdx.x
        </text>
        <text x={infoX} y={190} fill="currentColor" className="text-ink2">
          {'    '}= 1×4 + 2 = <tspan className="text-volt">6</tspan>
        </text>
        <text x={infoX} y={222} fill="currentColor" className="text-ink2">
          row = blockIdx.y·4 + threadIdx.y
        </text>
        <text x={infoX} y={244} fill="currentColor" className="text-ink2">
          {'    '}= 1×4 + 1 = <tspan className="text-volt">5</tspan>
        </text>
        <text x={infoX} y={284} fill="currentColor" className="text-volt">
          → 像素 (row 5, col 6)
        </text>
      </g>
    </svg>
  )
}
