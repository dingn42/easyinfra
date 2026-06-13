import { useT } from '@/lib/i18n'

/** SEC1 静态图：i = blockIdx.x * blockDim.x + threadIdx.x 的几何拆解 */
export function IndexFigure() {
  const t = useT()
  // 4 个 block，每个 8 线程；选中 block 2 的 thread 3 → i = 19
  const BD = 8 // blockDim.x
  const SEL_B = 2
  const SEL_T = 3
  const cell = 21
  const gap = 2
  const pad = 5
  const blockW = BD * (cell + gap) - gap + pad * 2 // 8*23-2+10 = 192
  const bGap = 14
  const x0 = 14
  const y0 = 42
  const bx = (b: number) => x0 + b * (blockW + bGap)
  const cx = (b: number, t: number) => bx(b) + pad + t * (cell + gap)
  const selX = cx(SEL_B, SEL_T)

  return (
    <svg
      viewBox="0 0 850 165"
      className="w-full"
      role="img"
      aria-label={t(
        'Geometric breakdown of the global index i: block start + offset within block',
        '全局索引 i 的几何拆解：block 起点 + 块内偏移',
      )}
    >
      {/* block 标签 */}
      {[0, 1, 2, 3].map((b) => (
        <text
          key={b}
          x={bx(b) + blockW / 2}
          y={y0 - 12}
          textAnchor="middle"
          fontSize={11}
          fontFamily="var(--font-mono)"
          fill="currentColor"
          className={b === SEL_B ? 'text-volt' : 'text-ink3'}
        >
          blockIdx.x = {b}
        </text>
      ))}
      {/* block 外框 + 线程格 */}
      {[0, 1, 2, 3].map((b) => (
        <g key={b}>
          <rect
            x={bx(b)}
            y={y0 - 5}
            width={blockW}
            height={cell + pad * 2}
            rx={5}
            fill="none"
            stroke={b === SEL_B ? 'var(--color-volt)' : 'var(--color-line2)'}
            strokeOpacity={b === SEL_B ? 0.6 : 1}
          />
          {Array.from({ length: BD }, (_, t) => {
            const sel = b === SEL_B && t === SEL_T
            return (
              <g key={t}>
                <rect
                  x={cx(b, t)}
                  y={y0}
                  width={cell}
                  height={cell}
                  rx={2}
                  fill={sel ? 'var(--color-volt)' : 'var(--color-cyan)'}
                  fillOpacity={sel ? 0.45 : 0.1}
                  stroke={sel ? 'var(--color-volt)' : 'var(--color-line)'}
                  strokeWidth={sel ? 1.5 : 1}
                />
                <text
                  x={cx(b, t) + cell / 2}
                  y={y0 + cell / 2 + 3.5}
                  textAnchor="middle"
                  fontSize={9.5}
                  fontFamily="var(--font-mono)"
                  fill="currentColor"
                  className={sel ? 'text-ink' : 'text-ink3'}
                >
                  {t}
                </text>
              </g>
            )
          })}
        </g>
      ))}
      {/* 全局编号刻度（每个 block 的第一个线程） */}
      {[0, 1, 2, 3].map((b) => (
        <text
          key={b}
          x={cx(b, 0) + cell / 2}
          y={y0 + cell + 16}
          textAnchor="middle"
          fontSize={9}
          fontFamily="var(--font-mono)"
          fill="currentColor"
          className="text-ink3"
        >
          i={b * BD}
        </text>
      ))}

      {/* 第一段：blockIdx.x × blockDim.x —— 从 0 跳到 block 2 起点 */}
      <g>
        <line x1={x0} y1={y0 + cell + 32} x2={bx(SEL_B)} y2={y0 + cell + 32} stroke="var(--color-cyan)" strokeWidth={1.5} />
        <line x1={x0} y1={y0 + cell + 27} x2={x0} y2={y0 + cell + 37} stroke="var(--color-cyan)" strokeWidth={1.5} />
        <path
          d={`M ${bx(SEL_B)} ${y0 + cell + 32} l -7 -3.5 v 7 z`}
          fill="var(--color-cyan)"
        />
        <text
          x={(x0 + bx(SEL_B)) / 2}
          y={y0 + cell + 48}
          textAnchor="middle"
          fontSize={11}
          fontFamily="var(--font-mono)"
          fill="currentColor"
          className="text-cyan"
        >
          {t(
            'blockIdx.x × blockDim.x = 2 × 8 = 16 (skip the first two blocks)',
            'blockIdx.x × blockDim.x = 2 × 8 = 16（跳过前两个 block）',
          )}
        </text>
      </g>
      {/* 第二段：+ threadIdx.x —— 块内偏移 */}
      <g>
        <line x1={bx(SEL_B) + pad} y1={y0 + cell + 70} x2={selX + cell / 2} y2={y0 + cell + 70} stroke="var(--color-amber)" strokeWidth={1.5} />
        <line x1={bx(SEL_B) + pad} y1={y0 + cell + 65} x2={bx(SEL_B) + pad} y2={y0 + cell + 75} stroke="var(--color-amber)" strokeWidth={1.5} />
        <path d={`M ${selX + cell / 2} ${y0 + cell + 70} l -7 -3.5 v 7 z`} fill="var(--color-amber)" />
        <line
          x1={selX + cell / 2}
          y1={y0 + cell + 2}
          x2={selX + cell / 2}
          y2={y0 + cell + 66}
          stroke="var(--color-amber)"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.7}
        />
        <text
          x={bx(SEL_B) + pad + 36}
          y={y0 + cell + 86}
          textAnchor="middle"
          fontSize={11}
          fontFamily="var(--font-mono)"
          fill="currentColor"
          className="text-amber"
        >
          + threadIdx.x = 3
        </text>
      </g>
      {/* 结果 */}
      <text
        x={selX + cell + 56}
        y={y0 + cell + 86}
        fontSize={12.5}
        fontFamily="var(--font-mono)"
        fontWeight={600}
        fill="currentColor"
        className="text-volt"
      >
        → i = 16 + 3 = 19
      </text>
    </svg>
  )
}
