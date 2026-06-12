import { Figure } from '@/components/ui'

/**
 * SEC4 静态图：Megatron 式 MLP 张量并行 —— W1 列切、W2 行切，
 * 中间零通信，末尾一次 AllReduce；右侧 NVLink vs IB 带宽对比条。
 */
export function TensorParallelFigure() {
  const mono = 'var(--font-mono, monospace)'
  return (
    <Figure caption="MLP 的 Megatron 切法：W1 按列切（各卡得到激活的不同列），GeLU 逐元素可独立算，W2 按行切（各卡得到 Y 的部分和），整个两层只在末尾通信一次。右：这次 AllReduce 走 NVLink 还是跨机 InfiniBand，带宽差约 18 倍。">
      <svg viewBox="0 0 720 320" className="w-full select-none" role="img" aria-label="张量并行列切/行切示意图">
        {/* ── GPU0 行 ── */}
        {[0, 1].map((g) => {
          const y0 = g === 0 ? 30 : 150
          const wColor = g === 0 ? 'var(--color-cyan)' : 'var(--color-amber)'
          return (
            <g key={g}>
              <rect x={10} y={y0 - 18} width={560} height={112} rx={8} fill="var(--color-panel)" stroke="var(--color-line)" />
              <text x={22} y={y0} fontSize={11} fontFamily={mono} fill="var(--color-ink3)">
                GPU {g}
              </text>
              {/* X（完整复制） */}
              <rect x={60} y={y0 + 10} width={56} height={56} rx={4} fill="var(--color-ink3)" opacity={0.25} />
              <text x={88} y={y0 + 40} textAnchor="middle" dominantBaseline="central" fontSize={13} fontFamily={mono} fill="var(--color-ink)">
                X
              </text>
              {/* W1 列切：高瘦半块 */}
              <text x={150} y={y0 + 40} textAnchor="middle" fontSize={13} fill="var(--color-ink2)">
                ×
              </text>
              <rect x={172} y={y0 + 2} width={30} height={72} rx={4} fill={wColor} opacity={0.55} />
              <rect x={172} y={y0 + 2} width={62} height={72} rx={4} fill="none" stroke="var(--color-line2)" strokeDasharray="3 3" />
              <text x={203} y={y0 + 90} textAnchor="middle" fontSize={10} fontFamily={mono} fill="var(--color-ink3)">
                W₁ 列切 {g === 0 ? '左半' : '右半'}
              </text>
              {/* 部分激活 */}
              <text x={258} y={y0 + 40} textAnchor="middle" fontSize={13} fill="var(--color-ink2)">
                →
              </text>
              <rect x={278} y={y0 + 10} width={30} height={56} rx={4} fill={wColor} opacity={0.4} />
              <text x={293} y={y0 + 90} textAnchor="middle" fontSize={10} fontFamily={mono} fill="var(--color-ink3)">
                GeLU(XW₁{g === 0 ? 'ᵃ' : 'ᵇ'})
              </text>
              {/* W2 行切：矮宽半块 */}
              <text x={332} y={y0 + 40} textAnchor="middle" fontSize={13} fill="var(--color-ink2)">
                ×
              </text>
              <rect x={352} y={y0 + 10} width={72} height={26} rx={4} fill={wColor} opacity={0.55} />
              <rect x={352} y={y0 + 10} width={72} height={56} rx={4} fill="none" stroke="var(--color-line2)" strokeDasharray="3 3" />
              <text x={388} y={y0 + 90} textAnchor="middle" fontSize={10} fontFamily={mono} fill="var(--color-ink3)">
                W₂ 行切 {g === 0 ? '上半' : '下半'}
              </text>
              {/* 部分和 */}
              <text x={448} y={y0 + 40} textAnchor="middle" fontSize={13} fill="var(--color-ink2)">
                →
              </text>
              <rect x={468} y={y0 + 10} width={56} height={56} rx={4} fill={wColor} opacity={0.4} />
              <text x={496} y={y0 + 40} textAnchor="middle" dominantBaseline="central" fontSize={11.5} fontFamily={mono} fill="var(--color-ink)">
                Y{g === 0 ? '₀' : '₁'}
              </text>
              <text x={496} y={y0 + 90} textAnchor="middle" fontSize={10} fontFamily={mono} fill="var(--color-ink3)">
                部分和
              </text>
            </g>
          )
        })}

        {/* 中间「零通信」标注 */}
        <text x={293} y={146} textAnchor="middle" fontSize={10.5} fontFamily={mono} fill="var(--color-volt)">
          ↕ 中间零通信
        </text>

        {/* AllReduce 汇合 */}
        <path d="M 530 58 C 575 58 575 100 600 138" fill="none" stroke="var(--color-line2)" strokeWidth={1.2} />
        <path d="M 530 208 C 575 208 575 176 600 142" fill="none" stroke="var(--color-line2)" strokeWidth={1.2} />
        <rect x={600} y={116} width={110} height={50} rx={6} fill="var(--color-volt)" opacity={0.12} />
        <rect x={600} y={116} width={110} height={50} rx={6} fill="none" stroke="var(--color-volt)" strokeWidth={1} />
        <text x={655} y={136} textAnchor="middle" fontSize={11} fontFamily={mono} fill="var(--color-volt)">
          AllReduce
        </text>
        <text x={655} y={152} textAnchor="middle" fontSize={10.5} fontFamily={mono} fill="var(--color-ink2)">
          Y = Y₀ + Y₁
        </text>
        <text x={655} y={180} textAnchor="middle" fontSize={9.5} fontFamily={mono} fill="var(--color-ink3)">
          ← 唯一通信点(g 算子)
        </text>

        {/* ── 底部：带宽对比条 ── */}
        <g transform="translate(10, 268)">
          <text fontSize={10.5} fontFamily={mono} fill="var(--color-ink3)" y={-6}>
            这次 AllReduce 走哪条路？
          </text>
          <text x={0} y={14} fontSize={10.5} fontFamily={mono} fill="var(--color-ink2)">
            NVLink (机内)
          </text>
          <rect x={120} y={5} width={560} height={12} rx={3} fill="var(--color-bg2)" />
          <rect x={120} y={5} width={560} height={12} rx={3} fill="var(--color-volt)" opacity={0.7} />
          <text x={686} y={14} fontSize={10} fontFamily={mono} fill="var(--color-volt)">
            900GB/s
          </text>
          <text x={0} y={38} fontSize={10.5} fontFamily={mono} fill="var(--color-ink2)">
            InfiniBand (跨机)
          </text>
          <rect x={120} y={29} width={560} height={12} rx={3} fill="var(--color-bg2)" />
          <rect x={120} y={29} width={31} height={12} rx={3} fill="var(--color-rose)" opacity={0.8} />
          <text x={158} y={38} fontSize={10} fontFamily={mono} fill="var(--color-rose)">
            ~50GB/s（差 18 倍）
          </text>
        </g>
      </svg>
    </Figure>
  )
}
