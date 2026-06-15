import { useState } from 'react'
import { Widget, Toggle } from '@/components/ui'
import { useRafLoop, useReducedMotion } from '@/lib/hooks'
import { pick, useLocale, useT, type Loc } from '@/lib/i18n'

/* ── LAB 01：一层的数据流 ─────────────────────────────────────────
   LLaMA-7B 单层 decoder 的 SVG 流程图。hover/点击节点弹出信息卡
   （张量形状 / 参数量 / 单 token FLOPs），可开关沿管线流动的动画。 */

type Kind = 'gemm' | 'attn' | 'norm' | 'elem' | 'io'

interface FlowNode {
  id: string
  label: Loc
  tag: Loc
  kind: Kind
  shape: Loc
  params: Loc
  flops: Loc
  desc: Loc
}

const KIND_COLOR: Record<Kind, string> = {
  gemm: 'var(--color-volt)',
  attn: 'var(--color-amber)',
  norm: 'var(--color-cyan)',
  elem: 'var(--color-violet)',
  io: 'var(--color-line2)',
}

const KIND_LABEL: Record<Kind, Loc> = {
  gemm: { en: 'GEMM projection', zh: 'GEMM 投影' },
  attn: { en: 'attention core', zh: 'attention 核心' },
  norm: { en: 'normalization', zh: '归一化' },
  elem: { en: 'elementwise', zh: '逐元素' },
  io: { en: 'residual stream', zh: '残差流' },
}

/* 信息卡数据：以 decode 单 token、上下文 S=2048 计 */
const ROW1_NODES: FlowNode[] = [
  {
    id: 'input', label: { en: 'input x', zh: '输入 x' }, tag: { en: 'residual', zh: 'residual' }, kind: 'io',
    shape: { en: 'x: [1, 4096]', zh: 'x: [1, 4096]' },
    params: { en: '—', zh: '—' }, flops: { en: '—', zh: '—' },
    desc: {
      en: 'A token vector on the residual stream. The entire job of this layer is to compute a delta and add it back to this vector.',
      zh: '残差流（residual stream）上的一个 token 向量。整层的全部工作，就是算出一个增量加回这个向量。',
    },
  },
  {
    id: 'norm1', label: { en: 'RMSNorm₁', zh: 'RMSNorm₁' }, tag: { en: 'pre-norm', zh: 'pre-norm' }, kind: 'norm',
    shape: { en: '[1, 4096] → [1, 4096]', zh: '[1, 4096] → [1, 4096]' },
    params: { en: '4,096 (γ)', zh: '4,096（γ）' }, flops: { en: '≈ 16 K', zh: '≈ 16 K' },
    desc: {
      en: 'RMSNorm: divide by the vector\'s RMS, then scale by a learnable γ. Its cost is proportional only to d, so it\'s negligible on the ledger.',
      zh: '均方根归一化（RMSNorm）：除以向量的 RMS 再乘可学习缩放 γ。计算量只与 d 成正比，在账本上可以忽略。',
    },
  },
  {
    id: 'qkv', label: { en: 'QKV projection', zh: 'QKV 投影' }, tag: { en: '3 × GEMV', zh: '3 × GEMV' }, kind: 'gemm',
    shape: {
      en: '[1,4096] × three [4096,4096] → Q/K/V each [32 heads, 128]',
      zh: '[1,4096] × 三个 [4096,4096] → Q/K/V 各 [32 head, 128]',
    },
    params: { en: '50.3 M (3·d²)', zh: '50.3 M（3·d²）' }, flops: { en: '100.7 M (2·3·d²)', zh: '100.7 M（2·3·d²）' },
    desc: {
      en: 'Three d×d matrices project x into Q, K, V, each then reshaped into 32 heads × 128 dims. This is the most parameter-heavy step in attention.',
      zh: '三个 d×d 矩阵把 x 投影成 Q、K、V，再各自重排成 32 个 head × 128 维。这是 attention 部分参数最多的一笔。',
    },
  },
  {
    id: 'rope', label: { en: 'RoPE', zh: 'RoPE' }, tag: { en: 'rotary', zh: 'rotary' }, kind: 'elem',
    shape: { en: 'Q, K: [32, 128] rotated in place', zh: 'Q、K: [32, 128] 原位旋转' },
    params: { en: '0', zh: '0' }, flops: { en: '≈ 33 K', zh: '≈ 33 K' },
    desc: {
      en: 'Rotary Position Embedding: encode position as a rotation angle on a 2D plane, applied to Q and K. Zero parameters, near-zero cost.',
      zh: '旋转位置编码（Rotary Position Embedding）：把位置编码成二维平面上的旋转角，作用在 Q、K 上。零参数、近乎零开销。',
    },
  },
  {
    id: 'attn', label: { en: 'Attention', zh: 'Attention' }, tag: { en: 'QKᵀ → softmax → AV', zh: 'QKᵀ → softmax → AV' }, kind: 'attn',
    shape: { en: 'score: [32, S=2048], output [32, 128] → [1, 4096]', zh: 'score: [32, S=2048]，输出 [32, 128] → [1, 4096]' },
    params: { en: '0', zh: '0' }, flops: { en: '≈ 33.6 M (4·S·d)', zh: '≈ 33.6 M（4·S·d）' },
    desc: {
      en: 'Each head uses its own Q to dot against 2048 cached K vectors, softmax, then weighted-sum the V. The only step in the layer that grows with context length S.',
      zh: '每个 head 用自己的 Q 对缓存里 2048 个 K 做点积、softmax、再加权求和 V。整层唯一随上下文长度 S 增长的环节。',
    },
  },
  {
    id: 'oproj', label: { en: 'O projection', zh: 'O 投影' }, tag: { en: 'GEMV', zh: 'GEMV' }, kind: 'gemm',
    shape: { en: '[1, 4096] × [4096, 4096]', zh: '[1, 4096] × [4096, 4096]' },
    params: { en: '16.8 M (d²)', zh: '16.8 M（d²）' }, flops: { en: '33.6 M (2·d²)', zh: '33.6 M（2·d²）' },
    desc: {
      en: 'Mix the concatenated 32-head vector one more time. The last matrix before the attention sublayer writes back to the residual stream.',
      zh: '把 32 个 head 拼回的向量再混合一次。attention 子层写回残差流之前的最后一个矩阵。',
    },
  },
  {
    id: 'add1', label: { en: '⊕ residual', zh: '⊕ 残差' }, tag: { en: 'add', zh: 'add' }, kind: 'elem',
    shape: { en: '[1, 4096] + [1, 4096]', zh: '[1, 4096] + [1, 4096]' },
    params: { en: '0', zh: '0' }, flops: { en: '4 K', zh: '4 K' },
    desc: {
      en: 'Add the attention output back to the input. The residual connection gives gradients a path through a 32-layer network.',
      zh: 'attention 的输出加回输入。残差连接（residual connection）让 32 层网络的梯度有路可走。',
    },
  },
]

const ROW2_NODES: FlowNode[] = [
  {
    id: 'norm2', label: { en: 'RMSNorm₂', zh: 'RMSNorm₂' }, tag: { en: 'pre-norm', zh: 'pre-norm' }, kind: 'norm',
    shape: { en: '[1, 4096] → [1, 4096]', zh: '[1, 4096] → [1, 4096]' },
    params: { en: '4,096 (γ)', zh: '4,096（γ）' }, flops: { en: '≈ 16 K', zh: '≈ 16 K' },
    desc: {
      en: 'Normalize once more before the MLP. Each layer has two RMSNorms — 8192 parameters total, which rounds to zero against the layer\'s 200 million.',
      zh: '进 MLP 前再归一化一次。每层两个 RMSNorm，合计 8192 个参数——在 2 亿的层参数里四舍五入等于零。',
    },
  },
  {
    id: 'gateup', label: { en: 'gate / up projection', zh: 'gate / up 投影' }, tag: { en: '2 × GEMV', zh: '2 × GEMV' }, kind: 'gemm',
    shape: { en: '[1,4096] × two [4096,11008] → two paths [1, 11008]', zh: '[1,4096] × 两个 [4096,11008] → 两路 [1, 11008]' },
    params: { en: '90.2 M (2·d·d_ff)', zh: '90.2 M（2·d·d_ff）' }, flops: { en: '180.4 M', zh: '180.4 M' },
    desc: {
      en: 'SwiGLU\'s two up-projection paths — the single largest step in the layer for both parameters and FLOPs, more than the entire attention sublayer combined.',
      zh: 'SwiGLU 的两路升维投影，单层里参数和 FLOPs 都最大的一笔——比整个 attention 子层加起来还多。',
    },
  },
  {
    id: 'act', label: { en: 'SiLU · ⊙', zh: 'SiLU · ⊙' }, tag: { en: 'gating', zh: 'gating' }, kind: 'elem',
    shape: { en: '[1, 11008] elementwise', zh: '[1, 11008] 逐元素' },
    params: { en: '0', zh: '0' }, flops: { en: '≈ 55 K', zh: '≈ 55 K' },
    desc: {
      en: 'SiLU(gate) ⊙ up: elementwise gating. Its cost is proportional to d_ff and negligible against the two GEMMs beside it.',
      zh: 'SiLU(gate) ⊙ up：逐元素门控。开销与 d_ff 成正比，和旁边两个 GEMM 比可以忽略。',
    },
  },
  {
    id: 'down', label: { en: 'down projection', zh: 'down 投影' }, tag: { en: 'GEMV', zh: 'GEMV' }, kind: 'gemm',
    shape: { en: '[1, 11008] × [11008, 4096]', zh: '[1, 11008] × [11008, 4096]' },
    params: { en: '45.1 M (d·d_ff)', zh: '45.1 M（d·d_ff）' }, flops: { en: '90.2 M', zh: '90.2 M' },
    desc: {
      en: 'Project back down to d=4096. With this, the MLP\'s three matrices total 3·d·d_ff ≈ 135 million parameters.',
      zh: '降维回 d=4096。至此 MLP 三个矩阵合计 3·d·d_ff ≈ 1.35 亿参数。',
    },
  },
  {
    id: 'add2', label: { en: '⊕ residual', zh: '⊕ 残差' }, tag: { en: 'add', zh: 'add' }, kind: 'elem',
    shape: { en: '[1, 4096] + [1, 4096]', zh: '[1, 4096] + [1, 4096]' },
    params: { en: '0', zh: '0' }, flops: { en: '4 K', zh: '4 K' },
    desc: { en: 'Add the MLP output back to the residual stream. The layer is done.', zh: 'MLP 的输出加回残差流。一层结束。' },
  },
  {
    id: 'out', label: { en: 'output', zh: '输出' }, tag: { en: '→ next layer', zh: '→ 下一层' }, kind: 'io',
    shape: { en: '[1, 4096]', zh: '[1, 4096]' },
    params: { en: '—', zh: '—' }, flops: { en: '—', zh: '—' },
    desc: {
      en: 'The same structure repeats 32 times, then a final RMSNorm and a [4096, 32000] vocabulary projection produce the logits for the next token.',
      zh: '同样的结构重复 32 次，最后过一个 RMSNorm 和 [4096, 32000] 的词表投影，得到下一个 token 的 logits。',
    },
  },
]

const ALL_NODES = [...ROW1_NODES, ...ROW2_NODES]

/* ── 几何布局（viewBox 坐标系） ── */
const VB_W = 900
const VB_H = 348
const NODE_H = 50
const R1_Y = 36
const R2_Y = 230
const LANE_SKIP1 = 12 // 残差跳线 1（行上方）
const LANE_MAIN = 152 // 行间主连接
const LANE_SKIP2 = 192 // 残差跳线 2（行间下方）

interface Rect {
  x: number
  y: number
  w: number
  h: number
  cx: number
  cy: number
}

function rowRects(count: number, y: number): Rect[] {
  const m = 10
  const gap = 16
  const w = (VB_W - m * 2 - gap * (count - 1)) / count
  return Array.from({ length: count }, (_, i) => {
    const x = m + i * (w + gap)
    return { x, y, w, h: NODE_H, cx: x + w / 2, cy: y + NODE_H / 2 }
  })
}

const R1 = rowRects(ROW1_NODES.length, R1_Y)
const R2 = rowRects(ROW2_NODES.length, R2_Y)

/* 动画小球的折线路径：行 1 → 折返 → 行 2 */
const WAYPOINTS: [number, number][] = [
  ...R1.map((r) => [r.cx, r.cy] as [number, number]),
  [R1[R1.length - 1].cx, LANE_MAIN],
  [R2[0].cx, LANE_MAIN],
  ...R2.map((r) => [r.cx, r.cy] as [number, number]),
]
const SEG_LEN: number[] = []
let PATH_LEN = 0
for (let i = 0; i < WAYPOINTS.length - 1; i++) {
  const dx = WAYPOINTS[i + 1][0] - WAYPOINTS[i][0]
  const dy = WAYPOINTS[i + 1][1] - WAYPOINTS[i][1]
  const len = Math.hypot(dx, dy)
  SEG_LEN.push(len)
  PATH_LEN += len
}

function pointAt(dist: number): [number, number] {
  let d = ((dist % PATH_LEN) + PATH_LEN) % PATH_LEN
  for (let i = 0; i < SEG_LEN.length; i++) {
    if (d <= SEG_LEN[i]) {
      const t = SEG_LEN[i] === 0 ? 0 : d / SEG_LEN[i]
      const [x0, y0] = WAYPOINTS[i]
      const [x1, y1] = WAYPOINTS[i + 1]
      return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]
    }
    d -= SEG_LEN[i]
  }
  return WAYPOINTS[WAYPOINTS.length - 1]
}

const DOT_COUNT = 3

function NodeBox({
  node,
  rect,
  active,
  onSelect,
  onHover,
}: {
  node: FlowNode
  rect: Rect
  active: boolean
  onSelect: () => void
  onHover: (id: string | null) => void
}) {
  const { lang } = useLocale()
  const color = KIND_COLOR[node.kind]
  return (
    <g
      className="cursor-pointer select-none"
      onClick={onSelect}
      onMouseEnter={() => onHover(node.id)}
      onMouseLeave={() => onHover(null)}
    >
      <rect
        x={rect.x}
        y={rect.y}
        width={rect.w}
        height={rect.h}
        rx={6}
        fill={active ? 'var(--color-panel2)' : 'var(--color-panel)'}
        stroke={active ? color : 'var(--color-line2)'}
        strokeWidth={active ? 1.6 : 1}
      />
      {/* 顶部类别色条 */}
      <rect x={rect.x + 5} y={rect.y + 4} width={rect.w - 10} height={3} rx={1.5} fill={color} opacity={active ? 0.95 : 0.55} />
      <text
        x={rect.cx}
        y={rect.y + 26}
        textAnchor="middle"
        fontSize={14}
        fontWeight={600}
        fill="currentColor"
        className={active ? 'text-ink' : 'text-text'}
      >
        {pick(node.label, lang)}
      </text>
      <text
        x={rect.cx}
        y={rect.y + 42}
        textAnchor="middle"
        fontSize={10}
        fill="currentColor"
        className="font-mono text-ink3"
      >
        {pick(node.tag, lang)}
      </text>
    </g>
  )
}

export function LayerFlowLab() {
  const t = useT()
  const { lang } = useLocale()
  const reduced = useReducedMotion()
  const [anim, setAnim] = useState(true)
  const [sel, setSel] = useState('qkv')
  const [hover, setHover] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  const playing = anim && !reduced
  useRafLoop((dt) => setTick((p) => (p + dt * 0.18) % PATH_LEN), playing)

  const activeId = hover ?? sel
  const active = ALL_NODES.find((n) => n.id === activeId) ?? ALL_NODES[2]

  const reset = () => {
    setSel('qkv')
    setHover(null)
    setTick(0)
    setAnim(true)
  }

  return (
    <Widget
      index={1}
      title={t('The data flow of one layer', '一层的数据流')}
      subtitle={t('LLaMA-7B · d=4096 · 32 heads × 128 · d_ff=11008 · 32 layers', 'LLaMA-7B · d=4096 · 32 heads × 128 · d_ff=11008 · 共 32 层')}
      wide
      onReset={reset}
      footer={t(
        <>
          Hover / click a node to see its tensor shape, parameter count, and single-token FLOPs (computed for decode,
          context S=2048). The top color bar marks the node category. Notice that almost all of the layer&apos;s
          parameters and FLOPs land on <span className="text-volt">GEMM projections</span>, while the{' '}
          <span className="text-amber">attention core</span> is the only step that grows with S.
        </>,
        <>
          hover / 点击节点查看张量形状、参数量与单 token FLOPs（按 decode、上下文 S=2048 计）。
          顶部色条标记节点类别。注意整层的参数和 FLOPs 几乎全部落在 <span className="text-volt">GEMM 投影</span> 上，
          <span className="text-amber"> attention 核心</span>是唯一随 S 增长的环节。
        </>,
      )}
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {(Object.keys(KIND_COLOR) as Kind[]).map((k) => (
            <span key={k} className="inline-flex items-center gap-1.5 font-mono text-[11px] text-ink2">
              <span className="inline-block size-2 rounded-[2px]" style={{ background: KIND_COLOR[k] }} />
              {pick(KIND_LABEL[k], lang)}
            </span>
          ))}
        </div>
        <Toggle
          label={reduced ? t('Animation (off per system preference)', '动画（已按系统偏好关闭）') : t('Animation', '动画')}
          checked={playing}
          onChange={(v) => setAnim(v && !reduced)}
        />
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={t('Data flow of one LLaMA decoder layer', 'LLaMA 一层 decoder 的数据流图')}>
        <defs>
          <marker id="ch7-arrow" viewBox="0 0 8 8" refX={7} refY={4} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
            <path d="M0,0.5 L7.5,4 L0,7.5 Z" fill="var(--color-line2)" />
          </marker>
        </defs>

        {/* 行内箭头 */}
        {R1.slice(0, -1).map((r, i) => (
          <line key={`a1-${i}`} x1={r.x + r.w} y1={r.cy} x2={R1[i + 1].x - 1.5} y2={r.cy} stroke="var(--color-line2)" strokeWidth={1.2} markerEnd="url(#ch7-arrow)" />
        ))}
        {R2.slice(0, -1).map((r, i) => (
          <line key={`a2-${i}`} x1={r.x + r.w} y1={r.cy} x2={R2[i + 1].x - 1.5} y2={r.cy} stroke="var(--color-line2)" strokeWidth={1.2} markerEnd="url(#ch7-arrow)" />
        ))}
        {/* 行间折返连接 */}
        <path
          d={`M ${R1[R1.length - 1].cx} ${R1_Y + NODE_H} V ${LANE_MAIN} H ${R2[0].cx} V ${R2_Y - 2}`}
          fill="none"
          stroke="var(--color-line2)"
          strokeWidth={1.2}
          markerEnd="url(#ch7-arrow)"
        />
        {/* 残差跳线 1：输入 x → ⊕ */}
        <path
          d={`M ${R1[0].cx} ${R1_Y} V ${LANE_SKIP1} H ${R1[6].cx} V ${R1_Y - 2}`}
          fill="none"
          stroke="var(--color-cyan)"
          strokeWidth={1}
          strokeDasharray="4 4"
          opacity={0.6}
          markerEnd="url(#ch7-arrow)"
        />
        <text x={(R1[0].cx + R1[6].cx) / 2} y={LANE_SKIP1 - 2} textAnchor="middle" fontSize={10} fill="currentColor" className="font-mono text-ink3">
          residual
        </text>
        {/* 残差跳线 2：attention 输出 → MLP 后的 ⊕ */}
        <path
          d={`M ${R2[0].cx} ${LANE_SKIP2} H ${R2[4].cx} V ${R2_Y - 2}`}
          fill="none"
          stroke="var(--color-cyan)"
          strokeWidth={1}
          strokeDasharray="4 4"
          opacity={0.6}
          markerEnd="url(#ch7-arrow)"
        />
        <circle cx={R2[0].cx} cy={LANE_SKIP2} r={2.5} fill="var(--color-cyan)" opacity={0.6} />
        <text x={(R2[0].cx + R2[4].cx) / 2} y={LANE_SKIP2 - 4} textAnchor="middle" fontSize={10} fill="currentColor" className="font-mono text-ink3">
          residual
        </text>

        {/* 节点 */}
        {ROW1_NODES.map((n, i) => (
          <NodeBox key={n.id} node={n} rect={R1[i]} active={activeId === n.id} onSelect={() => setSel(n.id)} onHover={setHover} />
        ))}
        {ROW2_NODES.map((n, i) => (
          <NodeBox key={n.id} node={n} rect={R2[i]} active={activeId === n.id} onSelect={() => setSel(n.id)} onHover={setHover} />
        ))}

        {/* 流动的小方块 */}
        {playing &&
          Array.from({ length: DOT_COUNT }, (_, k) => {
            const [x, y] = pointAt(tick + (k * PATH_LEN) / DOT_COUNT)
            return (
              <g key={k} transform={`translate(${x}, ${y})`} pointerEvents="none">
                <rect x={-5} y={-5} width={10} height={10} rx={2} fill="var(--color-volt)" opacity={0.9} />
                <rect x={-5} y={-5} width={10} height={10} rx={2} fill="none" stroke="var(--color-volt)" opacity={0.35} strokeWidth={5} />
              </g>
            )
          })}
      </svg>

      {/* 信息卡 */}
      <div className="mt-3 rounded-md border border-line bg-bg2 p-4">
        <div className="mb-2.5 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="inline-flex items-center gap-2 text-[15px] font-medium text-ink">
            <span className="inline-block size-2 rounded-[2px]" style={{ background: KIND_COLOR[active.kind] }} />
            {pick(active.label, lang)}
          </span>
          <span className="font-mono text-[11px] text-ink3">{pick(active.tag, lang)}</span>
        </div>
        <div className="grid gap-x-6 gap-y-3 sm:grid-cols-3">
          <div>
            <div className="microlabel mb-1">{t('tensor shape', '张量形状')}</div>
            <div className="font-mono text-[12.5px] leading-relaxed text-cyan">{pick(active.shape, lang)}</div>
          </div>
          <div>
            <div className="microlabel mb-1">{t('parameters', '参数量')}</div>
            <div className="font-mono text-[15px] tabular-nums text-volt">{pick(active.params, lang)}</div>
          </div>
          <div>
            <div className="microlabel mb-1">{t('FLOPs / token', '单 token FLOPs')}</div>
            <div className="font-mono text-[15px] tabular-nums text-amber">{pick(active.flops, lang)}</div>
          </div>
        </div>
        <p className="mt-3 text-[13.5px] leading-[1.9] text-text">{pick(active.desc, lang)}</p>
      </div>

      {/* 单层合计 */}
      <div className="mt-3 flex flex-wrap items-baseline gap-x-8 gap-y-2 rounded-md border border-line bg-panel2/40 px-4 py-2.5 font-mono text-[12.5px] text-ink2">
        <span>
          {t('layer total ', '本层合计 ')}
          <span className="tabular-nums text-volt">202.4 M</span>
          {t(' params', ' 参数')}
        </span>
        <span>
          ≈ <span className="tabular-nums text-amber">438.4 M</span>
          {t(' FLOPs / token (S=2048)', ' FLOPs / token（S=2048）')}
        </span>
        <span className="text-ink3">{t('× 32 layers + embedding ⇒ 6.74 B params', '× 32 层 + embedding ⇒ 6.74 B 参数')}</span>
      </div>
    </Widget>
  )
}
