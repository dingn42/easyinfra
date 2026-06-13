import { Figure } from '@/components/ui'
import { useT } from '@/lib/i18n'

/** ── SEC5 静态示意图集 ── */

/** chunked prefill：长 prompt 切片混入 decode 批，别人 TPOT 不再被卡 */
export function ChunkedPrefillSVG() {
  const t = useT()
  // 上行：不切片 —— 一整块 prefill 占满若干迭代，他人 decode 停摆
  // 下行：切片 —— prefill 切成小块与 decode token 交错
  const iterW = 34
  const row = (y: number, cells: { c: string; t?: string; tc?: string }[]) =>
    cells.map((cell, i) => (
      <g key={i}>
        <rect x={120 + i * (iterW + 3)} y={y} width={iterW} height={16} rx={2} className={cell.c} />
        {cell.t && (
          <text
            x={120 + i * (iterW + 3) + iterW / 2}
            y={y + 11.5}
            textAnchor="middle"
            className={`fill-current ${cell.tc ?? 'text-bg'}`}
            fontSize={7.5}
            fontFamily="monospace"
          >
            {cell.t}
          </text>
        )}
      </g>
    ))
  return (
    <Figure caption={t(
      'Chunked prefill: slice a 2000-token prefill into several chunks and pack them into the same iterations as other requests’ decode tokens — a long prompt no longer stalls the neighbors’ TPOT.',
      'chunked prefill：把 2000-token 的 prefill 切成若干 chunk，与其他请求的 decode token 拼进同一个迭代 —— 长 prompt 不再让邻居的 TPOT 卡顿。',
    )}>
      <svg viewBox="0 0 640 132" className="w-full" role="img" aria-label={t('Chunked prefill comparison diagram', 'chunked prefill 对比示意')}>
        <text x={0} y={26} className="fill-current font-mono text-ink3" fontSize={9}>{t('unchunked', '不切片')}</text>
        {/* 一整块 prefill 横跨 3 个迭代 */}
        <rect x={120} y={14} width={iterW * 3 + 6} height={16} rx={2} className="fill-cyan/90" />
        <text x={120 + (iterW * 3 + 6) / 2} y={25.5} textAnchor="middle" className="fill-current text-bg" fontSize={7.5} fontFamily="monospace">
          PREFILL 2000 tok
        </text>
        {row(14, [
          { c: 'fill-none' },
          { c: 'fill-none' },
          { c: 'fill-none' },
          { c: 'fill-rose/30', t: t('stalled', '停摆'), tc: 'text-rose' },
          { c: 'fill-rose/30', t: t('stalled', '停摆'), tc: 'text-rose' },
          { c: 'fill-volt/80', t: 'decode' },
        ])}
        <text x={352} y={26} className="fill-current font-mono text-rose" fontSize={8}>{t('← neighbor TPOT spike', '← 邻居 TPOT 尖刺')}</text>

        <text x={0} y={76} className="fill-current font-mono text-ink3" fontSize={9}>{t('chunked mix', '切片混批')}</text>
        {row(64, [
          { c: 'fill-cyan/90', t: 'chunk' },
          { c: 'fill-volt/80', t: 'decode' },
          { c: 'fill-cyan/90', t: 'chunk' },
          { c: 'fill-volt/80', t: 'decode' },
          { c: 'fill-cyan/90', t: 'chunk' },
          { c: 'fill-volt/80', t: 'decode' },
        ])}
        <text x={352} y={76} className="fill-current font-mono text-volt" fontSize={8}>{t('← smoothly interleaved', '← 平滑交错')}</text>
        <text x={120} y={108} className="fill-current font-mono text-ink3" fontSize={8}>{t('→ each cell = one engine iteration; cyan = prefill chunk, volt = another request’s decode token', '→ 每格 = 引擎的一个迭代（iteration），cyan = prefill chunk，volt = 其他请求的 decode token')}</text>
      </svg>
    </Figure>
  )
}

/** 抢占：显存吃紧时换出/丢弃低优请求的 KV，腾出空间 */
export function PreemptionSVG() {
  const t = useT()
  return (
    <Figure caption={t(
      'Preemption: when the KV cache overflows memory, the scheduler evicts a request — either swapping its KV out to host memory wholesale, or simply discarding the KV and recomputing it later.',
      '抢占（preemption）：KV cache 把显存挤爆时，调度器把某个请求请出去 —— 要么整体换出（swap）到主机内存，要么直接丢弃 KV、回头重算（recompute）。',
    )}>
      <svg viewBox="0 0 640 120" className="w-full" role="img" aria-label={t('Preemption and swap diagram', '抢占与换出示意')}>
        {/* GPU 显存框 */}
        <rect x={10} y={14} width={300} height={88} rx={6} className="fill-bg2 stroke-line2" strokeWidth={1} />
        <text x={20} y={32} className="fill-current font-mono text-ink3" fontSize={9}>{t('GPU MEMORY · KV CACHE', 'GPU 显存 · KV CACHE')}</text>
        {[0, 1, 2].map((i) => (
          <g key={i}>
            <rect x={22 + i * 92} y={42} width={84} height={22} rx={3} className={i === 2 ? 'fill-rose/30 stroke-rose' : 'fill-amber/30 stroke-amber/60'} strokeWidth={1} />
            <text x={64 + i * 92} y={56.5} textAnchor="middle" className={`fill-current font-mono ${i === 2 ? 'text-rose' : 'text-amber'}`} fontSize={8}>
              {i === 2 ? t('REQ C ← preempted', 'REQ C ← 被抢占') : t(`REQ ${'AB'[i]} KV`, `REQ ${'AB'[i]} 的 KV`)}
            </text>
          </g>
        ))}
        <rect x={22} y={72} width={130} height={22} rx={3} className="fill-volt/20 stroke-volt/60" strokeWidth={1} strokeDasharray="4 3" />
        <text x={87} y={86.5} textAnchor="middle" className="fill-current font-mono text-volt" fontSize={8}>{t('freed space → new request', '腾出的空间 → 新请求')}</text>
        {/* 两条出路 */}
        <path d="M 314 50 C 360 38, 380 38, 420 44" fill="none" className="stroke-amber" strokeWidth={1.4} markerEnd="url(#pre-arrow)" />
        <path d="M 314 62 C 360 80, 380 84, 420 88" fill="none" className="stroke-rose" strokeWidth={1.4} markerEnd="url(#pre-arrow)" />
        <defs>
          <marker id="pre-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="fill-ink3" />
          </marker>
        </defs>
        <rect x={424} y={28} width={206} height={28} rx={4} className="fill-panel2 stroke-line2" strokeWidth={1} />
        <text x={527} y={46} textAnchor="middle" className="fill-current font-mono text-amber" fontSize={9}>{t('SWAP: KV out to host memory', 'SWAP：KV 换出到主机内存')}</text>
        <rect x={424} y={74} width={206} height={28} rx={4} className="fill-panel2 stroke-line2" strokeWidth={1} />
        <text x={527} y={92} textAnchor="middle" className="fill-current font-mono text-rose" fontSize={9}>{t('RECOMPUTE: discard, redo later', 'RECOMPUTE：丢弃，回头重算')}</text>
      </svg>
    </Figure>
  )
}

/** speculative decoding：小模型起草，大模型一次并行验证 */
export function SpeculativeSVG() {
  const t = useT()
  const tok = (x: number, y: number, cls: string, label?: string) => (
    <g>
      <rect x={x} y={y} width={26} height={18} rx={3} className={cls} />
      {label && (
        <text x={x + 13} y={y + 12.5} textAnchor="middle" className="fill-current text-bg" fontSize={8} fontFamily="monospace">
          {label}
        </text>
      )}
    </g>
  )
  return (
    <Figure caption={t(
      'Speculative decoding: the draft model serially drafts k tokens (cheap), then the target model verifies them in one parallel forward pass. Accept the whole matching prefix, void everything after the first mismatch — one "big-model time" buys back several tokens.',
      'speculative decoding：草稿模型串行起草 k 个 token（便宜），目标大模型一次前向并行验证。前缀命中的全收下，第一个不一致处之后作废 —— 一次「大模型时间」换回多个 token。',
    )}>
      <svg viewBox="0 0 640 150" className="w-full" role="img" aria-label={t('Speculative decoding flow diagram', 'speculative decoding 流程示意')}>
        {/* 草稿模型 */}
        <rect x={10} y={20} width={120} height={34} rx={5} className="fill-panel2 stroke-line2" strokeWidth={1} />
        <text x={70} y={36} textAnchor="middle" className="fill-current font-mono text-ink2" fontSize={9}>DRAFT 1B</text>
        <text x={70} y={48} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={7.5}>{t('serially guess k=4', '串行猜 k=4 个')}</text>
        {tok(160, 28, 'fill-ink3/50', 't1')}
        {tok(192, 28, 'fill-ink3/50', 't2')}
        {tok(224, 28, 'fill-ink3/50', 't3')}
        {tok(256, 28, 'fill-ink3/50', 't4')}
        <path d="M 296 37 L 330 37" className="stroke-ink3" strokeWidth={1.4} markerEnd="url(#sp-arrow)" />
        <defs>
          <marker id="sp-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="fill-ink3" />
          </marker>
        </defs>
        {/* 目标模型 */}
        <rect x={338} y={14} width={140} height={46} rx={5} className="fill-volt/12 stroke-volt/60" strokeWidth={1.2} />
        <text x={408} y={34} textAnchor="middle" className="fill-current font-mono text-volt" fontSize={9}>TARGET 70B</text>
        <text x={408} y={48} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={7.5}>{t('one forward, 4 verified in parallel', '一次前向，4 个并行验证')}</text>
        {/* 验证结果 */}
        {tok(160, 100, 'fill-volt', 't1✓')}
        {tok(192, 100, 'fill-volt', 't2✓')}
        {tok(224, 100, 'fill-volt', 't3✓')}
        {tok(256, 100, 'fill-rose', 't4✗')}
        {tok(288, 100, 'fill-cyan', "t4'")}
        <path d="M 408 64 C 408 84, 300 86, 290 96" fill="none" className="stroke-line2" strokeWidth={1.2} markerEnd="url(#sp-arrow)" />
        <text x={160} y={134} className="fill-current font-mono text-ink3" fontSize={8}>
          {t('3 hits + the big model’s own correction token: net 4 tokens in one step', '命中 3 个 + 大模型自己给出的修正 token：一步净赚 4 个 token')}
        </text>
        <text x={494} y={37} className="fill-current font-mono text-ink3" fontSize={8}>{t('← turns serial decode', '← 把串行 decode')}</text>
        <text x={494} y={49} className="fill-current font-mono text-ink3" fontSize={8}>{t('into parallel verification', '变成并行验证')}</text>
      </svg>
    </Figure>
  )
}

/** 推理栈全景：Router → Scheduler → Engine → Kernels */
export function ServingStackSVG() {
  const t = useT()
  const layers = [
    { id: 'router', name: t('ROUTER / GATEWAY', 'ROUTER / 网关'), desc: t('multi-replica load balancing · prefix-affinity routing · rate limiting', '多副本负载均衡 · prefix 亲和路由 · 限流'), tone: 'text-ink2', stroke: 'stroke-line2' },
    { id: 'scheduler', name: t('SCHEDULER', 'SCHEDULER / 调度器'), desc: t('continuous batching · chunked prefill · priority & preemption', '连续批处理 · chunked prefill · 优先级与抢占'), tone: 'text-volt', stroke: 'stroke-volt/60' },
    { id: 'engine', name: t('ENGINE', 'ENGINE / 执行引擎'), desc: t('paged KV cache management · prefix reuse · speculative', 'KV cache 分页管理 · prefix 复用 · speculative'), tone: 'text-cyan', stroke: 'stroke-cyan/60' },
    { id: 'kernels', name: t('KERNELS', 'KERNELS / 算子层'), desc: t('FlashAttention · Paged KV · fused GEMM', 'FlashAttention · Paged KV · 融合 GEMM'), tone: 'text-amber', stroke: 'stroke-amber/60' },
  ]
  return (
    <Figure caption={t(
      'A layered panorama of an inference serving stack: requests flow top-down, and each layer solves one class of problem. This chapter stars the middle two layers — the scheduler and the engine.',
      '一个推理服务栈的分层全景：请求自上而下，每一层解决一类问题。本章的主角是中间两层 —— 调度器与引擎。',
    )}>
      <svg viewBox="0 0 640 248" className="w-full" role="img" aria-label={t('Inference serving stack layer diagram', '推理服务栈分层示意')}>
        {layers.map((l, i) => (
          <g key={l.id}>
            <rect x={90} y={10 + i * 56} width={460} height={42} rx={6} className={`fill-panel2 ${l.stroke}`} strokeWidth={1.2} />
            <text x={108} y={28 + i * 56} className={`fill-current font-mono ${l.tone}`} fontSize={11}>
              {l.name}
            </text>
            <text x={108} y={44 + i * 56} className="fill-current text-ink3" fontSize={9} fontFamily="monospace">
              {l.desc}
            </text>
            {i < layers.length - 1 && (
              <path d={`M 320 ${52 + i * 56} L 320 ${66 + i * 56}`} className="stroke-ink3" strokeWidth={1.4} markerEnd="url(#st-arrow)" />
            )}
          </g>
        ))}
        <defs>
          <marker id="st-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="fill-ink3" />
          </marker>
        </defs>
        <text x={62} y={30} textAnchor="end" className="fill-current font-mono text-ink3" fontSize={8}>{t('request ↓', '请求 ↓')}</text>
        <text x={582} y={232} textAnchor="end" className="fill-current font-mono text-ink3" fontSize={8}>↓ GPU</text>
      </svg>
    </Figure>
  )
}
