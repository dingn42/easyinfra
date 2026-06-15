import { useState } from 'react'
import { Segmented, Slider, Stat, Toggle, Widget } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { C, rgba } from '@/lib/palette'
import { fmtBytes, fmtInt, fmtSI } from '@/lib/format'

interface ModelSpec {
  name: string
  L: number // 层数
  dModel: number
  nHeads: number
  dHead: number
  params: number // 参数量
  gqaNative: boolean // 官方权重是否原生 GQA(n_kv=8)
}

const MODELS: Record<string, ModelSpec> = {
  '7b': { name: 'LLaMA-7B', L: 32, dModel: 4096, nHeads: 32, dHead: 128, params: 6.74e9, gqaNative: false },
  '13b': { name: 'LLaMA-13B', L: 40, dModel: 5120, nHeads: 40, dHead: 128, params: 13.0e9, gqaNative: false },
  '70b': { name: 'LLaMA-70B', L: 80, dModel: 8192, nHeads: 64, dHead: 128, params: 69.0e9, gqaNative: true },
}

const HBM = 80 * 2 ** 30 // A100 80GB

/** LAB 02 KV Cache 计算器：逐项拆解 bytes 公式 + 权重/KV 堆叠条 vs 80GB 红线 */
export default function KVCalcLab() {
  const t = useT()
  const [modelKey, setModelKey] = useState<string>('7b')
  const [gqa, setGqa] = useState(false)
  const [seqExp, setSeqExp] = useState(12) // 2^12 = 4096
  const [batchExp, setBatchExp] = useState(3) // 2^3 = 8

  const m = MODELS[modelKey]
  const nKv = gqa ? 8 : m.nHeads
  const S = Math.round(2 ** seqExp)
  const B = 2 ** batchExp

  const perToken = 2 * m.L * nKv * m.dHead * 2 // bytes / token
  const perReq = perToken * S
  const totalKv = perReq * B
  const weights = m.params * 2 // fp16
  const total = weights + totalKv
  const maxBatch = weights >= HBM ? 0 : Math.floor((HBM - weights) / perReq)

  // 堆叠条：横轴比例尺
  const scale = Math.max(total, HBM) * 1.06
  const wPct = (weights / scale) * 100
  const kvPct = (totalKv / scale) * 100
  const linePct = (HBM / scale) * 100

  const reset = () => {
    setModelKey('7b')
    setGqa(false)
    setSeqExp(12)
    setBatchExp(3)
  }

  const factor = (v: string, label: string, tone = 'text-ink') => (
    <span className="flex flex-col items-center rounded-md border border-line bg-bg2 px-2.5 py-1.5">
      <span className={`font-mono text-sm tabular-nums ${tone}`}>{v}</span>
      <span className="microlabel mt-0.5 text-[9.5px]">{label}</span>
    </span>
  )
  const times = <span className="font-mono text-ink3">×</span>

  return (
    <Widget
      index={2}
      title={t('KV Cache calculator', 'KV Cache 计算器')}
      subtitle={t('How much context can one 80 GB card hold?', '一张 80GB 卡到底能塞多少上下文')}
      onReset={reset}
      footer={t(
        <>
          Flip GQA on and KV instantly shrinks by a factor of n_heads/n_kv (32→8 on 7B, a full 4×). This is why LLaMA-2
          70B and the entire LLaMA-3 line ship GQA natively: trade under 1% of quality for 4–8× the concurrency.
        </>,
        <>
          把 GQA 打开，KV 立刻缩小 n_heads/n_kv 倍（7B 上是 32→8，整整 4×）。
          这就是 LLaMA-2 70B、LLaMA-3 全系原生带 GQA 的原因：拿不到 1% 的质量损失，换回 4~8 倍的并发。
        </>,
      )}
    >
      <div className="mb-5 flex flex-wrap items-center gap-x-6 gap-y-3">
        <Segmented
          options={[
            { value: '7b', label: 'LLaMA-7B' },
            { value: '13b', label: 'LLaMA-13B' },
            { value: '70b', label: 'LLaMA-70B' },
          ]}
          value={modelKey}
          onChange={(v) => setModelKey(v)}
        />
        <Toggle label={<>GQA (n_kv: {m.nHeads} → 8)</>} checked={gqa} onChange={setGqa} />
        {m.gqaNative && !gqa && (
          <span className="font-mono text-[11px] text-amber">
            {t('Note: LLaMA-2 70B’s official weights are natively GQA (n_kv=8)', '注：LLaMA-2 70B 官方权重原生即 GQA（n_kv=8）')}
          </span>
        )}
      </div>

      <div className="mb-5 grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
        <Slider label={t('Sequence length S', '序列长度 S')} value={seqExp} min={10} max={17} step={0.5}
          onChange={setSeqExp} fmt={(v) => fmtSI(2 ** v, 1)} unit="token" />
        <Slider label={t('Concurrent batch', '并发 batch')} value={batchExp} min={0} max={8} step={1}
          onChange={setBatchExp} fmt={(v) => fmtInt(2 ** v)} unit={t('req', '请求')} />
      </div>

      {/* 公式逐项拆解 */}
      <div className="mb-5 rounded-md border border-line bg-panel2/40 p-3">
        <div className="microlabel mb-2">BYTES = 2 · L · n_kv · d_head · 2B · S · batch</div>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-2">
          {factor('2', t('K and V', 'K 和 V'))}
          {times}
          {factor(String(m.L), t('layers L', '层数 L'))}
          {times}
          {factor(String(nKv), t('KV heads n_kv', 'KV 头 n_kv'), gqa ? 'text-volt' : 'text-ink')}
          {times}
          {factor(String(m.dHead), 'd_head')}
          {times}
          {factor('2 B', 'fp16')}
          <span className="font-mono text-ink3">=</span>
          {factor(fmtBytes(perToken), t('per token', '每 token'), 'text-cyan')}
          {times}
          {factor(fmtSI(S, 1), 'S')}
          <span className="font-mono text-ink3">=</span>
          {factor(fmtBytes(perReq), t('per request', '每请求'), 'text-amber')}
          {times}
          {factor(fmtInt(B), 'batch')}
          <span className="font-mono text-ink3">=</span>
          {factor(fmtBytes(totalKv), t('total KV', 'KV 总量'), 'text-amber')}
        </div>
      </div>

      {/* 堆叠条 vs 80GB 红线 */}
      <div className="mb-5">
        <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
          <span className="microlabel">{t('MEMORY USE (fp16 WEIGHTS + KV CACHE)', '显存占用（fp16 权重 + KV CACHE）')}</span>
          <span className="flex items-center gap-3 font-mono text-[10.5px] text-ink3">
            <span><span className="mr-1 inline-block size-2 rounded-sm bg-cyan/70" />{t('weights', '权重')}</span>
            <span><span className="mr-1 inline-block size-2 rounded-sm bg-amber/80" />KV</span>
            <span><span className="mr-1 inline-block h-2.5 w-px bg-rose" /> A100 80GB</span>
          </span>
        </div>
        <div className="relative h-9 overflow-hidden rounded-md border border-line bg-bg2">
          <div className="absolute inset-y-0 left-0 bg-cyan/55" style={{ width: `${wPct}%` }} />
          <div className="absolute inset-y-0 bg-amber/70 transition-[width,left] duration-150"
            style={{ left: `${wPct}%`, width: `${kvPct}%` }} />
          <div className="absolute inset-y-0 w-px bg-rose" style={{ left: `${linePct}%`, boxShadow: `0 0 8px ${rgba(C.rose, 0.8)}` }} />
          <span className="absolute top-0.5 -translate-x-full pr-1.5 font-mono text-[10px] text-rose" style={{ left: `${linePct}%` }}>
            80GB
          </span>
        </div>
        <div className="mt-1.5 font-mono text-[11px] tabular-nums text-ink3">
          {t('weights', '权重')} <span className="text-cyan">{fmtBytes(weights)}</span>
          {' + '}KV <span className="text-amber">{fmtBytes(totalKv)}</span>
          {' = '}<span className={total > HBM ? 'text-rose' : 'text-ink'}>{fmtBytes(total)}</span>
          {total > HBM && <span className="text-rose">{t(` (over by ${fmtBytes(total - HBM)}, OOM)`, `（超出 ${fmtBytes(total - HBM)}，OOM）`)}</span>}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
        <Stat label={t('KV PER TOKEN', '每 TOKEN KV')} value={fmtBytes(perToken)} tone="cyan" size="sm" />
        <Stat label={t('KV PER REQUEST', '单请求 KV')} value={fmtBytes(perReq)} tone="amber" size="sm" />
        <Stat label={t('KV / WEIGHTS RATIO', 'KV / 权重之比')} value={`${(totalKv / weights).toFixed(2)}×`} size="sm" />
        <Stat label={t('MAX CONCURRENCY @ 80GB', '80GB 最大并发')}
          value={maxBatch > 0 ? fmtInt(maxBatch) : '0'}
          unit={maxBatch > 0 ? t('req', '请求') : t('weights don’t even fit', '权重都放不下')}
          tone={maxBatch > 0 ? 'volt' : 'rose'} size="sm" />
      </div>
    </Widget>
  )
}
