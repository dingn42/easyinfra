import { useMemo, useState } from 'react'
import { Figure, Segmented } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { fmtSI } from '@/lib/format'

/** ── SEC2 交互插图：吞吐 vs batch、TPOT vs batch，切换硬件看拐点移动 ──
 * 模型：13B 参数 FP16（权重 26 GB，三张卡都装得下）。
 * 每步 decode：读权重 t_mem = bytes/BW；算 B 个 token t_comp = B·2P/FLOPS。
 * 单步时延用 p-范数平滑 max(t_mem, B·t_comp)，更接近实测的圆滑拐点。
 */

const P = 13e9
const BYTES = 2 * P

const HW = {
  A100: { flops: 312e12, bw: 1.9e12, label: 'A100' },
  H100: { flops: 989e12, bw: 3.35e12, label: 'H100' },
  H200: { flops: 989e12, bw: 4.8e12, label: 'H200' },
} as const

type HwKey = keyof typeof HW

const BMAX = 512
const LOG_BMAX = Math.log2(BMAX)

function stepMs(b: number, hw: { flops: number; bw: number }): number {
  const tMem = (BYTES / hw.bw) * 1000
  const tComp = ((b * 2 * P) / hw.flops) * 1000
  const p = 3
  return Math.pow(Math.pow(tMem, p) + Math.pow(tComp, p), 1 / p)
}

export function BatchingCurves() {
  const t = useT()
  const [hwKey, setHwKey] = useState<HwKey>('H100')
  const hw = HW[hwKey]

  const ridge = (hw.flops / hw.bw) // = B*：t_mem == B·t_comp 的批大小
  const maxTput = 1000 / ((2 * P / hw.flops) * 1000) // 1/t_comp，tok/s 上限
  const tputCeil = 42000 // 固定坐标，便于跨硬件对比
  const tpotCeil = 50 // ms

  const pts = useMemo(() => {
    const arr: { b: number; tput: number; tpot: number }[] = []
    for (let i = 0; i <= 96; i++) {
      const b = Math.pow(2, (i / 96) * LOG_BMAX)
      const s = stepMs(b, hw)
      arr.push({ b, tput: (b / s) * 1000, tpot: s })
    }
    return arr
  }, [hw])

  // 坐标映射（两幅图共用 x：log2(B) → px）
  const W = 320, H = 190, L = 44, R = 10, T = 16, B0 = 36
  const x = (b: number) => L + (Math.log2(b) / LOG_BMAX) * (W - L - R)
  const yT = (v: number) => H - B0 - (Math.min(v, tputCeil) / tputCeil) * (H - T - B0)
  const yP = (v: number) => H - B0 - (Math.min(v, tpotCeil) / tpotCeil) * (H - T - B0)

  const tputPath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.b).toFixed(1)},${yT(p.tput).toFixed(1)}`).join(' ')
  const tpotPath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.b).toFixed(1)},${yP(p.tpot).toFixed(1)}`).join(' ')
  const ridgeX = x(Math.min(ridge, BMAX))
  const xticks = [1, 8, 64, 512]

  const axis = (yFn: (v: number) => number, yMax: number, n: number) => (
    <>
      <line x1={L} y1={H - B0} x2={W - R} y2={H - B0} className="stroke-line2" strokeWidth={1} />
      <line x1={L} y1={T} x2={L} y2={H - B0} className="stroke-line2" strokeWidth={1} />
      {xticks.map((b) => (
        <g key={b}>
          <line x1={x(b)} y1={H - B0} x2={x(b)} y2={H - B0 + 3} className="stroke-line2" strokeWidth={1} />
          <text x={x(b)} y={H - B0 + 13} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={8}>{b}</text>
        </g>
      ))}
      {Array.from({ length: n + 1 }, (_, i) => (yMax / n) * i).map((v) => (
        <text key={v} x={L - 5} y={yFn(v) + 3} textAnchor="end" className="fill-current font-mono text-ink3" fontSize={8}>
          {fmtSI(v, 0)}
        </text>
      ))}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="fill-current font-mono text-ink3" fontSize={8}>
        batch size B (log)
      </text>
    </>
  )

  return (
    <Figure
      caption={t(
        <>
          decode per-step latency ≈ max(weight read, compute). At small B the weight read dominates, so
          throughput rises linearly with B while TPOT barely moves. Once B passes the ridge (≈ compute/bandwidth
          ratio) compute becomes the bottleneck, throughput saturates, and TPOT climbs steeply. Estimated for a
          13B FP16 model.
        </>,
        <>
          decode 单步时延 ≈ max(权重读取, 计算)。B 小时读权重独大，吞吐随 B 线性涨、TPOT 几乎不变；
          B 越过 ridge（≈ 算力/带宽比）后计算成了瓶颈，吞吐饱和、TPOT 开始陡增。按 13B FP16 模型估算。
        </>,
      )}
    >
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="microlabel">{t('HARDWARE', '硬件')}</span>
        <Segmented
          options={(Object.keys(HW) as HwKey[]).map((k) => ({ value: k, label: HW[k].label }))}
          value={hwKey}
          onChange={setHwKey}
        />
        <span className="ml-auto font-mono text-[11px] text-ink3">
          ridge B* ≈ <span className="text-volt">{Math.round(ridge)}</span>
          {' '}
          {t('throughput ceiling', '吞吐上限')} ≈ <span className="text-volt">{fmtSI(maxTput, 0)}</span> tok/s
        </span>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {/* 吞吐 vs B */}
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={t('Throughput rises linearly with batch, then saturates', '吞吐随 batch 先线性后饱和')}>
          <text x={L} y={10} className="fill-current font-mono text-ink2" fontSize={9}>{t('throughput tok/s', '吞吐 tok/s')}</text>
          {axis(yT, tputCeil, 3)}
          <line x1={ridgeX} y1={T} x2={ridgeX} y2={H - B0} className="stroke-volt/50" strokeWidth={1} strokeDasharray="4 3" />
          <text
            x={ridgeX > W - 70 ? ridgeX - 4 : ridgeX + 4}
            y={T + 9}
            textAnchor={ridgeX > W - 70 ? 'end' : 'start'}
            className="fill-current font-mono text-volt"
            fontSize={8}
          >
            B*≈{Math.round(ridge)}
          </text>
          <path d={tputPath} fill="none" className="stroke-volt" strokeWidth={2} />
          <text x={x(2)} y={yT(tputCeil * 0.52)} className="fill-current font-mono text-ink3" fontSize={8}>{t('linear region: free throughput', '线性区：免费的吞吐')}</text>
          <text x={x(80)} y={yT(maxTput) + (maxTput < tputCeil * 0.5 ? -8 : 14)} className="fill-current font-mono text-ink3" fontSize={8}>{t('saturated', '饱和区')}</text>
        </svg>
        {/* TPOT vs B */}
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={t('TPOT is flat before the knee, then climbs steeply', 'TPOT 在拐点前平坦，之后陡增')}>
          <text x={L} y={10} className="fill-current font-mono text-ink2" fontSize={9}>TPOT ms / token</text>
          {axis(yP, tpotCeil, 5)}
          <line x1={ridgeX} y1={T} x2={ridgeX} y2={H - B0} className="stroke-volt/50" strokeWidth={1} strokeDasharray="4 3" />
          <text
            x={ridgeX > W - 70 ? ridgeX - 4 : ridgeX + 4}
            y={T + 9}
            textAnchor={ridgeX > W - 70 ? 'end' : 'start'}
            className="fill-current font-mono text-volt"
            fontSize={8}
          >
            B*≈{Math.round(ridge)}
          </text>
          <path d={tpotPath} fill="none" className="stroke-amber" strokeWidth={2} />
          <text x={x(2.4)} y={yP(stepMs(1, hw)) - 7} className="fill-current font-mono text-ink3" fontSize={8}>
            {t('≈ weight-read time', '≈ 权重读取时间')} {stepMs(1, hw).toFixed(0)}ms
          </text>
          <text x={x(110)} y={yP(tpotCeil * 0.72)} className="fill-current font-mono text-amber" fontSize={8}>{t('latency degrades', '延迟开始变差')}</text>
        </svg>
      </div>
    </Figure>
  )
}
