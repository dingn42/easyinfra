import { useState } from 'react'
import { MathTex, Segmented, Slider, Stat, Widget } from '@/components/ui'
import { fmtSI } from '@/lib/format'
import { HARDWARE, fmtAI, fmtTime, hwById, ridgeOf } from './data'

/** LAB 01 算术强度计算器 */

type OpId = 'vecadd' | 'saxpy' | 'gemv' | 'gemm' | 'attn'
type Prec = 'half' | 'fp32'

const DEF = {
  op: 'gemm' as OpId,
  hw: 'a100',
  prec: 'half' as Prec,
  /** 向量长度指数（vecadd / saxpy） */
  nVec: 24,
  /** GEMV 方阵边长指数 */
  nMat: 12,
  m: 12,
  n: 12,
  k: 12,
  /** attention：序列长 S 指数、head 维度 d 指数 */
  s: 12,
  d: 7,
}

interface CalcOut {
  flops: number
  bytes: number
  /** FLOPs 公式（代入数值后的 KaTeX 串） */
  fTex: string
  /** 字节公式 */
  bTex: string
}

/** 在 KaTeX 里展示 SI 缩写数（如 16.8M） */
const si = (x: number) => String.raw`\text{${fmtSI(x, 1)}}`

function compute(op: OpId, b: number, st: typeof DEF): CalcOut {
  switch (op) {
    case 'vecadd': {
      const N = 2 ** st.nVec
      const flops = N
      const bytes = 3 * N * b
      return {
        flops,
        bytes,
        fTex: String.raw`\text{FLOPs} = N = ${si(N)}`,
        bTex: String.raw`\text{Bytes} = \underbrace{3N}_{\text{读 a,b 写 c}} \times ${b}\,\text{B} = ${si(bytes)}\,\text{B}`,
      }
    }
    case 'saxpy': {
      const N = 2 ** st.nVec
      const flops = 2 * N
      const bytes = 3 * N * b
      return {
        flops,
        bytes,
        fTex: String.raw`\text{FLOPs} = \underbrace{2N}_{\text{乘 + 加}} = ${si(flops)}`,
        bTex: String.raw`\text{Bytes} = \underbrace{3N}_{\text{读 x,y 写 y}} \times ${b}\,\text{B} = ${si(bytes)}\,\text{B}`,
      }
    }
    case 'gemv': {
      const n = 2 ** st.nMat
      const flops = 2 * n * n
      const bytes = (n * n + 2 * n) * b
      return {
        flops,
        bytes,
        fTex: String.raw`\text{FLOPs} = 2n^2 = 2 \times ${si(n)}^2 = ${si(flops)}`,
        bTex: String.raw`\text{Bytes} = (\underbrace{n^2}_{\text{矩阵 A}} + \underbrace{2n}_{\text{x, y}}) \times ${b}\,\text{B} = ${si(bytes)}\,\text{B}`,
      }
    }
    case 'gemm': {
      const M = 2 ** st.m
      const N = 2 ** st.n
      const K = 2 ** st.k
      const flops = 2 * M * N * K
      const bytes = (M * N + M * K + K * N) * b
      return {
        flops,
        bytes,
        fTex: String.raw`\text{FLOPs} = 2MNK = 2 \times ${si(M)} \times ${si(N)} \times ${si(K)} = ${si(flops)}`,
        bTex: String.raw`\text{Bytes} = (MN + MK + KN) \times ${b}\,\text{B} = ${si(bytes)}\,\text{B}`,
      }
    }
    case 'attn': {
      const S = 2 ** st.s
      const d = 2 ** st.d
      const flops = 2 * S * S * d
      const bytes = (2 * S * d + S * S) * b
      return {
        flops,
        bytes,
        fTex: String.raw`\text{FLOPs} = 2S^2 d = 2 \times ${si(S)}^2 \times ${d} = ${si(flops)}`,
        bTex: String.raw`\text{Bytes} = (\underbrace{2Sd}_{Q,K} + \underbrace{S^2}_{\text{分数矩阵}}) \times ${b}\,\text{B} = ${si(bytes)}\,\text{B}`,
      }
    }
  }
}

export function IntensityCalc() {
  const [op, setOp] = useState<OpId>(DEF.op)
  const [hwId, setHwId] = useState<string>(DEF.hw)
  const [prec, setPrec] = useState<Prec>(DEF.prec)
  const [nVec, setNVec] = useState(DEF.nVec)
  const [nMat, setNMat] = useState(DEF.nMat)
  const [m, setM] = useState(DEF.m)
  const [n, setN] = useState(DEF.n)
  const [k, setK] = useState(DEF.k)
  const [s, setS] = useState(DEF.s)
  const [d, setD] = useState(DEF.d)

  const reset = () => {
    setOp(DEF.op)
    setHwId(DEF.hw)
    setPrec(DEF.prec)
    setNVec(DEF.nVec)
    setNMat(DEF.nMat)
    setM(DEF.m)
    setN(DEF.n)
    setK(DEF.k)
    setS(DEF.s)
    setD(DEF.d)
  }

  const hw = hwById(hwId)
  const b = prec === 'half' ? 2 : 4
  const peak = prec === 'half' ? hw.tensor : hw.fp32
  const out = compute(op, b, { ...DEF, nVec, nMat, m, n, k, s, d })
  const ai = out.flops / out.bytes
  const ridge = ridgeOf(peak, hw.bw)
  const tMem = out.bytes / (hw.bw * 1e12)
  const tComp = out.flops / (peak * 1e12)
  const memBound = ai < ridge

  const exp2 = (e: number) => fmtSI(2 ** e, 2 ** e >= 1024 ? 1 : 0)

  return (
    <Widget
      index={1}
      title="算术强度计算器"
      subtitle="同一台机器，不同操作的「体质」差几个数量级"
      onReset={reset}
      footer={
        <>
          GEMM 的 AI = (2/b) · 1/(1/M + 1/N + 1/K)：<b>三个维度里最小的那个说了算</b>。把 K 拖到 16
          试试——再大的 M、N 也救不回来，这正是「瘦矩阵乘吃不满算力」的根源。
        </>
      }
    >
      <div className="flex flex-col gap-4">
        {/* 操作选择 */}
        <Segmented
          block
          options={[
            { value: 'vecadd', label: '向量加' },
            { value: 'saxpy', label: 'SAXPY' },
            { value: 'gemv', label: 'GEMV' },
            { value: 'gemm', label: 'GEMM' },
            { value: 'attn', label: 'Attn score' },
          ]}
          value={op}
          onChange={setOp}
        />
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="microlabel">硬件</span>
          <Segmented
            options={HARDWARE.map((h) => ({ value: h.id, label: h.name }))}
            value={hwId}
            onChange={setHwId}
          />
          <span className="microlabel">精度</span>
          <Segmented
            options={[
              { value: 'half', label: 'BF16 · Tensor Core' },
              { value: 'fp32', label: 'FP32 · CUDA Core' },
            ]}
            value={prec}
            onChange={setPrec}
          />
        </div>

        {/* 形状滑杆 */}
        <div className="grid grid-cols-1 gap-x-5 gap-y-3 sm:grid-cols-3">
          {(op === 'vecadd' || op === 'saxpy') && (
            <Slider label="N · 向量长度" value={nVec} min={16} max={28} onChange={setNVec} fmt={exp2} unit="元素" className="sm:col-span-3" />
          )}
          {op === 'gemv' && (
            <Slider label="n · 方阵边长" value={nMat} min={8} max={14} onChange={setNMat} fmt={exp2} className="sm:col-span-3" />
          )}
          {op === 'gemm' && (
            <>
              <Slider label="M" value={m} min={4} max={13} onChange={setM} fmt={exp2} />
              <Slider label="N" value={n} min={4} max={13} onChange={setN} fmt={exp2} />
              <Slider label="K" value={k} min={4} max={13} onChange={setK} fmt={exp2} />
            </>
          )}
          {op === 'attn' && (
            <>
              <Slider label="S · 序列长度" value={s} min={7} max={14} onChange={setS} fmt={exp2} className="sm:col-span-2" />
              <Slider label="d · head 维度" value={d} min={6} max={8} onChange={setD} fmt={exp2} />
            </>
          )}
        </div>

        {/* 公式区 */}
        <div className="rounded-md border border-line bg-bg2 px-4 py-1">
          <MathTex block tex={out.fTex} />
          <MathTex block tex={out.bTex} />
          <MathTex
            block
            tex={String.raw`\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{${si(out.flops)}}{${si(out.bytes)}\,\text{B}} = ${fmtAI(ai)}\ \text{FLOP/B}`}
          />
        </div>

        {/* 读数 + 判定 */}
        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <Stat label="算术强度 AI" value={fmtAI(ai)} unit="FLOP/B" tone="volt" size="lg" />
          <Stat label={`ridge（${hw.name}）`} value={fmtAI(ridge)} unit="FLOP/B" tone="ink" />
          <Stat label="搬数据耗时" value={fmtTime(tMem)} tone={memBound ? 'amber' : 'ink'} />
          <Stat label="纯计算耗时" value={fmtTime(tComp)} tone={memBound ? 'ink' : 'cyan'} />
          <span
            className={`mb-1 inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 font-mono text-xs tracking-wider ${
              memBound
                ? 'border-amber/50 bg-amber/10 text-amber'
                : 'border-volt/50 bg-volt/10 text-volt'
            }`}
          >
            {memBound ? '◀ MEMORY-BOUND' : '▶ COMPUTE-BOUND'}
          </span>
        </div>
        <p className="text-[13px] leading-relaxed text-ink2">
          {memBound ? (
            <>
              AI（{fmtAI(ai)}）&lt; ridge（{fmtAI(ridge)}）：搬数据的时间（
              <span className="font-mono text-amber">{fmtTime(tMem)}</span>）盖过了计算（
              <span className="font-mono">{fmtTime(tComp)}</span>）。理论上限只有{' '}
              <span className="font-mono text-ink">{(ai * hw.bw).toFixed(ai * hw.bw < 10 ? 2 : 0)} TFLOPS</span>
              ——峰值算力 {peak} TFLOPS 大部分在围观。
            </>
          ) : (
            <>
              AI（{fmtAI(ai)}）&gt; ridge（{fmtAI(ridge)}）：计算时间（
              <span className="font-mono text-cyan">{fmtTime(tComp)}</span>）盖过了搬数据（
              <span className="font-mono">{fmtTime(tMem)}</span>
              ）。带宽喂得饱，性能上限就是峰值算力 {peak} TFLOPS——接下来拼的是利用率。
            </>
          )}
        </p>
      </div>
    </Widget>
  )
}
