/** CH06 共享数据：硬件规格 + roofline 数学 + 数字格式化 */

export interface Hardware {
  id: 'a100' | 'h100' | 'rtx4090'
  name: string
  /** 显存带宽，TB/s */
  bw: number
  /** 半精度（BF16/FP16）Tensor Core 峰值，TFLOPS（dense，不算稀疏） */
  tensor: number
  /** FP32 CUDA Core 峰值，TFLOPS */
  fp32: number
  /** 显存类型 */
  mem: string
}

export const HARDWARE: Hardware[] = [
  { id: 'a100', name: 'A100', bw: 1.9, tensor: 312, fp32: 19.5, mem: 'HBM2e' },
  { id: 'h100', name: 'H100 SXM', bw: 3.35, tensor: 989, fp32: 67, mem: 'HBM3' },
  { id: 'rtx4090', name: 'RTX 4090', bw: 1.0, tensor: 165, fp32: 82.6, mem: 'GDDR6X' },
]

export function hwById(id: string): Hardware {
  return HARDWARE.find((h) => h.id === id) ?? HARDWARE[0]
}

/** 机器平衡点 ridge = 峰值算力 / 带宽。TFLOPS / (TB/s) 正好约掉 1e12，单位是 FLOP/B */
export const ridgeOf = (peakTflops: number, bwTBs: number) => peakTflops / bwTBs

/** roofline：给定 AI 与硬件，可达性能上限（TFLOPS） */
export const attainable = (ai: number, bwTBs: number, peakTflops: number) =>
  Math.min(ai * bwTBs, peakTflops)

/** AI 的自适应小数位显示 */
export function fmtAI(ai: number): string {
  if (ai >= 100) return ai.toFixed(0)
  if (ai >= 10) return ai.toFixed(1)
  if (ai >= 1) return ai.toFixed(2)
  return ai.toFixed(3)
}

/** 秒 → 人类可读时间 */
export function fmtTime(sec: number): string {
  if (sec >= 1) return sec.toFixed(2) + ' s'
  if (sec >= 1e-3) return (sec * 1e3).toFixed(2) + ' ms'
  if (sec >= 1e-6) return (sec * 1e6).toFixed(2) + ' µs'
  return (sec * 1e9).toFixed(0) + ' ns'
}
