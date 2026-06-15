import type { Loc } from './i18n'

/**
 * 全站统一的参考 GPU 规格 —— 单一事实来源。
 * Canonical reference-GPU specs, used by <HardwareBaseline> so every chapter
 * declares which card its numbers assume. Keep figures consistent with the
 * course convention (A100 HBM = 1.9 TB/s; see ch6).
 */

export type GpuId = 'a100' | 'h100' | 'rtx4090'

export interface GpuSpec {
  id: GpuId
  /** 完整型号名（两种语言通用） */
  name: string
  /** 简称，用于行内 */
  short: string
  /** 头部规格条目（标签双语，值用通用记法） */
  specs: { label: Loc; value: string }[]
}

export const GPUS: Record<GpuId, GpuSpec> = {
  a100: {
    id: 'a100',
    name: 'A100 80GB SXM',
    short: 'A100',
    specs: [
      { label: { en: 'SMs', zh: 'SM 数' }, value: '108' },
      { label: { en: 'BF16 Tensor', zh: 'BF16 张量' }, value: '312 TFLOPS' },
      { label: { en: 'FP32', zh: 'FP32' }, value: '19.5 TFLOPS' },
      { label: { en: 'HBM2e', zh: '显存带宽' }, value: '1.9 TB/s' },
      { label: { en: 'L2', zh: 'L2' }, value: '40 MB' },
      { label: { en: 'Shared/SM', zh: '共享内存/SM' }, value: '≤164 KB' },
    ],
  },
  h100: {
    id: 'h100',
    name: 'H100 SXM',
    short: 'H100',
    specs: [
      { label: { en: 'SMs', zh: 'SM 数' }, value: '132' },
      { label: { en: 'BF16 Tensor', zh: 'BF16 张量' }, value: '989 TFLOPS' },
      { label: { en: 'FP32', zh: 'FP32' }, value: '67 TFLOPS' },
      { label: { en: 'HBM3', zh: '显存带宽' }, value: '3.35 TB/s' },
      { label: { en: 'L2', zh: 'L2' }, value: '50 MB' },
      { label: { en: 'Shared/SM', zh: '共享内存/SM' }, value: '≤228 KB' },
    ],
  },
  rtx4090: {
    id: 'rtx4090',
    name: 'RTX 4090',
    short: 'RTX 4090',
    specs: [
      { label: { en: 'SMs', zh: 'SM 数' }, value: '128' },
      { label: { en: 'BF16 Tensor', zh: 'BF16 张量' }, value: '165 TFLOPS' },
      { label: { en: 'FP32', zh: 'FP32' }, value: '82.6 TFLOPS' },
      { label: { en: 'GDDR6X', zh: '显存带宽' }, value: '1.0 TB/s' },
      { label: { en: 'L2', zh: 'L2' }, value: '72 MB' },
    ],
  },
}
