/** 数值格式化工具 */

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t

/** 千分位整数 */
export const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US')

/** SI 缩写：1.2K / 3.4M / 5.6B / 7.8T */
export function fmtSI(n: number, digits = 1): string {
  const abs = Math.abs(n)
  if (abs >= 1e12) return (n / 1e12).toFixed(digits) + 'T'
  if (abs >= 1e9) return (n / 1e9).toFixed(digits) + 'B'
  if (abs >= 1e6) return (n / 1e6).toFixed(digits) + 'M'
  if (abs >= 1e3) return (n / 1e3).toFixed(digits) + 'K'
  return String(Math.round(n * 100) / 100)
}

/** 字节：KB / MB / GB / TB（1024 进制） */
export function fmtBytes(n: number, digits = 1): string {
  const abs = Math.abs(n)
  if (abs >= 2 ** 40) return (n / 2 ** 40).toFixed(digits) + ' TB'
  if (abs >= 2 ** 30) return (n / 2 ** 30).toFixed(digits) + ' GB'
  if (abs >= 2 ** 20) return (n / 2 ** 20).toFixed(digits) + ' MB'
  if (abs >= 2 ** 10) return (n / 2 ** 10).toFixed(digits) + ' KB'
  return Math.round(n) + ' B'
}

/** FLOPs：GFLOPs / TFLOPs / PFLOPs */
export function fmtFlops(n: number, digits = 1): string {
  const abs = Math.abs(n)
  if (abs >= 1e15) return (n / 1e15).toFixed(digits) + ' PFLOPs'
  if (abs >= 1e12) return (n / 1e12).toFixed(digits) + ' TFLOPs'
  if (abs >= 1e9) return (n / 1e9).toFixed(digits) + ' GFLOPs'
  if (abs >= 1e6) return (n / 1e6).toFixed(digits) + ' MFLOPs'
  return fmtSI(n, digits) + ' FLOPs'
}

/** 百分比 */
export const pct = (x: number, digits = 0) => (x * 100).toFixed(digits) + '%'
