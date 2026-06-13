/**
 * 语义色板（JS 常量）—— 供 canvas / 内联样式 / CodeMirror 等无法使用 Tailwind 类或
 * CSS 变量的场景。必须与 src/index.css 的 @theme 保持同步。
 *
 * Semantic palette as JS constants — for canvas / inline-style / CodeMirror where
 * Tailwind classes and CSS variables aren't available. KEEP IN SYNC with @theme in index.css.
 */
export const C = {
  bg: '#f7f8f5',
  bg2: '#eef0ea',
  panel: '#ffffff',
  panel2: '#f3f5f0',
  line: '#e4e6df',
  line2: '#ccd0c5',
  ink: '#15181c',
  text: '#3a4048',
  ink2: '#59616c',
  ink3: '#8a929c',
  volt: '#1c8a3f',
  voltBright: '#3aa856',
  cyan: '#0e78a8',
  amber: '#b06d09',
  rose: '#c92a4d',
  violet: '#6a3bd0',
} as const

export type PaletteKey = keyof typeof C

/** 把 #rrggbb 转成 rgba(...) 字符串 */
export function rgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const n = parseInt(
    h.length === 3
      ? h
          .split('')
          .map((c) => c + c)
          .join('')
      : h,
    16,
  )
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
