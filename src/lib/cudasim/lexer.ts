/**
 * CUDA C 子集 —— 词法分析器
 * 产出带行号/列号（1-based）的 Token 流；遇到不支持的字符级语法直接抛 CompileErrorException。
 */
import type { CompileError } from './types'

/** 编译期错误（内部用异常传递，index.ts 捕获后转成 CompileOutcome） */
export class CompileErrorException extends Error {
  readonly line: number
  readonly col: number
  constructor(message: string, line: number, col: number) {
    super(message)
    this.name = 'CompileErrorException'
    this.line = line
    this.col = col
  }
  toCompileError(): CompileError {
    return { kind: 'compile', message: this.message, line: this.line, col: this.col }
  }
}

export interface Token {
  kind: 'ident' | 'num' | 'punct' | 'eof'
  text: string
  /** kind === 'num' 时的数值 */
  value?: number
  /** kind === 'num' 时：是否浮点字面量（含小数点 / 指数 / f 后缀） */
  isFloat?: boolean
  line: number
  col: number
}

const PUNCT2 = ['++', '--', '+=', '-=', '*=', '/=', '%=', '==', '!=', '<=', '>=', '&&', '||', '<<', '>>', '->']
const PUNCT1 = '+-*/%=<>!?:;,.()[]{}&|^~'

const isIdentStart = (c: string) => /[A-Za-z_]/.test(c)
const isIdentChar = (c: string) => /[A-Za-z0-9_]/.test(c)
const isDigit = (c: string) => c >= '0' && c <= '9'

export function lex(src: string): Token[] {
  const toks: Token[] = []
  let i = 0
  let line = 1
  let col = 1
  const n = src.length

  const fail = (msg: string, l: number, c: number): never => {
    throw new CompileErrorException(msg, l, c)
  }

  while (i < n) {
    const ch = src[i]
    if (ch === '\n') {
      i++
      line++
      col = 1
      continue
    }
    if (ch === ' ' || ch === '\t' || ch === '\r') {
      i++
      col++
      continue
    }
    // 行注释
    if (ch === '/' && src[i + 1] === '/') {
      while (i < n && src[i] !== '\n') i++
      continue
    }
    // 块注释
    if (ch === '/' && src[i + 1] === '*') {
      const sl = line
      const sc = col
      i += 2
      col += 2
      for (;;) {
        if (i >= n) fail('未闭合的块注释 /* ... */', sl, sc)
        if (src[i] === '\n') {
          i++
          line++
          col = 1
        } else if (src[i] === '*' && src[i + 1] === '/') {
          i += 2
          col += 2
          break
        } else {
          i++
          col++
        }
      }
      continue
    }

    const sl = line
    const sc = col

    if (ch === '#') fail('模拟器暂不支持 预处理指令（#include / #define 等），请删除以 # 开头的行', sl, sc)
    if (ch === '"' || ch === "'") fail('模拟器暂不支持 字符串/字符字面量', sl, sc)

    // 标识符 / 关键字
    if (isIdentStart(ch)) {
      let j = i
      while (j < n && isIdentChar(src[j])) j++
      const text = src.slice(i, j)
      col += j - i
      i = j
      toks.push({ kind: 'ident', text, line: sl, col: sc })
      continue
    }

    // 数字字面量：123 / 1.5 / .5 / 1e-5 / 2.f / 1e-5f
    if (isDigit(ch) || (ch === '.' && isDigit(src[i + 1] ?? ''))) {
      if (ch === '0' && (src[i + 1] === 'x' || src[i + 1] === 'X')) {
        fail('模拟器暂不支持 十六进制字面量', sl, sc)
      }
      let j = i
      let isFloat = false
      while (j < n && isDigit(src[j])) j++
      if (src[j] === '.') {
        isFloat = true
        j++
        while (j < n && isDigit(src[j])) j++
      }
      if (src[j] === 'e' || src[j] === 'E') {
        isFloat = true
        j++
        if (src[j] === '+' || src[j] === '-') j++
        if (!isDigit(src[j] ?? '')) fail('无效的数字字面量：指数部分缺少数字', sl, sc)
        while (j < n && isDigit(src[j])) j++
      }
      const numText = src.slice(i, j)
      const value = parseFloat(numText)
      if (src[j] === 'f' || src[j] === 'F') {
        isFloat = true
        j++
      }
      if (j < n && (isIdentChar(src[j]) || src[j] === '.')) {
        fail(`无效的数字字面量 '${src.slice(i, j + 1)}…'（暂不支持该写法/后缀）`, sl, sc)
      }
      col += j - i
      i = j
      toks.push({ kind: 'num', text: numText, value, isFloat, line: sl, col: sc })
      continue
    }

    // 标点 / 运算符（先匹配双字符）
    const two = src.slice(i, i + 2)
    if (PUNCT2.includes(two)) {
      toks.push({ kind: 'punct', text: two, line: sl, col: sc })
      i += 2
      col += 2
      continue
    }
    if (PUNCT1.includes(ch)) {
      toks.push({ kind: 'punct', text: ch, line: sl, col: sc })
      i++
      col++
      continue
    }
    fail(`无法识别的字符 '${ch}'`, sl, sc)
  }

  toks.push({ kind: 'eof', text: '<文件结尾>', line, col })
  return toks
}
