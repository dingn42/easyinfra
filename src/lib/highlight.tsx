import type { ReactNode } from 'react'

/** 轻量语法高亮（静态代码块用，不依赖 CodeMirror） */

export type Lang = 'cuda' | 'cpp' | 'python' | 'js' | 'bash' | 'text'

interface Token {
  text: string
  cls: string | null
}

const CPP_KEYWORDS = new Set([
  'if', 'else', 'for', 'while', 'do', 'return', 'break', 'continue', 'switch', 'case', 'default',
  'struct', 'class', 'template', 'typename', 'const', 'constexpr', 'static', 'extern', 'inline',
  'sizeof', 'new', 'delete', 'namespace', 'using', 'public', 'private', 'true', 'false', 'nullptr', 'NULL',
])
const CPP_TYPES = new Set([
  'void', 'int', 'float', 'double', 'char', 'bool', 'long', 'short', 'unsigned', 'signed', 'auto',
  'size_t', 'int8_t', 'int32_t', 'uint32_t', 'int64_t', 'half', 'float4', 'float2', 'dim3',
  'cudaError_t', 'cudaStream_t', 'cudaEvent_t',
])
const CUDA_BUILTINS = new Set([
  'threadIdx', 'blockIdx', 'blockDim', 'gridDim', 'warpSize',
  '__global__', '__device__', '__host__', '__shared__', '__restrict__', '__constant__',
  '__syncthreads', '__syncwarp', '__shfl_down_sync', '__shfl_xor_sync', 'atomicAdd', 'atomicMax',
  'cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaMemcpyAsync', 'cudaDeviceSynchronize',
  'cudaMemcpyHostToDevice', 'cudaMemcpyDeviceToHost', 'cudaGetLastError', 'cudaStreamCreate',
])
const PY_KEYWORDS = new Set([
  'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and', 'or', 'is',
  'import', 'from', 'as', 'with', 'try', 'except', 'finally', 'raise', 'lambda', 'yield', 'pass',
  'break', 'continue', 'None', 'True', 'False', 'assert', 'global', 'del', 'async', 'await',
])
const JS_KEYWORDS = new Set([
  'const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
  'break', 'continue', 'new', 'typeof', 'instanceof', 'class', 'extends', 'import', 'export', 'from',
  'default', 'try', 'catch', 'finally', 'throw', 'async', 'await', 'yield', 'true', 'false', 'null', 'undefined',
])

function tokenize(code: string, lang: Lang): Token[] {
  if (lang === 'text') return [{ text: code, cls: null }]
  const tokens: Token[] = []
  // 注释 | 字符串 | 数字 | 标识符 | 其他
  const re =
    lang === 'python' || lang === 'bash'
      ? /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\b0[xX][\da-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[fF]?\b)|([A-Za-z_]\w*)|([\s\S])/g
      : /(\/\/[^\n]*|\/\*[\s\S]*?\*\/|^[ \t]*#\w+[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b0[xX][\da-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[fFuUlL]*\b)|([A-Za-z_]\w*)|([\s\S])/gm

  let m: RegExpExecArray | null
  let lastIndex = 0
  while ((m = re.exec(code))) {
    if (m.index > lastIndex) tokens.push({ text: code.slice(lastIndex, m.index), cls: null })
    lastIndex = re.lastIndex
    const [full, com, str, num, ident] = m
    if (com != null) {
      tokens.push({ text: full, cls: full.startsWith('#') && lang !== 'python' && lang !== 'bash' ? 'tok-pre' : 'tok-com' })
    } else if (str != null) {
      tokens.push({ text: full, cls: 'tok-str' })
    } else if (num != null) {
      tokens.push({ text: full, cls: 'tok-num' })
    } else if (ident != null) {
      let cls: string | null = null
      if (lang === 'cuda' || lang === 'cpp') {
        if (CUDA_BUILTINS.has(ident)) cls = 'tok-cuda'
        else if (CPP_KEYWORDS.has(ident)) cls = 'tok-kw'
        else if (CPP_TYPES.has(ident)) cls = 'tok-type'
      } else if (lang === 'python') {
        if (PY_KEYWORDS.has(ident)) cls = 'tok-kw'
      } else if (lang === 'js') {
        if (JS_KEYWORDS.has(ident)) cls = 'tok-kw'
      }
      if (!cls) {
        // 函数调用着色：标识符后紧跟 (
        const rest = code.slice(lastIndex)
        if (/^\s*\(/.test(rest)) cls = 'tok-fn'
      }
      tokens.push({ text: full, cls })
    } else {
      tokens.push({ text: full, cls: /[(){}[\];,<>=+\-*/%&|^!~?:.]/.test(full) ? 'tok-punc' : null })
    }
  }
  if (lastIndex < code.length) tokens.push({ text: code.slice(lastIndex), cls: null })
  return tokens
}

/** 将一段代码渲染为带高亮的 React 节点数组（按行拆分） */
export function highlightLines(code: string, lang: Lang): ReactNode[][] {
  const tokens = tokenize(code.replace(/\n$/, ''), lang)
  const lines: ReactNode[][] = [[]]
  let key = 0
  for (const tok of tokens) {
    const parts = tok.text.split('\n')
    parts.forEach((part, i) => {
      if (i > 0) lines.push([])
      if (part.length === 0) return
      const node = tok.cls ? (
        <span key={key++} className={tok.cls}>
          {part}
        </span>
      ) : (
        part
      )
      lines[lines.length - 1].push(node)
    })
  }
  return lines
}
