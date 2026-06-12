/**
 * EasyInfra CUDA 模拟器 —— 引擎入口
 *
 *   compile(source)                      → CompileOutcome
 *   run(kernel, config, buffers, opts?)  → SimOutcome
 *
 * 支持的语法子集见 ./LANGUAGE.md；公共契约见 ./types.ts。
 */
import type {
  BufferDecl,
  CompileOutcome,
  CompiledKernel,
  CudaSimEngine,
  KernelInfo,
  LaunchConfig,
  RunOptions,
  SimOutcome,
} from './types'
import { CompileErrorException, lex } from './lexer'
import { analyze, parse, type KernelAst } from './parser'
import { runKernel } from './interp'

export function compile(source: string): CompileOutcome {
  try {
    const tokens = lex(source)
    const ast = parse(tokens)
    analyze(ast)
    const sharedArrays: Record<string, number> = {}
    for (const s of ast.shared) sharedArrays[s.name] = s.length
    const info: KernelInfo = {
      name: ast.name,
      params: ast.params.map((p) => ({ name: p.name, type: p.type, isPointer: p.isPointer })),
      sharedArrays,
    }
    return { ok: true, kernel: { info, _ast: ast } }
  } catch (e) {
    if (e instanceof CompileErrorException) return { ok: false, error: e.toCompileError() }
    throw e
  }
}

export function run(
  kernel: CompiledKernel,
  config: LaunchConfig,
  buffers: BufferDecl[],
  opts?: RunOptions,
): SimOutcome {
  return runKernel(kernel._ast as KernelAst, config, buffers, opts)
}

export const engine: CudaSimEngine = { compile, run }
export default engine

export type * from './types'
