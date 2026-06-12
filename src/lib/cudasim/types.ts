/**
 * EasyInfra CUDA 模拟器 —— 公共类型契约
 *
 * 引擎实现一个 CUDA C 子集的解释器，在浏览器中"伪执行"核函数：
 *  - compile(source) 把源码编译为 CompiledKernel（或返回带行号的编译错误）
 *  - run(kernel, config, buffers, opts) 按 grid/block 配置执行所有线程，
 *    记录每次全局/共享内存访问，统计合并度、bank conflict、分支分化与同步，
 *    返回输出缓冲区与完整访问轨迹，供可视化使用。
 *
 * 实现要求见 src/lib/cudasim/LANGUAGE.md（支持的语法子集）。
 */

export interface Dim3 {
  x: number
  y: number
  z: number
}

export interface LaunchConfig {
  grid: Dim3
  block: Dim3
}

/** 设备端缓冲区声明。按 name 与 kernel 的指针参数绑定（同名绑定）。 */
export interface BufferDecl {
  name: string
  /** 元素个数（float/int 均按 1 元素计） */
  length: number
  /**
   * 初始化方式：
   *  - 'zero'   全 0（默认）
   *  - 'iota'   0,1,2,3,...
   *  - 'random' [0,1) 伪随机（固定种子，可复现）
   *  - number[] 显式给值（不足补 0）
   */
  init?: 'zero' | 'iota' | 'random' | number[]
}

export interface ThreadCoord {
  block: Dim3
  thread: Dim3
}

/** 一次内存访问事件（可视化的核心数据源） */
export interface AccessEvent {
  /** 该线程的逻辑步号（用于 warp 级聚合：同 warp 同 step 的访问视为同拍发出） */
  step: number
  /** 全局 warp 编号 */
  warpId: number
  /** warp 内 lane（0-31） */
  laneId: number
  thread: ThreadCoord
  space: 'global' | 'shared'
  /** 缓冲区名；shared 空间为共享数组变量名 */
  buffer: string
  /** 元素下标 */
  index: number
  kind: 'read' | 'write'
  /** 源码行号（1-based） */
  line: number
}

export interface SimStats {
  threads: number
  warps: number
  /** 所有线程的最大逻辑步数 */
  maxSteps: number
  globalReads: number
  globalWrites: number
  /**
   * 全局内存事务数：把同 warp 同 step 的全局访问按 32 字节段聚合
   * （每元素按 4 字节计），事务数 = 触及的不同段数。完美合并时
   * 32 个 float 读 = 4 个事务。
   */
  globalTransactions: number
  sharedReads: number
  sharedWrites: number
  /**
   * bank conflict 计数：同 warp 同 step 的 shared 访问按 32 banks
   * （bank = index % 32）聚合，conflict = max(同 bank 不同地址的访问数) - 1 的累计。
   */
  bankConflicts: number
  /** __syncthreads() 障栅次数（按 block 计一次） */
  syncBarriers: number
  /** 发生 warp 内分支分化的 (分支点, warp) 数 */
  divergentBranches: number
}

export interface CompileError {
  kind: 'compile'
  message: string
  /** 1-based */
  line: number
  col: number
}

export interface RuntimeErrorInfo {
  kind: 'runtime'
  /** 例如 "数组越界：A[1024]（长度 1024）" */
  message: string
  line?: number
  /** 触发错误的线程 */
  thread?: ThreadCoord
}

export interface KernelParam {
  name: string
  /** 'float*' | 'int*' | 'float' | 'int' */
  type: string
  isPointer: boolean
}

export interface KernelInfo {
  name: string
  params: KernelParam[]
  /** 声明的 __shared__ 数组：名字 → 长度 */
  sharedArrays: Record<string, number>
}

/** 编译产物（内部结构对调用方不透明） */
export interface CompiledKernel {
  info: KernelInfo
  /** 引擎内部表示（AST 等），调用方不应访问 */
  _ast: unknown
}

export type CompileOutcome = { ok: true; kernel: CompiledKernel } | { ok: false; error: CompileError }

export interface RunOptions {
  /** 标量参数赋值（按参数名），如 { n: 1024 } */
  scalarArgs?: Record<string, number>
  /** 单线程最大逻辑步数，超出判定为死循环（默认 100_000） */
  maxStepsPerThread?: number
  /** 是否记录访问轨迹（默认 true；大规模运行可关闭以省内存） */
  recordAccesses?: boolean
  /** 轨迹最多记录条数（默认 200_000，超出截断但统计仍完整） */
  maxTraceEvents?: number
}

export type SimOutcome =
  | {
      ok: true
      /** 运行结束后的所有缓冲区内容 */
      buffers: Record<string, number[]>
      stats: SimStats
      /** 访问轨迹（可能按 maxTraceEvents 截断） */
      accesses: AccessEvent[]
      traceTruncated: boolean
    }
  | {
      ok: false
      error: RuntimeErrorInfo
      /** 出错前累计的部分统计/轨迹（尽力提供） */
      stats?: SimStats
      accesses?: AccessEvent[]
    }

/** 引擎入口（由 src/lib/cudasim/index.ts 导出实现） */
export interface CudaSimEngine {
  compile(source: string): CompileOutcome
  run(kernel: CompiledKernel, config: LaunchConfig, buffers: BufferDecl[], opts?: RunOptions): SimOutcome
}
