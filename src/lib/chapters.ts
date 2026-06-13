import type { Loc } from './i18n'

/** 课程目录元数据 —— 全站唯一的章节注册表（组件懒加载见 src/chapters/index.ts）
 *  Curriculum metadata — the single source of truth for chapters. */

export interface PartMeta {
  num: 1 | 2 | 3
  /** localized: "Part I" / "第一部分" */
  label: Loc
  title: Loc
  /** decorative uppercase codename (English in both locales) */
  titleEn: string
  blurb: Loc
}

export interface ChapterMeta {
  /** URL id，如 /learn/why-gpu */
  id: string
  /** 目录序号 1-12 */
  num: number
  part: 1 | 2 | 3
  title: Loc
  /** decorative uppercase codename (English in both locales) */
  titleEn: string
  /** 一句话副标题 */
  tagline: Loc
  /** 目录页摘要 */
  summary: Loc
  /** 预估阅读分钟数 */
  minutes: number
  /** 本章交互实验名称列表 */
  labs: Loc[]
}

export const PARTS: PartMeta[] = [
  {
    num: 1,
    label: { en: 'Part I', zh: '第一部分' },
    title: { en: 'GPUs & Parallel Computing', zh: 'GPU 与并行计算' },
    titleEn: 'PARALLEL FOUNDATIONS',
    blurb: {
      en: 'Before writing a single line of CUDA, understand why the machine is shaped the way it is.',
      zh: '在写下任何一行 CUDA 之前，先理解这台机器为什么长这样。',
    },
  },
  {
    num: 2,
    label: { en: 'Part II', zh: '第二部分' },
    title: { en: 'CUDA Programming', zh: 'CUDA 编程' },
    titleEn: 'CUDA PROGRAMMING',
    blurb: {
      en: 'From your first kernel to near-cuBLAS speed: the programming model, the memory system, and a methodology for optimization.',
      zh: '从第一个 kernel 到逼近 cuBLAS：编程模型、内存系统与优化方法论。',
    },
  },
  {
    num: 3,
    label: { en: 'Part III', zh: '第三部分' },
    title: { en: 'LLM Inference Systems', zh: '大模型推理系统' },
    titleEn: 'LLM INFERENCE SYSTEMS',
    blurb: {
      en: 'FlashAttention, KV cache, continuous batching, quantization, and distributed parallelism — the core ideas behind production LLM infrastructure.',
      zh: 'FlashAttention、KV Cache、连续批处理、量化与分布式并行 —— 生产级 LLM Infra 的核心思想。',
    },
  },
]

export const CHAPTERS: ChapterMeta[] = [
  {
    id: 'why-gpu',
    num: 1,
    part: 1,
    title: { en: 'Why GPUs?', zh: '为什么需要 GPU' },
    titleEn: 'WHY GPUS',
    tagline: { en: 'From a latency machine to a throughput machine', zh: '从「延迟机器」到「吞吐机器」' },
    summary: {
      en: 'How CPUs and GPUs differ in design philosophy, the intuition behind parallel computing, and why large models cannot live without GPUs.',
      zh: 'CPU 与 GPU 的设计哲学差异、并行计算的基本直觉，以及大模型为什么离不开 GPU。',
    },
    minutes: 25,
    labs: [
      { en: 'Throughput Race', zh: '吞吐量赛道' },
      { en: 'Transistor Budget Allocator', zh: '晶体管预算分配器' },
    ],
  },
  {
    id: 'gpu-architecture',
    num: 2,
    part: 1,
    title: { en: 'Anatomy of a GPU', zh: 'GPU 硬件解剖' },
    titleEn: 'GPU ANATOMY',
    tagline: { en: 'From die to SM, from warp to core', zh: '从 die 到 SM，从 warp 到核心' },
    summary: {
      en: 'Take a modern GPU apart layer by layer: GPCs, SMs, warp schedulers, Tensor Cores, and the memory hierarchy.',
      zh: '逐层拆开一块现代 GPU：GPC、SM、warp 调度器、Tensor Core 与显存层级。',
    },
    minutes: 30,
    labs: [
      { en: 'GPU Dissection Table', zh: 'GPU 解剖台' },
      { en: 'Memory Hierarchy Latency', zh: '显存层级延迟实验' },
      { en: 'Warp Latency Hiding', zh: 'Warp 延迟隐藏' },
    ],
  },
  {
    id: 'cuda-model',
    num: 3,
    part: 2,
    title: { en: 'The CUDA Model', zh: 'CUDA 编程模型' },
    titleEn: 'THE CUDA MODEL',
    tagline: { en: 'You write one function; the hardware runs it a million times', zh: '你写一个函数，硬件运行一百万次' },
    summary: {
      en: 'Kernels, grids, blocks, and threads: the CUDA execution model and your very first kernel.',
      zh: 'kernel、grid、block、thread：CUDA 的执行模型与你的第一个核函数。',
    },
    minutes: 30,
    labs: [
      { en: 'Grid/Block Configurator', zh: 'Grid/Block 配置器' },
      { en: 'vecAdd Execution Sim', zh: 'vecAdd 执行模拟器' },
    ],
  },
  {
    id: 'memory',
    num: 4,
    part: 2,
    title: { en: 'Memory Is King', zh: '访存为王' },
    titleEn: 'MEMORY IS KING',
    tagline: { en: 'Coalescing, shared memory, and bank conflicts', zh: '合并访存、Shared Memory 与 bank conflict' },
    summary: {
      en: 'Why most kernels are bottlenecked by memory rather than compute, and how to keep the bandwidth fed.',
      zh: '为什么大多数 kernel 的瓶颈不是计算而是访存，以及如何喂饱显存带宽。',
    },
    minutes: 35,
    labs: [
      { en: 'Coalescing Bench', zh: '合并访存实验台' },
      { en: 'Bank Conflict Checker', zh: 'Bank Conflict 检查器' },
      { en: 'AoS vs SoA', zh: 'AoS vs SoA' },
    ],
  },
  {
    id: 'matmul',
    num: 5,
    part: 2,
    title: { en: 'Making Matmul Fast', zh: '实战：把矩阵乘法做快' },
    titleEn: 'FAST MATMUL',
    tagline: { en: 'The journey from naive to near-cuBLAS', zh: '从 naive 到逼近 cuBLAS 的旅程' },
    summary: {
      en: 'Use tiling, shared memory, and register reuse to speed up GEMM tens of times, one step at a time.',
      zh: '用 tiling、shared memory 与寄存器重用，一步步把 GEMM 加速几十倍。',
    },
    minutes: 40,
    labs: [
      { en: 'Tiling Animation', zh: 'Tiling 动画' },
      { en: 'Memory Traffic Counter', zh: '访存计数器' },
      { en: 'Occupancy Calculator', zh: '占用率计算器' },
    ],
  },
  {
    id: 'roofline',
    num: 6,
    part: 2,
    title: { en: 'Roofline & Profiling', zh: 'Roofline 与性能分析' },
    titleEn: 'ROOFLINE MODEL',
    tagline: { en: 'One chart that reveals any kernel’s bottleneck', zh: '一张图看懂任何 kernel 的瓶颈' },
    summary: {
      en: 'Arithmetic intensity, the memory wall, and the compute wall: use the roofline model to diagnose exactly where your program is stuck.',
      zh: '算术强度、内存墙与计算墙：用 Roofline 模型诊断你的程序到底卡在哪。',
    },
    minutes: 25,
    labs: [
      { en: 'Interactive Roofline', zh: 'Roofline 交互图' },
      { en: 'Arithmetic Intensity Calculator', zh: '算术强度计算器' },
    ],
  },
  {
    id: 'transformer',
    num: 7,
    part: 3,
    title: { en: 'Inside the Transformer', zh: 'Transformer 计算解剖' },
    titleEn: 'INSIDE TRANSFORMERS',
    tagline: { en: 'Where every FLOP of a large model goes', zh: '大模型的每一个 FLOP 花在哪里' },
    summary: {
      en: 'Break one Transformer layer into matrix multiplies: a precise ledger of parameters, FLOPs, and memory.',
      zh: '把一层 Transformer 拆成矩阵乘法：参数量、FLOPs 与显存的精确账本。',
    },
    minutes: 35,
    labs: [
      { en: 'FLOPs Explorer', zh: 'FLOPs 巡览器' },
      { en: 'Attention Memory Growth', zh: 'Attention 显存增长' },
      { en: 'One Layer’s Dataflow', zh: '一层的数据流' },
    ],
  },
  {
    id: 'flash-attention',
    num: 8,
    part: 3,
    title: { en: 'FlashAttention', zh: 'FlashAttention' },
    titleEn: 'FLASH ATTENTION',
    tagline: { en: 'Attention without ever writing the S×S matrix', zh: '不写回 S×S 矩阵，注意力照样算' },
    summary: {
      en: 'Online softmax and tiled computation: the elegant algorithm that cuts attention’s memory from O(S²) to O(S).',
      zh: 'Online softmax 与分块计算：把注意力的显存从 O(S²) 降到 O(S) 的精妙算法。',
    },
    minutes: 35,
    labs: [
      { en: 'Online Softmax Lab', zh: 'Online Softmax 实验' },
      { en: 'Tiled Attention Animation', zh: '分块注意力动画' },
    ],
  },
  {
    id: 'kv-cache',
    num: 9,
    part: 3,
    title: { en: 'KV Cache & PagedAttention', zh: 'KV Cache 与 PagedAttention' },
    titleEn: 'KV CACHE',
    tagline: { en: 'The main battlefield of inference memory', zh: '推理显存的主战场' },
    summary: {
      en: 'Why autoregressive generation depends on the KV cache, and how vLLM ends memory fragmentation with paging.',
      zh: '为什么自回归生成离不开 KV Cache，以及 vLLM 如何用分页管理终结显存碎片。',
    },
    minutes: 30,
    labs: [
      { en: 'KV Cache Calculator', zh: 'KV Cache 计算器' },
      { en: 'PagedAttention Simulator', zh: 'PagedAttention 模拟器' },
    ],
  },
  {
    id: 'serving',
    num: 10,
    part: 3,
    title: { en: 'Serving Systems', zh: '推理服务系统' },
    titleEn: 'SERVING SYSTEMS',
    tagline: { en: 'Continuous batching and the life of a token', zh: '连续批处理与一个 token 的一生' },
    summary: {
      en: 'Prefill, decode, TTFT, and the throughput trade-off: the scheduling core of a modern inference engine.',
      zh: 'Prefill、decode、TTFT 与吞吐的权衡：现代推理引擎的调度核心。',
    },
    minutes: 30,
    labs: [
      { en: 'Continuous Batching Sim', zh: '连续批处理模拟器' },
      { en: 'Prefill vs Decode', zh: 'Prefill/Decode 对比' },
    ],
  },
  {
    id: 'quantization',
    num: 11,
    part: 3,
    title: { en: 'Quantization', zh: '量化' },
    titleEn: 'QUANTIZATION',
    tagline: { en: 'Run a bigger model with fewer bits', zh: '用更少的比特，跑更大的模型' },
    summary: {
      en: 'The bit-level structure of FP16/BF16/FP8/INT4, where quantization error comes from, and why fewer bits speed up decoding.',
      zh: 'FP16/BF16/FP8/INT4 的位级结构、量化误差的来源，以及为什么量化能直接提升解码速度。',
    },
    minutes: 30,
    labs: [
      { en: 'Float Bit Inspector', zh: '浮点位拆解器' },
      { en: 'Quantization Error Playground', zh: '量化误差实验场' },
      { en: 'Memory Savings Calculator', zh: '显存收益计算器' },
    ],
  },
  {
    id: 'parallelism',
    num: 12,
    part: 3,
    title: { en: 'Distributed Parallelism', zh: '分布式并行' },
    titleEn: 'PARALLELISM',
    tagline: { en: 'What to do when one card isn’t enough', zh: '一张卡放不下，怎么办' },
    summary: {
      en: 'Data, tensor, and pipeline parallelism plus ZeRO: the art of slicing trillion-parameter models.',
      zh: '数据并行、张量并行、流水线并行与 ZeRO：万亿参数模型的切分艺术。',
    },
    minutes: 35,
    labs: [
      { en: 'Parallelism Sandbox', zh: '并行策略沙盘' },
      { en: 'Ring AllReduce Animation', zh: 'Ring AllReduce 动画' },
      { en: 'Pipeline Bubble', zh: '流水线气泡' },
    ],
  },
]

export function getChapter(id: string): ChapterMeta | undefined {
  return CHAPTERS.find((c) => c.id === id)
}

export function getPart(num: 1 | 2 | 3): PartMeta {
  return PARTS[num - 1]
}

export function prevNext(id: string): { prev?: ChapterMeta; next?: ChapterMeta } {
  const i = CHAPTERS.findIndex((c) => c.id === id)
  if (i < 0) return {}
  return { prev: CHAPTERS[i - 1], next: CHAPTERS[i + 1] }
}
