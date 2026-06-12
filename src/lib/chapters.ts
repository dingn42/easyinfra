/** 课程目录元数据 —— 全站唯一的章节注册表（组件懒加载见 src/chapters/index.ts） */

export interface PartMeta {
  num: 1 | 2 | 3
  label: string
  title: string
  titleEn: string
  blurb: string
}

export interface ChapterMeta {
  /** URL id，如 /learn/why-gpu */
  id: string
  /** 目录序号 1-12 */
  num: number
  part: 1 | 2 | 3
  title: string
  titleEn: string
  /** 一句话副标题 */
  tagline: string
  /** 目录页摘要 */
  summary: string
  /** 预估阅读分钟数 */
  minutes: number
  /** 本章交互实验名称列表 */
  labs: string[]
}

export const PARTS: PartMeta[] = [
  {
    num: 1,
    label: '第一部分',
    title: 'GPU 与并行计算',
    titleEn: 'PARALLEL FOUNDATIONS',
    blurb: '在写下任何一行 CUDA 之前，先理解这台机器为什么长这样。',
  },
  {
    num: 2,
    label: '第二部分',
    title: 'CUDA 编程',
    titleEn: 'CUDA PROGRAMMING',
    blurb: '从第一个 kernel 到逼近 cuBLAS：编程模型、内存系统与优化方法论。',
  },
  {
    num: 3,
    label: '第三部分',
    title: '大模型推理系统',
    titleEn: 'LLM INFERENCE SYSTEMS',
    blurb: 'FlashAttention、KV Cache、连续批处理、量化与分布式并行 —— 生产级 LLM Infra 的核心思想。',
  },
]

export const CHAPTERS: ChapterMeta[] = [
  {
    id: 'why-gpu',
    num: 1,
    part: 1,
    title: '为什么需要 GPU',
    titleEn: 'WHY GPUS',
    tagline: '从「延迟机器」到「吞吐机器」',
    summary: 'CPU 与 GPU 的设计哲学差异、并行计算的基本直觉，以及大模型为什么离不开 GPU。',
    minutes: 25,
    labs: ['吞吐量赛道', '晶体管预算分配器'],
  },
  {
    id: 'gpu-architecture',
    num: 2,
    part: 1,
    title: 'GPU 硬件解剖',
    titleEn: 'GPU ANATOMY',
    tagline: '从 die 到 SM，从 warp 到核心',
    summary: '逐层拆开一块现代 GPU：GPC、SM、warp 调度器、Tensor Core 与显存层级。',
    minutes: 30,
    labs: ['GPU 解剖台', '显存层级延迟实验', 'Warp 延迟隐藏'],
  },
  {
    id: 'cuda-model',
    num: 3,
    part: 2,
    title: 'CUDA 编程模型',
    titleEn: 'THE CUDA MODEL',
    tagline: '你写一个函数，硬件运行一百万次',
    summary: 'kernel、grid、block、thread：CUDA 的执行模型与你的第一个核函数。',
    minutes: 30,
    labs: ['Grid/Block 配置器', 'vecAdd 执行模拟器'],
  },
  {
    id: 'memory',
    num: 4,
    part: 2,
    title: '访存为王',
    titleEn: 'MEMORY IS KING',
    tagline: '合并访存、Shared Memory 与 bank conflict',
    summary: '为什么大多数 kernel 的瓶颈不是计算而是访存，以及如何喂饱显存带宽。',
    minutes: 35,
    labs: ['合并访存实验台', 'Bank Conflict 检查器', 'AoS vs SoA'],
  },
  {
    id: 'matmul',
    num: 5,
    part: 2,
    title: '实战：把矩阵乘法做快',
    titleEn: 'FAST MATMUL',
    tagline: '从 naive 到逼近 cuBLAS 的旅程',
    summary: '用 tiling、shared memory 与寄存器重用，一步步把 GEMM 加速几十倍。',
    minutes: 40,
    labs: ['Tiling 动画', '访存计数器', '占用率计算器'],
  },
  {
    id: 'roofline',
    num: 6,
    part: 2,
    title: 'Roofline 与性能分析',
    titleEn: 'ROOFLINE MODEL',
    tagline: '一张图看懂任何 kernel 的瓶颈',
    summary: '算术强度、内存墙与计算墙：用 Roofline 模型诊断你的程序到底卡在哪。',
    minutes: 25,
    labs: ['Roofline 交互图', '算术强度计算器'],
  },
  {
    id: 'transformer',
    num: 7,
    part: 3,
    title: 'Transformer 计算解剖',
    titleEn: 'INSIDE TRANSFORMERS',
    tagline: '大模型的每一个 FLOP 花在哪里',
    summary: '把一层 Transformer 拆成矩阵乘法：参数量、FLOPs 与显存的精确账本。',
    minutes: 35,
    labs: ['FLOPs 巡览器', 'Attention 显存增长', '一层的数据流'],
  },
  {
    id: 'flash-attention',
    num: 8,
    part: 3,
    title: 'FlashAttention',
    titleEn: 'FLASH ATTENTION',
    tagline: '不写回 S×S 矩阵，注意力照样算',
    summary: 'Online softmax 与分块计算：把注意力的显存从 O(S²) 降到 O(S) 的精妙算法。',
    minutes: 35,
    labs: ['Online Softmax 实验', '分块注意力动画'],
  },
  {
    id: 'kv-cache',
    num: 9,
    part: 3,
    title: 'KV Cache 与 PagedAttention',
    titleEn: 'KV CACHE',
    tagline: '推理显存的主战场',
    summary: '为什么自回归生成离不开 KV Cache，以及 vLLM 如何用分页管理终结显存碎片。',
    minutes: 30,
    labs: ['KV Cache 计算器', 'PagedAttention 模拟器'],
  },
  {
    id: 'serving',
    num: 10,
    part: 3,
    title: '推理服务系统',
    titleEn: 'SERVING SYSTEMS',
    tagline: '连续批处理与一个 token 的一生',
    summary: 'Prefill、decode、TTFT 与吞吐的权衡：现代推理引擎的调度核心。',
    minutes: 30,
    labs: ['连续批处理模拟器', 'Prefill/Decode 对比'],
  },
  {
    id: 'quantization',
    num: 11,
    part: 3,
    title: '量化',
    titleEn: 'QUANTIZATION',
    tagline: '用更少的比特，跑更大的模型',
    summary: 'FP16/BF16/FP8/INT4 的位级结构、量化误差的来源，以及为什么量化能直接提升解码速度。',
    minutes: 30,
    labs: ['浮点位拆解器', '量化误差实验场', '显存收益计算器'],
  },
  {
    id: 'parallelism',
    num: 12,
    part: 3,
    title: '分布式并行',
    titleEn: 'PARALLELISM',
    tagline: '一张卡放不下，怎么办',
    summary: '数据并行、张量并行、流水线并行与 ZeRO：万亿参数模型的切分艺术。',
    minutes: 35,
    labs: ['并行策略沙盘', 'Ring AllReduce 动画', '流水线气泡'],
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
