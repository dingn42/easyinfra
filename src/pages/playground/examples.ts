/**
 * Playground 示例与挑战定义（纯数据，UI 无关）。
 * 注意：用相对路径导入引擎类型，便于脱离 Vite 别名做脚本级验证。
 */
import type { BufferDecl, KernelParam } from '../../lib/cudasim/types'

export interface ExampleDef {
  id: string
  title: string
  badge: string
  /** 卡片上的一句话 */
  blurb: string
  /** 「观察什么」提示 */
  hint: string
  source: string
  grid: number
  block: number
  buffers: BufferDecl[]
  scalars: Record<string, number>
}

export interface ChallengeDef {
  id: string
  num: number
  title: string
  desc: string
  goal: string
  source: string
  /** 正确参考实现：与用户同配置运行，输出逐元素比对 */
  referenceSource: string
  /** 需要比对的输出缓冲区 */
  compareBuffers: string[]
  /** 可选：通关还要求全局事务数不超过该值（访存优化类挑战） */
  maxGlobalTransactions?: number
  grid: number
  block: number
  buffers: BufferDecl[]
  scalars: Record<string, number>
}

/* ──────────────────────── 示例 ──────────────────────── */

export const EXAMPLES: ExampleDef[] = [
  {
    id: 'vecadd',
    title: 'vecAdd',
    badge: '入门',
    blurb: '逐元素向量加法，CUDA 的 Hello World',
    hint: '看 BUFFERS 里 C 的 volt 高亮（前 1000 个被写入）；ACCESS MAP 中读写呈整齐竖条 —— 这就是完美合并访存的样子。',
    source: `__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}
`,
    grid: 4,
    block: 256,
    buffers: [
      { name: 'A', length: 1000, init: 'iota' },
      { name: 'B', length: 1000, init: 'random' },
      { name: 'C', length: 1000, init: 'zero' },
    ],
    scalars: { n: 1000 },
  },
  {
    id: 'reverse',
    title: '数组反转',
    badge: 'shared + sync',
    blurb: '经 shared memory 中转，两次 __syncthreads()',
    hint: 'STATS 里 BARRIERS = 2；ACCESS MAP 中 amber 色块是 shared 访问，夹在两次全局读写之间。试着删掉一个 __syncthreads() 再跑。',
    source: `__global__ void reverse(float* A, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  s[t] = A[t];
  __syncthreads();
  float v = s[n - 1 - t];
  __syncthreads();
  A[t] = v;
}
`,
    grid: 1,
    block: 256,
    buffers: [{ name: 'A', length: 256, init: 'iota' }],
    scalars: { n: 256 },
  },
  {
    id: 'stride',
    title: '跨步访问',
    badge: '访存合并',
    blurb: '把 stride 从 1 调到 8，看事务数翻倍',
    hint: '改 stride 标量再 RUN：GLOBAL TXN 从 16（stride=1，完美合并）→ 24 → 40 → 72（stride=8）；合并效率 100% → 67% → 40% → 22%。写 B 的 8 个事务不变，翻倍的全是读 A。',
    source: `__global__ void strided(const float* A, float* B, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  /* 相邻线程读取相距 stride 的元素：
     stride=1 时一拍 32 个读落在 4 个 32B 段里；
     stride 翻倍，触及的段数也翻倍 */
  B[i] = A[i * stride];
}
`,
    grid: 1,
    block: 64,
    buffers: [
      { name: 'A', length: 512, init: 'iota' },
      { name: 'B', length: 64, init: 'zero' },
    ],
    scalars: { stride: 2 },
  },
  {
    id: 'bank',
    title: 'bank conflict',
    badge: 'shared 冲突',
    blurb: '同 warp 多线程撞同一个 bank',
    hint: 'BANK CONFLICTS 显示最坏 bank 的「冲突路数 − 1」：stride=1 → 0；stride=2 → 1（2 路）；stride=4 → 3（4 路，lane 0/8/16/24 全撞 bank 0）。冲突路数 = 这次访问要串行的拍数。',
    source: `__global__ void bankDemo(float* out, int stride) {
  __shared__ float s[128];
  int i = threadIdx.x;
  for (int k = 0; k < 4; k++) {
    s[k * 32 + i] = k * 32 + i;
  }
  __syncthreads();
  /* bank = index % 32：stride=4 时 lane 0/8/16/24
     读的地址都落在 bank 0，硬件只能串行 4 拍 */
  out[i] = s[(i * stride) % 128];
}
`,
    grid: 1,
    block: 32,
    buffers: [{ name: 'out', length: 32, init: 'zero' }],
    scalars: { stride: 2 },
  },
  {
    id: 'diverge',
    title: '分支分化',
    badge: 'warp divergence',
    blurb: '同 warp 两条路径，谁等谁？',
    hint: 'STATS 的 DIVERGENT BRANCHES > 0；ACCESS MAP 里上半 warp（lane<16）的写入很早完成，下半被 for 循环拖到很晚 —— 同 warp 不再同拍。',
    source: `__global__ void diverge(float* out) {
  int lane = threadIdx.x % warpSize;
  if (lane < 16) {
    out[threadIdx.x] = lane * 2.0f;
  } else {
    /* 下半 warp 走一条更长的路 */
    float v = 0.0f;
    for (int k = 0; k < lane; k++) {
      v = v + 0.5f;
    }
    out[threadIdx.x] = v;
  }
}
`,
    grid: 1,
    block: 64,
    buffers: [{ name: 'out', length: 64, init: 'zero' }],
    scalars: {},
  },
  {
    id: 'matmul',
    title: '4×4 tiled matmul',
    badge: '综合',
    blurb: '把整块矩阵搬进 shared 再做乘法',
    hint: 'C = A × B（行主序 4×4）。BUFFERS 里核对 C[0] = 0·0+1·4+2·8+3·12 = 56；ACCESS MAP 能看到「先全局读 → 屏障 → 反复 shared 读」的节奏。',
    source: `__global__ void matmul(const float* A, const float* B, float* C, int n) {
  __shared__ float sA[16];
  __shared__ float sB[16];
  int t = threadIdx.x;
  int row = t / n;
  int col = t % n;
  sA[t] = A[t];
  sB[t] = B[t];
  __syncthreads();
  float acc = 0.0f;
  for (int k = 0; k < n; k++) {
    acc += sA[row * n + k] * sB[k * n + col];
  }
  C[t] = acc;
}
`,
    grid: 1,
    block: 16,
    buffers: [
      { name: 'A', length: 16, init: 'iota' },
      { name: 'B', length: 16, init: 'iota' },
      { name: 'C', length: 16, init: 'zero' },
    ],
    scalars: { n: 4 },
  },
]

/* ──────────────────────── 挑战 ──────────────────────── */

export const CHALLENGES: ChallengeDef[] = [
  {
    id: 'ch-bounds',
    num: 1,
    title: '修复越界',
    desc: '这个 vecAdd 启动了 4×256 = 1024 个线程，但数组只有 1000 个元素。RUN 一下，看哪个线程先撞墙。',
    goal: '加上边界检查，让输出 C 与参考实现一致。',
    source: `__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
  /* BUG：线程数 1024 > n=1000，越界只是时间问题 */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}
`,
    referenceSource: `__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}
`,
    compareBuffers: ['C'],
    grid: 4,
    block: 256,
    buffers: [
      { name: 'A', length: 1000, init: 'iota' },
      { name: 'B', length: 1000, init: 'random' },
      { name: 'C', length: 1000, init: 'zero' },
    ],
    scalars: { n: 1000 },
  },
  {
    id: 'ch-sync',
    num: 2,
    title: '消失的屏障',
    desc: '这个反转 kernel 里，奇数线程走了一条更慢的路才写入 shared。偶数线程不等它们，直接去读 —— 读到的是没写过的 0。',
    goal: '在正确的位置补上 __syncthreads()，让 A 变成完美的逆序。',
    source: `__global__ void reverse(float* A, int n) {
  __shared__ float s[64];
  int t = threadIdx.x;
  if (t % 2 == 0) {
    s[t] = A[t];
  } else {
    /* 奇数线程多绕两步才写入 */
    float v = A[t];
    v = v + 0.0f;
    s[t] = v;
  }
  /* BUG：这里少了一次 __syncthreads() */
  A[t] = s[n - 1 - t];
}
`,
    referenceSource: `__global__ void reverse(float* A, int n) {
  __shared__ float s[64];
  int t = threadIdx.x;
  if (t % 2 == 0) {
    s[t] = A[t];
  } else {
    float v = A[t];
    v = v + 0.0f;
    s[t] = v;
  }
  __syncthreads();
  A[t] = s[n - 1 - t];
}
`,
    compareBuffers: ['A'],
    grid: 1,
    block: 64,
    buffers: [{ name: 'A', length: 64, init: 'iota' }],
    scalars: { n: 64 },
  },
  {
    id: 'ch-coalesce',
    num: 3,
    title: '合并访存',
    desc: '每个线程乘 2 并搬运 4 个「连续」元素 —— 结果是对的，但一拍之内 32 个 lane 散落在 16 个内存段上，事务数是理想值的 4 倍。',
    goal: '改成跨线程连续的索引（如 idx = k * (n / 4) + t），输出不变、GLOBAL TXN ≤ 128。',
    source: `__global__ void scaleCopy(const float* A, float* B, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < 4; k++) {
    /* BUG（性能）：thread0 摸 0..3，thread1 摸 4..7 ……
       同一拍里 warp 的 32 次访问横跨 128 个元素 */
    int idx = t * 4 + k;
    if (idx < n) {
      B[idx] = A[idx] * 2.0f;
    }
  }
}
`,
    referenceSource: `__global__ void scaleCopy(const float* A, float* B, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < 4; k++) {
    int idx = k * (n / 4) + t;
    if (idx < n) {
      B[idx] = A[idx] * 2.0f;
    }
  }
}
`,
    compareBuffers: ['B'],
    maxGlobalTransactions: 128,
    grid: 1,
    block: 64,
    buffers: [
      { name: 'A', length: 256, init: 'iota' },
      { name: 'B', length: 256, init: 'zero' },
    ],
    scalars: { n: 256 },
  },
]

/* ──────────────────────── 初值复算 ──────────────────────── */

/**
 * 复刻引擎的缓冲区初始化（含全局共享的 seed=42 LCG 与 int 截断），
 * 用于 BUFFERS 视图里「与初值不同」的 diff 高亮。
 */
export function materializeInitial(
  decls: BufferDecl[],
  params: KernelParam[],
): Record<string, number[]> {
  let s = 42 >>> 0
  const lcg = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
  const elemOf = new Map<string, string>()
  for (const p of params) if (p.isPointer) elemOf.set(p.name, p.type.startsWith('int') ? 'int' : 'float')
  const out: Record<string, number[]> = {}
  for (const d of decls) {
    const data = new Array<number>(Math.max(0, d.length))
    const init = d.init ?? 'zero'
    if (init === 'zero') data.fill(0)
    else if (init === 'iota') for (let i = 0; i < d.length; i++) data[i] = i
    else if (init === 'random') for (let i = 0; i < d.length; i++) data[i] = lcg()
    else for (let i = 0; i < d.length; i++) data[i] = init[i] ?? 0
    if (elemOf.get(d.name) === 'int') for (let i = 0; i < data.length; i++) data[i] = Math.trunc(data[i])
    out[d.name] = data
  }
  return out
}
