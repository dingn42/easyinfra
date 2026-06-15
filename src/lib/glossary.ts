import type { Loc } from './i18n'

/**
 * 术语表数据 —— 全站关键术语的双语定义 + 引入它的章节。
 * Glossary: bilingual definitions for the course's key terms, each linked to
 * the chapter that introduces it. Grouped by theme.
 */

export interface GlossaryTerm {
  /** 锚点用的稳定 slug */
  slug: string
  /** 显示术语（含中文译名/英文原词） */
  term: Loc
  /** 缩写（可选），如 SM / GQA */
  abbr?: string
  /** 一两句定义 */
  def: Loc
  /** 引入该术语的章节序号（链到该章） */
  chapter: number
}

export interface GlossaryGroup {
  title: Loc
  terms: GlossaryTerm[]
}

export const GLOSSARY: GlossaryGroup[] = [
  {
    title: { en: 'GPU Hardware', zh: 'GPU 硬件' },
    terms: [
      {
        slug: 'sm',
        term: { en: 'Streaming Multiprocessor', zh: '流多处理器 (SM)' },
        abbr: 'SM',
        def: {
          en: 'The GPU’s basic compute building block. A modern GPU stamps out 100+ identical SMs; each holds warp schedulers, CUDA cores, Tensor Cores, a register file, and L1/shared memory. A thread block runs entirely on one SM.',
          zh: 'GPU 的基本计算单元。一块现代 GPU 复制出 100+ 个相同的 SM，每个含 warp 调度器、CUDA 核、Tensor Core、寄存器堆与 L1/共享内存。一个线程块完整地跑在一个 SM 上。',
        },
        chapter: 2,
      },
      {
        slug: 'warp',
        term: { en: 'Warp', zh: '线程束 (warp)' },
        def: {
          en: '32 threads that execute in lockstep — the same instruction in the same cycle. The hardware’s real scheduling unit; you write scalar threads, but they run 32 at a time.',
          zh: '32 个线程锁步执行——同一周期跑同一条指令。这是硬件真正的调度单位：你写的是标量线程，但它们 32 个一组一起跑。',
        },
        chapter: 2,
      },
      {
        slug: 'simt',
        term: { en: 'SIMT', zh: 'SIMT（单指令多线程）' },
        abbr: 'SIMT',
        def: {
          en: 'Single-Instruction, Multiple-Thread: a warp’s 32 threads share one instruction stream but each has its own registers. When threads take different branches, both paths run serially under a mask (divergence).',
          zh: '单指令多线程：一个 warp 的 32 个线程共享一条指令流，但各有自己的寄存器。当线程走不同分支时，两条路径在掩码下串行执行（分化）。',
        },
        chapter: 2,
      },
      {
        slug: 'tensor-core',
        term: { en: 'Tensor Core', zh: 'Tensor Core' },
        def: {
          en: 'A dedicated unit that does a small matrix multiply-accumulate in one instruction. It dwarfs the FP32 CUDA cores (H100: 989 BF16 TFLOPS vs 67 FP32) and is why modern training/inference is matmul-bound on Tensor Cores.',
          zh: '专用单元，一条指令完成一个小矩阵乘加。算力远超 FP32 CUDA 核（H100：989 BF16 TFLOPS vs 67 FP32），是现代训练/推理把矩阵乘压在 Tensor Core 上的原因。',
        },
        chapter: 2,
      },
      {
        slug: 'hbm',
        term: { en: 'High-Bandwidth Memory', zh: '高带宽显存 (HBM)' },
        abbr: 'HBM',
        def: {
          en: 'The GPU’s main memory (VRAM), stacked beside the die for huge bandwidth (A100 ~1.9 TB/s, H100 ~3.35 TB/s). Wide, but high-latency — feeding it efficiently is the central performance problem.',
          zh: 'GPU 的主显存，堆叠在芯片旁以获得极高带宽（A100 约 1.9 TB/s，H100 约 3.35 TB/s）。带宽宽但延迟高——如何高效喂饱它是核心性能问题。',
        },
        chapter: 1,
      },
      {
        slug: 'shared-memory',
        term: { en: 'Shared Memory', zh: '共享内存 (shared memory)' },
        def: {
          en: 'A small, fast, programmer-managed scratchpad on each SM, shared by a block’s threads. Used to stage reused data on-chip and reshape access patterns — the key to tiling.',
          zh: '每个 SM 上一块小而快、由程序员管理的片上暂存区，供一个块内的线程共享。用来把复用数据缓存在片上、改写访问模式——是 tiling 的关键。',
        },
        chapter: 4,
      },
      {
        slug: 'bank-conflict',
        term: { en: 'Bank Conflict', zh: 'Bank 冲突' },
        def: {
          en: 'Shared memory is split into 32 banks. If lanes in a warp hit different addresses in the same bank, those accesses serialize (an N-way conflict costs N×). Padding arrays (e.g. [32][33]) sidesteps it.',
          zh: '共享内存分成 32 个 bank。若一个 warp 的多条 lane 命中同一 bank 的不同地址，这些访问会串行化（N 路冲突慢 N 倍）。给数组加 padding（如 [32][33]）可避开。',
        },
        chapter: 4,
      },
    ],
  },
  {
    title: { en: 'CUDA Programming', zh: 'CUDA 编程' },
    terms: [
      {
        slug: 'kernel',
        term: { en: 'Kernel', zh: '核函数 (kernel)' },
        def: {
          en: 'A function marked `__global__` that runs on the GPU. You write the body once; the hardware launches it across thousands of threads in parallel — the for-loop "disappears."',
          zh: '标了 `__global__`、在 GPU 上运行的函数。你只写一遍函数体，硬件会把它分给成千上万个线程并行执行——for 循环「消失」了。',
        },
        chapter: 3,
      },
      {
        slug: 'grid-block-thread',
        term: { en: 'Grid / Block / Thread', zh: 'Grid / Block / Thread' },
        def: {
          en: 'CUDA’s thread hierarchy. A kernel launches a grid of blocks; each block holds threads. Threads in a block share memory and can sync; blocks are independent, so the program scales to any number of SMs.',
          zh: 'CUDA 的线程层级。一个 kernel 启动一个由 block 组成的 grid，每个 block 含若干 thread。块内线程共享内存、可同步；块间相互独立，因此程序能随 SM 数量自动扩展。',
        },
        chapter: 3,
      },
      {
        slug: 'thread-index',
        term: { en: 'threadIdx / blockIdx', zh: 'threadIdx / blockIdx' },
        def: {
          en: 'Built-in coordinates that tell each thread who it is. The canonical global index is `i = blockIdx.x * blockDim.x + threadIdx.x`, which maps a thread to the data element it processes.',
          zh: '内建坐标，告诉每个线程「我是谁」。经典的全局下标是 `i = blockIdx.x * blockDim.x + threadIdx.x`，把线程映射到它要处理的数据元素。',
        },
        chapter: 3,
      },
      {
        slug: 'syncthreads',
        term: { en: '__syncthreads()', zh: '__syncthreads() 障栅' },
        def: {
          en: 'A barrier: every thread in a block must reach it before any moves on. Required around shared-memory staging so no thread reads data another hasn’t written yet.',
          zh: '一个障栅：块内每个线程都到达后，谁才能继续。在共享内存中转前后必须用它，避免某线程读到别人还没写入的数据。',
        },
        chapter: 4,
      },
      {
        slug: 'coalescing',
        term: { en: 'Memory Coalescing', zh: '合并访存 (coalescing)' },
        def: {
          en: 'When a warp’s 32 threads touch consecutive addresses, the hardware fuses them into the fewest 32-byte transactions (32 contiguous floats = 128 B = 4 transactions). Strided or scattered access multiplies the transactions and wastes bandwidth.',
          zh: '当一个 warp 的 32 条线程访问连续地址时，硬件把它们合并成尽量少的 32 字节事务（32 个连续 float = 128 B = 4 个事务）。跨步或散乱访问会成倍增加事务数、浪费带宽。',
        },
        chapter: 4,
      },
      {
        slug: 'occupancy',
        term: { en: 'Occupancy', zh: '占用率 (occupancy)' },
        def: {
          en: 'The ratio of resident warps to the SM’s maximum. Bounded by registers/thread, shared memory/block, and thread limits. Enough occupancy hides latency — but more is not always better; high-performance GEMM often trades occupancy for registers.',
          zh: '驻留 warp 数与 SM 上限的比值，受寄存器/线程、共享内存/块、线程数上限约束。够用即可隐藏延迟——但并非越高越好，高性能 GEMM 常用占用率换寄存器。',
        },
        chapter: 5,
      },
      {
        slug: 'tiling',
        term: { en: 'Tiling', zh: '分块 (tiling)' },
        def: {
          en: 'Cutting a problem into tiles that fit in shared memory so each loaded value is reused by many threads. In matmul, a T×T tile cuts global-memory traffic by ~T× — the single biggest matmul speedup.',
          zh: '把问题切成能放进共享内存的小块，让每个加载进来的值被许多线程复用。矩阵乘里 T×T 的瓦片把全局访存降低约 T 倍——单项最大的 matmul 提速手段。',
        },
        chapter: 5,
      },
    ],
  },
  {
    title: { en: 'Performance Model', zh: '性能模型' },
    terms: [
      {
        slug: 'arithmetic-intensity',
        term: { en: 'Arithmetic Intensity', zh: '算术强度 (arithmetic intensity)' },
        def: {
          en: 'FLOPs performed per byte moved from memory (FLOP/Byte). The horizontal axis of the Roofline. Low intensity → bandwidth-limited; high intensity → compute-limited.',
          zh: '每从内存搬一个字节所做的浮点运算数（FLOP/Byte），是 Roofline 的横轴。强度低 → 受带宽限制；强度高 → 受算力限制。',
        },
        chapter: 6,
      },
      {
        slug: 'roofline',
        term: { en: 'Roofline Model', zh: 'Roofline 模型' },
        def: {
          en: 'One log-log chart that bounds achievable performance: a slanted bandwidth ceiling meets a flat compute ceiling. Plot a kernel’s arithmetic intensity to see instantly whether it’s memory- or compute-bound, and how far it is from the roof.',
          zh: '一张 log-log 图给出可达性能上限：倾斜的带宽屋顶接上水平的算力屋顶。把 kernel 的算术强度画上去，立刻看出它是 memory-bound 还是 compute-bound、离屋顶还差多远。',
        },
        chapter: 6,
      },
      {
        slug: 'ridge-point',
        term: { en: 'Ridge Point', zh: '机器平衡点 (ridge point)' },
        def: {
          en: 'Where the Roofline’s bandwidth slope meets its compute ceiling: ridge = peak FLOPS ÷ bandwidth (A100 BF16 ≈ 164 FLOP/B). A kernel must exceed this intensity to be compute-bound. A property of the machine, not your code.',
          zh: 'Roofline 上带宽斜线与算力平台的交点：ridge = 峰值算力 ÷ 带宽（A100 BF16 ≈ 164 FLOP/B）。kernel 的强度超过它才可能 compute-bound。它是机器的属性，与你的代码无关。',
        },
        chapter: 6,
      },
      {
        slug: 'memory-compute-bound',
        term: { en: 'Memory-bound / Compute-bound', zh: 'Memory-bound / Compute-bound' },
        def: {
          en: 'Whether runtime is dominated by moving data or by doing math. Below the ridge a kernel is memory-bound (the fix is fewer/bigger transfers or higher reuse); above it, compute-bound (the fix is more FLOPS, e.g. Tensor Cores).',
          zh: '运行时间是被搬数据还是被算数主导。低于 ridge 是 memory-bound（解法：减少/增大传输或提高复用）；高于 ridge 是 compute-bound（解法：堆算力，如 Tensor Core）。',
        },
        chapter: 6,
      },
      {
        slug: 'memory-wall',
        term: { en: 'Memory Wall', zh: '内存墙 (memory wall)' },
        def: {
          en: 'The widening gap between how fast cores compute and how fast memory delivers data. Bandwidth can be widened with parallelism; latency barely improves — so most real kernels wait on memory.',
          zh: '核心算得多快与内存供数多快之间不断拉大的差距。带宽能靠并行做宽，延迟却几乎不进步——所以多数真实 kernel 都在等内存。',
        },
        chapter: 1,
      },
      {
        slug: 'latency-hiding',
        term: { en: 'Latency Hiding', zh: '延迟隐藏 (latency hiding)' },
        def: {
          en: 'The GPU’s answer to slow memory: keep many warps resident, and whenever one stalls on data, instantly switch to another that’s ready. Parallelism, not a big cache, hides the latency.',
          zh: 'GPU 应对慢内存的办法：让大量 warp 驻留，一旦某个 warp 因等数据停顿，立刻切到另一个就绪的。靠并行而非大缓存来隐藏延迟。',
        },
        chapter: 2,
      },
      {
        slug: 'mfu',
        term: { en: 'Model FLOPs Utilization', zh: '模型算力利用率 (MFU)' },
        abbr: 'MFU',
        def: {
          en: 'The fraction of a GPU’s peak FLOPS actually used by useful model math. Prefill can hit ~50–70%; single-stream decode is memory-bound and often under 5% — which is what batching fixes.',
          zh: 'GPU 峰值算力中真正用于有用模型计算的比例。prefill 可达约 50–70%；单流 decode 受带宽限制常低于 5%——这正是批处理要解决的。',
        },
        chapter: 10,
      },
      {
        slug: 'amdahl',
        term: { en: 'Amdahl’s Law', zh: '阿姆达尔定律 (Amdahl’s law)' },
        def: {
          en: 'Speedup from parallelism is capped by the serial fraction: speedup = 1 / ((1−p) + p/s). Even a tiny non-parallel part bounds how much the GPU can ever help.',
          zh: '并行带来的加速被串行部分卡住上限：speedup = 1 / ((1−p) + p/s)。哪怕极小的非并行部分，也限制了 GPU 最终能帮上多少忙。',
        },
        chapter: 1,
      },
    ],
  },
  {
    title: { en: 'Transformers & Attention', zh: 'Transformer 与注意力' },
    terms: [
      {
        slug: 'attention',
        term: { en: 'Attention', zh: '注意力 (attention)' },
        def: {
          en: 'Each token forms a query and scores it against every other token’s key (QKᵀ), softmaxes the scores, and takes a weighted sum of values. The S×S score matrix makes naive attention cost and memory grow as O(S²).',
          zh: '每个 token 形成一个 query，与其它所有 token 的 key 打分（QKᵀ），对分数做 softmax，再对 value 加权求和。S×S 的分数矩阵让朴素注意力的开销与显存随 O(S²) 增长。',
        },
        chapter: 7,
      },
      {
        slug: 'online-softmax',
        term: { en: 'Online Softmax', zh: '在线 softmax (online softmax)' },
        def: {
          en: 'A one-pass softmax that processes values in blocks, keeping a running max and sum and rescaling earlier output by exp(m_old − m_new). Exact (not an approximation) — the trick that lets FlashAttention avoid the full score matrix.',
          zh: '单遍 softmax：分块处理，维护运行中的最大值与求和，并用 exp(m_old − m_new) 对已累计输出做修正。它是精确的（非近似）——FlashAttention 借此免于物化整张分数矩阵。',
        },
        chapter: 8,
      },
      {
        slug: 'flash-attention',
        term: { en: 'FlashAttention', zh: 'FlashAttention' },
        def: {
          en: 'A tiled attention algorithm that never writes the S×S matrix to HBM, using online softmax in on-chip SRAM. It does the same FLOPs but cuts memory traffic, dropping attention memory from O(S²) to O(S) and running 2–4× faster.',
          zh: '一种分块注意力算法：借助片上 SRAM 里的在线 softmax，从不把 S×S 矩阵写回 HBM。FLOPs 不变但削减了访存，把注意力显存从 O(S²) 降到 O(S)，快 2–4 倍。',
        },
        chapter: 8,
      },
      {
        slug: 'gqa',
        term: { en: 'MHA / GQA / MQA', zh: '多头 / 分组 / 多查询注意力' },
        abbr: 'GQA',
        def: {
          en: 'Variants that share key/value heads across query heads. MHA: one KV head per query head. GQA: a few KV heads shared by groups. MQA: a single KV head. Fewer KV heads shrink the KV cache by n_heads / n_kv.',
          zh: '在 query 头之间共享 key/value 头的几种变体。MHA：每个 query 头配一个 KV 头；GQA：几个 KV 头被分组共享；MQA：只有一个 KV 头。KV 头越少，KV cache 按 n_heads / n_kv 倍缩小。',
        },
        chapter: 9,
      },
      {
        slug: 'autoregressive',
        term: { en: 'Autoregressive Generation', zh: '自回归生成 (autoregressive)' },
        def: {
          en: 'Generating one token at a time, appending each to the input as the condition for the next. Step t needs the keys/values of all prior tokens — which is exactly why the KV cache exists.',
          zh: '每次只生成一个 token，并把它拼回输入、作为下一步预测的条件。第 t 步需要前面所有 token 的 key/value——这正是 KV cache 存在的理由。',
        },
        chapter: 9,
      },
    ],
  },
  {
    title: { en: 'Inference Systems', zh: '推理系统' },
    terms: [
      {
        slug: 'kv-cache',
        term: { en: 'KV Cache', zh: 'KV 缓存 (KV cache)' },
        def: {
          en: 'The stored keys and values of past tokens, so each decode step recomputes attention in O(S·d) instead of O(S²·d). It trades compute for memory — and that memory becomes inference’s main constraint.',
          zh: '把历史 token 的 key 与 value 存下来，使每步 decode 的注意力从 O(S²·d) 降到 O(S·d)。它用显存换计算——而这块显存成了推理的主要瓶颈。',
        },
        chapter: 9,
      },
      {
        slug: 'paged-attention',
        term: { en: 'PagedAttention', zh: 'PagedAttention（分页注意力）' },
        def: {
          en: 'vLLM’s idea of storing the KV cache in fixed-size blocks (like OS paging) instead of one contiguous slab per request. It nearly eliminates memory fragmentation, raising usable KV memory from ~20–40% to near 100%.',
          zh: 'vLLM 的做法：把 KV cache 存成固定大小的 block（类似操作系统分页），而非每个请求一整段连续显存。它几乎消除碎片，把可用 KV 显存利用率从约 20–40% 提到接近 100%。',
        },
        chapter: 9,
      },
      {
        slug: 'prefill-decode',
        term: { en: 'Prefill / Decode', zh: '预填充 / 解码 (prefill / decode)' },
        def: {
          en: 'The two phases of inference. Prefill processes the whole prompt in parallel (compute-bound, sets TTFT). Decode emits one token at a time (memory-bound, sets the gap between tokens). They have opposite performance profiles.',
          zh: '推理的两个阶段。prefill 并行处理整段 prompt（compute-bound，决定 TTFT）；decode 一次吐一个 token（memory-bound，决定 token 间隔）。二者性能特征相反。',
        },
        chapter: 10,
      },
      {
        slug: 'continuous-batching',
        term: { en: 'Continuous Batching', zh: '连续批处理 (continuous batching)' },
        def: {
          en: 'Scheduling at the token level instead of the request level: as soon as one sequence finishes, a waiting request fills its slot — no waiting for the whole batch. The key to high decode throughput.',
          zh: '以 token 为粒度而非请求为粒度调度：某条序列一完成，等待中的请求立刻补上它的槽位，不必等整批结束。是 decode 高吞吐的关键。',
        },
        chapter: 10,
      },
      {
        slug: 'ttft-tpot',
        term: { en: 'TTFT / TPOT', zh: '首 token 延迟 / 每 token 间隔' },
        abbr: 'TTFT/TPOT',
        def: {
          en: 'The two latency metrics users feel. TTFT (Time To First Token) is set by prefill; TPOT (Time Per Output Token) by decode. Larger batches raise throughput but worsen TPOT — a tuning trade-off against an SLO.',
          zh: '用户最能感知的两个延迟指标。TTFT（首 token 延迟）由 prefill 决定，TPOT（每输出 token 间隔）由 decode 决定。增大 batch 提吞吐但拖差 TPOT——要按 SLO 权衡调参。',
        },
        chapter: 10,
      },
      {
        slug: 'speculative-decoding',
        term: { en: 'Speculative Decoding', zh: '投机解码 (speculative decoding)' },
        def: {
          en: 'A small draft model proposes several tokens; the big model verifies them in one parallel pass, accepting the longest correct prefix. Turns serial decode into parallel verification, often 2–3× faster with identical outputs.',
          zh: '用小的草稿模型一次提议多个 token，大模型用一次并行前向验证，接受最长的正确前缀。把串行 decode 变成并行验证，常快 2–3 倍且输出不变。',
        },
        chapter: 10,
      },
    ],
  },
  {
    title: { en: 'Quantization & Parallelism', zh: '量化与并行' },
    terms: [
      {
        slug: 'quantization',
        term: { en: 'Quantization', zh: '量化 (quantization)' },
        def: {
          en: 'Storing weights/activations in fewer bits (FP16→INT8/INT4). For memory-bound decode, halving the bytes roughly doubles speed — the win is moving fewer bytes, not faster math.',
          zh: '用更少的比特存权重/激活（FP16→INT8/INT4）。对 memory-bound 的 decode，字节减半≈速度翻倍——收益来自搬的字节更少，而非乘法更快。',
        },
        chapter: 11,
      },
      {
        slug: 'float-formats',
        term: { en: 'FP16 / BF16 / FP8 / INT4', zh: 'FP16 / BF16 / FP8 / INT4' },
        def: {
          en: 'Number formats trading bits for range/precision. BF16 keeps FP32’s exponent range but fewer mantissa bits; FP8 (E4M3/E5M2) and INT4 push further for inference. Fewer bits = less memory and bandwidth.',
          zh: '用比特换范围/精度的数值格式。BF16 保留 FP32 的指数范围但尾数更少；FP8（E4M3/E5M2）与 INT4 在推理中进一步压缩。位数越少，显存与带宽越省。',
        },
        chapter: 11,
      },
      {
        slug: 'absmax-quant',
        term: { en: 'Absmax / Scale', zh: 'Absmax 量化 / 缩放因子' },
        def: {
          en: 'Symmetric quantization maps reals to integers with q = round(x / s), where the scale s = max|x| / qmax. A single outlier inflates s and crushes precision for everything else — which is why per-group/per-channel scales and outlier handling matter.',
          zh: '对称量化用 q = round(x / s) 把实数映射为整数，缩放因子 s = max|x| / qmax。单个离群值会撑大 s、压垮其余值的精度——所以才需要分组/分通道缩放与离群值处理。',
        },
        chapter: 11,
      },
      {
        slug: 'parallelism-types',
        term: { en: 'Data / Tensor / Pipeline Parallelism', zh: '数据 / 张量 / 流水线并行' },
        def: {
          en: 'Three ways to split training across GPUs. Data: replicate the model, split the batch. Tensor: split each layer’s matrices (chatty — needs NVLink). Pipeline: split layers into stages. Real runs combine all three.',
          zh: '把训练切到多卡的三种方式。数据并行：复制模型、切分 batch；张量并行：切开每层的矩阵（通信频繁，需 NVLink）；流水线并行：把层分成多段。真实训练三者并用。',
        },
        chapter: 12,
      },
      {
        slug: 'zero',
        term: { en: 'ZeRO', zh: 'ZeRO（零冗余优化）' },
        abbr: 'ZeRO',
        def: {
          en: 'Shards the optimizer states, then gradients, then parameters across data-parallel GPUs (stages 1/2/3), gathering them only when needed. Removes the memory redundancy of plain data parallelism so huge models fit.',
          zh: '把优化器状态、梯度、参数依次切分到数据并行的各卡（stage 1/2/3），用时再 all-gather。消除朴素数据并行的显存冗余，让超大模型放得下。',
        },
        chapter: 12,
      },
      {
        slug: 'ring-allreduce',
        term: { en: 'Ring All-Reduce', zh: '环形 All-Reduce (ring all-reduce)' },
        def: {
          en: 'A bandwidth-optimal way to sum gradients across N GPUs arranged in a ring: 2(N−1) steps, with per-GPU traffic ≈ 2× the data regardless of N. The workhorse collective behind data parallelism.',
          zh: '把 N 张卡排成环来求和梯度的带宽最优方法：2(N−1) 步，每卡通信量约为数据量的 2 倍且与 N 无关。是数据并行背后的主力集合通信。',
        },
        chapter: 12,
      },
      {
        slug: 'pipeline-bubble',
        term: { en: 'Pipeline Bubble', zh: '流水线气泡 (pipeline bubble)' },
        def: {
          en: 'The idle time at the start/end of pipeline parallelism while stages fill and drain. Bubble fraction = (P−1) / (M + P−1) for P stages and M micro-batches — more micro-batches amortize it away.',
          zh: '流水线并行在启动/收尾阶段、各段填充与排空时的空转。气泡率 = (P−1) / (M + P−1)（P 段、M 个 micro-batch）——micro-batch 越多越能摊薄。',
        },
        chapter: 12,
      },
      {
        slug: 'nvlink',
        term: { en: 'NVLink', zh: 'NVLink' },
        def: {
          en: 'NVIDIA’s high-bandwidth GPU-to-GPU interconnect within a node (A100 600 GB/s, H100 900 GB/s) — roughly 12–18× faster than cross-node InfiniBand, which is why chatty tensor parallelism stays inside an NVLink domain.',
          zh: 'NVIDIA 节点内 GPU 互连（A100 600 GB/s，H100 900 GB/s），约比跨节点 InfiniBand 快 12–18 倍——这就是通信频繁的张量并行要留在一个 NVLink 域内的原因。',
        },
        chapter: 12,
      },
    ],
  },
]

/** 扁平化所有术语（搜索用） */
export const ALL_TERMS: GlossaryTerm[] = GLOSSARY.flatMap((g) => g.terms)
