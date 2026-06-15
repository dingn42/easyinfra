import { Callout, ChapterLink, CodeBlock, Figure, MathTex, Quiz, Section, Term } from '@/components/ui'
import { pick, useLocale, useT, type Loc } from '@/lib/i18n'
import { CoalescingLab } from './CoalescingLab'
import { AosSoaLab } from './AosSoaLab'
import { BankConflictLab } from './BankConflictLab'

const HOOK_CODE_EN = `// kernel A: each warp fires 4 32B transactions, 100% bandwidth utilization
out[i] = a[i] * 2.0f;

// kernel B: identical instruction count, but 32 transactions, 12.5% utilization
out[i] = a[i * 8] * 2.0f;`
const HOOK_CODE_ZH = `// kernel A：每个 warp 触发 4 个 32B 事务，带宽利用率 100%
out[i] = a[i] * 2.0f;

// kernel B：指令数一模一样，却是 32 个事务，利用率 12.5%
out[i] = a[i * 8] * 2.0f;`

const AOS_CODE_EN = `// AoS (Array of Structures): pack each particle into a struct, then array them
struct Particle { float x, y, z; };          // 12 B apiece

__global__ void scale_x(float* out, const Particle* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = p[i].x * 2.0f;         // neighboring threads are 12 B apart
}`
const AOS_CODE_ZH = `// AoS（Array of Structures）：粒子打包成结构体，再排成数组
struct Particle { float x, y, z; };          // 12 B 一个

__global__ void scale_x(float* out, const Particle* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = p[i].x * 2.0f;         // 相邻线程的地址差 12 B
}`

const SOA_CODE_EN = `// SoA (Structure of Arrays): one independent array per field
struct Particles { float *x, *y, *z; };

__global__ void scale_x(float* out, Particles p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = p.x[i] * 2.0f;         // neighbors are 4 B apart -> perfectly coalesced
}`
const SOA_CODE_ZH = `// SoA（Structure of Arrays）：每个字段一条独立数组
struct Particles { float *x, *y, *z; };

__global__ void scale_x(float* out, Particles p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = p.x[i] * 2.0f;         // 相邻线程的地址差 4 B → 完美合并
}`

const TRANSPOSE_CODE_EN = `__global__ void transpose(float* out, const float* in, int n) {
  __shared__ float tile[32][33];             // 33 = 32 + 1 pad column; see section 5
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (x < n && y < n)
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];   // read global by row -> coalesced
  __syncthreads();                           // barrier: wait for the whole block to finish writing shared
  x = blockIdx.y * 32 + threadIdx.x;         // note: the block coordinates are swapped
  y = blockIdx.x * 32 + threadIdx.y;
  if (x < n && y < n)
    out[y * n + x] = tile[threadIdx.x][threadIdx.y];  // write global by row too -> coalesced
}`
const TRANSPOSE_CODE_ZH = `__global__ void transpose(float* out, const float* in, int n) {
  __shared__ float tile[32][33];             // 33 = 32 + 1 列 padding，第 5 节揭晓
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (x < n && y < n)
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];   // 读全局内存：按行 → 合并 ✓
  __syncthreads();                           // 屏障：等全 block 写完 shared 再开始读
  x = blockIdx.y * 32 + threadIdx.x;         // 注意：block 坐标交换了
  y = blockIdx.x * 32 + threadIdx.y;
  if (x < n && y < n)
    out[y * n + x] = tile[threadIdx.x][threadIdx.y];  // 写全局内存：也按行 → 合并 ✓
}`

const DOUBLE_SYNC_CODE_EN = `for (int t = 0; t < numTiles; ++t) {
  tile[ty][tx] = A[...];        // write this round's tile into shared
  __syncthreads();              // (1) write-then-read barrier: read only after everyone has written
  for (int k = 0; k < 32; ++k)
    acc += tile[ty][k] * ...;   // the whole block reuses this tile
  __syncthreads();              // (2) read-then-write barrier: wait for everyone to finish reading
}                               //     before a fast thread overwrites with the next round's data`
const DOUBLE_SYNC_CODE_ZH = `for (int t = 0; t < numTiles; ++t) {
  tile[ty][tx] = A[...];        // 把本轮 tile 写进 shared
  __syncthreads();              // ① 写后读屏障：等所有人写完，才能放心读
  for (int k = 0; k < 32; ++k)
    acc += tile[ty][k] * ...;   // 全 block 复用这块 tile
  __syncthreads();              // ② 读后写屏障：等所有人读完，
}                               //    才能让快线程写入下一轮数据`

type HierRow = [Loc, Loc, Loc, Loc, Loc]
const HIER_ROWS: HierRow[] = [
  [
    { en: 'Registers', zh: '寄存器' },
    { en: 'on-chip · per-thread', zh: '片上 · 每线程' },
    { en: '~1 cycle', zh: '~1 cycle' },
    { en: 'thread-private', zh: '线程私有' },
    { en: 'where all computation starts; capped at 255 per thread', zh: '一切计算的起点；每线程上限 255 个' },
  ],
  [
    { en: 'Local memory', zh: 'Local memory' },
    { en: 'DRAM (yes, off-chip)', zh: '显存（没错，在板上）' },
    { en: '~400+ cycles', zh: '~400+ cycles' },
    { en: 'thread-private', zh: '线程私有' },
    { en: 'where register spills land — the most misleading name', zh: '寄存器溢出（spill）的去处，名字最骗人' },
  ],
  [
    { en: 'Shared memory', zh: 'Shared memory' },
    { en: 'on-chip · per-SM', zh: '片上 · 每 SM' },
    { en: '~20–30 cycles', zh: '~20–30 cycles' },
    { en: 'shared within a block', zh: 'block 内共享' },
    { en: 'a programmable cache — the star of this chapter', zh: '可编程缓存，本章主角' },
  ],
  [
    { en: 'Constant', zh: 'Constant' },
    { en: 'DRAM + dedicated cache', zh: '显存 + 专用缓存' },
    { en: 'register-like on a hit', zh: '命中后近似寄存器' },
    { en: 'grid-wide read-only', zh: '全 grid 只读' },
    { en: 'one broadcast when a whole warp reads the same address', zh: '全 warp 读同一地址时一次广播' },
  ],
  [
    { en: 'L2 cache', zh: 'L2 cache' },
    { en: 'on-chip · device-wide', zh: '片上 · 全卡共享' },
    { en: '~200 cycles', zh: '~200 cycles' },
    { en: 'every access passes through', zh: '所有访问路过' },
    { en: 'transparent cache; 40 MB on A100 / 50 MB on H100', zh: '透明缓存；A100 40 MB / H100 50 MB' },
  ],
  [
    { en: 'Global (HBM)', zh: 'Global (HBM)' },
    { en: 'on-board DRAM', zh: '板上显存' },
    { en: '~400–600 cycles', zh: '~400–600 cycles' },
    { en: 'grid-wide', zh: '全 grid' },
    { en: 'the main arena; ~1.9 TB/s on A100 / 3.35 TB/s on H100', zh: '主战场；A100 约 1.9 TB/s / H100 3.35 TB/s' },
  ],
  [
    { en: 'Pinned host mem', zh: 'Pinned host mem' },
    { en: 'host memory (page-locked)', zh: '主机内存（页锁定）' },
    { en: 'over PCIe / NVLink', zh: '走 PCIe / NVLink' },
    { en: 'Host <-> Device', zh: 'Host ↔ Device' },
    { en: 'prerequisite for overlapping cudaMemcpyAsync with compute', zh: 'cudaMemcpyAsync 重叠拷贝与计算的前提' },
  ],
]

export default function Chapter() {
  const t = useT()
  const { lang } = useLocale()
  return (
    <>
      <p>
        {t(
          <>
            Here's a fact that should bother you. Two kernels, identical instruction count, identical
            floating-point work. They differ in one thing only: the array index. One writes <code>a[i]</code>,
            the other <code>a[i * 8]</code>. Run them and the first is nearly 10x faster. If you're still
            reasoning in <ChapterLink n={3} />'s "thread grid" model, that makes no sense. Same number of threads, same
            number of multiply-adds. Where does an order of magnitude come from? The answer lives on the road
            between the moment a thread computes its index and the moment the data actually lands in a register.
            This chapter stares at that road: how global memory bills you by the <em>transaction</em>, how a
            warp's access pattern decides how much bandwidth you throw away, and how that small, fast slab of
            on-chip shared memory launders a bad access pattern into a good one. The punchline goes first.{' '}
            <strong>Most kernels are bound by memory, not compute</strong>. Learn to read memory access and
            you've got most of GPU performance tuning in hand.
          </>,
          <>
            先看一个本该让你坐立不安的现象。两个 kernel，指令数完全一样，浮点运算量完全一样，唯一的区别只在数组下标：
            一个写 <code>a[i]</code>，另一个写 <code>a[i * 8]</code>。跑出来，前者比后者快接近 10 倍。如果你还停留在
            <ChapterLink n={3} />的「线程网格」视角，这根本说不通。同样多的线程，同样多的乘加，凭什么差出一个数量级？答案藏在
            线程算出下标之后、数据真正抵达寄存器之前的那段路上。这一章我们就盯着这段路看：全局内存如何按「事务」
            收费、warp 的访问模式怎样决定你浪费掉多少带宽、以及片上那块小而快的 shared memory 怎么把烂访问模式
            洗成好的。结论先放这里：<strong>大多数 kernel 的瓶颈不是计算，是访存</strong>。学会看访存，你就握住了
            GPU 性能优化的大半。
          </>,
        )}
      </p>

      <Section
        index={1}
        title={t('Global memory bills you by the transaction', '全局内存按「事务」收费')}
        lead={t(
          'DRAM has no retail counter: ask for one byte and it still ships you a 32-byte minimum.',
          '显存不做零售生意：你要 1 个字节，它也按 32 字节起步价给你搬。',
        )}
      >
        <p>
          {t(
            <>
              Global memory is the tens of gigabytes of HBM on the card. Its bandwidth looks terrifying, around
              1.9 TB/s on A100 and 3.35 TB/s on H100 (and the Blackwell generation pushes that further still), but
              the number comes with a condition: you have to access it the way it likes. The HBM interface's
              minimum transfer granularity is not one byte but a 32 B <strong>sector</strong>. GPU hardware places
              a coalescer in front of the memory subsystem. When a{' '}
              <Term t="warp">A scheduling unit of 32 threads that execute the same instruction in lockstep.</Term>{' '}
              issues a load/store, the hardware gathers the 32 addresses, works out which 32 B sectors they fall
              on, and <strong>issues one memory transaction per sector touched</strong>.
            </>,
            <>
              全局内存（global memory）就是显卡上那几十 GB 的 HBM。它的带宽看起来吓人，A100 约 1.9 TB/s，H100
              约 3.35 TB/s（Blackwell 一代还会更高），但这个数字有个前提：你得按它喜欢的方式访问。HBM 的接口一次传输的最小粒度不是
              1 字节，而是一个 32 B 的<strong>段（sector）</strong>。GPU 硬件在内存子系统前面放了一个合并器。当一个{' '}
              <Term t="warp">32 个线程组成的调度单位，同一拍里执行同一条指令。</Term> 执行一条 load/store
              指令时，硬件会把 32 个线程给出的地址收集起来，算一算它们一共落在哪几个 32 B 段上，然后
              <strong>每个被触及的段发起一次内存事务（memory transaction）</strong>。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              This is the most important billing rule in the chapter:{' '}
              <strong>the bandwidth you pay for = transactions x 32 B, not the bytes you actually wanted</strong>.
              The ratio of the two is bandwidth utilization:
            </>,
            <>
              这就是整章最重要的计费规则：<strong>你付出的带宽 = 事务数 × 32 B，而不是你真正想要的字节数</strong>。
              两者的比值就是带宽利用率：
            </>,
          )}
        </p>
        <MathTex block tex="\text{utilization} = \frac{\text{requested bytes}}{\text{transactions} \times 32\,\text{B}}" />
        <p>
          {t(
            <>
              Run the arithmetic. 32 threads each read a 4 B float, total demand 128 B. If thread{' '}
              <code>i</code> reads <code>a[i]</code>, the addresses are contiguous and the base is aligned, so
              those 128 B tile exactly across 4 adjacent 32 B sectors. The hardware issues 4 transactions, every
              byte fetched has a taker, and you're at 100% utilization. This ideal state, where a warp's accesses
              pack into as few sectors as possible, is called <strong>memory coalescing</strong>. It isn't a
              compiler optimization and needs no special instruction from you. The hardware decides it on the fly,
              purely from <em>the shape of those 32 addresses in a single cycle</em>.
            </>,
            <>
              来算几笔账。32 个线程每人读一个 4 B 的 float，总需求 128 B。如果线程 <code>i</code> 读{' '}
              <code>a[i]</code>，地址连续、首地址对齐，这 128 B 恰好铺满 4 个相邻的 32 B 段，硬件发 4 个事务，
              搬回来的每一个字节都有人要，利用率 100%。这种「warp 内的访问被打包进尽可能少的内存段」的理想状态，
              就叫<strong>合并访存（memory coalescing）</strong>。它不是编译器优化，也不需要你写任何特殊指令，
              纯粹由硬件根据<em>那一拍里 32 个地址的形状</em>即时决定。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Now change the index to <code>a[i * 2]</code> (stride = 2). The 32 threads spread across 256 B,
              touching 8 sectors, 8 transactions hauling back 256 B. You still only wanted 128 B, so utilization
              drops to 50%. Crank the stride further and at <code>a[i * 8]</code> each float owns its own 32 B
              sector: 32 transactions fetch 1024 B, only 128 B of which is useful. That's 12.5% utilization, rock
              bottom. Random access (gather) ends up similar, addresses scattering wherever they land, worst case
              again one transaction per thread. There's the entire secret behind the 10x gap from the opening.
              Kernel B drives a 1.9 TB/s bus at 12.5% efficiency, so its <em>effective</em> bandwidth is barely
              two hundred-some GB/s.
            </>,
            <>
              现在把下标改成 <code>a[i * 2]</code>（跨步 stride=2）。32 个线程铺开在 256 B 的范围上，触及 8 个段，
              8 个事务搬回 256 B，但你要的还是那 128 B，利用率掉到 50%。继续加大步长，到 <code>a[i * 8]</code>{' '}
              时每个 float 都独占一个 32 B 段，32 个事务搬回 1024 B，只有 128 B 有用，利用率 12.5%，触底。随机
              访问（gather）的下场类似，地址散到哪算哪，最坏同样是一人一个事务。这就是开头那个 10 倍差距的全部
              秘密：kernel B 用 12.5% 的效率使用一条 1.9 TB/s 的内存总线，<em>有效</em>带宽只剩两百多 GB/s。
            </>,
          )}
        </p>
        <CodeBlock
          code={t(HOOK_CODE_EN, HOOK_CODE_ZH)}
          lang="cuda"
          title={t('Same instruction count, a 10x gap', '同样的指令数，10 倍的差距')}
        />
        <p>
          {t(
            <>
              Two footnotes. Alignment is part of the bill too: a 128 B contiguous access that doesn't start on a
              32 B boundary straddles one extra sector, turning 4 transactions into 5. That's why the pointer from{' '}
              <code>cudaMalloc</code> is naturally 256 B aligned, and why your own pointer offsets deserve a second
              look. And the L1/L2 caches do exist (a 128 B cache line is made of 4 sectors), but for a big array
              streamed through once, the cache can't help. Streaming workloads are entirely transaction-count
              bound.
            </>,
            <>
              两个补充细节。一是对齐（alignment）也参与计费：128 B 的连续访问首地址若不在 32 B 边界上，
              会多骑跨一个段，4 个事务变 5 个，所以 <code>cudaMalloc</code> 返回的指针天然 256 B 对齐，自己做
              指针偏移时要留心。二是 L1/L2 缓存确实存在（cache line 128 B，由 4 个 sector 组成），但对一次性
              流过的大数组，缓存帮不上忙，流式负载的性能完全由事务数决定。
            </>,
          )}
        </p>
        <Callout type="insight" title={t('Bandwidth is billed per transaction', '带宽是按事务计费的')}>
          <p>
            {t(
              <>
                Picture DRAM as a wholesale-only warehouse. The minimum shipment is 32 B, and asking for 4 B still
                ships 32 B; the wasted part rides the same trucks and burns the same power. So "optimize memory
                access," translated into hardware terms, comes down to one sentence:{' '}
                <strong>make every byte you move actually get used</strong>. Keeping addresses contiguous within a
                warp is the simplest way to get there, and the one that matters most.
              </>,
              <>
                把显存想成一个只做批发的仓库。最小发货单位 32 B，你要 4 B 它也发 32 B，浪费的部分照样占运力、
                照样耗电。所以「优化访存」这四个字翻译成硬件语言只有一句话：
                <strong>让每一个被搬运的字节都被用上</strong>。warp 内地址连续，是达成这件事最简单、也最要紧的手段。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title={t('Lab: touch a transaction with your own hands', '实验：亲手摸一摸事务')}
        lead={t(
          "Rather than memorize the rules, drag a slider and watch where 32 lines land on the memory strip.",
          '与其背规则，不如拖一拖滑杆，看 32 条连线怎么砸在内存条带上。',
        )}
      >
        <p>
          {t(
            <>
              The lab below visualizes the previous section's billing rule. The top row of 32 dots is the warp's
              32 lanes, the bottom row is global memory partitioned into 32 B cells. Each cyan line is one lane's
              read request, and a touched sector lights up amber, so{' '}
              <strong>counting the lit cells gives you the transaction count</strong>. Switch to "stride" mode and
              drag the stride from 1 to 32 to watch transactions climb from 4 to 32 and utilization fall from 100%
              to 12.5%. Switch to "random" to see the misery of a gather (fixed seed, identical every time, so you
              have something to compare against).
            </>,
            <>
              下面的实验台把上一节的计费规则做成了可视化。上排 32 个圆点是 warp 的 32 个 lane，下排是按 32 B
              分格的全局内存条带。每条 cyan 连线是一个 lane 的读请求，被触及的段会亮起 amber，
              <strong>数一数亮了几格，就是几个事务</strong>。切到「跨步」模式把 stride 从 1 拖到 32，看事务数怎么从
              4 爬到 32、利用率怎么从 100% 跌到 12.5%；切到「随机」看 gather 的惨状（固定种子，每次都长一样，
              方便你对照）。
            </>,
          )}
        </p>
        <CoalescingLab />
        <p>
          {t(
            <>
              Two moments worth pausing on. Going from stride 1 to 2 instantly doubles the transaction count; the
              waste isn't gradual, it starts the instant you leave "contiguous." And past stride 8 the transaction
              count pins at 32 and won't budge, because each float already owns a sector and the penalty is capped
              (the cache layer keeps degrading, but transactions have bottomed out). Performance doesn't fall
              forever. It falls only to the floor of "one transaction per request."
            </>,
            <>
              有两个值得停下来体会的点。stride 从 1 变到 2，事务数立刻翻倍，浪费不是渐进发生的，从你离开
              「连续」的那一刻就开始了。而 stride 涨到 8 以后再涨，事务数钉死在 32 不动：每个 float 已经独占
              一个段，惩罚封顶（缓存层面还会继续恶化，但事务数见底了）。性能不会无限下坠，只会坠到「每个请求
              一个事务」的地板上。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            <>
              A warp's 32 threads read a float array at index <code>tid * 2</code> (stride = 2). How many 32 B
              transactions does this warp generate?
            </>,
            <>
              一个 warp 的 32 个线程读 float 数组，下标是 <code>tid * 2</code>（stride=2）。这个 warp
              产生几个 32 B 事务？
            </>,
          )}
          options={[
            {
              text: t('4 — the hardware can skip the bytes nobody wants', '4 个——硬件能跳过没人要的字节'),
              explain: t(
                "The hardware can't fetch with gaps: a 32 B sector is the minimum transfer granularity, so even one wanted byte drags the whole sector along.",
                '硬件不能跳着搬：32 B 段是最小传输粒度，段里哪怕只有一个字节被要也得整段搬。',
              ),
            },
            {
              text: t('8', '8 个'),
              correct: true,
              explain: t(
                '32 threads spread over 256 B (32 x 2 x 4 B), touching 8 consecutive 32 B sectors, so 8 transactions. Each sector has only half its bytes used — exactly 50% utilization.',
                '32 个线程铺在 256 B 上（32 × 2 × 4 B），触及 8 个连续的 32 B 段，所以 8 个事务。每段里只有一半字节有用，利用率恰好 50%。',
              ),
            },
            {
              text: t('16', '16 个'),
              explain: t(
                'Each 32 B sector holds 8 floats; at stride 2, 4 of them get read per sector, for a total of 256 B / 32 B = 8 sectors, not 16.',
                '每个 32 B 段装 8 个 float，stride=2 时每段里有 4 个被读到，共 256 B / 32 B = 8 段，不是 16。',
              ),
            },
            {
              text: t('32 — one transaction per thread', '32 个——每个线程一个事务'),
              explain: t(
                'That is the capped case at stride >= 8; at stride 2 several threads still share the same sector.',
                '那是 stride ≥ 8 时的封顶情况；stride=2 时多个线程还能共享同一个段。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title={t('AoS vs SoA: how a struct murders bandwidth', 'AoS vs SoA：struct 是怎么谋杀带宽的')}
        lead={t(
          "Nobody writes a[i*3] on purpose, but people who write p[i].x are everywhere — it's the same thing.",
          '没人会故意写 a[i*3]，但写 p[i].x 的人到处都是——这是同一件事。',
        )}
      >
        <p>
          {t(
            <>
              Strided access sounds like a contrived bad example, but it has an extremely common disguise:{' '}
              <strong>the array of structs</strong>. Imagine a particle simulation where each particle is{' '}
              <code>{'struct Particle { float x, y, z; }'}</code> and a million particles sit in an array. This
              "array of structures" layout is called <strong>AoS (Array of Structures)</strong>. Now write a
              kernel that touches only the x coordinate, so thread <code>i</code> reads <code>p[i].x</code>. Looks
              harmless. Work out the addresses, though, and neighboring threads land 12 B apart: a strided access
              with a stride of 3 floats, wearing a struct as a costume.
            </>,
            <>
              跨步访问听起来像故意写出来的反面教材，但它有一个极其常见的伪装：<strong>结构体数组</strong>。设想一个
              粒子模拟，每个粒子是 <code>{'struct Particle { float x, y, z; }'}</code>，一百万个粒子排成数组，这种
              「数组的结构体」布局叫 <strong>AoS（Array of Structures）</strong>。现在来一个只处理 x 坐标的
              kernel：线程 <code>i</code> 读 <code>p[i].x</code>。看起来人畜无害。可一算地址：相邻线程读的地址差
              12 B，这就是一个 stride 为 3 个 float 的跨步访问，只不过给套了件 struct 的外衣。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The 32 x-values a warp wants are spread over 384 B (32 x 12 B), touching 12 sectors: 12
              transactions, 33% utilization. The y and z hauled back have no takers, and you pay for that
              bandwidth anyway. The cure is to rotate the layout and store x, y, z each in its own independent
              array. This is <strong>SoA (Structure of Arrays)</strong>. The same kernel now reads{' '}
              <code>p.x[i]</code>, the addresses are contiguous again, 4 transactions, 100% utilization, two
              thirds of the bandwidth saved outright. The lab below lets you flip between the two layouts. Watch
              the memory strip rearrange and the 12 amber sectors collapse into 4:
            </>,
            <>
              一个 warp 要的 32 个 x 散布在 384 B（32 × 12 B）的范围上，触及 12 个 32 B 段，12 个事务，利用率
              33%。搬回来的 y 和 z 没人要，带宽却照付。解药是把布局转个方向：x、y、z 各自存成一条独立的数组，
              这叫 <strong>SoA（Structure of Arrays）</strong>。同一个 kernel 改读 <code>p.x[i]</code>，地址重新连续，
              4 个事务、100% 利用率，带宽立省三分之二。下面的实验台可以来回切换两种布局，注意内存条带怎么重新
              排列、12 个 amber 段怎么缩成 4 个：
            </>,
          )}
        </p>
        <AosSoaLab />
        <p>
          {t(
            "It's clearer side by side with the code. The kernel logic doesn't change a single word; only the data's seating chart in memory does:",
            '对照着代码看会更清楚：kernel 逻辑一个字没变，变的只是数据在内存里的「站位」：',
          )}
        </p>
        <CodeBlock
          code={t(AOS_CODE_EN, AOS_CODE_ZH)}
          lang="cuda"
          title={t('aos.cu — 12 transactions / warp', 'aos.cu —— 12 个事务 / warp')}
          highlight={[6]}
        />
        <CodeBlock
          code={t(SOA_CODE_EN, SOA_CODE_ZH)}
          lang="cuda"
          title={t('soa.cu — 4 transactions / warp', 'soa.cu —— 4 个事务 / warp')}
          highlight={[6]}
        />
        <p>
          {t(
            <>
              On the CPU this lesson is a nice-to-have, since the cache covers for you. On the GPU it's
              life-or-death. A CPU scanning AoS sequentially on one thread can still reuse the cache line, whereas
              on a GPU 32 threads each poke a different struct <em>in the same cycle</em> and the waste is
              multiplied by 32. The tensors in deep learning frameworks, one big contiguous array sliced by
              dimension, are SoA thinking taken all the way. And if the kernel really does need x/y/z together,
              there's a third option: a vector type like <code>float4</code> lets one thread move 16 B in one
              instruction, contiguous and instruction-thrifty at once.
            </>,
            <>
              这条经验在 CPU 世界是「锦上添花」，缓存会兜底；在 GPU 世界则是「生死攸关」。CPU 单线程顺序扫
              AoS 时缓存行还能复用，而 GPU 上 32 个线程<em>同一拍</em>各自戳一个 struct，浪费被乘以 32。深度学习
              框架里的 tensor，一整块连续的大数组、按维度切分，本质上就是把 SoA 思想贯彻到底的产物。
              如果 kernel 确实同时要 x/y/z，还有第三条路：用 <code>float4</code> 之类的向量类型让一个线程一条指令
              搬 16 B，既连续又省指令。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={4}
        title={t('Shared memory: the on-chip programmable cache', 'Shared memory：片上的可编程缓存')}
        lead={t(
          "Can't change the algorithm's access pattern? Then claim a plot on-chip and be your own cache manager.",
          '改不了算法的访问模式？那就在片上找块地，自己当缓存管理员。',
        )}
      >
        <p>
          {t(
            <>
              Some access patterns simply cannot be coalesced. The classic is matrix transpose: read by row,
              write by column, and one of the two is doomed to be strided. That's not bad code, it's the shape of
              the problem itself. This is where <strong>shared memory</strong> steps in. It's a slab of SRAM on
              each SM (up to 164 KB per SM on A100, 228 KB on H100), with ~20–30 cycle latency, more than an order
              of magnitude below global memory, and far higher aggregate bandwidth. The part that matters most is
              that it's <strong>under your programmatic control</strong>. A hardware cache can only guess what you
              want; shared memory does what you tell it.
            </>,
            <>
              有些访问模式天生没法合并。最经典的是矩阵转置：读按行、写按列，二者注定有一个是跨步的，这不是代码
              写得差，是问题本身的形状。这时就轮到 <strong>shared memory（共享内存）</strong>出场了。它是每个 SM
              片上的一块 SRAM（A100 上每 SM 最多 164 KB，H100 228 KB），延迟约 20–30 个周期，比全局内存低一个
              数量级以上，聚合带宽更是高出一大截。最要紧的一点是：它<strong>由你编程管理</strong>。硬件缓存只能猜你
              要什么，shared memory 听你指挥。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              There are only two new faces in the syntax. One is the <code>__shared__</code> qualifier, which
              declares an array shared by all threads in a block, living as long as the block does. The other is{' '}
              <code>__syncthreads()</code>, a block-level barrier that stalls at the call site until <em>every</em>{' '}
              thread in the block has arrived. There's exactly one rule, and it's the one that bites:{' '}
              <strong>all threads must reach the same barrier</strong>. Put it inside a branch only some threads
              take (say, inside <code>{'if (tid < 16)'}</code>) and they never all arrive. Deadlock at best,
              undefined behavior at worst.
            </>,
            <>
              用法上只有两个新面孔。一是 <code>__shared__</code> 修饰符，声明一块 block 内所有线程共享的数组，
              生命周期同 block。二是 <code>__syncthreads()</code>，block 级别的屏障（barrier），调用处等待 block
              内<em>所有</em>线程到齐才放行。规则只有一条，也正是最容易踩的一条：<strong>所有线程都必须到达同一个屏障</strong>。
              把它放进只有部分线程会走的分支里（比如 <code>{'if (tid < 16)'}</code> 内部），就永远到不齐，轻则死锁，
              重则未定义行为。
            </>,
          )}
        </p>
        <p>{t('Two canonical uses of shared memory:', 'shared memory 的两大典型用法：')}</p>
        <p>
          {t(
            <>
              <strong>(1) Data reuse within a block.</strong> When the same data gets read many times by many
              threads in the block (the matmul tile in the next chapter is the textbook case), pay once for a
              coalesced load to bring it on-chip, and every reuse afterward is billed at on-chip rates.
              <br />
              <strong>(2) A staging area to rewrite the access pattern.</strong> For problems like transpose where
              one of read/write must be strided, move the strided end into shared memory. Both ends of global
              memory stay on the coalesced path, and the messy scrambling work happens on-chip, where there's no
              32 B sector tax.
            </>,
            <>
              <strong>① block 内数据复用。</strong>同一块数据要被 block 里的很多线程读很多遍（下一章矩阵乘的
              tile 就是教科书案例），先花一次合并访存的钱把它搬进片上，之后的每次复用都按片上价格计费。
              <br />
              <strong>② 改写访问模式的中转站。</strong>转置这种「读写必有一头跨步」的问题，把跨步的那一头挪进
              shared memory 里完成，全局内存两头都走合并路线，乱序的脏活在片上做，片上不按 32 B 段计费。
            </>,
          )}
        </p>
        <CodeBlock
          code={t(TRANSPOSE_CODE_EN, TRANSPOSE_CODE_ZH)}
          lang="cuda"
          title={t('transpose.cu — shared memory as a staging area', 'transpose.cu —— shared memory 当中转站')}
          highlight={[2, 7, 12]}
        />
        <p>
          {t(
            <>
              Reading this, notice why the barrier on line 7 can't be dropped. The threads that <em>write</em>{' '}
              <code>tile</code> are not the same ones that read it: thread (tx, ty) writes{' '}
              <code>tile[ty][tx]</code>, but later reads <code>tile[tx][ty]</code>, a cell written by a different
              thread. There's no execution-order guarantee between warps, so reading without syncing may return
              uninitialized garbage. And when shared memory gets reused repeatedly inside a loop (like the next
              chapter's tiled matmul), you need a <strong>second barrier</strong> too:
            </>,
            <>
              读这段代码时注意第 7 行的屏障为什么不可省略：写 <code>tile</code> 的线程和读它的线程<em>不是同一批</em>。
              线程 (tx, ty) 写入 <code>tile[ty][tx]</code>，之后读的却是 <code>tile[tx][ty]</code>，那个格子是别的
              线程写的。warp 之间没有任何执行顺序保证，不等齐就读，读到的可能是没初始化的垃圾值。而当 shared
              memory 在循环里被反复复用时（比如下一章的 tiled matmul），还需要<strong>第二道屏障</strong>：
            </>,
          )}
        </p>
        <CodeBlock
          code={t(DOUBLE_SYNC_CODE_EN, DOUBLE_SYNC_CODE_ZH)}
          lang="cuda"
          title={t('Why a loop needs two syncs', '为什么循环里要 sync 两次')}
          highlight={[3, 6]}
        />
        <p>
          {t(
            <>
              Barrier (1) guards against a write-then-read race, where a thread reads before the write is done.
              Barrier (2) guards against a read-then-write race, where a fast thread enters the next iteration and
              overwrites data a slow thread hasn't finished reading. The bug from omitting the second sync is
              insidious. The result is usually correct by luck, then fails intermittently the moment you swap GPUs
              or change the block size.
            </>,
            <>
              屏障 ① 防「写后读」竞争：没写完就被读。屏障 ② 防「读后写」竞争：跑得快的线程进入下一轮迭代，把慢
              线程还没读完的数据覆盖掉。少写第二道 sync 的 bug 极其阴险，多数时候结果碰巧正确，换张卡、换个
              block 大小就偶发出错。
            </>,
          )}
        </p>
        <Callout type="warn" title={t('__syncthreads() only covers within a block', '__syncthreads() 只管 block 内')}>
          <p>
            {t(
              <>
                It can't synchronize anything between blocks. Cross-block dependencies must either be split into
                two kernels (a kernel boundary is a natural global sync point) or use grid-level synchronization
                from cooperative groups, a story for a later chapter.
              </>,
              <>
                它同步不了 block 之间的任何东西。跨 block 的依赖要么拆成两个 kernel（kernel 边界天然是全局同步
                点），要么用 cooperative groups 的 grid 级同步，那是后面章节的故事。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={5}
        title={t("Bank conflicts: shared memory's own pitfall", 'Bank conflict：shared memory 自己的坑')}
        lead={t(
          "On-chip charges no 32 B sector tax, but it levies a tax of its own.",
          '片上不收 32B 段的税，但它有自己的税种。',
        )}
      >
        <p>
          {t(
            <>
              Shared memory has no coalescing problem, but it has its own parallel structure: the whole SRAM is
              sliced into <strong>32 banks</strong>, addressed round-robin by 4 B words. Word 0 sits in bank 0,
              word 1 in bank 1, and word 32 wraps back to bank 0. Each bank can serve <em>one address</em> per
              cycle. So a warp's 32 accesses complete in a single cycle if they fall evenly across 32 distinct
              banks (or if several threads read the <em>same</em> address, which the hardware just broadcasts). If
              N threads instead land on <strong>different addresses in the same bank</strong>, that's an{' '}
              <strong>N-way bank conflict</strong>, and the hardware can only serialize those N accesses into N
              cycles. Shared memory instantly runs N times slower.
            </>,
            <>
              shared memory 没有合并访存的问题，却有自己的并行结构：整块 SRAM 被切成 <strong>32 个 bank</strong>，
              按 4 B 的 word 轮流编址。word 0 在 bank 0，word 1 在 bank 1，word 32 又绕回 bank 0。每个 bank
              一个周期只能伺候<em>一个地址</em>。于是一个 warp 的 32 个访问，如果均匀落在 32 个不同 bank 上
              （或者多个线程读<em>同一个</em>地址，硬件直接广播），一拍完成。但如果 N 个线程落在<strong>同一个
              bank 的不同地址</strong>上，这就是 <strong>N 路 bank conflict（bank 冲突）</strong>，硬件只能把这
              N 个访问串行成 N 拍，shared memory 瞬间慢 N 倍。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Go back to the transpose kernel and the trap becomes clear. If the tile is declared{' '}
              <code>[32][32]</code>, reading by column (<code>tile[threadIdx.x][threadIdx.y]</code>, with tx
              running 0 to 31 across the warp) puts neighboring threads 32 words apart, and 32 mod 32 is 0, so{' '}
              <strong>all 32 accesses in the warp slam into the same bank</strong>: a 32-way conflict, 32x
              serialization. Use the inspector below to sweep stride from 1 to 33, watching four ticks in
              particular: 1 (perfect), 2 (two-way), 32 (everyone collides), 33 (perfect again):
            </>,
            <>
              回头看转置 kernel 就明白坑在哪了。如果 tile 声明成 <code>[32][32]</code>，按列读（
              <code>tile[threadIdx.x][threadIdx.y]</code>，warp 内 tx 从 0 跑到 31）时相邻线程的地址差 32 个
              word，32 模 32 等于 0，<strong>整个 warp 的 32 个访问全砸在同一个 bank 上</strong>，32 路冲突，
              32 倍串行。用下面的检查器亲手扫一遍 stride 从 1 到 33，重点观察四个刻度：1（完美）、2（两路）、
              32（全员撞车）、33（又完美）：
            </>,
          )}
        </p>
        <BankConflictLab />
        <p>
          {t(
            <>
              Why is stride 33 magically conflict-free? Because 33 and 32 are coprime: lane <code>i</code> accesses
              word <code>33i</code>, landing in bank <code>33i mod 32 = i</code>, so the 32 lanes spread evenly
              across all 32 banks. That's the whole principle behind the <strong>padding trick</strong>. Declare
              the tile <code>[32][33]</code>, leaving one never-used word at the end of each row, and consecutive
              rows are offset by one in bank-space, so the column-access stride goes from 32 to 33. The conflict
              vanishes. The cost is 128 B of on-chip space for up to a 32x speedup, about the best return on a
              single line you'll find anywhere in GPU programming.
            </>,
            <>
              stride=33 为什么神奇地无冲突？因为 33 和 32 互质：lane <code>i</code> 访问 word <code>33i</code>，
              落在 bank <code>33i mod 32 = i</code>，32 个 lane 恰好均匀铺满 32 个 bank。这就是{' '}
              <strong>padding（填充）技巧</strong>的全部原理。把 tile 声明成 <code>[32][33]</code>，每行末尾留一个
              永远不用的 word，行与行之间在 bank 视角下整体错开一格，「按列访问」的 stride 就从 32 变成 33，
              冲突消失。代价不过 128 B 的片上空间，换回最多 32 倍的速度，大概是 GPU 编程里一行代码能买到的最高回报。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            <>
              In the transpose kernel, declaring the tile <code>[32][33]</code> instead of <code>[32][32]</code>{' '}
              eliminates the column-access bank conflict. Why?
            </>,
            <>
              转置 kernel 里把 tile 声明成 <code>[32][33]</code> 而不是 <code>[32][32]</code>，为什么能消除
              按列访问的 bank conflict？
            </>,
          )}
          options={[
            {
              text: t(
                'One extra word per row offsets the same column of consecutive rows by one in bank-space; the column-access stride becomes 33, which is coprime with 32',
                '每行多 1 个 word，相邻行的同一列在 bank 视角下错开一格；列访问的 stride 变成与 32 互质的 33',
              ),
              correct: true,
              explain: t(
                'Correct. On column access the address stride per lane equals the row width (33 words), so bank = 33i mod 32 = i, and the 32 accesses land evenly on 32 distinct banks — one cycle.',
                '正解。按列访问时 lane i 的地址 stride 等于行宽（33 个 word），bank = 33i mod 32 = i，32 个访问均匀落到 32 个不同 bank，一拍完成。',
              ),
            },
            {
              text: t('The array is bigger, so the L1 cache hit rate improves', '数组更大，L1 缓存命中率更高'),
              explain: t(
                'Shared memory is on-chip SRAM and never goes through L1; the extra column has nothing to do with caching.',
                'shared memory 本身就是片上 SRAM，不经过 L1；多出的一列与缓存无关。',
              ),
            },
            {
              text: t(
                'The compiler sees a non-power-of-two and automatically rearranges the data layout',
                '编译器看到非 2 的幂会自动重排数据布局',
              ),
              explain: t(
                "The compiler doesn't quietly change the layout you declared; padding is a manual trick, completely transparent to the compiler.",
                '编译器不会偷偷改你声明的布局；padding 是程序员手动完成的、对编译器完全透明的技巧。',
              ),
            },
            {
              text: t(
                "The extra column stores __syncthreads()'s flag bits",
                '多出的一列用来存 __syncthreads() 的标志位',
              ),
              explain: t(
                "The barrier is implemented by a hardware instruction and uses no shared memory; that column is purely deliberately wasted space.",
                '屏障由硬件指令实现，不占 shared memory；那一列纯粹是被故意浪费掉的空间。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={6}
        title={t('The rest of the memory hierarchy', '存储层级的其他角色')}
        lead={t(
          "The leads are done; the supporting cast each get one line — but any of them can kill you at the wrong moment.",
          '主角讲完了，配角们各有一句话的戏份——但关键时刻都能要你的命。',
        )}
      >
        <p>
          {t(
            <>
              <strong>Registers</strong> are the fastest storage, bar none: on-chip, single-cycle, thread-private.
              But there are at most 255 per thread, and the whole SM's register file (256 KB per SM on A100) is
              shared among all resident threads. The more one thread uses, the fewer threads the SM can keep alive
              at once (foreshadowing the occupancy chapter). The subtler trap is the{' '}
              <strong>register spill</strong>. When there are too many variables to hold, the compiler stuffs the
              overflow into local memory, a deeply deceptive name, since it physically lives in DRAM with
              global-memory-class latency. A loop variable you assumed was spinning in a register costs hundreds
              of cycles per access once it spills, and that's exactly how a performance cliff appears (compile
              with <code>-Xptxas -v</code> to see the spill byte count).
            </>,
            <>
              <strong>寄存器（register）</strong>是最快的存储，没有之一：片上、单周期、每线程私有。但每个线程
              最多 255 个，并且整个 SM 的寄存器堆（A100 每 SM 256 KB）由所有常驻线程瓜分，单线程用得越多，
              SM 上能同时养活的线程越少（这是后面占用率一章的伏笔）。更隐蔽的坑是<strong>寄存器溢出（register
              spill）</strong>：变量太多放不下时，编译器把多出来的塞进 local memory，这名字起得极具欺骗性，它
              物理上就在显存里，延迟和 global memory 一个档次。一个你以为在寄存器里打转的循环变量，溢出后每次
              读写都是几百周期，性能悬崖就是这么来的（编译时加 <code>-Xptxas -v</code> 能看到 spill 字节数）。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>Constant memory</strong> is a 64 KB read-only region with a dedicated on-chip cache. Its
              trump card is the <strong>broadcast</strong>: when a whole warp reads the same address, one hit
              feeds all 32 threads at near-register speed. If the addresses within the warp all differ, though,
              the access serializes and it's actually slower. Good for small "everyone reads the same copy" data
              like convolution kernel coefficients or hyperparameters. <strong>L2 cache</strong> is the
              device-wide transparent cache (40 MB on A100, 50 MB on H100). Every global access passes through it,
              beyond your control either way, but it explains many "performance is mysteriously good when the data
              is small" phenomena.
            </>,
            <>
              <strong>Constant memory</strong> 是 64 KB 的只读区域，配专用片上缓存。它的杀手锏是<strong>广播</strong>：
              整个 warp 读同一个地址时，一次命中喂饱 32 个线程，近似寄存器速度。但若 warp 内地址各不相同，访问
              会被串行化，反而更慢。适合放卷积核系数、超参数这类「全员读同一份」的小数据。<strong>L2 cache</strong>{' '}
              则是全卡共享的透明缓存（A100 40 MB、H100 50 MB），所有全局访问都路过它，不用管也管不着，但它
              解释了很多「数据小的时候性能莫名其妙地好」的现象。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The last role is over on the host side: <strong>pinned host memory</strong> (page-locked). Memory
              from a plain <code>malloc</code> can be paged out by the OS, so the GPU's DMA engine can't move it
              directly and has to go through an extra staging copy first. Page-locked memory from{' '}
              <code>cudaMallocHost</code> can be DMA'd directly, which isn't just faster to copy but the
              prerequisite for asynchronous <code>cudaMemcpyAsync</code>. Only with it can a PCIe transfer overlap
              with kernel compute on the timeline, the basic skill for hiding data-movement latency in the later
              inference-serving chapter.
            </>,
            <>
              最后一个角色在主机那头：<strong>pinned host memory（页锁定内存）</strong>。普通 <code>malloc</code>{' '}
              出来的内存可能被操作系统换页，GPU 的 DMA 引擎不能直接搬，得先经过一次额外的中转拷贝；用{' '}
              <code>cudaMallocHost</code> 申请的页锁定内存则可以直接 DMA，不仅拷贝更快，更是{' '}
              <code>cudaMemcpyAsync</code> 异步拷贝的前提。有了它，PCIe 传输才能和 kernel 计算在时间轴上重叠，
              这是后面推理服务章节里隐藏数据搬运延迟的基本功。
            </>,
          )}
        </p>
        <Figure
          caption={t(
            'GPU memory-hierarchy cheat sheet: latencies are order-of-magnitude estimates that vary by architecture; capacity and bandwidth reference A100 / H100.',
            'GPU 存储层级速查表：延迟为数量级估计，随架构浮动；容量与带宽以 A100 / H100 为参考',
          )}
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse font-mono text-[12.5px]">
              <thead>
                <tr className="border-b border-line2 text-left">
                  <th className="microlabel py-2 pr-3 font-normal">{t('Storage', '存储')}</th>
                  <th className="microlabel py-2 pr-3 font-normal">{t('Location', '位置')}</th>
                  <th className="microlabel py-2 pr-3 font-normal">{t('Latency (approx)', '延迟（约）')}</th>
                  <th className="microlabel py-2 pr-3 font-normal">{t('Scope', '作用域')}</th>
                  <th className="microlabel py-2 font-normal">{t('In one line', '一句话定位')}</th>
                </tr>
              </thead>
              <tbody className="text-text">
                {HIER_ROWS.map((r, i) => (
                  <tr key={i} className="border-b border-line last:border-0">
                    <td className="py-2 pr-3 whitespace-nowrap text-ink">{pick(r[0], lang)}</td>
                    <td className="py-2 pr-3 text-ink2">{pick(r[1], lang)}</td>
                    <td className="py-2 pr-3 tabular-nums text-amber">{pick(r[2], lang)}</td>
                    <td className="py-2 pr-3 text-ink2">{pick(r[3], lang)}</td>
                    <td className="py-2 text-ink2">{pick(r[4], lang)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Figure>
      </Section>

      <Section
        index={7}
        title={t('Summary and further reading', '总结与延伸阅读')}
        lead={t(
          'This chapter has only one central idea: make every byte you fetch get used.',
          '这一章只有一个中心思想：让搬来的每个字节都被用上。',
        )}
      >
        <p>
          {t(
            'Compressed into five points, the intuition here is enough to diagnose most memory-access problems:',
            '把本章的直觉压缩成五条，够你诊断大多数访存问题：',
          )}
        </p>
        <p>
          {t(
            <>
              <strong>(1)</strong> Global memory is billed in 32 B sector transactions: however many sectors a
              warp's 32 addresses land on, that's how many transactions you pay for. A contiguous float access is
              4 fully-useful transactions, while strided and random access bloat up to 32.
              <br />
              <strong>(2)</strong> Coalescing is decided on the fly by hardware from each cycle's address shape;
              your only job is to keep addresses contiguous within a warp.
              <br />
              <strong>(3)</strong> <code>p[i].x</code> is a strided access in disguise; use SoA layout on hot
              paths that read only some fields.
              <br />
              <strong>(4)</strong> Shared memory is an on-chip programmable cache, the cure for data reuse and
              "inherently un-coalescable" access patterns; <code>__syncthreads()</code> must be reached by all
              threads, and loop reuse needs one barrier each for write-then-read and read-then-write.
              <br />
              <strong>(5)</strong> Shared memory parallelizes across 32 banks; N threads hitting different
              addresses in the same bank serialize N-fold, and one padding column (<code>[32][33]</code>) fixes
              it.
            </>,
            <>
              <strong>①</strong> 全局内存按 32 B 段事务计费：warp 内 32 个地址落在几个段上，就付几个事务的钱。
              连续 float 访问 4 个事务全有用，跨步和随机最多膨胀到 32 个。
              <br />
              <strong>②</strong> 合并访存（coalescing）由硬件根据每一拍的地址形状即时决定，你的任务只是让 warp
              内地址连续。
              <br />
              <strong>③</strong> <code>p[i].x</code> 就是穿了马甲的跨步访问；只读部分字段的热路径请用 SoA 布局。
              <br />
              <strong>④</strong> shared memory 是片上的可编程缓存，专治数据复用和「天生没法合并」的访问模式；
              <code>__syncthreads()</code> 必须全员到达，循环复用时写后读、读后写各要一道屏障。
              <br />
              <strong>⑤</strong> shared memory 按 32 个 bank 并行，N 个线程撞同一 bank 的不同地址就串行 N 倍，
              一列 padding（<code>[32][33]</code>）就能解。
            </>,
          )}
        </p>
        <p>{t('To dig deeper, here are the primary sources:', '想继续深挖，下面这些是一手资料：')}</p>
        <ul>
          <li>
            <a
              className="text-cyan hover:underline"
              href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses"
              target="_blank"
              rel="noreferrer"
            >
              CUDA C++ Programming Guide — Device Memory Accesses
            </a>
            {t(
              ' — the official definition of the coalescing rules and bank structure, the source for every secondhand account.',
              '：合并规则与 bank 结构的官方定义，所有二手资料的源头。',
            )}
          </li>
          <li>
            <a
              className="text-cyan hover:underline"
              href="https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA Developer Blog — How to Access Global Memory Efficiently
            </a>
            {t(
              ' — an intro to coalescing with measured bandwidth curves, the paper version of this chapter\'s LAB 01.',
              '：带实测带宽曲线的合并访存入门，本章 LAB 01 的纸面版。',
            )}
          </li>
          <li>
            <a
              className="text-cyan hover:underline"
              href="https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA Developer Blog — Using Shared Memory in CUDA C/C++
            </a>
            {t(
              ' — the official tutorial on shared memory, synchronization, and bank conflicts.',
              '：shared memory、同步与 bank conflict 的官方教程。',
            )}
          </li>
          <li>
            <a
              className="text-cyan hover:underline"
              href="https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA Developer Blog — An Efficient Matrix Transpose in CUDA C/C++
            </a>
            {t(
              " — the full version of this chapter's transpose kernel, measured step by step from naive to padding.",
              '：本章转置 kernel 的完整版，从 naive 到 padding 的逐步实测。',
            )}
          </li>
          <li>
            <a
              className="text-cyan hover:underline"
              href="https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors"
              target="_blank"
              rel="noreferrer"
            >
              {t(
                'PMPP (Programming Massively Parallel Processors, 4th ed.), chapters 5–6',
                'PMPP（Programming Massively Parallel Processors, 4th ed.）第 5–6 章',
              )}
            </a>
            {t(
              ' — the systematic textbook treatment of memory architecture and performance optimization.',
              '：内存架构与性能优化的系统化教材讲法。',
            )}
          </li>
        </ul>
        <p>
          {t(
            <>
              Next chapter we aim this chapter's entire arsenal at a single problem: matrix multiplication.
              Starting from a naive implementation, we close in on cuBLAS step by step through tiling, shared
              memory, and register reuse, and you'll watch "memory is king" prove itself over and over on a real
              kernel.
            </>,
            <>
              下一章我们把本章的全部武器对准同一个问题：矩阵乘法。从 naive 实现出发，靠 tiling、shared memory
              和寄存器复用一步步逼近 cuBLAS，你会看到「访存为王」这四个字在一个真实 kernel 上反复应验。
            </>,
          )}
        </p>
      </Section>
    </>
  )
}
