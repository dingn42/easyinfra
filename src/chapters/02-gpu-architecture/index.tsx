import { Callout, ChapterLink, CodeBlock, Figure, HardwareBaseline, Quiz, Section, Term } from '@/components/ui'
import { useT, useLocale, pick, type Loc } from '@/lib/i18n'
import { AnatomyLab } from './AnatomyLab'
import { WarpLatencyLab } from './WarpLatencyLab'
import { MemoryLatencyLab } from './MemoryLatencyLab'

/* ── SEC06 配图：CUDA core vs Tensor Core 吞吐对比 ── */
const FLOPS_BARS: { label: Loc; v: number; cls: string; txt: string }[] = [
  { label: { en: 'FP64 CUDA core', zh: 'FP64 CUDA 核' }, v: 34, cls: 'fill-ink3', txt: 'fill-ink2' },
  { label: { en: 'FP32 CUDA core', zh: 'FP32 CUDA 核' }, v: 67, cls: 'fill-cyan/70', txt: 'fill-cyan' },
  { label: { en: 'TF32 Tensor Core', zh: 'TF32 Tensor Core' }, v: 495, cls: 'fill-violet/70', txt: 'fill-violet' },
  { label: { en: 'BF16 Tensor Core', zh: 'BF16 Tensor Core' }, v: 989, cls: 'fill-volt', txt: 'fill-volt' },
  { label: { en: 'FP8 Tensor Core', zh: 'FP8 Tensor Core' }, v: 1979, cls: 'fill-volt/60', txt: 'fill-volt' },
]

function FlopsChart() {
  const t = useT()
  const { lang } = useLocale()
  const max = 1979
  return (
    <svg viewBox="0 0 720 196" className="w-full select-none" role="img" aria-label={t('H100 peak throughput by precision', 'H100 各精度峰值吞吐对比')}>
      {FLOPS_BARS.map((b, i) => {
        const y = 10 + i * 36
        const w = Math.max(3, (b.v / max) * 520)
        return (
          <g key={b.label.en}>
            <text x={148} y={y + 15} fontSize={11} textAnchor="end" className="fill-ink2 font-mono">
              {pick(b.label, lang)}
            </text>
            <rect x={156} y={y} width={520} height={22} rx={3} className="fill-bg2" />
            <rect x={156} y={y} width={w} height={22} rx={3} className={b.cls} />
            <text x={162 + w} y={y + 15} fontSize={11} className={`${b.txt} font-mono`}>
              {b.v.toLocaleString('en-US')} TFLOPS
            </text>
          </g>
        )
      })}
      <text x={156} y={192} fontSize={9.5} className="fill-ink3 font-mono">
        {t(
          'H100 SXM peak (dense, excludes 2:4 sparsity) · BF16 Tensor ≈ 15× FP32 CUDA core',
          'H100 SXM 峰值（dense，不含 2:4 稀疏加成）· BF16 Tensor ≈ 15× FP32 CUDA 核',
        )}
      </text>
    </svg>
  )
}

/* ── SEC05 配表：存储层级 ── */
interface MemRow {
  level: Loc
  cap: Loc
  bw: Loc
  lat: Loc
  cls: string
}
const MEM_TABLE: MemRow[] = [
  {
    level: { en: 'Registers', zh: '寄存器（Registers）' },
    cap: { en: '256 KB / SM (≈33 MB whole GPU)', zh: '256 KB / SM（全卡 ≈ 33 MB）' },
    bw: { en: 'all partitions, every cycle', zh: '每周期全分区并发' },
    lat: { en: '≈1 cycle', zh: '≈1 周期' },
    cls: 'text-volt',
  },
  {
    level: { en: 'Shared memory / L1', zh: '共享内存 / L1' },
    cap: { en: '256 KB / SM (up to 228 KB shared)', zh: '256 KB / SM（Shared 可配 228 KB）' },
    bw: { en: '~30 TB/s aggregate', zh: '聚合约 30 TB/s' },
    lat: { en: '≈30 cycles', zh: '≈30 周期' },
    cls: 'text-cyan',
  },
  {
    level: { en: 'L2 cache', zh: 'L2 缓存' },
    cap: { en: '50 MB (chip-wide)', zh: '50 MB（全片共享）' },
    bw: { en: '~5–6 TB/s', zh: '约 5~6 TB/s' },
    lat: { en: '≈200–300 cycles', zh: '≈200~300 周期' },
    cls: 'text-violet',
  },
  {
    level: { en: 'HBM3 memory', zh: 'HBM3 显存' },
    cap: { en: '80 GB', zh: '80 GB' },
    bw: { en: '3.35 TB/s', zh: '3.35 TB/s' },
    lat: { en: '≈500+ cycles', zh: '≈500 周期以上' },
    cls: 'text-amber',
  },
]

function MemTable() {
  const { lang } = useLocale()
  const headers: Loc[] = [
    { en: 'Level', zh: '层级' },
    { en: 'Capacity', zh: '容量' },
    { en: 'Bandwidth', zh: '带宽' },
    { en: 'Latency', zh: '延迟' },
  ]
  return (
    <div className="my-6 overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-line2 text-left">
            {headers.map((h) => (
              <th key={h.en} className="microlabel py-2 pr-4 font-normal">
                {pick(h, lang)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MEM_TABLE.map((row) => (
            <tr key={row.level.en} className="border-b border-line">
              <td className={`py-2.5 pr-4 font-medium ${row.cls}`}>{pick(row.level, lang)}</td>
              <td className="py-2.5 pr-4 font-mono text-[12px] tabular-nums text-text">{pick(row.cap, lang)}</td>
              <td className="py-2.5 pr-4 font-mono text-[12px] tabular-nums text-text">{pick(row.bw, lang)}</td>
              <td className="py-2.5 font-mono text-[12px] tabular-nums text-ink">{pick(row.lat, lang)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const DIVERGE_CODE_EN = `// The 32 threads in a warp run in lockstep. What happens at a branch?
if (threadIdx.x % 2 == 0) {
    out[i] = expensive_a(x);   // even threads want this path
} else {
    out[i] = expensive_b(x);   // odd threads want this path
}
// The hardware's answer: walk both paths.
// Run branch a first (odd threads masked off), then branch b (even threads masked off).
// Total time ≈ a + b — half the execution slots burned for nothing.`

const DIVERGE_CODE_ZH = `// warp 内的 32 个线程锁步执行。遇到分支怎么办？
if (threadIdx.x % 2 == 0) {
    out[i] = expensive_a(x);   // 偶数线程想走这里
} else {
    out[i] = expensive_b(x);   // 奇数线程想走这里
}
// 硬件的答案：两条路都走一遍。
// 先执行 a 分支（奇数线程被掩码挂起），再执行 b 分支（偶数线程挂起）。
// 总耗时 ≈ a + b，一半的执行槽位被白白烧掉。`

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            The H100 spec sheet lists 16,896 FP32 cores, well over a hundred times the core count of a
            top-end server CPU. Picture that as 16,896 little CPUs each doing its own thing and you will be
            lost from your very first line of CUDA. Why can a single <code>if</code> statement halve your
            performance? Why do you have to launch tens of times more threads than there are cores just to
            keep the chip busy? It all comes down to how the hardware is wired up. These cores move in groups
            of 32, <strong>in lockstep</strong>, sharing one fetch and one scheduler. This chapter takes an
            H100 apart, from the silicon down to the execution units, and works out the rules this machine
            actually runs by. The performance intuition behind every later chapter starts here.
          </>,
          <>
            H100 的规格表上写着 16896 个 FP32 核心，是顶级服务器 CPU 核数的一百多倍。可要是你把它想成 16896
            个能各干各事的小 CPU，那从第一行 CUDA 代码开始就要犯迷糊：为什么一个 <code>if</code>{' '}
            语句能让性能掉一半？为什么线程数要开到核心数的几十倍才能把芯片喂饱？这一切都取决于硬件是怎么接线的。
            这些核心以 32 个为一组<strong>锁步（lockstep）行动</strong>，共享取指和调度。这一章我们把一块 H100
            拆开，从硅片一路看到执行单元，弄清这台机器究竟按什么规则运转。后面每一章的性能直觉，根子都在这里。
          </>,
        )}
      </p>

      <HardwareBaseline ids={['h100']} />

      <Section
        index={1}
        title={t('Top-down: from die to SM', '自顶向下：从 die 到 SM')}
        lead={t(
          'A GPU is not one processor. It is a hundred-plus copies of the same processor, plus a pipeline to feed them data.',
          'GPU 不是一个处理器，而是同一个处理器的一百多份拷贝，外加一套喂数据的管线。',
        )}
      >
        <p>
          {t(
            <>
              Start with a sense of scale. An H100 die (codename GH100) is 814 mm² and packs roughly 80 billion
              transistors on TSMC's 4N process, close to the physical limit of a single lithography exposure.
              Zoom in and the organization is a strict set of nested dolls: the die splits into 8{' '}
              <strong>GPCs</strong> (Graphics Processing Clusters), each GPC holds 9 <strong>TPCs</strong>{' '}
              (Texture Processing Clusters), and each TPC carries 2{' '}
              <Term t="SM (Streaming Multiprocessor)">
                The GPU's fundamental compute unit. A thread block resides entirely on one SM, which supplies its
                registers, shared memory, and execution units. Remember it; every later chapter comes back to it.
              </Term>
              . That multiplies out to 144 SMs; for yield, the H100 SXM fuses off 12, leaving{' '}
              <strong>132 SMs</strong> enabled (the previous-gen A100 had 108). GPCs and TPCs are nearly invisible to
              the programmer. The only two levels you truly need to hold in your head are <strong>die and SM</strong>.
            </>,
            <>
              先建立层级感。一块 H100 的硅片（die，代号 GH100）面积 814 mm²，用台积电 4N 工艺塞进了约
              800 亿个晶体管，已经逼近光刻机单次曝光的物理上限。往里看，它的组织是严格的套娃结构：整片 die
              划分为 8 个 <strong>GPC</strong>（Graphics Processing Cluster，图形处理簇），每个 GPC 里有 9 个{' '}
              <strong>TPC</strong>（Texture Processing Cluster，纹理处理簇），每个 TPC 装着 2 个{' '}
              <Term t="SM（Streaming Multiprocessor，流式多处理器）">
                GPU 的基本计算单元。线程块整块驻留在 SM 上，由它提供寄存器、共享内存和执行单元。记住它，
                后面每一章都绕不开。
              </Term>
              。乘起来是 144 个 SM；考虑良率，H100 SXM 版屏蔽掉 12 个，实际启用 <strong>132 个 SM</strong>
              （上一代 A100 是 108 个）。GPC 和 TPC 对程序员几乎透明，真正需要记住的层级只有两个：
              <strong>die 和 SM</strong>。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The SM is what really deserves the name "core" on a GPU. Inside each H100 SM are four structurally
              identical processing partitions; each partition has 1 warp scheduler and 32 FP32 cores, for a total of{' '}
              <strong>128 FP32 cores, 4 warp schedulers, and 4 Tensor Cores</strong>. The four partitions each carry a
              64 KB register file (<strong>256 KB</strong> in total) and share one 256 KB block of L1 data
              cache / shared memory, of which up to 228 KB can be configured as shared memory. Hold an SM's area
              budget against a CPU core and two opposite design philosophies jump out. A CPU core spends most of its
              transistors on out-of-order execution, branch prediction, and deep caches, all to make a{' '}
              <strong>single thread</strong> run as fast as possible. An SM throws all of that out and pours the
              reclaimed area into execution units and registers. It never intended to make any one thread fast; what
              it wants is <strong>enough threads in flight at once</strong>.
            </>,
            <>
              SM 才是 GPU 真正意义上的「核」。每个 H100 的 SM 内部是四个结构完全相同的处理分区，每个分区有
              1 个 warp 调度器和 32 个 FP32 核心，合计 <strong>128 个 FP32 核心、4 个 warp 调度器、4 个
              Tensor Core</strong>；四个分区各自挂着 64 KB 寄存器堆（合计 <strong>256 KB</strong>），共用一块
              256 KB 的 L1 数据缓存/共享内存，其中最多 228 KB 可配置为共享内存。把 SM 的面积分配和 CPU
              核摆在一起看，两种截然相反的设计哲学就浮出来了。CPU 核把大部分晶体管花在乱序执行、分支预测和多级大缓存上，
              图的是让<strong>单个线程</strong>跑得尽量快。SM 把这些全砍掉，省下的面积全部还给执行单元和寄存器。
              它从一开始就没想让任何一个线程跑得快，它要的是<strong>同时在飞的线程足够多</strong>。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Note one counterintuitive number: an SM's register file (256 KB) is larger than its L1 cache, whereas a
              single CPU core's registers add up to just a few KB. This "inversion" is no design mistake. It is the
              physical foundation of the GPU's entire latency-hiding scheme, a thread we pick back up in section 3. The
              rest of the chip is the logistics that serve those 132 SMs: a 50 MB L2 cache sits chip-wide in the
              center of the die, 5 HBM3 stacks line its sides to provide 80 GB of capacity and 3.35 TB/s of bandwidth,
              the NVLink interface up top handles multi-GPU interconnect, and the GigaThread engine at the bottom hands
              thread blocks out to the SMs. (The newer Blackwell generation keeps this same skeleton and pushes the
              bandwidth and low-precision throughput further still.) The dissection bench below lets you take one apart
              by hand.
            </>,
            <>
              注意一个反直觉的数字：SM 的寄存器堆（256 KB）比它的 L1 缓存还要大，而 CPU 单核的寄存器加起来
              不过几 KB。这个「倒挂」不是设计失误，而是整个 GPU 延迟隐藏机制的物质基础，第 3 节会回收这个伏笔。
              芯片的其余部分都在为这 132 个 SM 跑后勤：die 中央一条 50 MB 的 L2 缓存为全片共享，
              die 两侧贴着 5 颗 HBM3 显存堆叠提供 80 GB 容量和 3.35 TB/s 带宽，顶部的 NVLink 接口负责多卡互联，
              底部的 GigaThread 引擎负责把线程块分发到各个 SM。（更新一代的 Blackwell 沿用了这套骨架，
              把带宽和低精度吞吐又往上推了一截。）下面这台解剖台可以亲手拆一遍。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={2}
        title={t('LAB · GPU dissection bench', 'LAB · GPU 解剖台')}
        lead={t(
          'Layer 1 is the whole die; click any lit-up SM to zoom into layer 2; click any component for its file on the right.',
          '层 1 是整片 die，点击任意发亮的 SM 可以放大到层 2；点击任何部件，右侧给出它的档案。',
        )}
      >
        <AnatomyLab />
        <p>
          {t(
            <>
              A tour leaves you with three things to take away. The first: <strong>an SM is copy-pasted</strong>. NVIDIA designs
              one SM and stamps out 144 of them to tile the die, which is why a GPU's generational upgrade is largely
              "stronger SMs plus more SMs," and why your program, to saturate the hardware, has to split into enough
              independent chunks to feed every SM. The second: <strong>the gray cells are real</strong>. Yield on a chip
              this large is limited, fusing off defective SMs at the factory is industry standard, and the same GH100 die
              is binned and sold by how many are disabled. The third: look inside an SM and notice that each partition has
              exactly 32 FP32 cores. That number is about to become the protagonist.
            </>,
            <>
              逛完一圈，有三件事值得留意。其一，<strong>SM 是「复制粘贴」出来的</strong>。NVIDIA 设计好一个 SM，
              复制 144 份铺满整片 die，所以 GPU 的代际升级很大程度上就是「SM 更强 + SM 更多」，而你写的程序
              要想吃满硬件，就得切成足够多的独立任务块去喂饱所有 SM。其二，<strong>灰色的格子是真实存在的</strong>。
              大芯片良率有限，出厂时屏蔽掉有缺陷的 SM 是行业惯例，同一颗 GH100 die 按屏蔽数量分档出售。其三，
              看 SM 内部时注意每个分区恰好是 32 个 FP32 核。这个数字马上就要变成主角。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={3}
        title={t('Warps and SIMT: 32 threads, one instruction', 'Warp 与 SIMT：32 个线程，一条指令')}
        lead={t(
          "The smallest unit of hardware scheduling isn't a thread. It's a warp of 32 threads.",
          '硬件调度的最小单位不是线程，是 32 个线程组成的 warp。',
        )}
      >
        <p>
          {t(
            <>
              CUDA teaches you to "launch tens of thousands of threads," but from the hardware's point of view there is
              no such thing as a single thread. When an SM receives a thread block, it slices the threads into groups of
              32 called{' '}
              <Term t="warp">
                A scheduling unit of 32 consecutive threads that execute the same instruction at the same instant. The
                name comes from the warp threads on a weaving loom.
              </Term>
              , and the warp is the atomic unit of fetch, issue, and execution: the 32 threads in a warp execute{' '}
              <strong>the same instruction</strong> in the same cycle, exactly filling a partition's 32 FP32 cores.
              NVIDIA calls this model <strong>SIMT</strong> (Single Instruction, Multiple Threads).
            </>,
            <>
              CUDA 教你「启动几万个线程」，但硬件视角里根本没有单个线程这回事。SM 接收线程块后，把其中的线程按
              32 个一组切成{' '}
              <Term t="warp（线程束）">
                32 个连续线程组成的调度单位，同一时刻执行同一条指令。名字来自织布机上的「经线」。
              </Term>
              ，warp 才是取指、发射、执行的原子单位：一个 warp 里的 32 个线程在同一个周期执行<strong>同一条指令</strong>，
              正好铺满一个分区的 32 个 FP32 核。NVIDIA 把这种模式叫 <strong>SIMT</strong>（Single Instruction,
              Multiple Threads，单指令多线程）。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              At the hardware level SIMT is a close cousin of the CPU's SIMD (Single Instruction, Multiple Data): both
              drive a row of execution units from one instruction. The <strong>programming model, though, is completely
              different</strong>. Writing AVX-512, the vector width is your problem. You explicitly manipulate vector
              registers of 16 floats and handle alignment, tails, and masking yourself. Writing CUDA, you write an{' '}
              <strong>ordinary scalar thread</strong>: each thread has its own registers, its own program counter (per-thread
              PCs since Volta, even), its own branch direction, and "bundling them 32 at a time to run in lockstep" is
              something the hardware does behind your back. That is where CUDA's usability comes from. Anyone can write
              scalar code; almost nobody wants to touch vector intrinsics.
            </>,
            <>
              在硬件层面，SIMT 和 CPU 的 SIMD（Single Instruction, Multiple Data，单指令多数据）是近亲：
              都是一条指令驱动一排运算单元。可<strong>编程模型完全不同</strong>。写 AVX-512 时，向量宽度是你的事：
              你显式操作 16 个 float 的向量寄存器，对齐、收尾、掩码都得自己处理。写 CUDA 时，你写的是一个
              <strong>普通的标量线程</strong>：每个线程有自己的寄存器、自己的程序计数器（Volta 之后甚至每线程独立
              PC）、自己的分支走向，把这些线程「捆成 32 一组锁步执行」是硬件背着你做的。CUDA 的易用性就是这么来的：
              标量代码人人会写，向量 intrinsics 没几个人愿意碰。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              But abstractions always leak somewhere. The 32 threads share one fetch unit, so when threads
              inside a warp head down different branches, the hardware can only <strong>walk both paths in turn</strong>,
              using an active mask to make the threads on the wrong path tag along: mask off the B threads while
              executing branch A, then mask off the A threads while executing branch B. This is{' '}
              <Term t="warp divergence">
                When threads in one warp take different branches, the hardware executes every branch path serially and uses
                a mask to switch off the irrelevant threads. The cost of the two sides adds up rather than taking the max.
              </Term>
              :
            </>,
            <>
              但抽象总有漏出来的地方。32 个线程共用一个取指单元，于是当 warp 内部走向不同分支时，硬件只能
              <strong>把两条路径都执行一遍</strong>，靠活动掩码（active mask）让走错路的线程「陪跑」：
              执行 A 分支时屏蔽走 B 的线程，再执行 B 分支时屏蔽走 A 的线程。这就是{' '}
              <Term t="分支分化（warp divergence）">
                同一 warp 内线程走向不同分支时，硬件串行执行所有分支路径、用掩码关闭不相关线程的现象。
                分支两侧的耗时是相加而不是取最大。
              </Term>
              ：
            </>,
          )}
        </p>
        <CodeBlock code={t(DIVERGE_CODE_EN, DIVERGE_CODE_ZH)} lang="cuda" title="divergence.cu" highlight={[2, 4]} />
        <p>
          {t(
            <>
              Note that divergence only happens <strong>within a warp</strong>: if all 32 threads of a warp take the same
              branch in unison (say, branching on <code>blockIdx</code>, or on <code>threadIdx.x / 32</code>), there is no
              penalty at all. So high-performance CUDA code either aligns its branches to warp boundaries or rewrites them
              into branch-free arithmetic.
            </>,
            <>
              注意分化只发生在 <strong>warp 内部</strong>：如果整个 warp 的 32 个线程齐刷刷走同一条分支
              （比如按 <code>blockIdx</code> 分支，或按 <code>threadIdx.x / 32</code> 分支），没有任何惩罚。
              所以高性能 CUDA 代码的分支要么对齐到 warp 边界，要么干脆改写成无分支的算术形式。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The warp's second identity matters even more: it is <strong>the currency of latency hiding</strong>. Each
              partition's warp scheduler keeps several warps resident at once (up to 64 per SM on H100, 16 per partition).
              Every cycle it glances at which warps have their next instruction ready, operands and all, and picks one to
              issue. A warp waiting on memory? No problem. It sits quietly while the scheduler turns to issue a different
              warp, filling the execution units by swapping in someone else to work rather than making one worker wait
              less. This is the watershed between CPU and GPU: a CPU <strong>shortens</strong> latency with big
              caches and out-of-order execution, while a GPU <strong>hides</strong> it under a flood of resident warps. And
              the reason a warp switch costs zero is precisely that inverted register file from section 1. All the registers
              of 64 warps <strong>stay resident simultaneously</strong> in the 256 KB register file, so a switch saves and
              restores no state; the scheduler simply fetches from a different number. A CPU thread context
              switch, by contrast, has to shuttle a whole register set in and out of memory, at a cost measured in microseconds.
            </>,
            <>
              warp 的第二重身份更重要：它是<strong>延迟隐藏的筹码</strong>。每个分区的 warp 调度器手里同时
              「驻留」着多个 warp（H100 每 SM 最多 64 个，每分区 16 个），每个周期它扫一眼哪些 warp 的下一条指令
              连操作数都就绪了，从中挑一个发射。某个 warp 在等显存？没关系，它安静地挂着，调度器转头发射
              别的 warp，靠「换人干活」而不是「让一个人少等点」来填满执行单元。这就是 CPU 与 GPU 的分水岭：
              CPU 用大缓存和乱序执行<strong>缩短</strong>延迟，GPU 用海量驻留 warp <strong>隐藏</strong>延迟。
              warp 切换之所以能做到零开销，正是因为第 1 节那个倒挂的寄存器堆：64 个 warp 的全部寄存器
              <strong>同时常驻</strong>在 256 KB 的寄存器堆里，切换不用保存恢复任何状态，调度器只是换一个编号去取指。
              CPU 的线程上下文切换则相反，要进出内存搬运整套寄存器状态，开销是微秒级的。
            </>,
          )}
        </p>
      </Section>

      <Quiz
        question={t(
          'In one warp, 13 of the 32 threads satisfy the if condition and 19 take the else. How does the hardware execute this?',
          '一个 warp 内的 32 个线程，有 13 个满足 if 条件、19 个走 else。硬件会怎么执行？',
        )}
        options={[
          {
            text: t('Split the warp into two groups and run both branches in parallel', '把 warp 拆成两个小组，两条分支并行执行'),
            explain: t(
              'A warp is an indivisible scheduling unit and can only issue one instruction per cycle; there is no "split and parallelize."',
              'warp 是不可拆分的调度单位，一个周期只能发射一条指令，不存在「拆开并行」。',
            ),
          },
          {
            text: t('Walk both branch paths in turn, using the active mask to make the irrelevant threads tag along', '两条分支路径都执行一遍，用活动掩码让不相关的线程陪跑'),
            correct: true,
            explain: t(
              'Correct. The 32 threads share a fetch unit, so on divergence the hardware runs every path serially: it masks off 19 threads while running the if, masks off 13 while running the else, and the total time is roughly the sum of both paths.',
              '正确。32 个线程共用取指单元，分化时硬件串行执行所有路径：跑 if 时掩蔽 19 个线程，跑 else 时掩蔽 13 个，总耗时约等于两条路径之和。',
            ),
          },
          {
            text: t('Run only the branch the majority (19 threads) chose, discarding the minority\'s results', '只执行多数线程（19 个）选择的分支，少数线程的结果作废'),
            explain: t(
              "Every thread's semantics must be correct; the hardware never discards any thread's computation. It guarantees correctness with masking, at the cost of time.",
              '每个线程的语义都必须正确，硬件不会丢弃任何线程的计算，而是靠掩码保证正确性，代价是时间。',
            ),
          },
          {
            text: t('Raise an exception; CUDA does not allow branches within a warp', '触发异常，CUDA 不允许 warp 内出现分支'),
            explain: t(
              'Branching is perfectly legal and the result is correct; performance just drops because the two paths run serially. It is a performance trap, not an error.',
              '分支完全合法，程序结果也正确，只是性能会因为两路串行执行而下降。这是性能陷阱，不是错误。',
            ),
          },
        ]}
      />

      <Section
        index={4}
        title={t('LAB · Warp latency hiding', 'LAB · Warp 延迟隐藏')}
        lead={t(
          "The timeline of one SM partition: watch the ALU's bubbles get filled in, one warp at a time.",
          '一个 SM 分区的时间轴：看 ALU 的空泡如何被一个个 warp 填满。',
        )}
      >
        <p>
          {t(
            <>
              The simulator below reconstructs the scheduling of one partition. Every warp does the same thing on repeat:
              issue a few compute instructions, then issue a memory instruction and stall for hundreds of cycles. Each
              cycle the scheduler picks one ready warp to issue. First drag the warp count down to 1 and see how big a hole
              the memory latency tears in the timeline; then add warps and adjust the compute density, and watch ALU
              utilization climb toward 100%.
            </>,
            <>
              下面的模拟器还原一个分区的调度过程。每个 warp 反复做同一件事：发射几条计算指令，然后发射一条访存指令、
              进入上百个周期的等待。调度器每周期从就绪的 warp 里挑一个发射。先把 warp 数拖到 1，看看访存延迟
              把时间轴撕出多大的空洞；再逐渐加 warp、调整计算密度，观察 ALU 利用率怎么爬向 100%。
            </>,
          )}
        </p>
        <WarpLatencyLab />
        <p>
          {t(
            <>
              Behind this toy model is a formula worth writing on the wall. To fully hide a memory access of latency{' '}
              <code>L</code>, there has to be other work ready to issue during the wait. Each warp contributes about{' '}
              <code>C+1</code> instructions per round and then goes silent for <code>L</code> cycles, so utilization is
              roughly <code>W×(C+1) / (C+1+L)</code>, capped at 100%. That gives you two dials for raising utilization:{' '}
              <strong>increase W</strong> (more resident warps, i.e. higher occupancy) or <strong>increase C</strong> (more
              compute between each memory access, i.e. a higher compute-to-memory ratio). The optimization stories of{' '}
              <ChapterLink n={5} /> and <ChapterLink n={6} /> all turn on these two dials.
            </>,
            <>
              这个玩具模型背后有一条可以写在墙上的公式。要把延迟为 <code>L</code> 的访存完全藏住，等待期间就得有别的活
              可干。每个 warp 每轮贡献约 <code>C+1</code> 条指令、随后沉默 <code>L</code> 个周期，所以利用率约为{' '}
              <code>W×(C+1) / (C+1+L)</code>，封顶 100%。这就给了你两个提高利用率的旋钮：<strong>加 W</strong>
              （更多驻留 warp，即更高 occupancy）或者<strong>加 C</strong>（每次访存之间做更多计算，
              即更高的计算访存比）。<ChapterLink n={5} />、<ChapterLink n={6} />的优化故事，全部是围绕这两个旋钮展开的。
            </>,
          )}
        </p>
        <Callout type="insight" title={t("GPUs don't eliminate latency, they hide it", 'GPU 不消灭延迟，只是把延迟藏起来')}>
          <p>
            {t(
              <>
                An HBM access takes hundreds of cycles, and that physical fact is one the GPU cannot change, nor does
                it intend to; its caches are even smaller than a CPU's. The GPU's bet is simple: as long as there is enough
                resident work, the wait can always be covered by someone else's computation. This is why a GPU program
                launches tens of times more threads than there are cores. The extra threads aren't waste; they are the
                cards the scheduler holds to fill the bubbles. Run short of cards and even an ocean of cores just idles.
              </>,
              <>
                HBM 一次访问几百个周期，这个物理事实 GPU 改变不了，它甚至也不打算改变，毕竟它的缓存比 CPU
                的还小。GPU 的赌注很简单：只要同时驻留的工作足够多，等待就永远可以被别人的计算盖住。
                这就是为什么 GPU 程序要启动比核心数多几十倍的线程。多出来的线程不是浪费，
                而是调度器手里用来填空泡的牌。牌不够，再多的核心也只能空转。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Quiz
        question={t(
          'Why should an SM keep far more warps resident than it has execution units (up to 64 on H100)?',
          '为什么一个 SM 上要同时驻留远多于执行单元数量的 warp（H100 上最多 64 个）？',
        )}
        options={[
          {
            text: t('To run more threads in parallel and raise peak compute', '为了让更多线程并行执行，提高峰值算力'),
            explain: t(
              'Peak compute is fixed by the number of execution units; no amount of extra resident warps raises it, since each partition still issues only one instruction per cycle.',
              '峰值算力由执行单元数量决定，驻留再多 warp 也不会提高峰值，因为每个分区每周期仍然只能发射一条指令。',
            ),
          },
          {
            text: t('So that when some warps are waiting on memory, the scheduler has other ready warps to issue, hiding the latency', '为了在某些 warp 等待访存时，调度器有别的就绪 warp 可以发射，从而隐藏延迟'),
            correct: true,
            explain: t(
              'Correct. Resident warps are the capital for latency hiding: a waiting warp ties up no execution units, and the scheduler switches to a ready warp at zero cost, keeping the ALU busy at all times.',
              '正确。驻留 warp 是延迟隐藏的本钱：等待中的 warp 不占执行单元，调度器零开销切换到就绪 warp，让 ALU 始终有活干。',
            ),
          },
          {
            text: t('To reduce the chance of warp divergence', '为了减少分支分化的概率'),
            explain: t(
              'Divergence happens among the 32 threads inside a single warp and has nothing to do with how many warps are resident.',
              '分化发生在单个 warp 内部的 32 个线程之间，与驻留多少个 warp 无关。',
            ),
          },
          {
            text: t('Because a thread block must reside as a whole, a side effect of the programming model', '因为线程块必须整块驻留，这是编程模型的副作用'),
            explain: t(
              'A thread block does reside as a whole, but that is a consequence, not the goal. The hardware deliberately built a 256 KB register file precisely so it can keep many warps resident for latency hiding.',
              '线程块确实整块驻留，但这是结果不是目的。硬件特意把寄存器堆做到 256 KB，就是为了能驻留大量 warp 用于延迟隐藏。',
            ),
          },
        ]}
      />

      <Section
        index={5}
        title={t('The memory hierarchy: an order of magnitude per level', '存储层级：每一级都差一个数量级')}
        lead={t(
          'Data is the raw material of compute, and where the data lives decides how many cycles one fetch burns.',
          '算力的食材是数据，而数据放在哪里，决定了取一次要烧掉多少周期。',
        )}
      >
        <p>
          {t(
            <>
              Latency hiding is about "doing something else while you wait," but a better strategy is always "wait less."
              GPU storage is a pyramid: from inside the SM out to off-die memory, each step down grows an order of magnitude
              in capacity and costs an order of magnitude more in latency.
            </>,
            <>
              延迟隐藏讲的是「等待时干别的」，但更好的策略永远是「少等」。GPU 的存储是一座金字塔，
              从 SM 内部到片外显存，每往下走一层，容量大一个数量级，延迟也贵一个数量级：
            </>,
          )}
        </p>
        <MemTable />
        <p>
          {t(
            <>
              This table rewards repeated chewing. <strong>Registers</strong> sit right at the execution units, almost free
              to read and write, but capped at 255 per thread; exceed that and they spill to memory, a pitfall you'll hit
              in <ChapterLink n={5} />. <strong>Shared memory and L1</strong> are two uses of the same on-chip SRAM. L1 is
              hardware-managed; shared memory is a programmer-controlled scratchpad that threads within one block use to
              exchange data at ~30 cycles, more than an order of magnitude cheaper than going to memory. <strong>L2</strong> is
              a chip-wide relay shared by all 132 SMs; 50 MB sounds generous, but spread across tens of thousands of concurrent
              threads it gets tight fast. At the bottom, <strong>HBM3</strong> latency runs 500+ cycles, measured at over 600
              on Hopper, and one round trip takes long enough for a partition to issue hundreds of compute instructions.
            </>,
            <>
              这张表值得反复咀嚼。<strong>寄存器</strong>就在执行单元手边，读写几乎免费，但每个线程最多 255 个，
              用超了就会「溢出」到显存，<ChapterLink n={5} />会撞上这个坑。<strong>共享内存和 L1</strong> 是同一块片上
              SRAM 的两种用法：L1 由硬件自动管理，共享内存则是程序员显式控制的便笺纸，同一个线程块内的线程
              用它交换数据，延迟约 30 个周期，比走显存便宜一个数量级以上。<strong>L2</strong> 是全片 132 个 SM
              共享的中转站，50 MB 听起来不小，但被几万个并发线程一摊就很紧张了。最底层的 <strong>HBM3</strong>{' '}
              延迟在 500 周期以上，实测 Hopper 上甚至能到 600 多个周期，一次往返的时间足够一个分区发射几百条
              计算指令。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Feel the gap for yourself: each time you click "fetch" below, a data block flies one round trip along its real
              path and the cycle counter keeps honest books. Click HBM a few times, then registers a few times, and the bar
              chart on the right lays the brutal contrast in front of you.
            </>,
            <>
              亲手感受一下这个落差：下面每点一次「取数」，就有一个数据方块沿真实路径飞一个来回，周期计数器会诚实记账。
              连点几次 HBM，再连点几次寄存器，右侧的柱状图会把残酷的对比摆在你面前。
            </>,
          )}
        </p>
        <MemoryLatencyLab />
        <p>
          {t(
            <>
              While you're at it, recalibrate one intuition: HBM's <strong>bandwidth</strong> (3.35 TB/s) is tens of times that
              of CPU memory, but its <strong>latency</strong> is no better, arguably worse. The GPU memory
              system is a truck convoy built for throughput, not a sports car built for latency. So batching up data and moving it
              in one shot beats fetching piecemeal on demand by a wide margin, a principle that runs through coalesced access in{' '}
              <ChapterLink n={4} /> and FlashAttention in <ChapterLink n={8} />.
            </>,
            <>
              顺带校准一个直觉：HBM 的<strong>带宽</strong>（3.35 TB/s）是 CPU 内存的几十倍，但<strong>延迟</strong>
              并不比 CPU 内存好，甚至更差。GPU 的显存系统是为吞吐设计的卡车车队，而非为延迟设计的跑车，
              所以「攒一批数据一次搬运」远胜「零碎地按需取用」，这个原则会贯穿<ChapterLink n={4} />的合并访存和<ChapterLink n={8} />的
              FlashAttention。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={6}
        title={t('A glance at Tensor Cores: dedicated matrix-multiply circuitry', 'Tensor Core 一瞥：矩阵乘的专用电路')}
        lead={t(
          'When 95% of the FLOPs go into matrix multiplies, building dedicated hardware for them is the natural move.',
          '当 95% 的 FLOPs 都花在矩阵乘上，为它造专门的硬件就是顺理成章的事。',
        )}
      >
        <p>
          {t(
            <>
              Everything so far has been about FP32 CUDA cores, one scalar multiply-add per cycle. But the H100 spec sheet
              carries a far more extravagant number: 989 TFLOPS in BF16, roughly 15× the FP32 CUDA cores (67 TFLOPS). That gap
              comes from the purple block in each partition, the <strong>Tensor Core</strong>. It does no scalar arithmetic.
              One <strong>MMA</strong> (Matrix Multiply-Accumulate) instruction multiplies a small matrix tile and accumulates
              it: a group of warps cooperating on a <code>16×8×16</code> tile multiply-add, say, one instruction doing
              the work of hundreds of scalar ones.
            </>,
            <>
              到目前为止我们聊的都是 FP32 CUDA 核心，每个周期做一次标量乘加。但 H100 的规格表上还有一个
              夸张得多的数字：BF16 精度下 989 TFLOPS，是 FP32 CUDA 核心（67 TFLOPS）的约 15 倍。这个差距来自每个
              分区里那个紫色的块，<strong>Tensor Core</strong>。它不做标量运算，一条 <strong>MMA</strong>
              （Matrix Multiply-Accumulate，矩阵乘加）指令直接完成一小块矩阵的乘法并累加：比如一组 warp
              协作完成 <code>16×8×16</code> 的矩阵片乘加，一条指令顶过去几百条标量指令。
            </>,
          )}
        </p>
        <Figure
          caption={t(
            'H100 SXM peak throughput by precision. Tensor Cores beat CUDA cores by about an order of magnitude, and throughput nearly doubles with each step down in precision.',
            'H100 SXM 各精度峰值吞吐。Tensor Core 对 CUDA 核心的优势约一个数量级，精度每降一档吞吐近乎翻倍。',
          )}
        >
          <FlopsChart />
        </Figure>
        <p>
          {t(
            <>
              Why is dedicated circuitry so much faster? Because matrix multiplication is intensely structured. A 16×16 tile has
              heavy operand reuse inside it, so dedicated hardware can amortize the cost of fetch, decode, and operand movement
              across hundreds of multiply-adds, where a scalar core pays all that administrative overhead for every single
              one. Add deep learning's tolerance for low precision (BF16 has become the default for training, FP8 increasingly so
              for both training and inference) and each halving of precision doubles throughput for the same silicon area and
              bandwidth. That is why Tensor Cores became the undisputed star of the LLM era: by FLOP count, 95%+ of a Transformer's
              forward and backward passes are matrix multiplies (<ChapterLink n={7} /> does the arithmetic).
            </>,
            <>
              为什么专用电路能快这么多？因为矩阵乘有极强的结构性。一块 16×16 的数据片内部有大量操作数复用，
              专用电路可以把取指、译码、操作数搬运的开销摊薄到几百次乘加上，而标量核心每做一次乘加都要付一遍这些
              「行政成本」。再加上深度学习对低精度的宽容（BF16 已是训练的默认，FP8 在训练和推理里也越来越常见），
              精度每砍一半，同样的硅片面积和带宽就能再翻一倍吞吐。这就是大模型时代 Tensor Core 成为绝对主角的原因：
              Transformer 的前向反向，按 FLOPs 算 95% 以上是矩阵乘（<ChapterLink n={7} />会算这笔账）。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The good news is you almost never have to write Tensor Core instructions directly: NVIDIA's own libraries like
              cuBLAS and cuDNN select them automatically, and PyTorch's <code>torch.matmul</code> routes through Tensor Cores by
              default in half precision. The bad news is that it raises the bar on feeding data by an order of magnitude. The
              faster it computes, the easier it starves, and a starved Tensor Core might as well not be there at all. That
              is the central tension of the Roofline model in <ChapterLink n={6} /> and the entire reason FlashAttention exists in{' '}
              <ChapterLink n={8} />.
            </>,
            <>
              好消息是你几乎不需要直接编写 Tensor Core 指令：cuBLAS、cuDNN 这些 NVIDIA 官方库会自动选用它，
              PyTorch 的 <code>torch.matmul</code> 在半精度下默认走 Tensor Core。坏消息是它把对「喂数据」的要求
              抬高了一个数量级：算得越快，越容易饿，一颗吃不饱的 Tensor Core 和没有 Tensor Core 没什么两样。
              这是<ChapterLink n={6} /> Roofline 模型要讲的核心矛盾，也是<ChapterLink n={8} /> FlashAttention 存在的全部理由。
            </>,
          )}
        </p>
      </Section>

      <Section index={7} title={t('Summary and further reading', '总结与延伸阅读')}>
        <p>{t("Having taken an H100 apart, carry these intuitions into the next chapter:", '这一章拆完了一块 H100，把这几条直觉带去下一章：')}</p>
        <ul>
          <li>
            {t(
              <>
                A GPU is <strong>a hundred-plus copies of the same SM</strong>: 132 on H100 SXM, 108 on A100. The SM is the unit
                of allocation for compute, registers, and shared memory, and the place your thread blocks reside.
              </>,
              <>
                GPU 是<strong>同一个 SM 的一百多份拷贝</strong>：H100 SXM 132 个、A100 108 个。SM 是计算、
                寄存器、共享内存的分配单位，也是你的线程块的驻留地。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Each SM = 4 partitions × (1 warp scheduler + 32 FP32 cores + 1 Tensor Core), with a 256 KB register file and
                256 KB of L1 / shared memory (up to 228 KB configurable as shared).
              </>,
              <>
                每个 SM = 4 个分区 × (1 warp 调度器 + 32 FP32 核 + 1 Tensor Core)，配 256 KB 寄存器堆和
                256 KB L1/共享内存（Shared 可配至 228 KB）。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                The atomic unit of hardware scheduling is the <strong>warp (32 threads in lockstep)</strong>. You write scalar
                threads; the hardware bundles them into warps, and on divergence it walks both paths, using a mask to keep
                results correct.
              </>,
              <>
                硬件调度的原子单位是 <strong>warp（32 线程锁步）</strong>。你写标量线程，硬件捆成 warp 执行；
                分支分化时两路都要走，靠掩码保证正确性。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                A GPU hides memory latency with <strong>many resident warps plus zero-cost switching</strong>, not by shortening
                latency with big caches. The register file is larger than L1 precisely so every resident warp's state can stay
                on board at once.
              </>,
              <>
                GPU 靠<strong>驻留大量 warp + 零成本切换</strong>隐藏访存延迟，而不是靠大缓存缩短延迟。
                寄存器堆比 L1 还大，就是为了让所有驻留 warp 的状态同时在板上。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                The storage pyramid differs by an order of magnitude per level: registers ≈1 cycle, Shared/L1 ≈30, L2 ≈200,
                HBM ≈500+. Where the data lives is the first question of performance optimization.
              </>,
              <>
                存储金字塔每层差一个数量级：寄存器 ≈1 周期、Shared/L1 ≈30、L2 ≈200、HBM ≈500+。
                数据放哪里，是性能优化的第一问题。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                A Tensor Core does a whole tile's matrix multiply-accumulate in one MMA instruction, with BF16 throughput about
                15× the FP32 CUDA cores. It is the compute protagonist of large models, invoked automatically by cuBLAS/cuDNN/PyTorch.
              </>,
              <>
                Tensor Core 用一条 MMA 指令做整片矩阵乘加，BF16 吞吐约为 FP32 CUDA 核的 15 倍，
                是大模型的算力主角，由 cuBLAS/cuDNN/PyTorch 自动调用。
              </>,
            )}
          </li>
        </ul>
        <p>{t('To dig deeper, these are worth your time:', '想钻得更深，这几份材料值得花时间：')}</p>
        <ul>
          <li>
            <a href="https://resources.nvidia.com/en-us-tensor-core" target="_blank" rel="noreferrer">
              NVIDIA H100 Tensor Core GPU Architecture Whitepaper
            </a>{' '}
            {t(
              '— the official Hopper whitepaper, the source of every spec number in this chapter; its SM diagrams are well worth a close read.',
              '—— Hopper 架构官方白皮书，本章所有规格数字的出处，SM 结构图非常值得细看。',
            )}
          </li>
          <li>
            <a
              href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA A100 Tensor Core GPU Architecture Whitepaper
            </a>{' '}
            {t(
              '— the Ampere whitepaper. Read it against the Hopper specs to trace how the architecture evolved.',
              '—— Ampere 白皮书。对照两代规格读，能看出架构演进的脉络。',
            )}
          </li>
          <li>
            <a
              href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA Hopper Architecture In-Depth
            </a>{' '}
            {t(
              '— the official blog deep-dive, lighter than the whitepaper, with a focus on new features like TMA and thread block clusters.',
              '—— 官方博客版深度导览，比白皮书轻松，重点讲了 TMA 和线程块簇等新特性。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2402.13499" target="_blank" rel="noreferrer">
              Benchmarking and Dissecting the Nvidia Hopper GPU Architecture
            </a>{' '}
            {t(
              "— a paper that measures Hopper's latencies and bandwidths at each level with microbenchmarks, the empirical reference for this chapter's latency figures.",
              '—— 用微基准实测 Hopper 各级延迟与带宽的论文，本章延迟数字的实测参照。',
            )}
          </li>
          <li>
            <a
              href="https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors"
              target="_blank"
              rel="noreferrer"
            >
              {t('Programming Massively Parallel Processors (4th ed.), Chapter 4', 'Programming Massively Parallel Processors（第 4 版）第 4 章')}
            </a>{' '}
            {t(
              '— the architecture-and-scheduling chapter of the PMPP textbook, the most systematic treatment of SIMT and latency hiding in print.',
              '—— PMPP 教材中讲架构与调度的一章，把 SIMT 与延迟隐藏讲得最系统的教科书材料。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
