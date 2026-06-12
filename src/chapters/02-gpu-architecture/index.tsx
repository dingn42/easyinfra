import { Callout, CodeBlock, Figure, Quiz, Section, Term } from '@/components/ui'
import { AnatomyLab } from './AnatomyLab'
import { WarpLatencyLab } from './WarpLatencyLab'
import { MemoryLatencyLab } from './MemoryLatencyLab'

/* ── SEC06 配图：CUDA core vs Tensor Core 吞吐对比 ── */
const FLOPS_BARS = [
  { label: 'FP64 CUDA 核', v: 34, cls: 'fill-ink3', txt: 'fill-ink2' },
  { label: 'FP32 CUDA 核', v: 67, cls: 'fill-cyan/70', txt: 'fill-cyan' },
  { label: 'TF32 Tensor Core', v: 495, cls: 'fill-violet/70', txt: 'fill-violet' },
  { label: 'BF16 Tensor Core', v: 989, cls: 'fill-volt', txt: 'fill-volt' },
  { label: 'FP8 Tensor Core', v: 1979, cls: 'fill-volt/60', txt: 'fill-volt' },
]

function FlopsChart() {
  const max = 1979
  return (
    <svg viewBox="0 0 720 196" className="w-full select-none" role="img" aria-label="H100 各精度峰值吞吐对比">
      {FLOPS_BARS.map((b, i) => {
        const y = 10 + i * 36
        const w = Math.max(3, (b.v / max) * 520)
        return (
          <g key={b.label}>
            <text x={148} y={y + 15} fontSize={11} textAnchor="end" className="fill-ink2 font-mono">
              {b.label}
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
        H100 SXM 峰值（dense，不含 2:4 稀疏加成）· BF16 Tensor ≈ 15× FP32 CUDA 核
      </text>
    </svg>
  )
}

/* ── SEC05 配表：存储层级 ── */
const MEM_TABLE = [
  ['寄存器（Registers）', '256 KB / SM（全卡 ≈ 33 MB）', '每周期全分区并发', '≈1 周期', 'text-volt'],
  ['共享内存 / L1', '256 KB / SM（Shared 可配 228 KB）', '聚合约 30 TB/s', '≈30 周期', 'text-cyan'],
  ['L2 缓存', '50 MB（全片共享）', '约 5~6 TB/s', '≈200~300 周期', 'text-violet'],
  ['HBM3 显存', '80 GB', '3.35 TB/s', '≈500 周期以上', 'text-amber'],
] as const

function MemTable() {
  return (
    <div className="my-6 overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-line2 text-left">
            {['层级', '容量', '带宽', '延迟'].map((h) => (
              <th key={h} className="microlabel py-2 pr-4 font-normal">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MEM_TABLE.map((row) => (
            <tr key={row[0]} className="border-b border-line">
              <td className={`py-2.5 pr-4 font-medium ${row[4]}`}>{row[0]}</td>
              <td className="py-2.5 pr-4 font-mono text-[12px] tabular-nums text-text">{row[1]}</td>
              <td className="py-2.5 pr-4 font-mono text-[12px] tabular-nums text-text">{row[2]}</td>
              <td className="py-2.5 font-mono text-[12px] tabular-nums text-ink">{row[3]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const DIVERGE_CODE = `// warp 内的 32 个线程锁步执行。遇到分支怎么办？
if (threadIdx.x % 2 == 0) {
    out[i] = expensive_a(x);   // 偶数线程想走这里
} else {
    out[i] = expensive_b(x);   // 奇数线程想走这里
}
// 硬件的答案：两条路都走一遍。
// 先执行 a 分支（奇数线程被掩码挂起），再执行 b 分支（偶数线程挂起）。
// 总耗时 ≈ a + b，一半的执行槽位被白白烧掉。`

export default function Chapter() {
  return (
    <>
      <p>
        H100 的规格表上写着 16896 个 FP32 核心，是顶级服务器 CPU 核数的一百多倍。但如果你以为这是
        16896 个可以各干各事的小 CPU，那从第一行 CUDA 代码开始就会困惑不断：为什么一个 <code>if</code>{' '}
        语句能让性能掉一半？为什么线程数要开到核心数的几十倍才能跑满？答案藏在硬件组织方式里 ——
        这些核心以 32 个为一组<strong>锁步（lockstep）行动</strong>，共享取指和调度。这一章我们把一块
        H100 拆开，从硅片一路看到执行单元，搞清楚这台机器到底是按什么规则运转的。后面每一章的性能直觉，
        根子都在这里。
      </p>

      <Section
        index={1}
        title="自顶向下：从 die 到 SM"
        lead="GPU 不是一个处理器，而是同一个处理器的一百多份拷贝，外加一套喂数据的管线。"
      >
        <p>
          先建立层级感。一块 H100 的硅片（die，代号 GH100）面积 814 mm²，用台积电 4N 工艺塞进了约
          800 亿个晶体管 —— 接近光刻机单次曝光的物理上限。往里看，它的组织是严格的套娃结构：整片 die
          划分为 8 个 <strong>GPC</strong>（Graphics Processing Cluster，图形处理簇），每个 GPC 里有 9 个{' '}
          <strong>TPC</strong>（Texture Processing Cluster，纹理处理簇），每个 TPC 装着 2 个{' '}
          <Term t="SM（Streaming Multiprocessor，流式多处理器）">
            GPU 的基本计算单元。线程块整块驻留在 SM 上，由它提供寄存器、共享内存和执行单元。记住它，
            后面每一章都绕不开。
          </Term>
          。乘起来是 144 个 SM；考虑良率，H100 SXM 版屏蔽掉 12 个，实际启用 <strong>132 个 SM</strong>
          （上一代 A100 是 108 个）。GPC 和 TPC 对程序员几乎透明，真正需要记住的层级只有两个：
          <strong>die 和 SM</strong>。
        </p>
        <p>
          SM 才是 GPU 真正意义上的「核」。每个 H100 的 SM 内部是四个结构完全相同的处理分区，每个分区有
          1 个 warp 调度器和 32 个 FP32 核心，合计 <strong>128 个 FP32 核心、4 个 warp 调度器、4 个
          Tensor Core</strong>；四个分区各自挂着 64 KB 寄存器堆（合计 <strong>256 KB</strong>），共用一块
          256 KB 的 L1 数据缓存/共享内存（其中最多 228 KB 可配置为共享内存）。把 SM 的面积分配和 CPU
          核对比一下，你会看到两种完全相反的设计哲学：CPU 核把大部分晶体管花在乱序执行、分支预测和多级大缓存上，
          目标是让<strong>单个线程</strong>跑得尽量快；SM 把这些全部砍掉，省下的面积全部换成执行单元和寄存器 ——
          它从一开始就没打算让任何一个线程跑得快，它要的是<strong>同时在飞的线程足够多</strong>。
        </p>
        <p>
          注意一个反直觉的数字：SM 的寄存器堆（256 KB）比它的 L1 缓存还要大，而 CPU 单核的寄存器加起来
          不过几 KB。这个「倒挂」不是设计失误，而是整个 GPU 延迟隐藏机制的物质基础 —— 第 3 节会回收这个伏笔。
          芯片的其余部分都是为这 132 个 SM 服务的「后勤系统」：die 中央一条 50 MB 的 L2 缓存为全片共享，
          die 两侧贴着 5 颗 HBM3 显存堆叠提供 80 GB 容量和 3.35 TB/s 带宽，顶部的 NVLink 接口负责多卡互联，
          底部的 GigaThread 引擎负责把线程块分发到各个 SM。下面这台解剖台可以亲手拆一遍。
        </p>
      </Section>

      <Section
        index={2}
        title="LAB · GPU 解剖台"
        lead="层 1 是整片 die，点击任意发亮的 SM 可以放大到层 2；点击任何部件，右侧给出它的档案。"
      >
        <AnatomyLab />
        <p>
          逛完一圈值得留意三件事。其一，<strong>SM 是「复制粘贴」出来的</strong>：NVIDIA 设计好一个 SM，
          复制 144 份铺满整片 die —— 所以 GPU 的代际升级很大程度上就是「SM 更强 + SM 更多」，而你写的程序
          要想吃满硬件，就必须切成足够多的独立任务块去喂饱所有 SM。其二，<strong>灰色的格子是真实存在的</strong>：
          大芯片良率有限，出厂时屏蔽掉有缺陷的 SM 是行业惯例，同一颗 GH100 die 按屏蔽数量分档出售。其三，
          看 SM 内部时注意每个分区恰好是 32 个 FP32 核 —— 这个数字马上就要变成主角。
        </p>
      </Section>

      <Section
        index={3}
        title="Warp 与 SIMT：32 个线程，一条指令"
        lead="硬件调度的最小单位不是线程，是 32 个线程组成的 warp。"
      >
        <p>
          CUDA 教你「启动几万个线程」，但硬件视角里根本没有单个线程这回事。SM 接收线程块后，把其中的线程按
          32 个一组切成{' '}
          <Term t="warp（线程束）">
            32 个连续线程组成的调度单位，同一时刻执行同一条指令。名字来自织布机上的「经线」。
          </Term>
          ，warp 才是取指、发射、执行的原子单位：一个 warp 里的 32 个线程在同一个周期执行<strong>同一条指令</strong>，
          正好铺满一个分区的 32 个 FP32 核。NVIDIA 把这种模式叫 <strong>SIMT</strong>（Single Instruction,
          Multiple Threads，单指令多线程）。
        </p>
        <p>
          SIMT 和 CPU 的 SIMD（Single Instruction, Multiple Data，单指令多数据）在硬件层面是近亲 ——
          都是一条指令驱动一排运算单元 —— 但<strong>编程模型完全不同</strong>。写 AVX-512 时，向量宽度是你的事：
          你显式操作 16 个 float 的向量寄存器，要自己处理对齐、收尾和掩码。写 CUDA 时，你写的是一个
          <strong>普通的标量线程</strong>：每个线程有自己的寄存器、自己的程序计数器（Volta 之后甚至每线程独立
          PC）、自己的分支走向，把这些线程「捆成 32 一组锁步执行」是硬件偷偷做的。这是 CUDA 易用性的来源 ——
          标量代码人人会写，向量 intrinsics 没几个人愿意碰。
        </p>
        <p>
          但抽象总有漏出来的地方。32 个线程共用一个取指单元，意味着当 warp 内部走向不同分支时，硬件只能
          <strong>把两条路径都执行一遍</strong>，靠活动掩码（active mask）让走错路的线程「陪跑」——
          执行 A 分支时屏蔽走 B 的线程，再执行 B 分支时屏蔽走 A 的线程。这就是{' '}
          <Term t="分支分化（warp divergence）">
            同一 warp 内线程走向不同分支时，硬件串行执行所有分支路径、用掩码关闭不相关线程的现象。
            分支两侧的耗时是相加而不是取最大。
          </Term>
          ：
        </p>
        <CodeBlock code={DIVERGE_CODE} lang="cuda" title="divergence.cu" highlight={[2, 4]} />
        <p>
          注意分化只发生在 <strong>warp 内部</strong>：如果整个 warp 的 32 个线程齐刷刷走同一条分支
          （比如按 <code>blockIdx</code> 分支，或按 <code>threadIdx.x / 32</code> 分支），没有任何惩罚。
          所以高性能 CUDA 代码的分支要么对齐到 warp 边界，要么干脆改写成无分支的算术形式。
        </p>
        <p>
          warp 的第二重身份更重要：它是<strong>延迟隐藏的筹码</strong>。每个分区的 warp 调度器手里同时
          「驻留」着多个 warp（H100 每 SM 最多 64 个，每分区 16 个），每个周期它扫一眼哪些 warp 的下一条指令
          已经就绪（operands ready），从中挑一个发射。某个 warp 在等显存？没关系，它安静地挂着，调度器转头发射
          别的 warp —— 用「换人干活」而不是「让一个人等得更短」来填满执行单元。这正是 CPU 与 GPU 的分水岭：
          CPU 用大缓存和乱序执行<strong>缩短</strong>延迟，GPU 用海量驻留 warp <strong>隐藏</strong>延迟。
          而 warp 切换之所以能做到零开销，就是因为第 1 节那个倒挂的寄存器堆：64 个 warp 的全部寄存器
          <strong>同时常驻</strong>在 256 KB 的寄存器堆里，切换不需要保存恢复任何状态，调度器只是换一个编号取指而已。
          对比之下，CPU 的线程上下文切换要进出内存搬运整套寄存器状态，开销是微秒级的。
        </p>
      </Section>

      <Quiz
        question="一个 warp 内的 32 个线程，有 13 个满足 if 条件、19 个走 else。硬件会怎么执行？"
        options={[
          {
            text: '把 warp 拆成两个小组，两条分支并行执行',
            explain: 'warp 是不可拆分的调度单位，一个周期只能发射一条指令，不存在「拆开并行」。',
          },
          {
            text: '两条分支路径都执行一遍，用活动掩码让不相关的线程陪跑',
            correct: true,
            explain:
              '正确。32 个线程共用取指单元，分化时硬件串行执行所有路径：跑 if 时掩蔽 19 个线程，跑 else 时掩蔽 13 个，总耗时约等于两条路径之和。',
          },
          {
            text: '只执行多数线程（19 个）选择的分支，少数线程的结果作废',
            explain: '每个线程的语义都必须正确，硬件不会丢弃任何线程的计算 —— 它靠掩码保证正确性，代价是时间。',
          },
          {
            text: '触发异常，CUDA 不允许 warp 内出现分支',
            explain: '分支完全合法，程序结果也正确，只是性能会因为两路串行执行而下降 —— 这是性能陷阱，不是错误。',
          },
        ]}
      />

      <Section
        index={4}
        title="LAB · Warp 延迟隐藏"
        lead="一个 SM 分区的时间轴：看 ALU 的空泡如何被一个个 warp 填满。"
      >
        <p>
          下面的模拟器还原一个分区的调度过程。每个 warp 反复做同一件事：发射几条计算指令，然后发射一条访存指令、
          进入上百个周期的等待。调度器每周期从就绪的 warp 里挑一个发射。先把 warp 数拖到 1，看看访存延迟
          把时间轴撕出多大的空洞；再逐渐加 warp、调整计算密度，观察 ALU 利用率怎么爬向 100%。
        </p>
        <WarpLatencyLab />
        <p>
          这个玩具模型背后有一条可以写在墙上的公式：要把延迟为 <code>L</code> 的访存完全藏住，期间就得有别的指令
          可发。每个 warp 每轮贡献约 <code>C+1</code> 条指令、随后沉默 <code>L</code> 个周期，所以利用率约为{' '}
          <code>W×(C+1) / (C+1+L)</code>，封顶 100%。它告诉你两条提高利用率的路：<strong>加 W</strong>
          （更多驻留 warp，即更高 occupancy）或者<strong>加 C</strong>（每次访存之间做更多计算，
          即更高的计算访存比）。第 5、6 章的优化故事，全部是围绕这两个旋钮展开的。
        </p>
        <Callout type="insight" title="GPU 不消灭延迟，只是把延迟藏起来">
          <p>
            HBM 一次访问几百个周期，这个物理事实 GPU 改变不了，它甚至不打算改变 —— 它的缓存比 CPU
            的还小。GPU 的赌注是：只要同时驻留的工作足够多，等待就永远可以被别人的计算盖住。
            这就是为什么 GPU 程序要启动比核心数多几十倍的线程 —— 多出来的线程不是浪费，
            它们是调度器手里用来填空泡的牌。牌不够，再多的核心也只能空转。
          </p>
        </Callout>
      </Section>

      <Quiz
        question="为什么一个 SM 上要同时驻留远多于执行单元数量的 warp（H100 上最多 64 个）？"
        options={[
          {
            text: '为了让更多线程并行执行，提高峰值算力',
            explain:
              '峰值算力由执行单元数量决定，驻留再多 warp 也不会提高峰值 —— 每个分区每周期仍然只能发射一条指令。',
          },
          {
            text: '为了在某些 warp 等待访存时，调度器有别的就绪 warp 可以发射，从而隐藏延迟',
            correct: true,
            explain:
              '正确。驻留 warp 是延迟隐藏的本钱：等待中的 warp 不占执行单元，调度器零开销切换到就绪 warp，让 ALU 始终有活干。',
          },
          {
            text: '为了减少分支分化的概率',
            explain: '分化发生在单个 warp 内部的 32 个线程之间，与驻留多少个 warp 无关。',
          },
          {
            text: '因为线程块必须整块驻留，这是编程模型的副作用',
            explain:
              '线程块确实整块驻留，但这是结果不是目的 —— 硬件特意把寄存器堆做到 256 KB，就是为了能驻留大量 warp 用于延迟隐藏。',
          },
        ]}
      />

      <Section
        index={5}
        title="存储层级：每一级都差一个数量级"
        lead="算力的食材是数据，而数据放在哪里，决定了取一次要烧掉多少周期。"
      >
        <p>
          延迟隐藏讲的是「等待时干别的」，但更好的策略永远是「少等」。GPU 的存储是一座金字塔，
          从 SM 内部到片外显存，每往下走一层，容量大一个数量级，延迟也贵一个数量级：
        </p>
        <MemTable />
        <p>
          这张表值得反复咀嚼。<strong>寄存器</strong>就在执行单元手边，读写几乎免费，但每个线程最多 255 个，
          用超了就会「溢出」到显存（第 5 章会撞上这个坑）。<strong>共享内存和 L1</strong> 其实是同一块片上
          SRAM 的两种用法：L1 由硬件自动管理，共享内存则是程序员显式控制的便笺纸 —— 同一个线程块内的线程
          用它交换数据，延迟约 30 个周期，比走显存便宜一个数量级以上。<strong>L2</strong> 是全片 132 个 SM
          共享的中转站，50 MB 听起来不小，但被几万个并发线程一摊就很紧张了。最底层的 <strong>HBM3</strong>{' '}
          延迟在 500 周期以上 —— 实测 Hopper 上甚至能到 600 多个周期 —— 一次往返的时间，足够一个分区发射几百条
          计算指令。
        </p>
        <p>
          亲手感受一下这个落差：下面每点一次「取数」，就有一个数据方块沿真实路径飞一个来回，周期计数器会诚实记账。
          连点几次 HBM，再连点几次寄存器，右侧的柱状图会把残酷的对比摆在你面前。
        </p>
        <MemoryLatencyLab />
        <p>
          顺带校准一个直觉：HBM 的<strong>带宽</strong>（3.35 TB/s）是 CPU 内存的几十倍，但<strong>延迟</strong>
          并不比 CPU 内存好，甚至更差。GPU 的显存系统是为吞吐设计的卡车车队，不是为延迟设计的跑车 ——
          所以「攒一批数据一次搬运」远胜「零碎地按需取用」，这个原则会贯穿第 4 章的合并访存和第 8 章的
          FlashAttention。
        </p>
      </Section>

      <Section
        index={6}
        title="Tensor Core 一瞥：矩阵乘的专用电路"
        lead="当 95% 的 FLOPs 都花在矩阵乘上，为它造专门的硬件就是顺理成章的事。"
      >
        <p>
          到目前为止我们聊的都是 FP32 CUDA 核心 —— 每个周期做一次「标量乘加」。但 H100 的规格表上还有一个
          夸张得多的数字：BF16 精度下 989 TFLOPS，是 FP32 CUDA 核心（67 TFLOPS）的约 15 倍。这个差距来自每个
          分区里那个紫色的块：<strong>Tensor Core</strong>。它不做标量运算，一条 <strong>MMA</strong>
          （Matrix Multiply-Accumulate，矩阵乘加）指令直接完成一小块矩阵的乘法并累加 —— 例如一组 warp
          协作完成 <code>16×8×16</code> 的矩阵片乘加，一条指令顶过去几百条标量指令。
        </p>
        <Figure caption="H100 SXM 各精度峰值吞吐。Tensor Core 对 CUDA 核心的优势约一个数量级，精度每降一档吞吐近乎翻倍。">
          <FlopsChart />
        </Figure>
        <p>
          为什么专用电路能快这么多？因为矩阵乘有极强的结构性：一块 16×16 的数据片内部有大量操作数复用，
          专用电路可以把取指、译码、操作数搬运的开销摊薄到几百次乘加上，而标量核心每做一次乘加都要付一遍这些
          「行政成本」。再加上深度学习对低精度的宽容 —— BF16 训练、FP8 推理已是常态 —— 精度每砍一半，
          同样的硅片面积和带宽就能再翻一倍吞吐。这就是大模型时代 Tensor Core 成为绝对主角的原因：
          Transformer 的前向反向，按 FLOPs 算 95% 以上是矩阵乘（第 7 章会算这笔账）。
        </p>
        <p>
          好消息是你几乎不需要直接编写 Tensor Core 指令：cuBLAS、cuDNN 这些 NVIDIA 官方库会自动选用它，
          PyTorch 的 <code>torch.matmul</code> 在半精度下默认走 Tensor Core。坏消息是它把对「喂数据」的要求
          抬高了一个数量级 —— 算得越快，越容易饿。一颗吃不饱的 Tensor Core 和没有 Tensor Core 没什么区别，
          这是第 6 章 Roofline 模型要讲的核心矛盾，也是第 8 章 FlashAttention 存在的全部理由。
        </p>
      </Section>

      <Section index={7} title="总结与延伸阅读">
        <p>这一章拆完了一块 H100，把这几条直觉带去下一章：</p>
        <ul>
          <li>
            GPU 是<strong>同一个 SM 的一百多份拷贝</strong>：H100 SXM 132 个、A100 108 个。SM 是计算、
            寄存器、共享内存的分配单位，也是你的线程块的驻留地。
          </li>
          <li>
            每个 SM = 4 个分区 × (1 warp 调度器 + 32 FP32 核 + 1 Tensor Core)，配 256 KB 寄存器堆和
            256 KB L1/共享内存（Shared 可配至 228 KB）。
          </li>
          <li>
            硬件调度的原子单位是 <strong>warp（32 线程锁步）</strong>。你写标量线程，硬件捆成 warp 执行 ——
            分支分化时两路都要走，靠掩码保证正确性。
          </li>
          <li>
            GPU 靠<strong>驻留大量 warp + 零成本切换</strong>隐藏访存延迟，而不是靠大缓存缩短延迟。
            寄存器堆比 L1 还大，就是为了让所有驻留 warp 的状态同时在板上。
          </li>
          <li>
            存储金字塔每层差一个数量级：寄存器 ≈1 周期、Shared/L1 ≈30、L2 ≈200、HBM ≈500+。
            数据放哪里，是性能优化的第一问题。
          </li>
          <li>
            Tensor Core 用一条 MMA 指令做整片矩阵乘加，BF16 吞吐约为 FP32 CUDA 核的 15 倍 ——
            大模型的算力主角，由 cuBLAS/cuDNN/PyTorch 自动调用。
          </li>
        </ul>
        <p>想钻得更深，这几份材料值得花时间：</p>
        <ul>
          <li>
            <a href="https://resources.nvidia.com/en-us-tensor-core" target="_blank" rel="noreferrer">
              NVIDIA H100 Tensor Core GPU Architecture Whitepaper
            </a>{' '}
            —— Hopper 架构官方白皮书，本章所有规格数字的出处，SM 结构图非常值得细看。
          </li>
          <li>
            <a
              href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA A100 Tensor Core GPU Architecture Whitepaper
            </a>{' '}
            —— Ampere 白皮书。对照两代规格读，能看出架构演进的脉络。
          </li>
          <li>
            <a
              href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA Hopper Architecture In-Depth
            </a>{' '}
            —— 官方博客版深度导览，比白皮书轻松，重点讲了 TMA 和线程块簇等新特性。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2402.13499" target="_blank" rel="noreferrer">
              Benchmarking and Dissecting the Nvidia Hopper GPU Architecture
            </a>{' '}
            —— 用微基准实测 Hopper 各级延迟与带宽的论文，本章延迟数字的实测参照。
          </li>
          <li>
            <a
              href="https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors"
              target="_blank"
              rel="noreferrer"
            >
              Programming Massively Parallel Processors（第 4 版）第 4 章
            </a>{' '}
            —— PMPP 教材中讲架构与调度的一章，把 SIMT 与延迟隐藏讲得最系统的教科书材料。
          </li>
        </ul>
      </Section>
    </>
  )
}
