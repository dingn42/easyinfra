import { Callout, CodeBlock, MathTex, Quiz, Section, Term } from '@/components/ui'
import { MemCounterLab } from './MemCounterLab'
import { OccupancyLab } from './OccupancyLab'
import { PerfLadder } from './PerfLadder'
import { TilingAnimLab } from './TilingAnimLab'

const NAIVE_KERNEL = `__global__ void matmul_naive(const float* A, const float* B,
                             float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];  // 每次迭代：读 2 个 float，算 2 个 FLOP
    C[row * N + col] = acc;
}`

const TILED_KERNEL = `#define T 32

__global__ void matmul_tiled(const float* A, const float* B,
                             float* C, int N) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;

    for (int k0 = 0; k0 < N; k0 += T) {
        // 全 block 1024 个线程协作：每人各搬 A、B 的一个元素进 shared
        As[threadIdx.y][threadIdx.x] = A[row * N + (k0 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
        __syncthreads();  // ① 等全员搬完，shared 里的瓦片才完整可用

        for (int k = 0; k < T; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();  // ② 等全员算完，才允许覆盖 shared 装下一段
    }
    C[row * N + col] = acc;
}`

export default function Chapter() {
  return (
    <>
      <p>
        照着第 3 章的套路写一个矩阵乘法 kernel —— 每个线程负责一个输出点 ——
        在 A100 上跑 4096×4096 的 FP32 矩阵乘，实测大约只有 cuBLAS 的 2%。同一块芯片、
        同样的浮点单元、同样的理论算力，差了 50 倍。这一章我们不换硬件、不改算法复杂度，
        只靠重写 kernel 把这 50 倍一点点追回来。需要的弹药前两章都已经备好：合并访存、shared
        memory、occupancy。而你会发现，所有优化最终都指向同一句话：<strong>少从 HBM 搬数据</strong>。
      </p>

      <Section index={1} title="基线：先记一笔访存账" lead="在优化任何 kernel 之前，先算清楚它到底搬了多少字节。">
        <p>
          基线长这样：C = A×B，三个矩阵都是 N×N。把 C 摊平成 N² 个点，每个线程认领一个，
          对 K 维做一次长度为 N 的内积：
        </p>
        <CodeBlock code={NAIVE_KERNEL} lang="cuda" title="matmul_naive.cu" highlight={[9]} />
        <p>
          这段代码在功能上无可挑剔，问题全在第 9 行那个循环的<strong>访存账</strong>上。
          每个输出点要扫一遍 A 的一行和 B 的一列 —— 从全局内存读 2N 个 float。N² 个输出点，
          总读取量就是 2N³ 个 float。但矩阵本身一共才多大？A、B、C 三个矩阵加起来 3N² 个 float：
        </p>
        <MathTex
          block
          tex="\underbrace{2N^3 \times 4\,\mathrm{B}}_{\text{naive 全局读取}} \quad\text{vs}\quad \underbrace{3N^2 \times 4\,\mathrm{B}}_{\text{数据总量}}"
        />
        <p>
          代入 N=4096：naive 要从 HBM 搬 2×4096³×4 B = <strong>512 GB</strong>，
          而三个矩阵总共只有 192 MB。也就是说，A 的每个元素被原封不动地重读了几千次 ——
          它本来就该被复用 N 次（A 的一行参与 N 个输出点的计算），但 naive 把每一次复用都变成了一次 HBM 往返。
          512 GB 除以 A100 的 1.9 TB/s 带宽，光搬数据就要约 290 ms；而这点计算量在 Tensor Core
          上不到 1 ms 就能算完。第 4 章说「访存为王」，矩阵乘法是最极端的例证：
          <strong>计算量是 O(N³)，数据量是 O(N²)，这多出来的一个 N 全是复用的机会</strong>。
          能把复用做到芯片上的 kernel 快如闪电，做不到的就只能排队等 HBM。
        </p>
        <p>
          顺带一提，naive 实测只有 2% 而不是「带宽算出来的 5%」，因为它的访存模式往往还没合并 ——
          相邻线程读 A 时踩出 N×4B 的大跨步（stride），一条 cache line 只用到 4 个字节。
          访存量大和访存效率低，两笔账叠在一起。
        </p>
      </Section>

      <Section
        index={2}
        title="用数字感受瓶颈"
        lead="拖一拖滑杆，看不同策略下访存量、复用倍数和瓶颈的变化。"
      >
        <p>
          下面这个计数器把刚才的账本做成了可以拨弄的仪器。tiled T=16 / T=32 是下一节要讲的分块策略
          （先剧透结论：全局读取量降为 2N³/T）；「理论下限」是 A、B 各读一遍、C 写一遍 ——
          任何 kernel 都不可能比它更省。重点观察：<strong>访存时间和计算时间谁压过谁</strong>，
          以及对数刻度下四个条形之间隔着多少个数量级。
        </p>
        <MemCounterLab />
        <p>
          有两个现象值得停下来想。第一，naive 与理论下限之间差了三个数量级 ——
          优化空间不是 20%、30% 这种抠细节的量级，而是「整个 kernel 的结构都不对」的量级。第二，
          即使 T=32，估算出的访存时间仍然比计算时间大一个量级：shared memory tiling
          只是把差距从 1000× 缩到 30×，要真正翻越 memory-bound 的山头，还得靠后面的寄存器分块。
        </p>
      </Section>

      <Section
        index={3}
        title="Tiling：把复用搬到芯片上"
        lead="C 切成 T×T 的瓦片，每个 block 认领一片，K 维分段、段内全员复用。"
      >
        <p>
          复用机会摆在那里，问题是去哪复用。寄存器是每线程私有的，装不下别人要的数据；HBM
          太远；答案是第 4 章的 <Term t="shared memory">块内共享的片上缓存，A100 上每 SM 最多 164KB，
          延迟约为全局内存的 1/10，带宽高一个数量级以上。</Term> —— 一个 block 内所有线程共享，
          延迟和带宽都接近 L1。于是有了 <Term t="tiling（分块）">把大矩阵切成小瓦片（tile），
          让每个瓦片在片上缓存里被充分复用后再丢弃的循环重组技术，CPU 上对应 cache blocking。</Term>：
        </p>
        <p>
          把 C 切成 T×T 的瓦片，每个 block 负责算其中一片。这一片需要 A 的 T 行和 B 的 T 列 ——
          还是太大，放不进 shared。所以再沿 K 维切成长度为 T 的段：每一段只涉及 A 的一个 T×T 子块和
          B 的一个 T×T 子块。循环就变成：<strong>搬两个子块进 shared → 块内 T×T 个线程全员用它算一段内积
          → 换下一段</strong>。每个从 HBM 搬进来的元素，进了 shared 之后会被同一行/列的 T 个线程各用一次：
        </p>
        <MathTex block tex="\text{全局读取}: 2N^3 \;\longrightarrow\; \frac{2N^3}{T}, \qquad \text{复用倍数} = T" />
        <CodeBlock code={TILED_KERNEL} lang="cuda" title="matmul_tiled.cu" highlight={[16, 20]} />
        <p>
          两行高亮的 <code>__syncthreads()</code> 缺一不可，而且拦的是两种方向相反的事故。
          <strong>第一次同步（第 16 行）拦「还没写完就读」</strong>：搬运是 1024 个线程每人一个元素拼出来的，
          warp 之间没有任何执行顺序保证；如果某个跑得快的线程直接开始内积，它读到的 As、Bs
          里可能还有上一段的旧值或未初始化的数据。<strong>第二次同步（第 20 行）拦「还没读完就覆盖」</strong>：
          下一轮循环的第一件事就是往同一块 shared 里写新子块，如果有线程还在用当前段做内积，
          数据就会在它脚下被换掉。删掉任何一个，错误都不会报错 —— 只会算出一堆悄悄错掉的数字，
          这是 CUDA 里最难查的一类 bug（race condition，竞态）。
        </p>
        <Callout type="insight" title="优化 GEMM 的全部秘密：提高每字节的计算量">
          <p>
            naive 每从 HBM 读 8 个字节只做 2 个 FLOP —— <Term t="算术强度（arithmetic intensity）">
            计算量与访存量之比，单位 FLOP/Byte，第 6 章 Roofline 模型的横轴。</Term>只有 0.25 FLOP/B。
            tiled T=32 把它抬到 16 FLOP/B。而 A100 的「收支平衡点」是 312 TFLOPS ÷ 1.9 TB/s ≈ 164 FLOP/B
            ——低于这个数，算力就在等数据。之后的每一级优化（寄存器 tiling、向量化、Tensor Core）
            本质上都在干同一件事：让每个搬进来的字节被更多计算摊薄。记住这个数字游戏，
            第 6 章的 Roofline 会把它变成一张图。
          </p>
        </Callout>
        <Quiz
          question="T=32 的 shared memory tiling，相比 naive 把全局内存读取量降低了多少倍？"
          options={[
            { text: '2 倍 —— A、B 各省一半', explain: '再想想：每个元素进 shared 之后被多少个线程复用？' },
            {
              text: '32 倍',
              correct: true,
              explain:
                '每个搬进 shared 的元素被同行/同列的 T=32 个线程各用一次，全局读取从 2N³ 降到 2N³/32。复用倍数恰好等于瓦片边长 T —— 这也是为什么大家总想把 T 做大，直到 shared 容量和寄存器预算卡住为止。',
            },
            { text: '1024 倍 —— T×T 个线程都在复用', explain: '复用发生在一行或一列方向上，是 T 倍而不是 T² 倍：每个元素被 32 个线程用，不是被 1024 个线程用。' },
            { text: '取决于 N，N 越大降得越多', explain: '降低的倍数只和 T 有关：2N³ ÷ (2N³/T) = T，与 N 无关。N 决定的是绝对量，不是比例。' },
          ]}
        />
      </Section>

      <Section index={4} title="看见 tiling" lead="一个 12×12 的玩具矩阵，把「装载 → 同步 → 复用」逐帧放给你看。">
        <p>
          文字描述的循环结构，不如直接看它跑。下面是 N=12、T=4 的微缩版：左下是 A，右上是 B，
          右下是 C 的九个瓦片。每一段你会看到 A 的子块（cyan）和 B 的子块（amber）飞进中间的 shared
          棋盘，然后当前 C 瓦片（volt 框）的 16 个格子逐个点亮 —— 每个格子点亮一次，
          代表它从 shared 里白嫖了一段内积。右侧两个计数器在记同一件事的两种代价：
          这些计算如果由 naive 来做要读多少次 HBM，由 tiled 来做又读了多少次。
        </p>
        <TilingAnimLab />
        <p>
          看完一整遍（或者直接拖到结尾），两个计数器的比值会停在 4.0× —— 正好是 T。
          真实 kernel 里 T=32，这个差距就是 32 倍。另外注意一个容易忽略的细节：装载阶段是「全员搬运」，
          16 个线程每人搬 A、B 各一个元素，而不是某个线程独自搬完 ——
          这让装载本身也是合并访存的，搬运的代价被整个 block 均摊。
        </p>
      </Section>

      <Section index={5} title="性能阶梯：从 2% 爬到 100%" lead="shared tiling 只是半山腰，上面还有四级台阶。">
        <p>
          把各级优化的实测性能排成一列，你会得到一条经典的「阶梯」。
          数字会因矩阵尺寸和 GPU 型号浮动，但形状是普适的：
        </p>
        <PerfLadder />
        <p>
          <strong>合并访存（2% → ~8%）</strong>：不改一行算法，只调整线程到行列的映射，
          让同一 warp 的 32 个线程在每次迭代里读连续地址。这是第 4 章的内容，也是性价比最高的一步 ——
          几行代码换 4 倍。它解决的是「访存效率」，还没动「访存总量」。
        </p>
        <p>
          <strong>shared memory tiling（~8% → ~40%）</strong>：本章的主角，访存总量除以 T。
          到这里 kernel 的结构已经和 cuBLAS 同源，但还有一个隐蔽的新瓶颈：每做 2 个 FLOP
          仍要从 shared 读 8 个字节。shared 虽快，带宽也是有限的 —— 瓶颈从 HBM 搬进了片上。
        </p>
        <p>
          <strong>寄存器 tiling（~40% → ~70%）</strong>：同样的复用逻辑再下沉一层。
          让每个线程不再只算 1 个输出点，而是算一个 4×4 的 micro-tile：从 shared 读 4 个 A 值、4
          个 B 值进寄存器，做 4×4=16 次乘加。每字节 shared 读取摊到的计算量翻了 4 倍，
          shared 带宽压力骤降。代价是每线程要养 16 个累加器外加装载缓冲，寄存器占用飙到 100 个上下
          —— 这正是下一节 occupancy 之争的伏笔。
        </p>
        <p>
          <strong>向量化 + 双缓冲（~70% → ~85%）</strong>：两个独立的技巧。向量化是把读写改成{' '}
          <code>float4</code>，一条 128-bit 指令顶四条 32-bit，指令数少了，访存单元的利用率高了。
          双缓冲（double buffering）则是在 shared 里开两份瓦片缓冲：计算第 k 段时，同步预取第 k+1
          段进另一份缓冲 —— 搬运和计算重叠起来，把原本「搬、停、算、停」的流水线填满，
          还能省掉一次同步等待。
        </p>
        <p>
          <strong>cuBLAS / Tensor Core（100%）</strong>：最后一段差距靠的不只是软件。Tensor Core 用{' '}
          <code>mma</code>（WMMA）指令让一个 warp 在硬件流水线里一口气完成 16×16×16
          的小矩阵乘，等效算力是 CUDA Core 的十几倍。再叠加 warp 级分块、bank conflict
          消除、指令调度微调 —— 这也是 CUTLASS 模板库做的事。手写 kernel 摸到 cuBLAS 的 90%
          是完全现实的目标，最后 10% 是汇编级的体力活。
        </p>
        <Callout type="deep" title="为什么每一级都在「再分一次块」？">
          <p>
            回头看这条阶梯：HBM → shared 是 block 级分块，shared → 寄存器是线程级分块，Tensor Core
            内部还有 warp 级分块。GPU 的存储层级有几层，GEMM 就要分几次块 ——
            每一层缓存都值得一个为它的容量和带宽量身定制的瓦片尺寸。CUTLASS
            干脆把这套「瓦片尺寸的层级」做成了 C++ 模板参数。
          </p>
        </Callout>
      </Section>

      <Section
        index={6}
        title="Occupancy：寄存器与并行度的交易"
        lead="寄存器 tiling 不是免费的 —— 它吃掉的是 SM 的驻留能力。"
      >
        <p>
          第 2 章讲过，GPU 靠「驻留大量 warp、随时切换」来隐藏延迟，
          <Term t="occupancy（占用率）">SM 上实际驻留的 warp 数与硬件上限（A100 为 64 个 warp）之比，
          衡量延迟隐藏的「弹药储备」。</Term>就是这个能力的量化。一个 SM 的资源是死的：A100 上是 65536
          个寄存器、164 KB shared memory、2048 个线程、32 个 block 的驻留上限。
          每个 block 按自己的需求从这四个池子里扣资源，<strong>哪个池子先耗尽，哪个就决定了驻留的 block 数</strong>
          —— 四个约束取最小值。寄存器 tiling 让每线程寄存器从 30 个涨到 100+，
          直接把寄存器池变成最先见底的那个。拖一拖下面的滑杆，看看瓶颈怎么换位：
        </p>
        <OccupancyLab />
        <p>
          那寄存器 tiling 岂不是搬起石头砸自己的脚？并没有 —— 这正是本节最反直觉的一课：
          <strong>occupancy 是手段，不是目的</strong>。它存在的意义是保证「访存等待时永远有别的 warp 可切」，
          只要驻留的 warp 足以盖住延迟，再高就是浪费。而寄存器 tiling 让每个 warp 自带 16
          倍的独立计算（指令级并行），对「切换隐藏」的依赖本身就低了。高性能 GEMM 的典型画像是
          occupancy 只有 25%~50%，却把算力跑到 90% 以上；反过来，如果某线程的变量多到寄存器装不下，
          编译器会把溢出的部分挪到 local memory —— 名字叫 local，物理上在显存里，
          一次溢出读写就是几百个周期，比低 occupancy 致命得多。调优时盯着 nvcc 的{' '}
          <code>-maxrregcount</code> 和 profiler 里的 register spilling 指标，比盯着 occupancy 数字有用。
        </p>
        <Quiz
          question="把 kernel 改成每线程算 4×4 micro-tile 后，每线程寄存器用量从 32 涨到 120。最可能的直接后果是？"
          options={[
            { text: '编译失败，CUDA 不允许单线程用这么多寄存器', explain: '编译没问题：单线程寄存器上限是 255 个，120 远没到顶。问题出在「合在一起够不够分」。' },
            {
              text: 'SM 上能同时驻留的 warp 变少（occupancy 下降）；若超过预算还会溢出到 local memory',
              correct: true,
              explain:
                '寄存器池是整个 SM 共享的：65536 ÷ (120×256) ≈ 2 个 block，occupancy 掉到 25%。这通常是笔划算的交易 —— 每线程干的活翻了 16 倍。但如果编译器实在塞不下，变量会溢出（spill）到 local memory，那是显存速度，性能立刻跳水。',
            },
            { text: '硬件自动把多出来的变量放进 shared memory，性能不受影响', explain: '没有这种自动机制。溢出目标是 local memory（物理上在显存），而且溢出由编译器在编译期决定，不是运行时调度。' },
            { text: '没有任何影响，每个线程的寄存器是独立无限的', explain: '恰恰相反：寄存器是 SM 上最紧俏的共享资源之一，65536 个要分给所有驻留线程。每线程用得越多，驻留得越少。' },
          ]}
        />
      </Section>

      <Section index={7} title="总结与延伸阅读">
        <p>这一章把一个 50 倍的性能差距拆成了五级台阶，每一级都对应一个明确的硬件事实：</p>
        <ul>
          <li>
            <strong>先记账，再动手</strong>：GEMM 计算 O(N³)、数据 O(N²)，naive 在 N=4096 时要搬 512 GB
            —— 复用率是矩阵乘法优化的唯一主线。
          </li>
          <li>
            <strong>tiling 的收益等于瓦片边长</strong>：T×T 分块 + shared memory 让全局读取降为
            2N³/T；两次 <code>__syncthreads()</code> 分别拦住「没写完就读」和「没读完就覆盖」。
          </li>
          <li>
            <strong>每级存储层级分一次块</strong>：HBM→shared 是 block 级瓦片，shared→寄存器是 4×4
            micro-tile，Tensor Core 里还有 warp 级瓦片 —— 全都是同一个复用思想。
          </li>
          <li>
            <strong>occupancy 是手段不是 KPI</strong>：四个资源约束取最小值；高性能 GEMM
            常故意用低 occupancy 换寄存器，真正的红线是 register spilling。
          </li>
        </ul>
        <p>想继续往深处走，这几份材料按顺序读：</p>
        <ul>
          <li>
            <a href="https://siboehm.com/articles/22/CUDA-MMM" target="_blank" rel="noreferrer" className="text-cyan">
              Simon Boehm — How to Optimize a CUDA Matmul Kernel
            </a>
            ：<strong>必读</strong>。从 naive 到 cuBLAS 94% 的十个 kernel，每一步都有代码、profiler
            数据和失败尝试，本章的性能阶梯就以它为蓝本。
          </li>
          <li>
            <a href="https://github.com/NVIDIA/cutlass" target="_blank" rel="noreferrer" className="text-cyan">
              NVIDIA CUTLASS（GitHub）
            </a>
            ：生产级的分块 GEMM 模板库，把「层级化 tiling」做成了可组合的模板参数，cuBLAS 的开源近亲。
          </li>
          <li>
            <a
              href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md"
              target="_blank"
              rel="noreferrer"
              className="text-cyan"
            >
              CUTLASS 文档 — Efficient GEMM in CUDA
            </a>
            ：一页讲清 threadblock tile / warp tile / thread tile 三层分块与双缓冲流水线的官方图解。
          </li>
          <li>
            <a href="https://docs.nvidia.com/cuda/cublas/" target="_blank" rel="noreferrer" className="text-cyan">
              cuBLAS Library Documentation
            </a>
            ：实际工程里你大概率调用而不是手写 GEMM —— 了解它的 API 约定（列主序！）和算法选择接口。
          </li>
          <li>
            <a
              href="https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0"
              target="_blank"
              rel="noreferrer"
              className="text-cyan"
            >
              PMPP（Programming Massively Parallel Processors）第 5 章
            </a>
            ：tiling 的教科书级推导，包含边界处理、任意尺寸矩阵等本章略过的工程细节。
          </li>
        </ul>
        <p>
          下一章我们把「访存时间 vs 计算时间」这笔反复出现的账正式画成一张图 —— Roofline 模型，
          让任何 kernel 的瓶颈一眼可辨。
        </p>
      </Section>
    </>
  )
}
