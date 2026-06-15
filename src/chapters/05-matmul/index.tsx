import { Callout, ChapterLink, CodeBlock, HardwareBaseline, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { MemCounterLab } from './MemCounterLab'
import { OccupancyLab } from './OccupancyLab'
import { PerfLadder } from './PerfLadder'
import { TilingAnimLab } from './TilingAnimLab'

const NAIVE_KERNEL_EN = `__global__ void matmul_naive(const float* A, const float* B,
                             float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];  // per iter: 2 floats read, 2 FLOPs done
    C[row * N + col] = acc;
}`

const NAIVE_KERNEL_ZH = `__global__ void matmul_naive(const float* A, const float* B,
                             float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];  // 每次迭代：读 2 个 float，算 2 个 FLOP
    C[row * N + col] = acc;
}`

const TILED_KERNEL_EN = `#define T 32

__global__ void matmul_tiled(const float* A, const float* B,
                             float* C, int N) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;

    for (int k0 = 0; k0 < N; k0 += T) {
        // all 1024 threads cooperate: each fetches one A and one B element into shared
        As[threadIdx.y][threadIdx.x] = A[row * N + (k0 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
        __syncthreads();  // (1) wait until every thread has loaded — only then is the tile complete

        for (int k = 0; k < T; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();  // (2) wait until every thread is done before overwriting shared with the next slice
    }
    C[row * N + col] = acc;
}`

const TILED_KERNEL_ZH = `#define T 32

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
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            Write a matmul kernel the way <ChapterLink n={3} /> taught you, one thread per output element, and run a
            4096×4096 FP32 multiply on an A100. You will measure roughly 2% of cuBLAS. Identical silicon, identical
            floating-point units, identical theoretical throughput, and a 50× gap. This chapter closes that gap
            without touching the hardware or changing the algorithmic complexity, purely by rewriting the kernel.
            The last two chapters already handed you the ammunition: coalesced access, shared memory, occupancy.
            What you will notice is that every optimization traces back to a single sentence: <strong>move less data
            out of HBM</strong>.
          </>,
          <>
            照着<ChapterLink n={3} />的套路写一个矩阵乘法 kernel，每个线程负责一个输出点，
            在 A100 上跑 4096×4096 的 FP32 矩阵乘，实测大约只有 cuBLAS 的 2%。同一块芯片，
            同样的浮点单元，同样的理论算力，却差了 50 倍。这一章不换硬件，也不改算法复杂度，
            只靠重写 kernel 就能把这 50 倍一点点追回来。需要的弹药前两章已经备齐：合并访存、shared
            memory、occupancy。你会发现，所有优化最后都指向同一句话：<strong>少从 HBM 搬数据</strong>。
          </>,
        )}
      </p>

      <HardwareBaseline ids={['a100']} />

      <Section
        index={1}
        title={t('The baseline: start by tallying the memory bill', '基线：先记一笔访存账')}
        lead={t(
          'Before optimizing any kernel, work out exactly how many bytes it actually moves.',
          '在优化任何 kernel 之前，先算清楚它到底搬了多少字节。',
        )}
      >
        <p>
          {t(
            'The baseline looks like this: C = A×B, all three matrices N×N. Flatten C into N² elements, give each thread one of them, and let it run a length-N dot product along the K dimension:',
            '基线长这样：C = A×B，三个矩阵都是 N×N。把 C 摊平成 N² 个点，每个线程认领一个，对 K 维做一次长度为 N 的内积：',
          )}
        </p>
        <CodeBlock
          code={t(NAIVE_KERNEL_EN, NAIVE_KERNEL_ZH)}
          lang="cuda"
          title="matmul_naive.cu"
          highlight={[9]}
        />
        <p>
          {t(
            <>
              The code is functionally flawless. The entire problem lives in the <strong>memory bill</strong> of the loop on line 9.
              Every output element sweeps a full row of A and a full column of B, which is 2N floats read from global memory. With N²
              output elements, total reads come to 2N³ floats. But how big are the matrices themselves? A, B, and C together are
              only 3N² floats:
            </>,
            <>
              这段代码在功能上无可挑剔，问题全在第 9 行那个循环的<strong>访存账</strong>上。
              每个输出点要扫一遍 A 的一行和 B 的一列，从全局内存读 2N 个 float。N² 个输出点，
              总读取量就是 2N³ 个 float。但矩阵本身一共才多大？A、B、C 三个加起来才 3N² 个 float：
            </>,
          )}
        </p>
        <MathTex
          block
          tex={t(
            '\\underbrace{2N^3 \\times 4\\,\\mathrm{B}}_{\\text{naive global reads}} \\quad\\text{vs}\\quad \\underbrace{3N^2 \\times 4\\,\\mathrm{B}}_{\\text{total data}}',
            '\\underbrace{2N^3 \\times 4\\,\\mathrm{B}}_{\\text{naive 全局读取}} \\quad\\text{vs}\\quad \\underbrace{3N^2 \\times 4\\,\\mathrm{B}}_{\\text{数据总量}}',
          )}
        />
        <p>
          {t(
            <>
              Plug in N=4096: naive moves 2×4096³×4 B = <strong>512 GB</strong> out of HBM, while the three matrices total just 192 MB.
              Every element of A gets re-read verbatim thousands of times. It was supposed to be reused N times (one row of
              A feeds N output elements), but naive turns each of those reuses into a fresh HBM round trip. Divide 512 GB by the A100's
              1.9 TB/s bandwidth and the data movement alone costs about 290 ms; the arithmetic itself finishes in under 1 ms on the Tensor
              Cores. <ChapterLink n={4} /> said "memory is king," and matmul is where that hits hardest:{' '}
              <strong>the compute is O(N³), the data is O(N²), and that extra factor of N is pure reuse opportunity</strong>. Kernels that land
              that reuse on-chip fly; the ones that can't just sit queued behind HBM.
            </>,
            <>
              代入 N=4096：naive 要从 HBM 搬 2×4096³×4 B = <strong>512 GB</strong>，
              而三个矩阵总共只有 192 MB。A 的每个元素都被原封不动地重读了几千次。
              它本该被复用 N 次（A 的一行参与 N 个输出点的计算），可 naive 把每一次复用都变成了一次 HBM 往返。
              512 GB 除以 A100 的 1.9 TB/s 带宽，光搬数据就要约 290 ms；而这点计算量在 Tensor Core
              上不到 1 ms 就算完了。<ChapterLink n={4} />说「访存为王」，矩阵乘法正是这句话体现得最明显的地方：
              <strong>计算量是 O(N³)，数据量是 O(N²)，这多出来的一个 N 全是复用的机会</strong>。
              能把复用做到芯片上的 kernel 跑得飞快，做不到的就只能一直排队等 HBM。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              One aside: naive measures 2%, not the "5% the bandwidth math predicts," because its access pattern usually isn't coalesced
              either. Neighboring threads reading A stride by N×4B, so a single cache line yields only 4 useful bytes. High traffic and low
              access efficiency are two separate bills, and you pay both.
            </>,
            <>
              顺带一提，naive 实测只有 2%，而不是「带宽算出来的 5%」，因为它的访存模式往往还没合并：
              相邻线程读 A 时踩出 N×4B 的大跨步（stride），一条 cache line 只用上 4 个字节。
              访存量大和访存效率低是两笔账，你两笔都得付。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={2}
        title={t('Feel the bottleneck in numbers', '用数字感受瓶颈')}
        lead={t(
          'Drag the sliders to watch how traffic, reuse factor, and the binding bottleneck shift across strategies.',
          '拖一拖滑杆，看不同策略下访存量、复用倍数和瓶颈的变化。',
        )}
      >
        <p>
          {t(
            <>
              The counter below turns that bill into an instrument you can dial. <code>tiled T=16</code> / <code>T=32</code> are the
              blocking strategy of the next section (spoiler: global reads drop to 2N³/T); the "theoretical floor" reads A and B once each
              and writes C once, and no kernel can possibly do better. Two things are worth watching: <strong>whether memory time or
              compute time dominates</strong>, and how many orders of magnitude separate the four bars on the log scale.
            </>,
            <>
              下面这个计数器把刚才的账本做成了一台可以拨弄的仪器。tiled T=16 / T=32 是下一节要讲的分块策略
              （先剧透结论：全局读取量降为 2N³/T）；「理论下限」是 A、B 各读一遍、C 写一遍，
              任何 kernel 都不可能比它更省。有两点值得盯着看：<strong>访存时间和计算时间谁压过谁</strong>，
              以及对数刻度下四个条形之间隔着多少个数量级。
            </>,
          )}
        </p>
        <MemCounterLab />
        <p>
          {t(
            <>
              Two observations are worth pausing on. naive and the theoretical floor are three orders of magnitude apart, which is not
              the 20%-or-30% world of shaving details; it's the "the whole kernel structure is wrong" world. And even at T=32 the estimated
              memory time still beats compute time by an order of magnitude. Shared-memory tiling only shrinks the gap from 1000× to 30×.
              Actually cresting the memory-bound ridge takes the register tiling that comes later.
            </>,
            <>
              有两个现象值得停下来想。一是 naive 与理论下限之间差了三个数量级，
              这不是 20%、30% 这种抠细节的量级，而是「整个 kernel 的结构都错了」的量级。二是
              即使 T=32，估算出的访存时间仍比计算时间大一个量级。shared memory tiling
              只把差距从 1000× 缩到 30×，要真正翻越 memory-bound 那道山脊，还得靠后面的寄存器分块。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={3}
        title={t('Tiling: bringing reuse on-chip', 'Tiling：把复用搬到芯片上')}
        lead={t(
          'Cut C into T×T tiles, give each block one tile, slice the K dimension into segments, and reuse each segment across the whole block.',
          'C 切成 T×T 的瓦片，每个 block 认领一片，K 维分段、段内全员复用。',
        )}
      >
        <p>
          {t(
            <>
              The reuse opportunity is sitting right there; the question is <em>where</em> to reuse it. Registers are private to each thread and
              can't hold data that other threads need, and HBM is too far away. The answer is <ChapterLink n={4} />'s{' '}
              <Term t="shared memory">An on-chip cache shared within a block: up to 164 KB per SM on the A100, roughly 1/10 the latency of
              global memory and more than an order of magnitude more bandwidth.</Term>, shared by every thread in a block, with latency and
              bandwidth close to L1. That hands us{' '}
              <Term t="tiling">A loop-restructuring technique that cuts a large matrix into small tiles, fully reuses each tile in on-chip
              cache, then discards it. The CPU equivalent is cache blocking.</Term>:
            </>,
            <>
              复用机会就摆在那里，问题是去哪复用。寄存器是每个线程私有的，装不下别人要用的数据；HBM
              又太远。答案是<ChapterLink n={4} />的 <Term t="shared memory">块内共享的片上缓存，A100 上每 SM 最多 164KB，
              延迟约为全局内存的 1/10，带宽高出一个数量级以上。</Term>，一个 block 内所有线程共享，
              延迟和带宽都接近 L1。于是就有了 <Term t="tiling（分块）">把大矩阵切成小瓦片（tile），
              让每个瓦片在片上缓存里被充分复用后再丢弃的循环重组技术，CPU 上对应 cache blocking。</Term>：
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Cut C into T×T tiles and make each block responsible for one of them. That tile needs T rows of A and T columns of B, still too
              large to fit in shared. So slice it again along K into segments of length T: each segment touches just one T×T sub-block of A and
              one T×T sub-block of B. The loop becomes <strong>load two sub-blocks into shared → the block's T×T threads all use them to compute
              one segment of the dot product → swap in the next segment</strong>. Every element pulled from HBM, once it lands in shared, gets used
              once by each of the T threads in its row or column:
            </>,
            <>
              把 C 切成 T×T 的瓦片，每个 block 负责算其中一片。这一片需要 A 的 T 行和 B 的 T 列，
              还是太大，放不进 shared，所以再沿 K 维切成长度为 T 的段：每一段只涉及 A 的一个 T×T 子块和
              B 的一个 T×T 子块。循环就变成 <strong>搬两个子块进 shared → 块内 T×T 个线程全员用它算一段内积
              → 换下一段</strong>。每个从 HBM 搬进来的元素，一旦进了 shared，就会被同一行或同一列的 T 个线程各用一次：
            </>,
          )}
        </p>
        <MathTex
          block
          tex={t(
            '\\text{global reads}: 2N^3 \\;\\longrightarrow\\; \\frac{2N^3}{T}, \\qquad \\text{reuse factor} = T',
            '\\text{全局读取}: 2N^3 \\;\\longrightarrow\\; \\frac{2N^3}{T}, \\qquad \\text{复用倍数} = T',
          )}
        />
        <CodeBlock
          code={t(TILED_KERNEL_EN, TILED_KERNEL_ZH)}
          lang="cuda"
          title="matmul_tiled.cu"
          highlight={[16, 20]}
        />
        <p>
          {t(
            <>
              The two highlighted <code>__syncthreads()</code> are both essential, and they guard against accidents in opposite directions.
              <strong> The first barrier (line 16) stops "reading before the write is finished."</strong> The load is stitched together by 1024
              threads each contributing one element, with no execution-order guarantee between warps; if a fast thread jumps straight into the dot
              product, the As/Bs it reads may still hold stale values from the previous segment or uninitialized data.{' '}
              <strong>The second barrier (line 20) stops "overwriting before the read is finished."</strong> The very first thing the next loop
              iteration does is write new sub-blocks into the same shared buffer, so if some thread is still using the current segment, the data
              gets swapped out from under its feet. Delete either one and nothing throws an error. You just get a pile of quietly wrong numbers,
              one of the hardest classes of bug in CUDA (a race condition).
            </>,
            <>
              两行高亮的 <code>__syncthreads()</code> 缺一不可，拦的是两种方向相反的事故。
              <strong>第一次同步（第 16 行）拦「还没写完就读」。</strong>搬运是 1024 个线程每人一个元素拼出来的，
              warp 之间没有任何执行顺序保证；要是某个跑得快的线程直接开始内积，它读到的 As、Bs
              里可能还留着上一段的旧值或未初始化的数据。<strong>第二次同步（第 20 行）拦「还没读完就覆盖」。</strong>
              下一轮循环干的第一件事就是往同一块 shared 里写新子块，如果有线程还在用当前段做内积，
              数据就会在它脚下被换掉。删掉任意一个都不会报错，只会算出一堆悄悄错掉的数字，
              这是 CUDA 里最难查的一类 bug（race condition，竞态）。
            </>,
          )}
        </p>
        <Callout
          type="insight"
          title={t('The whole secret of optimizing GEMM: raise the compute per byte', '优化 GEMM 的全部秘密：提高每字节的计算量')}
        >
          <p>
            {t(
              <>
                Naive does only 2 FLOPs for every 8 bytes read from HBM, an{' '}
                <Term t="arithmetic intensity">The ratio of compute to memory traffic, measured in FLOP/Byte; it's the horizontal axis of the
                Roofline model in Chapter 6.</Term> of just 0.25 FLOP/B. Tiled T=32 lifts it to 16 FLOP/B. The A100's break-even point is
                312 TFLOPS ÷ 1.9 TB/s ≈ 164 FLOP/B; below that number, the math units sit there waiting on data. Every later optimization (register
                tiling, vectorization, Tensor Cores) is doing the same one thing: amortizing each fetched byte across more compute.
                Keep this FLOP/B arithmetic in your head. <ChapterLink n={6} />'s Roofline turns it into a picture.
              </>,
              <>
                naive 每从 HBM 读 8 个字节只做 2 个 FLOP，<Term t="算术强度（arithmetic intensity）">
                计算量与访存量之比，单位 FLOP/Byte，第 6 章 Roofline 模型的横轴。</Term>只有 0.25 FLOP/B；
                tiled T=32 把它抬到 16 FLOP/B。而 A100 的收支平衡点是 312 TFLOPS ÷ 1.9 TB/s ≈ 164 FLOP/B，
                低于这个数，算力就只能干等数据。之后的每一级优化（寄存器 tiling、向量化、Tensor Core）
                干的都是同一件事：让每个搬进来的字节被更多计算摊薄。把这笔 FLOP/B 账记在脑子里，
                <ChapterLink n={6} />的 Roofline 会把它画成一张图。
              </>,
            )}
          </p>
        </Callout>
        <Quiz
          question={t(
            'Compared with naive, by what factor does shared-memory tiling at T=32 cut global-memory reads?',
            'T=32 的 shared memory tiling，相比 naive 把全局内存读取量降低了多少倍？',
          )}
          options={[
            {
              text: t('2× — A and B each save half', '2 倍 —— A、B 各省一半'),
              explain: t(
                'Think again: once an element is in shared, how many threads reuse it?',
                '再想想：每个元素进 shared 之后被多少个线程复用？',
              ),
            },
            {
              text: t('32×', '32 倍'),
              correct: true,
              explain: t(
                'Each element loaded into shared is used once by each of the T=32 threads in its row or column, so global reads drop from 2N³ to 2N³/32. The reuse factor equals the tile edge length T exactly, which is why everyone wants T larger, right up until shared capacity and the register budget say no.',
                '每个搬进 shared 的元素，会被同行或同列的 T=32 个线程各用一次，全局读取从 2N³ 降到 2N³/32。复用倍数恰好等于瓦片边长 T，这也是大家总想把 T 做大的原因，直到 shared 容量和寄存器预算把它卡住为止。',
              ),
            },
            {
              text: t('1024× — all T×T threads reuse it', '1024 倍 —— T×T 个线程都在复用'),
              explain: t(
                'Reuse happens along a single row or column, so it is T×, not T²: each element is used by 32 threads, not by 1024.',
                '复用发生在一行或一列方向上，是 T 倍而不是 T² 倍：每个元素被 32 个线程用，不是被 1024 个线程用。',
              ),
            },
            {
              text: t('It depends on N — bigger N cuts more', '取决于 N，N 越大降得越多'),
              explain: t(
                'The reduction factor depends only on T: 2N³ ÷ (2N³/T) = T, independent of N. N sets the absolute volume, not the ratio.',
                '降低的倍数只和 T 有关：2N³ ÷ (2N³/T) = T，与 N 无关。N 决定的是绝对量，不是比例。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={4}
        title={t('Seeing tiling', '看见 tiling')}
        lead={t(
          'A 12×12 toy matrix that plays "load → sync → reuse" back to you frame by frame.',
          '一个 12×12 的玩具矩阵，把「装载 → 同步 → 复用」逐帧放给你看。',
        )}
      >
        <p>
          {t(
            <>
              A loop structure described in prose is no match for watching it run. Below is a miniature with N=12, T=4: A is bottom-left, B is
              top-right, and C's nine tiles are bottom-right. In each segment you'll see a sub-block of A (cyan) and a sub-block of B (amber) fly
              into the shared board in the middle, then the 16 cells of the current C tile (volt outline) light up one by one. Each cell lighting
              once means it grabbed one segment of the dot product for free out of shared. The two counters on the right track the same work at two
              different prices: how many HBM reads naive would need to do this compute, versus how many tiled actually did.
            </>,
            <>
              文字描述的循环结构，不如直接看它跑。下面是 N=12、T=4 的微缩版：左下是 A，右上是 B，
              右下是 C 的九个瓦片。每一段你会看到 A 的子块（cyan）和 B 的子块（amber）飞进中间的 shared
              棋盘，然后当前 C 瓦片（volt 框）的 16 个格子逐个点亮。每个格子点亮一次，
              就代表它从 shared 里白嫖了一段内积。右侧两个计数器记的是同一件事的两种代价：
              这些计算如果交给 naive 要读多少次 HBM，交给 tiled 又只读了多少次。
            </>,
          )}
        </p>
        <TilingAnimLab />
        <p>
          {t(
            <>
              Watch it through once (or just drag to the end) and the ratio of the two counters settles at 4.0×, exactly T. In a real kernel
              T=32, so the gap is 32×. One easily-missed detail: the load phase is a team effort, with 16 threads each fetching one element
              of A and one of B, rather than a single thread fetching the whole tile. That keeps the load itself coalesced, and spreads its cost
              across the entire block.
            </>,
            <>
              看完一整遍（或者直接拖到结尾），两个计数器的比值会停在 4.0×，正好是 T。
              真实 kernel 里 T=32，这个差距就是 32 倍。还有一个容易忽略的细节：装载阶段是全员搬运，
              16 个线程每人搬 A、B 各一个元素，而不是某个线程独自把整片搬完。
              这让装载本身也是合并访存的，搬运的代价由整个 block 均摊。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={5}
        title={t('The performance ladder: climbing from 2% to 100%', '性能阶梯：从 2% 爬到 100%')}
        lead={t(
          'Shared tiling is only halfway up the hill; four more steps wait above it.',
          'shared tiling 只是半山腰，上面还有四级台阶。',
        )}
      >
        <p>
          {t(
            <>
              Line up the measured performance of each optimization level and you get a classic "ladder." The exact numbers float with matrix
              size and GPU model, but the shape is universal:
            </>,
            <>
              把各级优化的实测性能排成一列，你会得到一条经典的「阶梯」。
              数字会因矩阵尺寸和 GPU 型号浮动，但形状是普适的：
            </>,
          )}
        </p>
        <PerfLadder />
        <p>
          {t(
            <>
              <strong>Coalesced access (2% → ~8%)</strong>: not a single line of the algorithm changes. You only adjust how threads map to rows
              and columns, so the 32 threads of a warp read contiguous addresses each iteration. This is <ChapterLink n={4} />'s material, and it's the best
              bang for the buck: a 4× win for a few lines of code. It fixes access efficiency without touching total traffic yet.
            </>,
            <>
              <strong>合并访存（2% → ~8%）</strong>：算法一行不改，只调整线程到行列的映射，
              让同一 warp 的 32 个线程每次迭代都读连续地址。这是<ChapterLink n={4} />的内容，也是性价比最高的一步：
              几行代码换来 4 倍。它解决的是访存效率，还没动访存总量。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>Shared-memory tiling (~8% → ~40%)</strong>: the star of this chapter, total traffic divided by T. By now the kernel
              structure shares its DNA with cuBLAS, but a subtle new bottleneck appears: it still reads 8 bytes from shared for every 2 FLOPs.
              Shared is fast, but its bandwidth is finite too. The bottleneck has simply moved from HBM onto the chip.
            </>,
            <>
              <strong>shared memory tiling（~8% → ~40%）</strong>：本章的主角，访存总量除以 T。
              到这一步 kernel 的结构已经和 cuBLAS 同源，但又冒出一个隐蔽的新瓶颈：每做 2 个 FLOP
              仍要从 shared 读 8 个字节。shared 虽快，带宽同样有限。瓶颈只是从 HBM 搬进了片上。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>Register tiling (~40% → ~70%)</strong>: the same reuse logic, pushed down one more level. Instead of computing 1 output
              element, each thread now computes a 4×4 micro-tile: read 4 A values and 4 B values from shared into registers and do 4×4=16
              multiply-adds. The compute amortized over each byte of shared read jumps 4×, and the pressure on shared bandwidth plummets. The price
              is that each thread must keep 16 accumulators plus load buffers, pushing register usage to around 100. That sets up the
              occupancy fight in the next section.
            </>,
            <>
              <strong>寄存器 tiling（~40% → ~70%）</strong>：同样的复用逻辑再下沉一层。
              让每个线程不再只算 1 个输出点，改算一个 4×4 的 micro-tile：从 shared 读 4 个 A 值、4
              个 B 值进寄存器，做 4×4=16 次乘加。每字节 shared 读取摊到的计算量翻了 4 倍，
              shared 带宽压力骤降。代价是每个线程要养 16 个累加器外加装载缓冲，寄存器占用飙到 100 个上下。
              这正是下一节 occupancy 之争的伏笔。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>Vectorization + double buffering (~70% → ~85%)</strong>: two independent tricks. Vectorization rewrites loads and stores as{' '}
              <code>float4</code>, so one 128-bit instruction does the work of four 32-bit ones, cutting instruction count and raising memory-unit
              utilization. Double buffering keeps two tile buffers in shared: while computing segment k, it prefetches segment k+1 into the other buffer.
              Load and compute now overlap, the "load, stall, compute, stall" pipeline fills up, and you save a sync wait on top.
            </>,
            <>
              <strong>向量化 + 双缓冲（~70% → ~85%）</strong>：两个互不相干的技巧。向量化是把读写改成{' '}
              <code>float4</code>，一条 128-bit 指令顶四条 32-bit，指令数少了，访存单元的利用率上去了。
              双缓冲（double buffering）则是在 shared 里开两份瓦片缓冲：计算第 k 段时，顺手预取第 k+1
              段进另一份缓冲。搬运和计算就此重叠起来，原本「搬、停、算、停」的流水线被填满，
              还顺带省掉一次同步等待。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>cuBLAS / Tensor Core (100%)</strong>: the final stretch isn't won by software alone. Tensor Cores use the <code>mma</code>{' '}
              (WMMA) instruction to let one warp finish a 16×16×16 small matrix multiply in a single hardware pipeline pass, with effective
              throughput more than ten times that of CUDA Cores. On top of that go warp-level tiling, bank-conflict elimination, and instruction-scheduling
              tuning, which is exactly what the CUTLASS template library handles. Reaching 90% of cuBLAS with a hand-written kernel is a perfectly
              realistic goal; the last 10% is assembly-level grunt work. (Newer Blackwell-class cards push the low-precision side of this even
              harder, but the tiling story stays the same; the A100 numbers above are the ones to reason from.)
            </>,
            <>
              <strong>cuBLAS / Tensor Core（100%）</strong>：最后这一段差距靠的不只是软件。Tensor Core 用{' '}
              <code>mma</code>（WMMA）指令让一个 warp 在硬件流水线里一口气算完 16×16×16
              的小矩阵乘，等效算力是 CUDA Core 的十几倍。再往上叠 warp 级分块、bank conflict
              消除、指令调度微调，这些正是 CUTLASS 模板库在做的事。手写 kernel 摸到 cuBLAS 的 90%
              是完全现实的目标，最后 10% 是汇编级的体力活。（更新的 Blackwell 系列卡把低精度这一侧推得更狠，
              但 tiling 的逻辑不变；上面那组 A100 数字才是我们用来推算的基准。）
            </>,
          )}
        </p>
        <Callout type="deep" title={t('Why is every level "blocking one more time"?', '为什么每一级都在「再分一次块」？')}>
          <p>
            {t(
              <>
                Look back at the ladder: HBM → shared is block-level tiling, shared → registers is thread-level tiling, and inside the Tensor Core
                there's warp-level tiling. However many levels the GPU's memory hierarchy has, GEMM gets blocked that many times, with each cache
                layer earning a tile size tailored to its own capacity and bandwidth. CUTLASS simply turns this hierarchy of tile sizes into C++ template
                parameters.
              </>,
              <>
                回头看这条阶梯：HBM → shared 是 block 级分块，shared → 寄存器是线程级分块，Tensor Core
                内部还有 warp 级分块。GPU 的存储层级有几层，GEMM 就分几次块，
                每一层缓存都配一个为它的容量和带宽量身定制的瓦片尺寸。CUTLASS
                干脆把这套瓦片尺寸的层级做成了 C++ 模板参数。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={6}
        title={t('Occupancy: trading registers for parallelism', 'Occupancy：寄存器与并行度的交易')}
        lead={t(
          "Register tiling isn't free; what it eats is the SM's residency.",
          '寄存器 tiling 不是免费的，它吃掉的是 SM 的驻留能力。',
        )}
      >
        <p>
          {t(
            <>
              <ChapterLink n={2} /> explained that GPUs hide latency by keeping many warps resident and switching among them at will.{' '}
              <Term t="occupancy">The ratio of warps actually resident on an SM to the hardware ceiling (64 warps on the A100); it measures the
              ammunition reserve available for hiding latency.</Term> is the quantification of that ability. An SM's resources are fixed: on the
              A100 that's 65,536 registers, 164 KB of shared memory, 2,048 threads, and a residency ceiling of 32 blocks. Each block draws from
              these four pools according to its own needs, and <strong>whichever pool runs dry first determines the number of resident blocks</strong>,
              the minimum of the four constraints. Register tiling pushes per-thread registers from 30 to 100+, which turns the register pool into the
              first one to bottom out. Drag the sliders below to watch the bottleneck change hands:
            </>,
            <>
              <ChapterLink n={2} />讲过，GPU 靠「驻留大量 warp、随时切换」来隐藏延迟，
              <Term t="occupancy（占用率）">SM 上实际驻留的 warp 数与硬件上限（A100 为 64 个 warp）之比，
              衡量延迟隐藏的「弹药储备」。</Term>就是对这个能力的量化。一个 SM 的资源是死的：A100 上是 65536
              个寄存器、164 KB shared memory、2048 个线程、32 个 block 的驻留上限。
              每个 block 按自己的需求从这四个池子里扣资源，<strong>哪个池子先耗尽，哪个就决定了驻留的 block 数</strong>，
              也就是四个约束取最小值。寄存器 tiling 让每线程寄存器从 30 个涨到 100+，
              一下就把寄存器池变成最先见底的那个。拖一拖下面的滑杆，看瓶颈怎么换位：
            </>,
          )}
        </p>
        <OccupancyLab />
        <p>
          {t(
            <>
              So isn't register tiling shooting itself in the foot? No, and this is the section's most counterintuitive lesson:{' '}
              <strong>occupancy is a means, not an end</strong>. Its whole job is to guarantee there's always another warp to switch to
              while we wait on memory. Once resident warps cover the latency, more of them is just waste. Register tiling also gives each warp
              16× the independent compute (instruction-level parallelism), so its reliance on switch-to-hide drops to begin with. The typical
              profile of a high-performance GEMM is occupancy of only 25%–50% while running at over 90% of peak throughput. Go the other way and
              give a thread more variables than registers can hold, and the compiler moves the overflow to local memory: "local" in name only,
              physically in DRAM, where a single spill read or write costs hundreds of cycles and hurts far more than low occupancy ever does.
              When tuning, watch nvcc's <code>-maxrregcount</code> and the register-spilling counters in the profiler. They tell you more than the
              occupancy number does.
            </>,
            <>
              那寄存器 tiling 岂不是搬起石头砸自己的脚？不会，这恰恰是本节最反直觉的一课：
              <strong>occupancy 是手段，不是目的</strong>。它存在的意义是保证「访存等待时永远有别的 warp 可切」，
              只要驻留的 warp 足以盖住延迟，再多就是浪费。何况寄存器 tiling 让每个 warp 自带 16
              倍的独立计算（指令级并行），对「切换隐藏」的依赖本来就低了。高性能 GEMM 的典型画像，
              是 occupancy 只有 25%~50%，算力却跑到 90% 以上。反过来，要是某个线程的变量多到寄存器装不下，
              编译器就把溢出的部分挪到 local memory：名字叫 local，物理上却在显存里，
              一次溢出读写就是几百个周期，比低 occupancy 致命得多。调优时盯着 nvcc 的{' '}
              <code>-maxrregcount</code> 和 profiler 里的 register spilling 指标，比盯着 occupancy 数字有用得多。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'After rewriting the kernel so each thread computes a 4×4 micro-tile, per-thread register usage rises from 32 to 120. What is the most likely direct consequence?',
            '把 kernel 改成每线程算 4×4 micro-tile 后，每线程寄存器用量从 32 涨到 120。最可能的直接后果是？',
          )}
          options={[
            {
              text: t(
                "Compilation fails, because CUDA doesn't allow a single thread to use that many registers",
                '编译失败，CUDA 不允许单线程用这么多寄存器',
              ),
              explain: t(
                "Compilation is fine: the per-thread register ceiling is 255, and 120 is nowhere near it. The problem is whether they all fit together.",
                '编译没问题：单线程寄存器上限是 255 个，120 远没到顶。问题出在「合在一起够不够分」。',
              ),
            },
            {
              text: t(
                'Fewer warps can be resident on the SM at once (occupancy drops); if it overruns the budget, variables spill to local memory',
                'SM 上能同时驻留的 warp 变少（occupancy 下降）；若超过预算还会溢出到 local memory',
              ),
              correct: true,
              explain: t(
                'The register pool is shared across the whole SM: 65536 ÷ (120×256) ≈ 2 blocks, so occupancy falls to 25%. Usually a worthwhile trade, since each thread now does 16× the work. But if the compiler genuinely cannot fit everything, variables spill to local memory, which runs at DRAM speed, and performance falls off a cliff.',
                '寄存器池是整个 SM 共享的：65536 ÷ (120×256) ≈ 2 个 block，occupancy 掉到 25%。这通常是笔划算的交易，每线程干的活翻了 16 倍。但如果编译器实在塞不下，变量会溢出（spill）到 local memory，那是显存速度，性能立刻跳水。',
              ),
            },
            {
              text: t(
                'The hardware automatically puts the extra variables in shared memory, with no performance impact',
                '硬件自动把多出来的变量放进 shared memory，性能不受影响',
              ),
              explain: t(
                'No such automatic mechanism exists. The spill target is local memory (physically in DRAM), and spilling is decided by the compiler at compile time, not by the runtime scheduler.',
                '没有这种自动机制。溢出目标是 local memory（物理上在显存），而且溢出由编译器在编译期决定，不是运行时调度。',
              ),
            },
            {
              text: t(
                "No effect at all, since each thread's registers are private and unlimited",
                '没有任何影响，每个线程的寄存器是独立无限的',
              ),
              explain: t(
                'Quite the opposite: registers are one of the scarcest shared resources on an SM, 65,536 of them split across every resident thread. The more each thread uses, the fewer can be resident.',
                '恰恰相反：寄存器是 SM 上最紧俏的共享资源之一，65536 个要分给所有驻留线程。每线程用得越多，驻留得越少。',
              ),
            },
          ]}
        />
      </Section>

      <Section index={7} title={t('Summary & further reading', '总结与延伸阅读')}>
        <p>
          {t(
            'This chapter broke a 50× performance gap into five steps, each tied to one concrete hardware fact:',
            '这一章把一个 50 倍的性能差距拆成了五级台阶，每一级都对应一个明确的硬件事实：',
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>Tally first, then touch the code</strong>: GEMM is O(N³) compute over O(N²) data, and naive moves 512 GB at N=4096. Reuse
                is the one main thread running through all of matmul optimization.
              </>,
              <>
                <strong>先记账，再动手</strong>：GEMM 计算 O(N³)、数据 O(N²)，naive 在 N=4096 时要搬 512 GB。
                复用率是贯穿矩阵乘法优化的唯一主线。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Tiling's payoff equals the tile edge length</strong>: T×T blocking plus shared memory cuts global reads to 2N³/T; the two{' '}
                <code>__syncthreads()</code> guard against "reading before the write finished" and "overwriting before the read finished"
                respectively.
              </>,
              <>
                <strong>tiling 的收益等于瓦片边长</strong>：T×T 分块加上 shared memory 把全局读取降到
                2N³/T；两次 <code>__syncthreads()</code> 分别拦住「没写完就读」和「没读完就覆盖」。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>One blocking pass per memory level</strong>: HBM→shared is a block-level tile, shared→registers is a 4×4 micro-tile, and
                inside the Tensor Core there's a warp-level tile, all the same reuse idea.
              </>,
              <>
                <strong>每级存储层级分一次块</strong>：HBM→shared 是 block 级瓦片，shared→寄存器是 4×4
                micro-tile，Tensor Core 里还有 warp 级瓦片，全都是同一个复用思想。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Occupancy is a means, not a KPI</strong>: take the minimum of the four resource constraints; high-performance GEMM often
                trades occupancy for registers on purpose, and the real red line is register spilling.
              </>,
              <>
                <strong>occupancy 是手段不是 KPI</strong>：四个资源约束取最小值；高性能 GEMM
                常故意用低 occupancy 换寄存器，真正的红线是 register spilling。
              </>,
            )}
          </li>
        </ul>
        <p>{t('To go deeper, read these in order:', '想继续往深处走，这几份材料按顺序读：')}</p>
        <ul>
          <li>
            <a href="https://siboehm.com/articles/22/CUDA-MMM" target="_blank" rel="noreferrer" className="text-cyan">
              Simon Boehm — How to Optimize a CUDA Matmul Kernel
            </a>
            {t(
              <>
                : <strong>required reading</strong>. Ten kernels from naive to 94% of cuBLAS, each step backed by code, profiler data, and failed
                attempts. This chapter's performance ladder is modeled on it.
              </>,
              <>
                ：<strong>必读</strong>。从 naive 到 cuBLAS 94% 的十个 kernel，每一步都有代码、profiler
                数据和失败尝试，本章的性能阶梯就以它为蓝本。
              </>,
            )}
          </li>
          <li>
            <a href="https://github.com/NVIDIA/cutlass" target="_blank" rel="noreferrer" className="text-cyan">
              NVIDIA CUTLASS (GitHub)
            </a>
            {t(
              ': a production-grade blocked-GEMM template library that turns hierarchical tiling into composable template parameters; the open-source cousin of cuBLAS.',
              '：生产级的分块 GEMM 模板库，把层级化 tiling 做成了可组合的模板参数，cuBLAS 的开源近亲。',
            )}
          </li>
          <li>
            <a
              href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md"
              target="_blank"
              rel="noreferrer"
              className="text-cyan"
            >
              CUTLASS Docs — Efficient GEMM in CUDA
            </a>
            {t(
              ': one page of official diagrams that makes threadblock tile / warp tile / thread tile and the double-buffered pipeline click.',
              '：一页讲清 threadblock tile / warp tile / thread tile 三层分块与双缓冲流水线的官方图解。',
            )}
          </li>
          <li>
            <a href="https://docs.nvidia.com/cuda/cublas/" target="_blank" rel="noreferrer" className="text-cyan">
              cuBLAS Library Documentation
            </a>
            {t(
              ": in real engineering you'll most likely call GEMM rather than hand-write it, so learn its API conventions (column-major!) and algorithm-selection interface.",
              '：实际工程里你大概率是调用 GEMM 而不是手写，所以了解它的 API 约定（列主序！）和算法选择接口。',
            )}
          </li>
          <li>
            <a
              href="https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0"
              target="_blank"
              rel="noreferrer"
              className="text-cyan"
            >
              {t('PMPP (Programming Massively Parallel Processors), Chapter 5', 'PMPP（Programming Massively Parallel Processors）第 5 章')}
            </a>
            {t(
              ': a textbook-grade derivation of tiling, covering boundary handling and arbitrary-size matrices and the other engineering details this chapter skips.',
              '：tiling 的教科书级推导，包含边界处理、任意尺寸矩阵等本章略过的工程细节。',
            )}
          </li>
        </ul>
        <p>
          {t(
            <>
              Next chapter turns the recurring "memory time vs compute time" ledger into a proper picture, the Roofline model, so that any
              kernel's bottleneck is obvious at a glance.
            </>,
            <>
              下一章我们把「访存时间 vs 计算时间」这笔反复出现的账正式画成一张图，也就是 Roofline 模型，
              让任何 kernel 的瓶颈一眼可辨。
            </>,
          )}
        </p>
      </Section>
    </>
  )
}
