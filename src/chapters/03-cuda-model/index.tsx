import { Callout, ChapterLink, CodeBlock, Figure, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { IndexFigure } from './IndexFigure'
import { Grid2DFigure } from './Grid2DFigure'
import { GridConfigurator } from './GridConfigurator'
import { VecAddSim } from './VecAddSim'

const VEC_ADD = `__global__ void add(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}`

const LAUNCH_EN = `int blockSize = 256;                              // 256 threads per block
int numBlocks = (n + blockSize - 1) / blockSize;  // round up: ceil(n / 256)
add<<<numBlocks, blockSize>>>(dA, dB, dC, n);     // launch!`

const LAUNCH_ZH = `int blockSize = 256;                              // 每个 block 256 个线程
int numBlocks = (n + blockSize - 1) / blockSize;  // 向上取整：⌈n / 256⌉
add<<<numBlocks, blockSize>>>(dA, dB, dC, n);     // 启动！`

const HOST_EN = `#define CUDA_CHECK(call)                                               \\
    do {                                                               \\
        cudaError_t err = (call);                                      \\
        if (err != cudaSuccess) {                                      \\
            fprintf(stderr, "CUDA error: %s (%s:%d)\\n",                \\
                    cudaGetErrorString(err), __FILE__, __LINE__);      \\
            exit(1);                                                   \\
        }                                                              \\
    } while (0)

int main() {
    int n = 1 << 20;                       // 1M elements
    size_t bytes = n * sizeof(float);

    // host (CPU) memory, 'h' prefix
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

    // (1) allocate VRAM on the GPU ('d' prefix = device)
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // (2) copy inputs host -> device (H2D)
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // (3) launch the kernel (async! the CPU races ahead immediately)
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());        // catch bad launch parameters

    // (4) copy the result back to host (D2H) -- implicitly waits for the kernel
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // (5) free everything
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}`

const HOST_ZH = `#define CUDA_CHECK(call)                                               \\
    do {                                                               \\
        cudaError_t err = (call);                                      \\
        if (err != cudaSuccess) {                                      \\
            fprintf(stderr, "CUDA error: %s (%s:%d)\\n",                \\
                    cudaGetErrorString(err), __FILE__, __LINE__);      \\
            exit(1);                                                   \\
        }                                                              \\
    } while (0)

int main() {
    int n = 1 << 20;                       // 1M 个元素
    size_t bytes = n * sizeof(float);

    // host（CPU）侧内存，h 前缀
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

    // ① 在 GPU 上分配显存（d 前缀 = device）
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // ② 把输入从 host 拷到 device（H2D）
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // ③ 启动 kernel（异步！CPU 立刻继续往下跑）
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());        // 捕获启动参数错误

    // ④ 把结果拷回 host（D2H）—— 隐式等 kernel 跑完
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // ⑤ 释放
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}`

const MAT_2D_EN = `dim3 block(16, 16);                       // 16x16 = 256 threads per block
dim3 grid((W + 15) / 16, (H + 15) / 16);  // round up in both dimensions
matAdd<<<grid, block>>>(A, B, C, W, H);

__global__ void matAdd(float* A, float* B, float* C, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        int idx = row * W + col;          // 2D coords -> 1D memory address
        C[idx] = A[idx] + B[idx];
    }
}`

const MAT_2D_ZH = `dim3 block(16, 16);                       // 每个 block 16×16 = 256 线程
dim3 grid((W + 15) / 16, (H + 15) / 16);  // 两个维度都向上取整
matAdd<<<grid, block>>>(A, B, C, W, H);

__global__ void matAdd(float* A, float* B, float* C, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        int idx = row * W + col;          // 2D 坐标 → 1D 内存地址
        C[idx] = A[idx] + B[idx];
    }
}`

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            If you have written CPU code for a decade, the most disorienting thing about reading your first CUDA
            program isn't some new keyword — it's that <strong>something is missing: the loop</strong>. To add two
            arrays of a million elements, a CPU programmer's instinct is to write{' '}
            <code>for (i = 0; i &lt; n; i++)</code> and let one core grind through a million iterations. The first
            lesson of CUDA is to delete that for loop — the loop body stays, the loop itself vanishes, and in its
            place <strong>a million threads each run the body exactly once</strong>. "The i-th iteration" becomes
            "thread number i," and the loop variable <code>i</code> is no longer produced by <code>i++</code> but
            by each thread asking the hardware: "who am I?" This chapter makes that model crisp: how threads are
            numbered, why they're organized in layers, and the full round trip a CUDA program takes from CPU to GPU
            and back.
          </>,
          <>
            如果你写了十年 CPU 代码，第一次读 CUDA 程序时最不适应的不是哪个新关键字，而是<strong>有个东西不见了：循环</strong>。
            给两个一百万元素的数组做加法，CPU 程序员的本能是写 <code>for (i = 0; i &lt; n; i++)</code>，
            让一个核心吭哧吭哧跑一百万圈。CUDA 的第一课是把这个 for 循环删掉——循环体留下，循环本身消失，
            取而代之的是<strong>一百万个线程同时各执行一次循环体</strong>。「第 i 次迭代」变成了「第 i 号线程」，
            迭代变量 <code>i</code> 不再由 <code>i++</code> 产生，而是每个线程问硬件：「我是谁？」这一章就讲清楚这个模型：
            线程怎么编号、为什么要分层组织、以及一段 CUDA 程序从 CPU 到 GPU 再回来的完整旅程。
          </>,
        )}
      </p>

      <Section
        index={1}
        title={t('Your first kernel: vector addition', '第一个 kernel：向量加法')}
        lead={t(
          'Six lines of code that pack the three most central conventions of the CUDA programming model.',
          '六行代码，包含了 CUDA 编程模型最核心的三个约定。',
        )}
      >
        <p>
          {t(
            <>
              In CUDA, a function that runs on the GPU is called a{' '}
              <Term t={t('kernel', 'kernel（核函数）')}>
                {t(
                  'A function launched from the host and executed simultaneously by thousands of threads on the GPU. Every thread runs the same code and tells its data apart by its own index.',
                  '由 host 启动、在 GPU 上被成千上万个线程同时执行的函数。每个线程执行同一份代码，靠自己的索引区分要处理的数据。',
                )}
              </Term>
              . Here is CUDA's "Hello World" — vector addition <code>C = A + B</code>:
            </>,
            <>
              在 CUDA 里，跑在 GPU 上的函数叫{' '}
              <Term t={t('kernel', 'kernel（核函数）')}>
                {t(
                  'A function launched from the host and executed simultaneously by thousands of threads on the GPU. Every thread runs the same code and tells its data apart by its own index.',
                  '由 host 启动、在 GPU 上被成千上万个线程同时执行的函数。每个线程执行同一份代码，靠自己的索引区分要处理的数据。',
                )}
              </Term>
              。下面是 CUDA 世界的 “Hello World”——向量加法 <code>C = A + B</code>：
            </>,
          )}
        </p>
        <CodeBlock
          code={VEC_ADD}
          lang="cuda"
          title={t('vector_add.cu — the kernel', 'vector_add.cu — kernel 部分')}
          highlight={[2, 3]}
        />
        <p>{t('Line by line:', '逐行拆开看：')}</p>
        <ul>
          <li>
            {t(
              <>
                <strong>Line 1</strong>: <code>__global__</code> is CUDA's extension qualifier on top of C++,
                meaning "this function is called from the CPU side and executed on the GPU." Its return type must be{' '}
                <code>void</code> — who would tens of thousands of threads return a value to, anyway? Results can
                only be written into VRAM and carried back.
              </>,
              <>
                <strong>第 1 行</strong>：<code>__global__</code> 是 CUDA 对 C++ 的扩展修饰符，意思是「这个函数由 CPU
                侧调用、在 GPU 上执行」。它的返回值必须是 <code>void</code>——几万个线程同时返回值给谁呢？结果只能写进显存里带回来。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Line 2 (highlighted)</strong>: the soul of the entire CUDA model. As each thread reaches
                this line, it reads "who am I" from the built-in variables and computes a globally unique index{' '}
                <code>i</code>. Note that every thread executes this line, yet each one computes a different{' '}
                <code>i</code> — and this is exactly how "same code, different data" is realized.
              </>,
              <>
                <strong>第 2 行（高亮）</strong>：整个 CUDA 模型的灵魂。每个线程跑到这里时，从内建变量里读出「我是谁」，
                算出一个全局唯一的编号 <code>i</code>。注意这行代码所有线程都执行，但每个线程算出的 <code>i</code> 不同——
                这正是「同一份代码、不同的数据」的实现方式。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Line 3 (highlighted)</strong>: a bounds check. Looks redundant, but it's a lifesaver — more
                on that below.
              </>,
              <>
                <strong>第 3 行（高亮）</strong>：边界检查。看似多余，实则保命，下面单独说。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Line 4</strong>: the loop body itself. The one line that lived inside the for loop is
                carried over untouched — only the source of <code>i</code> has changed.
              </>,
              <>
                <strong>第 4 行</strong>：循环体本体。原来 for 循环里的那行代码原封不动搬过来，只是 <code>i</code> 的来源变了。
              </>,
            )}
          </li>
        </ul>
        <h3>{t('"Who am I": dissecting the index formula', '「我是谁」：拆解索引公式')}</h3>
        <p>
          {t(
            <>
              CUDA organizes threads in two layers (detailed in the next section): a number of <strong>blocks</strong>,
              each holding a number of threads. Three built-in variables assemble the global index:{' '}
              <code>blockIdx.x</code> is "which block am I in," <code>blockDim.x</code> is "how many threads per
              block," and <code>threadIdx.x</code> is "which thread am I within the block." So:
            </>,
            <>
              CUDA 把线程组织成两层（下一节细讲）：若干个 <strong>block</strong>，每个 block 里有若干个线程。三个内建变量
              (built-in variable) 拼出全局编号：<code>blockIdx.x</code> 是「我在第几个 block」，<code>blockDim.x</code>{' '}
              是「每个 block 有多少线程」，<code>threadIdx.x</code> 是「我是 block 内的第几号」。于是：
            </>,
          )}
        </p>
        <MathTex
          block
          tex={t(
            'i \\;=\\; \\underbrace{\\texttt{blockIdx.x} \\times \\texttt{blockDim.x}}_{\\text{start of my block}} \\;+\\; \\underbrace{\\texttt{threadIdx.x}}_{\\text{offset within block}}',
            'i \\;=\\; \\underbrace{\\texttt{blockIdx.x} \\times \\texttt{blockDim.x}}_{\\text{我所在 block 的起点}} \\;+\\; \\underbrace{\\texttt{threadIdx.x}}_{\\text{块内偏移}}',
          )}
        />
        <p>
          {t(
            <>
              This is the same arithmetic as "building number × apartments per floor + door number": first skip the
              entire index range occupied by all preceding blocks, then add your own position within the block. In
              the figure below, thread 3 of block 2 computes <code>i = 2 × 8 + 3 = 19</code>:
            </>,
            <>
              这和「楼号 × 每层户数 + 门牌号」是同一个算术：先跳过前面所有 block 占据的索引区间，再加上自己在块内的位置。
              下图里 block 2 的 3 号线程算出 <code>i = 2 × 8 + 3 = 19</code>：
            </>,
          )}
        </p>
        <Figure
          caption={t(
            'Global index = block start (cyan) + offset within block (amber). Each thread does this one multiply-add and gets the slice of data that belongs to it.',
            '全局索引 = block 起点（cyan）+ 块内偏移（amber）。每个线程做一次这个乘加，就拿到了属于自己的那份数据。',
          )}
        >
          <IndexFigure />
        </Figure>
        <h3>{t('Why if (i < n) is mandatory', '为什么必须 if (i < n)')}</h3>
        <p>
          {t(
            <>
              Threads aren't launched one at a time — they're launched in whole blocks. Say <code>n = 1000</code>{' '}
              with 256 threads per block. You can't launch "3.9 blocks," only round up to 4, so 1024 threads
              actually run — 24 more than the data. Those 24 threads compute indices <code>i</code> from 1000 to
              1023, and if you don't stop them, <code>C[i] = A[i] + B[i]</code> becomes a genuine{' '}
              <strong>out-of-bounds read and write</strong>: at best you corrupt data, at worst the whole kernel
              dies with <code>illegal memory access</code>. So that <code>if</code> isn't defensive-programming
              fussiness — it's the inevitable consequence of one structural fact: "thread count is rounded up to
              whole blocks, so it almost always exceeds the data size." Nearly every production kernel opens with a
              line of bounds checking just like this.
            </>,
            <>
              线程不是按 1 个的粒度启动的，而是按 block 的粒度启动。假设 <code>n = 1000</code>、每 block 256 线程，
              你不可能启动「3.9 个 block」，只能向上取整启动 4 个，于是实际有 1024 个线程在跑——比数据多出 24 个。
              这 24 个线程算出的 <code>i</code> 是 1000 到 1023，如果不拦住它们，<code>C[i] = A[i] + B[i]</code>{' '}
              就是一次实打实的<strong>越界读写</strong>：轻则数据损坏，重则整个 kernel 报{' '}
              <code>illegal memory access</code> 崩掉。所以那行 <code>if</code> 不是防御性编程的洁癖，
              而是「线程数按 block 取整、几乎永远多于数据量」这一结构性事实的必然要求。几乎每一个生产 kernel
              的开头都站着这么一行边界检查。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={2}
        title={t('The thread hierarchy: thread → block → grid', '线程层级：thread → block → grid')}
        lead={t(
          'Why two layers instead of one? This single design choice is what makes CUDA programs scale for free.',
          '为什么是两层而不是一层？这个设计决定了 CUDA 程序天然可扩展。',
        )}
      >
        <p>
          {t(
            <>
              All the threads produced by one kernel launch are collectively called a <strong>grid</strong>. A grid
              is made of <strong>blocks</strong>, and each block is made of <strong>threads</strong>. The launch
              syntax writes these two layers of parameters inside the signature triple-angle-brackets:
            </>,
            <>
              一次 kernel 启动产生的所有线程统称一个 <strong>grid（网格）</strong>。grid 由若干{' '}
              <strong>block（线程块）</strong>组成，每个 block 由若干 <strong>thread（线程）</strong>组成。
              启动语法把这两层参数写在标志性的三对尖括号里：
            </>,
          )}
        </p>
        <CodeBlock
          code={t(LAUNCH_EN, LAUNCH_ZH)}
          lang="cuda"
          title={t('kernel launch syntax', 'kernel 启动语法')}
        />
        <p>
          {t(
            <>
              <code>&lt;&lt;&lt;numBlocks, blockSize&gt;&gt;&gt;</code> reads as "launch numBlocks blocks, each with
              blockSize threads." The natural first question is: why bother with two layers? Why can't I just write
              "launch 1048576 threads"? The answer lies in what each layer can and can't do:
            </>,
            <>
              <code>&lt;&lt;&lt;numBlocks, blockSize&gt;&gt;&gt;</code> 读作「启动 numBlocks 个 block，每个 block 含
              blockSize 个线程」。第一个问题自然是：为什么要费劲分两层？让我直接写「启动 1048576 个线程」不行吗？
              答案藏在两层各自的能力差异里：
            </>,
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>Inside a block: threads are "coworkers."</strong> Threads in the same block are guaranteed
                to execute on the same{' '}
                <Term t={t('SM (Streaming Multiprocessor)', 'SM（Streaming Multiprocessor）')}>
                  {t(
                    "The GPU's basic compute unit, containing execution units, a register file, and shared memory. A block stays pinned to one SM from launch to finish.",
                    'GPU 的基本计算单元，含执行单元、寄存器堆和 shared memory。一个 block 从启动到结束都钉在一个 SM 上。',
                  )}
                </Term>
                , so they can wait on each other via <code>__syncthreads()</code> (a barrier) and share a fast cache
                inside the SM —{' '}
                <Term t={t('shared memory', 'shared memory（共享内存）')}>
                  {t(
                    'A programmable on-chip cache inside the SM, on the order of a hundred KB, with latency an order of magnitude lower than VRAM. The star of Chapter 4.',
                    '位于 SM 片上的可编程缓存，约百 KB 量级，延迟比显存低一个数量级。第 4 章的主角。',
                  )}
                </Term>{' '}
                — to exchange intermediate results.
              </>,
              <>
                <strong>block 内：线程是「同事」。</strong>同一个 block 的线程保证在同一个{' '}
                <Term t={t('SM (Streaming Multiprocessor)', 'SM（Streaming Multiprocessor）')}>
                  {t(
                    "The GPU's basic compute unit, containing execution units, a register file, and shared memory. A block stays pinned to one SM from launch to finish.",
                    'GPU 的基本计算单元，含执行单元、寄存器堆和 shared memory。一个 block 从启动到结束都钉在一个 SM 上。',
                  )}
                </Term>{' '}
                上执行，因此它们可以通过 <code>__syncthreads()</code> 互相等待（路障同步，barrier），还能共用一块
                SM 内的高速缓存——
                <Term t={t('shared memory', 'shared memory（共享内存）')}>
                  {t(
                    'A programmable on-chip cache inside the SM, on the order of a hundred KB, with latency an order of magnitude lower than VRAM. The star of Chapter 4.',
                    '位于 SM 片上的可编程缓存，约百 KB 量级，延迟比显存低一个数量级。第 4 章的主角。',
                  )}
                </Term>
                ，用来交换中间结果。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Between blocks: they're "strangers."</strong> CUDA explicitly states that blocks cannot
                synchronize with each other, are not guaranteed any execution order, and aren't even guaranteed to
                coexist — the hardware might finish 50 blocks first and then schedule the next 50.
              </>,
              <>
                <strong>block 间：彼此是「陌生人」。</strong>CUDA 明确规定 block 之间不能同步、不保证执行顺序、
                甚至不保证同时存在——可能先跑完 50 个 block，再调度下 50 个。
              </>,
            )}
          </li>
        </ul>
        <Callout
          type="insight"
          title={t(
            'Block independence is not a limitation — it is the source of scalability',
            'block 间独立不是限制，而是可扩展性的来源',
          )}
        >
          <p>
            {t(
              <>
                Precisely because blocks are independent, the hardware scheduler can treat them as{' '}
                <strong>a bag of unrelated tasks</strong> and toss one at whichever SM just went idle. The same
                compiled binary runs on a laptop GPU with 10 SMs — the hardware lays out 10 blocks at once — and on
                an H100 with 132 SMs — 132 at once — without changing a single line, performance scaling
                automatically with the SM count. This is what CUDA calls{' '}
                <strong>transparent scalability</strong>: you describe "which units of work are independent," and
                the hardware decides "how to spread them out." Conversely, the moment you quietly assume an
                execution order between blocks (say, block 1 waits on data written by block 0), your program
                deadlocks or produces wrong results on certain cards — independence is the contract between you and
                the hardware.
              </>,
              <>
                正因为 block 互相独立，硬件调度器才可以把它们<strong>当作一袋互不相干的任务</strong>，哪个 SM
                空了就扔一个过去。同一份编译产物，在 10 个 SM 的笔记本 GPU 上跑，硬件一次摆 10 个 block；
                在 132 个 SM 的 H100 上跑，一次摆 132 个——程序一行不改，性能随 SM 数量自动伸缩。
                这就是 CUDA 所谓的「透明可扩展性（transparent scalability）」：你描述「有哪些独立的工作」，
                硬件决定「怎么铺开」。反过来，一旦你偷偷假设了 block 间的执行顺序（比如 block 1 等 block 0
                写好的数据），程序就会在某些卡上死锁或出错——独立性是你和硬件之间的契约。
              </>,
            )}
          </p>
        </Callout>
        <p>
          {t(
            <>
              The two-layer structure maps cleanly onto two layers of hardware reality: a block's ability to
              cooperate comes from the physical fact that it's "pinned to one SM"; the independence between blocks
              comes from the commercial fact that "the SM count varies card to card." One layer is too flat (who
              cooperates with whom among a million threads?), three is too baroque, and two cuts "the scope of
              cooperation" cleanly apart from "the scale of parallelism." As an aside: within a block, the hardware
              further slices threads into groups of 32 called{' '}
              <Term t={t('warps', 'warp（线程束）')}>
                {t(
                  'The smallest unit the hardware schedules — 32 threads executing the same instruction in lockstep. The star of the previous chapter.',
                  '硬件调度的最小单位，32 个线程锁步执行同一条指令。上一章的主角。',
                )}
              </Term>
              . This is a third layer from the hardware's point of view, but it doesn't appear in the launch syntax
              of the programming model — you declare threads and blocks, and the hardware slices warps on its own.
            </>,
            <>
              两层结构正好映射到硬件的两层现实：block 内的协作能力来自「钉死在一个 SM 上」这个物理事实；block
              间的独立性来自「SM 数量因卡而异」这个商业事实。一层太扁（百万线程谁跟谁协作？），三层太繁，
              两层刚好把「能协作的范围」和「能并行的规模」切开。顺带一提：block 内的线程在硬件上还会被切成 32
              个一组的{' '}
              <Term t={t('warps', 'warp（线程束）')}>
                {t(
                  'The smallest unit the hardware schedules — 32 threads executing the same instruction in lockstep. The star of the previous chapter.',
                  '硬件调度的最小单位，32 个线程锁步执行同一条指令。上一章的主角。',
                )}
              </Term>
              ，这是硬件视角的第三层，但它不出现在编程模型的启动语法里——你声明 thread 和 block，硬件自己去切 warp。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'Why does a CUDA kernel almost always need an if (i < n) bounds check?',
            '为什么 CUDA kernel 里几乎总要写 if (i < n) 边界检查？',
          )}
          options={[
            {
              text: t(
                "Because the GPU's floating-point math is imprecise, so an if is needed to filter out wrong results",
                '因为 GPU 的浮点运算不精确，需要用 if 过滤掉错误结果',
              ),
              explain: t(
                'The bounds check has nothing to do with floating-point precision; it only cares whether the index i falls inside the valid data range.',
                '边界检查与浮点精度毫无关系，它只关心索引 i 是否落在合法的数据范围内。',
              ),
            },
            {
              text: t(
                'Because threads launch in whole blocks, and after rounding the grid size up, the total thread count almost always exceeds the data size n — the extra threads compute out-of-bounds indices',
                '因为线程按 block 的粒度整批启动，grid 大小向上取整后，线程总数几乎总是大于数据量 n，多出来的线程索引会越界',
              ),
              correct: true,
              explain: t(
                "Correct. gridDim = ceil(n / blockDim), so as long as n isn't an exact multiple of blockDim, the last block contains threads with indices >= n. Not stopping them is an out-of-bounds read/write — a structural problem, not a matter of coding style.",
                '正解。gridDim = ⌈n / blockDim⌉，只要 n 不是 blockDim 的整数倍，最后一个 block 里就有线程拿到 ≥ n 的索引。不拦住它们就是越界读写——这是结构性问题，不是编程风格问题。',
              ),
            },
            {
              text: t(
                'To let the compiler optimize branch prediction better',
                '为了让编译器能更好地优化分支预测',
              ),
              explain: t(
                "The GPU has nothing like a CPU's branch predictor. This if is a correctness requirement, not a performance trick.",
                'GPU 没有 CPU 那种分支预测器。这行 if 是正确性需求，不是性能技巧。',
              ),
            },
            {
              text: t(
                'Only needed while debugging; the release build can drop it for speed',
                '只有调试时需要，发布版可以去掉以提升性能',
              ),
              explain: t(
                'As long as n is not guaranteed to divide blockDim evenly, a "release build" without the bounds check is a program with an out-of-bounds write that can crash or corrupt other data at any moment.',
                '只要 n 不能保证整除 blockDim，去掉边界检查的「发布版」就是一个带越界写的程序，随时可能崩溃或写坏别人的数据。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title={t('Lab: Grid/Block configurator', '实验：Grid/Block 配置器')}
        lead={t(
          'How gridDim is computed, how warps get sliced, where the wasted threads hide — drag a few sliders and it clicks.',
          'gridDim 怎么算、warp 怎么切、浪费的线程藏在哪——拖一拖就有手感了。',
        )}
      >
        <p>
          {t(
            <>
              Choosing launch parameters is daily bread for every CUDA programmer: blockDim is usually 128–256 (must
              be a multiple of 32, otherwise the last warp is half-empty — pure waste), and gridDim is computed with
              the classic integer ceiling trick <code>(n + blockSize - 1) / blockSize</code>. The configurator below
              puts the whole tally on the table: each little cell is a thread, the light/dark stripes mark the warp
              boundaries, and rose cells are the out-of-bounds wasted threads. <strong>Click any thread cell</strong>{' '}
              and the right-hand panel walks through its index derivation.
            </>,
            <>
              配置 launch 参数是每个 CUDA 程序员的日常：blockDim 通常选 128～256（必须是 32 的倍数，否则最后一个
              warp 半满，纯浪费），gridDim 用 <code>(n + blockSize - 1) / blockSize</code> 这个经典的整数向上取整算出来。
              下面的配置器把这笔账摆在台面上：每个小格是一个线程，深浅条纹标出 warp 的切分，rose
              色是越界的浪费线程。<strong>点击任意线程格</strong>，右侧面板会演算它的索引推导。
            </>,
          )}
        </p>
        <GridConfigurator />
        <p>
          {t(
            <>
              What's worth noticing is the proportion of wasted threads: at N=1000, blockDim=256 you waste only
              24/1024 ≈ 2.3%, harmless; but at N=16, blockDim=1024 the waste hits 98% — the whole card has just 16
              threads doing actual work. This is exactly why "a job with too little data isn't worth shipping to the
              GPU": no launch configuration can rescue a problem that simply can't keep the machine fed. Another
              intuition: bigger blockDim isn't always better. It affects the scope of cooperation and occupancy
              (detailed in <ChapterLink n={5} />), but has no effect on "whether the result is correct" — correctness
              comes from the index formula and the bounds check.
            </>,
            <>
              值得注意的是浪费线程的占比：N=1000、blockDim=256 时只浪费 24/1024 ≈ 2.3%，无伤大雅；但若 N=16、blockDim=1024，
              浪费高达 98%——整卡只有 16 个线程在干活。这也是为什么「数据量太小的活不值得下发到 GPU」：
              launch 配置救不了本来就喂不饱机器的问题。另一个直觉：blockDim 不是越大越好，它影响的是协作范围和占用率（
              <ChapterLink n={5} />细讲），对「能不能算对」没有影响——算对靠的是索引公式和边界检查。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'N = 1000, blockSize = 256, launching this vecAdd kernel: what is gridDim, and how many threads are wasted?',
            'N = 1000、blockSize = 256，启动这个 vecAdd kernel：gridDim 取多少？浪费多少个线程？',
          )}
          options={[
            {
              text: t('gridDim = 3, 0 wasted', 'gridDim = 3，浪费 0 个'),
              explain: t(
                '3 × 256 = 768 < 1000, so 232 elements would go unprocessed — the result is simply wrong. You must round up.',
                '3 × 256 = 768 < 1000，会有 232 个元素没人处理——结果直接算错。必须向上取整。',
              ),
            },
            {
              text: t('gridDim = 4, 0 wasted', 'gridDim = 4，浪费 0 个'),
              explain: t(
                'gridDim = 4 is right, but 4 × 256 = 1024 > 1000, so in the last block the 24 threads with indices 1000–1023 fail the bounds check and spin idle.',
                'gridDim = 4 对了，但 4 × 256 = 1024 > 1000，最后一个 block 里索引 1000–1023 的 24 个线程过不了边界检查，是空转的。',
              ),
            },
            {
              text: t('gridDim = 4, 24 wasted', 'gridDim = 4，浪费 24 个'),
              correct: true,
              explain: t(
                'ceil(1000 / 256) = 4 blocks, 1024 threads total; 1024 − 1000 = 24 threads have i >= 1000 and are stopped by if (i < n), spinning idle but harmless. This 2.3% waste is the inherent cost of launching in whole blocks.',
                '⌈1000 / 256⌉ = 4 个 block，共 1024 个线程；1024 − 1000 = 24 个线程的 i ≥ 1000，被 if (i < n) 拦下，空转但无害。这 2.3% 的浪费是按 block 整批启动的固有代价。',
              ),
            },
            {
              text: t('gridDim = 5, 280 wasted', 'gridDim = 5，浪费 280 个'),
              explain: t(
                '5 blocks is 1280 threads, which computes correctly (the extra 280 are stopped by the bounds check), but wastes a whole extra block of scheduling overhead beyond the necessary 4.',
                '5 个 block 是 1280 个线程，能算对（多出的 280 个被边界检查拦住），但比必要的 4 个 block 多浪费了一整个 block 的调度开销。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={4}
        title={t('Lab: vecAdd execution simulator', '实验：vecAdd 执行模拟器')}
        lead={t(
          'Slow a toy case of N=24, blockSize=8 down by 10,000× and watch what happens at every step.',
          '把 N=24、blockSize=8 的玩具例放慢一万倍，看每一步发生了什么。',
        )}
      >
        <p>
          {t(
            <>
              Now that the formula makes sense, watch it execute. The simulator below slows one{' '}
              <code>add&lt;&lt;&lt;3, 8&gt;&gt;&gt;</code> launch into a single-steppable animation: at each step,
              the 8 threads of the current block do the same thing <strong>simultaneously</strong> — compute their
              index together, clear the bounds check together, read A and B together (cyan flash), write C together
              (volt flash). Notice that no single thread is "looping" — the loop has been spatialized into 24 cells
              side by side.
            </>,
            <>
              公式看懂了，再看执行。下面的模拟器把一次 <code>add&lt;&lt;&lt;3, 8&gt;&gt;&gt;</code>{' '}
              启动放慢成可以单步的动画：每一步里，当前 block 的 8 个线程<strong>同时</strong>做同一件事——同时算索引、
              同时过边界检查、同时读 A 和 B（cyan 闪烁）、同时写 C（volt 闪烁）。注意没有任何一个线程在「循环」，
              循环被空间化成了 24 个并排的格子。
            </>,
          )}
        </p>
        <VecAddSim />
        <p>
          {t(
            <>
              The button to really play with is "shuffle scheduling": it randomly reorders the execution of the 3
              blocks and replays. Block 2 runs first, block 0 brings up the rear? C's result is identical to the
              last bit. This is the visualization of the contract from the previous section — each element is
              touched exactly once by "its own" thread, there's no data dependency between threads, so the schedule
              order is irrelevant. On a real GPU this kind of reordering happens constantly: whichever SM frees up
              first gets the next block. <strong>Never assume an execution order between blocks</strong> — a line
              worth repeating to yourself three times.
            </>,
            <>
              重点玩「乱序调度」按钮：它随机打乱 3 个 block 的执行顺序重放一遍。block 2 先跑、block 0
              垫底？C 的结果分毫不差。这正是上一节那条契约的可视化——每个元素只被「自己的」那个线程碰一次，
              线程之间没有任何数据依赖，所以调度顺序无关紧要。真实 GPU 上这种乱序时刻都在发生：哪个 SM 先空出来，
              哪个 block 就先上。<strong>永远不要假设 block 间的执行顺序</strong>，这句话值得在心里默念三遍。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={5}
        title={t('The host-side five steps: a complete CUDA program', 'host 侧五部曲：一段完整的 CUDA 程序')}
        lead={t(
          'The kernel is just the tip of the iceberg; below the waterline is VRAM management and data movement.',
          'kernel 只是冰山一角，水面下是显存管理和数据搬运。',
        )}
      >
        <p>
          {t(
            <>
              The GPU has its own separate VRAM. A CPU (host) pointer is meaningless on the GPU (device), and vice
              versa. So the host side of every CUDA program follows the same rhythm, known colloquially as the five
              steps: <strong>allocate VRAM → copy in → launch → copy back → free</strong>.
            </>,
            <>
              GPU 有自己独立的显存，CPU（host）的指针在 GPU（device）上没有意义，反之亦然。所以每个 CUDA
              程序的 host 侧都绕不开同一套节奏，江湖人称五部曲：<strong>分配显存 → 拷入 → 启动 → 拷回 → 释放</strong>。
            </>,
          )}
        </p>
        <CodeBlock
          code={t(HOST_EN, HOST_ZH)}
          lang="cuda"
          title={t('vector_add.cu — the host side', 'vector_add.cu — host 部分')}
          highlight={[31, 35]}
        />
        <p>{t('Three spots that trip people up:', '有三个容易栽跟头的点：')}</p>
        <ul>
          <li>
            {t(
              <>
                <strong>The kernel launch is asynchronous (line 31).</strong> The line{' '}
                <code>add&lt;&lt;&lt;...&gt;&gt;&gt;</code> only <em>submits</em> the work to the GPU; the CPU
                doesn't wait for it to finish and runs straight on to the next line. That's a good thing — the CPU
                can use the time to prep the next batch of data — but it also means reading the result right after
                the launch reads nothing. To wait explicitly, use <code>cudaDeviceSynchronize()</code>.
              </>,
              <>
                <strong>kernel 启动是异步的（第 31 行）。</strong>
                <code>add&lt;&lt;&lt;...&gt;&gt;&gt;</code> 这行只是把任务<em>提交</em>给 GPU，CPU 不等它跑完就继续执行下一行。
                这是好事——CPU 可以趁机准备下一批数据——但也意味着你在 launch 之后立刻读结果是读不到的。
                想显式等待用 <code>cudaDeviceSynchronize()</code>。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>cudaMemcpy is an implicit sync point (line 35).</strong> A D2H copy can't start until the
                kernel has finished writing C, so this line naturally "waits" for the kernel to end. Many beginner
                programs compute correctly without any explicit sync, riding entirely on this safety net — but know
                the why behind the what, because the moment you switch to an async copy (
                <code>cudaMemcpyAsync</code>) this free insurance is gone.
              </>,
              <>
                <strong>cudaMemcpy 是隐式同步点（第 35 行）。</strong>D2H 拷贝必须等 kernel 把 C
                写完才开始搬，所以这一行天然「等」了 kernel 结束。很多入门程序没写任何显式同步也能算对，全靠它兜底——知其然也要知其所以然，
                等你换成异步拷贝（<code>cudaMemcpyAsync</code>）时这层免费保险就没了。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Every CUDA API call returns an error code, and you must check it.</strong> CUDA errors are
                "sticky" and often surface late: an out-of-bounds access inside a kernel frequently isn't reported
                until the next synchronizing operation, and a program that doesn't check error codes will crash
                mysteriously a thousand miles from the actual error site. So everyone wraps every call in a{' '}
                <code>CUDA_CHECK</code> macro (see the top of the code) that stops the moment something fails, with
                the file name and line number attached. This isn't an optional good habit — it's the passing grade
                for CUDA engineering.
              </>,
              <>
                <strong>每个 CUDA API 调用都返回错误码，必须检查。</strong>CUDA 的错误是「粘性」的而且经常延迟暴露：kernel
                里的越界访问往往要到下一次同步操作才报出来，不查错误码的程序会在错误现场的十万八千里外神秘崩溃。
                所以所有人都用一个 <code>CUDA_CHECK</code> 宏把每个调用包起来（见代码开头），出错立刻带着文件名和行号停下。
                这不是可选的好习惯，是 CUDA 工程的及格线。
              </>,
            )}
          </li>
        </ul>
        <Callout type="note" title={t('A preview from the performance angle', '性能视角的预告')}>
          <p>
            {t(
              <>
                The most expensive part of the five steps is often not the compute but the PCIe movement in steps 2
                and 4: PCIe 4.0 x16 measures around 25 GB/s, while H100 VRAM bandwidth is around 3.35 TB/s — two
                orders of magnitude apart. For a kernel like vecAdd that "moves 12 bytes to do 1 addition," the
                end-to-end time is spent almost entirely on movement. How to save on movement, and how to overlap
                compute with movement, is the through-line of <ChapterLink n={4} label="Chapters 4" /> and{' '}
                <ChapterLink n={6} />.
              </>,
              <>
                五部曲里最贵的常常不是计算而是②和④的 PCIe 搬运：PCIe 4.0 x16 实测约 25 GB/s 上下，而 H100 显存带宽约
                3.35 TB/s，差出两个数量级。vecAdd 这种「搬 12 字节算 1 次加法」的 kernel，端到端时间几乎全花在搬运上。
                怎么省搬运、怎么让计算和搬运重叠，是 <ChapterLink n={4} /> 和 <ChapterLink n={6} /> 的主线。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={6}
        title={t('Multidimensional indexing: handling matrices with a 2D grid', '多维索引：用 2D grid 处理矩阵')}
        lead={t(
          "However many dimensions the data has, the thread coordinates can match — purely to keep the code hugging the data's shape.",
          '数据是几维的，线程坐标就可以是几维的——纯粹为了让代码贴着数据长。',
        )}
      >
        <p>
          {t(
            <>
              When handling two-dimensional data like images or matrices, manually flattening them to 1D and
              computing the index by hand works, of course — but CUDA offers a more natural notation: grids and
              blocks themselves can be 1D, 2D, or 3D, declared with the <code>dim3</code> type. The index formula is
              independent per dimension, each one the familiar "block start + offset within block":
            </>,
            <>
              处理图像、矩阵这类二维数据时，把它们手动压扁成一维再算索引当然可行，但 CUDA 提供了更顺手的写法：grid 和
              block 本身可以是 1D、2D 或 3D 的，用 <code>dim3</code> 类型声明。索引公式逐维独立，每一维都是熟悉的「block
              起点 + 块内偏移」：
            </>,
          )}
        </p>
        <CodeBlock
          code={t(MAT_2D_EN, MAT_2D_ZH)}
          lang="cuda"
          title={t('matrix_add.cu — 2D indexing', 'matrix_add.cu — 2D 索引')}
          highlight={[6, 7, 9]}
        />
        <p>
          {t(
            <>
              Picture an H×W image tiled by 16×16 tiles (blocks): <code>blockIdx.x/y</code> is the tile's column and
              row number, <code>threadIdx.x/y</code> is the local coordinate inside the tile, and a multiply-add per
              dimension gives the global <code>col</code> and <code>row</code>. The bounds check is per-dimension
              too — image width and height are rarely exact multiples of 16, so the tiles on the right and bottom
              edges always poke out a little. The last line, <code>idx = row * W + col</code>, reminds us:
              multidimensionality is just syntactic sugar the programming model gives you. VRAM is always a 1D byte
              sequence, and 2D coordinates ultimately convert back to a linear address (row-major).
            </>,
            <>
              想象一张 H×W 的图被 16×16 的瓷砖（block）铺满：<code>blockIdx.x/y</code> 是瓷砖的列号和行号，
              <code>threadIdx.x/y</code> 是瓷砖内的局部坐标，两维分别做乘加就得到全局的 <code>col</code> 和{' '}
              <code>row</code>。边界检查也要逐维做——图像宽高一般都不是 16 的整数倍，右边缘和下边缘的瓷砖都会探出去一截。
              最后一行 <code>idx = row * W + col</code> 提醒我们：多维只是编程模型给的语法糖，显存永远是一维的字节序列，
              2D 坐标终究要换算回线性地址（行优先，row-major）。
            </>,
          )}
        </p>
        <Figure
          caption={t(
            'dim3 block(4,4) × dim3 grid(3,2) tiling a 12×8 image. The cyan frame is the tile at blockIdx=(1,1); the volt cell is its thread threadIdx=(2,1), responsible for pixel (row 5, col 6).',
            'dim3 block(4,4) × dim3 grid(3,2) 铺满 12×8 的图像。cyan 框是 blockIdx=(1,1) 的瓷砖，volt 格是其中 threadIdx=(2,1) 的线程，负责像素 (row 5, col 6)。',
          )}
        >
          <Grid2DFigure />
        </Figure>
        <p>
          {t(
            <>
              A conventional gotcha: <code>x</code> is horizontal (column) and <code>y</code> is vertical (row), so{' '}
              <code>row</code> maps to <code>y</code> and <code>col</code> maps to <code>x</code> — the exact
              reverse of how the matrix subscript <code>(row, col)</code> is written. And this mapping isn't just
              pretty: letting <code>threadIdx.x</code> run along the memory-contiguous direction (within a row)
              means adjacent threads access adjacent addresses, which is the setup for the next chapter's "memory
              coalescing." Pick the wrong indexing scheme and performance can differ by several times to an order of
              magnitude.
            </>,
            <>
              约定俗成的坑：<code>x</code> 是横向（列），<code>y</code> 是纵向（行），所以 <code>row</code> 对应{' '}
              <code>y</code>、<code>col</code> 对应 <code>x</code>，和矩阵下标 <code>(row, col)</code>{' '}
              的书写顺序正好相反。另外这个映射不只是好看——让 <code>threadIdx.x</code> 沿着内存连续的方向（行内）走，
              相邻线程就会访问相邻地址，这是下一章「合并访存」的伏笔：索引方式选错，性能可以差出几倍到一个数量级。
            </>,
          )}
        </p>
      </Section>

      <Section index={7} title={t('Summary and further reading', '总结与延伸阅读')}>
        <p>
          {t(
            'This chapter builds CUDA’s worldview; every later optimization grows on top of these few intuitions:',
            '这一章建立的是 CUDA 的世界观，所有后续优化都长在这几条直觉上：',
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>The loop is gone</strong>: the for loop is spatialized into a vast number of threads, and
                each thread claims its slice of data with{' '}
                <code>blockIdx.x * blockDim.x + threadIdx.x</code>.
              </>,
              <>
                <strong>循环消失了</strong>：for 循环被空间化成海量线程，每个线程用{' '}
                <code>blockIdx.x * blockDim.x + threadIdx.x</code> 认领自己的那份数据。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The bounds check is not optional</strong>: threads launch in whole blocks and the grid is
                rounded up, so the thread count almost always exceeds the data size, and out-of-bounds threads must
                be stopped by <code>if (i &lt; n)</code>.
              </>,
              <>
                <strong>边界检查不可省</strong>：线程按 block 整批启动、grid 向上取整，线程数几乎总是多于数据量，
                越界线程必须被 <code>if (i &lt; n)</code> 拦住。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The two layers each have their job</strong>: within a block, threads can synchronize and
                share memory (the unit of cooperation); between blocks they're fully independent (the unit of
                scheduling) — and that independence buys you a program that scales transparently with the SM count.
              </>,
              <>
                <strong>两层层级各司其职</strong>：block 内可同步、可共享内存（协作单位）；block
                间完全独立（调度单位）——独立性换来了程序随 SM 数量透明扩展。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The host five steps</strong>: cudaMalloc → H2D → launch → D2H → cudaFree; the launch is
                async, cudaMemcpy syncs implicitly, and every call needs a <code>CUDA_CHECK</code>.
              </>,
              <>
                <strong>host 五部曲</strong>：cudaMalloc → H2D → launch → D2H → cudaFree；launch 异步、cudaMemcpy
                隐式同步、每个调用都要 <code>CUDA_CHECK</code>。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Multidimensional indexing is syntactic sugar</strong>: dim3 lets thread coordinates grow to
                match the data's shape, but VRAM is forever one-dimensional.
              </>,
              <>
                <strong>多维索引是语法糖</strong>：dim3 让线程坐标贴着数据形状长，但显存永远是一维的。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Further reading, ordered from "quick start" to "deep dive":', '延伸阅读，按「上手快 → 钻得深」排序：')}</p>
        <ul>
          <li>
            <a href="https://developer.nvidia.com/blog/even-easier-introduction-cuda/" target="_blank" rel="noreferrer">
              {t(
                'An Even Easier Introduction to CUDA (NVIDIA Developer Blog)',
                'An Even Easier Introduction to CUDA（NVIDIA 开发者博客）',
              )}
            </a>
            {t(
              " — Mark Harris's classic intro, using unified memory to simplify the five steps; you can run this chapter's code in half an hour.",
              '—— Mark Harris 的经典入门文，用统一内存简化五部曲，半小时就能跑通本章代码。',
            )}
          </li>
          <li>
            <a
              href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model"
              target="_blank"
              rel="noreferrer"
            >
              {t(
                'CUDA C++ Programming Guide — Ch.2 Programming Model (NVIDIA official docs)',
                'CUDA C++ Programming Guide — Ch.2 Programming Model（NVIDIA 官方文档）',
              )}
            </a>
            {t(
              ' — the authoritative source for every concept in this chapter; the thread hierarchy section is worth reading word for word.',
              '—— 本章所有概念的权威出处，thread hierarchy 一节值得逐字读。',
            )}
          </li>
          <li>
            <a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41487/" target="_blank" rel="noreferrer">
              {t(
                'How CUDA Programming Works (GTC 2022, Stephen Jones)',
                'How CUDA Programming Works（GTC 2022, Stephen Jones）',
              )}
            </a>
            {t(
              " — an NVIDIA CUDA architect explaining the hardware motivation behind the programming model in person; the best answer to \"why is it designed this way.\"",
              '—— NVIDIA CUDA 架构师亲述编程模型背后的硬件动机，是「为什么这么设计」的最佳答案。',
            )}
          </li>
          <li>
            <a
              href="https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors"
              target="_blank"
              rel="noreferrer"
            >
              {t(
                'Programming Massively Parallel Processors (PMPP, 4th ed.) Ch.2–3',
                'Programming Massively Parallel Processors（PMPP, 4th ed.）Ch.2–3',
              )}
            </a>
            {t(
              ' — Hwu & Kirk’s textbook; Chapters 2 and 3 spend dozens of pages unfolding this chapter to every detail, with plenty of exercises.',
              '—— Hwu & Kirk 的教科书，第 2、3 章用几十页把本章内容展开到每个细节，配大量习题。',
            )}
          </li>
          <li>
            <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html" target="_blank" rel="noreferrer">
              {t(
                'CUDA Runtime API — Error Handling (NVIDIA official docs)',
                'CUDA Runtime API — Error Handling（NVIDIA 官方文档）',
              )}
            </a>
            {t(
              ' — the full semantics of error codes; understand "sticky errors" and why async errors surface late.',
              '—— 错误码的完整语义，理解「粘性错误」和异步报错为什么会延迟暴露。',
            )}
          </li>
        </ul>
        <p>
          {t(
            <>
              In the next chapter we follow this thread downward: we have threads, the indices are right, but
              whether a kernel runs fast comes down ninety percent to <em>what pattern</em> these threads touch VRAM
              with — memory access is king.
            </>,
            <>
              下一章我们沿着这条线往下走：线程有了、索引对了，但 kernel 跑得快不快，九成取决于这些线程<em>以什么模式</em>
              touch 显存——访存为王。
            </>,
          )}
        </p>
      </Section>
    </>
  )
}
