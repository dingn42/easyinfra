import { Callout, CodeBlock, Figure, MathTex, Quiz, Section, Term } from '@/components/ui'
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

const LAUNCH = `int blockSize = 256;                              // 每个 block 256 个线程
int numBlocks = (n + blockSize - 1) / blockSize;  // 向上取整：⌈n / 256⌉
add<<<numBlocks, blockSize>>>(dA, dB, dC, n);     // 启动！`

const HOST = `#define CUDA_CHECK(call)                                               \\
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

const MAT_2D = `dim3 block(16, 16);                       // 每个 block 16×16 = 256 线程
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
  return (
    <>
      <p>
        如果你写了十年 CPU 代码，第一次读 CUDA 程序时最不适应的不是哪个新关键字，而是<strong>有个东西不见了：循环</strong>。
        给两个一百万元素的数组做加法，CPU 程序员的本能是写 <code>for (i = 0; i &lt; n; i++)</code>，
        让一个核心吭哧吭哧跑一百万圈。CUDA 的第一课是把这个 for 循环删掉——循环体留下，循环本身消失，
        取而代之的是<strong>一百万个线程同时各执行一次循环体</strong>。「第 i 次迭代」变成了「第 i 号线程」，
        迭代变量 <code>i</code> 不再由 <code>i++</code> 产生，而是每个线程问硬件：「我是谁？」这一章就讲清楚这个模型：
        线程怎么编号、为什么要分层组织、以及一段 CUDA 程序从 CPU 到 GPU 再回来的完整旅程。
      </p>

      <Section
        index={1}
        title="第一个 kernel：向量加法"
        lead="六行代码，包含了 CUDA 编程模型最核心的三个约定。"
      >
        <p>
          在 CUDA 里，跑在 GPU 上的函数叫{' '}
          <Term t="kernel（核函数）">
            由 host 启动、在 GPU 上被成千上万个线程同时执行的函数。每个线程执行同一份代码，靠自己的索引区分要处理的数据。
          </Term>
          。下面是 CUDA 世界的 “Hello World”——向量加法 <code>C = A + B</code>：
        </p>
        <CodeBlock code={VEC_ADD} lang="cuda" title="vector_add.cu — kernel 部分" highlight={[2, 3]} />
        <p>逐行拆开看：</p>
        <ul>
          <li>
            <strong>第 1 行</strong>：<code>__global__</code> 是 CUDA 对 C++ 的扩展修饰符，意思是「这个函数由 CPU
            侧调用、在 GPU 上执行」。它的返回值必须是 <code>void</code>——几万个线程同时返回值给谁呢？结果只能写进显存里带回来。
          </li>
          <li>
            <strong>第 2 行（高亮）</strong>：整个 CUDA 模型的灵魂。每个线程跑到这里时，从内建变量里读出「我是谁」，
            算出一个全局唯一的编号 <code>i</code>。注意这行代码所有线程都执行，但每个线程算出的 <code>i</code> 不同——
            这正是「同一份代码、不同的数据」的实现方式。
          </li>
          <li>
            <strong>第 3 行（高亮）</strong>：边界检查。看似多余，实则保命，下面单独说。
          </li>
          <li>
            <strong>第 4 行</strong>：循环体本体。原来 for 循环里的那行代码原封不动搬过来，只是 <code>i</code> 的来源变了。
          </li>
        </ul>
        <h3>「我是谁」：拆解索引公式</h3>
        <p>
          CUDA 把线程组织成两层（下一节细讲）：若干个 <strong>block</strong>，每个 block 里有若干个线程。三个内建变量
          (built-in variable) 拼出全局编号：<code>blockIdx.x</code> 是「我在第几个 block」，<code>blockDim.x</code>{' '}
          是「每个 block 有多少线程」，<code>threadIdx.x</code> 是「我是 block 内的第几号」。于是：
        </p>
        <MathTex block tex="i \;=\; \underbrace{\texttt{blockIdx.x} \times \texttt{blockDim.x}}_{\text{我所在 block 的起点}} \;+\; \underbrace{\texttt{threadIdx.x}}_{\text{块内偏移}}" />
        <p>
          这和「楼号 × 每层户数 + 门牌号」是同一个算术：先跳过前面所有 block 占据的索引区间，再加上自己在块内的位置。
          下图里 block 2 的 3 号线程算出 <code>i = 2 × 8 + 3 = 19</code>：
        </p>
        <Figure caption="全局索引 = block 起点（cyan）+ 块内偏移（amber）。每个线程做一次这个乘加，就拿到了属于自己的那份数据。">
          <IndexFigure />
        </Figure>
        <h3>为什么必须 if (i &lt; n)</h3>
        <p>
          线程不是按 1 个的粒度启动的，而是按 block 的粒度启动。假设 <code>n = 1000</code>、每 block 256 线程，
          你不可能启动「3.9 个 block」，只能向上取整启动 4 个，于是实际有 1024 个线程在跑——比数据多出 24 个。
          这 24 个线程算出的 <code>i</code> 是 1000 到 1023，如果不拦住它们，<code>C[i] = A[i] + B[i]</code>{' '}
          就是一次实打实的<strong>越界读写</strong>：轻则数据损坏，重则整个 kernel 报{' '}
          <code>illegal memory access</code> 崩掉。所以那行 <code>if</code> 不是防御性编程的洁癖，
          而是「线程数按 block 取整、几乎永远多于数据量」这一结构性事实的必然要求。几乎每一个生产 kernel
          的开头都站着这么一行边界检查。
        </p>
      </Section>

      <Section
        index={2}
        title="线程层级：thread → block → grid"
        lead="为什么是两层而不是一层？这个设计决定了 CUDA 程序天然可扩展。"
      >
        <p>
          一次 kernel 启动产生的所有线程统称一个 <strong>grid（网格）</strong>。grid 由若干{' '}
          <strong>block（线程块）</strong>组成，每个 block 由若干 <strong>thread（线程）</strong>组成。
          启动语法把这两层参数写在标志性的三对尖括号里：
        </p>
        <CodeBlock code={LAUNCH} lang="cuda" title="kernel 启动语法" />
        <p>
          <code>&lt;&lt;&lt;numBlocks, blockSize&gt;&gt;&gt;</code> 读作「启动 numBlocks 个 block，每个 block 含
          blockSize 个线程」。第一个问题自然是：为什么要费劲分两层？让我直接写「启动 1048576 个线程」不行吗？
          答案藏在两层各自的能力差异里：
        </p>
        <ul>
          <li>
            <strong>block 内：线程是「同事」。</strong>同一个 block 的线程保证在同一个{' '}
            <Term t="SM（Streaming Multiprocessor）">
              GPU 的基本计算单元，含执行单元、寄存器堆和 shared memory。一个 block 从启动到结束都钉在一个 SM 上。
            </Term>{' '}
            上执行，因此它们可以通过 <code>__syncthreads()</code> 互相等待（路障同步，barrier），还能共用一块
            SM 内的高速缓存——
            <Term t="shared memory（共享内存）">
              位于 SM 片上的可编程缓存，约百 KB 量级，延迟比显存低一个数量级。第 4 章的主角。
            </Term>
            ，用来交换中间结果。
          </li>
          <li>
            <strong>block 间：彼此是「陌生人」。</strong>CUDA 明确规定 block 之间不能同步、不保证执行顺序、
            甚至不保证同时存在——可能先跑完 50 个 block，再调度下 50 个。
          </li>
        </ul>
        <Callout type="insight" title="block 间独立不是限制，而是可扩展性的来源">
          <p>
            正因为 block 互相独立，硬件调度器才可以把它们<strong>当作一袋互不相干的任务</strong>，哪个 SM
            空了就扔一个过去。同一份编译产物，在 10 个 SM 的笔记本 GPU 上跑，硬件一次摆 10 个 block；
            在 132 个 SM 的 H100 上跑，一次摆 132 个——程序一行不改，性能随 SM 数量自动伸缩。
            这就是 CUDA 所谓的「透明可扩展性（transparent scalability）」：你描述「有哪些独立的工作」，
            硬件决定「怎么铺开」。反过来，一旦你偷偷假设了 block 间的执行顺序（比如 block 1 等 block 0
            写好的数据），程序就会在某些卡上死锁或出错——独立性是你和硬件之间的契约。
          </p>
        </Callout>
        <p>
          两层结构正好映射到硬件的两层现实：block 内的协作能力来自「钉死在一个 SM 上」这个物理事实；block
          间的独立性来自「SM 数量因卡而异」这个商业事实。一层太扁（百万线程谁跟谁协作？），三层太繁，
          两层刚好把「能协作的范围」和「能并行的规模」切开。顺带一提：block 内的线程在硬件上还会被切成 32
          个一组的 <Term t="warp（线程束）">硬件调度的最小单位，32 个线程锁步执行同一条指令。上一章的主角。</Term>
          ，这是硬件视角的第三层，但它不出现在编程模型的启动语法里——你声明 thread 和 block，硬件自己去切 warp。
        </p>
        <Quiz
          question="为什么 CUDA kernel 里几乎总要写 if (i < n) 边界检查？"
          options={[
            {
              text: '因为 GPU 的浮点运算不精确，需要用 if 过滤掉错误结果',
              explain: '边界检查与浮点精度毫无关系，它只关心索引 i 是否落在合法的数据范围内。',
            },
            {
              text: '因为线程按 block 的粒度整批启动，grid 大小向上取整后，线程总数几乎总是大于数据量 n，多出来的线程索引会越界',
              correct: true,
              explain:
                '正解。gridDim = ⌈n / blockDim⌉，只要 n 不是 blockDim 的整数倍，最后一个 block 里就有线程拿到 ≥ n 的索引。不拦住它们就是越界读写——这是结构性问题，不是编程风格问题。',
            },
            {
              text: '为了让编译器能更好地优化分支预测',
              explain: 'GPU 没有 CPU 那种分支预测器。这行 if 是正确性需求，不是性能技巧。',
            },
            {
              text: '只有调试时需要，发布版可以去掉以提升性能',
              explain: '只要 n 不能保证整除 blockDim，去掉边界检查的「发布版」就是一个带越界写的程序，随时可能崩溃或写坏别人的数据。',
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title="实验：Grid/Block 配置器"
        lead="gridDim 怎么算、warp 怎么切、浪费的线程藏在哪——拖一拖就有手感了。"
      >
        <p>
          配置 launch 参数是每个 CUDA 程序员的日常：blockDim 通常选 128～256（必须是 32 的倍数，否则最后一个
          warp 半满，纯浪费），gridDim 用 <code>(n + blockSize - 1) / blockSize</code> 这个经典的整数向上取整算出来。
          下面的配置器把这笔账摆在台面上：每个小格是一个线程，深浅条纹标出 warp 的切分，rose
          色是越界的浪费线程。<strong>点击任意线程格</strong>，右侧面板会演算它的索引推导。
        </p>
        <GridConfigurator />
        <p>
          值得注意的是浪费线程的占比：N=1000、blockDim=256 时只浪费 24/1024 ≈ 2.3%，无伤大雅；但若 N=16、blockDim=1024，
          浪费高达 98%——整卡只有 16 个线程在干活。这也是为什么「数据量太小的活不值得下发到 GPU」：
          launch 配置救不了本来就喂不饱机器的问题。另一个直觉：blockDim 不是越大越好，它影响的是协作范围和占用率（第
          5 章细讲），对「能不能算对」没有影响——算对靠的是索引公式和边界检查。
        </p>
        <Quiz
          question="N = 1000、blockSize = 256，启动这个 vecAdd kernel：gridDim 取多少？浪费多少个线程？"
          options={[
            {
              text: 'gridDim = 3，浪费 0 个',
              explain: '3 × 256 = 768 < 1000，会有 232 个元素没人处理——结果直接算错。必须向上取整。',
            },
            {
              text: 'gridDim = 4，浪费 0 个',
              explain: 'gridDim = 4 对了，但 4 × 256 = 1024 > 1000，最后一个 block 里索引 1000–1023 的 24 个线程过不了边界检查，是空转的。',
            },
            {
              text: 'gridDim = 4，浪费 24 个',
              correct: true,
              explain:
                '⌈1000 / 256⌉ = 4 个 block，共 1024 个线程；1024 − 1000 = 24 个线程的 i ≥ 1000，被 if (i < n) 拦下，空转但无害。这 2.3% 的浪费是按 block 整批启动的固有代价。',
            },
            {
              text: 'gridDim = 5，浪费 280 个',
              explain: '5 个 block 是 1280 个线程，能算对（多出的 280 个被边界检查拦住），但比必要的 4 个 block 多浪费了一整个 block 的调度开销。',
            },
          ]}
        />
      </Section>

      <Section
        index={4}
        title="实验：vecAdd 执行模拟器"
        lead="把 N=24、blockSize=8 的玩具例放慢一万倍，看每一步发生了什么。"
      >
        <p>
          公式看懂了，再看执行。下面的模拟器把一次 <code>add&lt;&lt;&lt;3, 8&gt;&gt;&gt;</code>{' '}
          启动放慢成可以单步的动画：每一步里，当前 block 的 8 个线程<strong>同时</strong>做同一件事——同时算索引、
          同时过边界检查、同时读 A 和 B（cyan 闪烁）、同时写 C（volt 闪烁）。注意没有任何一个线程在「循环」，
          循环被空间化成了 24 个并排的格子。
        </p>
        <VecAddSim />
        <p>
          重点玩「乱序调度」按钮：它随机打乱 3 个 block 的执行顺序重放一遍。block 2 先跑、block 0
          垫底？C 的结果分毫不差。这正是上一节那条契约的可视化——每个元素只被「自己的」那个线程碰一次，
          线程之间没有任何数据依赖，所以调度顺序无关紧要。真实 GPU 上这种乱序时刻都在发生：哪个 SM 先空出来，
          哪个 block 就先上。<strong>永远不要假设 block 间的执行顺序</strong>，这句话值得在心里默念三遍。
        </p>
      </Section>

      <Section
        index={5}
        title="host 侧五部曲：一段完整的 CUDA 程序"
        lead="kernel 只是冰山一角，水面下是显存管理和数据搬运。"
      >
        <p>
          GPU 有自己独立的显存，CPU（host）的指针在 GPU（device）上没有意义，反之亦然。所以每个 CUDA
          程序的 host 侧都绕不开同一套节奏，江湖人称五部曲：<strong>分配显存 → 拷入 → 启动 → 拷回 → 释放</strong>。
        </p>
        <CodeBlock code={HOST} lang="cuda" title="vector_add.cu — host 部分" highlight={[31, 35]} />
        <p>有三个容易栽跟头的点：</p>
        <ul>
          <li>
            <strong>kernel 启动是异步的（第 31 行）。</strong>
            <code>add&lt;&lt;&lt;...&gt;&gt;&gt;</code> 这行只是把任务<em>提交</em>给 GPU，CPU 不等它跑完就继续执行下一行。
            这是好事——CPU 可以趁机准备下一批数据——但也意味着你在 launch 之后立刻读结果是读不到的。
            想显式等待用 <code>cudaDeviceSynchronize()</code>。
          </li>
          <li>
            <strong>cudaMemcpy 是隐式同步点（第 35 行）。</strong>D2H 拷贝必须等 kernel 把 C
            写完才开始搬，所以这一行天然「等」了 kernel 结束。很多入门程序没写任何显式同步也能算对，全靠它兜底——知其然也要知其所以然，
            等你换成异步拷贝（<code>cudaMemcpyAsync</code>）时这层免费保险就没了。
          </li>
          <li>
            <strong>每个 CUDA API 调用都返回错误码，必须检查。</strong>CUDA 的错误是「粘性」的而且经常延迟暴露：kernel
            里的越界访问往往要到下一次同步操作才报出来，不查错误码的程序会在错误现场的十万八千里外神秘崩溃。
            所以所有人都用一个 <code>CUDA_CHECK</code> 宏把每个调用包起来（见代码开头），出错立刻带着文件名和行号停下。
            这不是可选的好习惯，是 CUDA 工程的及格线。
          </li>
        </ul>
        <Callout type="note" title="性能视角的预告">
          <p>
            五部曲里最贵的常常不是计算而是②和④的 PCIe 搬运：PCIe 4.0 x16 实测约 25 GB/s 上下，而 H100 显存带宽约
            3.35 TB/s，差出两个数量级。vecAdd 这种「搬 12 字节算 1 次加法」的 kernel，端到端时间几乎全花在搬运上。
            怎么省搬运、怎么让计算和搬运重叠，是第 4 章和第 6 章的主线。
          </p>
        </Callout>
      </Section>

      <Section
        index={6}
        title="多维索引：用 2D grid 处理矩阵"
        lead="数据是几维的，线程坐标就可以是几维的——纯粹为了让代码贴着数据长。"
      >
        <p>
          处理图像、矩阵这类二维数据时，把它们手动压扁成一维再算索引当然可行，但 CUDA 提供了更顺手的写法：grid 和
          block 本身可以是 1D、2D 或 3D 的，用 <code>dim3</code> 类型声明。索引公式逐维独立，每一维都是熟悉的「block
          起点 + 块内偏移」：
        </p>
        <CodeBlock code={MAT_2D} lang="cuda" title="matrix_add.cu — 2D 索引" highlight={[6, 7, 9]} />
        <p>
          想象一张 H×W 的图被 16×16 的瓷砖（block）铺满：<code>blockIdx.x/y</code> 是瓷砖的列号和行号，
          <code>threadIdx.x/y</code> 是瓷砖内的局部坐标，两维分别做乘加就得到全局的 <code>col</code> 和{' '}
          <code>row</code>。边界检查也要逐维做——图像宽高一般都不是 16 的整数倍，右边缘和下边缘的瓷砖都会探出去一截。
          最后一行 <code>idx = row * W + col</code> 提醒我们：多维只是编程模型给的语法糖，显存永远是一维的字节序列，
          2D 坐标终究要换算回线性地址（行优先，row-major）。
        </p>
        <Figure caption="dim3 block(4,4) × dim3 grid(3,2) 铺满 12×8 的图像。cyan 框是 blockIdx=(1,1) 的瓷砖，volt 格是其中 threadIdx=(2,1) 的线程，负责像素 (row 5, col 6)。">
          <Grid2DFigure />
        </Figure>
        <p>
          约定俗成的坑：<code>x</code> 是横向（列），<code>y</code> 是纵向（行），所以 <code>row</code> 对应{' '}
          <code>y</code>、<code>col</code> 对应 <code>x</code>，和矩阵下标 <code>(row, col)</code>{' '}
          的书写顺序正好相反。另外这个映射不只是好看——让 <code>threadIdx.x</code> 沿着内存连续的方向（行内）走，
          相邻线程就会访问相邻地址，这是下一章「合并访存」的伏笔：索引方式选错，性能可以差出几倍到一个数量级。
        </p>
      </Section>

      <Section index={7} title="总结与延伸阅读">
        <p>这一章建立的是 CUDA 的世界观，所有后续优化都长在这几条直觉上：</p>
        <ul>
          <li>
            <strong>循环消失了</strong>：for 循环被空间化成海量线程，每个线程用{' '}
            <code>blockIdx.x * blockDim.x + threadIdx.x</code> 认领自己的那份数据。
          </li>
          <li>
            <strong>边界检查不可省</strong>：线程按 block 整批启动、grid 向上取整，线程数几乎总是多于数据量，
            越界线程必须被 <code>if (i &lt; n)</code> 拦住。
          </li>
          <li>
            <strong>两层层级各司其职</strong>：block 内可同步、可共享内存（协作单位）；block
            间完全独立（调度单位）——独立性换来了程序随 SM 数量透明扩展。
          </li>
          <li>
            <strong>host 五部曲</strong>：cudaMalloc → H2D → launch → D2H → cudaFree；launch 异步、cudaMemcpy
            隐式同步、每个调用都要 <code>CUDA_CHECK</code>。
          </li>
          <li>
            <strong>多维索引是语法糖</strong>：dim3 让线程坐标贴着数据形状长，但显存永远是一维的。
          </li>
        </ul>
        <p>延伸阅读，按「上手快 → 钻得深」排序：</p>
        <ul>
          <li>
            <a href="https://developer.nvidia.com/blog/even-easier-introduction-cuda/" target="_blank" rel="noreferrer">
              An Even Easier Introduction to CUDA（NVIDIA 开发者博客）
            </a>
            —— Mark Harris 的经典入门文，用统一内存简化五部曲，半小时就能跑通本章代码。
          </li>
          <li>
            <a
              href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model"
              target="_blank"
              rel="noreferrer"
            >
              CUDA C++ Programming Guide — Ch.2 Programming Model（NVIDIA 官方文档）
            </a>
            —— 本章所有概念的权威出处，thread hierarchy 一节值得逐字读。
          </li>
          <li>
            <a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41487/" target="_blank" rel="noreferrer">
              How CUDA Programming Works（GTC 2022, Stephen Jones）
            </a>
            —— NVIDIA CUDA 架构师亲述编程模型背后的硬件动机，是「为什么这么设计」的最佳答案。
          </li>
          <li>
            <a
              href="https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors"
              target="_blank"
              rel="noreferrer"
            >
              Programming Massively Parallel Processors（PMPP, 4th ed.）Ch.2–3
            </a>
            —— Hwu & Kirk 的教科书，第 2、3 章用几十页把本章内容展开到每个细节，配大量习题。
          </li>
          <li>
            <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html" target="_blank" rel="noreferrer">
              CUDA Runtime API — Error Handling（NVIDIA 官方文档）
            </a>
            —— 错误码的完整语义，理解「粘性错误」和异步报错为什么会延迟暴露。
          </li>
        </ul>
        <p>
          下一章我们沿着这条线往下走：线程有了、索引对了，但 kernel 跑得快不快，九成取决于这些线程<em>以什么模式</em>
          touch 显存——访存为王。
        </p>
      </Section>
    </>
  )
}
