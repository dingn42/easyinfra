import { Callout, ChapterLink, CodeBlock, HardwareBaseline, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { DieAreaFigure } from './DieAreaFigure'
import { ThroughputRace } from './ThroughputRace'
import { TransistorBudget } from './TransistorBudget'

const VEC_ADD_CPU_EN = `// CPU mindset: one worker, doing the whole thing in order
void vec_add(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];   // note: iteration i does not depend on i-1 at all
  }
}`
const VEC_ADD_CPU_ZH = `// CPU 思维：一个工人，从头到尾按顺序做完
void vec_add(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];   // 注意：第 i 次迭代完全不依赖第 i-1 次
  }
}`

const VEC_ADD_GPU_EN = `// GPU mindset: hire n workers, each owns exactly one element
// Below is one worker's entire job (Chapter 3 introduces this syntax properly)
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // which worker am I?
  if (i < n) c[i] = a[i] + b[i];                  // do only my own share
}`
const VEC_ADD_GPU_ZH = `// GPU 思维：雇 n 个工人，每人只负责一个元素
// 下面是"第 i 号工人"的全部工作（第 3 章会正式介绍这套语法）
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 我是第几号工人？
  if (i < n) c[i] = a[i] + b[i];                  // 只做属于我的那一份
}`

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            Training a GPT-4-class model burns through roughly 1×10²⁵{' '}
            <Term t="FLOPs">
              Floating-point Operations — a count of arithmetic done. Don&apos;t confuse it with FLOPS
              (operations per second, a measure of speed): the former is the amount of work, the latter is how fast
              you do it.
            </Term>{' '}
            (floating-point operations). The number is too large to feel, so convert it: a top-tier server CPU
            peaks at a few TFLOPS (trillions per second), and even running flat out, sleepless, never making a
            mistake, it would take tens of thousands of years. In reality the training ran on tens of thousands of
            GPUs for about three months. Same silicon, same transistor process, single-chip power in the same
            ballpark — yet throughput differs by five or six orders of magnitude. More counterintuitive still: the
            gap does not come from &ldquo;GPU cores being faster.&rdquo; Quite the opposite — a single GPU core is
            slower and dumber than a CPU core. This chapter writes not one line of CUDA; it drives home a single
            point: what kind of machine a GPU actually is, where its speed comes from, and when it loses.
          </>,
          <>
            训练一个 GPT-4 量级的大模型，大约要消耗 1×10²⁵ 次浮点运算（
            <Term t="FLOPs">
              Floating-point Operations，浮点运算次数。注意与 FLOPS（每秒浮点运算次数，衡量算力）区分：前者是
              「工作量」，后者是「干活速度」。
            </Term>
            ，floating-point operations）。这个数字大到失去直觉，不妨换算一下：一颗顶级服务器 CPU
            的峰值算力约为几个 TFLOPS（每秒万亿次），就算它满负荷、不眠不休、永不出错，也要连续算上几万年；
            而现实中的训练，是数万张 GPU 跑了大约三个月。同样的硅片、同样的晶体管工艺、单芯片功耗也在同一个
            数量级，吞吐却差出五六个数量级。更反直觉的是：这差距并不来自「GPU 的核心更快」——恰恰相反，GPU
            的单个核心比 CPU 的核心更慢、更笨。这一章我们一行 CUDA 都不写，只把一件事讲透：GPU
            到底是一台什么样的机器，它的快从哪里来，又会在什么时候输。
          </>,
        )}
      </p>

      <HardwareBaseline
        ids={['a100', 'h100']}
        note={t(
          'A100 is the running example; H100 appears for contrast.',
          '以 A100 为主线示例，H100 用于对照。',
        )}
      />

      <Section
        index={1}
        title={t('Two design philosophies: the latency machine and the throughput machine', '两种设计哲学：延迟机器与吞吐机器')}
        lead={t(
          'Given the same transistor budget, the CPU and the GPU hand in two completely opposite answers.',
          '同样一笔晶体管预算，CPU 和 GPU 交出了两份完全相反的答卷。',
        )}
      >
        <p>
          {t(
            <>
              The CPU is designed to <strong>finish a single task as fast as possible</strong> — to minimize
              latency. To that end it packs the chip with circuitry that does no arithmetic at all: multi-level
              caches, tens of megabytes of them, to hide how slow memory is; a branch predictor that bets which
              way an if/else will go and pockets dozens of cycles when it bets right; an out-of-order execution
              engine that tears instructions apart and reorders them so whichever has its data ready goes first;
              plus speculative execution, prefetchers, and more. The entire machine exists for one purpose: to
              keep those few cores from <strong>ever stalling to wait</strong>. The cost is blunt — on a modern
              flagship CPU die, the ALUs (Arithmetic Logic Units) that actually do the arithmetic occupy only a
              small patch; the vast majority of the area goes to cache and control logic.
            </>,
            <>
              CPU 的设计目标是<strong>把单个任务尽快做完</strong>，也就是最小化延迟（latency）。为了这个目标，
              它在芯片上堆了大量「不直接做计算」的电路：动辄几十 MB 的多级缓存（cache）用来掩盖内存的慢；
              分支预测器（branch predictor）赌 if/else 会往哪边走，赌对了就白赚几十个周期；乱序执行
              （out-of-order execution）引擎把指令拆开重排，谁的数据先就绪谁先上；还有投机执行、预取器……
              这一整套机器的唯一目的，是让那少数几个核心<strong>永远不要停下来等</strong>。代价也很直白：在一颗
              现代旗舰 CPU 的 die（裸片）上，真正做算术的 ALU（Arithmetic Logic Unit，算术逻辑单元）
              只占很小一块，绝大部分面积都给了缓存和控制逻辑。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The GPU&apos;s answer is the exact opposite: its goal is to <strong>maximize total tasks completed
              per unit time</strong> — throughput. A single core is allowed to be slow and dumb (no reordering,
              almost no branch prediction, a clock only a third of the CPU&apos;s) as long as there are many of
              them. An H100 has 16,896 FP32 cores spread across 132 SMs (Streaming Multiprocessors); all the
              control logic and cache area it saved is converted into ALUs. A task on a single GPU core might take
              eight times as long as on a single CPU core, but once there are enough tasks, two thousand slow
              workers against eight fast ones is still a rout.
            </>,
            <>
              GPU 的答卷完全相反：它的目标是<strong>单位时间内完成的任务总量最大</strong>，也就是吞吐
              （throughput）。单个核心可以慢、可以笨——不做乱序、几乎不做分支预测、主频只有 CPU 的三分之一
              ——但要多。一颗 H100 上有 16896 个 FP32 核心，分布在 132 个 SM（Streaming Multiprocessor，
              流式多处理器）里；省下来的控制逻辑和缓存面积，全部换成了 ALU。一个任务在 GPU 单核上可能要花
              CPU 单核八倍的时间，但只要任务足够多，两千个慢工人对八个快工人，依然是碾压。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              A familiar analogy: the CPU is a world-class surgeon, the GPU is ten thousand assembly-line workers.
              A delicate operation — every step depending on the last, the plan changing on the fly with the
              bleeding — can only go to the surgeon; ten thousand workers crowding in just get in the way. But if
              the job is &ldquo;tighten one screw on each of a million parts,&rdquo; the worker army wins hands
              down, even if each worker tightens a screw at a fraction of the surgeon&apos;s speed. The analogy
              also spells out the GPU&apos;s <strong>precondition</strong>: the work must split into a great many
              independent pieces — done simultaneously, not waiting on each other, not constantly swapping
              intermediate results. When that precondition fails, the GPU&apos;s advantage evaporates instantly —
              which we&apos;ll verify with an experiment in a moment.
            </>,
            <>
              一个常用的类比：CPU 是一位顶尖外科医生，GPU 是一万个流水线工人。一台精密手术——每一步都依赖
              上一步的结果、随时要根据出血情况改变方案——只能交给医生，一万个工人围上来反而碍事。但如果活儿是
              「给一百万个零件各拧一颗螺丝」，工人军团完胜，哪怕每个工人拧一颗螺丝的速度只有医生的几分之一。
              这个类比同时也交代了 GPU 的<strong>适用前提</strong>：任务必须能拆成大量彼此独立的小块——可以
              同时做、互相不等待、不需要频繁交换中间结果。前提不成立时，GPU 的优势会瞬间消失，这一点我们马上
              用实验验证。
            </>,
          )}
        </p>
        <DieAreaFigure />
        <Callout type="insight" title={t('A GPU is not a “faster CPU”', 'GPU 不是「更快的 CPU」')}>
          <p>
            {t(
              <>
                The two optimize fundamentally different quantities: the CPU minimizes <strong>the time to finish
                a single task</strong> (latency), the GPU maximizes <strong>the total tasks finished per unit
                time</strong> (throughput). So &ldquo;how many times faster is a GPU than a CPU&rdquo; is an
                ill-posed question — first ask whether your problem contains tens of thousands of small chores
                that can run at once without waiting on each other. If it does, the GPU is two orders of magnitude
                faster; if it doesn&apos;t, the GPU may be slower than your laptop.
              </>,
              <>
                两者优化的根本就不是同一个量：CPU 最小化<strong>单个任务的完成时间</strong>（延迟），GPU 最大化
                <strong>单位时间完成的任务总量</strong>（吞吐）。所以「GPU 比 CPU 快多少倍」这个问题本身是不完整
                的——先问你的问题里有没有几万个可以同时做、互不等待的小活。有，GPU 快两个数量级；没有，GPU
                可能比你的笔记本还慢。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title={t('Experiment: the throughput race', '实验：吞吐量赛道')}
        lead={t(
          '8 fast cores against 2048 slow ones — who finishes first? Drag the task count and find the crossover yourself.',
          '8 个快核对 2048 个慢核，谁先把活干完？拖动任务数，亲手找到分界点。',
        )}
      >
        <p>
          {t(
            <>
              The rules are simple: the CPU has 8 cores, each task takes 1 µs; the GPU has 2048 cores, each k×
              slower (8× by default), plus a fixed 20 µs launch overhead (the real-world kernel-launch cost —
              handing work to the GPU costs time in itself). All tasks are independent. First drag the task count
              down to the minimum of 64 and see who wins, then push it to a million and see how wide the gap gets;
              also crank the GPU per-core slowdown all the way to 32× and see whether &ldquo;many cores&rdquo; can
              hold the line against &ldquo;slow cores.&rdquo;
            </>,
            <>
              规则很简单：CPU 有 8 个核，每个任务耗时 1 µs；GPU 有 2048 个核，单核慢 k 倍（默认 8×），
              外加一笔固定的 20 µs 启动开销（对应真实世界里的 kernel launch 开销——把活儿派给 GPU
              本身就要花时间）。所有任务彼此独立。先把任务数拖到最小的 64 个看看谁赢，再拖到一百万个看看
              差距能拉到多大；也试试把 GPU 单核慢倍数拉满到 32×，看看「核多」能不能撑住「核慢」。
            </>,
          )}
        </p>
        <ThroughputRace index={1} />
        <p>
          {t(
            <>
              You should have noticed three things. First, when tasks are few, the CPU wins: 64 tasks take it 8 µs
              to finish, while the GPU spends 20 µs just on launch — bringing a sledgehammer to crack a nut, and
              the nut is cracked before you&apos;ve even raised the hammer. Second, when tasks are plentiful the
              GPU dominates, and the more tasks there are the closer the speedup approaches its theoretical limit
              (2048 ÷ 8 ÷ slowdown). Third, the GPU&apos;s progress bar jumps a notch at a time: 2048 cores start
              together and finish together, advancing one &ldquo;wave&rdquo; at a time — a notion that arrives
              formally in <ChapterLink n={2} /> as the wave/warp.
            </>,
            <>
              你应该观察到了三件事。第一，任务很少时 CPU 赢：64 个任务它 8 µs 就干完了，GPU 光启动就要
              20 µs——杀鸡用牛刀，牛刀抬起来的功夫鸡已经杀完了。第二，任务足够多时 GPU 碾压，而且任务越多
              加速比越接近理论极限（2048 ÷ 8 ÷ 慢倍数）。第三，GPU 的进度条是一格一格跳的：2048
              个核同时开工、同时交活，一「波」一波地推进——这个「波」的概念在<ChapterLink n={2} />会以 wave/warp
              的形式正式登场。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              One hidden condition is worth saying out loud: in this race every task is <strong>independent</strong>.
              The moment a program contains a serial part that can&apos;t be parallelized, the speedup hits a
              ceiling — that&apos;s Amdahl&apos;s Law. Let s be the serial fraction of the total work and P the
              number of processors; then the overall speedup is
            </>,
            <>
              还有一个隐藏条件值得说破：这场赛跑里所有任务<strong>彼此独立</strong>。一旦程序里存在无法并行的
              串行部分，加速比就有天花板——这就是 Amdahl 定律（Amdahl&apos;s Law）。设串行部分占总工作量的
              比例为 s，处理器数量为 P，则总加速比
            </>,
          )}
        </p>
        <MathTex block tex="\text{Speedup} \le \dfrac{1}{s + \dfrac{1 - s}{P}} \xrightarrow{P \to \infty} \dfrac{1}{s}" />
        <p>
          {t(
            <>
              Even if just 5% of the work must run serially, no matter how many tens of thousands of cores you pile
              on, the speedup can never exceed 20×. That is why the first question about &ldquo;can I use a
              GPU&rdquo; is always &ldquo;how much parallelism does my problem have&rdquo; — and in the next section
              we look at deep learning&apos;s answer.
            </>,
            <>
              哪怕只有 5% 的工作必须串行执行，无论你堆多少万个核心，加速比也不可能超过 20×。这就是为什么
              「能不能用 GPU」的第一个问题永远是「你的问题里有多少并行度」——下一节我们就去看深度学习的答案。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'Which of these tasks is the worst fit for a GPU?',
            '下面哪类任务交给 GPU 最不划算？',
          )}
          options={[
            {
              text: t('Brighten each of a hundred million pixels by 20%', '把一亿个像素各自调亮 20%'),
              explain: t(
                'Each pixel&apos;s computation is completely independent — a textbook parallel task, the GPU&apos;s home turf.',
                '每个像素的计算完全独立，是教科书级的并行任务，GPU 的主场。',
              ),
            },
            {
              text: t(
                'Walk a linked list node by node: each step&apos;s address depends on the pointer read in the previous step',
                '沿一条链表逐结点查找：每一步的地址都取决于上一步读到的指针',
              ),
              correct: true,
              explain: t(
                'Correct. This is a hard serial dependency: until step i finishes, step i+1 doesn&apos;t even know where its data is, so out of tens of thousands of cores only one can work and the rest stand and watch. Add in the GPU&apos;s slower single core and weaker branch handling, and this kind of pointer-chasing task actually runs slower on a GPU than on a CPU.',
                '对。这是强串行依赖：第 i 步不做完，第 i+1 步连数据在哪都不知道，几万个核里只有一个能干活，' +
                  '其余全部围观。再叠加 GPU 单核本身更慢、分支处理更弱，这类指针追逐（pointer chasing）任务 GPU 反而比 CPU 慢。',
              ),
            },
            {
              text: t('Compute a 4096×4096 matrix multiply', '计算一个 4096×4096 的矩阵乘法'),
              explain: t(
                'Each output element of a matmul is an independent dot product, giving tens of millions of degrees of parallelism — exactly the workload a GPU is best at.',
                '矩阵乘的每个输出元素都是一次独立点积，有千万级并行度，是 GPU 最擅长的负载。',
              ),
            },
            {
              text: t('Compute one loss value for each of a million independent samples', '给一百万个独立样本各算一次损失函数'),
              explain: t(
                'The samples don&apos;t depend on each other — naturally parallel, and batch work like this is precisely the GPU&apos;s strength.',
                '样本之间互不依赖，天然并行，GPU 处理这类批量任务正是强项。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title={t('Where the parallelism comes from', '并行度从哪来')}
        lead={t(
          'Deep learning happens to be exactly the kind of problem that delivers parallelism by the ton.',
          '深度学习恰好是把并行度成吨送上门的那类问题。',
        )}
      >
        <p>
          {t(
            <>
              Start with the simplest example: vector addition. In c[i] = a[i] + b[i], each i reads only its own
              two inputs and writes its own one output, with no relationship to any other i. A length-one-million
              vector add is a million naturally independent little tasks. On a CPU we&apos;re used to writing it as
              a loop:
            </>,
            <>
              先看最简单的例子：向量加法。c[i] = a[i] + b[i]，每个 i 的计算只读自己的两个输入、写自己的一个
              输出，跟其他任何 i 都没有关系。一个长度一百万的向量加法，就是一百万个天然独立的小任务。在 CPU
              上我们习惯写成循环：
            </>,
          )}
        </p>
        <CodeBlock code={t(VEC_ADD_CPU_EN, VEC_ADD_CPU_ZH)} lang="cpp" title="vec_add.c" highlight={[4]} />
        <p>
          {t(
            <>
              GPU programming demands a <strong>shift in mindset</strong>: stop asking &ldquo;how do I make this
              loop run faster&rdquo; and start asking &ldquo;what is the work each element has to do.&rdquo; Lift
              the loop body out, write it as one worker&apos;s job description, then hire n workers to start at
              once — the loop disappears, replaced by an identity query: &ldquo;which worker am I?&rdquo;
            </>,
            <>
              GPU 编程要求的是一次<strong>思维转换</strong>：别再问「这个循环怎么跑得更快」，改问「每个元素
              要做的活是什么」。把循环体抠出来，写成一个工人的工作说明书，然后雇 n 个工人同时开工——循环消失了，
              取而代之的是「我是第几号工人」这个身份查询：
            </>,
          )}
        </p>
        <CodeBlock code={t(VEC_ADD_GPU_EN, VEC_ADD_GPU_ZH)} lang="cuda" title="vec_add.cu" highlight={[4, 5]} />
        <p>
          {t(
            <>
              Up another level: matrix multiply C = A·B. Each output element C[i][j] is the dot product of row i
              of A and column j of B — again completely independent computations. For an N×N matrix the total work
              is <MathTex tex="2N^3" /> floating-point operations (N multiply-adds per output element), while the
              number of independent tasks is <MathTex tex="N^2" />. Take N = 4096: about 1.4×10¹¹ operations,
              split into 16.7 million mutually independent dot products. That degree of parallelism is more than
              enough to feed tens of thousands of cores.
            </>,
            <>
              再上一个量级：矩阵乘法 C = A·B。输出矩阵的每个元素 C[i][j]，是 A 的第 i 行和 B 的第 j
              列的点积——又是彼此完全独立的计算。对于 N×N 的矩阵，总计算量是 <MathTex tex="2N^3" /> 次浮点
              运算（每个输出元素 N 次乘加），而独立任务数是 <MathTex tex="N^2" /> 个。取 N = 4096：约
              1.4×10¹¹ 次运算，拆成 1670 万个互不依赖的点积。这样的并行度，喂饱几万个核心绰绰有余。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The final inference follows naturally: the body of a neural network is just stacked matrix multiplies.
              A fully connected layer is a matmul, the heart of attention is a matmul, convolution can be cast as a
              matmul too — a single forward pass of a large model is essentially a few hundred big matmuls executed
              in sequence, each carrying tens of millions of degrees of parallelism inside. This class of problem,
              where the parallelism shows up on its own with no painstaking reworking, has a name of its own:
              embarrassingly parallel. The relationship between deep learning and the GPU is less a
              &ldquo;choice&rdquo; than a mutual making: the GPU finally got a workload that fills all its cores,
              and deep learning finally got a machine that can grind through its computation in a reasonable time.
            </>,
            <>
              最后一步推论就水到渠成了：神经网络的主体就是矩阵乘的堆叠。全连接层是矩阵乘，注意力机制
              （attention）的核心是矩阵乘，卷积也能化成矩阵乘——一次大模型的前向传播，本质上是几百个大矩阵乘
              排队执行，每一个内部都携带千万级的并行度。这类不需要费心改造、并行度自己送上门的问题，有个专门
              的称呼：embarrassingly parallel（「不好意思地并行」，意为并行得毫不费力）。深度学习和 GPU
              的关系与其说是「选择」，不如说是互相成就：GPU 等到了能填满它所有核心的负载，深度学习等到了能把
              它的计算量在合理时间内算完的机器。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'What is the precondition for a GPU to keep its thousands of weak cores fed and truly become a “throughput machine”?',
            'GPU 能把上千个弱核心都喂饱、真正成为「吞吐机器」的前提是什么？',
          )}
          options={[
            {
              text: t('A high enough clock speed', '主频足够高'),
              explain: t(
                'A GPU&apos;s clock is usually only about a third of a CPU&apos;s (roughly 1.5–2 GHz); its speed has never come from clock rate.',
                'GPU 主频通常只有 CPU 的三分之一左右（约 1.5～2 GHz），它的快从来不靠主频。',
              ),
            },
            {
              text: t(
                'Enough work that can run at once and is independent of itself (enough parallelism)',
                '程序里有足够多可以同时执行、彼此独立的工作（足够的并行度）',
              ),
              correct: true,
              explain: t(
                <>
                  Correct. Tens of thousands of cores means you need at least tens of thousands of independent tasks
                  (actually far more — you&apos;ll see why when <ChapterLink n={2} /> covers latency hiding) to fill
                  the machine. When parallelism falls short, all those extra cores can only idle.
                </>,
                <>
                  对。几万个核心意味着至少需要几万个（实际上远不止，<ChapterLink n={2} />讲延迟隐藏时你会看到需要更多）
                  互不依赖的任务才能填满机器。并行度不足时，再多的核也只能空转。
                </>,
              ),
            },
            {
              text: t('A big enough cache', '缓存做得足够大'),
              explain: t(
                'The GPU is precisely the side that trades cache area away for ALUs; it hides memory-access latency by switching among a huge number of threads, not with a big cache.',
                'GPU 恰恰是把缓存面积省下来换 ALU 的那一方；它靠海量线程切换而不是大缓存来掩盖访存延迟。',
              ),
            },
            {
              text: t('The code must be written in C++', '代码必须用 C++ 编写'),
              explain: t(
                'The language is irrelevant — PyTorch users write Python, and the matmuls still get dispatched to the GPU underneath. What matters is the structure of the workload itself.',
                '语言无关紧要——PyTorch 用户写的是 Python，底层照样把矩阵乘派给 GPU。关键在负载本身的结构。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={4}
        title={t('Experiment: the transistor budget allocator', '实验：晶体管预算分配器')}
        lead={t(
          'Now pretend you&apos;re the chip architect: 100% of the die area is in front of you — control, cache, ALU; how do you split it?',
          '现在假装你是芯片架构师：100% 的 die 面积摆在面前，控制、缓存、ALU，怎么分？',
        )}
      >
        <p>
          {t(
            <>
              This experiment turns that static figure from SEC 01 into something you can drag. The three sliders
              always sum to 100% — drag any one and the other two scale proportionally, just like real chip design:
              area given to cache can no longer go to ALUs. The two readouts on the right are the
              &ldquo;scores&rdquo; for the two design philosophies. The <strong>single-thread score</strong> comes
              from control logic and cache, but with diminishing returns — pushing cache from 30% to 50% lifts hit
              rate far less than going from 0% to 20% did, and branch prediction is the same: once accuracy nears
              98%, every further point costs multiplying transistors. The <strong>throughput score</strong> is
              strictly proportional to ALU count — one more core, one more unit of output; crude, linear, no
              diminishing returns.
            </>,
            <>
              这个实验把 SEC 01 的那张静态图变成可以拖的。三个滑杆的总和恒为 100%——拖动任何一个，另外两个会
              按比例缩放，就像真实的芯片设计：面积给了缓存，就不能再给 ALU。右侧两个读数对应两种设计哲学的
              「得分」：<strong>单线程性能分</strong>来自控制逻辑与缓存，但收益递减——把缓存从 30% 加到
              50%，命中率的提升远不如从 0% 加到 20% 来得猛，分支预测也一样，预测准确率逼近 98%
              之后每提高一点都要付出成倍的晶体管；<strong>吞吐分</strong>则严格正比于 ALU
              数量——多一个核就多一份产出，简单粗暴，没有递减。
            </>,
          )}
        </p>
        <TransistorBudget index={2} />
        <p>
          {t(
            <>
              Try the two extremes: pour everything into ALUs and the throughput score maxes out, but the
              single-thread score is dismal — this machine running serial code would get crushed by a phone. Pour
              everything into control and cache and the single-thread score can still only approach an asymptote,
              because diminishing returns lock the ceiling. The two preset buttons give exactly the typical
              trade-offs of the two real-world paths. One detail is worth noting: the &ldquo;Typical GPU&rdquo;
              preset does not put cache at 0% — a real GPU keeps a register file, shared memory, and a modest L2,
              whose roles are very different from CPU cache (more like a scratchpad the programmer manages by
              hand). That&apos;s a centerpiece of Chapters <ChapterLink n={2} label="2" /> and{' '}
              <ChapterLink n={4} label="4" />.
            </>,
            <>
              试试两个极端：全部砸给 ALU，吞吐分拉满，但单线程分惨不忍睹——这台机器跑串行代码会被手机吊打；
              全部砸给控制和缓存，单线程分也只能逼近渐近线，因为收益递减锁死了天花板。两个预设按钮给出的正是
              现实中两条路线的典型取舍。还有一个细节值得注意：「典型 GPU」预设里缓存并不是
              0%——真实 GPU 保留了寄存器堆、shared memory 和一块不大的 L2，它们的角色和 CPU
              缓存很不一样（更像程序员手动管理的便签纸），这是<ChapterLink n={2} />和<ChapterLink n={4} />的重头戏。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={5}
        title={t('Looking ahead: computing fast is half the battle, feeding fast is the other', '预告：算得快，还要喂得饱')}
        lead={t(
          'Piling ALUs sky-high solves only half the problem — if the data can&apos;t get in, all those cores just idle.',
          '把 ALU 堆上天只解决了一半问题——数据进不来，再多核心也只能空转。',
        )}
      >
        <p>
          {t(
            <>
              All chapter we&apos;ve assumed &ldquo;once a task reaches a core, it can compute,&rdquo; but in the
              real world the data sits in memory, and memory is far slower than arithmetic — a gap that widens
              year over year, as compute roughly more than doubles every two years while memory bandwidth and
              latency improve much more slowly. This is the famous memory wall; David Patterson distilled its more
              painful half into &ldquo;latency lags bandwidth&rdquo;: bandwidth can still be widened by piling on
              parallelism, but latency is nearly incurable.
            </>,
            <>
              本章我们一直假设「任务到了核心手里就能算」，但真实世界里数据躺在内存里，而内存比算术慢得多，
              且这个差距还在逐年拉大——算力大约每两年翻一倍以上，内存带宽和延迟的改善却慢得多。这就是著名的
              内存墙（memory wall）；David Patterson 把其中更扎心的那半句总结为「延迟落后于带宽」
              （latency lags bandwidth）：带宽还能靠堆并行做宽，延迟几乎无药可医。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The GPU&apos;s response is to equip itself with extremely wide memory: the A100&apos;s HBM2e (High
              Bandwidth Memory) delivers about 1.9–2.0 TB/s, the H100&apos;s HBM3 about 3.35 TB/s; for comparison,
              a desktop CPU&apos;s dual-channel DDR5 manages only around 100 GB/s — a 20–30× difference. But
              &ldquo;wide&rdquo; is not &ldquo;fast.&rdquo; Take a look at the latency magnitudes across the
              storage hierarchy (in GPU core clock cycles, all approximate):
            </>,
            <>
              GPU 的应对是给自己配上极宽的内存：A100 的 HBM2e（High Bandwidth Memory，高带宽内存）提供约
              1.9～2.0 TB/s 的带宽，H100 的 HBM3 约 3.35 TB/s；作为对比，桌面 CPU 的双通道 DDR5 大约只有
              100 GB/s——差了 20～30 倍。但「宽」不等于「快」，看一眼各级存储的延迟数量级（以 GPU 核心时钟
              周期计，均为约数）：
            </>,
          )}
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full min-w-[420px] border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="py-2 pr-4 font-mono text-[11px] font-medium uppercase tracking-wider text-ink2">{t('Operation', '操作')}</th>
                <th className="py-2 pr-4 font-mono text-[11px] font-medium uppercase tracking-wider text-ink2">{t('Latency (cycles)', '延迟（周期）')}</th>
                <th className="py-2 font-mono text-[11px] font-medium uppercase tracking-wider text-ink2">{t('Intuition', '直觉')}</th>
              </tr>
            </thead>
            <tbody className="text-text">
              <tr className="border-b border-line">
                <td className="py-2 pr-4">{t('One floating-point multiply-add (FMA)', '一次浮点乘加（FMA）')}</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-volt">~1</td>
                <td className="py-2 text-ink2">{t('the baseline unit', '基准单位')}</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4">{t('Shared memory / L1 hit', 'Shared memory / L1 命中')}</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-cyan">~30</td>
                <td className="py-2 text-ink2">{t('getting up to grab something a few steps away', '起身走几步去拿个东西')}</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4">{t('L2 cache hit', 'L2 缓存命中')}</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-cyan">~200</td>
                <td className="py-2 text-ink2">{t('walking to the next room to fetch it', '走去隔壁房间取一趟')}</td>
              </tr>
              <tr>
                <td className="py-2 pr-4">{t('DRAM / HBM access', 'DRAM / HBM 访问')}</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-amber">~400–600</td>
                <td className="py-2 text-ink2">
                  {t(
                    'going downstairs to the parcel locker — one trip is long enough to do hundreds of multiply-adds',
                    '下楼去快递柜——等一次的工夫够做几百次乘加',
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          {t(
            <>
              In the time it takes to wait for one trip to VRAM, a core could do hundreds of multiply-adds — if it
              just sat there waiting. Of course the GPU doesn&apos;t sit and wait; its solution is not to build a
              bigger cache like the CPU, but to have a huge number of threads cover for one another: while one
              batch of threads waits on data, it instantly switches to another batch and computes meanwhile,
              hiding latency with parallelism. How this latency-hiding mechanism works and how many threads it
              takes to hide the latency is the core of <ChapterLink n={2} />; how to organize access patterns so
              that 2 TB/s of bandwidth is actually saturated rather than squandered on scattered accesses is the
              entire subject of <ChapterLink n={4} />. Remember this chapter&apos;s conclusion:{' '}
              <strong>compute solves &ldquo;computing fast,&rdquo; the memory system decides &ldquo;whether you
              get fed&rdquo; — and the latter is the bottleneck for most real programs.</strong>
            </>,
            <>
              等一次显存的时间，核心足够做几百次乘加——如果它干等的话。GPU 当然不会干等，它的解法不是像 CPU
              那样修更大的缓存，而是用海量线程互相掩护：一批线程在等数据，就立刻切换到另一批先算着，用并行度把
              延迟「藏」起来。这套延迟隐藏（latency hiding）机制怎么运转、需要多少线程才藏得住，是<ChapterLink n={2} />的
              核心；而怎么组织访存模式让那 2 TB/s 的带宽真正跑满、而不是浪费在零碎的访问上，是<ChapterLink n={4} />的全部内容。记住本章的结论：<strong>算力解决「算得快」，内存系统决定「喂不喂得饱」——后者才是
              大多数真实程序的瓶颈。</strong>
            </>,
          )}
        </p>
      </Section>

      <Section index={6} title={t('Summary and further reading', '总结与延伸阅读')}>
        <p>
          {t(
            'This chapter wrote no code, but it is the foundation for every chapter that follows. Five takeaways:',
            '这一章没有写代码，但它是后面所有章节的地基。五个要点：',
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>The CPU is a latency machine</strong>: a few strong cores + a large cache + complex
                control logic (branch prediction, out-of-order execution), all to finish a single task as fast as
                possible.
              </>,
              <>
                <strong>CPU 是延迟机器</strong>：少量强核 + 大缓存 + 分支预测 / 乱序执行等复杂控制逻辑，一切为了
                单个任务尽快完成。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The GPU is a throughput machine</strong>: tens of thousands of simple slow cores, the
                transistor budget overwhelmingly poured into ALUs, trading quantity for total output per unit time.
              </>,
              <>
                <strong>GPU 是吞吐机器</strong>：上万个简单慢核，晶体管预算压倒性地砸给 ALU，用数量换单位时间的
                总产出。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The GPU&apos;s precondition is parallelism</strong>: the work must split into a great many
                independent pieces; when a serial part exists, Amdahl&apos;s Law locks a ceiling on the speedup.
              </>,
              <>
                <strong>GPU 的前提是并行度</strong>：任务必须能拆成大量彼此独立的小块；存在串行部分时，Amdahl
                定律给加速比锁死了上限。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Deep learning is the perfect customer</strong>: the body of a neural network is stacked
                matrix multiplies, each carrying tens of millions of independent tasks — embarrassingly parallel.
              </>,
              <>
                <strong>深度学习是完美客户</strong>：神经网络的主体是矩阵乘的堆叠，每个矩阵乘自带千万级独立任务，
                embarrassingly parallel。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The memory wall lies ahead</strong>: compute grows faster than memory bandwidth, so
                &ldquo;getting fed&rdquo; is harder than &ldquo;computing fast&rdquo; — the theme of{' '}
                <ChapterLink n={2} /> (latency hiding) and <ChapterLink n={4} /> (the memory hierarchy).
              </>,
              <>
                <strong>内存墙在前方</strong>：算力增长快于内存带宽，「喂得饱」比「算得快」更难——这是<ChapterLink n={2} />
                （延迟隐藏）和<ChapterLink n={4} />（内存层级）的主题。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Further reading (in recommended order):', '延伸阅读（按推荐顺序）：')}</p>
        <ul>
          <li>
            <a
              href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              CUDA C++ Programming Guide — Introduction
            </a>
            {t(
              ' — the official docs&apos; introductory chapter lays out this chapter&apos;s philosophy contrast in two pages, with the classic figure.',
              '：官方文档的引言一章，用两页讲清楚本章的设计哲学对比，配图经典。',
            )}
          </li>
          <li>
            <a
              href="https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              {t(
                'Programming Massively Parallel Processors (PMPP, 4th ed.), Chapter 1',
                'Programming Massively Parallel Processors（PMPP，第 4 版）第 1 章',
              )}
            </a>
            {t(
              ' — this course&apos;s &ldquo;textbook companion&rdquo;; Chapter 1 develops the motivation for heterogeneous computing most systematically.',
              '：本课程的「教科书伴侣」，第 1 章对异构计算的动机展开得最系统。',
            )}
          </li>
          <li>
            <a
              href="https://dl.acm.org/doi/10.1145/1022594.1022596"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              {t(
                'David Patterson, “Latency Lags Bandwidth” (CACM 2004)',
                'David Patterson, “Latency Lags Bandwidth”（CACM 2004）',
              )}
            </a>
            {t(
              ' — a four-page note on why bandwidth can be bought but latency cannot — the historical regularity behind that table in SEC 05.',
              '：四页短文，解释为什么带宽可以买、延迟买不到——SEC 05 那张表背后的历史规律。',
            )}
          </li>
          <li>
            <a
              href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              {t(
                'Stephen Jones, “How GPU Computing Works” (GTC 2021 talk)',
                'Stephen Jones, “How GPU Computing Works”（GTC 2021 讲座）',
              )}
            </a>
            {t(
              ' — an NVIDIA architect explains in person why the GPU looks the way it does; the best video version of this chapter.',
              '：NVIDIA 架构师亲自讲 GPU 为什么长这样，是本章内容最好的视频版。',
            )}
          </li>
          <li>
            <a
              href="https://safari.ethz.ch/architecture/"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              {t(
                'Onur Mutlu — Computer Architecture (open course, ETH Zürich)',
                'Onur Mutlu — Computer Architecture 公开课（ETH Zürich）',
              )}
            </a>
            {t(
              ' — if you want to firm up the CPU side too (caches, branch prediction, out-of-order execution), start here.',
              '：想把 CPU 那一侧（缓存、分支预测、乱序执行）也补扎实的话，从这里开始。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
