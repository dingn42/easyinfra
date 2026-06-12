import { Callout, CodeBlock, MathTex, Quiz, Section, Term } from '@/components/ui'
import { DieAreaFigure } from './DieAreaFigure'
import { ThroughputRace } from './ThroughputRace'
import { TransistorBudget } from './TransistorBudget'

const VEC_ADD_CPU = `// CPU 思维：一个工人，从头到尾按顺序做完
void vec_add(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];   // 注意：第 i 次迭代完全不依赖第 i-1 次
  }
}`

const VEC_ADD_GPU = `// GPU 思维：雇 n 个工人，每人只负责一个元素
// 下面是"第 i 号工人"的全部工作（第 3 章会正式介绍这套语法）
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 我是第几号工人？
  if (i < n) c[i] = a[i] + b[i];                  // 只做属于我的那一份
}`

export default function Chapter() {
  return (
    <>
      <p>
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
      </p>

      <Section
        index={1}
        title="两种设计哲学：延迟机器与吞吐机器"
        lead="同样一笔晶体管预算，CPU 和 GPU 交出了两份完全相反的答卷。"
      >
        <p>
          CPU 的设计目标是<strong>把单个任务尽快做完</strong>，也就是最小化延迟（latency）。为了这个目标，
          它在芯片上堆了大量「不直接做计算」的电路：动辄几十 MB 的多级缓存（cache）用来掩盖内存的慢；
          分支预测器（branch predictor）赌 if/else 会往哪边走，赌对了就白赚几十个周期；乱序执行
          （out-of-order execution）引擎把指令拆开重排，谁的数据先就绪谁先上；还有投机执行、预取器……
          这一整套机器的唯一目的，是让那少数几个核心<strong>永远不要停下来等</strong>。代价也很直白：在一颗
          现代旗舰 CPU 的 die（裸片）上，真正做算术的 ALU（Arithmetic Logic Unit，算术逻辑单元）
          只占很小一块，绝大部分面积都给了缓存和控制逻辑。
        </p>
        <p>
          GPU 的答卷完全相反：它的目标是<strong>单位时间内完成的任务总量最大</strong>，也就是吞吐
          （throughput）。单个核心可以慢、可以笨——不做乱序、几乎不做分支预测、主频只有 CPU 的三分之一
          ——但要多。一颗 H100 上有 16896 个 FP32 核心，分布在 132 个 SM（Streaming Multiprocessor，
          流式多处理器）里；省下来的控制逻辑和缓存面积，全部换成了 ALU。一个任务在 GPU 单核上可能要花
          CPU 单核八倍的时间，但只要任务足够多，两千个慢工人对八个快工人，依然是碾压。
        </p>
        <p>
          一个常用的类比：CPU 是一位顶尖外科医生，GPU 是一万个流水线工人。一台精密手术——每一步都依赖
          上一步的结果、随时要根据出血情况改变方案——只能交给医生，一万个工人围上来反而碍事。但如果活儿是
          「给一百万个零件各拧一颗螺丝」，工人军团完胜，哪怕每个工人拧一颗螺丝的速度只有医生的几分之一。
          这个类比同时也交代了 GPU 的<strong>适用前提</strong>：任务必须能拆成大量彼此独立的小块——可以
          同时做、互相不等待、不需要频繁交换中间结果。前提不成立时，GPU 的优势会瞬间消失，这一点我们马上
          用实验验证。
        </p>
        <DieAreaFigure />
        <Callout type="insight" title="GPU 不是「更快的 CPU」">
          <p>
            两者优化的根本就不是同一个量：CPU 最小化<strong>单个任务的完成时间</strong>（延迟），GPU 最大化
            <strong>单位时间完成的任务总量</strong>（吞吐）。所以「GPU 比 CPU 快多少倍」这个问题本身是不完整
            的——先问你的问题里有没有几万个可以同时做、互不等待的小活。有，GPU 快两个数量级；没有，GPU
            可能比你的笔记本还慢。
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title="实验：吞吐量赛道"
        lead="8 个快核对 2048 个慢核，谁先把活干完？拖动任务数，亲手找到分界点。"
      >
        <p>
          规则很简单：CPU 有 8 个核，每个任务耗时 1 µs；GPU 有 2048 个核，单核慢 k 倍（默认 8×），
          外加一笔固定的 20 µs 启动开销（对应真实世界里的 kernel launch 开销——把活儿派给 GPU
          本身就要花时间）。所有任务彼此独立。先把任务数拖到最小的 64 个看看谁赢，再拖到一百万个看看
          差距能拉到多大；也试试把 GPU 单核慢倍数拉满到 32×，看看「核多」能不能撑住「核慢」。
        </p>
        <ThroughputRace index={1} />
        <p>
          你应该观察到了三件事。第一，任务很少时 CPU 赢：64 个任务它 8 µs 就干完了，GPU 光启动就要
          20 µs——杀鸡用牛刀，牛刀抬起来的功夫鸡已经杀完了。第二，任务足够多时 GPU 碾压，而且任务越多
          加速比越接近理论极限（2048 ÷ 8 ÷ 慢倍数）。第三，GPU 的进度条是一格一格跳的：2048
          个核同时开工、同时交活，一「波」一波地推进——这个「波」的概念在第 2 章会以 wave/warp
          的形式正式登场。
        </p>
        <p>
          还有一个隐藏条件值得说破：这场赛跑里所有任务<strong>彼此独立</strong>。一旦程序里存在无法并行的
          串行部分，加速比就有天花板——这就是 Amdahl 定律（Amdahl&apos;s Law）。设串行部分占总工作量的
          比例为 s，处理器数量为 P，则总加速比
        </p>
        <MathTex block tex="\text{Speedup} \le \dfrac{1}{s + \dfrac{1 - s}{P}} \xrightarrow{P \to \infty} \dfrac{1}{s}" />
        <p>
          哪怕只有 5% 的工作必须串行执行，无论你堆多少万个核心，加速比也不可能超过 20×。这就是为什么
          「能不能用 GPU」的第一个问题永远是「你的问题里有多少并行度」——下一节我们就去看深度学习的答案。
        </p>
        <Quiz
          question="下面哪类任务交给 GPU 最不划算？"
          options={[
            {
              text: '把一亿个像素各自调亮 20%',
              explain: '每个像素的计算完全独立，是教科书级的并行任务，GPU 的主场。',
            },
            {
              text: '沿一条链表逐结点查找：每一步的地址都取决于上一步读到的指针',
              correct: true,
              explain:
                '对。这是强串行依赖：第 i 步不做完，第 i+1 步连数据在哪都不知道，几万个核里只有一个能干活，' +
                '其余全部围观。再叠加 GPU 单核本身更慢、分支处理更弱，这类指针追逐（pointer chasing）任务 GPU 反而比 CPU 慢。',
            },
            {
              text: '计算一个 4096×4096 的矩阵乘法',
              explain: '矩阵乘的每个输出元素都是一次独立点积，有千万级并行度，是 GPU 最擅长的负载。',
            },
            {
              text: '给一百万个独立样本各算一次损失函数',
              explain: '样本之间互不依赖，天然并行，GPU 处理这类批量任务正是强项。',
            },
          ]}
        />
      </Section>

      <Section index={3} title="并行度从哪来" lead="深度学习恰好是把并行度成吨送上门的那类问题。">
        <p>
          先看最简单的例子：向量加法。c[i] = a[i] + b[i]，每个 i 的计算只读自己的两个输入、写自己的一个
          输出，跟其他任何 i 都没有关系。一个长度一百万的向量加法，就是一百万个天然独立的小任务。在 CPU
          上我们习惯写成循环：
        </p>
        <CodeBlock code={VEC_ADD_CPU} lang="cpp" title="vec_add.c" highlight={[4]} />
        <p>
          GPU 编程要求的是一次<strong>思维转换</strong>：别再问「这个循环怎么跑得更快」，改问「每个元素
          要做的活是什么」。把循环体抠出来，写成一个工人的工作说明书，然后雇 n 个工人同时开工——循环消失了，
          取而代之的是「我是第几号工人」这个身份查询：
        </p>
        <CodeBlock code={VEC_ADD_GPU} lang="cuda" title="vec_add.cu" highlight={[4, 5]} />
        <p>
          再上一个量级：矩阵乘法 C = A·B。输出矩阵的每个元素 C[i][j]，是 A 的第 i 行和 B 的第 j
          列的点积——又是彼此完全独立的计算。对于 N×N 的矩阵，总计算量是 <MathTex tex="2N^3" /> 次浮点
          运算（每个输出元素 N 次乘加），而独立任务数是 <MathTex tex="N^2" /> 个。取 N = 4096：约
          1.4×10¹¹ 次运算，拆成 1670 万个互不依赖的点积。这样的并行度，喂饱几万个核心绰绰有余。
        </p>
        <p>
          最后一步推论就水到渠成了：神经网络的主体就是矩阵乘的堆叠。全连接层是矩阵乘，注意力机制
          （attention）的核心是矩阵乘，卷积也能化成矩阵乘——一次大模型的前向传播，本质上是几百个大矩阵乘
          排队执行，每一个内部都携带千万级的并行度。这类不需要费心改造、并行度自己送上门的问题，有个专门
          的称呼：embarrassingly parallel（「不好意思地并行」，意为并行得毫不费力）。深度学习和 GPU
          的关系与其说是「选择」，不如说是互相成就：GPU 等到了能填满它所有核心的负载，深度学习等到了能把
          它的计算量在合理时间内算完的机器。
        </p>
        <Quiz
          question="GPU 能把上千个弱核心都喂饱、真正成为「吞吐机器」的前提是什么？"
          options={[
            {
              text: '主频足够高',
              explain: 'GPU 主频通常只有 CPU 的三分之一左右（约 1.5～2 GHz），它的快从来不靠主频。',
            },
            {
              text: '程序里有足够多可以同时执行、彼此独立的工作（足够的并行度）',
              correct: true,
              explain:
                '对。几万个核心意味着至少需要几万个（实际上远不止，第 2 章讲延迟隐藏时你会看到需要更多）' +
                '互不依赖的任务才能填满机器。并行度不足时，再多的核也只能空转。',
            },
            {
              text: '缓存做得足够大',
              explain: 'GPU 恰恰是把缓存面积省下来换 ALU 的那一方；它靠海量线程切换而不是大缓存来掩盖访存延迟。',
            },
            {
              text: '代码必须用 C++ 编写',
              explain: '语言无关紧要——PyTorch 用户写的是 Python，底层照样把矩阵乘派给 GPU。关键在负载本身的结构。',
            },
          ]}
        />
      </Section>

      <Section
        index={4}
        title="实验：晶体管预算分配器"
        lead="现在假装你是芯片架构师：100% 的 die 面积摆在面前，控制、缓存、ALU，怎么分？"
      >
        <p>
          这个实验把 SEC 01 的那张静态图变成可以拖的。三个滑杆的总和恒为 100%——拖动任何一个，另外两个会
          按比例缩放，就像真实的芯片设计：面积给了缓存，就不能再给 ALU。右侧两个读数对应两种设计哲学的
          「得分」：<strong>单线程性能分</strong>来自控制逻辑与缓存，但收益递减——把缓存从 30% 加到
          50%，命中率的提升远不如从 0% 加到 20% 来得猛，分支预测也一样，预测准确率逼近 98%
          之后每提高一点都要付出成倍的晶体管；<strong>吞吐分</strong>则严格正比于 ALU
          数量——多一个核就多一份产出，简单粗暴，没有递减。
        </p>
        <TransistorBudget index={2} />
        <p>
          试试两个极端：全部砸给 ALU，吞吐分拉满，但单线程分惨不忍睹——这台机器跑串行代码会被手机吊打；
          全部砸给控制和缓存，单线程分也只能逼近渐近线，因为收益递减锁死了天花板。两个预设按钮给出的正是
          现实中两条路线的典型取舍。还有一个细节值得注意：「典型 GPU」预设里缓存并不是
          0%——真实 GPU 保留了寄存器堆、shared memory 和一块不大的 L2，它们的角色和 CPU
          缓存很不一样（更像程序员手动管理的便签纸），这是第 2 章和第 4 章的重头戏。
        </p>
      </Section>

      <Section
        index={5}
        title="预告：算得快，还要喂得饱"
        lead="把 ALU 堆上天只解决了一半问题——数据进不来，再多核心也只能空转。"
      >
        <p>
          本章我们一直假设「任务到了核心手里就能算」，但真实世界里数据躺在内存里，而内存比算术慢得多，
          且这个差距还在逐年拉大——算力大约每两年翻一倍以上，内存带宽和延迟的改善却慢得多。这就是著名的
          内存墙（memory wall）；David Patterson 把其中更扎心的那半句总结为「延迟落后于带宽」
          （latency lags bandwidth）：带宽还能靠堆并行做宽，延迟几乎无药可医。
        </p>
        <p>
          GPU 的应对是给自己配上极宽的内存：A100 的 HBM2e（High Bandwidth Memory，高带宽内存）提供约
          1.9～2.0 TB/s 的带宽，H100 的 HBM3 约 3.35 TB/s；作为对比，桌面 CPU 的双通道 DDR5 大约只有
          100 GB/s——差了 20～30 倍。但「宽」不等于「快」，看一眼各级存储的延迟数量级（以 GPU 核心时钟
          周期计，均为约数）：
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full min-w-[420px] border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="py-2 pr-4 font-mono text-[11px] font-medium uppercase tracking-wider text-ink2">操作</th>
                <th className="py-2 pr-4 font-mono text-[11px] font-medium uppercase tracking-wider text-ink2">延迟（周期）</th>
                <th className="py-2 font-mono text-[11px] font-medium uppercase tracking-wider text-ink2">直觉</th>
              </tr>
            </thead>
            <tbody className="text-text">
              <tr className="border-b border-line">
                <td className="py-2 pr-4">一次浮点乘加（FMA）</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-volt">~1</td>
                <td className="py-2 text-ink2">基准单位</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4">L1 缓存命中</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-cyan">~4</td>
                <td className="py-2 text-ink2">伸手拿桌上的东西</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4">L2 缓存命中</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-cyan">~40</td>
                <td className="py-2 text-ink2">走去隔壁房间取一趟</td>
              </tr>
              <tr>
                <td className="py-2 pr-4">DRAM / HBM 访问</td>
                <td className="py-2 pr-4 font-mono tabular-nums text-amber">~200+</td>
                <td className="py-2 text-ink2">下楼去快递柜——等一次的工夫够做几百次乘加</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          等一次显存的时间，核心足够做几百次乘加——如果它干等的话。GPU 当然不会干等，它的解法不是像 CPU
          那样修更大的缓存，而是用海量线程互相掩护：一批线程在等数据，就立刻切换到另一批先算着，用并行度把
          延迟「藏」起来。这套延迟隐藏（latency hiding）机制怎么运转、需要多少线程才藏得住，是第 2 章的
          核心；而怎么组织访存模式让那 2 TB/s 的带宽真正跑满、而不是浪费在零碎的访问上，是第 4
          章的全部内容。记住本章的结论：<strong>算力解决「算得快」，内存系统决定「喂不喂得饱」——后者才是
          大多数真实程序的瓶颈。</strong>
        </p>
      </Section>

      <Section index={6} title="总结与延伸阅读">
        <p>这一章没有写代码，但它是后面所有章节的地基。五个要点：</p>
        <ul>
          <li>
            <strong>CPU 是延迟机器</strong>：少量强核 + 大缓存 + 分支预测 / 乱序执行等复杂控制逻辑，一切为了
            单个任务尽快完成。
          </li>
          <li>
            <strong>GPU 是吞吐机器</strong>：上万个简单慢核，晶体管预算压倒性地砸给 ALU，用数量换单位时间的
            总产出。
          </li>
          <li>
            <strong>GPU 的前提是并行度</strong>：任务必须能拆成大量彼此独立的小块；存在串行部分时，Amdahl
            定律给加速比锁死了上限。
          </li>
          <li>
            <strong>深度学习是完美客户</strong>：神经网络的主体是矩阵乘的堆叠，每个矩阵乘自带千万级独立任务，
            embarrassingly parallel。
          </li>
          <li>
            <strong>内存墙在前方</strong>：算力增长快于内存带宽，「喂得饱」比「算得快」更难——这是第 2 章
            （延迟隐藏）和第 4 章（内存层级）的主题。
          </li>
        </ul>
        <p>延伸阅读（按推荐顺序）：</p>
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
            ：官方文档的引言一章，用两页讲清楚本章的设计哲学对比，配图经典。
          </li>
          <li>
            <a
              href="https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              Programming Massively Parallel Processors（PMPP，第 4 版）第 1 章
            </a>
            ：本课程的「教科书伴侣」，第 1 章对异构计算的动机展开得最系统。
          </li>
          <li>
            <a
              href="https://dl.acm.org/doi/10.1145/1022594.1022596"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              David Patterson, “Latency Lags Bandwidth”（CACM 2004）
            </a>
            ：四页短文，解释为什么带宽可以买、延迟买不到——SEC 05 那张表背后的历史规律。
          </li>
          <li>
            <a
              href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              Stephen Jones, “How GPU Computing Works”（GTC 2021 讲座）
            </a>
            ：NVIDIA 架构师亲自讲 GPU 为什么长这样，是本章内容最好的视频版。
          </li>
          <li>
            <a
              href="https://safari.ethz.ch/architecture/"
              target="_blank"
              rel="noreferrer"
              className="text-cyan underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              Onur Mutlu — Computer Architecture 公开课（ETH Zürich）
            </a>
            ：想把 CPU 那一侧（缓存、分支预测、乱序执行）也补扎实的话，从这里开始。
          </li>
        </ul>
      </Section>
    </>
  )
}
