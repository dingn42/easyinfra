import { Callout, CodeBlock, MathTex, Quiz, Section, Term } from '@/components/ui'
import { RingAllReduceLab } from './RingAllReduce'
import { PipelineBubbleLab } from './PipelineBubble'
import { StrategySandboxLab } from './StrategySandbox'
import { ZeroBars } from './ZeroBars'
import { TensorParallelFigure } from './TensorParallelFigure'

const MEGATRON_F_G = `# Megatron 式 MLP 张量并行的两个“算子”
# f: 前向恒等、反向 AllReduce      g: 前向 AllReduce、反向恒等
def mlp_forward(X):                  # X 在每张卡上都有完整副本
    X = f(X)
    H = gelu(X @ W1_col_shard)       # W1 按列切：各卡算出激活的不同列
    Y_partial = H @ W2_row_shard     # W2 按行切：各卡得到 Y 的部分和
    Y = g(Y_partial)                 # 唯一通信点：AllReduce 求和
    return Y`

export default function Chapter() {
  return (
    <>
      <p>
        把 LLaMA-405B 的权重用 BF16 摆出来是 810GB，一张 H100 只有 80GB —— 别说训练，连参数本身都放不进一张卡。这不是工程上的小麻烦，而是大模型时代的默认前提：<strong>模型必须被切开，摊在几十张、几千张卡上</strong>。而怎么切，绝不只是「分一分」那么简单。切错了，几千张卡里 90% 的时间在等网络；切对了，万卡集群能跑出接近线性的扩展效率。这一章我们把四把刀 —— 数据并行、ZeRO、张量并行、流水线并行 —— 一把一把讲清楚，最后给你一个沙盘，亲手切一个 70B、405B 乃至 1T 的模型。
      </p>
      <p>
        在动刀之前，先回答一个更基本的问题：显存到底被什么吃掉了？很多人的直觉是「7B 模型就是 7B 参数 ≈ 14GB，一张 80G 的卡绰绰有余」。这个直觉对推理大致成立，对训练则错得离谱 —— 差了整整一个数量级的零头。
      </p>

      <Section index={1} title="显存都花在哪：一本训练账" lead="混合精度训练的每个参数，背后挂着约 16 字节的状态 —— 权重只是冰山一角。">
        <p>
          现代大模型训练几乎都用<Term t="混合精度（mixed precision）">前向/反向用 BF16 算（快、省显存），但权重更新用 FP32 累积（避免小梯度被舍入吞掉）。</Term>配 Adam 优化器。把一个参数从头到尾伺候好，需要这些东西常驻显存：BF16 的工作权重和梯度各 2 字节；为了数值稳定，优化器里还存着一份 FP32 的 master 权重 4 字节；Adam 的一阶动量 m 和二阶动量 v 又是 FP32 各 4 字节。加起来 <strong>每参数约 16 字节</strong> —— 其中 12 字节是优化器状态（optimizer states），是纯粹的「训练管理费」。
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">项目</th>
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">精度</th>
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">每参数</th>
                <th className="py-2 font-mono text-[11px] uppercase tracking-wider text-ink3">7B 模型合计</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">工作权重</td>
                <td className="py-2 pr-4 text-cyan">BF16</td>
                <td className="py-2 pr-4">2 B</td>
                <td className="py-2">14 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">梯度</td>
                <td className="py-2 pr-4 text-amber">BF16</td>
                <td className="py-2 pr-4">2 B</td>
                <td className="py-2">14 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">master 权重</td>
                <td className="py-2 pr-4 text-violet">FP32</td>
                <td className="py-2 pr-4">4 B</td>
                <td className="py-2">28 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">Adam 动量 m</td>
                <td className="py-2 pr-4 text-violet">FP32</td>
                <td className="py-2 pr-4">4 B</td>
                <td className="py-2">28 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">Adam 方差 v</td>
                <td className="py-2 pr-4 text-violet">FP32</td>
                <td className="py-2 pr-4">4 B</td>
                <td className="py-2">28 GB</td>
              </tr>
              <tr className="border-b border-line2">
                <td className="py-2 pr-4 font-sans font-medium text-ink">训练态小计</td>
                <td className="py-2 pr-4" />
                <td className="py-2 pr-4 text-volt">≈16 B</td>
                <td className="py-2 text-volt">≈112 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">激活（activation）</td>
                <td className="py-2 pr-4 text-cyan">BF16</td>
                <td className="py-2 pr-4 font-sans text-ink3">随 batch/序列长</td>
                <td className="py-2 font-sans text-ink3">4K 序列约 +18 GB</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-sans text-text">对照：纯推理</td>
                <td className="py-2 pr-4 text-cyan">BF16</td>
                <td className="py-2 pr-4">2 B</td>
                <td className="py-2">14 GB + KV cache</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          所以一个区区 7B 的模型，训练态的「静态账」就是 112GB 起步，再叠上前向传播留给反向用的激活（activation）—— 4K 序列、即便有 FlashAttention 帮忙省掉注意力矩阵，也还要十几个 GB。<strong>单张 80GB 的 A100/H100 根本放不下</strong>。这就是分布式并行的第一性原理：不是为了快（那是第二目标），而是<em>根本装不下</em>。也请记住推理和训练的鸿沟：推理只要 2 字节每参数加上 KV cache（第九章），7B 推理一张消费级 24G 卡都行；训练同一个模型，显存需求是它的八倍。
        </p>
      </Section>

      <Section index={2} title="数据并行与 ZeRO：先切「状态」，再切「模型」" lead="DP 复制模型、切数据；ZeRO 发现那 16 字节里大部分根本不需要每张卡都留一份。">
        <p>
          最朴素的多卡方案是<Term t="数据并行（Data Parallelism, DP）">每张卡持有完整模型副本，各自处理不同的数据分片，每步结束后同步梯度。</Term>：8 张卡，每张一份完整模型，全局 batch 切成 8 份各算各的。前向反向完全独立，只在每个 step 末尾把 8 张卡的梯度做一次 <strong>AllReduce</strong>（求和取平均），保证大家用同样的梯度更新、权重永远一致。DP 的好处是简单到发指，通信也只有每步一次、还能和反向计算重叠；坏处也直白 —— <strong>它一个字节的显存都不省</strong>。每张卡照样扛着完整的 112GB，7B 模型 8 卡是「8 份都放不下」，不是「放下了 1/8」。
        </p>
        <p>
          2019 年微软的 <strong>ZeRO</strong>（Zero Redundancy Optimizer）论文戳破了这层窗户纸：DP 的 N 张卡上存着 N 份<em>一模一样</em>的优化器状态、梯度和权重 —— 这是纯冗余。既然 Adam 的更新是逐元素的（每个参数的更新只依赖它自己的 m、v 和梯度），那把这些状态按参数切成 N 份、每张卡只管 1/N，更新照样能算。于是有了三档：
        </p>
        <ul>
          <li><strong>ZeRO-1</strong>：只切优化器状态（12B/参数那一大块）。每卡显存从 16Ψ 降到约 4Ψ + 12Ψ/N。代价几乎为零 —— 反正 step 末尾要通信，顺手把各自负责那段的更新结果广播回去就行。</li>
          <li><strong>ZeRO-2</strong>：再切梯度。反向算出的梯度用 ReduceScatter 直接归给负责的卡，不再每卡留全量。通信量和普通 DP 相当。</li>
          <li><strong>ZeRO-3</strong>：连权重也切。每层前向/反向用到谁，就临时 AllGather 谁，用完即扔。显存逼近 16Ψ/N 的理论下限，但通信量变成约 1.5 倍，且 AllGather 在关键路径上 —— 这是用带宽换显存的明码标价。</li>
        </ul>
        <ZeroBars />
        <p>
          拖一下上面的滑杆能看到：N 越大，ZeRO-3 的柱子越矮，64 卡时每卡只剩约 1.75GB 的状态 —— 7B 模型用一堆游戏卡训练在显存上成立了。这也是 PyTorch FSDP（Fully Sharded Data Parallel）的本质：它就是 ZeRO-3 的原生实现。
        </p>
        <Callout type="insight" title="切的是「冗余」，不是「计算」">
          <p>
            ZeRO 全系列在计算图上和普通 DP <em>完全等价</em> —— 每张卡跑的还是完整模型的前向反向，损失曲线分毫不差。它切掉的只是「N 份一样的东西」这种冗余。这给了我们一个重要的分类法：<strong>DP/ZeRO 切状态，TP/PP 切计算</strong>。前者不改变单卡的计算量和激活内存，所以一旦单层的激活或临时缓冲都放不下，ZeRO 救不了你，必须请出后两把刀。
          </p>
        </Callout>
      </Section>

      <Section index={3} title="通信的基本功：Ring AllReduce" lead="所有并行策略的地基是同一个原语 —— 在 N 张卡之间求和。它的最优实现优雅得值得专门一节。">
        <p>
          往下讲 TP/PP 之前，必须先把 <strong>AllReduce</strong> 这个词从黑盒里拆出来，因为它是 DP 梯度同步和 TP 部分和合并共用的地基。任务很简单：N 张卡各有一份大小为 D 的数据（比如各自的梯度），结束时每张卡都要拿到 N 份之和。最笨的做法是选一张「主卡」收集再分发 —— 主卡的网口要进出 2(N-1)·D 的流量，卡越多越堵，这正是参数服务器（parameter server）架构的旧伤。
        </p>
        <p>
          <strong>Ring AllReduce</strong> 的做法漂亮得多：把 N 张卡连成一个环，每张卡的数据切成 N 块，然后分两幕走。第一幕 <strong>reduce-scatter</strong>，走 N-1 步：每步每张卡把手头一个块发给下家、同时收上家一个块并累加。N-1 步之后，每张卡恰好握有<em>某一个块</em>的完整和。第二幕 <strong>all-gather</strong>，再走 N-1 步：成品块沿环回流，每卡每步转发一个完整块。下面的动画值得你一步一步点：盯住任意一个块，看它怎么转一圈攒齐 N 份贡献、再转一圈送达所有人。
        </p>

        <RingAllReduceLab />

        <p>
          算一笔通信账。总共 2(N-1) 步，每步每张卡只发送 D/N 的数据，于是每卡总发送量为：
        </p>
        <MathTex block tex="V_{\text{per-GPU}} = \underbrace{(N-1)\cdot\frac{D}{N}}_{\text{reduce-scatter}} + \underbrace{(N-1)\cdot\frac{D}{N}}_{\text{all-gather}} = 2\,\frac{N-1}{N}\,D \;<\; 2D" />
        <p>
          注意这个式子里最反直觉、也最优雅的事实：<strong>每卡通信量与 N 几乎无关</strong>，N 从 4 涨到 1024，系数只从 1.5 爬到 1.998，永远压在 2D 以下。环上每条链路同一时刻都在满负荷工作，没有任何一张卡是瓶颈 —— 带宽利用是均匀且最优的（可以证明 2(N-1)D/N 是 AllReduce 每卡通信量的下界）。代价是步数线性增长，每步都有一次链路延迟，所以超大集群上 NCCL 会切换到 tree/双二叉树等延迟更优的算法，但「带宽最优」这个性质让 ring 成为大消息的默认选择。
        </p>
      </Section>

      <Section index={4} title="张量并行：把一层矩阵乘切开" lead="Megatron 式切法：列切 W1、行切 W2，两层之间零通信，每层只在出口处求一次和。">
        <p>
          ZeRO 解决了「状态放不下」，但如果单层计算本身就太大 —— 比如 1T 模型一层 MLP 的权重就好几个 GB、激活峰值更高 —— 就得把<em>一层之内的矩阵乘法</em>切到多张卡上，这就是<Term t="张量并行（Tensor Parallelism, TP）">把单个权重矩阵按行或列切分到多卡，各卡算部分结果再通信合并。也叫 intra-layer 模型并行。</Term>。NVIDIA 的 Megatron-LM 给出了至今仍是标准答案的切法。
        </p>
        <p>
          看 Transformer 的 MLP：<MathTex tex="Y = \mathrm{GeLU}(XW_1)W_2" />。如果把 W<sub>1</sub> <strong>按列切</strong>成两半放到两张卡，每张卡算出 GeLU(XW<sub>1</sub>) 的不同<em>列</em> —— 由于 GeLU 是逐元素的，这两半激活互不依赖，不用通信。接着把 W<sub>2</sub> <strong>按行切</strong>：恰好上一步每张卡手里的「半截激活」正对着 W<sub>2</sub> 的「半截行」，各卡算出的是完整 Y 的一个<strong>部分和（partial sum）</strong>。整个两层的旅程中只需要在最后做一次 AllReduce 把部分和加起来。列切配行切，中间零通信 —— 这是手术刀级别的精确设计。
        </p>
        <CodeBlock code={MEGATRON_F_G} lang="python" title="megatron_mlp.py" />
        <TensorParallelFigure />
        <p>
          注意力层更幸运：多头注意力天生就是并行结构，<strong>按 head 切</strong>即可 —— 每张卡分到一部分 head，各自算各自的注意力，互相完全不用说话，只在末尾的输出投影（行切）后做一次 AllReduce。于是一个 Transformer 层的 TP 通信账是：前向 2 次 AllReduce（attention 一次 + MLP 一次），反向再来 2 次。一个 80 层的模型每个 step 就是几百次 AllReduce，每次都在<strong>关键路径</strong>上 —— 计算必须停下来等它。
        </p>
        <p>
          这决定了 TP 的活动半径。机内 8 张卡之间有 NVLink/NVSwitch，A100 代 600GB/s、H100 代 900GB/s；跨机走 InfiniBand 通常只有每卡 ~50GB/s 量级，差着约 18 倍。让每层都发生的通信去挤跨机网络，GPU 会饿死在等待里。所以业界的铁律是：<strong>TP ≤ 8，锁死在 NVLink 域内</strong>；需要更多并行度，找别的刀。
        </p>
        <Quiz
          question="为什么张量并行（TP）几乎从不跨机器部署，而数据并行可以？"
          options={[
            {
              text: 'TP 的实现依赖 NVLink 专有指令，InfiniBand 硬件不支持 AllReduce',
              explain: 'AllReduce 在任何互联上都能跑（NCCL 对 IB 支持得很好），问题不在「能不能」而在「快不快」。',
            },
            {
              text: 'TP 每层前向+反向要做多次 AllReduce、且都在关键路径上，通信极其频繁，只有 NVLink 量级的带宽兜得住；DP 每步只通信一次还能和计算重叠',
              correct: true,
              explain:
                '正是频率 × 关键路径的双重压力：80 层模型每 step 几百次同步 AllReduce，带宽慢 18 倍意味着 GPU 大部分时间在等网。DP 的梯度 AllReduce 每 step 仅一次，还能藏进反向计算里。',
            },
            {
              text: '跨机时 TP 的数值精度会下降，导致训练不收敛',
              explain: '通信不改变数值语义，跨机的 AllReduce 结果和机内完全一致，收敛性无关。',
            },
            {
              text: '因为跨机的 GPU 显存不能共享',
              explain: '任何并行方式下显存都不跨卡共享，TP 传的是激活和部分和，不是显存页。',
            },
          ]}
        />
      </Section>

      <Section index={5} title="流水线并行：切层，然后填满气泡" lead="把模型按层切成 P 段像工厂流水线 —— 难点是别让工位闲着。">
        <p>
          第三把刀是<Term t="流水线并行（Pipeline Parallelism, PP）">把模型按层切成 P 段（stage），各段放在不同的卡/机器上，数据像流水线一样依次经过。也叫 inter-layer 模型并行。</Term>：80 层切成 4 段，每段 20 层住一台机器。段与段之间只需要在边界上传一次激活 —— 通信量比 TP 小几个数量级，所以 PP 是<strong>跨机扩展</strong>的天然选择。但它有个结构性缺陷：第 2 段必须等第 1 段算完才能开工。如果整个 batch 一口气穿过流水线，任意时刻只有一段在干活，其余 P-1 段全在围观 —— 利用率 1/P，惨不忍睹。
        </p>
        <p>
          GPipe 的解法是把 batch 剁成 M 个 <strong>micro-batch</strong> 流水进去：第 1 段算完第 1 个 micro-batch 就立刻开始第 2 个，同时第 2 段接手第 1 个。流水填满之后所有段同时工作，只有开头的「灌水」和结尾的「排空」存在空转 —— 这些灰色空隙有个形象的名字：<strong>气泡（bubble）</strong>。气泡占总时间的比率有个干净的公式：
        </p>
        <MathTex block tex="\text{bubble ratio} = \frac{P-1}{M + P - 1}" />

        <PipelineBubbleLab />

        <p>
          公式和甘特图说的是同一件事：<strong>P 越深气泡越贵，M 越多气泡被摊得越薄</strong>。P=4、M=4 时气泡率 3/7 ≈ 43%，近半算力蒸发；M 拉到 16，降到 3/19 ≈ 16%；M=32 时只剩 9%。所以用 PP 的第一守则是 micro-batch 数至少是 P 的 4 倍以上。但 GPipe 还有第二个隐患：它把所有 M 个前向全部算完才开始反向，意味着每段要同时攒着 M 份 micro-batch 的激活等反向来取 —— M 一大显存先爆了。<strong>1F1B</strong>（one-forward-one-backward，源自 PipeDream）调度把顺序改成「暖机几个前向之后，每做一个前向就紧跟一个反向」：在上面的实验里切到 1F1B 你会看到总时长和气泡率<em>一点没变</em>，但每段同时在飞的 micro-batch 从 M 个降到至多 P 个 —— 激活驻留和 M 解耦了。这是免费的午餐，所以 Megatron、DeepSpeed 的流水线一律默认 1F1B。
        </p>
        <Quiz
          question="P=4 个 stage、M=16 个 micro-batch，用 1F1B 调度，气泡率约是多少？"
          options={[
            { text: '约 25%，因为 P=4 意味着固定浪费 1/4', explain: '1/P 是「完全不用 micro-batch」时的利用率灾难，流水起来之后气泡只出现在灌水/排空段。' },
            {
              text: '(P-1)/(M+P-1) = 3/19 ≈ 16%',
              correct: true,
              explain: '代入公式：(4-1)/(16+4-1) = 3/19 ≈ 15.8%。注意 1F1B 不改变气泡率（总时长与 GPipe 相同），它省的是激活显存。',
            },
            { text: '0%，1F1B 调度消除了所有气泡', explain: '1F1B 只是重排前向/反向的交错顺序，灌水和排空阶段的空转仍然存在，气泡率公式不变。' },
            { text: '3/16 ≈ 19%', explain: '分母是 M+P-1=19 而不是 M=16：总时间里除了 M 份满载工作还有 P-1 份灌排开销。' },
          ]}
        />
      </Section>

      <Section index={6} title="并行策略沙盘：把三把刀组合起来" lead="真实的大模型训练是 TP × PP × DP 的三维切分 —— 在显存、带宽、利用率之间找那个甜点。">
        <p>
          实战中三把刀从来不是单选题，而是乘法：先用 <strong>TP</strong> 在 NVLink 域内（≤8 卡）把单层切到能装下、激活压到可控；层数还溢出就用 <strong>PP</strong> 跨机切段（通信最便宜的跨机方案）；剩下的卡全部交给 <strong>DP</strong> 堆吞吐，再视显存余量给 DP 维叠一档 ZeRO。一个 64 卡训 70B 的经典配方是 TP8 × PP2 × DP4 + ZeRO-1：每个模型副本占 16 卡（两台 8 卡机），4 个副本并行吃数据。下面的沙盘把这本账完整摊开 —— 换模型、换卡型、拖三个并行度，看每卡显存堆叠条什么时候越过红线、三类通信的压力落在哪一层网络上。建议从三个预设开始，然后试着回答：405B 为什么非得 PP16？70B 推理为什么 TP8 一台机器就够？
        </p>

        <StrategySandboxLab />

        <p>
          玩沙盘时有几条经验法则值得带走：<strong>TP 永远先填满机内</strong>，因为 NVLink 的带宽是「不用白不用」；<strong>PP 的段数够用就好</strong>，切太深气泡和灌排延迟都涨，还要陪着更大的 M；<strong>DP 是吞吐的来源</strong>，所有「装下之后多出来的卡」都该给它；<strong>ZeRO 在 DP 维上白捡显存</strong>，Z1 几乎无代价应开尽开，Z3 则要掂量带宽。还有一条容易忽略：DP=1 时 ZeRO 完全失效 —— 它切的是副本间冗余，没有副本就没有冗余可切。
        </p>
      </Section>

      <Section index={7} title="推理侧与 MoE：同样的刀，不同的握法" lead="推理没有梯度和优化器，刀法的目标从「装下」变成「快」和「便宜」。">
        <p>
          推理的显存账轻得多（2B/参数 + KV cache），所以并行的动机变了：<strong>TP 在推理里买的是延迟</strong> —— 8 张卡同时算一层，单 token 的解码时间直接除以并行效率，这对在线服务的 P99 至关重要；而<strong>多副本（即推理版 DP）买的是吞吐</strong> —— 副本之间连梯度同步都没有，是零通信的「复制粘贴」，加多少副本吞吐几乎线性涨。所以 70B 在线服务的标准姿势是「TP8 一台机一个副本，流量大了横向加机器」。PP 在推理里地位尴尬：它不降低单 token 延迟（token 还是要串行走完所有段），主要用于权重大到单机装不下的场景（405B/1T 级）。顺带一提超长上下文：当 100 万 token 的 KV cache 本身就装不下时，还有<strong>序列并行（sequence/context parallelism）</strong>这第四把刀 —— 沿序列维把 token 切到多卡，配合 Ring Attention 之类的算法在卡间环形传递 KV 块。
        </p>
        <p>
          最后是 <strong>MoE（Mixture of Experts）</strong>带来的新维度。MoE 层里有几十上百个专家（expert）FFN，每个 token 只路由给其中一两个 —— 参数巨大但计算稀疏。自然的切法是<strong>专家并行（Expert Parallelism, EP）</strong>：专家们摊到不同卡上，每层做两次 <strong>all-to-all</strong> 通信，把每个 token 发到它的专家所在的卡、算完再收回来。all-to-all 是比 AllReduce 更「碎」的通信模式 —— 流量取决于路由的去向，负载还可能不均（热门专家所在的卡被挤爆），这正是 DeepSeek-V3、Mixtral 这类模型的基础设施团队花大力气优化的地方：专家放置、容量因子、通信与计算重叠，一个比稠密模型再复杂一档的工程世界。
        </p>
      </Section>

      <Section index={8} title="总结与延伸阅读">
        <p>这一章的核心账目和刀法，浓缩成几条：</p>
        <ul>
          <li>混合精度训练每参数约 <strong>16 字节</strong>（2+2+4+4+4），7B 模型训练态 112GB 起步 —— 单卡装不下是分布式的第一性原理；推理只要 2B/参数 + KV。</li>
          <li><strong>DP</strong> 切数据不省显存；<strong>ZeRO-1/2/3</strong> 依次把优化器状态、梯度、权重切到 DP 各卡，显存逼近 1/N，计算图与 DP 完全等价。</li>
          <li><strong>Ring AllReduce</strong>：2(N-1) 步、每卡通信量 2(N-1)D/N —— 与 N 无关地有界且带宽最优，是 DP 和 TP 共同的地基。</li>
          <li><strong>TP</strong>（Megatron）列切+行切，每层前反向共 4 次关键路径 AllReduce → 锁死 NVLink 域内（≤8 卡）；<strong>PP</strong> 切层跨机最便宜，气泡率 (P-1)/(M+P-1)，用大 M 摊薄、用 1F1B 省激活。</li>
          <li>组合拳：TP 填满机内 → PP 跨机装下 → 剩余全给 DP（+ZeRO）堆吞吐；推理用 TP 买延迟、副本买吞吐，MoE 再加一维 EP 与 all-to-all。</li>
        </ul>
        <p>延伸阅读，都是值得精读的一手材料：</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noreferrer">Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism</a> —— 张量并行的原始论文，列切/行切的 f/g 算子设计至今未被超越。
          </li>
          <li>
            <a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a> —— 三档切分的显存公式与通信分析，FSDP 的理论源头。
          </li>
          <li>
            <a href="https://arxiv.org/abs/1811.06965" target="_blank" rel="noreferrer">GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism</a> —— micro-batch 流水与气泡分析的出处。
          </li>
          <li>
            <a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook" target="_blank" rel="noreferrer">HuggingFace Ultra-Scale Playbook</a> —— 在真实集群上系统性扫过 TP/PP/DP/ZeRO 组合的现代实战手册，配大量实测曲线。
          </li>
          <li>
            <a href="https://www.deepspeed.ai/training/" target="_blank" rel="noreferrer">DeepSpeed Training 文档</a> —— ZeRO 全家桶与流水线并行的工程参考实现。
          </li>
        </ul>
      </Section>
    </>
  )
}
