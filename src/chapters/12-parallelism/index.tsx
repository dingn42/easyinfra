import { Callout, ChapterLink, CodeBlock, HardwareBaseline, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { RingAllReduceLab } from './RingAllReduce'
import { PipelineBubbleLab } from './PipelineBubble'
import { StrategySandboxLab } from './StrategySandbox'
import { ZeroBars } from './ZeroBars'
import { TensorParallelFigure } from './TensorParallelFigure'

const MEGATRON_F_G_EN = `# The two "operators" of Megatron-style MLP tensor parallelism
# f: identity forward, AllReduce backward    g: AllReduce forward, identity backward
def mlp_forward(X):                  # X is fully replicated on every GPU
    X = f(X)
    H = gelu(X @ W1_col_shard)       # W1 split by column: each GPU computes a different slice of the activation
    Y_partial = H @ W2_row_shard     # W2 split by row: each GPU produces a partial sum of Y
    Y = g(Y_partial)                 # the only communication point: AllReduce to sum
    return Y`

const MEGATRON_F_G_ZH = `# Megatron 式 MLP 张量并行的两个“算子”
# f: 前向恒等、反向 AllReduce      g: 前向 AllReduce、反向恒等
def mlp_forward(X):                  # X 在每张卡上都有完整副本
    X = f(X)
    H = gelu(X @ W1_col_shard)       # W1 按列切：各卡算出激活的不同列
    Y_partial = H @ W2_row_shard     # W2 按行切：各卡得到 Y 的部分和
    Y = g(Y_partial)                 # 唯一通信点：AllReduce 求和
    return Y`

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            Lay out LLaMA-405B's weights in BF16 and you get 810GB; a single H100 has only 80GB — forget training, the parameters alone won't fit on one card. This isn't a minor engineering nuisance, it's the default premise of the large-model era: <strong>the model has to be cut open and spread across dozens, even thousands, of GPUs</strong>. And <em>how</em> you cut it is anything but a simple "divide it up." Cut it wrong and 90% of the time across thousands of GPUs goes to waiting on the network; cut it right and a ten-thousand-GPU cluster scales nearly linearly. This chapter walks through four blades — data parallelism, ZeRO, tensor parallelism, pipeline parallelism — one at a time, then hands you a sandbox to carve up a 70B, a 405B, even a 1T model with your own hands.
          </>,
          <>
            把 LLaMA-405B 的权重用 BF16 摆出来是 810GB，一张 H100 只有 80GB —— 别说训练，连参数本身都放不进一张卡。这不是工程上的小麻烦，而是大模型时代的默认前提：<strong>模型必须被切开，摊在几十张、几千张卡上</strong>。而怎么切，绝不只是「分一分」那么简单。切错了，几千张卡里 90% 的时间在等网络；切对了，万卡集群能跑出接近线性的扩展效率。这一章我们把四把刀 —— 数据并行、ZeRO、张量并行、流水线并行 —— 一把一把讲清楚，最后给你一个沙盘，亲手切一个 70B、405B 乃至 1T 的模型。
          </>,
        )}
      </p>
      <HardwareBaseline
        ids={['a100', 'h100']}
        note={t(
          'Training examples use A100; NVLink/IB figures cover both generations.',
          '训练示例用 A100；NVLink/IB 数字覆盖两代卡。',
        )}
      />
      <p>
        {t(
          'Before we pick up a blade, answer a more basic question: what actually eats the memory? Many people’s intuition is "a 7B model is just 7B params ≈ 14GB, an 80G card has plenty of room." That intuition is roughly right for inference and wildly wrong for training — off by an entire order of magnitude.',
          '在动刀之前，先回答一个更基本的问题：显存到底被什么吃掉了？很多人的直觉是「7B 模型就是 7B 参数 ≈ 14GB，一张 80G 的卡绰绰有余」。这个直觉对推理大致成立，对训练则错得离谱 —— 差了整整一个数量级的零头。',
        )}
      </p>

      <Section
        index={1}
        title={t('Where the memory goes: a training ledger', '显存都花在哪：一本训练账')}
        lead={t(
          'Every parameter in mixed-precision training drags about 16 bytes of state behind it — the weights are just the tip of the iceberg.',
          '混合精度训练的每个参数，背后挂着约 16 字节的状态 —— 权重只是冰山一角。',
        )}
      >
        <p>
          {t(
            <>
              Modern large-model training almost always runs in <Term t="Forward/backward compute in BF16 (fast, memory-thrifty), but weight updates accumulate in FP32 (so small gradients aren’t swallowed by rounding).">mixed precision</Term> with the Adam optimizer. To shepherd one parameter from start to finish, all of this has to stay resident in memory: 2 bytes each for the BF16 working weight and gradient; for numerical stability the optimizer also keeps an FP32 master copy of the weight at 4 bytes; and Adam's first-moment m and second-moment v are another 4 bytes each in FP32. Add it up and that's <strong>about 16 bytes per parameter</strong> — of which 12 bytes are optimizer states, pure "training overhead."
            </>,
            <>
              现代大模型训练几乎都用<Term t="前向/反向用 BF16 算（快、省显存），但权重更新用 FP32 累积（避免小梯度被舍入吞掉）。">混合精度（mixed precision）</Term>配 Adam 优化器。把一个参数从头到尾伺候好，需要这些东西常驻显存：BF16 的工作权重和梯度各 2 字节；为了数值稳定，优化器里还存着一份 FP32 的 master 权重 4 字节；Adam 的一阶动量 m 和二阶动量 v 又是 FP32 各 4 字节。加起来 <strong>每参数约 16 字节</strong> —— 其中 12 字节是优化器状态（optimizer states），是纯粹的「训练管理费」。
            </>,
          )}
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">{t('Item', '项目')}</th>
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">{t('Precision', '精度')}</th>
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">{t('Per param', '每参数')}</th>
                <th className="py-2 font-mono text-[11px] uppercase tracking-wider text-ink3">{t('7B model total', '7B 模型合计')}</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">{t('Working weights', '工作权重')}</td>
                <td className="py-2 pr-4 text-cyan">BF16</td>
                <td className="py-2 pr-4">2 B</td>
                <td className="py-2">14 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">{t('Gradients', '梯度')}</td>
                <td className="py-2 pr-4 text-amber">BF16</td>
                <td className="py-2 pr-4">2 B</td>
                <td className="py-2">14 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">{t('Master weights', 'master 权重')}</td>
                <td className="py-2 pr-4 text-violet">FP32</td>
                <td className="py-2 pr-4">4 B</td>
                <td className="py-2">28 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">{t('Adam moment m', 'Adam 动量 m')}</td>
                <td className="py-2 pr-4 text-violet">FP32</td>
                <td className="py-2 pr-4">4 B</td>
                <td className="py-2">28 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">{t('Adam variance v', 'Adam 方差 v')}</td>
                <td className="py-2 pr-4 text-violet">FP32</td>
                <td className="py-2 pr-4">4 B</td>
                <td className="py-2">28 GB</td>
              </tr>
              <tr className="border-b border-line2">
                <td className="py-2 pr-4 font-sans font-medium text-ink">{t('Training-state subtotal', '训练态小计')}</td>
                <td className="py-2 pr-4" />
                <td className="py-2 pr-4 text-volt">≈16 B</td>
                <td className="py-2 text-volt">≈112 GB</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-sans text-text">{t('Activations', '激活（activation）')}</td>
                <td className="py-2 pr-4 text-cyan">BF16</td>
                <td className="py-2 pr-4 font-sans text-ink3">{t('scales with batch/seq len', '随 batch/序列长')}</td>
                <td className="py-2 font-sans text-ink3">{t('~+18 GB at 4K seq', '4K 序列约 +18 GB')}</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-sans text-text">{t('For contrast: pure inference', '对照：纯推理')}</td>
                <td className="py-2 pr-4 text-cyan">BF16</td>
                <td className="py-2 pr-4">2 B</td>
                <td className="py-2">{t('14 GB + KV cache', '14 GB + KV cache')}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          {t(
            <>
              So a mere 7B model starts at 112GB just for the "static" training-state ledger, before you add the activations the forward pass saves for the backward — at a 4K sequence, even with FlashAttention sparing you the attention matrix, that's still another dozen-plus GB. <strong>A single 80GB A100/H100 simply can't hold it.</strong> This is the first principle of distributed parallelism: not for speed (that's the second goal), but because <em>it literally doesn't fit</em>. Keep the inference-vs-training gulf in mind too: inference needs only 2 bytes per param plus the KV cache (<ChapterLink n={9} />), so 7B inference fits on a single 24G consumer card; training the same model demands eight times the memory.
            </>,
            <>
              所以一个区区 7B 的模型，训练态的「静态账」就是 112GB 起步，再叠上前向传播留给反向用的激活（activation）—— 4K 序列、即便有 FlashAttention 帮忙省掉注意力矩阵，也还要十几个 GB。<strong>单张 80GB 的 A100/H100 根本放不下</strong>。这就是分布式并行的第一性原理：不是为了快（那是第二目标），而是<em>根本装不下</em>。也请记住推理和训练的鸿沟：推理只要 2 字节每参数加上 KV cache（<ChapterLink n={9} />），7B 推理一张消费级 24G 卡都行；训练同一个模型，显存需求是它的八倍。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={2}
        title={t('Data parallelism and ZeRO: shard the state first, then the model', '数据并行与 ZeRO：先切「状态」，再切「模型」')}
        lead={t(
          'DP replicates the model and splits the data; ZeRO realizes that most of those 16 bytes don’t need a copy on every GPU.',
          'DP 复制模型、切数据；ZeRO 发现那 16 字节里大部分根本不需要每张卡都留一份。',
        )}
      >
        <p>
          {t(
            <>
              The most naive multi-GPU scheme is <Term t="Every GPU holds a complete model replica, each processes a different data shard, and gradients are synchronized at the end of every step.">data parallelism (DP)</Term>: 8 GPUs, each with a full model copy, the global batch split into 8 shards each computed independently. Forward and backward are entirely independent; only at the end of each step do the 8 GPUs run one <strong>AllReduce</strong> on the gradients (sum then average), so everyone updates with the same gradient and the weights stay identical forever. DP's upside is that it's brain-dead simple, and communication happens just once per step and can even overlap with the backward pass; its downside is just as blunt — <strong>it saves not a single byte of memory</strong>. Every GPU still carries the full 112GB, so 8 GPUs for a 7B model means "all 8 fail to fit," not "1/8 fits."
            </>,
            <>
              最朴素的多卡方案是<Term t="每张卡持有完整模型副本，各自处理不同的数据分片，每步结束后同步梯度。">数据并行（Data Parallelism, DP）</Term>：8 张卡，每张一份完整模型，全局 batch 切成 8 份各算各的。前向反向完全独立，只在每个 step 末尾把 8 张卡的梯度做一次 <strong>AllReduce</strong>（求和取平均），保证大家用同样的梯度更新、权重永远一致。DP 的好处是简单到发指，通信也只有每步一次、还能和反向计算重叠；坏处也直白 —— <strong>它一个字节的显存都不省</strong>。每张卡照样扛着完整的 112GB，7B 模型 8 卡是「8 份都放不下」，不是「放下了 1/8」。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              In 2019 Microsoft's <strong>ZeRO</strong> (Zero Redundancy Optimizer) paper called the bluff: across DP's N GPUs sit N <em>identical</em> copies of the optimizer states, gradients, and weights — pure redundancy. Since Adam's update is element-wise (each parameter's update depends only on its own m, v, and gradient), you can shard those states by parameter into N pieces, let each GPU own just 1/N, and the update still works out. Hence three tiers:
            </>,
            <>
              2019 年微软的 <strong>ZeRO</strong>（Zero Redundancy Optimizer）论文戳破了这层窗户纸：DP 的 N 张卡上存着 N 份<em>一模一样</em>的优化器状态、梯度和权重 —— 这是纯冗余。既然 Adam 的更新是逐元素的（每个参数的更新只依赖它自己的 m、v 和梯度），那把这些状态按参数切成 N 份、每张卡只管 1/N，更新照样能算。于是有了三档：
            </>,
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>ZeRO-1</strong>: shard only the optimizer states (the big 12B/param chunk). Per-GPU memory drops from 16Ψ to about 4Ψ + 12Ψ/N. The cost is almost nil — there's a communication step at the end anyway, so each GPU just broadcasts back the update for the slice it owns.
              </>,
              <>
                <strong>ZeRO-1</strong>：只切优化器状态（12B/参数那一大块）。每卡显存从 16Ψ 降到约 4Ψ + 12Ψ/N。代价几乎为零 —— 反正 step 末尾要通信，顺手把各自负责那段的更新结果广播回去就行。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>ZeRO-2</strong>: also shard the gradients. The gradients the backward pass produces are handed straight to the owning GPU via ReduceScatter, so no GPU keeps the full set anymore. Communication volume matches plain DP.
              </>,
              <>
                <strong>ZeRO-2</strong>：再切梯度。反向算出的梯度用 ReduceScatter 直接归给负责的卡，不再每卡留全量。通信量和普通 DP 相当。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>ZeRO-3</strong>: shard the weights too. Whichever layer a forward/backward needs, you AllGather it on the fly and discard it after use. Memory approaches the theoretical floor of 16Ψ/N, but communication rises to about 1.5× and the AllGather sits on the critical path — this is the marked price of trading bandwidth for memory.
              </>,
              <>
                <strong>ZeRO-3</strong>：连权重也切。每层前向/反向用到谁，就临时 AllGather 谁，用完即扔。显存逼近 16Ψ/N 的理论下限，但通信量变成约 1.5 倍，且 AllGather 在关键路径上 —— 这是用带宽换显存的明码标价。
              </>,
            )}
          </li>
        </ul>
        <ZeroBars />
        <p>
          {t(
            <>
              Drag the slider above and you'll see: the larger N, the shorter the ZeRO-3 bar — at 64 GPUs each card holds only ~1.75GB of state, so training a 7B model on a pile of gaming cards becomes memory-feasible. This is also the essence of PyTorch FSDP (Fully Sharded Data Parallel): it's just a native implementation of ZeRO-3.
            </>,
            <>
              拖一下上面的滑杆能看到：N 越大，ZeRO-3 的柱子越矮，64 卡时每卡只剩约 1.75GB 的状态 —— 7B 模型用一堆游戏卡训练在显存上成立了。这也是 PyTorch FSDP（Fully Sharded Data Parallel）的本质：它就是 ZeRO-3 的原生实现。
            </>,
          )}
        </p>
        <Callout type="insight" title={t('It cuts "redundancy," not "computation"', '切的是「冗余」，不是「计算」')}>
          <p>
            {t(
              <>
                The entire ZeRO family is <em>exactly equivalent</em> to plain DP on the compute graph — every GPU still runs the full model's forward and backward, and the loss curve is identical to the byte. All it cuts is the "N copies of the same thing" redundancy. This gives us an important taxonomy: <strong>DP/ZeRO shard state, TP/PP shard compute</strong>. The former changes neither a single GPU's compute volume nor its activation memory, so the moment a single layer's activations or temporary buffers won't fit, ZeRO can't save you — you have to reach for the other two blades.
              </>,
              <>
                ZeRO 全系列在计算图上和普通 DP <em>完全等价</em> —— 每张卡跑的还是完整模型的前向反向，损失曲线分毫不差。它切掉的只是「N 份一样的东西」这种冗余。这给了我们一个重要的分类法：<strong>DP/ZeRO 切状态，TP/PP 切计算</strong>。前者不改变单卡的计算量和激活内存，所以一旦单层的激活或临时缓冲都放不下，ZeRO 救不了你，必须请出后两把刀。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={3}
        title={t('The fundamental of communication: Ring AllReduce', '通信的基本功：Ring AllReduce')}
        lead={t(
          'Every parallel strategy is built on the same primitive — summing across N GPUs. Its optimal implementation is elegant enough to deserve its own section.',
          '所有并行策略的地基是同一个原语 —— 在 N 张卡之间求和。它的最优实现优雅得值得专门一节。',
        )}
      >
        <p>
          {t(
            <>
              Before TP/PP, we have to pull the word <strong>AllReduce</strong> out of the black box, because it's the shared foundation of both DP gradient sync and TP partial-sum merging. The task is simple: N GPUs each hold a chunk of data of size D (say, their own gradients), and by the end every GPU must have the sum of all N. The dumbest approach picks one "master" GPU to gather and redistribute — its NIC has to push 2(N-1)·D of traffic in and out, getting more congested the more GPUs you add. This is exactly the old wound of the parameter-server architecture.
            </>,
            <>
              往下讲 TP/PP 之前，必须先把 <strong>AllReduce</strong> 这个词从黑盒里拆出来，因为它是 DP 梯度同步和 TP 部分和合并共用的地基。任务很简单：N 张卡各有一份大小为 D 的数据（比如各自的梯度），结束时每张卡都要拿到 N 份之和。最笨的做法是选一张「主卡」收集再分发 —— 主卡的网口要进出 2(N-1)·D 的流量，卡越多越堵，这正是参数服务器（parameter server）架构的旧伤。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>Ring AllReduce</strong> is far prettier: connect the N GPUs into a ring, split each GPU's data into N chunks, then play it in two acts. Act one, <strong>reduce-scatter</strong>, runs N-1 steps: each step every GPU sends one chunk to its downstream neighbor while receiving and accumulating one chunk from upstream. After N-1 steps, each GPU holds exactly the complete sum of <em>one</em> chunk. Act two, <strong>all-gather</strong>, runs another N-1 steps: the finished chunks flow back around the ring, each GPU forwarding one complete chunk per step. The animation below rewards stepping through it click by click: fix your eye on any one chunk and watch it circle once to gather all N contributions, then circle again to reach everyone.
            </>,
            <>
              <strong>Ring AllReduce</strong> 的做法漂亮得多：把 N 张卡连成一个环，每张卡的数据切成 N 块，然后分两幕走。第一幕 <strong>reduce-scatter</strong>，走 N-1 步：每步每张卡把手头一个块发给下家、同时收上家一个块并累加。N-1 步之后，每张卡恰好握有<em>某一个块</em>的完整和。第二幕 <strong>all-gather</strong>，再走 N-1 步：成品块沿环回流，每卡每步转发一个完整块。下面的动画值得你一步一步点：盯住任意一个块，看它怎么转一圈攒齐 N 份贡献、再转一圈送达所有人。
            </>,
          )}
        </p>

        <RingAllReduceLab />

        <p>
          {t(
            <>
              Let's tally the communication. There are 2(N-1) steps in total, and each step every GPU sends only D/N of data, so each GPU's total send volume is:
            </>,
            <>
              算一笔通信账。总共 2(N-1) 步，每步每张卡只发送 D/N 的数据，于是每卡总发送量为：
            </>,
          )}
        </p>
        <MathTex block tex="V_{\text{per-GPU}} = \underbrace{(N-1)\cdot\frac{D}{N}}_{\text{reduce-scatter}} + \underbrace{(N-1)\cdot\frac{D}{N}}_{\text{all-gather}} = 2\,\frac{N-1}{N}\,D \;<\; 2D" />
        <p>
          {t(
            <>
              Note the most counterintuitive — and most elegant — fact in this formula: <strong>per-GPU communication is nearly independent of N</strong>. As N grows from 4 to 1024 the coefficient only creeps from 1.5 to 1.998, forever pinned below 2D. Every link on the ring is saturated at the same moment, and no single GPU is a bottleneck — bandwidth utilization is uniform and optimal (one can prove 2(N-1)D/N is the lower bound on per-GPU AllReduce traffic). The cost is that the step count grows linearly, with one link latency per step, so on very large clusters NCCL switches to latency-friendlier algorithms like tree/double-binary-tree; but the "bandwidth-optimal" property makes the ring the default choice for large messages.
            </>,
            <>
              注意这个式子里最反直觉、也最优雅的事实：<strong>每卡通信量与 N 几乎无关</strong>，N 从 4 涨到 1024，系数只从 1.5 爬到 1.998，永远压在 2D 以下。环上每条链路同一时刻都在满负荷工作，没有任何一张卡是瓶颈 —— 带宽利用是均匀且最优的（可以证明 2(N-1)D/N 是 AllReduce 每卡通信量的下界）。代价是步数线性增长，每步都有一次链路延迟，所以超大集群上 NCCL 会切换到 tree/双二叉树等延迟更优的算法，但「带宽最优」这个性质让 ring 成为大消息的默认选择。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={4}
        title={t('Tensor parallelism: slicing a layer’s matmul', '张量并行：把一层矩阵乘切开')}
        lead={t(
          'The Megatron cut: column-shard W1, row-shard W2 — zero communication between the two layers, with a single sum at the exit of each.',
          'Megatron 式切法：列切 W1、行切 W2，两层之间零通信，每层只在出口处求一次和。',
        )}
      >
        <p>
          {t(
            <>
              ZeRO solves "the state won't fit," but if the single-layer computation itself is too big — say, a 1T model's MLP layer whose weights alone are several GB and whose activation peak is even higher — you have to slice the <em>matmul within a single layer</em> across multiple GPUs. That's <Term t="Splitting a single weight matrix by row or column across GPUs; each computes a partial result, then they communicate to merge. Also called intra-layer model parallelism.">tensor parallelism (TP)</Term>. NVIDIA's Megatron-LM gave a recipe that remains the standard answer to this day.
            </>,
            <>
              ZeRO 解决了「状态放不下」，但如果单层计算本身就太大 —— 比如 1T 模型一层 MLP 的权重就好几个 GB、激活峰值更高 —— 就得把<em>一层之内的矩阵乘法</em>切到多张卡上，这就是<Term t="把单个权重矩阵按行或列切分到多卡，各卡算部分结果再通信合并。也叫 intra-layer 模型并行。">张量并行（Tensor Parallelism, TP）</Term>。NVIDIA 的 Megatron-LM 给出了至今仍是标准答案的切法。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Take the Transformer's MLP: <MathTex tex="Y = \mathrm{GeLU}(XW_1)W_2" />. If you <strong>split W<sub>1</sub> by column</strong> into two halves on two GPUs, each computes a different <em>column</em> set of GeLU(XW<sub>1</sub>) — and since GeLU is element-wise, the two activation halves don't depend on each other, no communication needed. Then <strong>split W<sub>2</sub> by row</strong>: each GPU's "half activation" from the previous step lines up exactly with W<sub>2</sub>'s "half rows," so each GPU produces a <strong>partial sum</strong> of the full Y. Across both layers you only need one AllReduce at the very end to add the partial sums. Column-split paired with row-split, zero communication in between — this is surgical-scalpel-level precise design.
            </>,
            <>
              看 Transformer 的 MLP：<MathTex tex="Y = \mathrm{GeLU}(XW_1)W_2" />。如果把 W<sub>1</sub> <strong>按列切</strong>成两半放到两张卡，每张卡算出 GeLU(XW<sub>1</sub>) 的不同<em>列</em> —— 由于 GeLU 是逐元素的，这两半激活互不依赖，不用通信。接着把 W<sub>2</sub> <strong>按行切</strong>：恰好上一步每张卡手里的「半截激活」正对着 W<sub>2</sub> 的「半截行」，各卡算出的是完整 Y 的一个<strong>部分和（partial sum）</strong>。整个两层的旅程中只需要在最后做一次 AllReduce 把部分和加起来。列切配行切，中间零通信 —— 这是手术刀级别的精确设计。
            </>,
          )}
        </p>
        <CodeBlock code={t(MEGATRON_F_G_EN, MEGATRON_F_G_ZH)} lang="python" title="megatron_mlp.py" />
        <TensorParallelFigure />
        <p>
          {t(
            <>
              The attention layer is even luckier: multi-head attention is inherently a parallel structure, so you simply <strong>split by head</strong> — each GPU gets a subset of heads, computes its own attention, and never has to talk to the others, with a single AllReduce only after the row-split output projection at the end. So a Transformer layer's TP communication ledger is: 2 AllReduces in the forward (one for attention + one for MLP), and 2 more in the backward. An 80-layer model is hundreds of AllReduces per step, every one of them on the <strong>critical path</strong> — compute has to stop and wait for it.
            </>,
            <>
              注意力层更幸运：多头注意力天生就是并行结构，<strong>按 head 切</strong>即可 —— 每张卡分到一部分 head，各自算各自的注意力，互相完全不用说话，只在末尾的输出投影（行切）后做一次 AllReduce。于是一个 Transformer 层的 TP 通信账是：前向 2 次 AllReduce（attention 一次 + MLP 一次），反向再来 2 次。一个 80 层的模型每个 step 就是几百次 AllReduce，每次都在<strong>关键路径</strong>上 —— 计算必须停下来等它。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              This dictates TP's radius of action. Within a box, 8 GPUs have NVLink/NVSwitch — 600GB/s on the A100 generation, 900GB/s on the H100; cross-box over InfiniBand is typically only on the order of ~50GB/s per GPU, about 18× slower. Force the communication that happens every layer to squeeze through the cross-box network and the GPUs will starve waiting. So the industry's iron rule is: <strong>TP ≤ 8, locked inside the NVLink domain</strong>; need more parallelism, reach for a different blade.
            </>,
            <>
              这决定了 TP 的活动半径。机内 8 张卡之间有 NVLink/NVSwitch，A100 代 600GB/s、H100 代 900GB/s；跨机走 InfiniBand 通常只有每卡 ~50GB/s 量级，差着约 18 倍。让每层都发生的通信去挤跨机网络，GPU 会饿死在等待里。所以业界的铁律是：<strong>TP ≤ 8，锁死在 NVLink 域内</strong>；需要更多并行度，找别的刀。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'Why is tensor parallelism (TP) almost never deployed across machines, while data parallelism can be?',
            '为什么张量并行（TP）几乎从不跨机器部署，而数据并行可以？',
          )}
          options={[
            {
              text: t(
                'TP’s implementation relies on NVLink-proprietary instructions, and InfiniBand hardware doesn’t support AllReduce',
                'TP 的实现依赖 NVLink 专有指令，InfiniBand 硬件不支持 AllReduce',
              ),
              explain: t(
                'AllReduce runs over any interconnect (NCCL supports IB well); the issue isn’t "can it" but "how fast."',
                'AllReduce 在任何互联上都能跑（NCCL 对 IB 支持得很好），问题不在「能不能」而在「快不快」。',
              ),
            },
            {
              text: t(
                'TP does multiple AllReduces per layer in both forward and backward, all on the critical path, so communication is extremely frequent and only NVLink-class bandwidth can keep up; DP communicates just once per step and can overlap it with compute',
                'TP 每层前向+反向要做多次 AllReduce、且都在关键路径上，通信极其频繁，只有 NVLink 量级的带宽兜得住；DP 每步只通信一次还能和计算重叠',
              ),
              correct: true,
              explain: t(
                'Exactly the double pressure of frequency × critical path: an 80-layer model does hundreds of synchronous AllReduces per step, and 18× slower bandwidth means the GPUs spend most of their time waiting on the network. DP’s gradient AllReduce happens once per step and can be hidden inside the backward pass.',
                '正是频率 × 关键路径的双重压力：80 层模型每 step 几百次同步 AllReduce，带宽慢 18 倍意味着 GPU 大部分时间在等网。DP 的梯度 AllReduce 每 step 仅一次，还能藏进反向计算里。',
              ),
            },
            {
              text: t(
                'TP loses numerical precision across machines, causing training to not converge',
                '跨机时 TP 的数值精度会下降，导致训练不收敛',
              ),
              explain: t(
                'Communication doesn’t change numerical semantics; a cross-box AllReduce gives a result identical to an in-box one, with no bearing on convergence.',
                '通信不改变数值语义，跨机的 AllReduce 结果和机内完全一致，收敛性无关。',
              ),
            },
            {
              text: t('Because GPU memory can’t be shared across machines', '因为跨机的 GPU 显存不能共享'),
              explain: t(
                'Memory is never shared across GPUs under any parallelism; TP ships activations and partial sums, not memory pages.',
                '任何并行方式下显存都不跨卡共享，TP 传的是激活和部分和，不是显存页。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={5}
        title={t('Pipeline parallelism: cut by layer, then fill the bubble', '流水线并行：切层，然后填满气泡')}
        lead={t(
          'Cut the model by layer into P stages like a factory line — the hard part is not letting any station sit idle.',
          '把模型按层切成 P 段像工厂流水线 —— 难点是别让工位闲着。',
        )}
      >
        <p>
          {t(
            <>
              The third blade is <Term t="Cut the model by layer into P stages, place each stage on a different GPU/machine, and data flows through them in sequence like a pipeline. Also called inter-layer model parallelism.">pipeline parallelism (PP)</Term>: 80 layers cut into 4 stages, each stage of 20 layers living on one machine. Stages only need to pass activations across the boundary once — communication volume is orders of magnitude smaller than TP, which makes PP the natural choice for <strong>cross-machine scaling</strong>. But it has a structural flaw: stage 2 can't start until stage 1 finishes. If the entire batch passes through the pipeline in one shot, only one stage works at any moment while the other P-1 stand around watching — utilization 1/P, dismal.
            </>,
            <>
              第三把刀是<Term t="把模型按层切成 P 段（stage），各段放在不同的卡/机器上，数据像流水线一样依次经过。也叫 inter-layer 模型并行。">流水线并行（Pipeline Parallelism, PP）</Term>：80 层切成 4 段，每段 20 层住一台机器。段与段之间只需要在边界上传一次激活 —— 通信量比 TP 小几个数量级，所以 PP 是<strong>跨机扩展</strong>的天然选择。但它有个结构性缺陷：第 2 段必须等第 1 段算完才能开工。如果整个 batch 一口气穿过流水线，任意时刻只有一段在干活，其余 P-1 段全在围观 —— 利用率 1/P，惨不忍睹。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              GPipe's fix is to chop the batch into M <strong>micro-batches</strong> and stream them through: the moment stage 1 finishes micro-batch 1 it starts micro-batch 2, while stage 2 picks up micro-batch 1. Once the pipeline is full all stages work at once, and only the "fill" at the start and the "drain" at the end sit idle — these gray gaps have a vivid name: <strong>bubbles</strong>. The fraction of total time spent in bubbles has a clean formula:
            </>,
            <>
              GPipe 的解法是把 batch 剁成 M 个 <strong>micro-batch</strong> 流水进去：第 1 段算完第 1 个 micro-batch 就立刻开始第 2 个，同时第 2 段接手第 1 个。流水填满之后所有段同时工作，只有开头的「灌水」和结尾的「排空」存在空转 —— 这些灰色空隙有个形象的名字：<strong>气泡（bubble）</strong>。气泡占总时间的比率有个干净的公式：
            </>,
          )}
        </p>
        <MathTex block tex="\text{bubble ratio} = \frac{P-1}{M + P - 1}" />

        <PipelineBubbleLab />

        <p>
          {t(
            <>
              The formula and the Gantt chart say the same thing: <strong>the deeper P, the costlier the bubble; the more M, the thinner the bubble is spread</strong>. At P=4, M=4 the bubble ratio is 3/7 ≈ 43%, nearly half the compute evaporates; pull M to 16 and it drops to 3/19 ≈ 16%; at M=32 only 9% remains. So the first commandment of PP is: the micro-batch count should be at least 4× P. But GPipe has a second hazard: it computes all M forwards before starting any backward, meaning each stage has to stash M micro-batches' worth of activations waiting for the backward to fetch them — let M grow and memory blows up first. <strong>1F1B</strong> (one-forward-one-backward, from PipeDream) reorders things to "after a few warmup forwards, follow every forward immediately with a backward": in the experiment above, switch to 1F1B and you'll see total time and bubble ratio <em>don't change at all</em>, but the micro-batches in flight per stage drop from M to at most P — activation residency is decoupled from M. This is a free lunch, which is why Megatron and DeepSpeed pipelines default to 1F1B.
            </>,
            <>
              公式和甘特图说的是同一件事：<strong>P 越深气泡越贵，M 越多气泡被摊得越薄</strong>。P=4、M=4 时气泡率 3/7 ≈ 43%，近半算力蒸发；M 拉到 16，降到 3/19 ≈ 16%；M=32 时只剩 9%。所以用 PP 的第一守则是 micro-batch 数至少是 P 的 4 倍以上。但 GPipe 还有第二个隐患：它把所有 M 个前向全部算完才开始反向，意味着每段要同时攒着 M 份 micro-batch 的激活等反向来取 —— M 一大显存先爆了。<strong>1F1B</strong>（one-forward-one-backward，源自 PipeDream）调度把顺序改成「暖机几个前向之后，每做一个前向就紧跟一个反向」：在上面的实验里切到 1F1B 你会看到总时长和气泡率<em>一点没变</em>，但每段同时在飞的 micro-batch 从 M 个降到至多 P 个 —— 激活驻留和 M 解耦了。这是免费的午餐，所以 Megatron、DeepSpeed 的流水线一律默认 1F1B。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'With P=4 stages, M=16 micro-batches, and 1F1B scheduling, what is the bubble ratio approximately?',
            'P=4 个 stage、M=16 个 micro-batch，用 1F1B 调度，气泡率约是多少？',
          )}
          options={[
            {
              text: t('About 25%, since P=4 means a fixed 1/4 waste', '约 25%，因为 P=4 意味着固定浪费 1/4'),
              explain: t(
                '1/P is the utilization disaster when you use no micro-batching at all; once the pipeline is flowing, bubbles only appear in the fill/drain phases.',
                '1/P 是「完全不用 micro-batch」时的利用率灾难，流水起来之后气泡只出现在灌水/排空段。',
              ),
            },
            {
              text: '(P-1)/(M+P-1) = 3/19 ≈ 16%',
              correct: true,
              explain: t(
                'Plug into the formula: (4-1)/(16+4-1) = 3/19 ≈ 15.8%. Note 1F1B doesn’t change the bubble ratio (total time matches GPipe); what it saves is activation memory.',
                '代入公式：(4-1)/(16+4-1) = 3/19 ≈ 15.8%。注意 1F1B 不改变气泡率（总时长与 GPipe 相同），它省的是激活显存。',
              ),
            },
            {
              text: t('0%, 1F1B scheduling eliminates all bubbles', '0%，1F1B 调度消除了所有气泡'),
              explain: t(
                '1F1B only reorders the interleaving of forwards and backwards; the idle time in the fill and drain phases still exists, and the bubble-ratio formula is unchanged.',
                '1F1B 只是重排前向/反向的交错顺序，灌水和排空阶段的空转仍然存在，气泡率公式不变。',
              ),
            },
            {
              text: '3/16 ≈ 19%',
              explain: t(
                'The denominator is M+P-1=19, not M=16: total time includes M units of full-load work plus P-1 units of fill/drain overhead.',
                '分母是 M+P-1=19 而不是 M=16：总时间里除了 M 份满载工作还有 P-1 份灌排开销。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={6}
        title={t('Parallel-strategy sandbox: combining the three blades', '并行策略沙盘：把三把刀组合起来')}
        lead={t(
          'Real large-model training is a 3D split of TP × PP × DP — finding the sweet spot among memory, bandwidth, and utilization.',
          '真实的大模型训练是 TP × PP × DP 的三维切分 —— 在显存、带宽、利用率之间找那个甜点。',
        )}
      >
        <p>
          {t(
            <>
              In practice the three blades are never an either/or — they multiply: first use <strong>TP</strong> inside the NVLink domain (≤8 GPUs) to slice a single layer down to fit and tame the activations; if the layer count still overflows, use <strong>PP</strong> to cut stages across machines (the cheapest cross-machine option); hand all remaining GPUs to <strong>DP</strong> for throughput, then layer on a tier of ZeRO over the DP dimension as memory headroom allows. A classic recipe for training 70B on 64 GPUs is TP8 × PP2 × DP4 + ZeRO-1: each model replica occupies 16 GPUs (two 8-GPU boxes), and 4 replicas chew through data in parallel. The sandbox below lays this ledger out in full — swap the model, swap the card type, drag the three parallelism degrees, and watch when the per-GPU memory stack crosses the red line and where the three kinds of communication pressure land in the network. Start from the three presets, then try to answer: why does 405B absolutely need PP16? Why does 70B inference fit on TP8 in a single box?
            </>,
            <>
              实战中三把刀从来不是单选题，而是乘法：先用 <strong>TP</strong> 在 NVLink 域内（≤8 卡）把单层切到能装下、激活压到可控；层数还溢出就用 <strong>PP</strong> 跨机切段（通信最便宜的跨机方案）；剩下的卡全部交给 <strong>DP</strong> 堆吞吐，再视显存余量给 DP 维叠一档 ZeRO。一个 64 卡训 70B 的经典配方是 TP8 × PP2 × DP4 + ZeRO-1：每个模型副本占 16 卡（两台 8 卡机），4 个副本并行吃数据。下面的沙盘把这本账完整摊开 —— 换模型、换卡型、拖三个并行度，看每卡显存堆叠条什么时候越过红线、三类通信的压力落在哪一层网络上。建议从三个预设开始，然后试着回答：405B 为什么非得 PP16？70B 推理为什么 TP8 一台机器就够？
            </>,
          )}
        </p>

        <StrategySandboxLab />

        <p>
          {t(
            <>
              A few rules of thumb worth taking away from the sandbox: <strong>TP always fills the box first</strong>, because NVLink bandwidth is "use-it-or-lose-it"; <strong>keep PP stages just deep enough</strong>, since cutting too deep raises both bubbles and fill/drain latency and forces a larger M along for the ride; <strong>DP is the source of throughput</strong>, so every "extra GPU left after it fits" should go to it; <strong>ZeRO is free memory on the DP dimension</strong>, Z1 is almost cost-free so turn it on whenever you can, while Z3 makes you weigh the bandwidth. And one easily-overlooked rule: with DP=1, ZeRO does nothing — it cuts redundancy between replicas, and with no replicas there's no redundancy to cut.
            </>,
            <>
              玩沙盘时有几条经验法则值得带走：<strong>TP 永远先填满机内</strong>，因为 NVLink 的带宽是「不用白不用」；<strong>PP 的段数够用就好</strong>，切太深气泡和灌排延迟都涨，还要陪着更大的 M；<strong>DP 是吞吐的来源</strong>，所有「装下之后多出来的卡」都该给它；<strong>ZeRO 在 DP 维上白捡显存</strong>，Z1 几乎无代价应开尽开，Z3 则要掂量带宽。还有一条容易忽略：DP=1 时 ZeRO 完全失效 —— 它切的是副本间冗余，没有副本就没有冗余可切。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={7}
        title={t('Inference side and MoE: same blades, different grip', '推理侧与 MoE：同样的刀，不同的握法')}
        lead={t(
          'Inference has no gradients or optimizer, so the blades’ goal shifts from "fit" to "fast" and "cheap."',
          '推理没有梯度和优化器，刀法的目标从「装下」变成「快」和「便宜」。',
        )}
      >
        <p>
          {t(
            <>
              The inference memory ledger is far lighter (2B/param + KV cache), so the motive for parallelism changes: <strong>in inference, TP buys latency</strong> — 8 GPUs compute a layer at once, dividing single-token decode time by the parallel efficiency, which is crucial for an online service's P99; while <strong>multiple replicas (the inference flavor of DP) buy throughput</strong> — replicas don't even do gradient sync, it's a zero-communication "copy-paste," so add however many replicas and throughput rises almost linearly. So the standard posture for serving 70B online is "TP8, one replica per box, scale out by adding machines when traffic grows." PP's role in inference is awkward: it doesn't lower single-token latency (the token still has to traverse all stages serially), and is mainly for the case where the weights are too big to fit in one box (405B/1T class). A side note on ultra-long context: when a 1-million-token KV cache won't fit on its own, there's a fourth blade — <strong>sequence/context parallelism</strong> — that splits tokens across GPUs along the sequence dimension, working with algorithms like Ring Attention to pass KV blocks around the ring between GPUs.
            </>,
            <>
              推理的显存账轻得多（2B/参数 + KV cache），所以并行的动机变了：<strong>TP 在推理里买的是延迟</strong> —— 8 张卡同时算一层，单 token 的解码时间直接除以并行效率，这对在线服务的 P99 至关重要；而<strong>多副本（即推理版 DP）买的是吞吐</strong> —— 副本之间连梯度同步都没有，是零通信的「复制粘贴」，加多少副本吞吐几乎线性涨。所以 70B 在线服务的标准姿势是「TP8 一台机一个副本，流量大了横向加机器」。PP 在推理里地位尴尬：它不降低单 token 延迟（token 还是要串行走完所有段），主要用于权重大到单机装不下的场景（405B/1T 级）。顺带一提超长上下文：当 100 万 token 的 KV cache 本身就装不下时，还有<strong>序列并行（sequence/context parallelism）</strong>这第四把刀 —— 沿序列维把 token 切到多卡，配合 Ring Attention 之类的算法在卡间环形传递 KV 块。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Finally there's the new dimension that <strong>MoE (Mixture of Experts)</strong> brings. An MoE layer has dozens to hundreds of expert FFNs, and each token is routed to only one or two — huge in parameters but sparse in compute. The natural cut is <strong>expert parallelism (EP)</strong>: spread the experts across different GPUs, and each layer does two <strong>all-to-all</strong> communications to send each token to the GPU holding its expert and collect the result afterward. all-to-all is a more "shattered" communication pattern than AllReduce — traffic depends on where the routing sends things, and the load can be uneven (the GPU holding a popular expert gets swamped). This is exactly where the infrastructure teams behind models like DeepSeek-V3 and Mixtral pour their effort: expert placement, capacity factors, overlapping communication with compute — an engineering world one notch more complex than dense models.
            </>,
            <>
              最后是 <strong>MoE（Mixture of Experts）</strong>带来的新维度。MoE 层里有几十上百个专家（expert）FFN，每个 token 只路由给其中一两个 —— 参数巨大但计算稀疏。自然的切法是<strong>专家并行（Expert Parallelism, EP）</strong>：专家们摊到不同卡上，每层做两次 <strong>all-to-all</strong> 通信，把每个 token 发到它的专家所在的卡、算完再收回来。all-to-all 是比 AllReduce 更「碎」的通信模式 —— 流量取决于路由的去向，负载还可能不均（热门专家所在的卡被挤爆），这正是 DeepSeek-V3、Mixtral 这类模型的基础设施团队花大力气优化的地方：专家放置、容量因子、通信与计算重叠，一个比稠密模型再复杂一档的工程世界。
            </>,
          )}
        </p>
      </Section>

      <Section index={8} title={t('Summary and further reading', '总结与延伸阅读')}>
        <p>{t('The core ledgers and blades of this chapter, distilled to a few lines:', '这一章的核心账目和刀法，浓缩成几条：')}</p>
        <ul>
          <li>
            {t(
              <>
                Mixed-precision training is about <strong>16 bytes per parameter</strong> (2+2+4+4+4), so a 7B model's training state starts at 112GB — not fitting on one card is the first principle of distributed training; inference needs only 2B/param + KV.
              </>,
              <>
                混合精度训练每参数约 <strong>16 字节</strong>（2+2+4+4+4），7B 模型训练态 112GB 起步 —— 单卡装不下是分布式的第一性原理；推理只要 2B/参数 + KV。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>DP</strong> splits data but saves no memory; <strong>ZeRO-1/2/3</strong> progressively shard optimizer states, gradients, then weights across DP GPUs, driving memory toward 1/N while staying exactly equivalent to DP on the compute graph.
              </>,
              <>
                <strong>DP</strong> 切数据不省显存；<strong>ZeRO-1/2/3</strong> 依次把优化器状态、梯度、权重切到 DP 各卡，显存逼近 1/N，计算图与 DP 完全等价。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Ring AllReduce</strong>: 2(N-1) steps, per-GPU traffic 2(N-1)D/N — bounded independently of N and bandwidth-optimal, the shared foundation of both DP and TP.
              </>,
              <>
                <strong>Ring AllReduce</strong>：2(N-1) 步、每卡通信量 2(N-1)D/N —— 与 N 无关地有界且带宽最优，是 DP 和 TP 共同的地基。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>TP</strong> (Megatron) column-split + row-split, 4 critical-path AllReduces per layer per forward+backward → locked inside the NVLink domain (≤8 GPUs); <strong>PP</strong> cuts layers and is the cheapest cross-machine option, bubble ratio (P-1)/(M+P-1), thinned with a large M and saving activations via 1F1B.
              </>,
              <>
                <strong>TP</strong>（Megatron）列切+行切，每层前反向共 4 次关键路径 AllReduce → 锁死 NVLink 域内（≤8 卡）；<strong>PP</strong> 切层跨机最便宜，气泡率 (P-1)/(M+P-1)，用大 M 摊薄、用 1F1B 省激活。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                The combo: TP fills the box → PP fits across machines → all remaining GPUs go to DP (+ZeRO) for throughput; inference uses TP to buy latency and replicas to buy throughput, with MoE adding one more dimension of EP and all-to-all.
              </>,
              <>
                组合拳：TP 填满机内 → PP 跨机装下 → 剩余全给 DP（+ZeRO）堆吞吐；推理用 TP 买延迟、副本买吞吐，MoE 再加一维 EP 与 all-to-all。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Further reading, all first-hand material worth studying closely:', '延伸阅读，都是值得精读的一手材料：')}</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noreferrer">Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism</a>
            {t(
              ' — the original tensor-parallelism paper; its column-split/row-split f/g operator design has yet to be surpassed.',
              ' —— 张量并行的原始论文，列切/行切的 f/g 算子设计至今未被超越。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a>
            {t(
              ' — the memory formulas and communication analysis for the three sharding tiers, the theoretical source of FSDP.',
              ' —— 三档切分的显存公式与通信分析，FSDP 的理论源头。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/1811.06965" target="_blank" rel="noreferrer">GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism</a>
            {t(
              ' — the origin of micro-batch pipelining and bubble analysis.',
              ' —— micro-batch 流水与气泡分析的出处。',
            )}
          </li>
          <li>
            <a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook" target="_blank" rel="noreferrer">HuggingFace Ultra-Scale Playbook</a>
            {t(
              ' — a modern hands-on manual that systematically sweeps TP/PP/DP/ZeRO combinations on real clusters, with plenty of measured curves.',
              ' —— 在真实集群上系统性扫过 TP/PP/DP/ZeRO 组合的现代实战手册，配大量实测曲线。',
            )}
          </li>
          <li>
            <a href="https://www.deepspeed.ai/training/" target="_blank" rel="noreferrer">{t('DeepSpeed Training docs', 'DeepSpeed Training 文档')}</a>
            {t(
              ' — the engineering reference implementation of the full ZeRO family and pipeline parallelism.',
              ' —— ZeRO 全家桶与流水线并行的工程参考实现。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
