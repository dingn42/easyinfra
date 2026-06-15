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
            A frontier model like Llama 3.1 405B in BF16 weighs 810GB. A single H100 has 80GB. Forget training — the parameters alone don't fit on one card. That's not an engineering nuisance to route around; it's the founding constraint of the era. <strong>You have to cut the model open and spread it across dozens, sometimes thousands, of GPUs.</strong> And the cut is not "just divide it up." Cut it badly and a thousand GPUs spend 90% of their time waiting on the network; cut it well and the same cluster scales almost linearly. This chapter takes four blades one at a time — data parallelism, ZeRO, tensor parallelism, pipeline parallelism — then hands you a sandbox to carve up a 70B, a 405B, even a 1T model yourself.
          </>,
          <>
            一个前沿模型，比如 Llama 3.1 405B，用 BF16 摆出来是 810GB。一张 H100 只有 80GB。别说训练，连参数本身都放不进一张卡。这不是绕一绕就能躲开的工程麻烦，而是这个时代的奠基约束：<strong>你必须把模型切开，摊到几十张、有时几千张卡上</strong>。而这一刀绝不是「分一分」那么简单。切坏了，一千张卡九成时间在等网络；切好了，同一个集群能跑出接近线性的扩展。这一章把四把刀一把一把讲清楚 —— 数据并行、ZeRO、张量并行、流水线并行 —— 最后给你一个沙盘，亲手切一个 70B、405B 乃至 1T 的模型。
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
          'Before picking up a blade, answer a more basic question: what actually eats the memory? The common intuition runs "a 7B model is 7B params ≈ 14GB, an 80G card has room to spare." That holds for inference. For training it is wrong by an order of magnitude.',
          '动刀之前，先回答一个更基本的问题：显存到底被什么吃掉了？常见的直觉是「7B 模型就是 7B 参数 ≈ 14GB，80G 的卡绰绰有余」。这话对推理成立，对训练则错了整整一个数量级。',
        )}
      </p>

      <Section
        index={1}
        title={t('Where the memory goes: a training ledger', '显存都花在哪：一本训练账')}
        lead={t(
          'In mixed-precision training every parameter drags about 16 bytes of state behind it. The weights are the tip of the iceberg.',
          '混合精度训练里，每个参数背后挂着约 16 字节的状态。权重只是冰山一角。',
        )}
      >
        <p>
          {t(
            <>
              Large-model training almost always runs in <Term t="Forward/backward compute in BF16 (fast, memory-thrifty), but weight updates accumulate in FP32 (so small gradients aren’t swallowed by rounding).">mixed precision</Term> with the Adam optimizer. Carrying one parameter through a full step keeps all of this resident: the BF16 working weight and its gradient at 2 bytes each; an FP32 master copy of the weight at 4 bytes, kept for numerical stability; and Adam's first-moment m and second-moment v, another 4 bytes each in FP32. That adds up to <strong>about 16 bytes per parameter</strong>, 12 of which are optimizer state — pure training overhead.
            </>,
            <>
              大模型训练几乎都用<Term t="前向/反向用 BF16 算（快、省显存），但权重更新用 FP32 累积（避免小梯度被舍入吞掉）。">混合精度（mixed precision）</Term>配 Adam 优化器。把一个参数走完整步，这些东西都得常驻显存：BF16 的工作权重和梯度各 2 字节；为数值稳定保留的一份 FP32 master 权重 4 字节；Adam 的一阶动量 m 和二阶动量 v，又是 FP32 各 4 字节。加起来 <strong>每参数约 16 字节</strong>，其中 12 字节是优化器状态（optimizer states），纯属训练开销。
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
              So a mere 7B model starts at 112GB just for the static training-state ledger, before you add the activations the forward pass saves for the backward. At a 4K sequence, even with FlashAttention sparing you the attention matrix, that's another dozen-plus GB. <strong>A single 80GB A100/H100 can't hold it.</strong> This is the first reason distributed parallelism exists: not speed (that comes second) but the fact that the thing <em>does not fit</em>. The inference-vs-training gulf is worth keeping in mind too. Inference needs 2 bytes per param plus the KV cache (<ChapterLink n={9} />), so 7B inference fits on a single 24G consumer card; training the same model wants eight times the memory.
            </>,
            <>
              所以一个区区 7B 的模型，光是训练态的静态账就 112GB 起步，还没算上前向留给反向的激活（activation）。4K 序列下，即便有 FlashAttention 省掉注意力矩阵，激活也还要十几个 GB。<strong>单张 80GB 的 A100/H100 放不下</strong>。这就是分布式并行存在的第一个理由：不是为了快（那是其次），而是这东西<em>装不下</em>。推理和训练的鸿沟也值得记住：推理每参数 2 字节加上 KV cache（<ChapterLink n={9} />），7B 推理一张消费级 24G 卡就够；训练同一个模型，显存需求是它的八倍。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={2}
        title={t('Data parallelism and ZeRO: shard the state first, then the model', '数据并行与 ZeRO：先切「状态」，再切「模型」')}
        lead={t(
          'DP replicates the model and splits the data. ZeRO notices that most of those 16 bytes don’t need a copy on every GPU.',
          'DP 复制模型、切数据。ZeRO 注意到那 16 字节里大部分不必每张卡都留一份。',
        )}
      >
        <p>
          {t(
            <>
              The most naive multi-GPU scheme is <Term t="Every GPU holds a complete model replica, each processes a different data shard, and gradients are synchronized at the end of every step.">data parallelism (DP)</Term>: 8 GPUs, each with a full model copy, the global batch split into 8 shards computed independently. Forward and backward run in isolation; only at the end of each step do the 8 GPUs run one <strong>AllReduce</strong> on the gradients (sum then average), so everyone updates with the same gradient and the weights stay identical forever. DP is dead simple, and its one communication per step can overlap with the backward pass. The catch is just as blunt: <strong>it saves no memory at all.</strong> Every GPU still carries the full 112GB, so 8 GPUs for a 7B model means all 8 fail to fit, not "1/8 fits."
            </>,
            <>
              最朴素的多卡方案是<Term t="每张卡持有完整模型副本，各自处理不同的数据分片，每步结束后同步梯度。">数据并行（Data Parallelism, DP）</Term>：8 张卡，每张一份完整模型，全局 batch 切成 8 份各算各的。前向反向各自独立，只在每个 step 末尾把 8 张卡的梯度做一次 <strong>AllReduce</strong>（求和取平均），保证大家用同样的梯度更新、权重永远一致。DP 简单到发指，每步仅一次的通信还能和反向计算重叠。代价同样直白：<strong>它一个字节显存都不省。</strong>每张卡照样扛着完整的 112GB，7B 模型 8 卡是「8 份都放不下」，不是「放下了 1/8」。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Microsoft's 2019 <strong>ZeRO</strong> (Zero Redundancy Optimizer) paper called the bluff: across DP's N GPUs sit N <em>identical</em> copies of the optimizer states, gradients, and weights, all pure redundancy. Adam's update is element-wise — each parameter's update depends only on its own m, v, and gradient — so you can shard those states by parameter into N pieces, let each GPU own 1/N, and the math still works. Hence three tiers:
            </>,
            <>
              微软 2019 年的 <strong>ZeRO</strong>（Zero Redundancy Optimizer）论文戳破了这层窗户纸：DP 的 N 张卡上存着 N 份<em>一模一样</em>的优化器状态、梯度和权重，全是纯冗余。Adam 的更新是逐元素的，每个参数的更新只依赖它自己的 m、v 和梯度，那就把这些状态按参数切成 N 份、每张卡只管 1/N，更新照样算得出来。于是有了三档：
            </>,
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>ZeRO-1</strong>: shard only the optimizer states, the big 12B/param chunk. Per-GPU memory drops from 16Ψ to roughly 4Ψ + 12Ψ/N. The cost is almost nil: there's a communication step at the end of the step anyway, so each GPU just broadcasts back the update for the slice it owns.
              </>,
              <>
                <strong>ZeRO-1</strong>：只切优化器状态，也就是 12B/参数那一大块。每卡显存从 16Ψ 降到约 4Ψ + 12Ψ/N。代价几乎为零：反正 step 末尾要通信，顺手把各自负责那段的更新结果广播回去就行。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>ZeRO-2</strong>: shard the gradients too. As the backward pass produces gradients, ReduceScatter hands them straight to the owning GPU, so no GPU keeps the full set. Communication volume matches plain DP.
              </>,
              <>
                <strong>ZeRO-2</strong>：把梯度也切了。反向算出的梯度由 ReduceScatter 直接归给负责的卡，没有哪张卡再留全量。通信量和普通 DP 相当。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>ZeRO-3</strong>: shard the weights as well. Whatever layer a forward or backward needs, you AllGather it on the fly and throw it away after. Memory approaches the theoretical floor of 16Ψ/N, but communication rises to about 1.5× and the AllGather sits on the critical path. That's the sticker price of trading bandwidth for memory.
              </>,
              <>
                <strong>ZeRO-3</strong>：连权重一起切。前向或反向用到哪层，就临时 AllGather 哪层，用完即扔。显存逼近 16Ψ/N 的理论下限，但通信量涨到约 1.5 倍，而且 AllGather 在关键路径上。这就是用带宽换显存的明码标价。
              </>,
            )}
          </li>
        </ul>
        <ZeroBars />
        <p>
          {t(
            <>
              Drag the slider above: the larger N, the shorter the ZeRO-3 bar. At 64 GPUs each card holds about 1.75GB of state, which is what makes training a 7B model on a pile of gaming cards memory-feasible. This is also what PyTorch FSDP (Fully Sharded Data Parallel) is under the hood — a native implementation of ZeRO-3.
            </>,
            <>
              拖一下上面的滑杆：N 越大，ZeRO-3 的柱子越矮，64 卡时每卡只剩约 1.75GB 的状态，这才让 7B 模型用一堆游戏卡训练在显存上成立。PyTorch FSDP（Fully Sharded Data Parallel）骨子里就是这个 —— ZeRO-3 的原生实现。
            </>,
          )}
        </p>
        <Callout type="insight" title={t('It cuts "redundancy," not "computation"', '切的是「冗余」，不是「计算」')}>
          <p>
            {t(
              <>
                On the compute graph the entire ZeRO family is <em>exactly equivalent</em> to plain DP: every GPU still runs the full model's forward and backward, and the loss curve matches to the byte. All ZeRO removes is the "N copies of the same thing" redundancy. That gives us a clean taxonomy: <strong>DP/ZeRO shard state, TP/PP shard compute</strong>. The former touches neither a single GPU's compute volume nor its activation memory, so the moment a single layer's activations or temporary buffers won't fit, ZeRO can't help. You have to reach for the other two blades.
              </>,
              <>
                在计算图上，ZeRO 全系列和普通 DP <em>完全等价</em>：每张卡跑的还是完整模型的前向反向，损失曲线分毫不差。ZeRO 切掉的只是「N 份一样的东西」这种冗余。这给了我们一个干净的分类：<strong>DP/ZeRO 切状态，TP/PP 切计算</strong>。前者不动单卡的计算量和激活内存，所以一旦单层的激活或临时缓冲都放不下，ZeRO 就帮不上忙，得请出后两把刀。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={3}
        title={t('The fundamental of communication: Ring AllReduce', '通信的基本功：Ring AllReduce')}
        lead={t(
          'Every parallel strategy rests on the same primitive: summing across N GPUs. Its optimal implementation is elegant enough to deserve a section of its own.',
          '所有并行策略都建在同一个原语上：在 N 张卡之间求和。它的最优实现优雅得值得单开一节。',
        )}
      >
        <p>
          {t(
            <>
              Before TP and PP, we need to pull the word <strong>AllReduce</strong> out of the black box, because it's the shared foundation of both DP gradient sync and TP partial-sum merging. The task is simple: N GPUs each hold a chunk of data of size D (say, their own gradients), and by the end every GPU must have the sum of all N. The dumbest approach elects one "master" GPU to gather and redistribute. Its NIC then pushes 2(N-1)·D of traffic in and out, and the more GPUs you add the more it congests. This is the old wound of the parameter-server architecture.
            </>,
            <>
              往下讲 TP 和 PP 之前，得先把 <strong>AllReduce</strong> 这个词从黑盒里拆出来，因为它是 DP 梯度同步和 TP 部分和合并共用的地基。任务很简单：N 张卡各有一份大小为 D 的数据（比如各自的梯度），结束时每张卡都要拿到 N 份之和。最笨的做法是选一张「主卡」收集再分发，主卡的网口要进出 2(N-1)·D 的流量，卡越多越堵。这是参数服务器（parameter server）架构的旧伤。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>Ring AllReduce</strong> is far prettier. Connect the N GPUs into a ring, split each GPU's data into N chunks, then run it in two acts. Act one, <strong>reduce-scatter</strong>, takes N-1 steps: on each step every GPU sends one chunk downstream while receiving and accumulating one chunk from upstream. After N-1 steps, each GPU holds the complete sum of exactly <em>one</em> chunk. Act two, <strong>all-gather</strong>, takes another N-1 steps: the finished chunks flow back around the ring, each GPU forwarding one complete chunk per step. The animation below rewards stepping through it click by click. Fix your eye on any single chunk and watch it circle once to gather all N contributions, then circle again to reach everyone.
            </>,
            <>
              <strong>Ring AllReduce</strong> 的做法漂亮得多。把 N 张卡连成一个环，每张卡的数据切成 N 块，分两幕走。第一幕 <strong>reduce-scatter</strong>，走 N-1 步：每步每张卡把手头一个块发给下家，同时收上家一个块并累加。N-1 步之后，每张卡恰好握有<em>某一个块</em>的完整和。第二幕 <strong>all-gather</strong>，再走 N-1 步：成品块沿环回流，每卡每步转发一个完整块。下面的动画值得一步一步点。盯住任意一个块，看它转一圈攒齐 N 份贡献，再转一圈送达所有人。
            </>,
          )}
        </p>

        <RingAllReduceLab />

        <p>
          {t(
            <>
              Tally the communication. There are 2(N-1) steps total, and on each step every GPU sends only D/N of data, so each GPU's total send volume is:
            </>,
            <>
              算一笔通信账。总共 2(N-1) 步，每步每张卡只发送 D/N 的数据，于是每卡总发送量是：
            </>,
          )}
        </p>
        <MathTex block tex="V_{\text{per-GPU}} = \underbrace{(N-1)\cdot\frac{D}{N}}_{\text{reduce-scatter}} + \underbrace{(N-1)\cdot\frac{D}{N}}_{\text{all-gather}} = 2\,\frac{N-1}{N}\,D \;<\; 2D" />
        <p>
          {t(
            <>
              The counterintuitive, elegant fact in this formula: <strong>per-GPU communication is nearly independent of N</strong>. As N grows from 4 to 1024 the coefficient only creeps from 1.5 to 1.998, forever pinned below 2D. Every link on the ring is busy at the same instant and no single GPU is a bottleneck, so bandwidth utilization is uniform and optimal — you can prove 2(N-1)D/N is the lower bound on per-GPU AllReduce traffic. The cost is that the step count grows linearly, one link latency per step, so on very large clusters NCCL switches to latency-friendlier algorithms like tree and double-binary-tree. But being bandwidth-optimal makes the ring the default for large messages.
            </>,
            <>
              这个式子里那个反直觉又优雅的事实：<strong>每卡通信量与 N 几乎无关</strong>。N 从 4 涨到 1024，系数只从 1.5 爬到 1.998，永远压在 2D 以下。环上每条链路同一时刻都在干活，没有哪张卡是瓶颈，带宽利用均匀且最优 —— 可以证明 2(N-1)D/N 是 AllReduce 每卡通信量的下界。代价是步数线性增长，每步都有一次链路延迟，所以超大集群上 NCCL 会切到 tree、双二叉树这类延迟更优的算法。但带宽最优这一点，让 ring 成为大消息的默认选择。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={4}
        title={t('Tensor parallelism: slicing a layer’s matmul', '张量并行：把一层矩阵乘切开')}
        lead={t(
          'The Megatron cut: column-shard W1, row-shard W2. Zero communication between the two layers, with a single sum at each exit.',
          'Megatron 式切法：列切 W1、行切 W2。两层之间零通信，每层只在出口求一次和。',
        )}
      >
        <p>
          {t(
            <>
              ZeRO solves "the state won't fit." But when the single-layer computation itself is too big — say a 1T model's MLP layer, whose weights alone run several GB and whose activation peak is higher still — you have to slice the <em>matmul within one layer</em> across multiple GPUs. That's <Term t="Splitting a single weight matrix by row or column across GPUs; each computes a partial result, then they communicate to merge. Also called intra-layer model parallelism.">tensor parallelism (TP)</Term>. NVIDIA's Megatron-LM gave a recipe that's still the standard answer.
            </>,
            <>
              ZeRO 解决了「状态放不下」。但当单层计算本身就太大 —— 比如 1T 模型一层 MLP，光权重就好几个 GB、激活峰值更高 —— 就得把<em>一层之内的矩阵乘法</em>切到多张卡上，这就是<Term t="把单个权重矩阵按行或列切分到多卡，各卡算部分结果再通信合并。也叫 intra-layer 模型并行。">张量并行（Tensor Parallelism, TP）</Term>。NVIDIA 的 Megatron-LM 给出的切法，至今仍是标准答案。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Take the Transformer's MLP: <MathTex tex="Y = \mathrm{GeLU}(XW_1)W_2" />. <strong>Split W<sub>1</sub> by column</strong> into two halves on two GPUs and each computes a different <em>column</em> set of GeLU(XW<sub>1</sub>); since GeLU is element-wise, the two activation halves don't depend on each other, so no communication. Then <strong>split W<sub>2</sub> by row</strong>: each GPU's half-activation from the previous step lines up exactly with W<sub>2</sub>'s half-rows, and each GPU produces a <strong>partial sum</strong> of the full Y. Across both layers you need a single AllReduce at the very end to add the partial sums. Column-split paired with row-split, nothing communicated in between. The design is surgical.
            </>,
            <>
              看 Transformer 的 MLP：<MathTex tex="Y = \mathrm{GeLU}(XW_1)W_2" />。把 W<sub>1</sub> <strong>按列切</strong>成两半放到两张卡，每张卡算出 GeLU(XW<sub>1</sub>) 的不同<em>列</em>；GeLU 逐元素，这两半激活互不依赖，不用通信。接着把 W<sub>2</sub> <strong>按行切</strong>：上一步每张卡手里的半截激活，恰好对着 W<sub>2</sub> 的半截行，各卡算出的是完整 Y 的一个<strong>部分和（partial sum）</strong>。整个两层只需在最后做一次 AllReduce 把部分和加起来。列切配行切，中间什么都不传。这个设计精准得像手术刀。
            </>,
          )}
        </p>
        <CodeBlock code={t(MEGATRON_F_G_EN, MEGATRON_F_G_ZH)} lang="python" title="megatron_mlp.py" />
        <TensorParallelFigure />
        <p>
          {t(
            <>
              Attention is even luckier. Multi-head attention is already a parallel structure, so you <strong>split by head</strong>: each GPU takes a subset of heads, computes its own attention, never talks to the others, with one AllReduce after the row-split output projection at the end. So a Transformer layer's TP communication ledger is 2 AllReduces in the forward (one for attention, one for the MLP) and 2 more in the backward. An 80-layer model runs hundreds of AllReduces per step, every one of them on the <strong>critical path</strong>: compute stops and waits for it.
            </>,
            <>
              注意力层更幸运。多头注意力本就是并行结构，<strong>按 head 切</strong>即可：每张卡分到一部分 head，各算各的注意力，互不说话，只在末尾的输出投影（行切）后做一次 AllReduce。于是一个 Transformer 层的 TP 通信账是：前向 2 次 AllReduce（attention 一次、MLP 一次），反向再来 2 次。一个 80 层的模型每 step 就是几百次 AllReduce，每次都在<strong>关键路径</strong>上：计算停下来等它。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              This sets TP's radius of action. Within a box, 8 GPUs share NVLink/NVSwitch — 600GB/s on the A100 generation, 900GB/s on the H100, and the Blackwell generation (B200/GB200) pushes it higher again. Cross-box over InfiniBand is typically on the order of ~50GB/s per GPU, roughly 18× slower. Force the per-layer communication through the cross-box network and the GPUs starve waiting. Hence the industry's iron rule: <strong>TP ≤ 8, locked inside the NVLink domain.</strong> Need more parallelism, reach for a different blade.
            </>,
            <>
              这定下了 TP 的活动半径。机内 8 张卡之间有 NVLink/NVSwitch，A100 代 600GB/s、H100 代 900GB/s，到 Blackwell 代（B200/GB200）又往上抬了一截。跨机走 InfiniBand 通常只有每卡 ~50GB/s 量级，慢了约 18 倍。让每层都要发生的通信去挤跨机网络，GPU 会饿死在等待里。所以业界铁律是：<strong>TP ≤ 8，锁死在 NVLink 域内。</strong>需要更多并行度，找别的刀。
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
                'The double pressure of frequency × critical path: an 80-layer model fires hundreds of synchronous AllReduces per step, and at 18× slower bandwidth the GPUs spend most of their time waiting on the network. DP’s gradient AllReduce happens once per step and hides inside the backward pass.',
                '这是频率 × 关键路径的双重压力：80 层模型每 step 几百次同步 AllReduce，带宽慢 18 倍，GPU 大部分时间都在等网。DP 的梯度 AllReduce 每 step 仅一次，还能藏进反向计算里。',
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
          'Cut the model by layer into P stages, like a factory line. The hard part is keeping every station busy.',
          '把模型按层切成 P 段，像工厂流水线。难点是别让任何一个工位闲着。',
        )}
      >
        <p>
          {t(
            <>
              The third blade is <Term t="Cut the model by layer into P stages, place each stage on a different GPU/machine, and data flows through them in sequence like a pipeline. Also called inter-layer model parallelism.">pipeline parallelism (PP)</Term>: 80 layers cut into 4 stages, each stage of 20 layers living on one machine. Stages only pass activations across the boundary once, so the communication volume is orders of magnitude smaller than TP. That makes PP the natural choice for <strong>cross-machine scaling</strong>. It has one structural flaw: stage 2 can't start until stage 1 finishes. Push the whole batch through the pipeline in one shot and only one stage works at a time while the other P-1 stand around watching — utilization 1/P, dismal.
            </>,
            <>
              第三把刀是<Term t="把模型按层切成 P 段（stage），各段放在不同的卡/机器上，数据像流水线一样依次经过。也叫 inter-layer 模型并行。">流水线并行（Pipeline Parallelism, PP）</Term>：80 层切成 4 段，每段 20 层住一台机器。段与段之间只在边界上传一次激活，通信量比 TP 小几个数量级，所以 PP 是<strong>跨机扩展</strong>的天然选择。它有一个结构性缺陷：第 2 段必须等第 1 段算完才能开工。整个 batch 一口气穿过流水线，任意时刻只有一段在干活，其余 P-1 段全在围观，利用率 1/P，惨不忍睹。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              GPipe's fix is to chop the batch into M <strong>micro-batches</strong> and stream them through. The moment stage 1 finishes micro-batch 1 it starts micro-batch 2, while stage 2 picks up micro-batch 1. Once the pipeline fills, all stages work at once; only the fill at the start and the drain at the end sit idle. Those gray gaps have a vivid name: <strong>bubbles</strong>. The fraction of total time lost to bubbles has a clean formula:
            </>,
            <>
              GPipe 的解法是把 batch 剁成 M 个 <strong>micro-batch</strong> 流水进去。第 1 段算完第 1 个 micro-batch 就立刻开始第 2 个，同时第 2 段接手第 1 个。流水填满后所有段同时工作，只有开头的灌水和结尾的排空在空转。这些灰色空隙有个形象的名字：<strong>气泡（bubble）</strong>。气泡占总时间的比率有个干净的公式：
            </>,
          )}
        </p>
        <MathTex block tex="\text{bubble ratio} = \frac{P-1}{M + P - 1}" />

        <PipelineBubbleLab />

        <p>
          {t(
            <>
              The formula and the Gantt chart say the same thing: <strong>the deeper P, the costlier the bubble; the more M, the thinner it spreads</strong>. At P=4, M=4 the bubble ratio is 3/7 ≈ 43%, nearly half the compute gone; pull M to 16 and it drops to 3/19 ≈ 16%; at M=32 only 9% remains. So PP's first rule of thumb is to keep the micro-batch count at least 4× P. GPipe carries a second hazard, though: it runs all M forwards before any backward, so each stage stashes M micro-batches' worth of activations waiting for the backward to fetch them, and a large M blows up memory before it buys you anything. <strong>1F1B</strong> (one-forward-one-backward, from PipeDream) reorders the schedule so that after a few warmup forwards, every forward is followed immediately by a backward. Switch the experiment above to 1F1B and total time and bubble ratio <em>don't budge</em>, but the micro-batches in flight per stage drop from M to at most P — activation residency is decoupled from M. That's a free lunch, which is why Megatron and DeepSpeed pipelines default to 1F1B.
            </>,
            <>
              公式和甘特图说的是同一件事：<strong>P 越深气泡越贵，M 越多气泡摊得越薄</strong>。P=4、M=4 时气泡率 3/7 ≈ 43%，近半算力蒸发；M 拉到 16，降到 3/19 ≈ 16%；M=32 时只剩 9%。所以用 PP 的第一条经验是 micro-batch 数至少取 P 的 4 倍。GPipe 还有第二个隐患：它把所有 M 个前向算完才开始反向，每段要同时攒着 M 份 micro-batch 的激活等反向来取，M 一大显存先爆了。<strong>1F1B</strong>（one-forward-one-backward，源自 PipeDream）把调度顺序改成：暖机几个前向之后，每做一个前向就紧跟一个反向。把上面的实验切到 1F1B，总时长和气泡率<em>纹丝不动</em>，但每段同时在飞的 micro-batch 从 M 个降到至多 P 个，激活驻留和 M 解耦了。这是免费的午餐，所以 Megatron、DeepSpeed 的流水线一律默认 1F1B。
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
          'Real large-model training is a 3D split of TP × PP × DP, finding the sweet spot among memory, bandwidth, and utilization.',
          '真实的大模型训练是 TP × PP × DP 的三维切分，在显存、带宽、利用率之间找那个甜点。',
        )}
      >
        <p>
          {t(
            <>
              In practice the three blades are never an either/or; they multiply. Use <strong>TP</strong> inside the NVLink domain (≤8 GPUs) to slice a single layer down to fit and tame the activations. If the layer count still overflows, use <strong>PP</strong> to cut stages across machines, the cheapest cross-machine option. Hand all remaining GPUs to <strong>DP</strong> for throughput, then layer a tier of ZeRO over the DP dimension as memory headroom allows. A classic recipe for training 70B on 64 GPUs is TP8 × PP2 × DP4 + ZeRO-1: each model replica occupies 16 GPUs (two 8-GPU boxes), and 4 replicas chew through data in parallel. The sandbox below lays the whole ledger out. Swap the model, swap the card, drag the three parallelism degrees, and watch when the per-GPU memory stack crosses the red line and where the three kinds of communication pressure land on the network. Start from the three presets, then try to answer: why does 405B need PP16? Why does 70B inference fit on TP8 in a single box?
            </>,
            <>
              实战中三把刀从来不是单选题，而是乘法。先用 <strong>TP</strong> 在 NVLink 域内（≤8 卡）把单层切到能装下、激活压到可控；层数还溢出就用 <strong>PP</strong> 跨机切段，这是最便宜的跨机方案；剩下的卡全交给 <strong>DP</strong> 堆吞吐，再视显存余量给 DP 维叠一档 ZeRO。一个 64 卡训 70B 的经典配方是 TP8 × PP2 × DP4 + ZeRO-1：每个模型副本占 16 卡（两台 8 卡机），4 个副本并行吃数据。下面的沙盘把整本账摊开。换模型、换卡型、拖三个并行度，看每卡显存堆叠条什么时候越过红线、三类通信的压力落在哪一层网络上。先从三个预设开始，再试着回答：405B 为什么得用 PP16？70B 推理为什么 TP8 一台机器就够？
            </>,
          )}
        </p>

        <StrategySandboxLab />

        <p>
          {t(
            <>
              A few rules of thumb worth taking from the sandbox. <strong>TP fills the box first</strong>, because NVLink bandwidth is use-it-or-lose-it. <strong>Keep PP stages just deep enough</strong>: cut too deep and you raise both bubbles and fill/drain latency, and you drag a larger M along for the ride. <strong>DP is the source of throughput</strong>, so every GPU left over once the model fits should go to it. <strong>ZeRO is free memory on the DP dimension</strong>: Z1 is almost cost-free, so turn it on whenever you can, while Z3 makes you weigh the bandwidth. And one rule people miss: with DP=1, ZeRO does nothing. It cuts redundancy between replicas, and with no replicas there's no redundancy to cut.
            </>,
            <>
              玩沙盘能带走几条经验法则。<strong>TP 先填满机内</strong>，因为 NVLink 的带宽不用白不用。<strong>PP 的段数够用就好</strong>：切太深，气泡和灌排延迟都涨，还得陪着更大的 M。<strong>DP 是吞吐的来源</strong>，模型装下之后多出来的卡都该给它。<strong>ZeRO 在 DP 维上白捡显存</strong>：Z1 几乎无代价，能开就开，Z3 则要掂量带宽。还有一条容易漏掉：DP=1 时 ZeRO 完全失效，它切的是副本间冗余，没有副本就没有冗余可切。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={7}
        title={t('Inference side and MoE: same blades, different grip', '推理侧与 MoE：同样的刀，不同的握法')}
        lead={t(
          'Inference has no gradients or optimizer, so the goal of the blades shifts from "fit" to "fast" and "cheap."',
          '推理没有梯度和优化器，刀法的目标从「装下」变成「快」和「便宜」。',
        )}
      >
        <p>
          {t(
            <>
              The inference memory ledger is far lighter (2B/param + KV cache), so the motive for parallelism changes. <strong>In inference, TP buys latency</strong>: 8 GPUs compute a layer at once, dividing single-token decode time by the parallel efficiency, which is what an online service's P99 lives or dies on. <strong>Replicas — the inference flavor of DP — buy throughput</strong>: replicas don't even sync gradients, it's a zero-communication copy-paste, so add replicas and throughput rises almost linearly. The standard posture for serving 70B online is TP8, one replica per box, scale out by adding machines as traffic grows. PP sits awkwardly in inference: it doesn't lower single-token latency, since the token still traverses all stages serially, and it's mainly there for when the weights are too big for one box (405B/1T class). A side note on ultra-long context: when a million-token KV cache won't fit on its own, there's a fourth blade, <strong>sequence/context parallelism</strong>, which splits tokens across GPUs along the sequence dimension and works with algorithms like Ring Attention to pass KV blocks around the ring.
            </>,
            <>
              推理的显存账轻得多（2B/参数 + KV cache），所以并行的动机变了。<strong>TP 在推理里买的是延迟</strong>：8 张卡同时算一层，单 token 的解码时间直接除以并行效率，在线服务的 P99 全靠它。<strong>多副本（即推理版 DP）买的是吞吐</strong>：副本之间连梯度同步都没有，是零通信的复制粘贴，加多少副本吞吐就几乎线性涨。所以 70B 在线服务的标准姿势是 TP8、一台机一个副本，流量大了横向加机器。PP 在推理里位置尴尬：它不降低单 token 延迟，token 还是要串行走完所有段，主要用在权重大到单机装不下的场景（405B/1T 级）。顺带一提超长上下文：当 100 万 token 的 KV cache 本身就装不下时，还有第四把刀，<strong>序列并行（sequence/context parallelism）</strong>，沿序列维把 token 切到多卡，配合 Ring Attention 之类的算法在卡间环形传递 KV 块。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Then there's the new dimension <strong>MoE (Mixture of Experts)</strong> brings, now standard at frontier scale. An MoE layer has dozens to hundreds of expert FFNs, and each token is routed to only one or two: huge in parameters, sparse in compute. The natural cut is <strong>expert parallelism (EP)</strong>: spread the experts across GPUs, and each layer does two <strong>all-to-all</strong> communications to send each token to the GPU holding its expert and gather the result back. all-to-all is a more shattered pattern than AllReduce. Traffic depends on where the routing sends things, and the load can be uneven — the GPU holding a popular expert gets swamped. This is where the infra teams behind models like DeepSeek-V3 and Mixtral pour their effort: expert placement, capacity factors, overlapping communication with compute. It's an engineering world a notch more complex than dense models.
            </>,
            <>
              最后是 <strong>MoE（Mixture of Experts）</strong>带来的新维度，如今在前沿规模上已是标配。MoE 层里有几十上百个专家（expert）FFN，每个 token 只路由给其中一两个：参数巨大，计算稀疏。自然的切法是<strong>专家并行（Expert Parallelism, EP）</strong>：专家摊到不同卡上，每层做两次 <strong>all-to-all</strong> 通信，把每个 token 发到它的专家所在的卡、算完再收回来。all-to-all 比 AllReduce 更碎。流量取决于路由的去向，负载还可能不均，热门专家所在的卡会被挤爆。这正是 DeepSeek-V3、Mixtral 这类模型的基础设施团队花大力气的地方：专家放置、容量因子、通信与计算重叠。这是比稠密模型再复杂一档的工程世界。
            </>,
          )}
        </p>
      </Section>

      <Section index={8} title={t('Summary and further reading', '总结与延伸阅读')}>
        <p>{t('The core ledgers and blades of this chapter, in a few lines:', '这一章的核心账目和刀法，几条话讲完：')}</p>
        <ul>
          <li>
            {t(
              <>
                Mixed-precision training is about <strong>16 bytes per parameter</strong> (2+2+4+4+4), so a 7B model's training state starts at 112GB. Not fitting on one card is the first reason distributed training exists. Inference needs only 2B/param + KV.
              </>,
              <>
                混合精度训练每参数约 <strong>16 字节</strong>（2+2+4+4+4），7B 模型训练态 112GB 起步。单卡装不下，是分布式训练存在的第一个理由。推理只要 2B/参数 + KV。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>DP</strong> splits data but saves no memory. <strong>ZeRO-1/2/3</strong> progressively shard optimizer states, gradients, then weights across DP GPUs, driving memory toward 1/N while staying exactly equivalent to DP on the compute graph.
              </>,
              <>
                <strong>DP</strong> 切数据，不省显存。<strong>ZeRO-1/2/3</strong> 依次把优化器状态、梯度、权重切到 DP 各卡，显存逼近 1/N，计算图与 DP 完全等价。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Ring AllReduce</strong>: 2(N-1) steps, per-GPU traffic 2(N-1)D/N, bounded independently of N and bandwidth-optimal. It's the shared foundation of both DP and TP.
              </>,
              <>
                <strong>Ring AllReduce</strong>：2(N-1) 步，每卡通信量 2(N-1)D/N，与 N 无关地有界且带宽最优。它是 DP 和 TP 共同的地基。
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
        <p>{t('Further reading, all first-hand material worth reading closely:', '延伸阅读，都是值得精读的一手材料：')}</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noreferrer">Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism</a>
            {t(
              ' — the original tensor-parallelism paper. Its column-split/row-split f/g operator design still hasn’t been beaten.',
              ' —— 张量并行的原始论文，列切/行切的 f/g 算子设计至今没被超越。',
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
