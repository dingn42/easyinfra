import { Callout, MathTex, Quiz, Section, Term } from '@/components/ui'
import { LayerFlowLab } from './LayerFlowLab'
import { FlopsExplorer } from './FlopsExplorer'
import { AttnMemoryLab } from './AttnMemoryLab'

export default function Chapter() {
  return (
    <>
      <p>
        打开 LLaMA-7B 的 config.json，里面只有几个不起眼的数字：hidden_size 4096、32 层、FFN 11008、32 个
        head。但「7B」这个名字是怎么从这几个数里长出来的？生成一个 token 到底要烧多少次浮点运算？为什么序列一长，
        显存先于算力爆炸？这些问题听起来要翻论文，其实全部可以心算。Transformer 的奢侈之处恰恰在于它的简单——
        整个模型几乎只由一种运算构成：矩阵乘法。这一章我们把一层 Transformer 拆开，逐项记账，最后你会得到几条
        能在白板上三十秒推完的经验公式。它们是后面所有章节——FlashAttention、KV cache、推理服务、量化——共同的算盘。
      </p>

      <Section
        index={1}
        title="一层的数据流"
        lead="先把一层 decoder 在脑子里过一遍：一个 token 向量进去，绕两圈残差，出来。"
      >
        <p>
          现在主流的大语言模型几乎都是 decoder-only Transformer（仅解码器架构）：同一种层堆 N 次，没有别的花样。
          理解了一层，就理解了整个模型。一层里有两个子结构——注意力（attention）子层和 MLP（多层感知机，也叫
          FFN）子层——它们都挂在同一条<Term t="残差流">residual stream：贯穿所有层的主干向量，每个子层都从它读取、
          向它写回一个增量。可以把它理解成模型的「工作内存总线」。</Term>上。数据流是固定的套路：先做
          RMSNorm（均方根归一化，Root Mean Square Normalization），投影出 Q、K、V，给 Q 和 K 套上
          RoPE（旋转位置编码，Rotary Position Embedding），做分头注意力，过输出投影 O，加回残差；然后第二次
          RMSNorm，进 MLP——LLaMA 用的是 SwiGLU 结构：gate 和 up 两路升维、SiLU 门控逐元素相乘、down 投影降维——
          再加回残差。一层结束，下一层重复。
        </p>
        <p>
          下面这张图是 LLaMA-7B 单层的精确解剖。每个节点都标了类别色，点开能看到张量形状、参数量和单 token 的
          FLOPs（浮点运算次数，Floating Point Operations）。先别管公式，凭直觉感受一件事：
          <strong>哪些节点「重」，哪些节点「轻」</strong>。
        </p>

        <LayerFlowLab />

        <p>
          点过一轮你会发现一个干净得近乎无聊的结论：一层 Transformer 的参数和计算几乎全部落在五个矩阵乘法上——
          QKV 三个投影、O 投影、MLP 的三个投影（gate/up 算两个）。RMSNorm、RoPE、softmax、残差加法这些环节合起来
          连总量的千分之一都不到。这不是巧合，而是这套架构的设计哲学：把所有「重活」都塞进
          GEMM（通用矩阵乘法，General Matrix Multiply），因为第 5 章讲过，GEMM 是 GPU 上唯一能把 Tensor Core
          喂饱的运算。架构里剩下的部分只负责「调味」：归一化保证数值稳定，RoPE 注入位置信息，门控提供非线性。
        </p>
        <Callout type="note" title="为什么 norm、RoPE、softmax 可以从账本上划掉">
          <p>
            这些操作的计算量都正比于 d 或 S（一次方），而 GEMM 正比于 d²。d=4096 时，一次 RMSNorm 约 1.6 万次运算，
            旁边的 QKV 投影是 1 亿次——差了四个数量级。所以算总账时直接忽略它们，误差不会超过千分之一。但注意：
            「FLOPs 可以忽略」不等于「耗时可以忽略」，这些算子是典型的 memory-bound（受显存带宽限制），第 6 章的
            roofline 已经解释过为什么。FLOPs 账和时间账是两本账。
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title="参数账本：6.74B 是怎么数出来的"
        lead="一支笔、四行算式，把 LLaMA-7B 的每个参数数出来。"
      >
        <p>
          先记 attention 子层的账。四个投影矩阵 <MathTex tex="W_Q, W_K, W_V, W_O" /> 在标准多头注意力
          （Multi-Head Attention，MHA）里都是 <MathTex tex="d \times d" /> 的方阵（分头只是把输出向量重新解释成
          32 个 head × 128 维，不改变矩阵大小），所以：
        </p>
        <MathTex block tex="P_{\text{attn}} = 4d^2" />
        <p>
          再记 MLP 的账。SwiGLU 有三个矩阵：gate 和 up 都是 <MathTex tex="d \times d_{ff}" />，down 是{' '}
          <MathTex tex="d_{ff} \times d" />：
        </p>
        <MathTex block tex="P_{\text{mlp}} = 3 \, d \, d_{ff}" />
        <p>
          每层还有两个 RMSNorm（各 d 个缩放参数），层外有词嵌入（embedding）矩阵 <MathTex tex="V \times d" /> 和
          输出端的 lm_head（LLaMA 不共享这两个矩阵，所以各算一份），加上最后一个 norm。合计：
        </p>
        <MathTex block tex="P \;=\; L\left(4d^2 + 3\,d\,d_{ff} + 2d\right) \;+\; 2Vd \;+\; d" />
        <p>代入 LLaMA-7B 的数（d=4096，d_ff=11008，L=32，V=32000）：</p>
        <MathTex block tex="32 \times (\underbrace{4 \cdot 4096^2}_{67.1\text{M}} + \underbrace{3 \cdot 4096 \cdot 11008}_{135.3\text{M}}) + \underbrace{2 \cdot 32000 \cdot 4096}_{262.1\text{M}} + 266{,}240" />
        <MathTex block tex="= 6{,}738{,}415{,}616 \;\approx\; 6.74\text{B} \;\;\checkmark" />
        <p>
          和官方权重 checkpoint 数出来的一个不差。注意账本里的权力结构：MLP 一层 1.35 亿，attention 一层 0.67
          亿——<strong>MLP 拿走了大约三分之二的参数</strong>。「Transformer 的本体是 attention」是个流行的误解：
          按参数算，它的本体是 MLP，attention 只是那个负责在 token 之间搬运信息的、小而精的路由器。
        </p>
        <p>
          这套账还能立刻解释 <Term t="GQA">分组查询注意力（Grouped-Query Attention）：多个 Q head 共享同一组
          K/V head。n_kv=8、n_head=64 时，K/V 投影缩小到原来的 1/8，KV cache 也同比例缩小（第 9 章细讲）。</Term>{' '}
          为什么省参数：K、V 投影的输出维度从 d 变成 <MathTex tex="n_{kv} \cdot d_{head}" />，于是：
        </p>
        <MathTex block tex="P_{\text{attn}}^{\text{GQA}} = 2d^2\left(1 + \tfrac{n_{kv}}{n_{\text{head}}}\right)" />
        <p>
          LLaMA-2 70B 用 n_kv=8、n_head=64，attention 参数从 4d² 降到 2.25d²。用上面的公式算 70B 的总账：
          80 层 × (2.25·8192² + 3·8192·28672) + 2·32000·8192 ≈ 69.0B，又对上了。一套四行的公式，两代模型通吃。
        </p>
      </Section>

      <Section
        index={3}
        title="FLOPs：每个参数两次运算"
        lead="参数量数清楚了，计算量就是免费的——它们之间有一条铁打的换算率。"
      >
        <p>
          矩阵乘法里，每个权重在每次前向中恰好参与一次乘加（multiply-accumulate，MAC）：乘一下，加一下，
          2 FLOPs。一个 token 流过整个模型，每个线性层的权重都被用到恰好一次。所以 decode（逐 token 自回归生成）
          时，单 token 的前向计算量约等于：
        </p>
        <MathTex block tex="F_{\text{decode}} \;\approx\; 2P \;+\; \underbrace{4\,L\,S\,d}_{\text{score}\cdot\text{AV}}" />
        <p>
          第一项是所有 GEMM 的总和——2 倍参数量，与序列长度无关。第二项是 attention 里两次「非参数」的矩阵运算：
          Q 对 S 个缓存 K 做点积（<MathTex tex="2Sd" />），softmax 权重再对 S 个 V 加权求和（又一个{' '}
          <MathTex tex="2Sd" />），每层 <MathTex tex="4Sd" />，乘 L 层。这一项不消耗任何权重，却随上下文长度 S
          线性增长——记住它，它是整章唯一会「长大」的项。对 prefill（一次性并行处理整段 prompt），把 S 个 token
          的账加起来，causal mask 下平均每个 token 看 S/2 的上下文：
        </p>
        <MathTex block tex="F_{\text{prefill}} \;\approx\; 2PS \;+\; 2\,L\,d\,S^2" />
        <Callout type="insight" title="一个参数 = 两次运算">
          <p>
            「2 × 参数量」是整个 LLM 推理经济学的心算基础：7B 模型 decode 一个 token ≈ 14 GFLOPs，70B ≈ 140
            GFLOPs——参数翻 10 倍，每 token 计算量就翻 10 倍，不用查任何表。倒过来用更有杀伤力：一张 A100 的 BF16
            峰值是 312 TFLOPS，纯算力上限是每秒 312e12 ÷ 14e9 ≈ 22000 个 token——而实际单流 decode 只有一百多
            token/s。差出两个数量级的原因不在算力而在带宽，这正是第 6 章 roofline 和第 10 章批处理要回答的问题。
          </p>
        </Callout>
        <p>
          下面的巡览器把这两条公式做成了仪表盘。五根滑杆对应你在任何模型 config 里都能找到的超参，右侧读数实时
          重算。重点看那根堆叠条：它展示单 token 的 FLOPs 流向了哪里。
        </p>

        <FlopsExplorer />

        <p>
          几个值得亲手做的实验：① 保持 LLaMA-7B 预设，把 S 从 128 拉到 32K——<span className="text-amber">score·AV</span>{' '}
          的占比从不到 1% 涨到近六成，反超 MLP，而且这还只是 decode 的账，prefill 里它是平方项，涨得更凶；② 切到 GPT-3
          预设，注意 d 翻 3 倍带来的是 d² 即 9 倍的单层开销；③ 切换 MHA / GQA，参数量动了，但 score·AV 一项纹丝
          不动——GQA 省的是参数和 KV cache 显存，不省 attention 的计算（Q head 一个都没少）。
        </p>

        <Quiz
          question="LLaMA-7B 在 2K 上下文下 decode 一个 token，前向计算量大约是多少？"
          options={[
            { text: '约 1.4 GFLOPs', explain: '少了一个数量级。每个参数要做一次乘加，是 2 FLOPs / 参数，7B 参数就是约 14 GFLOPs。' },
            {
              text: '约 14 GFLOPs',
              correct: true,
              explain:
                '2 × 6.74B ≈ 13.5 GFLOPs，加上 attention 的 4·L·S·d（S=2048 时约 1.1 GFLOPs），合计约 14.5 GFLOPs。「2 倍参数量」这条心算规则值得焊死在脑子里。',
            },
            { text: '约 140 GFLOPs', explain: '这是 70B 模型的量级。FLOPs 与参数量成正比：≈2P。' },
            { text: '约 14 TFLOPs', explain: '差了三个数量级——14 TFLOPs 已经是 prefill 一千个 token 的总账了。decode 单 token 是 2P ≈ 14 GFLOPs。' },
          ]}
        />
      </Section>

      <Section
        index={4}
        title="Attention 的特殊性：S² 的临时账单"
        lead="FLOPs 账本上 attention 是配角，但它有一笔别的运算都没有的隐藏开销。"
      >
        <p>
          到目前为止 attention 看起来人畜无害：参数占三分之一，FLOPs 在短序列下也只是零头。但它和 MLP 有一个本质
          区别——MLP 对每个 token 独立计算，中间张量的大小与 S 无关；attention 却要让每个 token 和前面所有 token
          两两见面。naive（朴素）实现会把这场见面会物化成一个真实的张量：<MathTex tex="A = QK^\top" />，形状{' '}
          <MathTex tex="[B,\, h,\, S,\, S]" />。它不是参数，不进 checkpoint，softmax 和 AV 一做完就被丢掉——
          但在那一瞬间，它真实地躺在显存里，体积是：
        </p>
        <MathTex block tex="M_{\text{score}} = B \cdot h \cdot S^2 \cdot 2\ \text{bytes (fp16)}" />
        <p>
          注意指数：参数与 S 无关，KV cache 和激活随 S 一次方增长，唯独这个临时矩阵是 <strong>S²</strong>。
          平方曲线的可怕之处在于前半段毫无存在感：S=2048 时单层 score 只有 256MB，没人会注意它。把序列拉长
          16 倍到 32K，它就膨胀 256 倍——在下面的实验里，亲眼看它越过两条红线。
        </p>

        <AttnMemoryLab />

        <p>
          S=32K、batch=1 时，单层临时 score 矩阵约 64GB（32 个 head × 32768² × 2 字节）——是 7B 模型全部权重
          （13.5GB）的近五倍，直逼 A100/H100 的 80GB HBM 上限；batch 开到 2 就直接 OOM（Out of Memory，显存
          溢出）。荒谬感正在于此：模型本身不大，计算上也不过是和线性层同量级的 FLOPs，
          <strong>纯粹是一个用完即弃的中间结果撑爆了显存</strong>。更糟的是这个矩阵还要往 HBM 写一遍、softmax
          读一遍、AV 再读一遍——三趟 S² 级别的显存往返，带宽账同样难看。出路不是买更大的卡，而是根本不物化这个
          矩阵：把 attention 拆成小块，在片上 SRAM 里流式地算，score 永远只存活在寄存器和 shared memory 里。
          这就是 FlashAttention，下一章的全部内容。
        </p>

        <Quiz
          question="为什么短序列下 MLP 占计算大头，而超长序列下 attention 反超？"
          options={[
            {
              text: 'score·AV 项（4LSd）随 S 增长，而所有线性层每 token 的 FLOPs 是常数（≈2P）',
              correct: true,
              explain:
                '正是如此。线性层每 token 的计算量由参数量决定，与上下文无关；attention 的 QKᵀ 和 AV 要遍历全部 S 个缓存位置，随 S 线性涨（prefill 总量随 S² 涨）。S 一旦大到 4·S·d 与 6·d·d_ff 同量级（LLaMA-7B 约在 S≈16K 处与 MLP 打平），attention 必然反超。',
            },
            { text: '长序列下 MLP 的参数会被共享，计算量变少', explain: 'MLP 参数从不随 S 变化，每个 token 都完整地过一遍全部 MLP 权重，2·3·d·d_ff FLOPs 一分不少。' },
            { text: 'softmax 是指数运算，比乘法昂贵得多', explain: 'softmax 的 FLOPs 只有 O(h·S)，比旁边 O(S·d) 的点积低两个数量级，从来不是计算瓶颈（它是带宽瓶颈倒是真的，但那是另一本账）。' },
            { text: '长序列下 KV cache 命中率下降，需要重算', explain: 'KV cache 不存在「命中率」——它是精确缓存，从不重算。长序列下它带来的是显存和带宽压力（第 9 章），不改变 FLOPs。' },
          ]}
        />
      </Section>

      <Section
        index={5}
        title="训练 vs 推理：6PT 与两种瓶颈"
        lead="同一套公式，乘上不同的系数，就能横跨训练和推理两个世界。"
      >
        <p>
          训练的账只比推理多一个系数。前向是 2P FLOPs/token；反向传播要算两组梯度——对激活的梯度（让链式法则
          继续往前传）和对权重的梯度，每组都是一次与前向同规模的 GEMM，所以反向 ≈ 4P。合计：
        </p>
        <MathTex block tex="F_{\text{train}} \;\approx\; 6 \, P \, T" />
        <p>
          其中 T 是训练 token 总数。这条经验式出自 Kaplan 等人的 scaling laws 论文，是整个大模型训练预算学的
          地基。验证一下：LLaMA-7B 训了 1T token，6 × 6.74e9 × 1e12 ≈ 4×10²² FLOPs；按 A100 BF16 峰值 312
          TFLOPS、实际 <Term t="MFU">Model FLOPs Utilization：模型实际吞吐折算出的 FLOPs 除以硬件峰值算力。
          大规模训练能跑到 40%~55% 就算优秀，剩下的时间花在通信、数据加载和 memory-bound 算子上。</Term> 约
          45% 计，需要约 8 万 GPU·小时——Meta 论文里报的真实数字是 82,432 GPU·小时。四行公式，误差个位数百分比。
        </p>
        <p>
          推理这边，同样的 FLOPs 公式却劈出两个性格迥异的阶段。<strong>Prefill</strong>：prompt 里的 S 个 token
          一次性并行流过模型，每块权重搬进片上就能复用 S 次，算术强度（arithmetic intensity）高，是
          compute-bound（算力受限）的——roofline 上贴着算力屋顶，Tensor Core 火力全开，和训练前向是同一种负载。
          <strong>Decode</strong>：自回归生成一次只出一个 token，要把 13.5GB 权重从 HBM 完整搬一遍，却只做 2P ≈
          14 GFLOPs——每搬 2 个字节做 2 次运算，算术强度只有 1~2 FLOPs/Byte 的量级，而 A100 的脊点（ridge
          point）在 160 以上。差两个数量级，意味着 decode 时算力利用率常常不到 1%，时间几乎全花在等显存。所以
          单流 decode 的延迟下限可以直接用带宽心算：13.5GB ÷ 1.9TB/s ≈ 7ms/token，约 140 token/s——和实测惊人
          地接近。这本「时间账」第 10 章会逐毫秒展开。
        </p>
        <Callout type="deep" title="为什么说 decode 是「免费的 FLOPs、昂贵的字节」">
          <p>
            把 batch 从 1 加到 32，权重还是只搬一遍，FLOPs 却变成 32 份——算术强度线性上升，GPU 从带宽墙下被推向
            算力屋顶，吞吐近乎白赚 32 倍。这就是推理服务里 batching（批处理）近乎免费的原因，也是 continuous
            batching、PagedAttention 这些系统设计的出发点（第 10 章）。
          </p>
        </Callout>

        <Quiz
          question="训练一个 7B 模型过 1T token，总计算量的量级是？"
          options={[
            { text: '约 4×10²⁰ FLOPs', explain: '少了 100 倍——这大约是只过 10B token 的训练量。训练是 6PT：前向 2PT + 反向 4PT。' },
            {
              text: '约 4×10²² FLOPs',
              correct: true,
              explain: '6 × 6.74×10⁹ × 10¹² ≈ 4×10²² FLOPs。除以单卡有效算力（A100 312 TFLOPS × 45% MFU ≈ 1.4×10¹⁴）≈ 2.9×10⁸ GPU·秒 ≈ 8 万 GPU·小时，与 Meta 报告的 82,432 GPU·小时吻合。',
            },
            { text: '约 4×10²⁴ FLOPs', explain: '多了 100 倍，这已经接近 GPT-4 级别的传闻训练预算了。7B × 1T token 按 6PT 算是 4×10²²。' },
            { text: '无法估算，取决于具体实现', explain: '恰恰相反——6PT 这条经验式与实现无关，只看参数量和数据量，误差通常在 10% 以内。这正是它宝贵的地方。' },
          ]}
        />
      </Section>

      <Section index={6} title="总结与延伸阅读" lead="这一章的全部内容，浓缩成五条可以心算的规则。">
        <ul>
          <li>
            <strong>一层 = 五个 GEMM。</strong>norm、RoPE、softmax 的 FLOPs 可以从账上划掉（但它们的耗时不能——
            它们是 memory-bound 的）。
          </li>
          <li>
            <strong>参数：</strong>
            <MathTex tex="P \approx L(4d^2 + 3\,d\,d_{ff}) + 2Vd" />，代入 LLaMA-7B 得 6,738,415,616，一个不差。
            MLP 占约三分之二；GQA 把 attention 的 4d² 砍成 <MathTex tex="2d^2(1 + n_{kv}/n_h)" />。
          </li>
          <li>
            <strong>计算：</strong>decode 每 token ≈ 2P FLOPs（7B → 14 GFLOPs），外加唯一随 S 增长的 attention
            项 4LSd；prefill ≈ 2PS + 2LdS²。
          </li>
          <li>
            <strong>显存的暗雷：</strong>naive attention 的 S×S score 矩阵随 S² 膨胀，32K 上下文时单层 64GB，
            是权重的近五倍——FlashAttention 因此而生（第 8 章）。
          </li>
          <li>
            <strong>训练 ≈ 6PT；</strong>推理分两段：prefill compute-bound、decode memory-bound（算术强度仅
            1~2 FLOPs/Byte），decode 延迟下限 = 权重字节数 ÷ HBM 带宽。
          </li>
        </ul>
        <p>延伸阅读，按「读完立刻变强」的顺序排：</p>
        <ul>
          <li>
            <a href="https://kipp.ly/transformer-inference-arithmetic/" target="_blank" rel="noreferrer">
              kipply — Transformer Inference Arithmetic
            </a>
            ：本章推理部分的精神源头，把延迟、带宽、batch 的账算到了极致。
          </li>
          <li>
            <a href="https://blog.eleuther.ai/transformer-math/" target="_blank" rel="noreferrer">
              EleutherAI — Transformer Math 101
            </a>
            ：训练侧的对应篇，6PT、训练显存占用、并行开销一网打尽。
          </li>
          <li>
            <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">
              Vaswani et al. — Attention Is All You Need (2017)
            </a>
            ：原始论文。带着本章的账本回去重读 3.2 节，体验完全不同。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noreferrer">
              Touvron et al. — LLaMA: Open and Efficient Foundation Language Models (2023)
            </a>
            ：本章所有数字的出处，附录里有完整的超参表和 GPU·小时账单。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noreferrer">
              Kaplan et al. — Scaling Laws for Neural Language Models (2020)
            </a>
            ：6PT 经验式的出处，也是「规模决定一切」这一信仰的起点。
          </li>
        </ul>
      </Section>
    </>
  )
}
