import { Callout, ChapterLink, HardwareBaseline, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { LayerFlowLab } from './LayerFlowLab'
import { FlopsExplorer } from './FlopsExplorer'
import { AttnMemoryLab } from './AttnMemoryLab'

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            Open LLaMA-7B&apos;s config.json and you get a handful of unremarkable numbers: hidden_size 4096, 32 layers,
            FFN 11008, 32 heads. So where does the &quot;7B&quot; in the name come from? How many floating-point
            operations does it actually burn to produce one token? And why, as the sequence grows, does memory blow up
            before compute does? These sound like questions for a paper-reading session. You can answer all three in
            your head. A Transformer is almost embarrassingly simple under the hood: peel back the architecture and it
            is one operation repeated, matrix multiply, with a thin layer of seasoning on top. The same shape powers
            every model people actually run today, from Llama to DeepSeek-V3 to the frontier systems behind chat
            assistants. In this chapter we crack open a single layer and bookkeep it term by term. You walk away with a
            few back-of-the-envelope formulas you can rederive on a whiteboard in thirty seconds, and they turn out to
            be the shared abacus behind everything that follows: FlashAttention, KV cache, inference serving,
            quantization.
          </>,
          <>
            打开 LLaMA-7B 的 config.json，里面只有几个不起眼的数字：hidden_size 4096、32 层、FFN 11008、32 个
            head。但「7B」这个名字是怎么从这几个数里长出来的？生成一个 token 到底要烧多少次浮点运算？为什么序列一长，
            显存会先于算力爆炸？这些问题听起来得翻论文，但三个都能心算出来。Transformer 拆开看简单得有点过分：剥掉外壳，
            它就是一种运算的反复堆叠——矩阵乘法，外面再裹一层薄薄的调味。今天大家真正在跑的模型几乎都是这个形状，从
            Llama 到 DeepSeek-V3，再到聊天助手背后的前沿系统。这一章我们把一层拆开，逐项记账，最后你会得到几条能在
            白板上三十秒推完的经验公式。它们是后面所有章节共同的算盘：FlashAttention、KV cache、推理服务、量化。
          </>,
        )}
      </p>

      <HardwareBaseline ids={['a100']} />

      <Section
        index={1}
        title={t('The data flow of one layer', '一层的数据流')}
        lead={t(
          'Walk through one decoder layer in your head first: a token vector goes in, loops through two residual adds, and comes out.',
          '先把一层 decoder 在脑子里过一遍：一个 token 向量进去，绕两圈残差，出来。',
        )}
      >
        <p>
          {t(
            <>
              Almost every LLM in production is a decoder-only Transformer: one kind of layer stacked N times, and
              that&apos;s the whole architecture. Understand one layer and you understand the model. A layer has two
              substructures, the attention sublayer and the MLP (multi-layer perceptron, also called FFN) sublayer, and
              both hang off the same{' '}
              <Term t="residual stream">
                the trunk vector running through every layer; each sublayer reads from it and writes a delta back.
                Think of it as the model&apos;s working-memory bus.
              </Term>
              . The data flow follows a fixed routine. First RMSNorm (Root Mean Square Normalization), then project out
              Q, K, V, apply RoPE (Rotary Position Embedding) to Q and K, run multi-head attention, push the result
              through the output projection O, and add it back to the residual. Then a second RMSNorm and into the MLP.
              LLaMA uses a SwiGLU block: gate and up project upward along two paths, SiLU gating multiplies them
              elementwise, and the down projection brings the dimension back down before adding to the residual a
              second time. One layer done, and the next one repeats it verbatim.
            </>,
            <>
              现在跑在生产里的大语言模型几乎都是 decoder-only Transformer（仅解码器架构）：同一种层堆 N 次，整个架构
              就这么多。看懂一层，就看懂了整个模型。一层里有两个子结构，注意力（attention）子层和 MLP（多层感知机，
              也叫 FFN）子层，它们都挂在同一条
              <Term t="残差流">
                贯穿所有层的主干向量，每个子层都从它读取、向它写回一个增量。可以把它理解成模型的「工作内存总线」。
              </Term>
              上。数据流走的是一套固定流程。先做 RMSNorm（均方根归一化，Root Mean Square Normalization），投影出 Q、
              K、V，给 Q 和 K 套上 RoPE（旋转位置编码，Rotary Position Embedding），做分头注意力，结果过输出投影 O，
              加回残差。然后第二次 RMSNorm，进 MLP。LLaMA 用的是 SwiGLU 块：gate 和 up 两路升维，SiLU 门控逐元素相乘，
              down 投影再降维，最后第二次加回残差。一层结束，下一层一模一样地再来一遍。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The diagram below dissects one LLaMA-7B layer. Every node is color-coded by category; click one to see its
              tensor shape, parameter count, and single-token FLOPs. Forget the formulas for now and just get a feel for
              one thing: <strong>which nodes are &quot;heavy&quot; and which are &quot;light&quot;</strong>.
            </>,
            <>
              下面这张图把 LLaMA-7B 单层解剖开来。每个节点都标了类别色，点开能看到张量形状、参数量和单 token 的
              FLOPs（浮点运算次数，Floating Point Operations）。先别管公式，就凭直觉感受一件事：
              <strong>哪些节点「重」，哪些节点「轻」</strong>。
            </>,
          )}
        </p>

        <LayerFlowLab />

        <p>
          {t(
            <>
              Click through once and the conclusion is clean to the point of boredom: a Transformer layer&apos;s
              parameters and compute land almost entirely on five matrix multiplies. The three QKV projections, the O
              projection, and the MLP&apos;s three projections (gate and up count as two). RMSNorm, RoPE, softmax, and
              the residual adds together don&apos;t reach one part in a thousand of the total. That split is no accident.
              It is the design philosophy of the architecture: shove all the heavy lifting into GEMM (General Matrix
              Multiply), because, as <ChapterLink n={5} /> showed, GEMM is the only operation on a GPU that can keep the
              Tensor Cores fed. Everything else is seasoning. Normalization keeps the numerics stable, RoPE injects
              positional information, gating supplies the nonlinearity.
            </>,
            <>
              点过一轮，你会得到一个干净到有点无聊的结论：一层 Transformer 的参数和计算几乎全压在五个矩阵乘法上。
              QKV 三个投影、O 投影、MLP 的三个投影（gate 和 up 算两个）。RMSNorm、RoPE、softmax、残差加法这些环节
              加起来还不到总量的千分之一。这种分布不是偶然，而是这套架构的设计取向：把所有重活都塞进
              GEMM（通用矩阵乘法，General Matrix Multiply），因为<ChapterLink n={5} />讲过，GEMM 是 GPU 上唯一能把
              Tensor Core 喂饱的运算。剩下的部分都只是调味。归一化保证数值稳定，RoPE 注入位置信息，门控提供非线性。
            </>,
          )}
        </p>
        <Callout type="note" title={t('Why norm, RoPE, and softmax fall off the ledger', '为什么 norm、RoPE、softmax 可以从账本上划掉')}>
          <p>
            {t(
              <>
                Their compute scales with d or S (linear), while GEMM scales with d². At d=4096 one RMSNorm is about 16K
                operations; the QKV projection right next to it is 100 million, four orders of magnitude more. So when
                you tally the books you can just drop them and the error stays under one part in a thousand. The catch:
                negligible FLOPs do not mean negligible time. These operators are textbook memory-bound, and{' '}
                <ChapterLink n={6} />&apos;s roofline already explained why. FLOPs and time are two separate ledgers.
              </>,
              <>
                它们的计算量正比于 d 或 S（一次方），而 GEMM 正比于 d²。d=4096 时，一次 RMSNorm 约 1.6 万次运算，
                旁边的 QKV 投影是 1 亿次，差了四个数量级。所以算总账时把它们直接划掉，误差也不会超过千分之一。但有个
                陷阱：FLOPs 可以忽略，不等于耗时可以忽略。这些算子是典型的 memory-bound（受显存带宽限制），<ChapterLink n={6} />的
                roofline 已经讲过为什么。FLOPs 和时间是两本账。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title={t('The parameter ledger: how do you get to 6.74B?', '参数账本：6.74B 是怎么数出来的')}
        lead={t(
          'One pen, four lines of arithmetic, and you count every parameter in LLaMA-7B.',
          '一支笔、四行算式，把 LLaMA-7B 的每个参数数出来。',
        )}
      >
        <p>
          {t(
            <>
              Start with the attention sublayer. In standard multi-head attention (MHA), the four projection matrices{' '}
              <MathTex tex="W_Q, W_K, W_V, W_O" /> are all <MathTex tex="d \times d" /> square matrices (splitting into
              heads just reinterprets the output vector as 32 heads × 128 dims; it doesn&apos;t change the matrix size),
              so:
            </>,
            <>
              先记 attention 子层的账。四个投影矩阵 <MathTex tex="W_Q, W_K, W_V, W_O" /> 在标准多头注意力
              （Multi-Head Attention，MHA）里都是 <MathTex tex="d \times d" /> 的方阵（分头只是把输出向量重新解释成
              32 个 head × 128 维，不改变矩阵大小），所以：
            </>,
          )}
        </p>
        <MathTex block tex="P_{\text{attn}} = 4d^2" />
        <p>
          {t(
            <>
              Now the MLP. SwiGLU has three matrices: gate and up are both <MathTex tex="d \times d_{ff}" />, down is{' '}
              <MathTex tex="d_{ff} \times d" />:
            </>,
            <>
              再记 MLP 的账。SwiGLU 有三个矩阵：gate 和 up 都是 <MathTex tex="d \times d_{ff}" />，down 是{' '}
              <MathTex tex="d_{ff} \times d" />：
            </>,
          )}
        </p>
        <MathTex block tex="P_{\text{mlp}} = 3 \, d \, d_{ff}" />
        <p>
          {t(
            <>
              Each layer also has two RMSNorms (d scale parameters each); outside the layers there&apos;s the embedding
              matrix <MathTex tex="V \times d" /> and the output-side lm_head (LLaMA does not tie these two, so each
              counts once), plus a final norm. Summing up:
            </>,
            <>
              每层还有两个 RMSNorm（各 d 个缩放参数），层外有词嵌入（embedding）矩阵 <MathTex tex="V \times d" /> 和
              输出端的 lm_head（LLaMA 不共享这两个矩阵，所以各算一份），加上最后一个 norm。合计：
            </>,
          )}
        </p>
        <MathTex block tex="P \;=\; L\left(4d^2 + 3\,d\,d_{ff} + 2d\right) \;+\; 2Vd \;+\; d" />
        <p>{t('Plug in LLaMA-7B (d=4096, d_ff=11008, L=32, V=32000):', '代入 LLaMA-7B 的数（d=4096，d_ff=11008，L=32，V=32000）：')}</p>
        <MathTex block tex="32 \times (\underbrace{4 \cdot 4096^2}_{67.1\text{M}} + \underbrace{3 \cdot 4096 \cdot 11008}_{135.3\text{M}}) + \underbrace{2 \cdot 32000 \cdot 4096}_{262.1\text{M}} + 266{,}240" />
        <MathTex block tex="= 6{,}738{,}415{,}616 \;\approx\; 6.74\text{B} \;\;\checkmark" />
        <p>
          {t(
            <>
              Not one parameter off the official checkpoint count. Look at how the ledger splits: 135 million per MLP
              layer against 67 million per attention layer, so <strong>the MLP walks away with roughly two-thirds of the
              parameters</strong>. &quot;Attention is the body of the Transformer&quot; is a popular misconception. By
              parameter count the body is the MLP. Attention is the small router that ferries information between
              tokens.
            </>,
            <>
              和官方 checkpoint 数出来的一个不差。看账本怎么分配：MLP 一层 1.35 亿，attention 一层 0.67 亿，
              <strong>MLP 拿走了大约三分之二的参数</strong>。「Transformer 的本体是 attention」是个流行的误解。
              按参数算，本体是 MLP。attention 只是那个在 token 之间搬运信息的小路由器。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The same ledger explains why{' '}
              <Term t="GQA">
                Grouped-Query Attention: multiple Q heads share one set of K/V heads. With n_kv=8, n_head=64, the K/V
                projections shrink to 1/8, and the KV cache shrinks by the same factor (detailed in Chapter 9).
              </Term>{' '}
              saves parameters, and why it has become the default attention variant at scale. The output dimension of
              the K and V projections drops from d to <MathTex tex="n_{kv} \cdot d_{head}" />, so:
            </>,
            <>
              同一本账还能解释{' '}
              <Term t="GQA">
                分组查询注意力（Grouped-Query Attention）：多个 Q head 共享同一组 K/V head。n_kv=8、n_head=64
                时，K/V 投影缩小到原来的 1/8，KV cache 也同比例缩小（第 9 章细讲）。
              </Term>{' '}
              为什么省参数，以及它为什么成了大模型上默认的注意力变体。K、V 投影的输出维度从 d 降到{' '}
              <MathTex tex="n_{kv} \cdot d_{head}" />，于是：
            </>,
          )}
        </p>
        <MathTex block tex="P_{\text{attn}}^{\text{GQA}} = 2d^2\left(1 + \tfrac{n_{kv}}{n_{\text{head}}}\right)" />
        <p>
          {t(
            <>
              LLaMA-2 70B uses n_kv=8, n_head=64, dropping attention params from 4d² to 2.25d². Run the same formula for
              the 70B total: 80 layers × (2.25·8192² + 3·8192·28672) + 2·32000·8192 ≈ 69.0B, and it checks out again.
              One four-line formula, two generations of models.
            </>,
            <>
              LLaMA-2 70B 用 n_kv=8、n_head=64，attention 参数从 4d² 降到 2.25d²。用同一条公式算 70B 的总账：
              80 层 × (2.25·8192² + 3·8192·28672) + 2·32000·8192 ≈ 69.0B，又对上了。一套四行公式，吃下两代模型。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={3}
        title={t('FLOPs: two operations per parameter', 'FLOPs：每个参数两次运算')}
        lead={t(
          'Once you have the parameter count, the FLOPs are basically free — a hard conversion rate ties them together.',
          '参数量数清楚了，计算量就是免费的——它们之间有一条铁打的换算率。',
        )}
      >
        <p>
          {t(
            <>
              In a matrix multiply, every weight does exactly one multiply-accumulate (MAC) per forward pass: one
              multiply, one add, 2 FLOPs. As a token flows through the model, each linear-layer weight gets used once.
              So during decode (autoregressive token-by-token generation), the forward compute for a single token comes
              out to roughly:
            </>,
            <>
              矩阵乘法里，每个权重在每次前向中只做一次乘加（multiply-accumulate，MAC）：乘一下，加一下，2 FLOPs。
              一个 token 流过整个模型，每个线性层的权重都被用到一次。所以 decode（逐 token 自回归生成）时，单 token
              的前向计算量大致是：
            </>,
          )}
        </p>
        <MathTex block tex="F_{\text{decode}} \;\approx\; 2P \;+\; \underbrace{4\,L\,S\,d}_{\text{score}\cdot\text{AV}}" />
        <p>
          {t(
            <>
              The first term is the sum of all GEMMs: twice the parameter count, independent of sequence length. The
              second term is the two parameter-free matrix operations inside attention. Q dotted against the S cached K
              vectors (<MathTex tex="2Sd" />), then the softmax weights summed over the S V vectors (another{' '}
              <MathTex tex="2Sd" />), giving <MathTex tex="4Sd" /> per layer, times L layers. It burns no weights yet
              grows linearly with context length S. Keep an eye on it, because it is the one term in this whole chapter
              that grows. For prefill (processing the whole prompt in parallel at once), sum the tally over all S tokens;
              under a causal mask each token sees S/2 of context on average:
            </>,
            <>
              第一项是所有 GEMM 的总和：2 倍参数量，与序列长度无关。第二项是 attention 里两次「非参数」的矩阵运算。
              Q 对 S 个缓存 K 做点积（<MathTex tex="2Sd" />），softmax 权重再对 S 个 V 加权求和（又一个{' '}
              <MathTex tex="2Sd" />），每层 <MathTex tex="4Sd" />，乘 L 层。它不消耗任何权重，却随上下文长度 S 线性
              增长。盯紧这一项，它是整章唯一会长大的项。到了 prefill（一次性并行处理整段 prompt），把 S 个 token 的
              账加起来，causal mask 下平均每个 token 看 S/2 的上下文：
            </>,
          )}
        </p>
        <MathTex block tex="F_{\text{prefill}} \;\approx\; 2PS \;+\; 2\,L\,d\,S^2" />
        <Callout type="insight" title={t('One parameter = two operations', '一个参数 = 两次运算')}>
          <p>
            {t(
              <>
                &quot;2 × parameter count&quot; is the number you do all your LLM inference economics on. A 7B model
                decodes one token in ≈ 14 GFLOPs, a 70B in ≈ 140 GFLOPs: 10× the parameters, 10× the per-token compute,
                no tables required. Run it backwards and it gets brutal. An A100&apos;s BF16 peak is 312 TFLOPS, so the
                pure-compute ceiling is 312e12 ÷ 14e9 ≈ 22,000 tokens per second, yet real single-stream decode manages
                only a couple hundred tokens/s. That two-orders-of-magnitude gap is a bandwidth problem, not a compute
                problem, and it is exactly the question <ChapterLink n={6} />&apos;s roofline and{' '}
                <ChapterLink n={10} />&apos;s batching set out to answer.
              </>,
              <>
                「2 × 参数量」是你做所有 LLM 推理经济学时心算用的那个数。7B 模型 decode 一个 token ≈ 14 GFLOPs，
                70B ≈ 140 GFLOPs：参数翻 10 倍，每 token 计算量也翻 10 倍，不用查任何表。倒过来算就更狠了。一张
                A100 的 BF16 峰值是 312 TFLOPS，纯算力上限是每秒 312e12 ÷ 14e9 ≈ 22000 个 token，可实际单流 decode
                只有一百多 token/s。差出两个数量级，是带宽问题，不是算力问题，而这正是<ChapterLink n={6} /> roofline
                和<ChapterLink n={10} />批处理要回答的问题。
              </>,
            )}
          </p>
        </Callout>
        <p>
          {t(
            <>
              The explorer below turns these two formulas into a dashboard. The five sliders map to hyperparameters you
              can find in any model config, and the readouts recompute in real time. Watch the stacked bar in
              particular; it shows where a single token&apos;s FLOPs actually go.
            </>,
            <>
              下面的巡览器把这两条公式做成了仪表盘。五根滑杆对应你在任何模型 config 里都能找到的超参，右侧读数实时
              重算。重点看那根堆叠条，它告诉你单 token 的 FLOPs 到底花在了哪里。
            </>,
          )}
        </p>

        <FlopsExplorer />

        <p>
          {t(
            <>
              Three experiments worth running by hand. Keep the LLaMA-7B preset and pull S from 128 to 32K:{' '}
              <span className="text-amber">score·AV</span> climbs from under 1% to nearly 60% and overtakes the MLP, and
              this is only the decode ledger; in prefill it is a quadratic term and grows even more violently. Switch to
              the GPT-3 preset and watch tripling d cost a d², a 9× hit per layer. Toggle MHA / GQA and the parameter
              count moves while the score·AV term sits dead still: GQA buys back parameters and KV-cache memory, never
              attention compute, since not a single Q head was removed.
            </>,
            <>
              三个值得亲手做的实验。保持 LLaMA-7B 预设，把 S 从 128 拉到 32K：<span className="text-amber">score·AV</span>{' '}
              的占比从不到 1% 涨到近六成，反超 MLP，而这还只是 decode 的账，prefill 里它是平方项，涨得更凶。切到 GPT-3
              预设，看 d 翻 3 倍带来的是 d² 即 9 倍的单层开销。切换 MHA / GQA，参数量动了，score·AV 那一项却纹丝不动：
              GQA 省回来的是参数和 KV cache 显存，从来不是 attention 的计算，因为 Q head 一个都没少。
            </>,
          )}
        </p>

        <Quiz
          question={t(
            'Decoding one token with LLaMA-7B at 2K context, what is the approximate forward compute?',
            'LLaMA-7B 在 2K 上下文下 decode 一个 token，前向计算量大约是多少？',
          )}
          options={[
            {
              text: t('About 1.4 GFLOPs', '约 1.4 GFLOPs'),
              explain: t(
                'Off by an order of magnitude. Each parameter does one multiply-accumulate — 2 FLOPs per parameter — so 7B parameters is about 14 GFLOPs.',
                '少了一个数量级。每个参数要做一次乘加，是 2 FLOPs / 参数，7B 参数就是约 14 GFLOPs。',
              ),
            },
            {
              text: t('About 14 GFLOPs', '约 14 GFLOPs'),
              correct: true,
              explain: t(
                '2 × 6.74B ≈ 13.5 GFLOPs, plus attention\'s 4·L·S·d (≈ 1.1 GFLOPs at S=2048), totaling about 14.5 GFLOPs. The "2× parameter count" rule is worth burning into your brain.',
                '2 × 6.74B ≈ 13.5 GFLOPs，加上 attention 的 4·L·S·d（S=2048 时约 1.1 GFLOPs），合计约 14.5 GFLOPs。「2 倍参数量」这条心算规则值得焊死在脑子里。',
              ),
            },
            {
              text: t('About 140 GFLOPs', '约 140 GFLOPs'),
              explain: t(
                "That's the 70B scale. FLOPs are proportional to parameter count: ≈ 2P.",
                '这是 70B 模型的量级。FLOPs 与参数量成正比：≈2P。',
              ),
            },
            {
              text: t('About 14 TFLOPs', '约 14 TFLOPs'),
              explain: t(
                "Off by three orders of magnitude — 14 TFLOPs is already the total bill for prefilling a thousand tokens. Single-token decode is 2P ≈ 14 GFLOPs.",
                '差了三个数量级——14 TFLOPs 已经是 prefill 一千个 token 的总账了。decode 单 token 是 2P ≈ 14 GFLOPs。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={4}
        title={t("Attention's quirk: the S² invoice that doesn't stick around", 'Attention 的特殊性：S² 的临时账单')}
        lead={t(
          'On the FLOPs ledger attention is a supporting actor, but it carries a hidden cost no other operation has.',
          'FLOPs 账本上 attention 是配角，但它有一笔别的运算都没有的隐藏开销。',
        )}
      >
        <p>
          {t(
            <>
              So far attention looks harmless: a third of the parameters, and at short sequences a rounding error of
              FLOPs. It differs from the MLP in one essential way. The MLP computes each token independently and its
              intermediate tensors don&apos;t depend on S, while attention forces every token to meet every preceding
              token pairwise. A naive implementation materializes that meeting as a real tensor:{' '}
              <MathTex tex="A = QK^\top" />, shape <MathTex tex="[B,\, h,\, S,\, S]" />. It is not a parameter, never
              enters the checkpoint, and is discarded the instant softmax and AV finish. For that instant, though, it
              genuinely sits in memory, with a footprint of:
            </>,
            <>
              到目前为止 attention 看起来人畜无害：占三分之一参数，FLOPs 在短序列下也只是零头。它和 MLP 有一个本质
              区别。MLP 对每个 token 独立计算，中间张量的大小与 S 无关，而 attention 要让每个 token 和前面所有 token
              两两见面。naive（朴素）实现会把这场见面物化成一个真实的张量：<MathTex tex="A = QK^\top" />，形状{' '}
              <MathTex tex="[B,\, h,\, S,\, S]" />。它不是参数，不进 checkpoint，softmax 和 AV 一做完就被丢掉。但就在
              那一瞬间，它实打实地躺在显存里，体积是：
            </>,
          )}
        </p>
        <MathTex block tex="M_{\text{score}} = B \cdot h \cdot S^2 \cdot 2\ \text{bytes (fp16)}" />
        <p>
          {t(
            <>
              Mind the exponents. Parameters don&apos;t depend on S; KV cache and activations grow linearly with S; this
              one temporary matrix grows as <strong>S²</strong>. What makes a quadratic dangerous is how dull its first
              half looks. At S=2048 a single layer&apos;s score is just 256MB, and nobody gives it a second glance.
              Stretch the sequence 16× to 32K and it balloons 256×. The experiment below lets you watch it cross two red
              lines.
            </>,
            <>
              注意指数。参数与 S 无关，KV cache 和激活随 S 一次方增长，唯独这个临时矩阵随 <strong>S²</strong> 增长。
              平方曲线的危险，在于它前半段长得太不起眼。S=2048 时单层 score 只有 256MB，谁都不会多看一眼。把序列拉长
              16 倍到 32K，它就膨胀 256 倍。下面的实验让你亲眼看它越过两条红线。
            </>,
          )}
        </p>

        <AttnMemoryLab />

        <p>
          {t(
            <>
              At S=32K, batch=1, a single layer&apos;s temporary score matrix is about 64GB (32 heads × 32768² × 2
              bytes), nearly five times the entire 7B weight set (13.5GB), pressing right up against the 80GB HBM ceiling
              of an A100/H100. Bump the batch to 2 and you OOM (Out of Memory) outright. (Blackwell-class cards push the
              ceiling higher, but a quadratic chews through any fixed budget eventually.) Here is the absurd part: the
              model itself isn&apos;t big, and the FLOPs are the same order as the linear layers, yet{' '}
              <strong>a single throwaway intermediate result is what blows up memory</strong>. It gets worse on the
              bandwidth side. This matrix is written to HBM once, read by softmax once, and read again by AV, three
              S²-scale round trips to memory. The fix isn&apos;t a bigger card. It is to never materialize the matrix:
              tile attention into blocks, stream the computation on-chip in SRAM, and keep score alive only in registers
              and shared memory. That is FlashAttention, the whole of the next chapter.
            </>,
            <>
              S=32K、batch=1 时，单层临时 score 矩阵约 64GB（32 个 head × 32768² × 2 字节），是 7B 模型全部权重
              （13.5GB）的近五倍，直逼 A100/H100 的 80GB HBM 上限。batch 开到 2 就直接 OOM（Out of Memory，显存
              溢出）。（Blackwell 一代的卡把上限抬得更高，但平方项迟早会啃光任何固定预算。）荒谬就在这里：模型本身
              不大，FLOPs 也只和线性层同量级，<strong>偏偏是一个用完即弃的中间结果撑爆了显存</strong>。带宽这边更糟。
              这个矩阵要往 HBM 写一遍、softmax 读一遍、AV 再读一遍，三趟 S² 级别的显存往返。出路不是买更大的卡，而是
              根本不物化这个矩阵：把 attention 拆成小块，在片上 SRAM 里流式地算，让 score 永远只活在寄存器和 shared
              memory 里。这就是 FlashAttention，下一章的全部内容。
            </>,
          )}
        </p>

        <Quiz
          question={t(
            'Why does the MLP dominate compute at short sequences while attention overtakes it at very long ones?',
            '为什么短序列下 MLP 占计算大头，而超长序列下 attention 反超？',
          )}
          options={[
            {
              text: t(
                'The score·AV term (4LSd) grows with S, while every linear layer\'s per-token FLOPs are constant (≈2P)',
                'score·AV 项（4LSd）随 S 增长，而所有线性层每 token 的 FLOPs 是常数（≈2P）',
              ),
              correct: true,
              explain: t(
                "Exactly. A linear layer's per-token compute is set by parameter count and is context-independent; attention's QKᵀ and AV must sweep all S cached positions, so they grow linearly with S (prefill total grows as S²). Once S is large enough that 4·S·d is the same order as 6·d·d_ff (for LLaMA-7B that's around S≈16K, where it ties the MLP), attention inevitably pulls ahead.",
                '正是如此。线性层每 token 的计算量由参数量决定，与上下文无关；attention 的 QKᵀ 和 AV 要遍历全部 S 个缓存位置，随 S 线性涨（prefill 总量随 S² 涨）。S 一旦大到 4·S·d 与 6·d·d_ff 同量级（LLaMA-7B 约在 S≈16K 处与 MLP 打平），attention 必然反超。',
              ),
            },
            {
              text: t("MLP parameters get shared at long sequences, so its compute drops", '长序列下 MLP 的参数会被共享，计算量变少'),
              explain: t(
                "MLP parameters never change with S; every token passes through the full set of MLP weights, all 2·3·d·d_ff FLOPs of it, with nothing skipped.",
                'MLP 参数从不随 S 变化，每个 token 都完整地过一遍全部 MLP 权重，2·3·d·d_ff FLOPs 一分不少。',
              ),
            },
            {
              text: t('Softmax is an exponential operation, far more expensive than multiplication', 'softmax 是指数运算，比乘法昂贵得多'),
              explain: t(
                "Softmax's FLOPs are only O(h·S), two orders of magnitude below the O(S·d) dot products next to it; it's never the compute bottleneck (it is a bandwidth bottleneck, but that's a different book).",
                'softmax 的 FLOPs 只有 O(h·S)，比旁边 O(S·d) 的点积低两个数量级，从来不是计算瓶颈（它是带宽瓶颈倒是真的，但那是另一本账）。',
              ),
            },
            {
              text: t('At long sequences the KV cache hit rate drops and recomputation is needed', '长序列下 KV cache 命中率下降，需要重算'),
              explain: t(
                <>
                  The KV cache has no &quot;hit rate&quot; — it&apos;s an exact cache and never recomputes. At long
                  sequences it brings memory and bandwidth pressure (<ChapterLink n={9} />), but it doesn&apos;t change
                  FLOPs.
                </>,
                <>
                  KV cache 不存在「命中率」——它是精确缓存，从不重算。长序列下它带来的是显存和带宽压力（<ChapterLink n={9} />），不改变 FLOPs。
                </>,
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={5}
        title={t('Training vs. inference: 6PT and two kinds of bottleneck', '训练 vs 推理：6PT 与两种瓶颈')}
        lead={t(
          'The same formula, multiplied by different coefficients, spans both the training and inference worlds.',
          '同一套公式，乘上不同的系数，就能横跨训练和推理两个世界。',
        )}
      >
        <p>
          {t(
            <>
              The training ledger differs from inference by a single coefficient. The forward pass is 2P FLOPs/token.
              Backprop computes two sets of gradients, one with respect to activations (so the chain rule keeps
              propagating backward) and one with respect to weights, each a GEMM the same size as the forward, so
              backward ≈ 4P. Total:
            </>,
            <>
              训练的账只比推理多乘一个系数。前向是 2P FLOPs/token。反向传播要算两组梯度，一组对激活（让链式法则
              继续往前传），一组对权重，每组都是一次与前向同规模的 GEMM，所以反向 ≈ 4P。合计：
            </>,
          )}
        </p>
        <MathTex block tex="F_{\text{train}} \;\approx\; 6 \, P \, T" />
        <p>
          {t(
            <>
              where T is the total number of training tokens. This rule of thumb traces back to Kaplan et al.&apos;s
              scaling-laws paper, and it underpins how everyone budgets large-model training runs. Verify it: LLaMA-7B
              trained on 1T tokens, 6 × 6.74e9 × 1e12 ≈ 4×10²² FLOPs; at an A100 BF16 peak of 312 TFLOPS and a real{' '}
              <Term t="MFU">
                Model FLOPs Utilization: the FLOPs implied by the model&apos;s actual throughput divided by the
                hardware&apos;s peak compute. Hitting 40%–55% at large scale is considered excellent; the rest of the
                time goes to communication, data loading, and memory-bound operators.
              </Term>{' '}
              of about 45%, you need roughly 80,000 GPU-hours. The real figure Meta reports in the paper is 82,432
              GPU-hours. Four lines of arithmetic, error in the single-digit percent.
            </>,
            <>
              其中 T 是训练 token 总数。这条经验式可以追到 Kaplan 等人的 scaling laws 论文，大家给大模型训练算预算
              基本都靠它。验证一下：LLaMA-7B 训了 1T token，6 × 6.74e9 × 1e12 ≈ 4×10²² FLOPs；按 A100 BF16 峰值 312
              TFLOPS、实际{' '}
              <Term t="MFU">
                Model FLOPs Utilization：模型实际吞吐折算出的 FLOPs 除以硬件峰值算力。大规模训练能跑到 40%~55%
                就算优秀，剩下的时间花在通信、数据加载和 memory-bound 算子上。
              </Term>{' '}
              约 45% 计，需要约 8 万 GPU·小时。Meta 论文里报的真实数字是 82,432 GPU·小时。四行公式，误差只在个位数百分比。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              On the inference side the same FLOPs formula splits into two phases with wildly different personalities.{' '}
              <strong>Prefill</strong> streams all S tokens of the prompt through the model at once. Each block of
              weights, once brought on-chip, is reused S times, so arithmetic intensity is high and the phase is
              compute-bound: pinned to the compute roof on the roofline, Tensor Cores firing on all cylinders, the same
              workload as the training forward pass. <strong>Decode</strong> generates one token at a time, so it hauls
              the full 13.5GB of weights from HBM yet does only 2P ≈ 14 GFLOPs. That works out to 2 operations per 2
              bytes moved, an arithmetic intensity of 1–2 FLOPs/Byte, against an A100 ridge point above 160. Two orders
              of magnitude apart, so decode compute utilization is often under 1% and almost all the time goes to
              waiting on memory. The latency floor of single-stream decode then drops straight out of bandwidth:
              13.5GB ÷ 1.9TB/s ≈ 7ms/token, about 140 tokens/s, startlingly close to what you measure.{' '}
              <ChapterLink n={10} /> unfolds this time ledger millisecond by millisecond.
            </>,
            <>
              推理这边，同一条 FLOPs 公式劈出了两个性格迥异的阶段。<strong>Prefill</strong> 把 prompt 里的 S 个 token
              一次性并行喂进模型。每块权重搬进片上后能复用 S 次，算术强度（arithmetic intensity）很高，这一段是
              compute-bound（算力受限）的：贴着 roofline 的算力屋顶，Tensor Core 火力全开，和训练前向是同一种负载。
              <strong>Decode</strong> 一次只生成一个 token，要把 13.5GB 权重从 HBM 完整搬一遍，却只做 2P ≈ 14 GFLOPs。
              算下来每搬 2 个字节才做 2 次运算，算术强度只有 1~2 FLOPs/Byte，而 A100 的脊点（ridge point）在 160 以上。
              差出两个数量级，于是 decode 的算力利用率常常不到 1%，时间几乎全花在等显存上。这样一来，单流 decode 的
              延迟下限直接从带宽里掉出来：13.5GB ÷ 1.9TB/s ≈ 7ms/token，约 140 token/s，和实测惊人地接近。这本时间账，
              <ChapterLink n={10} />会逐毫秒展开。
            </>,
          )}
        </p>
        <Callout type="deep" title={t('Why decode is "free FLOPs, expensive bytes"', '为什么说 decode 是「免费的 FLOPs、昂贵的字节」')}>
          <p>
            {t(
              <>
                Raise the batch from 1 to 32 and the weights still get hauled only once, while the FLOPs become 32×.
                Arithmetic intensity rises linearly, the GPU comes off the bandwidth wall toward the compute roof, and
                throughput is a nearly free 32× win. That is why batching is close to free in inference serving, and it
                is the starting point for system designs like continuous batching and PagedAttention (<ChapterLink n={10} />).
              </>,
              <>
                把 batch 从 1 加到 32，权重还是只搬一遍，FLOPs 却变成 32 份。算术强度线性上升，GPU 从带宽墙下被推向
                算力屋顶，吞吐近乎白赚 32 倍。这就是推理服务里 batching（批处理）近乎免费的原因，也是 continuous
                batching、PagedAttention 这些系统设计的出发点（<ChapterLink n={10} />）。
              </>,
            )}
          </p>
        </Callout>

        <Quiz
          question={t(
            'Training a 7B model over 1T tokens, what is the order of magnitude of total compute?',
            '训练一个 7B 模型过 1T token，总计算量的量级是？',
          )}
          options={[
            {
              text: t('About 4×10²⁰ FLOPs', '约 4×10²⁰ FLOPs'),
              explain: t(
                "100× too small — that's roughly the training compute for just 10B tokens. Training is 6PT: 2PT forward + 4PT backward.",
                '少了 100 倍——这大约是只过 10B token 的训练量。训练是 6PT：前向 2PT + 反向 4PT。',
              ),
            },
            {
              text: t('About 4×10²² FLOPs', '约 4×10²² FLOPs'),
              correct: true,
              explain: t(
                '6 × 6.74×10⁹ × 10¹² ≈ 4×10²² FLOPs. Divide by single-card effective compute (A100 312 TFLOPS × 45% MFU ≈ 1.4×10¹⁴) ≈ 2.9×10⁸ GPU-seconds ≈ 80,000 GPU-hours, matching Meta\'s reported 82,432 GPU-hours.',
                '6 × 6.74×10⁹ × 10¹² ≈ 4×10²² FLOPs。除以单卡有效算力（A100 312 TFLOPS × 45% MFU ≈ 1.4×10¹⁴）≈ 2.9×10⁸ GPU·秒 ≈ 8 万 GPU·小时，与 Meta 报告的 82,432 GPU·小时吻合。',
              ),
            },
            {
              text: t('About 4×10²⁴ FLOPs', '约 4×10²⁴ FLOPs'),
              explain: t(
                "100× too large, getting into the range estimated for frontier-scale training runs. 7B × 1T tokens by 6PT is 4×10²².",
                '多了 100 倍，已经进到外界估计的前沿规模训练量级了。7B × 1T token 按 6PT 算是 4×10²²。',
              ),
            },
            {
              text: t('Impossible to estimate; depends on the implementation', '无法估算，取决于具体实现'),
              explain: t(
                "Quite the opposite — the 6PT rule is implementation-independent, looking only at parameter count and data volume, usually within 10% error. That's exactly what makes it valuable.",
                '恰恰相反——6PT 这条经验式与实现无关，只看参数量和数据量，误差通常在 10% 以内。这正是它宝贵的地方。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={6}
        title={t('Summary and further reading', '总结与延伸阅读')}
        lead={t('Everything in this chapter, distilled into five rules you can do in your head.', '这一章的全部内容，浓缩成五条可以心算的规则。')}
      >
        <ul>
          <li>
            {t(
              <>
                <strong>One layer = five GEMMs.</strong> The FLOPs of norm, RoPE, and softmax drop off the ledger; their
                time doesn&apos;t, because they&apos;re memory-bound.
              </>,
              <>
                <strong>一层 = 五个 GEMM。</strong>norm、RoPE、softmax 的 FLOPs 可以从账上划掉；它们的耗时不能，因为
                它们是 memory-bound 的。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Parameters: </strong>
                <MathTex tex="P \approx L(4d^2 + 3\,d\,d_{ff}) + 2Vd" />, which for LLaMA-7B gives 6,738,415,616, not
                one off. The MLP takes about two-thirds; GQA cuts attention&apos;s 4d² down to{' '}
                <MathTex tex="2d^2(1 + n_{kv}/n_h)" />.
              </>,
              <>
                <strong>参数：</strong>
                <MathTex tex="P \approx L(4d^2 + 3\,d\,d_{ff}) + 2Vd" />，代入 LLaMA-7B 得 6,738,415,616，一个不差。
                MLP 占约三分之二；GQA 把 attention 的 4d² 砍成 <MathTex tex="2d^2(1 + n_{kv}/n_h)" />。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Compute: </strong>decode per token ≈ 2P FLOPs (7B → 14 GFLOPs), plus the only S-growing attention
                term 4LSd; prefill ≈ 2PS + 2LdS².
              </>,
              <>
                <strong>计算：</strong>decode 每 token ≈ 2P FLOPs（7B → 14 GFLOPs），外加唯一随 S 增长的 attention
                项 4LSd；prefill ≈ 2PS + 2LdS²。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The hidden memory bomb: </strong>naive attention&apos;s S×S score matrix balloons as S², hitting
                64GB per layer at 32K context, nearly five times the weights. FlashAttention was born for this (<ChapterLink n={8} />).
              </>,
              <>
                <strong>显存的暗雷：</strong>naive attention 的 S×S score 矩阵随 S² 膨胀，32K 上下文时单层 64GB，
                是权重的近五倍。FlashAttention 因此而生（<ChapterLink n={8} />）。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Training ≈ 6PT;</strong> inference splits in two: prefill compute-bound, decode memory-bound
                (arithmetic intensity only 1–2 FLOPs/Byte), and decode&apos;s latency floor = weight bytes ÷ HBM
                bandwidth.
              </>,
              <>
                <strong>训练 ≈ 6PT；</strong>推理分两段：prefill compute-bound、decode memory-bound（算术强度仅
                1~2 FLOPs/Byte），decode 延迟下限 = 权重字节数 ÷ HBM 带宽。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Further reading, ordered by "stronger the moment you finish":', '延伸阅读，按「读完立刻变强」的顺序排：')}</p>
        <ul>
          <li>
            <a href="https://kipp.ly/transformer-inference-arithmetic/" target="_blank" rel="noreferrer">
              kipply — Transformer Inference Arithmetic
            </a>
            {t(
              ': the spiritual source of this chapter\'s inference half, taking the latency, bandwidth, and batch ledger to the limit.',
              '：本章推理部分的精神源头，把延迟、带宽、batch 的账算到了极致。',
            )}
          </li>
          <li>
            <a href="https://blog.eleuther.ai/transformer-math/" target="_blank" rel="noreferrer">
              EleutherAI — Transformer Math 101
            </a>
            {t(
              ': the training-side counterpart, covering 6PT, training memory footprint, and parallelism overhead.',
              '：训练侧的对应篇，6PT、训练显存占用、并行开销一网打尽。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">
              Vaswani et al. — Attention Is All You Need (2017)
            </a>
            {t(
              ': the original paper. Reread Section 3.2 with this chapter\'s ledger in hand — a completely different experience.',
              '：原始论文。带着本章的账本回去重读 3.2 节，体验完全不同。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noreferrer">
              Touvron et al. — LLaMA: Open and Efficient Foundation Language Models (2023)
            </a>
            {t(
              ': the source of every number in this chapter; the appendix has the full hyperparameter table and the GPU-hour bill.',
              '：本章所有数字的出处，附录里有完整的超参表和 GPU·小时账单。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noreferrer">
              Kaplan et al. — Scaling Laws for Neural Language Models (2020)
            </a>
            {t(
              ': the origin of the 6PT rule, and the starting point of the "scale is all you need" creed.',
              '：6PT 经验式的出处，也是「规模决定一切」这一信仰的起点。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
