import { Callout, CodeBlock, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import CacheCompareLab from './CacheCompareLab'
import KVCalcLab from './KVCalcLab'
import PagedLab from './PagedLab'

const DECODE_LOOP_EN = `# No cache: to emit each new token, the whole sequence is run through the model again
for t in range(max_new_tokens):
    logits = model(tokens)              # O(S²·d) every step, and S keeps growing
    tokens.append(sample(logits[-1]))

# With cache: the prompt is computed once, then each step feeds in only the newest token
kv_cache = model.prefill(prompt)        # compute every layer's K/V for the whole prompt, once
x = prompt[-1]
for t in range(max_new_tokens):
    logits, kv_cache = model.decode_step(x, kv_cache)   # O(S·d) per step
    x = sample(logits)`

const DECODE_LOOP_ZH = `# 无 cache：每生成一个 token，整段序列重新过一遍模型
for t in range(max_new_tokens):
    logits = model(tokens)              # 每步 O(S²·d)，S 还在不断变长
    tokens.append(sample(logits[-1]))

# 有 cache：prompt 只算一次，之后每步只喂入最新的那个 token
kv_cache = model.prefill(prompt)        # 一次性算出 prompt 所有层的 K/V
x = prompt[-1]
for t in range(max_new_tokens):
    logits, kv_cache = model.decode_step(x, kv_cache)   # 每步 O(S·d)
    x = sample(logits)`

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            Picture a chatbot with no cache at all: by the time it has produced its 1000th token, emitting the 1001st
            means re-running all 1000 prior tokens <em>through the model from scratch</em> — and it does this again for
            every single character it speaks. That is not hyperbole; it is the most naive form of autoregressive
            generation. The KV cache rescues us from this absurdity: history is computed once, and from then on you just
            look it up. But there is no free lunch — the compute you save turns into a memory-devouring monster. On an
            80&nbsp;GB A100, the model weights claim one slice and almost all of the rest belongs to the KV cache. How
            well you manage that territory decides whether one inference server can serve 5 users at once or 50. This
            chapter nails down two things: why the KV cache is even valid, and how vLLM's PagedAttention squeezes that
            memory to the limit.
          </>,
          <>
            想象一个不加任何缓存的聊天机器人：它回答到第 1000 个 token 时，为了吐出第 1001 个，
            要把前面 1000 个 token <em>从头到尾重新算一遍</em> —— 而且每吐一个字都要这样来一次。
            这不是夸张，这就是 Transformer 自回归生成最朴素的形态。KV Cache 把它从这种荒谬中救了下来：
            历史只算一次，之后查表即可。但天下没有免费的午餐 —— 省下的计算变成了吃显存的怪物，
            一张 80GB 的 A100，模型权重占掉一块，剩下的几乎全是 KV Cache 的地盘。
            这块地盘管得好不好，直接决定一台推理服务器能同时伺候 5 个用户还是 50 个。
            本章讲清楚两件事：KV Cache 为什么成立，以及 vLLM 的 PagedAttention 怎么把这块显存管到极致。
          </>,
        )}
      </p>

      <Section
        index={1}
        title={t('Autoregressive generation: only K and V are worth caching', '自回归生成：能缓存的，只有 K 和 V')}
        lead={t(
          'The legitimacy of caching rests on one quiet fact: under a causal mask, a history token’s K and V never change once computed.',
          '缓存的合法性来自一个安静的事实：在因果掩码下，历史 token 的 K、V 一旦算出，就永远不变。',
        )}
      >
        <p>
          {t(
            <>
              An LLM generates{' '}
              <Term t="autoregressively">
                one token at a time, appending each to the end of the input as the condition for the next prediction —
                the output is conditioned on all of its own prior outputs.
              </Term>{' '}
              : step <MathTex tex="t" /> takes all the preceding <MathTex tex="t-1" /> tokens and predicts the{' '}
              <MathTex tex="t" />
              th. Recall the attention mechanism from Chapter 7. At every layer, the new token does this: it takes its
              own query vector <MathTex tex="q_t" /> and dot-products it against the key vectors of{' '}
              <em>every history token</em>, then uses the resulting weights to take a weighted sum over all the history
              value vectors:
            </>,
            <>
              大语言模型的生成是<Term t="自回归（autoregressive）">
                每次只生成一个 token，并把它拼回输入末尾作为下一步预测的条件 —— 输出依赖自己之前的全部输出。
              </Term>的：第 <MathTex tex="t" /> 步拿到前面全部 <MathTex tex="t-1" /> 个 token，预测第{' '}
              <MathTex tex="t" /> 个。回忆第七章的注意力机制，新 token 在每一层要做的事是：
              用自己的查询向量 <MathTex tex="q_t" /> 去和<em>所有历史 token</em> 的键向量做点积，
              再用得到的权重对所有历史的值向量加权求和：
            </>,
          )}
        </p>
        <MathTex block tex="o_t = \mathrm{softmax}\!\left(\frac{q_t K_{1:t}^{\top}}{\sqrt{d}}\right) V_{1:t}" />
        <p>
          {t(
            <>
              Notice how differently the three roles are treated in this expression.{' '}
              <strong>Q only needs the current one</strong>: a history token's q vector was already used up on its own
              step and is never queried by anyone again, so storing it is pointless. <strong>K and V, by contrast, are
              the ones queried over and over</strong>: token 1's <MathTex tex="k_1, v_1" /> get reused by steps 2, 3, …,
              1000. More importantly, <MathTex tex="k_i = x_i W_K" /> and <MathTex tex="v_i = x_i W_V" /> depend only on
              position <MathTex tex="i" />'s own representation — the causal mask guarantees that later tokens cannot
              reach back and rewrite history's hidden states, so once these K and V are computed, they are valid for
              life. <strong>Depends only on itself, yet is read again and again — that is the textbook definition of
              "worth caching."</strong>
            </>,
            <>
              注意这个式子里三个角色的待遇完全不同。<strong>Q 只需要当前这一个</strong>：
              历史 token 的 q 向量在它们自己的那一步已经用完了，之后再也不会被任何人查询，存着没有意义。
              而 <strong>K 和 V 是被反复查询的一方</strong>：第 1 个 token 的{' '}
              <MathTex tex="k_1, v_1" /> 会被第 2、3、…、1000 步反复用到。更关键的是，
              <MathTex tex="k_i = x_i W_K" />、<MathTex tex="v_i = x_i W_V" /> 只依赖第{' '}
              <MathTex tex="i" /> 个位置自己的表示 —— 因果掩码（causal mask）保证后来的 token
              不会回头改写历史的隐状态，所以这些 K、V 算出来一次，终生有效。
              <strong>只依赖自身、又被反复读取 —— 这正是「值得缓存」的教科书定义。</strong>
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              So inference splits into two phases with utterly different personalities.{' '}
              <strong>Prefill</strong>: push the entire prompt through the model in parallel in one shot, computing the K
              and V at every layer and every position and storing them in the cache — one big matrix multiply that
              saturates the compute. <strong>Decode</strong>: from then on, each step feeds in just <em>one</em> new
              token, computes its q, k, v, appends the k, v to the cache, and queries the whole cache with q. The cost
              of attention per step drops from recomputing everything at <MathTex tex="O(S^2 \cdot d)" /> to{' '}
              <MathTex tex="O(S \cdot d)" /> — a full factor of S removed, and S routinely runs into the thousands or
              tens of thousands.
            </>,
            <>
              于是推理被切成了两个性格迥异的阶段。<strong>预填充（prefill）</strong>：
              把整段 prompt 一次性并行过模型，算出所有层、所有位置的 K、V 存进缓存 ——
              这是一次大矩阵乘，算力被吃满。<strong>解码（decode）</strong>：之后每步只把<em>一个</em>新
              token 喂进模型，算出它的 q、k、v，把 k、v 追加进缓存，再用 q 查询整个缓存。
              每步注意力的代价从重算全部的 <MathTex tex="O(S^2 \cdot d)" /> 降到{' '}
              <MathTex tex="O(S \cdot d)" /> —— 整整少了一个 S 的量级，而 S 动辄是几千上万。
            </>,
          )}
        </p>
        <CodeBlock code={t(DECODE_LOOP_EN, DECODE_LOOP_ZH)} lang="python" title="decode_loop.py" />
        <Callout type="insight" title={t('The KV cache is a "memory-for-compute" trade', 'KV Cache 是一笔「显存换计算」的交易')}>
          <p>
            {t(
              <>
                The cache shaves an order of magnitude off the compute per decode step, but every token's K and V, at
                every layer, must live resident in memory — and every step still has to read the whole cache back out of
                HBM. The result: decode goes from "can't compute it fast enough" to "can't read it fast enough." In the
                language of Chapter 6's roofline, it is a textbook memory-bound workload. This trade moves the
                battlefield from compute to memory — capacity decides how many people you can serve at once, bandwidth
                decides how fast each token comes out. This chapter owns capacity; the next one owns bandwidth.
              </>,
              <>
                缓存把 decode 每步的计算量砍掉一个数量级，但每个 token、每一层的 K 和 V
                都要常驻显存 —— 而且每步还得把整个缓存从 HBM 读一遍。结果是 decode
                从「算不过来」变成了「读不过来」：用第六章 roofline 的语言说，它是个典型的
                memory-bound 工作负载。这笔交易把战场从算力挪到了显存 ——
                容量决定你能同时服务多少人，带宽决定每个 token 出得多快。本章管容量，下一章管带宽。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title={t('Lab: watch the saved compute with your own eyes', '实验：亲眼看看省掉的计算')}
        lead={t(
          'The same 32-token generation: on the left, every step recomputes the entire attention matrix; on the right, each step appends just one row.',
          '同一段 32 个 token 的生成，左边每步重算整个注意力矩阵，右边只追加一行。',
        )}
      >
        <p>
          {t(
            <>
              The lab below plays both algorithms side by side, token by token. Each little cell is one entry of the
              attention-score matrix (one <MathTex tex="q \cdot k" /> dot product):{' '}
              <span className="text-volt">green</span> is the cell genuinely computed this step,{' '}
              <span className="text-rose">rose</span> is an old cell the no-cache side is needlessly recomputing every
              step, and <span className="text-amber">amber</span> is a cache hit — data sitting in memory, read but not
              recomputed. Notice that each step the right side adds only a thin new row, while the left side has to burn
              the entire triangle anew every step.
            </>,
            <>
              下面的实验把两种算法摆在一起逐 token 播放。每个小格子是注意力 score 矩阵里的一个单元
              （一次 <MathTex tex="q \cdot k" /> 点积）：<span className="text-volt">荧光绿</span>是本步真正新算的格子，
              <span className="text-rose">玫红</span>是无缓存一侧每步都在白白重算的旧格子，
              <span className="text-amber">琥珀色</span>是缓存命中 —— 数据躺在显存里，只读不算。
              注意右侧每步只新增薄薄一行，而左侧整个三角形每步都要重新烧一遍。
            </>,
          )}
        </p>
        <CacheCompareLab />
        <p>
          {t(
            <>
              The curve readings reward a second look: flip to "per step" and the no-cache cost grows quadratically with
              t (the whole t×t triangle), while with cache it is just a linear single row; flip to "cumulative" and the
              gap widens further into <MathTex tex="O(S^3)" /> versus <MathTex tex="O(S^2)" />. Push the target length
              from 32 to 48 and you'll see the savings multiplier keep climbing — the longer the sequence, the more the
              cache saves your life. This is exactly why no production inference engine dares run without a KV cache: it
              is not an optimization option, it is the ticket to entry.
            </>,
            <>
              曲线读数值得多看一眼：切到「每步」，无缓存的代价随 t 二次增长（整个 t×t 三角），
              有缓存只是线性的一行；切到「累计」，差距进一步拉开成 <MathTex tex="O(S^3)" /> 对{' '}
              <MathTex tex="O(S^2)" />。把目标长度从 32 拉到 48，你会看到节省倍数还在继续上涨 ——
              序列越长，缓存越是救命。这也是为什么没有任何一个生产推理引擎敢不开 KV Cache：
              它不是优化选项，是入场券。
            </>,
          )}
        </p>
        <Quiz
          question={t(
            'With the KV cache on, what is the attention compute complexity for each new token generated during decode (S = current sequence length, d = hidden dimension)?',
            '开启 KV Cache 之后，decode 阶段每生成一个新 token，注意力的计算复杂度是多少（S 为当前序列长度，d 为隐藏维度）？',
          )}
          options={[
            {
              text: <MathTex tex="O(S^2 \cdot d)" />,
              explain: t(
                'This is the no-cache cost of recomputing the entire attention matrix each step. The whole point of caching is to shave one factor of S off this S².',
                '这是无缓存时每步重算整个注意力矩阵的代价。缓存的意义恰恰是把这个 S² 砍掉一个 S。',
              ),
            },
            {
              text: <MathTex tex="O(S \cdot d)" />,
              correct: true,
              explain: t(
                'Exactly. The new token’s q dot-products against the S cached k vectors (S·d), then takes a weighted sum over the S v vectors (S·d) — O(S·d) overall. The price is that all S copies of K and V must stay resident in memory, and every step reads them back from HBM.',
                '正是。新 token 的 q 要和缓存里 S 个 k 做点积（S·d），再对 S 个 v 加权求和（S·d）—— 整体 O(S·d)。代价是这 S 份 K、V 都得常驻显存，且每步都要从 HBM 读一遍。',
              ),
            },
            {
              text: <MathTex tex="O(d^2)" />,
              explain: t(
                'O(d²) is the cost of pushing the new token through the linear projection layers (computing q/k/v), and that part is indeed independent of S — but attention itself still queries all S copies of historical K/V, which is O(S·d).',
                'O(d²) 是新 token 过线性投影层（算 q/k/v）的代价，那部分确实和 S 无关；但注意力本身还要查询全部 S 份历史 K/V，是 O(S·d)。',
              ),
            },
            {
              text: <MathTex tex="O(1)" />,
              explain: t(
                'No such luck — attention inherently has to look at all of history, so every step reads and multiplies at least S copies of K and V. There is no escaping O(S·d).',
                '可惜不行 —— 注意力天生要看全部历史，每步至少要把 S 份 K、V 读一遍、乘一遍，逃不开 O(S·d)。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title={t('The memory bill: every token pays rent', '显存账：每个 token 都在交房租')}
        lead={t(
          'One LLaMA-7B request at 4K context needs 2 GB of KV cache — two orders of magnitude larger than most people imagine.',
          '一条 4K 上下文的 LLaMA-7B 请求，KV Cache 要 2GB —— 比很多人想象的大两个数量级。',
        )}
      >
        <p>
          {t(
            <>
              How big is the KV cache, really? This bill is worth working out by hand, because every term maps to a real
              architectural decision. At fp16 precision, each token occupies, in the cache:
            </>,
            <>
              KV Cache 到底多大？这笔账值得手算一遍，因为每一项都对应一个真实的架构决策。
              fp16 精度下，每个 token 在缓存里占：
            </>,
          )}
        </p>
        <MathTex
          block
          tex={t(
            String.raw`\text{bytes/token} = \underbrace{2}_{K\text{ and }V} \times \underbrace{L}_{\text{layers}} \times \underbrace{n_{kv}}_{\text{KV heads}} \times \underbrace{d_{head}}_{\text{dim/head}} \times \underbrace{2}_{\text{fp16 bytes}}`,
            String.raw`\text{bytes/token} = \underbrace{2}_{K\,\text{和}\,V} \times \underbrace{L}_{\text{层数}} \times \underbrace{n_{kv}}_{\text{KV 头数}} \times \underbrace{d_{head}}_{\text{每头维度}} \times \underbrace{2}_{\text{fp16 字节}}`,
          )}
        />
        <p>
          {t(
            <>
              Multiply by sequence length S for one request's footprint, and by batch for the total on a whole card.
              Plug in LLaMA-7B's real parameters (L=32, 32 heads, 128 dims per head): 512 KB per token — a single token's
              cache is bigger than an entire high-res image. One 4K-context request is 2 GB, a seventh of the 7B model's
              13.5 GB of weights; stretch the context to 32K and one request is 16 GB, with weights now the minor party.
              This is the literal meaning of "memory becomes the new battlefield":{' '}
              <strong>in the long-context era, the main occupant of memory is not the weights, it is the KV cache.</strong>
            </>,
            <>
              再乘上序列长度 S 是单条请求的占用，乘上 batch 是整张卡上的总量。代入 LLaMA-7B
              的真实参数（L=32，32 个头，每头 128 维）：每个 token 512KB —— 一个 token 的缓存比一整张高清图片还大。
              4K 上下文的一条请求就是 2GB，是 7B 模型 13.5GB 权重的七分之一；上下文拉到 32K，
              一条请求 16GB，权重反倒成了小头。这就是「显存成为新战场」的字面意思：
              <strong>长上下文时代，显存的主要住户不是权重，是 KV Cache。</strong>
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The most intriguing term in the formula is <MathTex tex="n_{kv}" />. In standard multi-head attention
              (MHA), the KV head count equals the attention head count; whereas{' '}
              <strong>grouped-query attention (GQA)</strong> lets several Q heads share one group of K, V heads — LLaMA-2
              70B uses 64 Q heads to query just 8 KV heads, shrinking the cache by a factor of 8 with almost no quality
              loss. The more extreme <strong>MQA (multi-query attention)</strong> simply lets all Q heads share a single
              KV group. These are not showing off: they are architectural designs forced into being by the engineering
              bill — the pain of "the KV cache is too expensive" came first, and only then did GQA become standard issue
              in nearly every modern open-source model.
            </>,
            <>
              公式里最值得玩味的一项是 <MathTex tex="n_{kv}" />。标准的多头注意力（MHA, Multi-Head
              Attention）里 KV 头数等于注意力头数；而<strong>分组查询注意力（GQA, Grouped-Query
              Attention）</strong>让多个 Q 头共享同一组 K、V 头 —— LLaMA-2 70B 用 64 个 Q 头查询仅仅
              8 个 KV 头，缓存直接缩小 8 倍，模型质量几乎无损。更极端的 <strong>MQA（Multi-Query
              Attention）</strong>干脆让全部 Q 头共享 1 组 KV。这些不是炫技：它们是工程账倒逼出来的架构设计 ——
              先有「KV Cache 太贵」这个痛点，才有 GQA 进入几乎所有现代开源模型的标配清单。
            </>,
          )}
        </p>
        <KVCalcLab />
        <p>
          {t(
            <>
              Run three experiments with the calculator: (1) pick 7B, 4K context, and push batch from 8 to 64 — see when
              the KV bar hits the red line; (2) flip the GQA switch and the KV bar instantly shrinks to a quarter, the
              max concurrency quadruples — one line of architecture change worth half a card; (3) switch to 70B — the
              weights alone cross the 80 GB red line, which is why 70B-class models are off the table on a single card
              and have to wait for Chapter 12's tensor parallelism to be split apart.
            </>,
            <>
              用计算器做三个实验：① 选 7B、4K 上下文，把 batch 从 8 拉到 64 —— 看 KV
              什么时候撞上红线；② 打开 GQA 开关，KV 条瞬间缩成四分之一，最大并发翻四倍 ——
              一行架构改动顶得上半张卡；③ 切到 70B —— 权重一项就越过 80GB
              红线，这就是为什么 70B 级别的模型单卡免谈，必须等到第十二章的张量并行来拆。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={4}
        title={t('Fragmentation: more memory locked away than actually used', '碎片化：被锁死的显存比用掉的还多')}
        lead={t(
          'The vLLM paper measured it: in the mainstream inference systems of the day, only 20%–40% of memory actually held KV data.',
          'vLLM 论文实测：在当时主流的推理系统里，真正装着 KV 数据的显存只占 20%~40%。',
        )}
      >
        <p>
          {t(
            <>
              Knowing the KV cache is big, the next question is how to <em>place</em> it. Deep-learning framework tensors
              demand contiguous memory by default, so early inference systems (FasterTransformer, Orca, etc.) took a very
              blunt approach: when a request arrives, pre-allocate one contiguous block of memory sized to its{' '}
              <em>maximum possible length</em> (prompt + max_tokens) all at once. This seemingly harmless decision
              manufactured three kinds of waste.
            </>,
            <>
              知道了 KV Cache 很大，下一个问题是怎么<em>放</em>。深度学习框架的张量默认要求连续内存，
              所以早期推理系统（FasterTransformer、Orca 等）的做法非常直白：请求进来时，
              按它<em>可能达到的最大长度</em>（prompt + max_tokens）一次性预分配一整块连续显存。
              这个看似无害的决定，制造了三种浪费。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The first is{' '}
              <Term t="internal fragmentation">
                space inside an allocation unit that goes unused — fenced off, unusable by others, unused by yourself.
              </Term>{' '}
              : a user declares max_tokens=2048 but actually stops after generating 300 tokens — the slots for the
              remaining 1700-plus tokens are locked from the moment of allocation until the request ends. No one can know
              in advance when generation will stop, so this waste is incurable in any "fence off the worst case" scheme.
              The second is the waste reserved "just in case": even if the request really could grow to max_len, most of
              the reserved region sits empty until it gets there. The third is{' '}
              <Term t="external fragmentation">
                enough free total, but chopped into discontiguous little segments that nothing can fit into — the old
                affliction of the malloc era.
              </Term>{' '}
              : requests of varied lengths come and go, and memory gets riddled like a sieve — 30% still free, yet a new
              long request can't find a single contiguous span to fit in, and can only wait.
            </>,
            <>
              第一种是<Term t="内部碎片（internal fragmentation）">
                分配单元内部用不到的空间 —— 圈进来了，别人用不了，自己也没用上。
              </Term>：用户申报 max_tokens=2048，实际生成 300 个 token 就停了 ——
              剩下 1700 多个 token 的位置从分配那一刻起就被锁死，直到请求结束才释放。
              生成会在何时结束没人能提前知道，所以这种浪费在「按最坏情况圈地」的方案里无药可救。
              第二种是为「万一」预留的浪费：哪怕请求真能长到 max_len，在它长到之前，
              预留区的大部分也一直空着。第三种是<Term t="外部碎片（external fragmentation）">
                空闲总量足够，但被切成不连续的小段，谁都装不下 —— malloc 时代的老毛病。
              </Term>：长短不一的请求来来去去，显存被打成千疮百孔的筛子 ——
              明明还剩 30% 的空闲，新来的长请求却找不到一段放得下它的连续区间，只能干等。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The three wastes stacked together produce a shocking result: the vLLM paper (SOSP 2023) measured that in
              the mainstream systems of the day, only 20.4%–38.2% of memory actually held KV data —{' '}
              <strong>on an 80 GB card, more than half the memory idles under the name "already allocated."</strong> And
              memory utilization translates directly into concurrency, into throughput, into the cost per million
              tokens. The root cause is not the cache itself, but the assumption that "KV must be stored contiguously."
            </>,
            <>
              这三种浪费叠加的结果触目惊心：vLLM 论文（SOSP 2023）实测，
              当时的主流系统里真正存放着 KV 数据的显存只占 20.4%~38.2% ——
              <strong>一张 80GB 的卡，超过一半的显存以「已分配」的名义闲着。</strong>
              而显存利用率直接等于并发数、等于吞吐、等于每百万 token 的成本。
              问题的根源不在缓存本身，在「KV 必须连续存放」这个假设。
            </>,
          )}
        </p>
        <Callout type="note" title={t('The OS solved this problem back in the 1960s', '这个问题操作系统在 1960 年代就解过')}>
          <p>
            {t(
              <>
                The tension between the "contiguous address space" a program wants and physical memory's "fragmentation
                reality" was answered by the operating system long ago: virtual memory and paging — give the program a
                contiguous <em>logical</em> address space, and use a page table to map it onto physical page frames
                scattered anywhere. What PagedAttention does is lift this sixty-year-old wisdom intact into GPU memory:
                the token sequence is the logical address, a KV block is a page, and the block table is the page table.
              </>,
              <>
                程序要的「连续地址空间」和物理内存的「碎片现实」之间的矛盾，操作系统早就给出了答案：
                虚拟内存与分页 —— 给程序连续的<em>逻辑</em>地址，用页表映射到任意散落的<em>物理</em>页帧。
                PagedAttention 做的事，就是把这套六十年前的智慧原封不动搬进显存：
                token 序列是逻辑地址，KV block 是页，block table 就是页表。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={5}
        title={t('PagedAttention: fit the KV cache with a page table', 'PagedAttention：给 KV Cache 装上页表')}
        lead={t(
          'Slice memory into fixed-size blocks, allocate on demand, address through a table — waste drops from 60%+ to under 4%.',
          '把显存切成固定大小的 block，按需分配、查表寻址 —— 浪费从 60%+ 压到 4% 以下。',
        )}
      >
        <p>
          {t(
            <>
              vLLM's scheme has three steps. <strong>Chunking</strong>: slice the entire KV memory pool into fixed-size
              physical blocks, each holding a fixed number (say, 16) of tokens' K and V.{' '}
              <strong>On-demand allocation</strong>: when a request arrives, allocate just enough blocks for the prompt;
              afterward, only append one block per 16 tokens generated — no one fences off land for "a possible future"
              anymore, and internal fragmentation is compressed to the remainder of the last block (half a block on
              average, under 4%). <strong>Table-based addressing</strong>: each request keeps a block table mapping
              logical block numbers to physical block numbers — the physical blocks need not be contiguous at all, and
              external fragmentation vanishes: as long as the pool has any free block left, it can go to any request. The
              attention kernel is rewritten accordingly so that q, when computing, indexes through the table and reads
              the K, V scattered all over the place by jumping around.
            </>,
            <>
              vLLM 的方案分三步。<strong>切块</strong>：把整个 KV 显存池切成固定大小的物理块（block），
              每块存固定数量（如 16 个）token 的 K、V。<strong>按需分配</strong>：请求到来时只为
              prompt 分配刚好够用的块；之后每生成满 16 个 token 才追加一块 ——
              没人再为「可能的未来」圈地，内部碎片被压缩到最后一块的零头（平均半块，不到 4%）。
              <strong>查表寻址</strong>：每个请求维护一张 block table，把逻辑块号映射到物理块号 ——
              物理块完全不必连续，外部碎片就此消失：只要池子里还有任何一块空闲，就能分给任何请求。
              注意力 kernel 也相应改写，让 q 在计算时按表索引、跳着读散落各处的 K、V。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The simulator below feeds the same request stream (fixed random seed, identical timing of arrivals, growth,
              and early termination) to both allocators at once. Click the cards above or the segmented switch to toggle
              the view: the "contiguous pre-allocation" side shows large hatched areas — that's dead memory fenced off
              for max_len but never occupied by a token; the "PAGED" side is a colorful interlocking patchwork of blocks,
              nearly all of them full. Stare at the concurrency counts on both sides for a while: same pool, same request
              stream, and the paged side steadily holds nearly twice as many requests. Click any request's colored
              blocks and its block table unfolds on the right — the messier the lines, the better it proves that "not
              being physically contiguous" doesn't matter at all.
            </>,
            <>
              下面的模拟器把同一条请求流（固定随机种子，到达、增长、提前结束的时机完全一致）
              同时喂给两种分配器。点上方卡片或分段开关切换视图：「连续预分配」侧能看到大片斜线 ——
              那是按 max_len 圈下却没有 token 入住的死显存；「PAGED」侧五颜六色的块犬牙交错，
              几乎块块装满。盯着两边的并发数看一会儿：同一个池子、同一条请求流，
              分页侧稳定容纳的请求数接近翻倍。点击任意请求的色块，右侧会展开它的 block table ——
              连线越乱，越说明「物理上不连续」根本不碍事。
            </>,
          )}
        </p>
        <PagedLab />
        <p>
          {t(
            <>
              There are two more things the simulator doesn't draw but production systems can't live without. First,{' '}
              <strong>prefix sharing</strong>: the same system prompt often appears at the head of tens of thousands of
              requests, and after paging, these requests' block tables can point directly at the same batch of physical
              blocks — a thousand requests, one cache. Second, <strong>copy-on-write</strong>: in parallel sampling or
              beam search, multiple branches share a prefix, and only when a branch is about to write a new token is that
              block copied out — exactly how the OS's fork plays it. With this toolkit, vLLM compresses memory waste to
              under 4% and lifts throughput 2–4× over the systems of the day, all without touching a single one of the
              model's weights.
            </>,
            <>
              还有两件模拟器没画、但生产系统离不开的事。其一，<strong>前缀共享（prefix sharing）</strong>：
              同一个系统提示词（system prompt）往往出现在成千上万条请求开头，分页之后，
              这些请求的 block table 可以直接指向同一批物理块 —— 千份请求，一份缓存。
              其二，<strong>写时复制（copy-on-write）</strong>：并行采样或 beam search
              中多个分支共享前缀，只有当某个分支要写入新 token 时，才把那一块复制出去 ——
              和操作系统 fork 的玩法一模一样。靠这一套，vLLM 把显存浪费压到 4% 以下，
              吞吐相比当时的系统提升 2~4 倍，而这一切没有动模型的任何一个权重。
            </>,
          )}
        </p>
        <Quiz
          question={t('What core problem does PagedAttention solve?', 'PagedAttention 解决的核心问题是什么？')}
          options={[
            {
              text: t(
                'Reducing the FLOPs of attention computation so each token computes faster',
                '减少注意力计算的 FLOPs，让每个 token 算得更快',
              ),
              explain: t(
                "No. PagedAttention reduces no compute at all — the dot products that must be done are all still done, and reading by jumping around even adds a touch of overhead. What it optimizes is the way memory is occupied.",
                '不是。PagedAttention 不减少任何计算量 —— 该做的点积一次不少，甚至因为跳着读还略有开销。它优化的是显存的「占用方式」。',
              ),
            },
            {
              text: t(
                'Eliminating memory fragmentation, lifting the KV cache’s memory utilization from 20%–40% to over 96%',
                '消除显存碎片，把 KV Cache 的显存利用率从 20%~40% 提到 96% 以上',
              ),
              correct: true,
              explain: t(
                'Exactly. On-demand chunking kills internal fragmentation, table-based addressing kills external fragmentation — the same card can pack in twice as many concurrent requests, and throughput doubles with it. This is a victory of memory management, not of computation.',
                '正是。按需分块干掉内部碎片，查表寻址干掉外部碎片 —— 同一张卡能塞下两倍以上的并发请求，吞吐随之翻倍。这是内存管理的胜利，不是计算的胜利。',
              ),
            },
            {
              text: t(
                'Compressing the KV cache to lower precision to save space',
                '把 KV Cache 压缩到更低的精度以节省空间',
              ),
              explain: t(
                'That is KV cache quantization (e.g. fp8 KV), which is orthogonal to paging — the two can be used together. PagedAttention changes no numerical values, only the layout.',
                '那是 KV Cache 量化（如 fp8 KV），与分页正交 —— 两者可以同时用。PagedAttention 不改变任何数值，只改变摆放方式。',
              ),
            },
            {
              text: t(
                'Making attention no longer need to read all of the historical K, V',
                '让注意力不再需要读取全部历史 K、V',
              ),
              explain: t(
                'Wrong — every step of attention still reads all of history, it just jumps around following the block table’s directions. To reduce "how much is read," you take the route of sliding-window attention or GQA.',
                '不对，每一步注意力依然要读全部历史 —— 只是按 block table 的指引跳着读。想减少「读多少」，要靠滑动窗口注意力或 GQA 那条路。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={6}
        title={t('Summary and further reading', '总结与延伸阅读')}
        lead={t(
          'In one line: the KV cache trades memory for compute; PagedAttention uses a page table to rescue the memory.',
          '一句话：KV Cache 用显存换计算，PagedAttention 用页表救显存。',
        )}
      >
        <ul>
          <li>
            {t(
              <>
                <strong>Validity of caching</strong>: under a causal mask, a history token's K and V never change and
                are queried again every step — compute once, store it, and decode drops from O(S²·d) to O(S·d) per step.
                Q is used up immediately, so it is not cached.
              </>,
              <>
                <strong>缓存的合法性</strong>：因果掩码下历史 token 的 K、V 永不改变、且被每一步反复查询 ——
                算一次存起来，decode 每步从 O(S²·d) 降到 O(S·d)。Q 用完即弃，不缓存。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The memory bill</strong>: bytes = 2·L·n_kv·d_head·2B; LLaMA-7B is 512 KB per token, ~2 GB for one
                4K request — under long context, the KV cache is memory's largest occupant. GQA cuts n_kv from 32 to 8,
                shrinking the cache 4×.
              </>,
              <>
                <strong>显存账</strong>：bytes = 2·L·n_kv·d_head·2B，LLaMA-7B 每 token 512KB、4K 一条请求约
                2GB —— 长上下文下 KV Cache 是显存的最大住户。GQA 把 n_kv 从 32 砍到 8，缓存即缩 4 倍。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>The pain of fragmentation</strong>: contiguous pre-allocation by max_len causes internal +
                external fragmentation; mainstream systems measured KV memory utilization at just 20.4%–38.2%.
              </>,
              <>
                <strong>碎片之痛</strong>：按 max_len 连续预分配造成内部 + 外部碎片，
                主流系统实测 KV 显存利用率仅 20.4%~38.2%。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>PagedAttention</strong>: fixed-size blocks + on-demand allocation + block-table addressing pushes
                waste below 4%; prefix sharing and copy-on-write let identical prefixes be stored only once. Throughput
                up 2–4×, purely from memory management, without touching a single model weight.
              </>,
              <>
                <strong>PagedAttention</strong>：固定大小 block + 按需分配 + block table 查表寻址，
                浪费压到 4% 以下；prefix 共享与 copy-on-write 让相同前缀只存一份。
                吞吐提升 2~4 倍，纯靠内存管理，不动模型一个权重。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Dig deeper:', '继续深挖：')}</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">
              Efficient Memory Management for Large Language Model Serving with PagedAttention
            </a>{' '}
            {t(
              '— the vLLM paper (SOSP 2023); all the data in Sections 4 and 5 of this chapter come from here, and it reads beautifully.',
              '—— vLLM 论文（SOSP 2023），本章 SEC 4、5 的数据全部出自这里，写得非常好读。',
            )}
          </li>
          <li>
            <a href="https://docs.vllm.ai/" target="_blank" rel="noreferrer">
              {t('vLLM official docs', 'vLLM 官方文档')}
            </a>{' '}
            {t(
              '— see what knobs like block_size, gpu_memory_utilization, and prefix caching look like in a real system.',
              '—— 看看 block_size、gpu_memory_utilization、prefix caching 这些旋钮在真实系统里长什么样。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noreferrer">
              GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
            </a>{' '}
            {t(
              '— the GQA paper: how to convert an MHA model into GQA with just 5% of continued pre-training.',
              '—— GQA 论文：如何用 5% 的继续预训练把 MHA 模型改造成 GQA。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noreferrer">
              Fast Transformer Decoding: One Write-Head is All You Need
            </a>{' '}
            {t(
              '— Shazeer 2019’s MQA paper, which saw the KV-bandwidth bottleneck four years ahead of everyone else.',
              '—— Shazeer 2019 的 MQA 论文，比所有人早四年看到了 KV 带宽这个瓶颈。',
            )}
          </li>
          <li>
            <a href="https://kipp.ly/transformer-inference-arithmetic/" target="_blank" rel="noreferrer">
              Transformer Inference Arithmetic (kipply)
            </a>{' '}
            {t(
              '— the classic blog post that works out inference’s memory and latency bills to the last detail with pen and paper, the spiritual ancestor of this chapter’s calculator.',
              '—— 用纸笔把推理的显存与延迟账算到底的经典博客，本章计算器的精神前辈。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
