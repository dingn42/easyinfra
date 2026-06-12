import { Callout, CodeBlock, MathTex, Quiz, Section, Term } from '@/components/ui'
import CacheCompareLab from './CacheCompareLab'
import KVCalcLab from './KVCalcLab'
import PagedLab from './PagedLab'

const DECODE_LOOP = `# 无 cache：每生成一个 token，整段序列重新过一遍模型
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
  return (
    <>
      <p>
        想象一个不加任何缓存的聊天机器人：它回答到第 1000 个 token 时，为了吐出第 1001 个，
        要把前面 1000 个 token <em>从头到尾重新算一遍</em> —— 而且每吐一个字都要这样来一次。
        这不是夸张，这就是 Transformer 自回归生成最朴素的形态。KV Cache 把它从这种荒谬中救了下来：
        历史只算一次，之后查表即可。但天下没有免费的午餐 —— 省下的计算变成了吃显存的怪物，
        一张 80GB 的 A100，模型权重占掉一块，剩下的几乎全是 KV Cache 的地盘。
        这块地盘管得好不好，直接决定一台推理服务器能同时伺候 5 个用户还是 50 个。
        本章讲清楚两件事：KV Cache 为什么成立，以及 vLLM 的 PagedAttention 怎么把这块显存管到极致。
      </p>

      <Section
        index={1}
        title="自回归生成：能缓存的，只有 K 和 V"
        lead="缓存的合法性来自一个安静的事实：在因果掩码下，历史 token 的 K、V 一旦算出，就永远不变。"
      >
        <p>
          大语言模型的生成是<Term t="自回归（autoregressive）">
            每次只生成一个 token，并把它拼回输入末尾作为下一步预测的条件 —— 输出依赖自己之前的全部输出。
          </Term>的：第 <MathTex tex="t" /> 步拿到前面全部 <MathTex tex="t-1" /> 个 token，预测第{' '}
          <MathTex tex="t" /> 个。回忆第七章的注意力机制，新 token 在每一层要做的事是：
          用自己的查询向量 <MathTex tex="q_t" /> 去和<em>所有历史 token</em> 的键向量做点积，
          再用得到的权重对所有历史的值向量加权求和：
        </p>
        <MathTex block tex="o_t = \mathrm{softmax}\!\left(\frac{q_t K_{1:t}^{\top}}{\sqrt{d}}\right) V_{1:t}" />
        <p>
          注意这个式子里三个角色的待遇完全不同。<strong>Q 只需要当前这一个</strong>：
          历史 token 的 q 向量在它们自己的那一步已经用完了，之后再也不会被任何人查询，存着没有意义。
          而 <strong>K 和 V 是被反复查询的一方</strong>：第 1 个 token 的{' '}
          <MathTex tex="k_1, v_1" /> 会被第 2、3、…、1000 步反复用到。更关键的是，
          <MathTex tex="k_i = x_i W_K" />、<MathTex tex="v_i = x_i W_V" /> 只依赖第{' '}
          <MathTex tex="i" /> 个位置自己的表示 —— 因果掩码（causal mask）保证后来的 token
          不会回头改写历史的隐状态，所以这些 K、V 算出来一次，终生有效。
          <strong>只依赖自身、又被反复读取 —— 这正是「值得缓存」的教科书定义。</strong>
        </p>
        <p>
          于是推理被切成了两个性格迥异的阶段。<strong>预填充（prefill）</strong>：
          把整段 prompt 一次性并行过模型，算出所有层、所有位置的 K、V 存进缓存 ——
          这是一次大矩阵乘，算力被吃满。<strong>解码（decode）</strong>：之后每步只把<em>一个</em>新
          token 喂进模型，算出它的 q、k、v，把 k、v 追加进缓存，再用 q 查询整个缓存。
          每步注意力的代价从重算全部的 <MathTex tex="O(S^2 \cdot d)" /> 降到{' '}
          <MathTex tex="O(S \cdot d)" /> —— 整整少了一个 S 的量级，而 S 动辄是几千上万。
        </p>
        <CodeBlock code={DECODE_LOOP} lang="python" title="decode_loop.py" />
        <Callout type="insight" title="KV Cache 是一笔「显存换计算」的交易">
          <p>
            缓存把 decode 每步的计算量砍掉一个数量级，但每个 token、每一层的 K 和 V
            都要常驻显存 —— 而且每步还得把整个缓存从 HBM 读一遍。结果是 decode
            从「算不过来」变成了「读不过来」：用第六章 roofline 的语言说，它是个典型的
            memory-bound 工作负载。这笔交易把战场从算力挪到了显存 ——
            容量决定你能同时服务多少人，带宽决定每个 token 出得多快。本章管容量，下一章管带宽。
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title="实验：亲眼看看省掉的计算"
        lead="同一段 32 个 token 的生成，左边每步重算整个注意力矩阵，右边只追加一行。"
      >
        <p>
          下面的实验把两种算法摆在一起逐 token 播放。每个小格子是注意力 score 矩阵里的一个单元
          （一次 <MathTex tex="q \cdot k" /> 点积）：<span className="text-volt">荧光绿</span>是本步真正新算的格子，
          <span className="text-rose">玫红</span>是无缓存一侧每步都在白白重算的旧格子，
          <span className="text-amber">琥珀色</span>是缓存命中 —— 数据躺在显存里，只读不算。
          注意右侧每步只新增薄薄一行，而左侧整个三角形每步都要重新烧一遍。
        </p>
        <CacheCompareLab />
        <p>
          曲线读数值得多看一眼：切到「每步」，无缓存的代价随 t 二次增长（整个 t×t 三角），
          有缓存只是线性的一行；切到「累计」，差距进一步拉开成 <MathTex tex="O(S^3)" /> 对{' '}
          <MathTex tex="O(S^2)" />。把目标长度从 32 拉到 48，你会看到节省倍数还在继续上涨 ——
          序列越长，缓存越是救命。这也是为什么没有任何一个生产推理引擎敢不开 KV Cache：
          它不是优化选项，是入场券。
        </p>
        <Quiz
          question="开启 KV Cache 之后，decode 阶段每生成一个新 token，注意力的计算复杂度是多少（S 为当前序列长度，d 为隐藏维度）？"
          options={[
            {
              text: <MathTex tex="O(S^2 \cdot d)" />,
              explain: '这是无缓存时每步重算整个注意力矩阵的代价。缓存的意义恰恰是把这个 S² 砍掉一个 S。',
            },
            {
              text: <MathTex tex="O(S \cdot d)" />,
              correct: true,
              explain:
                '正是。新 token 的 q 要和缓存里 S 个 k 做点积（S·d），再对 S 个 v 加权求和（S·d）—— 整体 O(S·d)。代价是这 S 份 K、V 都得常驻显存，且每步都要从 HBM 读一遍。',
            },
            {
              text: <MathTex tex="O(d^2)" />,
              explain:
                'O(d²) 是新 token 过线性投影层（算 q/k/v）的代价，那部分确实和 S 无关；但注意力本身还要查询全部 S 份历史 K/V，是 O(S·d)。',
            },
            {
              text: <MathTex tex="O(1)" />,
              explain: '可惜不行 —— 注意力天生要看全部历史，每步至少要把 S 份 K、V 读一遍、乘一遍，逃不开 O(S·d)。',
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title="显存账：每个 token 都在交房租"
        lead="一条 4K 上下文的 LLaMA-7B 请求，KV Cache 要 2GB —— 比很多人想象的大两个数量级。"
      >
        <p>
          KV Cache 到底多大？这笔账值得手算一遍，因为每一项都对应一个真实的架构决策。
          fp16 精度下，每个 token 在缓存里占：
        </p>
        <MathTex block tex="\text{bytes/token} = \underbrace{2}_{K\,\text{和}\,V} \times \underbrace{L}_{\text{层数}} \times \underbrace{n_{kv}}_{\text{KV 头数}} \times \underbrace{d_{head}}_{\text{每头维度}} \times \underbrace{2}_{\text{fp16 字节}}" />
        <p>
          再乘上序列长度 S 是单条请求的占用，乘上 batch 是整张卡上的总量。代入 LLaMA-7B
          的真实参数（L=32，32 个头，每头 128 维）：每个 token 512KB —— 一个 token 的缓存比一整张高清图片还大。
          4K 上下文的一条请求就是 2GB，是 7B 模型 13.5GB 权重的七分之一；上下文拉到 32K，
          一条请求 16GB，权重反倒成了小头。这就是「显存成为新战场」的字面意思：
          <strong>长上下文时代，显存的主要住户不是权重，是 KV Cache。</strong>
        </p>
        <p>
          公式里最值得玩味的一项是 <MathTex tex="n_{kv}" />。标准的多头注意力（MHA, Multi-Head
          Attention）里 KV 头数等于注意力头数；而<strong>分组查询注意力（GQA, Grouped-Query
          Attention）</strong>让多个 Q 头共享同一组 K、V 头 —— LLaMA-2 70B 用 64 个 Q 头查询仅仅
          8 个 KV 头，缓存直接缩小 8 倍，模型质量几乎无损。更极端的 <strong>MQA（Multi-Query
          Attention）</strong>干脆让全部 Q 头共享 1 组 KV。这些不是炫技：它们是工程账倒逼出来的架构设计 ——
          先有「KV Cache 太贵」这个痛点，才有 GQA 进入几乎所有现代开源模型的标配清单。
        </p>
        <KVCalcLab />
        <p>
          用计算器做三个实验：① 选 7B、4K 上下文，把 batch 从 8 拉到 64 —— 看 KV
          什么时候撞上红线；② 打开 GQA 开关，KV 条瞬间缩成四分之一，最大并发翻四倍 ——
          一行架构改动顶得上半张卡；③ 切到 70B —— 权重一项就越过 80GB
          红线，这就是为什么 70B 级别的模型单卡免谈，必须等到第十二章的张量并行来拆。
        </p>
      </Section>

      <Section
        index={4}
        title="碎片化：被锁死的显存比用掉的还多"
        lead="vLLM 论文实测：在当时主流的推理系统里，真正装着 KV 数据的显存只占 20%~40%。"
      >
        <p>
          知道了 KV Cache 很大，下一个问题是怎么<em>放</em>。深度学习框架的张量默认要求连续内存，
          所以早期推理系统（FasterTransformer、Orca 等）的做法非常直白：请求进来时，
          按它<em>可能达到的最大长度</em>（prompt + max_tokens）一次性预分配一整块连续显存。
          这个看似无害的决定，制造了三种浪费。
        </p>
        <p>
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
        </p>
        <p>
          这三种浪费叠加的结果触目惊心：vLLM 论文（SOSP 2023）实测，
          当时的主流系统里真正存放着 KV 数据的显存只占 20.4%~38.2% ——
          <strong>一张 80GB 的卡，超过一半的显存以「已分配」的名义闲着。</strong>
          而显存利用率直接等于并发数、等于吞吐、等于每百万 token 的成本。
          问题的根源不在缓存本身，在「KV 必须连续存放」这个假设。
        </p>
        <Callout type="note" title="这个问题操作系统在 1960 年代就解过">
          <p>
            程序要的「连续地址空间」和物理内存的「碎片现实」之间的矛盾，操作系统早就给出了答案：
            虚拟内存与分页 —— 给程序连续的<em>逻辑</em>地址，用页表映射到任意散落的<em>物理</em>页帧。
            PagedAttention 做的事，就是把这套六十年前的智慧原封不动搬进显存：
            token 序列是逻辑地址，KV block 是页，block table 就是页表。
          </p>
        </Callout>
      </Section>

      <Section
        index={5}
        title="PagedAttention：给 KV Cache 装上页表"
        lead="把显存切成固定大小的 block，按需分配、查表寻址 —— 浪费从 60%+ 压到 4% 以下。"
      >
        <p>
          vLLM 的方案分三步。<strong>切块</strong>：把整个 KV 显存池切成固定大小的物理块（block），
          每块存固定数量（如 16 个）token 的 K、V。<strong>按需分配</strong>：请求到来时只为
          prompt 分配刚好够用的块；之后每生成满 16 个 token 才追加一块 ——
          没人再为「可能的未来」圈地，内部碎片被压缩到最后一块的零头（平均半块，不到 4%）。
          <strong>查表寻址</strong>：每个请求维护一张 block table，把逻辑块号映射到物理块号 ——
          物理块完全不必连续，外部碎片就此消失：只要池子里还有任何一块空闲，就能分给任何请求。
          注意力 kernel 也相应改写，让 q 在计算时按表索引、跳着读散落各处的 K、V。
        </p>
        <p>
          下面的模拟器把同一条请求流（固定随机种子，到达、增长、提前结束的时机完全一致）
          同时喂给两种分配器。点上方卡片或分段开关切换视图：「连续预分配」侧能看到大片斜线 ——
          那是按 max_len 圈下却没有 token 入住的死显存；「PAGED」侧五颜六色的块犬牙交错，
          几乎块块装满。盯着两边的并发数看一会儿：同一个池子、同一条请求流，
          分页侧稳定容纳的请求数接近翻倍。点击任意请求的色块，右侧会展开它的 block table ——
          连线越乱，越说明「物理上不连续」根本不碍事。
        </p>
        <PagedLab />
        <p>
          还有两件模拟器没画、但生产系统离不开的事。其一，<strong>前缀共享（prefix sharing）</strong>：
          同一个系统提示词（system prompt）往往出现在成千上万条请求开头，分页之后，
          这些请求的 block table 可以直接指向同一批物理块 —— 千份请求，一份缓存。
          其二，<strong>写时复制（copy-on-write）</strong>：并行采样或 beam search
          中多个分支共享前缀，只有当某个分支要写入新 token 时，才把那一块复制出去 ——
          和操作系统 fork 的玩法一模一样。靠这一套，vLLM 把显存浪费压到 4% 以下，
          吞吐相比当时的系统提升 2~4 倍，而这一切没有动模型的任何一个权重。
        </p>
        <Quiz
          question="PagedAttention 解决的核心问题是什么？"
          options={[
            {
              text: '减少注意力计算的 FLOPs，让每个 token 算得更快',
              explain:
                '不是。PagedAttention 不减少任何计算量 —— 该做的点积一次不少，甚至因为跳着读还略有开销。它优化的是显存的「占用方式」。',
            },
            {
              text: '消除显存碎片，把 KV Cache 的显存利用率从 20%~40% 提到 96% 以上',
              correct: true,
              explain:
                '正是。按需分块干掉内部碎片，查表寻址干掉外部碎片 —— 同一张卡能塞下两倍以上的并发请求，吞吐随之翻倍。这是内存管理的胜利，不是计算的胜利。',
            },
            {
              text: '把 KV Cache 压缩到更低的精度以节省空间',
              explain:
                '那是 KV Cache 量化（如 fp8 KV），与分页正交 —— 两者可以同时用。PagedAttention 不改变任何数值，只改变摆放方式。',
            },
            {
              text: '让注意力不再需要读取全部历史 K、V',
              explain:
                '不对，每一步注意力依然要读全部历史 —— 只是按 block table 的指引跳着读。想减少「读多少」，要靠滑动窗口注意力或 GQA 那条路。',
            },
          ]}
        />
      </Section>

      <Section index={6} title="总结与延伸阅读" lead="一句话：KV Cache 用显存换计算，PagedAttention 用页表救显存。">
        <ul>
          <li>
            <strong>缓存的合法性</strong>：因果掩码下历史 token 的 K、V 永不改变、且被每一步反复查询 ——
            算一次存起来，decode 每步从 O(S²·d) 降到 O(S·d)。Q 用完即弃，不缓存。
          </li>
          <li>
            <strong>显存账</strong>：bytes = 2·L·n_kv·d_head·2B，LLaMA-7B 每 token 512KB、4K 一条请求约
            2GB —— 长上下文下 KV Cache 是显存的最大住户。GQA 把 n_kv 从 32 砍到 8，缓存即缩 4 倍。
          </li>
          <li>
            <strong>碎片之痛</strong>：按 max_len 连续预分配造成内部 + 外部碎片，
            主流系统实测 KV 显存利用率仅 20.4%~38.2%。
          </li>
          <li>
            <strong>PagedAttention</strong>：固定大小 block + 按需分配 + block table 查表寻址，
            浪费压到 4% 以下；prefix 共享与 copy-on-write 让相同前缀只存一份。
            吞吐提升 2~4 倍，纯靠内存管理，不动模型一个权重。
          </li>
        </ul>
        <p>继续深挖：</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">
              Efficient Memory Management for Large Language Model Serving with PagedAttention
            </a>{' '}
            —— vLLM 论文（SOSP 2023），本章 SEC 4、5 的数据全部出自这里，写得非常好读。
          </li>
          <li>
            <a href="https://docs.vllm.ai/" target="_blank" rel="noreferrer">
              vLLM 官方文档
            </a>{' '}
            —— 看看 block_size、gpu_memory_utilization、prefix caching 这些旋钮在真实系统里长什么样。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noreferrer">
              GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
            </a>{' '}
            —— GQA 论文：如何用 5% 的继续预训练把 MHA 模型改造成 GQA。
          </li>
          <li>
            <a href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noreferrer">
              Fast Transformer Decoding: One Write-Head is All You Need
            </a>{' '}
            —— Shazeer 2019 的 MQA 论文，比所有人早四年看到了 KV 带宽这个瓶颈。
          </li>
          <li>
            <a href="https://kipp.ly/transformer-inference-arithmetic/" target="_blank" rel="noreferrer">
              Transformer Inference Arithmetic（kipply）
            </a>{' '}
            —— 用纸笔把推理的显存与延迟账算到底的经典博客，本章计算器的精神前辈。
          </li>
        </ul>
      </Section>
    </>
  )
}
