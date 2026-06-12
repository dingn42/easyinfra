import { Callout, Quiz, Section, Term } from '@/components/ui'
import { PrefillDecodeLab } from './PrefillDecodeLab'
import { BatchingCurves } from './BatchingCurves'
import { StaticBatchTimeline } from './StaticBatchTimeline'
import { ContinuousBatchingLab } from './ContinuousBatchingLab'
import { ChunkedPrefillSVG, PreemptionSVG, ServingStackSVG, SpeculativeSVG } from './StackDiagrams'

export default function Chapter() {
  return (
    <>
      <p>
        打开任何一个大模型对话产品，你都会觉得模型在专心陪你：你一发问，它马上开始逐字往外蹦。
        但在机房里，那张 H100 此刻可能同时挂着几十个会话 —— 你的问题、别人的代码补全、某个
        Agent 的第 17 次工具调用，全都挤在同一块芯片上轮流取暖。让每个用户都以为自己独占模型，
        靠的不是更多的卡，而是<strong>调度</strong>。这一章我们拆开推理引擎（inference engine），
        看看 vLLM、TensorRT-LLM 这类系统到底在调度什么、为什么一个好调度器能把同一张卡的吞吐
        抬高一个数量级。
      </p>

      <Section
        index={1}
        title="一个请求的两段人生"
        lead="同一个请求，在 GPU 上活成了两种完全不同的样子。"
      >
        <p>
          一个推理请求从进入引擎到吐完最后一个字，经历两个性格迥异的阶段。第一段叫{' '}
          <Term t="prefill（预填充）">
            把整段 prompt 一次性送进模型，并行计算出所有位置的 KV cache 和第一个输出 token。
          </Term>
          ：prompt 里的几百上千个 token 是<em>已知的</em>，它们可以拼成一个大矩阵、一次前向全部算完。
          这是一场酣畅淋漓的大矩阵乘 —— 算术强度（arithmetic intensity）随 prompt 长度线性上涨，
          GPU 的张量核心被喂得很饱，有效算力利用率能到 60%~70%。
        </p>
        <p>
          第二段叫{' '}
          <Term t="decode（解码）">
            自回归生成阶段：每次前向只处理一个新 token，算完采样出下一个，再喂回去。
          </Term>
          ，画风急转直下。自回归（autoregressive）生成天生串行：第 N+1 个 token 必须等第 N 个出来才能算。
          于是每一步前向只处理<em>一个</em> token —— 但模型的全部权重一个字节都不能少读。回忆第 9 章的账：
          70B 模型 FP16 权重 140 GB，H100 的 HBM 带宽 3.35 TB/s，光把权重过一遍就要约 42 ms；
          而这一步真正的计算量只有约 140 GFLOPs，对 989 TFLOPS 的峰值算力来说是九牛一毛 ——
          有效利用率掉到 1% 以下。同一张卡，从满载的炼钢炉变成了几乎空转的传送带。
        </p>
        <p>
          这两段人生分别对应两个用户最敏感的延迟指标：<strong>TTFT</strong>（Time To First Token，首
          token 延迟）几乎完全由 prefill 决定 —— prompt 越长、排队越久，你盯着光标发呆的时间就越长；
          <strong>TPOT</strong>（Time Per Output Token，每 token 间隔）由 decode 决定 ——
          它决定了字往外蹦的速度。下面这个实验把两段人生并排放在一起，注意右侧 roofline
          上两个点的位置：prefill 顶着算力屋顶，decode 深陷带宽斜坡。
        </p>
        <PrefillDecodeLab />
        <Quiz
          question="batch=1 时 decode 的算力利用率不到 1%，根本原因是？"
          options={[
            {
              text: '自回归生成是串行的，GPU 来不及做指令级并行',
              explain:
                '串行确实是表象，但单步内部 GPU 并不缺并行度（一个 token 也要过全部矩阵乘）。真正的瓶颈在访存。',
            },
            {
              text: 'decode 是 memory-bound：每生成一个 token 都要把全部权重从 HBM 读一遍，batch=1 时这笔读取成本完全摊不薄',
              correct: true,
              explain:
                '对。每步计算量 ≈ 2P FLOPs，权重读取 ≈ 2P 字节，算术强度 ≈ 1 FLOP/Byte，离 H100 的 ridge（约 295）差两个数量级。时间全花在搬权重上，算力大块闲置 —— 这也是后面所有优化的出发点。',
            },
            {
              text: 'KV cache 太大，挤占了计算单元',
              explain:
                'KV cache 占的是显存容量和一部分带宽，不会「挤占计算单元」。batch=1 时主要的搬运量是权重本身。',
            },
            {
              text: '采样（sampling）这一步只能在 CPU 上做，拖慢了整体节奏',
              explain: '现代引擎的采样都在 GPU 上完成，开销很小，不是主要矛盾。',
            },
          ]}
        />
      </Section>

      <Section
        index={2}
        title="为什么 batching 能救 decode"
        lead="权重只读一遍，token 算几十个 —— 这是推理系统第一性的省钱逻辑。"
      >
        <p>
          decode 的病根是「读得多、算得少」，那药方就很直接：<strong>让一次权重读取服务更多的计算</strong>。
          把 32 个用户的 decode 步拼成一批（batch），每步前向还是把 140 GB 权重读一遍，但这一遍现在产出
          32 个 token —— 算术强度直接乘 32。只要还没碰到算力屋顶，吞吐随 batch 大小<em>线性</em>上涨，
          而每个用户感受到的 TPOT 几乎不变：反正时间都花在读权重上，多算 31 个 token 是顺手的事。
          这就是推理服务里少见的「近乎免费的午餐」。
        </p>
        <p>
          免费午餐当然有边界。batch 大到一定程度，计算时间追上了访存时间 —— 这正是第 6 章 roofline 的
          ridge point。越过它，decode 从 memory-bound 翻身成 compute-bound：吞吐不再涨，单步时延开始随
          batch 线性变长，TPOT 肉眼可见地变差。下面的曲线图把这两条线画在一起，切换硬件看拐点怎么动：
          带宽越大的卡（H200），同样的吞吐上限来得越早 —— 用更小的 batch 就能喂饱。
        </p>
        <BatchingCurves />
        <Callout type="insight" title="推理系统的第一性原理">
          <p>
            把这章所有花哨的调度技术倒推回去，根都是同一句话：<strong>decode 每一步都要付一次「读全部权重」的
            固定成本，调度器的使命是让每一次付费服务尽可能多的 token</strong>。batching 是在「用户数」维度摊薄它，
            speculative decoding 是在「时间」维度摊薄它（一次前向验证多个草稿 token），而连续批处理保证这个
            摊薄在任何时刻都不掉线。看懂了这一点，vLLM 的源码就只剩工程细节。
          </p>
        </Callout>
      </Section>

      <Section
        index={3}
        title="静态 batching 的木桶效应"
        lead="把请求捆成一批容易，难的是它们不肯一起结束。"
      >
        <p>
          知道了 batching 是答案，最朴素的实现长这样：攒一批请求（比如 8 个），整批做 prefill，
          整批 decode，整批结束后再放下一批进来 —— 这叫<strong>静态批处理（static batching）</strong>，
          也是早年很多自研服务的真实写法。它的问题藏在一个不起眼的事实里：<em>没人知道一个请求会生成多长</em>。
          有人问「翻译这个词」，16 个 token 就完事；有人要「写一篇两千字总结」，要跑上几百步。
          整批同进同出，意味着所有人都要陪最长的那个跑完全程。
        </p>
        <p>
          看下面的时间线：先完成的请求并不能把槽位让出来，只能挂着空转（图里的灰色斜纹）——
          GPU 在为「已经没有活的槽位」白白付权重读取的钱，实际有效 batch 越跑越小。更糟的是批与批之间的
          硬边界：哪怕新请求在批次开始后 1 秒就到了，它也得等整批结束才能上车，平均等待时间随之暴涨。
          输出长度方差越大，木桶效应越狠 —— 这不是实现不够好，是「批」这个粒度本身错了。
        </p>
        <StaticBatchTimeline />
      </Section>

      <Section
        index={4}
        title="连续批处理：完成即让位"
        lead="把调度粒度从「一批请求」降到「一个迭代」，一切都变了。"
      >
        <p>
          2022 年的 Orca 论文捅破了这层窗户纸：既然 decode 本来就是一步一步迭代的，为什么调度也要以
          「整批」为单位？把决策点搬到<strong>每个迭代（iteration）之间</strong>：每走完一步，
          谁生成完了就立刻离场、返还槽位；队列里有新请求就立刻补位，先做它的 prefill 再加入 decode 大部队。
          这就是<strong>连续批处理（continuous batching，也叫 in-flight batching / iteration-level
          scheduling）</strong>—— 今天 vLLM、TensorRT-LLM、SGLang 的调度核心无一例外。
        </p>
        <p>
          效果不是百分之几十的改良。批内不再有人陪跑，GPU 的有效 batch 始终贴着并发上限；
          新请求的等待从「等整批结束」变成「等下一个迭代」。在输出长度方差大的真实负载下，
          吞吐提升常见 2~5 倍，尾延迟（P95/P99）的改善更夸张。下面的模拟器用同一条固定种子的请求流
          回放两种调度：先看 STATIC 跑一遍，注意利用率条的锯齿状塌陷；切到 CONTINUOUS，
          看新请求怎么随时填进空槽、利用率怎么被钉在高位。再试着把到达速率 λ 调高、把输出区间拉大 ——
          差距会更难看。
        </p>
        <ContinuousBatchingLab />
        <Quiz
          question="连续批处理相对静态批处理，本质的改变是什么？"
          options={[
            {
              text: '用了更大的 batch size，所以吞吐更高',
              explain:
                '并发上限（槽位数）两种调度可以完全一样。连续批处理赢在槽位的利用率，不在槽位的数量。',
            },
            {
              text: '把 prefill 和 decode 分到了两张不同的卡上',
              explain:
                '那是 prefill/decode 分离（disaggregation，如 DistServe），是另一项正交的技术，第 5 节会提到。',
            },
            {
              text: '把调度粒度从「请求批」降到「迭代/token 级」：每步之间谁完成谁立刻让位，新请求随时补位，有效 batch 永远是满的',
              correct: true,
              explain:
                '对。关键洞察是 decode 本身就以迭代为节拍，调度器没有理由用比它粗的粒度做决策。完成即释放 + 随到随补，木桶效应就消失了。',
            },
            {
              text: '它会预测每个请求的输出长度，把长度相近的请求分到一批',
              explain:
                '长度预测调度确实有人研究，但连续批处理的妙处恰恰是不需要预测 —— 谁先完成谁先走，调度天然自适应。',
            },
          ]}
        />
      </Section>

      <Section
        index={5}
        title="进阶调度：把每一毫秒都抠出来"
        lead="连续批处理解决了主要矛盾，剩下的尖刺要靠三件趁手的工具。"
      >
        <p>
          <strong>chunked prefill（分块预填充）。</strong>连续批处理有个隐藏的副作用：新请求插队做 prefill
          时，一段 2000 token 的 prompt 是一大块 compute-bound 的计算，会把那个迭代撑得很长 ——
          正在 decode 的老用户会突然感到字停了一下，TPOT 出现尖刺。解法是把长 prompt 切成几百 token
          的 chunk，每个迭代只混入一小块，和 decode token 拼成一个不大不小的混合批。代价是 TTFT
          稍微变长，换来的是所有人 TPOT 的平稳 —— 这是典型的「拿峰值换方差」。
        </p>
        <ChunkedPrefillSVG />
        <p>
          <strong>抢占与换出（preemption &amp; swap/recompute）。</strong>连续批处理让显存变成了动态资源：
          每个在跑的请求都拖着一条越长越胖的 KV cache，请求一多显存就可能爆。调度器必须有止损手段 ——
          挑一个倒霉蛋（通常是最晚到的）请出去：把它的 KV 整体换出（swap）到主机内存，等有空再搬回来；
          或者干脆丢弃，重新排队时用一次 prefill 重算（recompute）。换出费 PCIe 带宽，重算费算力，
          vLLM 会按 KV 大小权衡选择。对用户来说这只是延迟波动，比 OOM 崩掉整个引擎好得多。
        </p>
        <PreemptionSVG />
        <p>
          <strong>speculative decoding（投机解码）。</strong>前面的招都在摊薄权重读取，这一招直接向
          「串行」本身开刀。找一个小得多的草稿模型（draft model）先串行猜 k 个 token —— 它小，猜得快；
          然后让大模型<em>一次前向</em>并行验证这 k 个位置，数学上可以做到与大模型逐个采样完全同分布。
          命中 3 个，就等于用一次「大模型时间」生产了 4 个 token（含修正）。本质上它把串行 decode
          变成了并行验证 —— 用 prefill 的姿势跑 decode 的活。草稿越准、收益越大，代价是要多养一个小模型
          （或像 EAGLE/Medusa 那样在大模型头上长出草稿头）。
        </p>
        <SpeculativeSVG />
        <p>
          把这些零件装回整机：一个生产级推理栈大致分四层。最上面的 Router 在多个引擎副本之间做负载均衡、
          按 prefix 亲和性路由（同一会话尽量打到缓存了它 KV 的副本）；Scheduler 做本章的主角 ——
          连续批处理、chunked prefill、抢占；Engine 管 KV cache 的分页与复用（第 9 章的 PagedAttention
          就活在这里）；最底下是 Kernels —— FlashAttention、融合 GEMM 这些第 8 章的功臣。更激进的架构如
          DistServe 干脆把 prefill 和 decode 拆到不同的 GPU 池子里（disaggregation），让 compute-bound
          和 memory-bound 的两段人生各自找最适合的硬件配比。
        </p>
        <ServingStackSVG />
      </Section>

      <Section
        index={6}
        title="指标与权衡：吞吐和延迟没有买一送一"
        lead="调参之前，先把你到底在优化什么说清楚。"
      >
        <p>推理服务的运营语言由四个指标构成，每个都站在不同的立场上：</p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">指标</th>
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">定义</th>
                <th className="py-2 font-mono text-[11px] uppercase tracking-wider text-ink3">谁在乎 / 由什么决定</th>
              </tr>
            </thead>
            <tbody className="text-text">
              <tr className="border-b border-line">
                <td className="py-2.5 pr-4 font-mono text-cyan">TTFT</td>
                <td className="py-2.5 pr-4">从发出请求到收到第一个 token</td>
                <td className="py-2.5 text-ink2">用户的「开始响应」体感；由排队等待 + prefill 决定</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2.5 pr-4 font-mono text-cyan">TPOT</td>
                <td className="py-2.5 pr-4">相邻输出 token 的平均间隔</td>
                <td className="py-2.5 text-ink2">字往外蹦的流畅度；由 decode 单步时延决定，随 batch 变差</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2.5 pr-4 font-mono text-volt">吞吐</td>
                <td className="py-2.5 pr-4">整张卡每秒产出的 token 总数</td>
                <td className="py-2.5 text-ink2">运营者的账单；batch 越大越高，直到 ridge</td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 font-mono text-volt">goodput</td>
                <td className="py-2.5 pr-4">只统计满足 SLO（如 TTFT&lt;2s 且 TPOT&lt;50ms）的请求的吞吐</td>
                <td className="py-2.5 text-ink2">两边的合同；超时的 token 不算数 —— 真正该优化的目标</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          注意吞吐和延迟在 ridge 之后是<strong>此消彼长</strong>的：batch 再加大，吞吐每多挤一点，
          全体用户的 TPOT 就一起变差一截 —— 「吞吐与延迟买一送一」的好事不存在。所以生产系统的调参
          从来不是「越大越好」，而是一道约束优化题：先和业务定下 SLO（service level objective，
          比如 P95 TTFT &lt; 2 s、TPOT &lt; 50 ms），然后把并发上限、chunked prefill 的块大小、
          抢占阈值这些旋钮拧到 <em>goodput</em> 最大的位置。同一套引擎，对话产品和离线批量摘要会拧出
          完全不同的参数 —— 前者拿吞吐换延迟，后者反过来。指标定义清楚了，第 4 节模拟器里那几个滑杆，
          就是你未来在生产里要拧的旋钮的等比例缩小版。
        </p>
      </Section>

      <Section index={7} title="总结与延伸阅读" lead="一章的内容，五句话带走。">
        <ul>
          <li>
            一个请求两段人生：<strong>prefill</strong> 是 compute-bound 的大矩阵乘（决定 TTFT），
            <strong>decode</strong> 是 memory-bound 的串行搬运（决定 TPOT），利用率差出一个数量级。
          </li>
          <li>
            <strong>batching</strong> 让一次权重读取服务多个请求的 token，吞吐在 ridge 之前近乎免费地线性上涨。
          </li>
          <li>
            静态批处理输给了输出长度的方差：<strong>木桶效应</strong>让先完成的槽位空转、新请求干等。
          </li>
          <li>
            <strong>连续批处理</strong>把调度粒度降到迭代级 —— 完成即让位、随到随补 ——
            是现代推理引擎共同的心脏。
          </li>
          <li>
            chunked prefill 削 TPOT 尖刺、抢占保显存不爆、speculative decoding 把串行 decode 变成并行验证；
            运营按 <strong>SLO</strong> 调参，优化的是 goodput 而不是裸吞吐。
          </li>
        </ul>
        <p>延伸阅读，按「先看哪个」排序：</p>
        <ul>
          <li>
            <a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">
              Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI '22)
            </a>{' '}
            —— 连续批处理（iteration-level scheduling）的源头论文，第 3、4 节的思想全部出自这里。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">
              Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM, SOSP '23)
            </a>{' '}
            —— 把虚拟内存的分页思想搬进 KV cache，连续批处理因此真正落地。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2401.09670" target="_blank" rel="noreferrer">
              DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (OSDI '24)
            </a>{' '}
            —— prefill/decode 分离 + goodput 优化，把「两段人生」的隐喻推到架构层面。
          </li>
          <li>
            <a href="https://github.com/sgl-project/sglang" target="_blank" rel="noreferrer">
              SGLang
            </a>{' '}
            —— 用 RadixAttention 做 prefix 复用的高性能引擎，读它的 scheduler 源码是最好的实战课。
          </li>
          <li>
            <a href="https://nvidia.github.io/TensorRT-LLM/" target="_blank" rel="noreferrer">
              NVIDIA TensorRT-LLM 文档
            </a>{' '}
            —— in-flight batching、chunked prefill、speculative decoding 在生产级实现里的参数与最佳实践。
          </li>
        </ul>
      </Section>
    </>
  )
}
