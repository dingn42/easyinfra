import { Callout, ChapterLink, HardwareBaseline, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { PrefillDecodeLab } from './PrefillDecodeLab'
import { BatchingCurves } from './BatchingCurves'
import { StaticBatchTimeline } from './StaticBatchTimeline'
import { ContinuousBatchingLab } from './ContinuousBatchingLab'
import { ChunkedPrefillSVG, PreemptionSVG, ServingStackSVG, SpeculativeSVG } from './StackDiagrams'

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            Open any chat product and the model feels like it is sitting there just for you: you hit send and
            it starts spelling out tokens one by one. In the datacenter, that H100 is probably juggling dozens of
            sessions right now. Your question, someone else&apos;s code completion, an agent&apos;s 17th tool
            call, all crammed onto the same chip taking turns. The illusion that every user owns the model
            outright is bought with <strong>scheduling</strong>, not with more cards. This chapter cracks open
            the inference engine to see what systems like vLLM and TensorRT-LLM are actually scheduling, and why
            a good scheduler can lift a single card&apos;s throughput by an order of magnitude.
          </>,
          <>
            打开任何一个大模型对话产品，你都会觉得模型在专心陪你：一发问，它马上开始逐字往外蹦。
            可机房里那张 H100 此刻多半同时挂着几十个会话。你的问题、别人的代码补全、某个 Agent
            的第 17 次工具调用，全挤在同一块芯片上轮流上场。让每个用户都以为自己独占了模型，靠的
            是<strong>调度</strong>，不是更多的卡。这一章我们拆开推理引擎（inference engine），看看
            vLLM、TensorRT-LLM 这类系统到底在调度什么，以及一个好调度器凭什么能把同一张卡的吞吐
            抬高一个数量级。
          </>,
        )}
      </p>

      <HardwareBaseline ids={['h100']} />

      <Section
        index={1}
        title={t('Two lives of one request', '一个请求的两段人生')}
        lead={t(
          'The very same request lives two completely different lives on the GPU.',
          '同一个请求，在 GPU 上活成了两种完全不同的样子。',
        )}
      >
        <p>
          {t(
            <>
              From the moment a request enters the engine until it emits its final character, it passes through
              two stages of wildly opposite temperament. The first is{' '}
              <Term t={t('prefill', 'prefill（预填充）')}>
                {t(
                  'Push the entire prompt through the model at once, computing the KV cache for every position and the first output token in parallel.',
                  '把整段 prompt 一次性送进模型，并行计算出所有位置的 KV cache 和第一个输出 token。',
                )}
              </Term>
              : the hundreds or thousands of tokens in the prompt are <em>already known</em>, so they pack into
              one big matrix and compute in a single forward pass. This is one big matmul. Arithmetic intensity
              climbs linearly with prompt length, the tensor cores stay well fed, and effective compute
              utilization can reach 60%–70%.
            </>,
            <>
              一个推理请求从进入引擎到吐完最后一个字，经历两个性格迥异的阶段。第一段叫{' '}
              <Term t={t('prefill', 'prefill（预填充）')}>
                把整段 prompt 一次性送进模型，并行计算出所有位置的 KV cache 和第一个输出 token。
              </Term>
              ：prompt 里的几百上千个 token 都是<em>已知的</em>，能拼成一个大矩阵、一次前向全部算完。
              这是一次实打实的大矩阵乘：算术强度（arithmetic intensity）随 prompt 长度线性上涨，张量核心
              被喂得很饱，有效算力利用率能到 60%~70%。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The second stage is{' '}
              <Term t={t('decode', 'decode（解码）')}>
                {t(
                  'The autoregressive generation phase: each forward pass handles exactly one new token, samples the next one, and feeds it back in.',
                  '自回归生成阶段：每次前向只处理一个新 token，算完采样出下一个，再喂回去。',
                )}
              </Term>
              , and the mood turns grim. Autoregressive generation is serial by nature: token N+1 can&apos;t be
              computed until token N is out. So each forward pass processes <em>one</em> token, yet not a single
              byte of the model&apos;s weights can be skipped. Recall the arithmetic from <ChapterLink n={9} />.
              A 70B model in FP16 is 140 GB of weights, the H100&apos;s HBM bandwidth is 3.35 TB/s, so streaming
              the weights once already takes about 42 ms. The real compute for that step is only about 140
              GFLOPs, a drop in the bucket against 989 TFLOPS of peak, and effective utilization falls below 1%.
              The same card flips from a fully loaded blast furnace into an almost-idle conveyor belt.
            </>,
            <>
              第二段叫{' '}
              <Term t={t('decode', 'decode（解码）')}>
                自回归生成阶段：每次前向只处理一个新 token，算完采样出下一个，再喂回去。
              </Term>
              ，画风急转直下。自回归（autoregressive）生成天生串行：第 N+1 个 token 必须等第 N 个出来才能算。
              于是每一步前向只处理<em>一个</em> token，可模型的全部权重一个字节都不能少读。翻回<ChapterLink n={9} />那笔账：
              70B 模型 FP16 权重 140 GB，H100 的 HBM 带宽 3.35 TB/s，光把权重过一遍就要约 42 ms。
              而这一步真正的计算量只有约 140 GFLOPs，对 989 TFLOPS 的峰值算力来说是九牛一毛，有效利用率
              掉到 1% 以下。同一张卡，从满载的炼钢炉变成了几乎空转的传送带。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              These two lives map onto the two latency metrics users feel most acutely. <strong>TTFT</strong>{' '}
              (Time To First Token) is almost entirely set by prefill: the longer the prompt and the longer the
              queue, the longer you stare at a blinking cursor. <strong>TPOT</strong> (Time Per Output Token) is
              set by decode, and it governs how fast the words spill out. The experiment below puts the two
              lives side by side. Watch where the two points land on the roofline to the right: prefill pressing
              against the compute ceiling, decode mired on the bandwidth slope.
            </>,
            <>
              这两段人生分别对应用户最敏感的两个延迟指标。<strong>TTFT</strong>（Time To First Token，首
              token 延迟）几乎完全由 prefill 决定：prompt 越长、排队越久，你盯着光标发呆的时间就越长。
              <strong>TPOT</strong>（Time Per Output Token，每 token 间隔）由 decode 决定，它管着字往外蹦
              的速度。下面这个实验把两段人生并排放在一起，注意右侧 roofline 上两个点的位置：prefill
              顶着算力屋顶，decode 深陷带宽斜坡。
            </>,
          )}
        </p>
        <PrefillDecodeLab />
        <Quiz
          question={t(
            'At batch=1, decode utilizes less than 1% of compute. What is the root cause?',
            'batch=1 时 decode 的算力利用率不到 1%，根本原因是？',
          )}
          options={[
            {
              text: t(
                'Autoregressive generation is serial, so the GPU has no time for instruction-level parallelism',
                '自回归生成是串行的，GPU 来不及做指令级并行',
              ),
              explain: t(
                'Serialness is the surface symptom, but a single step is not short of internal parallelism (even one token has to traverse every matmul). The real bottleneck is memory traffic.',
                '串行确实是表象，但单步内部 GPU 并不缺并行度（一个 token 也要过全部矩阵乘）。真正的瓶颈在访存。',
              ),
            },
            {
              text: t(
                'Decode is memory-bound: every generated token requires reading the full weights from HBM, and at batch=1 that read cost is amortized over nothing',
                'decode 是 memory-bound：每生成一个 token 都要把全部权重从 HBM 读一遍，batch=1 时这笔读取成本完全摊不薄',
              ),
              correct: true,
              explain: t(
                'Right. Per step the compute is ≈ 2P FLOPs while the weight read is ≈ 2P bytes, so arithmetic intensity ≈ 1 FLOP/byte — two orders of magnitude below the H100 ridge (≈ 295). All the time goes into hauling weights while compute sits idle. This is the starting point for every optimization that follows.',
                '对。每步计算量 ≈ 2P FLOPs，权重读取 ≈ 2P 字节，算术强度 ≈ 1 FLOP/Byte，离 H100 的 ridge（约 295）差两个数量级。时间全花在搬权重上，算力大块闲置 —— 这也是后面所有优化的出发点。',
              ),
            },
            {
              text: t(
                'The KV cache is too large and crowds out the compute units',
                'KV cache 太大，挤占了计算单元',
              ),
              explain: t(
                "The KV cache consumes memory capacity and some bandwidth, but it doesn't 'crowd out compute units.' At batch=1 the dominant traffic is the weights themselves.",
                'KV cache 占的是显存容量和一部分带宽，不会「挤占计算单元」。batch=1 时主要的搬运量是权重本身。',
              ),
            },
            {
              text: t(
                'Sampling can only run on the CPU, dragging down the overall pace',
                '采样（sampling）这一步只能在 CPU 上做，拖慢了整体节奏',
              ),
              explain: t(
                'Modern engines do sampling on the GPU with negligible overhead — not the main issue.',
                '现代引擎的采样都在 GPU 上完成，开销很小，不是主要矛盾。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={2}
        title={t('Why batching rescues decode', '为什么 batching 能救 decode')}
        lead={t(
          'Read the weights once, compute dozens of tokens — this is the first-principles economics of inference systems.',
          '权重只读一遍，token 算几十个 —— 这是推理系统第一性的省钱逻辑。',
        )}
      >
        <p>
          {t(
            <>
              Decode&apos;s disease is &quot;reads a lot, computes a little,&quot; so the cure is direct:{' '}
              <strong>make one weight read serve more compute</strong>. Pack the decode steps of 32 users into
              one batch. Each forward pass still streams the 140 GB of weights once, but that one pass now
              produces 32 tokens, so arithmetic intensity multiplies by 32 outright. As long as you haven&apos;t
              hit the compute ceiling, throughput rises <em>linearly</em> with batch size while each user&apos;s
              perceived TPOT barely budges: the time goes into reading weights anyway, so computing 31 extra
              tokens rides along for free. That is the rare near-free lunch in serving.
            </>,
            <>
              decode 的病根是「读得多、算得少」，药方因此很直接：<strong>让一次权重读取服务更多的计算</strong>。
              把 32 个用户的 decode 步拼成一批（batch），每步前向还是把 140 GB 权重读一遍，但这一遍现在
              产出 32 个 token，算术强度直接乘 32。只要还没碰到算力屋顶，吞吐随 batch 大小<em>线性</em>上涨，
              每个用户感受到的 TPOT 却几乎不变：反正时间都花在读权重上，顺手多算 31 个 token 等于白送。
              这是推理服务里少见的近乎免费的午餐。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The free lunch has a limit, of course. Grow the batch far enough and compute time catches up with
              memory time, which is exactly the ridge point from the <ChapterLink n={6} /> roofline. Cross it
              and decode flips from memory-bound to compute-bound: throughput stops climbing, per-step latency
              starts growing linearly with batch, and TPOT visibly degrades. The curves below plot both lines
              together. Switch hardware to watch the knee move: a higher-bandwidth card (H200) reaches the
              throughput ceiling at a smaller batch, so it saturates sooner. The Blackwell generation pushes
              bandwidth and low-precision throughput further still, but the shape of the trade-off never
              changes.
            </>,
            <>
              免费午餐当然有边界。batch 大到一定程度，计算时间追上了访存时间，这正是<ChapterLink n={6} /> roofline 的
              ridge point。越过它，decode 从 memory-bound 翻身成 compute-bound：吞吐不再涨，单步时延开始随
              batch 线性变长，TPOT 肉眼可见地变差。下面的曲线图把这两条线画在一起，切换硬件看拐点怎么动：
              带宽越大的卡（H200）用更小的 batch 就能撞到吞吐上限，饱和得更早。Blackwell 这一代把带宽和
              低精度算力又往前推了一截，但这条权衡曲线的形状始终不变。
            </>,
          )}
        </p>
        <BatchingCurves />
        <Callout type="insight" title={t('The first principle of inference systems', '推理系统的第一性原理')}>
          <p>
            {t(
              <>
                Trace every fancy scheduling trick in this chapter back to its root and you land on the same
                sentence: <strong>every decode step pays a fixed cost to &quot;read all the weights,&quot; and
                the scheduler&apos;s job is to make each payment serve as many tokens as possible</strong>.
                Batching amortizes that cost across users. Speculative decoding amortizes it across time, since
                one forward pass verifies several draft tokens. Continuous batching keeps the amortization from
                ever lapsing. Once that clicks, vLLM&apos;s source code is just engineering detail.
              </>,
              <>
                把这章所有花哨的调度技术倒推回去，根都是同一句话：<strong>decode 每一步都要付一次「读全部权重」的
                固定成本，调度器的活就是让每一次付费服务尽可能多的 token</strong>。batching 沿「用户数」摊薄它，
                speculative decoding 沿「时间」摊薄它（一次前向验证多个草稿 token），连续批处理则保证这种摊薄
                一刻都不掉线。想通这一点，vLLM 的源码就只剩工程细节了。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={3}
        title={t("Static batching's bucket effect", '静态 batching 的木桶效应')}
        lead={t(
          "Bundling requests into a batch is easy; the hard part is they refuse to finish together.",
          '把请求捆成一批容易，难的是它们不肯一起结束。',
        )}
      >
        <p>
          {t(
            <>
              Now that we know batching is the answer, the most naive implementation looks like this:
              accumulate a batch of requests (say 8), prefill the whole batch, decode the whole batch, and only
              admit the next batch once the current one is fully done. This is <strong>static batching</strong>,
              and it&apos;s how plenty of early in-house serving stacks were actually written. Its flaw hides in
              one unremarkable fact: <em>nobody knows how long a request will generate</em>. One user asks
              &quot;translate this word&quot; and is done in 16 tokens; another wants &quot;write a
              two-thousand-word summary&quot; and runs for hundreds of steps. Boarding and leaving the batch
              together means everyone gets dragged along until the longest one finishes.
            </>,
            <>
              知道了 batching 是答案，最朴素的实现长这样：攒一批请求（比如 8 个），整批做 prefill，整批
              decode，整批跑完才放下一批进来。这叫<strong>静态批处理（static batching）</strong>，也是早年
              很多自研服务的真实写法。它的问题藏在一个不起眼的事实里：<em>没人知道一个请求会生成多长</em>。
              有人问「翻译这个词」，16 个 token 就完事；有人要「写一篇两千字总结」，得跑上几百步。整批同进
              同出，意味着所有人都得陪最长的那个跑完全程。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Look at the timeline below. A request that finishes early can&apos;t hand its slot back; it just
              hangs there idling, the gray hatching in the figure. The GPU keeps paying the weight-read cost for
              slots that have no live work, and the effective batch shrinks as the batch runs. Worse is the hard
              boundary between batches: even if a new request shows up one second after the batch starts, it
              waits for the whole batch to finish before it can board, so average wait time blows up. The larger
              the variance in output length, the more vicious the bucket effect. No amount of careful coding
              fixes this; the &quot;batch&quot; granularity itself is the wrong unit.
            </>,
            <>
              看下面的时间线。先完成的请求并不能把槽位让出来，只能挂着空转（图里的灰色斜纹）：GPU 在为
              「已经没活干的槽位」白白付权重读取的钱，有效 batch 越跑越小。更糟的是批与批之间的硬边界：
              哪怕新请求在批次开始后 1 秒就到了，也得等整批结束才能上车，平均等待时间随之暴涨。输出长度
              方差越大，木桶效应越狠。这不是代码写得糙，是「批」这个粒度本身就选错了。
            </>,
          )}
        </p>
        <StaticBatchTimeline />
      </Section>

      <Section
        index={4}
        title={t('Continuous batching: yield your slot the moment you finish', '连续批处理：完成即让位')}
        lead={t(
          'Drop scheduling granularity from "a batch of requests" to "a single iteration" and everything changes.',
          '把调度粒度从「一批请求」降到「一个迭代」，一切都变了。',
        )}
      >
        <p>
          {t(
            <>
              The 2022 Orca paper poked through this window. Decode already iterates one step at a time, so why
              should scheduling work at the granularity of a whole batch? Move the decision point to{' '}
              <strong>between each iteration</strong>: after every step, whoever has finished leaves immediately
              and returns its slot, and if the queue holds a new request it boards right away, doing its prefill
              first and then joining the decode crowd. This is <strong>continuous batching (also called
              in-flight batching, or iteration-level scheduling)</strong>, the scheduling core of today&apos;s
              vLLM, TensorRT-LLM, and SGLang alike.
            </>,
            <>
              2022 年的 Orca 论文捅破了这层窗户纸。decode 本来就是一步一步迭代的，调度凭什么非得以「整批」
              为单位？把决策点搬到<strong>每个迭代（iteration）之间</strong>：每走完一步，谁生成完了就立刻
              离场、返还槽位；队列里有新请求就马上补位，先做它的 prefill，再汇入 decode 大部队。这就是
              <strong>连续批处理（continuous batching，也叫 in-flight batching 或 iteration-level
              scheduling）</strong>，今天 vLLM、TensorRT-LLM、SGLang 的调度核心无一例外。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The gain isn&apos;t a couple of percent. Nobody gets dragged along inside the batch anymore, so
              the GPU&apos;s effective batch stays glued to the concurrency limit, and a new request&apos;s wait
              shrinks from &quot;until the whole batch ends&quot; to &quot;until the next iteration.&quot; On
              real workloads with high output-length variance, throughput gains of 2–5x are common, and the
              improvement in tail latency (P95/P99) is more dramatic still. The simulator below replays the same
              fixed-seed request stream under both schedulers. Run STATIC first and watch the sawtooth collapses
              in the utilization bar; switch to CONTINUOUS and watch new requests fill empty slots on the fly,
              pinning utilization high. Then crank up arrival rate λ and widen the output range, and the gap
              gets uglier.
            </>,
            <>
              效果不是百分之几十的小修小补。批内不再有人陪跑，GPU 的有效 batch 始终贴着并发上限，新请求
              的等待也从「等整批结束」变成「等下一个迭代」。在输出长度方差大的真实负载下，吞吐提升常见
              2~5 倍，尾延迟（P95/P99）的改善更夸张。下面的模拟器用同一条固定种子的请求流回放两种调度：
              先让 STATIC 跑一遍，注意利用率条的锯齿状塌陷；切到 CONTINUOUS，看新请求怎么随时填进空槽、
              利用率怎么被钉在高位。再把到达速率 λ 调高、把输出区间拉大，差距会更难看。
            </>,
          )}
        </p>
        <ContinuousBatchingLab />
        <Quiz
          question={t(
            'What is the essential change continuous batching makes over static batching?',
            '连续批处理相对静态批处理，本质的改变是什么？',
          )}
          options={[
            {
              text: t(
                'It uses a larger batch size, so throughput is higher',
                '用了更大的 batch size，所以吞吐更高',
              ),
              explain: t(
                'The concurrency limit (slot count) can be identical in both schedules. Continuous batching wins on slot utilization, not slot count.',
                '并发上限（槽位数）两种调度可以完全一样。连续批处理赢在槽位的利用率，不在槽位的数量。',
              ),
            },
            {
              text: t(
                'It splits prefill and decode onto two different cards',
                '把 prefill 和 decode 分到了两张不同的卡上',
              ),
              explain: t(
                'That is prefill/decode disaggregation (e.g. DistServe), an orthogonal technique covered in Section 5.',
                '那是 prefill/decode 分离（disaggregation，如 DistServe），是另一项正交的技术，第 5 节会提到。',
              ),
            },
            {
              text: t(
                'It drops scheduling granularity from "batch of requests" to "iteration/token level": between every step, whoever finishes yields immediately, new requests board anytime, and the effective batch is always full',
                '把调度粒度从「请求批」降到「迭代/token 级」：每步之间谁完成谁立刻让位，新请求随时补位，有效 batch 永远是满的',
              ),
              correct: true,
              explain: t(
                'Right. Decode already beats time in iterations, so the scheduler has no reason to decide at a coarser granularity. Release-on-finish plus admit-on-arrival makes the bucket effect vanish.',
                '对。decode 本身就以迭代为节拍，调度器没有理由用比它更粗的粒度做决策。完成即释放、随到随补，木桶效应就消失了。',
              ),
            },
            {
              text: t(
                'It predicts each request’s output length and groups requests of similar length into one batch',
                '它会预测每个请求的输出长度，把长度相近的请求分到一批',
              ),
              explain: t(
                'Length-predicting schedulers do exist as research, but continuous batching needs no prediction at all: whoever finishes first leaves first, so the schedule adapts on its own.',
                '长度预测调度确实有人研究，但连续批处理压根不需要预测：谁先完成谁先走，调度自己就适应了。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={5}
        title={t('Advanced scheduling: squeeze out every millisecond', '进阶调度：把每一毫秒都抠出来')}
        lead={t(
          'Continuous batching solves the main conflict; the remaining thorns take three handy tools.',
          '连续批处理解决了主要矛盾，剩下的尖刺要靠三件趁手的工具。',
        )}
      >
        <p>
          {t(
            <>
              <strong>Chunked prefill.</strong> Continuous batching has a hidden side effect. When a new request
              jumps in to do its prefill, a 2000-token prompt is one big compute-bound chunk that stretches that
              iteration long, and users already in decode suddenly feel the words pause: a spike in TPOT. The
              fix is to slice the long prompt into chunks of a few hundred tokens and mix only one small chunk
              into each iteration, pairing it with decode tokens into a modestly sized hybrid batch. You pay
              with slightly longer TTFT and buy everyone a smooth TPOT, a textbook case of trading peak for
              variance.
            </>,
            <>
              <strong>chunked prefill（分块预填充）。</strong>连续批处理有个隐藏的副作用。新请求插队做 prefill
              时，一段 2000 token 的 prompt 是一大块 compute-bound 的计算，会把那个迭代撑得很长，正在 decode
              的老用户会突然觉得字停了一下，TPOT 冒出尖刺。解法是把长 prompt 切成几百 token 的 chunk，每个
              迭代只混入一小块，和 decode token 拼成一个不大不小的混合批。代价是 TTFT 稍微变长，换来所有人
              TPOT 的平稳，是典型的「拿峰值换方差」。
            </>,
          )}
        </p>
        <ChunkedPrefillSVG />
        <p>
          {t(
            <>
              <strong>Preemption &amp; swap/recompute.</strong> Continuous batching turns memory into a dynamic
              resource. Every running request drags a KV cache that grows fatter the longer it runs, and with
              enough requests memory can overflow. The scheduler needs a stop-loss: pick an unlucky victim
              (usually the latest arrival) and evict it. Either swap its KV wholesale out to host memory and
              pull it back when there&apos;s room, or discard it outright and recompute with one prefill when it
              re-queues. Swapping costs PCIe bandwidth, recompute costs compute, and vLLM weighs the choice by
              KV size. To the user this is just a latency wobble, far better than an OOM that takes down the
              whole engine.
            </>,
            <>
              <strong>抢占与换出（preemption &amp; swap/recompute）。</strong>连续批处理让显存变成了动态资源。
              每个在跑的请求都拖着一条越长越胖的 KV cache，请求一多显存就可能爆。调度器必须有止损手段：挑
              一个倒霉蛋（通常是最晚到的）请出去。要么把它的 KV 整体换出（swap）到主机内存，等有空再搬回来；
              要么干脆丢弃，重新排队时用一次 prefill 重算（recompute）。换出费 PCIe 带宽，重算费算力，vLLM
              会按 KV 大小权衡选择。对用户来说这只是一点延迟波动，比 OOM 崩掉整个引擎好得多。
            </>,
          )}
        </p>
        <PreemptionSVG />
        <p>
          {t(
            <>
              <strong>Speculative decoding.</strong> The earlier moves all amortize the weight read; this one
              goes after &quot;serial&quot; itself. Take a much smaller draft model and let it serially guess k
              tokens. It is small, so it guesses fast. Then let the big model verify those k positions in{' '}
              <em>one</em> parallel forward pass, in a way that is provably identical in distribution to the big
              model sampling them one at a time. Hit 3 of them and you&apos;ve produced 4 tokens (including the
              correction) in a single &quot;big-model time.&quot; You&apos;ve converted serial decode into
              parallel verification, doing decode&apos;s work in prefill&apos;s compute-bound mode. The more accurate
              the draft, the bigger the win. The cost is keeping an extra small model around, or, like
              EAGLE/Medusa, growing a draft head off the big model itself.
            </>,
            <>
              <strong>speculative decoding（投机解码）。</strong>前面几招都在摊薄权重读取，这一招直接冲着
              「串行」本身去。找一个小得多的草稿模型（draft model）先串行猜 k 个 token；它小，猜得快。
              然后让大模型<em>一次前向</em>并行验证这 k 个位置，数学上能做到与大模型逐个采样完全同分布。
              命中 3 个，就等于用一次「大模型时间」生产了 4 个 token（含修正）。这就把串行 decode 变成了
              并行验证，让 decode 的活跑在 prefill 那种 compute-bound 的模式里。草稿越准，收益越大。代价是多养一个小模型，或者像
              EAGLE/Medusa 那样在大模型头上长出草稿头。
            </>,
          )}
        </p>
        <SpeculativeSVG />
        <p>
          {t(
            <>
              Assemble these parts into a whole machine and a production-grade serving stack splits into roughly
              four layers. At the top the Router load-balances across multiple engine replicas and routes by
              prefix affinity, steering a session toward the replica that already cached its KV. The Scheduler
              is this chapter&apos;s star: continuous batching, chunked prefill, preemption. The Engine manages
              the paging and reuse of the KV cache, where <ChapterLink n={9} />&apos;s PagedAttention lives. And
              at the bottom sit the Kernels: FlashAttention, fused GEMM, the heroes of <ChapterLink n={8} />.
              More radical architectures like DistServe go further and split prefill and decode into separate
              GPU pools (disaggregation), letting the compute-bound and memory-bound lives each find their
              best-fit hardware ratio.
            </>,
            <>
              把这些零件装回整机，一个生产级推理栈大致分四层。最上面的 Router 在多个引擎副本间做负载均衡，
              按 prefix 亲和性路由，同一会话尽量打到已经缓存了它 KV 的副本。Scheduler 是本章的主角：连续
              批处理、chunked prefill、抢占。Engine 管 KV cache 的分页与复用，<ChapterLink n={9} />的 PagedAttention
              就活在这里。最底下是 Kernels：FlashAttention、融合 GEMM 这些<ChapterLink n={8} />的功臣。更激进的架构
              比如 DistServe，干脆把 prefill 和 decode 拆到不同的 GPU 池子里（disaggregation），让 compute-bound
              和 memory-bound 这两段人生各自找最合适的硬件配比。
            </>,
          )}
        </p>
        <ServingStackSVG />
      </Section>

      <Section
        index={6}
        title={t('Metrics and trade-offs: throughput and latency are not buy-one-get-one', '指标与权衡：吞吐和延迟没有买一送一')}
        lead={t(
          'Before you tune anything, get clear on what exactly you are optimizing.',
          '调参之前，先把你到底在优化什么说清楚。',
        )}
      >
        <p>
          {t(
            'The operational language of serving rests on four metrics, each speaking from a different vantage point:',
            '推理服务的运营语言由四个指标搭起来，每个都站在不同的立场上：',
          )}
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">
                  {t('Metric', '指标')}
                </th>
                <th className="py-2 pr-4 font-mono text-[11px] uppercase tracking-wider text-ink3">
                  {t('Definition', '定义')}
                </th>
                <th className="py-2 font-mono text-[11px] uppercase tracking-wider text-ink3">
                  {t('Who cares / set by what', '谁在乎 / 由什么决定')}
                </th>
              </tr>
            </thead>
            <tbody className="text-text">
              <tr className="border-b border-line">
                <td className="py-2.5 pr-4 font-mono text-cyan">TTFT</td>
                <td className="py-2.5 pr-4">
                  {t('From sending the request to receiving the first token', '从发出请求到收到第一个 token')}
                </td>
                <td className="py-2.5 text-ink2">
                  {t(
                    "User's sense of 'it started responding'; set by queue wait + prefill",
                    '用户的「开始响应」体感；由排队等待 + prefill 决定',
                  )}
                </td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2.5 pr-4 font-mono text-cyan">TPOT</td>
                <td className="py-2.5 pr-4">
                  {t('Average gap between adjacent output tokens', '相邻输出 token 的平均间隔')}
                </td>
                <td className="py-2.5 text-ink2">
                  {t(
                    'Fluency of words spilling out; set by per-step decode latency, worsens with batch',
                    '字往外蹦的流畅度；由 decode 单步时延决定，随 batch 变差',
                  )}
                </td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2.5 pr-4 font-mono text-volt">{t('Throughput', '吞吐')}</td>
                <td className="py-2.5 pr-4">
                  {t('Total tokens produced per second across the whole card', '整张卡每秒产出的 token 总数')}
                </td>
                <td className="py-2.5 text-ink2">
                  {t(
                    "The operator's bill; higher with larger batch, up to the ridge",
                    '运营者的账单；batch 越大越高，直到 ridge',
                  )}
                </td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 font-mono text-volt">goodput</td>
                <td className="py-2.5 pr-4">
                  {t(
                    <>Throughput counting only requests that meet the SLO (e.g. TTFT&lt;2s and TPOT&lt;50ms)</>,
                    <>只统计满足 SLO（如 TTFT&lt;2s 且 TPOT&lt;50ms）的请求的吞吐</>,
                  )}
                </td>
                <td className="py-2.5 text-ink2">
                  {t(
                    "The contract between both sides; timed-out tokens don't count — the goal you should actually optimize",
                    '两边的合同；超时的 token 不算数 —— 真正该优化的目标',
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          {t(
            <>
              Note that past the ridge, throughput and latency <strong>trade off against each other</strong>.
              Bump the batch up further and every extra sliver of throughput drags down all users&apos; TPOT a
              notch; there is no throughput-and-latency buy-one-get-one deal. So tuning a production system is
              never &quot;bigger is better&quot; but a constrained optimization problem. First agree on the SLO
              with the business (a service level objective like P95 TTFT &lt; 2 s, TPOT &lt; 50 ms), then turn
              the knobs (concurrency limit, chunked-prefill chunk size, preemption threshold) to wherever{' '}
              <em>goodput</em> is maximized. The same engine tuned for a chat product versus offline batch
              summarization lands on completely different parameters: the former trades throughput for latency,
              the latter does the reverse. Once the metrics are defined, those few sliders in the Section 4
              simulator are the scaled-down version of the knobs you&apos;ll be turning in production.
            </>,
            <>
              注意吞吐和延迟在 ridge 之后是<strong>此消彼长</strong>的。batch 再加大，吞吐每多挤一点，全体
              用户的 TPOT 就一起差一截，「吞吐与延迟买一送一」的好事并不存在。所以生产系统调参从来不是
              「越大越好」，而是一道约束优化题。先和业务定下 SLO（service level objective，比如 P95 TTFT
              &lt; 2 s、TPOT &lt; 50 ms），再把并发上限、chunked prefill 的块大小、抢占阈值这些旋钮，拧到
              <em>goodput</em> 最大的位置。同一套引擎，调对话产品和调离线批量摘要会拧出完全不同的参数：前者
              拿吞吐换延迟，后者反过来。指标定义清楚了，第 4 节模拟器里那几个滑杆，就是你将来在生产里要拧
              的旋钮的等比例缩小版。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={7}
        title={t('Summary and further reading', '总结与延伸阅读')}
        lead={t('A whole chapter, carried away in five sentences.', '一章的内容，五句话带走。')}
      >
        <ul>
          <li>
            {t(
              <>
                Two lives of one request: <strong>prefill</strong> is a compute-bound big matmul (sets TTFT),
                <strong>decode</strong> is memory-bound serial hauling (sets TPOT), and their utilization
                differs by an order of magnitude.
              </>,
              <>
                一个请求两段人生：<strong>prefill</strong> 是 compute-bound 的大矩阵乘（决定 TTFT），
                <strong>decode</strong> 是 memory-bound 的串行搬运（决定 TPOT），利用率差出一个数量级。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Batching</strong> makes one weight read serve many requests&apos; tokens, so throughput
                rises nearly for free and linearly before the ridge.
              </>,
              <>
                <strong>batching</strong> 让一次权重读取服务多个请求的 token，吞吐在 ridge 之前近乎免费地线性上涨。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Static batching loses to the variance in output length: the <strong>bucket effect</strong>
                leaves finished slots idling while new requests wait.
              </>,
              <>
                静态批处理输给了输出长度的方差：<strong>木桶效应</strong>让先完成的槽位空转、新请求干等。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>Continuous batching</strong> drops scheduling granularity to the iteration level (yield
                on finish, admit on arrival) and is the shared heart of every modern inference engine.
              </>,
              <>
                <strong>连续批处理</strong>把调度粒度降到迭代级（完成即让位、随到随补），是现代推理引擎
                共同的心脏。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Chunked prefill shaves TPOT spikes, preemption keeps memory from overflowing, speculative
                decoding turns serial decode into parallel verification; operators tune by <strong>SLO</strong>,
                optimizing goodput rather than raw throughput.
              </>,
              <>
                chunked prefill 削 TPOT 尖刺、抢占保显存不爆、speculative decoding 把串行 decode 变成并行验证；
                运营按 <strong>SLO</strong> 调参，优化的是 goodput 而不是裸吞吐。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Further reading, ordered by "read this first":', '延伸阅读，按「先看哪个」排序：')}</p>
        <ul>
          <li>
            <a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">
              Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI &apos;22)
            </a>{' '}
            {t(
              '— the origin paper for continuous batching (iteration-level scheduling); the ideas in Sections 3 and 4 all come from here.',
              '—— 连续批处理（iteration-level scheduling）的源头论文，第 3、4 节的思想全部出自这里。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">
              Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM, SOSP &apos;23)
            </a>{' '}
            {t(
              '— ports the paging idea from virtual memory into the KV cache, which is what made continuous batching truly practical.',
              '—— 把虚拟内存的分页思想搬进 KV cache，连续批处理因此真正落地。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2401.09670" target="_blank" rel="noreferrer">
              DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (OSDI &apos;24)
            </a>{' '}
            {t(
              '— prefill/decode disaggregation + goodput optimization, pushing the "two lives" metaphor up to the architecture level.',
              '—— prefill/decode 分离 + goodput 优化，把「两段人生」的隐喻推到架构层面。',
            )}
          </li>
          <li>
            <a href="https://github.com/sgl-project/sglang" target="_blank" rel="noreferrer">
              SGLang
            </a>{' '}
            {t(
              '— a high-performance engine doing prefix reuse via RadixAttention; reading its scheduler source is the best hands-on lesson.',
              '—— 用 RadixAttention 做 prefix 复用的高性能引擎，读它的 scheduler 源码是最好的实战课。',
            )}
          </li>
          <li>
            <a href="https://nvidia.github.io/TensorRT-LLM/" target="_blank" rel="noreferrer">
              {t('NVIDIA TensorRT-LLM documentation', 'NVIDIA TensorRT-LLM 文档')}
            </a>{' '}
            {t(
              '— the parameters and best practices for in-flight batching, chunked prefill, and speculative decoding in a production-grade implementation.',
              '—— in-flight batching、chunked prefill、speculative decoding 在生产级实现里的参数与最佳实践。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
