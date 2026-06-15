import { Callout, ChapterLink, HardwareBaseline, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { IntensityCalc } from './IntensityCalc'
import { RooflineChart } from './RooflineChart'
import { HARDWARE, ridgeOf } from './data'

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            "This kernel is too slow." Hand a beginner that ticket and they open the code and read line by line. Hand it
            to someone who's done this a hundred times and they ask one yes/no question first:{' '}
            <strong>is it starved for compute, or starved for data?</strong> Starved for compute means the math units
            can't keep up. Starved for data means the memory system can't feed them. Both diseases present the same
            symptom, everything is just "slow," yet the cures are opposite, and prescribing the wrong one buys you a week
            of tuning with nothing to show for it. The Roofline model squeezes that diagnosis into one log-log plot.
            The x-axis is a program's "constitution," the y-axis is how fast it can possibly run, and any kernel you pin
            onto it shows its bottleneck at a glance. This chapter builds the plot from scratch, then shows you how to pin
            your own kernels onto it with a real profiler.
          </>,
          <>
            「这个 kernel 太慢了」——接到这种 issue，新手会打开代码逐行找问题，干过几百次的人会先问一个判断题：
            它到底是<strong>算不动</strong>还是<strong>喂不饱</strong>？算不动是算力跟不上，喂不饱是数据供不上。两种病的症状
            一模一样，都是慢，药方却完全相反。开错了药，优化一星期，性能纹丝不动。Roofline 模型把这个判断压缩成一张 log-log
            图：横轴是程序的「体质」，纵轴是它能跑多快，任何 kernel 往图上一钉，瓶颈一眼可见。这一章先把图从零建起来，再讲真实
            世界里怎么用 profiler 把你的 kernel 钉上去。
          </>,
        )}
      </p>

      <HardwareBaseline
        ids={['a100', 'h100', 'rtx4090']}
        note={t(
          'Three cards compared; A100 is the default for worked examples and the ridge ~ 164.',
          '三卡对比；worked example 默认用 A100，ridge 约 164。',
        )}
      />

      <Section
        index={1}
        title={t('Arithmetic intensity: a kernel’s constitution', '算术强度：kernel 的体质指标')}
        lead={t(
          'One ratio fixes the program’s constitution, another fixes the machine’s temperament. Diagnosis is just comparing the two.',
          '一个除法定义程序的体质，另一个除法定义机器的脾气。诊断就是比较这两个数。',
        )}
      >
        <p>
          {t(
            <>
              The last two chapters kept circling the same fact: a modern GPU has <strong>compute to spare and bandwidth
              in short supply</strong>. Put numbers on it with the A100. BF16 Tensor Core peak is about 312 TFLOPS, HBM
              bandwidth about 1.9 TB/s (the 80 GB SXM tops out near 2 TB/s; we use 1.9 as our reference figure
              throughout). So it can do 312 trillion floating-point ops a second but pull only 1.9 trillion bytes a
              second out of DRAM. Divide the two: for every byte you bring in, the machine hands you roughly 164 ops
              worth of free capacity. Any of that capacity your program can't spend, the compute units sit there idle.
            </>,
            <>
              前两章反复出现过同一个事实：现代 GPU <strong>算力过剩、带宽稀缺</strong>。拿 A100 的数字说话：BF16 Tensor
              Core 峰值约 312 TFLOPS，HBM 带宽约 1.9 TB/s（80GB SXM 实际接近 2 TB/s，本课统一以 1.9 为基准）。也就是说，它
              每秒能做 312 万亿次浮点运算，却每秒只能从显存搬 1.9 万亿个字节。两个数相除：每搬进 1 个字节，机器就白送你约
              164 次运算的余量。这些余量你的程序用不掉，算力就闲在那里围观。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              That motivates the definition of <strong>arithmetic intensity (AI)</strong>: for every byte a kernel moves
              from DRAM, how many floating-point ops does it do on average.
            </>,
            <>
              于是自然定义出<strong>算术强度（arithmetic intensity，AI）</strong>：一个 kernel 每从 DRAM
              搬运 1 字节数据，平均做多少次浮点运算。
            </>,
          )}
        </p>
        <MathTex block tex="\mathrm{AI} \;=\; \frac{\text{FLOPs}}{\text{DRAM Bytes}} \qquad [\text{FLOP/B}]" />
        <p>
          {t(
            <>
              Why does this one ratio decide performance? Because a kernel's runtime has two independent lower bounds:
              doing all the FLOPs takes at least FLOPs / peak-compute, and moving all the bytes takes at least
              Bytes / bandwidth. On a GPU compute and data movement overlap heavily, so total time is roughly the larger
              of the two:
            </>,
            <>
              为什么这个比值能决定性能？因为 kernel 的执行时间有两个互相独立的下限：算完所有 FLOPs 至少要
              FLOPs/峰值算力 这么久，搬完所有字节至少要 Bytes/带宽 这么久。计算和搬运在 GPU
              上高度重叠，所以总时间约等于两者中较大的那个：
            </>,
          )}
        </p>
        <MathTex
          block
          tex="t \;\ge\; \max\!\left(\frac{\text{FLOPs}}{P_{\text{peak}}},\; \frac{\text{Bytes}}{BW}\right) \;\;\Longrightarrow\;\; P_{\text{attain}} \;=\; \min\big(\mathrm{AI}\times BW,\;\; P_{\text{peak}}\big)"
        />
        <p>
          {t(
            <>
              That <code>min</code> on the right is all the math the Roofline contains. Performance is either
              bandwidth-limited (AI × BW) or compute-limited (P<sub>peak</sub>), whichever is smaller. The point where the
              two limits cross is the{' '}
              <Term
                t={
                  <>
                    machine balance point (<em>ridge point</em>)
                  </>
                }
              >
                ridge = peak compute ÷ bandwidth. A program whose AI lands exactly on the ridge "just barely" saturates
                compute and bandwidth at the same time. It's where the slanted line meets the flat plateau on the
                Roofline, and the dividing line between memory-bound and compute-bound.
              </Term>
              :
            </>,
            <>
              右边这个 min 就是 Roofline 的全部数学。性能要么被带宽限制（AI × BW），要么被算力限制（P
              <sub>peak</sub>），取较小者。两条限制交汇的位置叫
              <Term t="机器平衡点（ridge point）">
                ridge = 峰值算力 ÷ 带宽。AI 恰好等于 ridge 的程序「正好」同时吃满算力与带宽；它是 Roofline
                图上斜线与平台的交点，也是 memory-bound 与 compute-bound 的分界线。
              </Term>
              ：
            </>,
          )}
        </p>
        <MathTex
          block
          tex="\mathrm{ridge} \;=\; \frac{P_{\text{peak}}}{BW} \;=\; \frac{312\ \text{TFLOPS}}{1.9\ \text{TB/s}} \;\approx\; 164\ \text{FLOP/B}"
        />
        <p>
          {t(
            <>
              AI below the ridge means time goes mostly to moving data, which is{' '}
              <strong className="text-amber">memory-bound</strong>. Above the ridge, time goes mostly to compute, which is{' '}
              <strong>compute-bound</strong>. Notice how the two numbers divide the work. AI belongs to the{' '}
              <strong>program</strong>, fixed by the algorithm and its implementation, the same on whatever card you run.
              The ridge belongs to the <strong>machine</strong> and owes nothing to your code. Diagnosing a bottleneck is
              just laying these two numbers side by side. Here are the ridges of a few common cards:
            </>,
            <>
              AI 低于 ridge，时间主要花在搬数据上，叫 <strong className="text-amber">memory-bound</strong>（内存受限）；
              高于 ridge，时间主要花在计算上，叫 <strong>compute-bound</strong>（算力受限）。注意这两个数怎么分工：AI
              属于<strong>程序</strong>，由算法和实现决定，跑在哪块卡上都一样；ridge 属于<strong>机器</strong>，跟你的代码无关。
              诊断瓶颈，就是把这两个数拿出来比大小。几块常见卡的 ridge 如下：
            </>,
          )}
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="microlabel py-2 pr-4 font-normal">GPU</th>
                <th className="microlabel py-2 pr-4 font-normal">{t('Bandwidth', '带宽')}</th>
                <th className="microlabel py-2 pr-4 font-normal">BF16 Tensor</th>
                <th className="microlabel py-2 pr-4 font-normal">ridge · BF16</th>
                <th className="microlabel py-2 pr-4 font-normal">FP32 CUDA</th>
                <th className="microlabel py-2 font-normal">ridge · FP32</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {HARDWARE.map((h) => (
                <tr key={h.id} className="border-b border-line text-ink2">
                  <td className="py-2 pr-4 text-ink">{h.name}</td>
                  <td className="py-2 pr-4">
                    {h.bw} <span className="text-ink3">TB/s</span>
                  </td>
                  <td className="py-2 pr-4">
                    {h.tensor} <span className="text-ink3">TFLOPS</span>
                  </td>
                  <td className="py-2 pr-4 text-volt">≈ {ridgeOf(h.tensor, h.bw).toFixed(0)}</td>
                  <td className="py-2 pr-4">
                    {h.fp32} <span className="text-ink3">TFLOPS</span>
                  </td>
                  <td className="py-2 text-cyan">≈ {ridgeOf(h.fp32, h.bw).toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          {t(
            <>
              See the pattern? On the same A100, the FP32 CUDA-Core ridge is only about 10, so most programs are
              "qualified" to saturate FP32 compute. Switch to the BF16 Tensor Core and the ridge jumps to 164, and the
              programs that clear that bar become a precious few.{' '}
              <strong>Tensor Cores lifted compute 16×, bandwidth barely moved, and "starved for data" became the default
              state.</strong>
            </>,
            <>
              看出门道了吗：同一块 A100，用 FP32 CUDA Core 时 ridge 只有约 10，大多数程序都「够格」吃满 FP32 算力；换到
              BF16 Tensor Core，ridge 跳到 164，能跨过这道门槛的程序就屈指可数了。
              <strong>Tensor Core 把算力抬高 16 倍，带宽几乎没动，于是「喂不饱」成了常态。</strong>
            </>,
          )}
        </p>
        <Callout type="note" title={t('Counting the bytes in the denominator', '分母里的字节怎么数')}>
          <p>
            {t(
              <>
                AI's denominator is the <strong>traffic that actually crosses DRAM</strong>, not the number of memory
                accesses written in the code. Anything served from cache doesn't count. When you do the math by hand you
                usually assume the ideal case, each datum crossing DRAM exactly once, which gives an upper bound on AI.
                With imperfect caching, real AI only drops and the kernel skews even more memory-bound. The calculator in
                the next section uses exactly this ideal counting.
              </>,
              <>
                AI 的分母是<strong>真正走 DRAM 的流量</strong>，不是代码里写的访存次数；缓存里复用掉的不算。手算时通常取理想
                情况，每个数据只从 DRAM 过一次，得到的是 AI 的上界。实际缓存不完美时 AI 只会更低，kernel 更偏 memory-bound。
                下一节的计算器用的就是这种理想计法。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title={t('Counting FLOPs and bytes by hand', '动手数 FLOPs 和字节')}
        lead={t(
          'Once the numerator and denominator are right, AI is grade-school division. The hard part is knowing what to count.',
          '分子分母数对了，AI 就是个小学除法。难的是知道该数什么。',
        )}
      >
        <p>
          {t(
            <>
              Take the case that matters most, GEMM (general matrix multiply, M×K times K×N). Each output element does K
              multiply-adds, for 2MNK FLOPs total. In the ideal case the three matrices each cross memory once, so the
              byte count is (MN + MK + KN)·b, where b is the bytes per element (2 for BF16). Tidy it up and you get a form
              with some character to it:
            </>,
            <>
              拿最要紧的 GEMM（通用矩阵乘，M×K 乘 K×N）来说：每个输出元素要做 K 次乘加，共 2MNK 个 FLOPs；理想情况下三个
              矩阵各从显存过一遍，字节数是 (MN + MK + KN)·b，其中 b 是每个元素的字节数（BF16 为 2）。整理一下，会得到一个挺
              有意思的形式：
            </>,
          )}
        </p>
        <MathTex
          block
          tex="\mathrm{AI}_{\text{GEMM}} \;=\; \frac{2MNK}{(MN+MK+KN)\,b} \;=\; \frac{2}{b}\cdot\frac{1}{\frac{1}{M}+\frac{1}{N}+\frac{1}{K}}"
        />
        <p>
          {t(
            <>
              The right side is the harmonic mean of M, N, K, the same shape as resistors in parallel, where{' '}
              <strong>the smallest dimension calls the shots</strong>. For a square M=N=K=n, AI = 2n/(3b) (≈ n/3 in
              BF16), growing linearly with size: the bigger the matrix, the more times each element is reused, the better
              its constitution. That's the mathematical root of "matrices have to be big enough to saturate compute." It
              cuts the other way too. Even with M and N both at 8192, the moment K=16 the short edge pins AI to the floor.
              Verify it yourself with the calculator below:
            </>,
            <>
              右边是 M、N、K 的调和平均形式，和并联电阻一个形状，<strong>最小的那个维度说了算</strong>。方阵 M=N=K=n 时 AI =
              2n/(3b)（BF16 下约 n/3），随尺寸线性增长：矩阵越大，每个元素被复用的次数越多，体质越好。这就是「矩阵要够大才能
              吃满算力」的数学根源。反过来也成立：哪怕 M 和 N 都是 8192，只要 K=16，AI 就被这条短板按在地上。用下面的计算器
              亲手验证：
            </>,
          )}
        </p>

        <IntensityCalc />

        <p>
          {t(
            <>
              Run through all five operations and a brutal taxonomy emerges.{' '}
              <strong>Vector add, SAXPY, and GEMV have constant AI</strong>: numerator and denominator both scale with the
              data size, so scaling up never changes the constitution, and they're doomed to be memory-bound. Attention
              score's AI is set jointly by sequence length and head dimension. Only GEMM can keep improving its
              constitution by growing larger. That's why "batch small ops into one big matmul" is the universal move of
              GPU performance work, from cuBLAS batched GEMM to the continuous batching in LLM serving. Same trick
              underneath.
            </>,
            <>
              把五种操作切一遍，你会看到一个残酷的分类：<strong>向量加、SAXPY、GEMV 的 AI 是常数</strong>，分子分母都正比于
              数据量，规模再大体质也不变，注定 memory-bound；attention score 的 AI 由序列长度和 head 维度共同决定；只有 GEMM
              能靠加大尺寸持续改善体质。这就是为什么「把小操作攒成大矩阵乘」是 GPU 性能优化里的万能套路，从 cuBLAS 的 batched
              GEMM 到 LLM serving 的 continuous batching，底下都是这一招。
            </>,
          )}
        </p>
      </Section>

      <Quiz
        question={t(
          'Vector add c = a + b (FP16): each element does 1 addition and touches 6 bytes of memory. What is its arithmetic intensity and bottleneck? (For reference: A100 BF16 ridge ≈ 164, FP32 ridge ≈ 10.)',
          ' 向量加 c = a + b（FP16）：每个元素做 1 次加法、访存 6 字节。它的算术强度和瓶颈是？（参考：A100 BF16 ridge ≈ 164，FP32 ridge ≈ 10）',
        )}
        options={[
          {
            text: t(
              'AI is on the order of 0.25 or below (well under 1) — squarely memory-bound',
              'AI 在 0.25 以下的量级（远小于 1），妥妥的 memory-bound',
            ),
            correct: true,
            explain: t(
              'In FP16 that’s 1 FLOP / 6 B ≈ 0.17; in FP32, 1/12 ≈ 0.08. Whatever the precision or counting convention, it stays between 0.1 and 0.3, two to three orders of magnitude below any ridge. Run this kernel and the GPU spends nearly all its time moving data while the add units watch.',
              'FP16 下 1 FLOP / 6 B ≈ 0.17，FP32 下 1/12 ≈ 0.08。不管精度和计法怎么变，都在 0.1~0.3 之间，离任何一条 ridge 都差两三个数量级。跑这种 kernel 时 GPU 绝大部分时间在搬数据，加法单元基本在围观。',
            ),
          },
          {
            text: t(
              'AI ≈ 6 (6 bytes per FLOP) — already close to the FP32 ridge, so compute-bound',
              'AI ≈ 6（每 FLOP 摊 6 字节），已经接近 FP32 的 ridge，算 compute-bound',
            ),
            explain: t(
              'Numerator and denominator are flipped: AI is "FLOPs per byte," not "bytes per FLOP." With 1 FLOP paired to 6 bytes, AI is 1/6 ≈ 0.17.',
              '分子分母拿反了：AI 是「每字节多少 FLOP」，不是「每 FLOP 多少字节」。1 FLOP 配 6 字节，AI 是 1/6 ≈ 0.17。',
            ),
          },
          {
            text: t(
              'AI rises with N — a long enough vector turns it compute-bound',
              'AI 随 N 增大而提高，向量够长就能变成 compute-bound',
            ),
            explain: t(
              'Both FLOPs and bytes scale with N, so N cancels in the ratio. An elementwise op’s constitution is constant, and scaling up can’t save it. That’s exactly what separates it from GEMM.',
              'FLOPs 和字节数都正比于 N，相除之后 N 被约掉。elementwise 操作的体质是常数，加大规模救不了它，这正是它和 GEMM 的本质区别。',
            ),
          },
          {
            text: t(
              'It depends on occupancy — AI alone can’t tell you',
              '取决于 occupancy，光看 AI 无法判断',
            ),
            explain: t(
              'Occupancy affects "how far below the roof you are" (utilization), not "which region you’re in." Comparing AI against the ridge is decided purely by the algorithm and the hardware specs.',
              'occupancy 影响的是「离屋顶多远」（利用率），不改变「在哪个区」。AI 与 ridge 的比较只由算法和硬件规格决定。',
            ),
          },
        ]}
      />

      <Section
        index={3}
        title={t('Roofline: drawing the diagnosis as a plot', 'Roofline：把判断画成一张图')}
        lead={t(
          'x-axis is constitution (AI), y-axis is attainable performance, two roofs overhead. No kernel can hide on this plot.',
          'x 轴是体质（AI），y 轴是可达性能，两道屋顶罩在头上。所有 kernel 在这张图上都无所遁形。',
        )}
      >
        <p>
          {t(
            <>
              Plot the <code>min</code> formula from the last section: AI on the x-axis (log scale), attainable
              performance on the y-axis (log scale). The left half is a slanted line, performance = AI × bandwidth, where
              every bump in a byte's "value" lifts performance proportionally. Past the ridge, compute caps it and the
              line flattens into a horizontal plateau. Slope plus plateau looks like a roof, which is how the model got
              its name. A kernel is a point on this plot: its x-coordinate is set by the algorithm, its y-coordinate is
              measured performance.{' '}
              <strong>The point always sits below the roof, and the vertical gap from point to roof is the headroom you
              haven't cashed in yet.</strong>
            </>,
            <>
              把上一节的 min 公式画出来：横轴取 AI（对数刻度），纵轴取可达性能（对数刻度）。左半边是一条斜线，性能 = AI ×
              带宽，每个字节的「含金量」每涨一分，性能成比例涨一分。过了 ridge 被算力封顶，斜线变成一条水平的平台。斜线加平台
              像一面屋顶，模型因此得名 Roofline。一个 kernel 在图上是一个点：横坐标由算法决定，纵坐标是实测性能。
              <strong>点永远在屋顶之下，点到屋顶的垂直距离就是还没兑现的优化空间。</strong>
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The plot below is playable. Switch between the three cards to watch the roof deform, hover the preset
              kernel points to see why they land where they do, then drag "your kernel" around to feel what different
              positions imply. To the left is an abyss, to the right a plateau.
            </>,
            <>
              下面这张图可以玩：切换三块卡看屋顶怎么变形，悬停预置的 kernel 点看它们为什么落在那里，再拖动「你的 kernel」
              感受不同位置的处境。往左是深渊，往右是平原。
            </>,
          )}
        </p>

        <RooflineChart />

        <p>
          {t(
            <>
              Two things here reward a second look. First, four of the five preset points cluster to the left of the
              ridge. Among everyday kernels, compute-bound is the rare species and "starved for data" is the norm. Second,
              switch to the H100: bandwidth grew about 1.76× over the A100 while compute grew about 3.2×, pushing the
              ridge from 164 right to about 295.{' '}
              <strong>The newer the card, the more surplus compute it carries, and the fewer programs can saturate it.</strong>{' '}
              The Blackwell generation (B200/GB200) only widens the gap, lifting low-precision throughput faster than
              bandwidth again. Hardware has already spelled out where this goes: future performance engineering is, in all
              likelihood, a fight against bandwidth.
            </>,
            <>
              有两件事值得在图上多看两眼。第一，预置的五个点里四个挤在 ridge 左侧：日常 kernel 里 compute-bound 反而是稀有
              物种，「喂不饱」才是常态。第二，切到 H100 看看：相比 A100 带宽涨了约 1.76 倍，算力却涨了约 3.2 倍，ridge 从 164
              右移到约 295。<strong>卡越新，算力越过剩，能吃满算力的程序越少。</strong>到了 Blackwell 这一代（B200/GB200），
              低精度吞吐又一次跑在带宽前面，差距只会更大。硬件的演进方向已经把话挑明了：未来的性能工程，大概率是和带宽搏斗的工程。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={4}
        title={t('Two optimization paths: right, or up', '两条优化路线：向右，或向上')}
        lead={t(
          'Your spot on the Roofline isn’t fate, it’s a coordinate. Knowing where you are tells you which way to push.',
          'Roofline 上的位置不是命运，是坐标。知道自己在哪，才知道该往哪走。',
        )}
      >
        <p>
          {t(
            <>
              When the point hugs the <strong className="text-amber">bandwidth slope</strong> (memory-bound and near the
              roof), cranking the clock, piling on SMs, or swapping in a card with more compute all do nothing. That
              stretch of roof equals AI × BW and has nothing to do with peak compute. Two roads go faster. One is to{' '}
              <strong>move right (raise AI)</strong>: (1)<strong> kernel fusion</strong>, collapsing a chain like add →
              mul → GELU into one kernel so intermediate results stay in registers and never hit DRAM, slashing the DRAM
              byte count by two-thirds; (2) <strong>tiling for reuse</strong>, last chapter's GEMM shared-memory blocking,
              which at heart makes each byte from DRAM serve more ops; (3) <strong>quantization</strong>, FP16 to
              INT8/INT4, halving and halving again the bytes for the same data, doubling AI in place. The other road is to{' '}
              <strong>lift the roof itself</strong>: swap in a card with faster HBM, or move hot data into a faster
              storage tier (L2, shared memory), which effectively draws a steeper slope against a nearer "DRAM."
            </>,
            <>
              点贴着<strong className="text-amber">带宽斜线</strong>（memory-bound 且接近屋顶）时，提频率、堆 SM、换算力更
              强的卡统统无效，这一段屋顶等于 AI × BW，跟峰值算力一个字都不沾。提速只有两条路。一条是<strong>向右移（提高 AI）
              </strong>：① <strong>kernel 融合（fusion）</strong>，把 add → mul → GELU 这类链条合成一个 kernel，中间结果留在
              寄存器里不落显存，DRAM 字节数直接砍掉三分之二；② <strong>tiling 复用</strong>，上一章 GEMM 的共享内存分块，本质
              就是让每个从 DRAM 来的字节被运算更多次；③ <strong>量化（quantization）</strong>，FP16 换 INT8/INT4，同样的数据
              字节数减半再减半，AI 原地翻倍。另一条是<strong>抬高屋顶本身</strong>：换 HBM 更快的卡，或把热数据挪进更快的存储
              层级（L2、共享内存），相当于对更近的「DRAM」画一条更陡的斜线。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              When the point hugs the <strong className="text-volt">compute plateau</strong> (compute-bound) the
              prescription flips. Bandwidth is no longer the problem; the game is <strong>utilization</strong>. Confirm
              the matmul is actually going through Tensor Cores (data layout, alignment, and precision all correct), raise
              occupancy so the SMs always have work, and kill warp stalls. If the point hugs neither roof but hangs in
              mid-air, that's a more elementary problem: uncoalesced accesses, branch divergence, or kernels so tiny that
              launch overhead dominates. Fix utilization first, then talk direction.
            </>,
            <>
              点贴着<strong className="text-volt">算力平台</strong>（compute-bound）时药方反过来：带宽已经不是问题，拼的是
              <strong>利用率</strong>。确认矩阵乘真的走了 Tensor Core（数据布局、对齐、精度都对），提高占用率让 SM 始终有活干，
              消除 warp 停顿。如果点哪条屋顶都不贴、悬在半空，那是更初级的问题：访存不合并、分支发散、kernel 太碎导致启动开销占
              大头。先把利用率做上去，再谈方向。
            </>,
          )}
        </p>
        <Callout
          type="insight"
          title={t('LLM inference’s destiny was written into this plot long ago', 'LLM 推理的宿命，这张图早就写好了')}
        >
          <p>
            {t(
              <>
                When an LLM decodes token by token, every matmul at batch=1 degenerates into a GEMV, and each step also
                has to sweep the entire KV cache. That pins AI at 1–2 FLOP/B, two orders of magnitude short of the ridge,{' '}
                <strong>doomed to be memory-bound</strong>. This isn't a bad implementation; it's the shape of the
                algorithm. So the next several chapters are really about sliding this one point across the plot: KV cache
                optimization (<ChapterLink n={9} label="CH09" />) moves fewer bytes per step, batching (
                <ChapterLink n={10} label="CH10" />) reassembles GEMVs back into a GEMM to move right, and quantization (
                <ChapterLink n={11} label="CH11" />) shrinks each weight's byte count, also moving right. Once you can read
                this plot, you already know the shape of the answers in the next three chapters.
              </>,
              <>
                LLM 逐 token 解码（decode）时，batch=1 的矩阵乘全部退化成 GEMV，再加上每步都要扫一遍 KV cache。这就把 AI
                钉死在 1~2 FLOP/B，离 ridge 差两个数量级，<strong>注定 memory-bound</strong>。这不是实现不好，是算法形状决定的。
                所以接下来几章其实都是在这张图上挪点：KV cache 优化（<ChapterLink n={9} label="CH09" />）让每步少搬字节，batching（
                <ChapterLink n={10} label="CH10" />）把 GEMV 重新攒成 GEMM 向右移，量化（<ChapterLink n={11} label="CH11" />
                ）把每个权重的字节数压小，同样是向右移。读懂了这张图，后面三章你已经知道答案的形状了。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Quiz
        question={t(
          'An elementwise chain add → mul → GELU is built from three separate kernels, each reading from and writing back to HBM. The profiler shows DRAM bandwidth utilization around 90%. Which move is most effective?',
          '一条 elementwise 链 add → mul → GELU 由三个独立 kernel 组成，每步都从 HBM 读入再写回，profiler 显示 DRAM 带宽利用率约 90%。哪个手段最有效？',
        )}
        options={[
          {
            text: t(
              'Fuse the three kernels into one, keeping intermediates in registers',
              '把三个 kernel 融合成一个，中间结果留在寄存器里',
            ),
            correct: true,
            explain: t(
              'Three "read + write back" round-trips become one, cutting DRAM bytes to about 1/3 and tripling AI, so the whole point shifts right on the Roofline. This is the core payoff of fusing elementwise chains in torch.compile and Triton.',
              '三次「读入 + 写回」变成一次，DRAM 字节数约砍到 1/3，AI 翻三倍，点在 Roofline 上整体右移，这是 memory-bound 的第一药方。也是 torch.compile、Triton 融合 elementwise 链的核心收益来源。',
            ),
          },
          {
            text: t(
              'Raise occupancy, packing more warps onto the SMs',
              '提高 occupancy，往 SM 上多塞一些 warp',
            ),
            explain: t(
              'Bandwidth is already at 90%; the bottleneck is HBM throughput, not latency hiding, so more warps just lengthen the queue. Occupancy is the cure when the point is "far from the roof," not when it’s "hugging the slope."',
              '带宽已经吃到 90%，瓶颈在 HBM 吞吐不在延迟隐藏，再多 warp 也只是把队排得更长。occupancy 是「点离屋顶远」时的药，不是「贴着斜线」时的药。',
            ),
          },
          {
            text: t(
              'Rewrite it with Tensor Core instructions',
              '改写成 Tensor Core 指令',
            ),
            explain: t(
              'Elementwise has no matmul structure, so Tensor Cores can’t apply; and the bottleneck is bandwidth anyway, where no amount of compute pushes the roof in the slope region.',
              'elementwise 没有矩阵乘结构，Tensor Core 根本用不上；更何况瓶颈在带宽，算力侧再强也推不动斜线区的屋顶。',
            ),
          },
          {
            text: t(
              'Swap in a GPU with double the compute and the same bandwidth',
              '换一块算力翻倍、带宽不变的 GPU',
            ),
            explain: t(
              'The roof in the slope region = AI × BW, independent of peak compute. This is the priciest of the four options and the only one that does nothing at all.',
              '斜线区的屋顶 = AI × BW，与峰值算力无关。这是四个选项里最贵、也是唯一完全无效的方案。',
            ),
          },
        ]}
      />

      <Section
        index={5}
        title={t('Profiler mindset: pin your kernel onto the plot', 'Profiler 思维：把你的 kernel 钉到图上')}
        lead={t(
          'The Roofline is the map, the profiler is the GPS. Without measured data, even the best map is just decoration.',
          'Roofline 是地图，profiler 是 GPS。没有实测数据，再好的地图也只是装饰。',
        )}
      >
        <p>
          {t(
            <>
              NVIDIA's performance tools split the work cleanly, and remembering the order saves you half the detours:{' '}
              <strong>Nsight Systems first, then Nsight Compute</strong>. Nsight Systems gives you the whole program's
              timeline: what the CPU is doing, when each kernel launches, whether memory copies overlap with compute. A
              lot of "the GPU is slow" cases crack at this layer. The GPU is actually idle, time is lost in the gaps
              between kernels (CPU submitting too slowly, needless synchronization), or H2D/D2H copies sit serially in the
              middle of compute. For that disease, optimizing the kernel itself is barking up the wrong tree.
            </>,
            <>
              NVIDIA 的性能工具分工明确，记住这个顺序能少走一半弯路：<strong>先 Nsight Systems，后 Nsight Compute</strong>。
              Nsight Systems 给你整个程序的时间线（timeline）：CPU 在干什么、每个 kernel 什么时候启动、显存拷贝和计算有没有
              重叠。很多「GPU 慢」的案子在这一层就破了：GPU 实际在空转，时间花在 kernel 之间的间隙里（CPU 提交太慢、不必要的
              同步），或者 H2D/D2H 拷贝串行地横在计算中间。这种病去优化 kernel 本身是南辕北辙。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Only once you've confirmed the time really is being spent inside some kernel does Nsight Compute take over,
              the microscope for a single kernel. It replays the target kernel dozens of times, samples every hardware
              counter, and even ships with a built-in Roofline analysis page that computes your AI and measured FLOPS
              automatically and plots the point under the roof. Even without a GPU on hand right now, it's worth burning
              the reading of these few metrics into your brain. They're the common language of every performance report:
            </>,
            <>
              确认时间确实花在某个 kernel 内部之后，才轮到 Nsight Compute，单 kernel 的显微镜。它会重放（replay）目标 kernel
              几十次，采出每个硬件计数器，还直接内置了 Roofline 分析页，自动算出你的 AI 和实测 FLOPS，把点画在屋顶下。哪怕
              手头暂时没有 GPU 环境，下面这几个指标的读法也值得先刻进脑子，它们是所有性能报告的通用语言：
            </>,
          )}
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="microlabel py-2 pr-4 font-normal">{t('Metric', '指标')}</th>
                <th className="microlabel py-2 pr-4 font-normal">{t('Meaning', '含义')}</th>
                <th className="microlabel py-2 font-normal">{t('How to read it', '怎么读')}</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-cyan">DRAM Throughput %</td>
                <td className="py-2.5 pr-4 text-ink2">
                  {t('Measured memory bandwidth as a fraction of peak', '实测显存带宽占峰值的比例')}
                </td>
                <td className="py-2.5 text-ink2">
                  {t(
                    '>80% means you’re hugging the bandwidth slope; prescribe for memory-bound (fuse / reuse / quantize)',
                    '>80% 说明贴着带宽斜线，按 memory-bound 开药（融合 / 复用 / 量化）',
                  )}
                </td>
              </tr>
              <tr className="border-b border-line align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-volt">SM Busy / Compute %</td>
                <td className="py-2.5 pr-4 text-ink2">{t('How busy the arithmetic pipelines are', '算术管线的忙碌程度')}</td>
                <td className="py-2.5 text-ink2">
                  {t(
                    'Read it against DRAM%: whichever nears 100% is the main bottleneck; both low = utilization problem',
                    '和 DRAM% 对照着读：谁接近 100% 谁是主瓶颈；两个都低 = 利用率问题',
                  )}
                </td>
              </tr>
              <tr className="border-b border-line align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-ink">Achieved Occupancy</td>
                <td className="py-2.5 pr-4 text-ink2">
                  {t('Resident warps actually achieved / hardware limit', '实际驻留 warp 数 / 硬件上限')}
                </td>
                <td className="py-2.5 text-ink2">
                  {t(
                    'Low (<30%) often means register or shared-memory over-subscription, too few warps to hide memory latency',
                    '偏低（<30%）常因寄存器或共享内存超额：warp 不够多，访存延迟藏不住',
                  )}
                </td>
              </tr>
              <tr className="align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-amber">Warp Stall Reasons</td>
                <td className="py-2.5 pr-4 text-ink2">
                  {t('What a warp is waiting on when it can’t issue', 'warp 发不出指令时在等什么')}
                </td>
                <td className="py-2.5 text-ink2">
                  {t(
                    'long scoreboard = waiting on global memory (most common); barrier = waiting on __syncthreads; MIO throttle = shared-memory queueing',
                    'long scoreboard = 等全局内存（最常见）；barrier = 等 __syncthreads；MIO throttle = 共享内存排队',
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          {t(
            <>
              Of these,{' '}
              <Term t={<>occupancy</>}>
                the fraction of the hardware's warp limit that is actually resident and active on each SM. It measures
                your "budget for hiding latency": the more warps, the easier it is to find another one to step in while
                one is waiting on memory. But higher isn't always better, and enough is enough.
              </Term>{' '}
              is the easiest to misread. It isn't performance itself, just the budget for latency hiding, and "occupancy
              is high yet neither bandwidth nor compute is saturated" is exactly the sign that the problem lies elsewhere.
              Build the habit: with any profiler report in hand, find DRAM% and SM% first, plot the kernel onto the
              Roofline in your head, then decide whether to push right or up. That two-step move is the watershed between
              a performance engineer and a tune-by-superstition hacker.
            </>,
            <>
              其中{' '}
              <Term t="占用率（occupancy）">
                每个 SM 上实际驻留的活跃 warp 数占硬件上限的比例。它衡量的是「延迟隐藏的本钱」：warp 越多，某个 warp 等内存
                时越容易找到别的 warp 顶上。但它不是越高越好，够用就行。
              </Term>{' '}
              最容易被误读：它不是性能本身，只是延迟隐藏的本钱；「occupancy 很高但带宽和算力都没吃满」恰恰说明问题出在别处。
              养成一个习惯：拿到任何 profiler 报告，先找 DRAM% 和 SM%，在脑子里把这个 kernel 画到 Roofline 上，再决定往右还是
              往上使劲。这个两步动作，就是性能工程师和「玄学调参侠」的分水岭。
            </>,
          )}
        </p>
      </Section>

      <Section index={6} title={t('Summary and further reading', '总结与延伸阅读')} lead={t('One plot, two divisions, two roads.', '一张图、两个除法、两条路。')}>
        <ul>
          <li>
            {t(
              <>
                <strong>AI = FLOPs ÷ DRAM bytes</strong> is the program's constitution;{' '}
                <strong>ridge = peak compute ÷ bandwidth</strong> is the machine's temperament. AI &lt; ridge means
                memory-bound. The A100 BF16 ridge ≈ 164 FLOP/B, and not many kernels can clear it.
              </>,
              <>
                <strong>AI = FLOPs ÷ DRAM 字节</strong>是程序的体质；<strong>ridge = 峰值算力 ÷ 带宽</strong>是机器的脾气。
                AI &lt; ridge 即 memory-bound。A100 BF16 的 ridge ≈ 164 FLOP/B，能跨过去的 kernel 不多。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Elementwise, GEMV, and decode attention have <strong>constant</strong> AI (about 0.1–2); scaling up
                can't save them. GEMM's AI = (2/b)·1/(1/M+1/N+1/K), where{' '}
                <strong>the thinnest dimension calls the shots</strong>, and the bigger the square matrix, the better its
                constitution.
              </>,
              <>
                elementwise、GEMV、decode attention 的 AI 是<strong>常数</strong>（约 0.1~2），加大规模救不了它们；GEMM 的 AI =
                (2/b)·1/(1/M+1/N+1/K)，<strong>最瘦的维度说了算</strong>，方阵越大体质越好。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Hugging the slope → <strong>go right</strong>: kernel fusion, tiling reuse, quantization to cut bytes, or
                swap in higher bandwidth; hugging the plateau →{' '}
                <strong>go up</strong>: saturate the Tensor Cores, raise utilization. Prescribe the wrong cure = wasted
                effort.
              </>,
              <>
                贴斜线 → <strong>向右</strong>：kernel 融合、tiling 复用、量化减字节，或换更高带宽；贴平台 →{' '}
                <strong>向上</strong>：吃满 Tensor Core、提高利用率。开反药 = 白干。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                LLM decode's AI ≈ 1–2, doomed to be memory-bound. The coming chapters on KV cache, batching, and
                quantization are all about pushing this point right or up on the Roofline.
              </>,
              <>
                LLM decode 的 AI ≈ 1~2，注定 memory-bound。后面的 KV cache、batching、量化几章，全是在 Roofline 图上向右或
                向上推这个点。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Diagnostic order: <strong>Nsight Systems first for the timeline</strong> (gaps between kernels, whether
                copies overlap compute), <strong>then Nsight Compute to dig into a single kernel</strong> (DRAM%, SM%,
                occupancy, stall reasons).
              </>,
              <>
                诊断顺序：<strong>先 Nsight Systems 看时间线</strong>（kernel 间隙、拷贝是否与计算重叠），
                <strong>再 Nsight Compute 深挖单个 kernel</strong>（DRAM%、SM%、occupancy、stall 原因）。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Further reading:', '延伸阅读：')}</p>
        <ul>
          <li>
            <a href="https://dl.acm.org/doi/10.1145/1498765.1498785" target="_blank" rel="noreferrer">
              Williams, Waterman &amp; Patterson — Roofline: An Insightful Visual Performance Model (CACM 2009)
            </a>
            {t(
              <> — the original paper; first proposed for multicore CPUs, now the lingua franca of heterogeneous-compute performance analysis.</>,
              <>——原论文，最初为多核 CPU 提出，如今是异构计算性能分析的通用语言。</>,
            )}
          </li>
          <li>
            <a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html" target="_blank" rel="noreferrer">
              NVIDIA Nsight Compute — Kernel Profiling Guide
            </a>
            {t(
              <> — the official profiling guide, including the built-in Roofline analysis and the authoritative account of each warp-stall reason.</>,
              <>——官方剖析指南，含内置 Roofline 分析与各类 warp stall 原因的权威解释。</>,
            )}
          </li>
          <li>
            <a href="https://docs.nvidia.com/nsight-systems/UserGuide/index.html" target="_blank" rel="noreferrer">
              NVIDIA Nsight Systems — User Guide
            </a>
            {t(
              <> — the timeline-tool docs, the first step of the "macro before micro" diagnostic flow.</>,
              <>——时间线工具文档，「先宏观后微观」诊断流程的第一步。</>,
            )}
          </li>
          <li>
            <a
              href="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA — GPU Performance Background User&apos;s Guide
            </a>
            {t(
              <> — arithmetic intensity and performance ceilings from a deep-learning angle; the official companion read for this chapter's calculator.</>,
              <>——深度学习视角下的算术强度与性能上限，本章计算器的官方对照读物。</>,
            )}
          </li>
          <li>
            <a href="https://docs.nersc.gov/tools/performance/roofline/" target="_blank" rel="noreferrer">
              NERSC — Roofline Performance Model
            </a>
            {t(
              <> — a national lab's hands-on Roofline playbook, with a complete method for collecting roofline data on real GPUs.</>,
              <>——国家实验室的 Roofline 实战手册，含真实 GPU 上采集 roofline 数据的完整方法。</>,
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
