import { Callout, MathTex, Quiz, Section, Term } from '@/components/ui'
import { IntensityCalc } from './IntensityCalc'
import { RooflineChart } from './RooflineChart'
import { HARDWARE, ridgeOf } from './data'

export default function Chapter() {
  return (
    <>
      <p>
        「这个 kernel 太慢了」——接到这种 issue，新手的第一反应是打开代码逐行找问题，老手的第一反应是先问一个判断题：
        它到底是<strong>算不动</strong>，还是<strong>喂不饱</strong>？前者是算力不够用，后者是数据供不上。两种病的症状一模一样
        （都是慢），药方却完全相反——开错了药，优化一星期，性能纹丝不动。Roofline 模型把这个判断压缩成一张 log-log
        图：横轴是程序的「体质」，纵轴是它能跑多快，任何 kernel 往图上一钉，瓶颈一眼可见。这一章先把这张图从零建起来，
        再告诉你真实世界里怎么用 profiler 把你的 kernel 钉上去。
      </p>

      <Section
        index={1}
        title="算术强度：kernel 的体质指标"
        lead="一个除法定义程序的体质，另一个除法定义机器的脾气——诊断就是比较这两个数。"
      >
        <p>
          前两章反复出现过一个事实：现代 GPU 是一台<strong>算力过剩、带宽稀缺</strong>的机器。拿 A100 的数字说话：BF16
          Tensor Core 峰值约 312 TFLOPS，HBM 带宽约 1.9 TB/s。也就是说，它每秒能做 312 万亿次浮点运算，却每秒只能从显存搬
          1.9 万亿个字节。两个数相除——每搬进 1 个字节，机器「免费附送」大约 164 次运算的机会。
          你的程序用不掉这些免费额度，算力就在围观。
        </p>
        <p>
          于是自然定义出<strong>算术强度（arithmetic intensity，AI）</strong>：一个 kernel 每从 DRAM
          搬运 1 字节数据，平均做多少次浮点运算。
        </p>
        <MathTex block tex="\mathrm{AI} \;=\; \frac{\text{FLOPs}}{\text{DRAM Bytes}} \qquad [\text{FLOP/B}]" />
        <p>
          为什么这个比值能决定性能？因为 kernel 的执行时间有两个互相独立的下限：算完所有 FLOPs 至少要
          FLOPs/峰值算力 这么久，搬完所有字节至少要 Bytes/带宽 这么久。计算和搬运在 GPU
          上高度重叠，所以总时间约等于两者中较大的那个：
        </p>
        <MathTex
          block
          tex="t \;\ge\; \max\!\left(\frac{\text{FLOPs}}{P_{\text{peak}}},\; \frac{\text{Bytes}}{BW}\right) \;\;\Longrightarrow\;\; P_{\text{attain}} \;=\; \min\big(\mathrm{AI}\times BW,\;\; P_{\text{peak}}\big)"
        />
        <p>
          右边这个 min 就是 Roofline 的全部数学：性能要么被带宽限制（AI × BW），要么被算力限制（P
          <sub>peak</sub>），取较小者。两条限制交汇的位置叫
          <Term t="机器平衡点（ridge point）">
            ridge = 峰值算力 ÷ 带宽。AI 恰好等于 ridge 的程序「正好」同时吃满算力与带宽；它是 Roofline
            图上斜线与平台的交点，也是 memory-bound 与 compute-bound 的分界线。
          </Term>
          ：
        </p>
        <MathTex
          block
          tex="\mathrm{ridge} \;=\; \frac{P_{\text{peak}}}{BW} \;=\; \frac{312\ \text{TFLOPS}}{1.9\ \text{TB/s}} \;\approx\; 164\ \text{FLOP/B}"
        />
        <p>
          AI 低于 ridge，时间被搬数据主导，叫 <strong className="text-amber">memory-bound</strong>（内存受限）；高于
          ridge，时间被计算主导，叫 <strong>compute-bound</strong>（算力受限）。注意这两个数的分工：AI
          是<strong>程序</strong>的属性，由算法和实现决定，跟跑在哪块卡上无关；ridge 是<strong>机器</strong>
          的属性，跟你的代码无关。诊断瓶颈，就是把这两个数拿出来比大小。几块常见卡的 ridge：
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="microlabel py-2 pr-4 font-normal">GPU</th>
                <th className="microlabel py-2 pr-4 font-normal">带宽</th>
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
          看出门道了吗：同一块 A100，用 FP32 CUDA Core 时 ridge 只有约 10——大多数程序都「够格」吃满 FP32
          算力；而换到 BF16 Tensor Core，ridge 飙到 164，能跨过这道门槛的程序就屈指可数了。
          <strong>Tensor Core 把算力抬高了 16 倍，但带宽没变，于是「喂不饱」成了常态。</strong>
        </p>
        <Callout type="note" title="分母里的字节怎么数">
          <p>
            AI 的分母是<strong>真正走 DRAM 的流量</strong>，不是代码里写的访存次数——缓存里复用掉的不算。手算时通常取理想情况
            （每个数据只从 DRAM 过一次），得到的是 AI 的上界；实际缓存不完美时 AI 只会更低、kernel 更偏
            memory-bound。下一节的计算器用的就是这种理想计法。
          </p>
        </Callout>
      </Section>

      <Section
        index={2}
        title="动手数 FLOPs 和字节"
        lead="数清楚分子分母，AI 就是个小学除法——难的是知道该数什么。"
      >
        <p>
          以最重要的 GEMM（通用矩阵乘，M×K 乘 K×N）为例：每个输出元素要做 K 次乘加，共 2MNK 个
          FLOPs；理想情况下三个矩阵各从显存过一遍，字节数是 (MN + MK + KN)·b，其中 b 是每个元素的字节数（BF16 为
          2）。整理一下会得到一个很有味道的形式：
        </p>
        <MathTex
          block
          tex="\mathrm{AI}_{\text{GEMM}} \;=\; \frac{2MNK}{(MN+MK+KN)\,b} \;=\; \frac{2}{b}\cdot\frac{1}{\frac{1}{M}+\frac{1}{N}+\frac{1}{K}}"
        />
        <p>
          右边是 M、N、K 的调和平均形式——和并联电阻一个形状，<strong>最小的那个维度说了算</strong>。方阵
          M=N=K=n 时 AI = 2n/(3b)（BF16 下约 n/3），随尺寸线性增长：矩阵越大，每个元素被复用的次数越多，体质越好。
          这就是「矩阵要够大才能吃满算力」的数学根源；反过来，哪怕 M 和 N 都是 8192，只要 K=16，AI
          就被这条短板按在地上。用下面的计算器亲手验证：
        </p>

        <IntensityCalc />

        <p>
          把五种操作切一遍，你会发现一个残酷的分类：<strong>向量加、SAXPY、GEMV 的 AI 是常数</strong>
          ——分子分母都正比于数据量，规模再大体质也不变，注定 memory-bound；attention score 的 AI 由序列长度和 head
          维度共同决定；只有 GEMM 能靠加大尺寸持续改善体质。这就是为什么「把小操作攒成大矩阵乘」是 GPU
          性能优化里的万能母题——从 cuBLAS 的 batched GEMM 到 LLM serving 的 continuous batching，本质都是这一招。
        </p>
      </Section>

      <Quiz
        question="向量加 c = a + b（FP16）：每个元素做 1 次加法、访存 6 字节。它的算术强度和瓶颈是？（参考：A100 BF16 ridge ≈ 164，FP32 ridge ≈ 10）"
        options={[
          {
            text: 'AI 在 0.25 以下的量级（远小于 1），妥妥的 memory-bound',
            correct: true,
            explain:
              'FP16 下 1 FLOP / 6 B ≈ 0.17，FP32 下 1/12 ≈ 0.08——不管精度和计法怎么变，都在 0.1~0.3 之间打转，离任何一条 ridge 都差两三个数量级。跑这种 kernel 时 GPU 绝大部分时间在搬数据，加法单元基本在围观。',
          },
          {
            text: 'AI ≈ 6（每 FLOP 摊 6 字节），已经接近 FP32 的 ridge，算 compute-bound',
            explain:
              '分子分母拿反了：AI 是「每字节多少 FLOP」，不是「每 FLOP 多少字节」。1 FLOP 配 6 字节，AI 是 1/6 ≈ 0.17。',
          },
          {
            text: 'AI 随 N 增大而提高，向量够长就能变成 compute-bound',
            explain:
              'FLOPs 和字节数都正比于 N，相除之后 N 被约掉——elementwise 操作的体质是常数，加大规模救不了它，这正是它和 GEMM 的本质区别。',
          },
          {
            text: '取决于 occupancy，光看 AI 无法判断',
            explain: 'occupancy 影响的是「离屋顶多远」（利用率），不改变「在哪个区」。AI 与 ridge 的比较只由算法和硬件规格决定。',
          },
        ]}
      />

      <Section
        index={3}
        title="Roofline：把判断画成一张图"
        lead="x 轴是体质（AI），y 轴是可达性能，两道屋顶罩在头上——所有 kernel 在这张图上无所遁形。"
      >
        <p>
          把上一节的 min 公式画出来：横轴取 AI（对数刻度），纵轴取可达性能（对数刻度）。左半边是一条斜线——性能 = AI ×
          带宽，每个字节的「含金量」每涨一分，性能成比例涨一分；过了 ridge
          被算力封顶，变成一条水平的平台。斜线加平台像一面屋顶，模型因此得名 Roofline。一个 kernel
          在图上是一个点：横坐标由算法决定，纵坐标是实测性能——
          <strong>点永远在屋顶之下，而点到屋顶的垂直距离，就是还没兑现的优化空间</strong>。
        </p>
        <p>
          下面这张图可以玩：切换三块卡看屋顶怎么变形，悬停预置的 kernel 点看它们为什么落在那里，
          再拖动「你的 kernel」感受不同位置的处境——往左是深渊，往右是平原。
        </p>

        <RooflineChart />

        <p>
          有两件事值得在图上反复咀嚼。第一，预置的五个点里四个挤在 ridge 左侧——日常 kernel 里 compute-bound
          反而是稀有物种，「喂不饱」才是常态。第二，切到 H100 看看：相比 A100 带宽涨了约 1.76 倍，算力却涨了约 3.2 倍，ridge
          从 164 右移到约 295——<strong>卡越新，算力越过剩，能吃满算力的程序越少</strong>。硬件的演进方向已经把话挑明了：
          未来的性能工程，大概率是和带宽搏斗的工程。
        </p>
      </Section>

      <Section
        index={4}
        title="两条优化路线：向右，或向上"
        lead="Roofline 上的位置不是命运，是坐标——知道自己在哪，才知道该往哪走。"
      >
        <p>
          点贴着<strong className="text-amber">带宽斜线</strong>（memory-bound 且接近屋顶）时，提频率、堆
          SM、换算力更强的卡统统无效——这一段屋顶等于 AI × BW，跟峰值算力一个字都不沾。想提速只有两条路。
          第一条，<strong>向右移（提高 AI）</strong>：① <strong>kernel 融合（fusion）</strong>——把 add → mul → GELU
          这类链条合成一个 kernel，中间结果留在寄存器里不落显存，DRAM 字节数直接砍掉三分之二；②{' '}
          <strong>tiling 复用</strong>——上一章 GEMM 的共享内存分块，本质就是让每个从 DRAM 来的字节被运算更多次；③{' '}
          <strong>量化（quantization）</strong>——FP16 换 INT8/INT4，同样的数据字节数减半再减半，AI 原地翻倍。第二条，
          <strong>抬高屋顶本身</strong>：换 HBM 更快的卡，或把热数据挪进更快的存储层级（L2、共享内存——相当于对更近的
          「DRAM」画一条更陡的斜线）。
        </p>
        <p>
          点贴着<strong className="text-volt">算力平台</strong>（compute-bound）时药方反过来：带宽已经不是问题，拼的是
          <strong>利用率</strong>——确认矩阵乘真的走了 Tensor Core（数据布局、对齐、精度都对）、提高占用率让 SM
          始终有活干、消除 warp 停顿。而如果点哪条屋顶都不贴、悬在半空，那是更初级的问题：访存不合并、分支发散、kernel
          太碎导致启动开销占大头——先把利用率做上去，再谈方向。
        </p>
        <Callout type="insight" title="LLM 推理的宿命，这张图早就写好了">
          <p>
            LLM 逐 token 解码（decode）时，batch=1 的矩阵乘全部退化成 GEMV，再加上每步都要扫一遍 KV cache——AI
            被钉死在 1~2 FLOP/B，离 ridge 差两个数量级，<strong>注定 memory-bound</strong>。这不是实现不好，是算法形状决定的。
            所以接下来几章其实都是在这张图上挪点：KV cache 优化（CH09）让每步少搬字节，batching（CH10）把 GEMV
            重新攒成 GEMM 向右移，量化（CH11）把每个权重的字节数压小、还是向右移。读懂了这张图，后面三章你已经知道答案的形状了。
          </p>
        </Callout>
      </Section>

      <Quiz
        question="一条 elementwise 链 add → mul → GELU 由三个独立 kernel 组成，每步都从 HBM 读入再写回，profiler 显示 DRAM 带宽利用率约 90%。哪个手段最有效？"
        options={[
          {
            text: '把三个 kernel 融合成一个，中间结果留在寄存器里',
            correct: true,
            explain:
              '三次「读入 + 写回」变成一次，DRAM 字节数约砍到 1/3，AI 翻三倍，点在 Roofline 上整体右移——memory-bound 的第一药方。这也是 torch.compile、Triton 融合 elementwise 链的核心收益来源。',
          },
          {
            text: '提高 occupancy，往 SM 上多塞一些 warp',
            explain:
              '带宽已经吃到 90%，瓶颈在 HBM 吞吐不在延迟隐藏——再多 warp 也只是把队排得更长。occupancy 是「点离屋顶远」时的药，不是「贴着斜线」时的药。',
          },
          {
            text: '改写成 Tensor Core 指令',
            explain: 'elementwise 没有矩阵乘结构，Tensor Core 根本用不上；更何况瓶颈在带宽，算力侧再强也推不动斜线区的屋顶。',
          },
          {
            text: '换一块算力翻倍、带宽不变的 GPU',
            explain: '斜线区的屋顶 = AI × BW，与峰值算力无关——这是四个选项里最贵且唯一完全无效的方案。',
          },
        ]}
      />

      <Section
        index={5}
        title="Profiler 思维：把你的 kernel 钉到图上"
        lead="Roofline 是地图，profiler 是 GPS——没有实测数据，再好的地图也只是装饰。"
      >
        <p>
          NVIDIA 的性能工具分工明确，记住这个顺序能少走一半弯路：<strong>先 Nsight Systems，后 Nsight
          Compute</strong>。Nsight Systems 给你整个程序的时间线（timeline）——CPU 在干什么、每个 kernel
          什么时候启动、显存拷贝和计算有没有重叠。很多「GPU 慢」的案子在这一层就破了：GPU 实际在空转，时间花在 kernel
          之间的间隙里（CPU 提交太慢、不必要的同步），或者 H2D/D2H 拷贝串行地横在计算中间。这种病去优化 kernel
          本身是南辕北辙。
        </p>
        <p>
          确认时间确实花在某个 kernel 内部之后，才轮到 Nsight Compute——单 kernel 的显微镜。它会重放（replay）目标
          kernel 几十次，采出每个硬件计数器，甚至直接内置了 Roofline 分析页：自动算出你的 AI 和实测
          FLOPS，把点画在屋顶下。哪怕手头暂时没有 GPU 环境，下面这几个指标的读法也值得先刻进脑子——它们是所有性能报告的通用语言：
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-[13.5px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                <th className="microlabel py-2 pr-4 font-normal">指标</th>
                <th className="microlabel py-2 pr-4 font-normal">含义</th>
                <th className="microlabel py-2 font-normal">怎么读</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-cyan">DRAM Throughput %</td>
                <td className="py-2.5 pr-4 text-ink2">实测显存带宽占峰值的比例</td>
                <td className="py-2.5 text-ink2">&gt;80% 说明贴着带宽斜线——按 memory-bound 开药（融合 / 复用 / 量化）</td>
              </tr>
              <tr className="border-b border-line align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-volt">SM Busy / Compute %</td>
                <td className="py-2.5 pr-4 text-ink2">算术管线的忙碌程度</td>
                <td className="py-2.5 text-ink2">和 DRAM% 对照着读：谁接近 100% 谁是主瓶颈；两个都低 = 利用率问题</td>
              </tr>
              <tr className="border-b border-line align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-ink">Achieved Occupancy</td>
                <td className="py-2.5 pr-4 text-ink2">实际驻留 warp 数 / 硬件上限</td>
                <td className="py-2.5 text-ink2">偏低（&lt;30%）常因寄存器或共享内存超额——warp 不够多，访存延迟藏不住</td>
              </tr>
              <tr className="align-top">
                <td className="py-2.5 pr-4 font-mono text-[12.5px] text-amber">Warp Stall Reasons</td>
                <td className="py-2.5 pr-4 text-ink2">warp 发不出指令时在等什么</td>
                <td className="py-2.5 text-ink2">
                  long scoreboard = 等全局内存（最常见）；barrier = 等 __syncthreads；MIO throttle = 共享内存排队
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          其中{' '}
          <Term t="占用率（occupancy）">
            每个 SM 上实际驻留的活跃 warp 数占硬件上限的比例。它衡量的是「延迟隐藏的本钱」：warp
            越多，某个 warp 等内存时越容易找到别的 warp 顶上。但它不是越高越好——够用就行。
          </Term>{' '}
          最容易被误读：它不是性能本身，而是延迟隐藏的本钱；「occupancy 很高但带宽和算力都没吃满」恰恰说明问题出在别处。
          养成一个习惯：拿到任何 profiler 报告，先找 DRAM% 和 SM%，在脑子里把这个 kernel 画到 Roofline
          上，再决定往右还是往上使劲——这个两步动作，就是性能工程师和「玄学调参侠」的分水岭。
        </p>
      </Section>

      <Section index={6} title="总结与延伸阅读" lead="一张图、两个除法、两条路。">
        <ul>
          <li>
            <strong>AI = FLOPs ÷ DRAM 字节</strong>是程序的体质；<strong>ridge = 峰值算力 ÷ 带宽</strong>
            是机器的脾气。AI &lt; ridge 即 memory-bound——A100 BF16 的 ridge ≈ 164 FLOP/B，能跨过去的 kernel 不多。
          </li>
          <li>
            elementwise、GEMV、decode attention 的 AI 是<strong>常数</strong>（约 0.1~2），加大规模救不了它们；GEMM 的 AI =
            (2/b)·1/(1/M+1/N+1/K)，<strong>最瘦的维度说了算</strong>，方阵越大体质越好。
          </li>
          <li>
            贴斜线 → <strong>向右</strong>：kernel 融合、tiling 复用、量化减字节，或换更高带宽；贴平台 →{' '}
            <strong>向上</strong>：吃满 Tensor Core、提高利用率。开反药 = 白干。
          </li>
          <li>
            LLM decode 的 AI ≈ 1~2，注定 memory-bound——后面的 KV cache、batching、量化几章，全是在 Roofline
            图上向右或向上推这个点。
          </li>
          <li>
            诊断顺序：<strong>先 Nsight Systems 看时间线</strong>（kernel 间隙、拷贝是否与计算重叠），
            <strong>再 Nsight Compute 深挖单个 kernel</strong>（DRAM%、SM%、occupancy、stall 原因）。
          </li>
        </ul>
        <p>延伸阅读：</p>
        <ul>
          <li>
            <a href="https://dl.acm.org/doi/10.1145/1498765.1498785" target="_blank" rel="noreferrer">
              Williams, Waterman &amp; Patterson — Roofline: An Insightful Visual Performance Model (CACM 2009)
            </a>
            ——原论文，最初为多核 CPU 提出，如今是异构计算性能分析的通用语言。
          </li>
          <li>
            <a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html" target="_blank" rel="noreferrer">
              NVIDIA Nsight Compute — Kernel Profiling Guide
            </a>
            ——官方剖析指南，含内置 Roofline 分析与各类 warp stall 原因的权威解释。
          </li>
          <li>
            <a href="https://docs.nvidia.com/nsight-systems/UserGuide/index.html" target="_blank" rel="noreferrer">
              NVIDIA Nsight Systems — User Guide
            </a>
            ——时间线工具文档，「先宏观后微观」诊断流程的第一步。
          </li>
          <li>
            <a
              href="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html"
              target="_blank"
              rel="noreferrer"
            >
              NVIDIA — GPU Performance Background User&apos;s Guide
            </a>
            ——深度学习视角下的算术强度与性能上限，本章计算器的官方对照读物。
          </li>
          <li>
            <a href="https://docs.nersc.gov/tools/performance/roofline/" target="_blank" rel="noreferrer">
              NERSC — Roofline Performance Model
            </a>
            ——国家实验室的 Roofline 实战手册，含真实 GPU 上采集 roofline 数据的完整方法。
          </li>
        </ul>
      </Section>
    </>
  )
}
