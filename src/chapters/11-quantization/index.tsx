import { Callout, MathTex, Quiz, Section, Term } from '@/components/ui'
import { FloatBitLab } from './FloatBitLab'
import { QuantErrorLab } from './QuantErrorLab'
import { MemoryCalcLab } from './MemoryCalcLab'

export default function Chapter() {
  return (
    <>
      <p>
        一个 70B 参数的模型，用 FP16 存权重要 <strong>140 GB</strong> —— 两张 80G 的 A100
        都装不下推理时的全部开销。把它量化（quantization）到 INT4，只剩约{' '}
        <strong>35 GB</strong>，单卡即可起服务；更反直觉的是，decode 速度还快了接近{' '}
        <strong>4 倍</strong>。注意这句话的诡异之处：INT4 的乘法并没有比 FP16
        快多少，甚至很多实现里乘法还是用 FP16 做的。快的真正原因是<strong>字节变少了</strong>。
        上一章 roofline 已经给过答案：decode 阶段算术强度只有 1~2 FLOPs/Byte，是典型的
        memory-bound —— 每生成一个 token，都要把全部权重从 HBM 搬进片上一遍。权重字节砍到
        1/4，每个 token 的搬运时间就砍到 1/4。这一章我们从比特层面把「数是怎么存的」拆开看，
        再看量化误差从哪来、被什么放大，最后用计算器算清楚：为什么量化几乎是白拿的推理加速。
      </p>

      <Section
        index={1}
        title="浮点数是怎么存的"
        lead="一个浮点数 = 一个符号位 + 一个量程旋钮 + 一把刻度尺。"
      >
        <p>
          所有 IEEE 风格的浮点格式都是同一套结构：最高 1 位是<strong>符号位（sign）</strong>，
          接着若干位<strong>指数（exponent）</strong>，剩下的是<strong>尾数（mantissa）</strong>。
          值由这条公式给出：
        </p>
        <MathTex block tex="x = (-1)^{s} \times 1.m \times 2^{\,e-\text{bias}}" />
        <p>
          指数决定数落在哪个「2 的幂区间」里 —— 它是<strong>量程旋钮</strong>，多一位指数，
          可表示范围就翻倍地扩张；尾数决定在这个区间内能切多细 —— 它是<strong>刻度尺</strong>，
          多一位尾数，区间内的格点数翻倍。注意尾数前面有个隐含的 1（写作{' '}
          <code>1.m</code>）：因为正规数的最高有效位必然是 1，干脆不存，白赚一位精度。
          指数字段存的是无符号整数 <code>e</code>，减去固定偏移 bias 才是真实指数，
          这样不用单独的指数符号位。
        </p>
        <p>
          三个特殊态值得记住：指数全 0 且尾数非 0 是
          <Term t="次正规数（subnormal）">
            指数到底后放弃隐含 1、让尾数自然下溢的小数，用精度换出比最小正规值更小的表示范围，
            避免「突然归零」。
          </Term>
          ；指数全 1 且尾数为 0 是 ±Inf；指数全 1 且尾数非 0 是 NaN。 这些约定在 FP8-E4M3
          上被打破了 —— 8 位实在太挤，E4M3 干脆取消了 Inf，把指数全 1 的编码空间也让给正规数，
          只保留一个 NaN 槽位，换回半格动态范围。
        </p>
        <p>常用格式的布局与「三围」如下（s/e/m 分别是符号、指数、尾数位数）：</p>
        <div className="my-6 overflow-x-auto">
          <table className="w-full min-w-[600px] border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                {['格式', '布局 s/e/m', 'bias', '最大值', '最小正规值', '1.0 附近间隔'].map((h) => (
                  <th key={h} className="py-2 pr-4 font-mono text-[11px] font-medium uppercase tracking-wider text-ink3">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono text-[12.5px] text-text">
              {[
                ['FP32', '1/8/23', '127', '≈3.4×10³⁸', '≈1.2×10⁻³⁸', '2⁻²³ ≈ 1.2×10⁻⁷'],
                ['FP16', '1/5/10', '15', '65 504', '6.1×10⁻⁵', '2⁻¹⁰ ≈ 9.8×10⁻⁴'],
                ['BF16', '1/8/7', '127', '≈3.4×10³⁸', '≈1.2×10⁻³⁸', '2⁻⁷ ≈ 7.8×10⁻³'],
                ['FP8-E4M3', '1/4/3', '7', '448', '2⁻⁶ ≈ 0.016', '2⁻³ = 0.125'],
                ['FP8-E5M2', '1/5/2', '15', '57 344', '6.1×10⁻⁵', '2⁻² = 0.25'],
              ].map((row) => (
                <tr key={row[0]} className="border-b border-line">
                  {row.map((c, i) => (
                    <td key={i} className={`py-2 pr-4 ${i === 0 ? 'text-ink' : ''} ${i === 1 ? 'text-cyan' : ''}`}>
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          对比 FP16 和 BF16 这一行最有意思：同样 16 位，FP16 给了尾数 10 位，所以 1.0
          附近刻度细 8 倍；BF16 把 8 位指数原封不动从 FP32 抄过来，范围直达
          3.4×10³⁸，代价是尾数只剩 7 位。BF16 的设计动机非常工程化：它就是 FP32
          砍掉低 16 位尾数 —— 转换只需截断，且任何 FP32 张量转 BF16 永远不会溢出。
          深度学习训练里梯度的动态范围远比精度重要（FP16 的 65504 上限在训练中真的会被撞穿），
          所以 BF16 成了训练默认格式。FP8 的两个变体也是同样的取舍在 8 位上重演：E4M3
          刻度更细，适合前向的权重和激活；E5M2 范围更大，适合数值上蹿下跳的反向梯度。
        </p>
      </Section>

      <Section
        index={2}
        title="LAB：把浮点数拆开看"
        lead="纸面规则不如亲手翻一位。下面每一个比特都可以点。"
      >
        <p>
          先按预设「0.1」—— 你会发现没有一个格式能精确存下 0.1（它在二进制下是无限循环小数），
          FP16 存成 0.0999755859375，BF16 存成 0.10009765625。再点一下指数的最低位，
          看值如何整段翻倍；点尾数最低位，看值挪动一小格。然后切到「最大值」，
          把指数位逐个点亮，亲眼撞上 Inf。最下面的两条数轴是本实验的核心：全景图里
          BF16 的条和 FP32 一样长，FP16 短一大截；放大图里 FP16 一格内挤了 1024
          个格点而 BF16 只有 128 个 —— 范围与精度的交易在这两张图里一目了然。
        </p>
        <FloatBitLab />
        <p>
          还有个值得注意的细节：浮点格点在数轴上<strong>不是均匀的</strong>。每跨过一个 2
          的幂，格距就翻倍 —— 在 1.0 附近 FP16 的格距约 0.001，到 60000 附近格距已经是
          32。这种「对数均匀」恰好契合神经网络权重的分布（大多数值挤在 0
          附近），这也是浮点格式天然比均匀整数格点更适合存权重的原因 ——
          记住这一点，下一节整数量化的麻烦正是从「格点均匀」来的。
        </p>
        <Quiz
          question="BF16 相对 FP16，本质上是用什么换了什么？"
          options={[
            {
              text: '牺牲尾数精度（10 位 → 7 位），换来与 FP32 相同的指数范围',
              correct: true,
              explain:
                '对。BF16 = FP32 砍掉低 16 位尾数，指数 8 位原样保留，所以范围同 FP32（≈3.4×10³⁸），而 FP16 只能到 65504。训练中梯度容易超出 FP16 范围，所以宁可要范围不要精度。',
            },
            {
              text: '牺牲表示范围，换取更细的刻度',
              explain: '说反了 —— 这是 FP16 的取舍。BF16 恰恰是范围派：指数 8 位照搬 FP32，尾数只剩 7 位。',
            },
            {
              text: '用更多的总位数换更高精度',
              explain: '两者都是 16 位，总位数一样。差别只在 16 位怎么在指数和尾数之间分配。',
            },
            {
              text: '没有取舍，BF16 全面优于 FP16',
              explain:
                '天下没有免费的位。BF16 在 1.0 附近的格距是 FP16 的 8 倍（2⁻⁷ vs 2⁻¹⁰），对精度敏感的推理场景 FP16 仍有优势。',
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title="整数量化：把实数轴铺成均匀格点"
        lead="浮点是对数格点，整数是均匀格点 —— 量化就是在两者之间做映射。"
      >
        <p>
          INT8 只有 256 个取值，本身表示不了 0.0123 这样的实数。量化的做法是给整个张量配一个
          <strong>缩放因子（scale）</strong>：把实数除以 scale、四舍五入到最近的整数格点。最常用的{' '}
          <Term t="absmax 对称量化">
            用张量绝对值的最大值确定 scale，使格点对称覆盖 [-absmax, +absmax]，零点恰好落在整数 0 上。
          </Term>
          长这样：
        </p>
        <MathTex block tex="s=\frac{\max\left|x\right|}{127},\qquad q=\mathrm{round}\!\left(\frac{x}{s}\right),\qquad \hat{x}=q\cdot s" />
        <p>
          反量化 <MathTex tex="\hat{x}" /> 与原值的差就是量化误差，每个数的误差最多半个格距（±s/2）。
          非对称量化只多一步：再加一个零点（zero-point）<MathTex tex="z" />，即{' '}
          <MathTex tex="\hat{x}=(q-z)\cdot s" />，把格点窗口整体平移到数据的真实区间上 ——
          对 ReLU 之后恒为正的激活，这能省下白白浪费的负半轴。
        </p>
        <p>
          剩下的问题是：<strong>一个 scale 管多大一片数？</strong>这叫量化粒度（granularity），
          它是误差和开销之间的标尺：
        </p>
        <ul>
          <li>
            <strong>per-tensor</strong>：整个张量共用 1 个 scale。开销最小，但只要张量里有一个异常大的值，
            所有数都得跟着用粗格距；
          </li>
          <li>
            <strong>per-channel</strong>：每个输出通道一个 scale。矩阵每一行自己定标尺，
            行与行之间互不拖累，开销仍可忽略；
          </li>
          <li>
            <strong>per-group（g=128）</strong>：每 128 个权重一个 scale，是 GPTQ/AWQ 等 INT4
            方案的标配。一个 FP16 的 scale 摊到 128 个权重头上只多 0.125 位/权重，
            却把异常值的破坏半径压缩到了 128 个数以内。
          </li>
        </ul>
        <p>
          粒度越细误差越小、元数据越多，但 scale 的存储开销实在不大，所以实践中 INT4
          几乎都用 per-group。真正的对手不是开销，而是下一节的主角：outlier。
        </p>
      </Section>

      <Section
        index={4}
        title="量化误差实验场：outlier 如何搞砸一切"
        lead="absmax 的命门写在名字里 —— 它把标尺交给了张量里最大的那一个数。"
      >
        <p>
          下面的实验场里有 4096 个服从高斯分布的「权重」，你可以注入一小撮
          outlier（异常值）并控制它们的幅度。先看默认状态：INT8-absmax 下注入 0.5% 的
          ×12σ outlier，SNR（信噪比）立刻掉了一截 —— 然后把幅度滑杆拉到
          ×30，看格点（竖线）发生什么：absmax 跟着最大值跑，<strong>格距被一个数撑大十几倍</strong>，
          主体 ±4σ 内只剩下零星几条 volt 色格点，其余全是接不到值的 rose 色摆设。
          再切到 INT4-absmax，7 个正格点几乎全部被浪费，主体直接糊成一团。
          最后切 INT4 g=128 —— 同样的 INT4，因为 outlier 被关在自己的 128 人小组里，SNR 反而回升。
        </p>
        <QuantErrorLab />
        <Callout type="insight" title="量化的敌人不是位数，是 outlier">
          <p>
            从 INT8 到 INT4 只丢 4 位，理论上 SNR 掉约 24 dB；而一个 ×30σ 的 outlier
            在 absmax 下能把有效格点压缩 30 倍，损失同样量级的精度 ——{' '}
            <strong>一个数毁掉一个张量</strong>。所以现代量化方法的主线剧情不是「发明更密的格点」，
            而是「怎么安置 outlier」：隔离它、抹平它，或者保护它。
          </p>
        </Callout>
        <p>
          麻烦的是，LLM 里的 outlier 不是实验里这种人造噪声，而是<strong>系统性现象</strong>。
          LLM.int8() 的作者 Dettmers 发现：6.7B 以上的模型中，激活（activation）会在固定的少数几个特征维度上
          出现比其他值大几十倍的 outlier，且这些维度对模型质量至关重要。他的解法是
          <strong>混合精度分解</strong>：把这极少数（约 0.1%）outlier 通道挑出来仍用 FP16
          算，其余 99.9% 走 INT8 —— 精度几乎无损，代价是 kernel 复杂了、还要做在线检测。
        </p>
        <p>
          SmoothQuant 换了个思路：既然激活的 outlier 难量化、而权重分布平整好量化，那就做个数学等价变换{' '}
          <MathTex tex="Y=(X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W)" /> ——
          把激活按通道除以一个平滑系数、再把这个系数乘进权重，
          <strong>把量化难度从激活搬到权重</strong>。两边都变得「平庸」之后，W8A8
          可以全程整数矩阵乘，吞吐直接受益于 INT8 Tensor Core。
        </p>
        <p>
          AWQ（Activation-aware Weight Quantization）则盯住 weight-only 场景：它观察到权重里约 1%
          的通道对应着激活 outlier 经过的「要道」，量化坏这些通道伤害最大。AWQ
          不把它们留成 FP16，而是给重要通道乘一个放大系数再量化 ——
          等效于让这些通道独享更细的格距，硬件上仍是干净统一的 INT4 格式。
          这三条路线殊途同归：<strong>识别谁重要，然后别让 absmax 一刀切。</strong>
        </p>
      </Section>

      <Section
        index={5}
        title="为什么量化 = 加速：搬得少就是快"
        lead="回到 roofline：decode 在斜坡最左端，那里带宽就是一切。"
      >
        <p>
          很多人对量化加速的第一直觉是「整数乘法快」，这在 decode 场景下是错的。decode
          每生成一个 token，要把全部权重读一遍，却只对每个权重做约 2 次浮点运算 ——
          算术强度 1~2 FLOPs/Byte，离 A100 约 100 的脊点（ridge point）差着两个数量级，
          GPU 的算力利用率常常不足 5%。<strong>此时性能公式退化为：tok/s ≈ 带宽 ÷ 权重字节数。</strong>
          分母砍半，速度翻倍 —— 这就是 INT8 ≈ 2×、INT4 ≈ 4× 的全部来源。
          主流的 W4A16 方案（GPTQ/AWQ 都是）权重以 INT4 存放，kernel 把权重块搬进片上后
          <strong>现场反量化回 FP16</strong>，乘加仍走 FP16 Tensor Core：HBM 流量按 INT4 算，
          数学精度按 FP16 算，两头占便宜。
        </p>
        <MemoryCalcLab />
        <p>
          用计算器验证一下钩子里的数字：70B + FP16 = 140 GB，24G 和 80G 卡都放不下，理论上限
          ~14 tok/s（还得先有 175G 显存才轮得到谈速度）；切到 INT4 只剩 35 GB，单张 80G
          卡放得下，理论上限冲到 ~57 tok/s。也要诚实地说清边界：这个加速只属于
          memory-bound 的场景。prefill 阶段和大 batch 服务是 compute-bound 的，weight-only
          量化帮不上忙（反量化甚至略有开销），那里需要的是 W8A8/FP8 这类让计算本身也降精度的方案 ——
          这正是 SmoothQuant 和 H100 FP8 的主场。
        </p>
        <Quiz
          question="INT4 weight-only 量化能把 decode 加速近 4 倍，根本原因是？"
          options={[
            {
              text: 'INT4 乘法器比 FP16 乘法器快 4 倍',
              explain:
                '不对 —— W4A16 里乘法压根还是 FP16 做的（kernel 里先反量化）。就算用整数乘法，decode 的算力利用率本来就只有几个百分点，算得再快也不解决问题。',
            },
            {
              text: '每个 token 要从 HBM 搬的权重字节数变成了 1/4，而 decode 是 memory-bound',
              correct: true,
              explain:
                '对。decode 算术强度只有 1~2 FLOPs/Byte，时间几乎全花在搬权重上：tok/s ≈ 带宽 ÷ 权重字节。字节砍到 1/4，每 token 的搬运时间就砍到 1/4 —— 瓶颈是搬字节，不是算。',
            },
            {
              text: '量化删掉了不重要的参数，模型变小了',
              explain: '量化不删参数 —— 70B 量化后还是 70B 个数，只是每个数从 16 位变成 4 位。删参数的那叫剪枝（pruning）。',
            },
            {
              text: 'INT4 数据能让 GPU 跑在更高的频率上',
              explain: '频率与数据格式无关。加速完全来自 HBM 流量变小，这是带宽瓶颈下的直接收益。',
            },
          ]}
        />
      </Section>

      <Section
        index={6}
        title="训练侧一瞥：混合精度与 FP8"
        lead="推理量化是「存得小」，训练降精度是「算得快还不能炸」。"
      >
        <p>
          训练的标准玩法是<strong>混合精度（mixed precision）</strong>：前向/反向用 BF16
          算（吃满 Tensor Core，A100 上 BF16 是 FP32 的 16 倍吞吐），但每个参数同时保留一份{' '}
          <strong>FP32 master 权重</strong>专门做更新。为什么必须留这份 FP32 副本？因为权重更新量{' '}
          <MathTex tex="\eta\cdot g" /> 往往比权重本身小 4~6 个数量级，而 BF16 只有 7
          位尾数 —— 当更新量小于权重的 2⁻⁸ 时，<MathTex tex="w + \eta g" /> 四舍五入后等于{' '}
          <MathTex tex="w" />，权重「纹丝不动」，训练悄无声息地停滞。FP32 的 23
          位尾数才能把这些微小更新一点点积累起来。
        </p>
        <p>
          如果用的是 FP16（范围只到 65504、最小正规值 6×10⁻⁵），还需要{' '}
          <strong>loss scaling</strong>：反向传播前把 loss 乘一个大系数（如
          1024），所有梯度按链式法则等比放大，从 FP16 的下溢区（&lt;2⁻²⁴
          会变成 0）抬回可表示范围，更新前再除回去；系数动态调整 —— 一旦梯度出现
          Inf/NaN 就跳过这步并把系数减半。BF16 因为范围同 FP32，基本免掉了这套体操，这正是它赢得训练默认地位的原因。
        </p>
        <p>
          再往下走一档是 <strong>FP8 训练</strong>：H100 的 Transformer Engine（TE）用 E4M3
          存前向的权重和激活、用 E5M2 存反向梯度，FP8 Tensor Core 吞吐约 2 倍于
          BF16（H100 上约 1979 vs 989 TFLOPS）。FP8 动态范围太窄，TE 给每个张量维护一个{' '}
          <strong>per-tensor scaling factor</strong>：记录最近若干步的 amax（绝对值最大值）历史，
          推算下一步的合适缩放，把张量「居中」放进 FP8 的可表示窗口里。累加仍在
          FP32 中进行，master 权重照旧 —— 降精度的永远只是「算」的那一段，
          「积累」的那一段从来不敢省。
        </p>
        <Callout type="deep" title="FP8 与 INT8：同是 8 位，分工不同">
          <p>
            INT8 格点均匀，配合 absmax/zero-point 适合分布稳定的<strong>推理</strong>权重；FP8
            自带对数格点和指数位，对动态范围剧烈变化的<strong>训练</strong>张量（尤其是梯度）
            宽容得多。所以业界的分工大体是：训练 BF16/FP8，推理 INT8/INT4/FP8 —— 格式跟着张量的统计性格走。
          </p>
        </Callout>
      </Section>

      <Section index={7} title="总结与延伸阅读">
        <ul>
          <li>
            浮点 = 符号 + 指数（量程）+ 尾数（刻度），<MathTex tex="(-1)^s\times 1.m\times 2^{e-\text{bias}}" />；
            指数位换范围，尾数位换精度。BF16 拿 FP16 的尾数精度换 FP32 的指数范围；FP8 的 E4M3/E5M2
            把同样的取舍在 8 位上重演。
          </li>
          <li>
            整数量化把实数映射到均匀格点：<MathTex tex="q=\mathrm{round}(x/s)" />，absmax 定
            scale。粒度 per-tensor → per-channel → per-group(128)，误差递减、开销略增。
          </li>
          <li>
            量化的头号敌人是 outlier：一个异常值就能撑爆 absmax 的格距。LLM.int8()
            把 outlier 通道隔离成 FP16，SmoothQuant 把难度从激活搬到权重，AWQ 给重要通道更细的等效格距。
          </li>
          <li>
            量化 = 加速的原因是 decode 是 memory-bound：tok/s ≈ 带宽 ÷ 权重字节。INT4 搬的字节是
            FP16 的 1/4，所以快 ~4 倍 —— 搬运变少了，不是乘法变快了。
          </li>
          <li>
            训练侧：BF16 算 + FP32 master 权重积累微小更新；FP16 需要 loss scaling 防梯度下溢；H100
            Transformer Engine 用 per-tensor scaling 把 FP8 塞进训练。
          </li>
        </ul>
        <p>想继续深挖，这几篇是绕不开的原始文献：</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/2208.07339" target="_blank" rel="noreferrer">
              LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
            </a>{' '}
            —— 系统性 outlier 现象的发现与混合精度分解。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2210.17323" target="_blank" rel="noreferrer">
              GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
            </a>{' '}
            —— 基于二阶信息的逐层 INT4/INT3 权重量化。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2306.00978" target="_blank" rel="noreferrer">
              AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
            </a>{' '}
            —— 用激活统计找出并保护 1% 的关键权重通道。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2211.10438" target="_blank" rel="noreferrer">
              SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs
            </a>{' '}
            —— 等价变换把激活的量化难度迁移给权重，实现全 INT8。
          </li>
          <li>
            <a href="https://arxiv.org/abs/2209.05433" target="_blank" rel="noreferrer">
              FP8 Formats for Deep Learning（NVIDIA / Arm / Intel）
            </a>{' '}
            —— E4M3/E5M2 的格式定义与 FP8 训练实践，Transformer Engine 的理论基础。
          </li>
        </ul>
      </Section>
    </>
  )
}
