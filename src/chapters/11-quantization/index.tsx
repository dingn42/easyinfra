import { Callout, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { FloatBitLab } from './FloatBitLab'
import { QuantErrorLab } from './QuantErrorLab'
import { MemoryCalcLab } from './MemoryCalcLab'

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            A 70B-parameter model stores its weights in <strong>140 GB</strong> under FP16 — enough that
            two 80G A100s still can't hold the full inference footprint. Quantize it to INT4 and you're
            left with about <strong>35 GB</strong>, servable on a single card; more counterintuitively,
            decode also speeds up by nearly <strong>4×</strong>. Notice what's strange about that sentence:
            INT4 multiplication isn't meaningfully faster than FP16 — in many implementations the multiply
            is still done in FP16. The real reason it's faster is that <strong>there are fewer bytes</strong>.
            The previous chapter's roofline already handed us the answer: decode has an arithmetic intensity
            of just 1–2 FLOPs/Byte, the textbook memory-bound regime — every token generated drags the
            entire weight set out of HBM and onto the chip once. Cut the weight bytes to 1/4 and you cut the
            per-token transfer time to 1/4. This chapter pries open "how a number is actually stored" at the
            bit level, then traces where quantization error comes from and what amplifies it, and finally
            uses a calculator to nail down why quantization is nearly free inference speedup.
          </>,
          <>
            一个 70B 参数的模型，用 FP16 存权重要 <strong>140 GB</strong> —— 两张 80G 的 A100
            都装不下推理时的全部开销。把它量化（quantization）到 INT4，只剩约{' '}
            <strong>35 GB</strong>，单卡即可起服务；更反直觉的是，decode 速度还快了接近{' '}
            <strong>4 倍</strong>。注意这句话的诡异之处：INT4 的乘法并没有比 FP16
            快多少，甚至很多实现里乘法还是用 FP16 做的。快的真正原因是<strong>字节变少了</strong>。
            上一章 roofline 已经给过答案：decode 阶段算术强度只有 1~2 FLOPs/Byte，是典型的
            memory-bound —— 每生成一个 token，都要把全部权重从 HBM 搬进片上一遍。权重字节砍到
            1/4，每个 token 的搬运时间就砍到 1/4。这一章我们从比特层面把「数是怎么存的」拆开看，
            再看量化误差从哪来、被什么放大，最后用计算器算清楚：为什么量化几乎是白拿的推理加速。
          </>,
        )}
      </p>

      <Section
        index={1}
        title={t('How a floating-point number is stored', '浮点数是怎么存的')}
        lead={t(
          'A float = one sign bit + a range knob + a measuring stick.',
          '一个浮点数 = 一个符号位 + 一个量程旋钮 + 一把刻度尺。',
        )}
      >
        <p>
          {t(
            <>
              Every IEEE-style float shares the same skeleton: the top bit is the{' '}
              <strong>sign</strong>, followed by some <strong>exponent</strong> bits, with the rest given
              to the <strong>mantissa</strong>. The value comes from this formula:
            </>,
            <>
              所有 IEEE 风格的浮点格式都是同一套结构：最高 1 位是<strong>符号位（sign）</strong>，
              接着若干位<strong>指数（exponent）</strong>，剩下的是<strong>尾数（mantissa）</strong>。
              值由这条公式给出：
            </>,
          )}
        </p>
        <MathTex block tex="x = (-1)^{s} \times 1.m \times 2^{\,e-\text{bias}}" />
        <p>
          {t(
            <>
              The exponent decides which "power-of-2 interval" the number lands in — it's the{' '}
              <strong>range knob</strong>: one more exponent bit doubles the representable range. The
              mantissa decides how finely you can slice within that interval — it's the{' '}
              <strong>measuring stick</strong>: one more mantissa bit doubles the grid points inside the
              interval. Notice the implicit leading 1 in front of the mantissa (written <code>1.m</code>):
              since a normal number's most significant bit is always 1, you simply don't store it and pocket
              a free bit of precision. The exponent field holds an unsigned integer <code>e</code>; subtract
              a fixed bias to get the true exponent, which avoids needing a separate sign bit for the exponent.
            </>,
            <>
              指数决定数落在哪个「2 的幂区间」里 —— 它是<strong>量程旋钮</strong>，多一位指数，
              可表示范围就翻倍地扩张；尾数决定在这个区间内能切多细 —— 它是<strong>刻度尺</strong>，
              多一位尾数，区间内的格点数翻倍。注意尾数前面有个隐含的 1（写作{' '}
              <code>1.m</code>）：因为正规数的最高有效位必然是 1，干脆不存，白赚一位精度。
              指数字段存的是无符号整数 <code>e</code>，减去固定偏移 bias 才是真实指数，
              这样不用单独的指数符号位。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Three special states are worth memorizing. Exponent all-zero with a non-zero mantissa is a{' '}
              <Term t="subnormal">
                Once the exponent bottoms out, the implicit 1 is dropped and the mantissa underflows
                gracefully — trading precision for representable values smaller than the minimum normal,
                so you avoid "snapping straight to zero."
              </Term>
              ; exponent all-ones with a zero mantissa is ±Inf; exponent all-ones with a non-zero mantissa
              is NaN. These conventions get broken in FP8-E4M3 — 8 bits is simply too cramped, so E4M3 drops
              Inf entirely, hands the all-ones exponent encoding space back to normal numbers, keeps just one
              NaN slot, and buys back half a step of dynamic range.
            </>,
            <>
              三个特殊态值得记住：指数全 0 且尾数非 0 是
              <Term t="次正规数（subnormal）">
                指数到底后放弃隐含 1、让尾数自然下溢的小数，用精度换出比最小正规值更小的表示范围，
                避免「突然归零」。
              </Term>
              ；指数全 1 且尾数为 0 是 ±Inf；指数全 1 且尾数非 0 是 NaN。 这些约定在 FP8-E4M3
              上被打破了 —— 8 位实在太挤，E4M3 干脆取消了 Inf，把指数全 1 的编码空间也让给正规数，
              只保留一个 NaN 槽位，换回半格动态范围。
            </>,
          )}
        </p>
        <p>
          {t(
            'The layouts and "vital stats" of the common formats are below (s/e/m are the sign, exponent, and mantissa bit counts):',
            '常用格式的布局与「三围」如下（s/e/m 分别是符号、指数、尾数位数）：',
          )}
        </p>
        <div className="my-6 overflow-x-auto">
          <table className="w-full min-w-[600px] border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-line2 text-left">
                {[
                  t('format', '格式'),
                  t('layout s/e/m', '布局 s/e/m'),
                  'bias',
                  t('max value', '最大值'),
                  t('min normal', '最小正规值'),
                  t('step near 1.0', '1.0 附近间隔'),
                ].map((h) => (
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
          {t(
            <>
              The FP16-vs-BF16 row is the most interesting: at the same 16 bits, FP16 gives the mantissa 10
              bits, so its grid near 1.0 is 8× finer; BF16 copies FP32's 8-bit exponent over untouched,
              reaching all the way to 3.4×10³⁸ at the cost of only 7 mantissa bits. BF16's design rationale is
              deeply pragmatic: it is literally FP32 with the low 16 mantissa bits lopped off — conversion is
              just truncation, and any FP32 tensor can become BF16 without ever overflowing. In deep-learning
              training the dynamic range of gradients matters far more than precision (FP16's 65504 ceiling
              really does get smashed through during training), which is why BF16 became the default training
              format. FP8's two variants replay the same trade-off at 8 bits: E4M3 has a finer grid, suiting
              forward-pass weights and activations; E5M2 has more range, suiting the wildly fluctuating
              backward gradients.
            </>,
            <>
              对比 FP16 和 BF16 这一行最有意思：同样 16 位，FP16 给了尾数 10 位，所以 1.0
              附近刻度细 8 倍；BF16 把 8 位指数原封不动从 FP32 抄过来，范围直达
              3.4×10³⁸，代价是尾数只剩 7 位。BF16 的设计动机非常工程化：它就是 FP32
              砍掉低 16 位尾数 —— 转换只需截断，且任何 FP32 张量转 BF16 永远不会溢出。
              深度学习训练里梯度的动态范围远比精度重要（FP16 的 65504 上限在训练中真的会被撞穿），
              所以 BF16 成了训练默认格式。FP8 的两个变体也是同样的取舍在 8 位上重演：E4M3
              刻度更细，适合前向的权重和激活；E5M2 范围更大，适合数值上蹿下跳的反向梯度。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={2}
        title={t('LAB: Take a float apart', 'LAB：把浮点数拆开看')}
        lead={t(
          'Rules on paper beat nothing — but flipping a bit by hand beats both. Every bit below is clickable.',
          '纸面规则不如亲手翻一位。下面每一个比特都可以点。',
        )}
      >
        <p>
          {t(
            <>
              Start with the "0.1" preset — you'll find that no format stores 0.1 exactly (in binary it's a
              repeating fraction): FP16 stores it as 0.0999755859375, BF16 as 0.10009765625. Now flip the
              lowest exponent bit and watch the value double in one jump; flip the lowest mantissa bit and
              watch it nudge over by a single step. Then switch to "max value", light up the exponent bits one
              by one, and crash straight into Inf with your own eyes. The two number lines at the bottom are
              the heart of this lab: in the panorama BF16's bar is exactly as long as FP32's while FP16 falls
              well short; in the zoom view FP16 crams 1024 grid points into one bucket while BF16 has only 128
              — the range-vs-precision deal is laid bare in those two pictures.
            </>,
            <>
              先按预设「0.1」—— 你会发现没有一个格式能精确存下 0.1（它在二进制下是无限循环小数），
              FP16 存成 0.0999755859375，BF16 存成 0.10009765625。再点一下指数的最低位，
              看值如何整段翻倍；点尾数最低位，看值挪动一小格。然后切到「最大值」，
              把指数位逐个点亮，亲眼撞上 Inf。最下面的两条数轴是本实验的核心：全景图里
              BF16 的条和 FP32 一样长，FP16 短一大截；放大图里 FP16 一格内挤了 1024
              个格点而 BF16 只有 128 个 —— 范围与精度的交易在这两张图里一目了然。
            </>,
          )}
        </p>
        <FloatBitLab />
        <p>
          {t(
            <>
              One more detail worth flagging: floating-point grid points are <strong>not uniform</strong> on
              the number line. Every time you cross a power of 2, the spacing doubles — near 1.0 FP16's spacing
              is about 0.001, but near 60000 the spacing is already 32. This "logarithmically uniform" layout
              happens to match the distribution of neural-network weights (most values cluster near 0), which
              is exactly why float formats are naturally better than uniform integer grids for storing weights
              — keep this in mind, because the trouble with integer quantization in the next section comes
              precisely from its "uniform grid."
            </>,
            <>
              还有个值得注意的细节：浮点格点在数轴上<strong>不是均匀的</strong>。每跨过一个 2
              的幂，格距就翻倍 —— 在 1.0 附近 FP16 的格距约 0.001，到 60000 附近格距已经是
              32。这种「对数均匀」恰好契合神经网络权重的分布（大多数值挤在 0
              附近），这也是浮点格式天然比均匀整数格点更适合存权重的原因 ——
              记住这一点，下一节整数量化的麻烦正是从「格点均匀」来的。
            </>,
          )}
        </p>
        <Quiz
          question={t('What does BF16 fundamentally trade away, and for what, versus FP16?', 'BF16 相对 FP16，本质上是用什么换了什么？')}
          options={[
            {
              text: t('Sacrifices mantissa precision (10 bits → 7 bits) for the same exponent range as FP32', '牺牲尾数精度（10 位 → 7 位），换来与 FP32 相同的指数范围'),
              correct: true,
              explain: t(
                'Right. BF16 = FP32 with the low 16 mantissa bits cut, keeping all 8 exponent bits, so its range equals FP32 (≈3.4×10³⁸) while FP16 tops out at 65504. Gradients in training easily exceed FP16\'s range, so you take range over precision.',
                '对。BF16 = FP32 砍掉低 16 位尾数，指数 8 位原样保留，所以范围同 FP32（≈3.4×10³⁸），而 FP16 只能到 65504。训练中梯度容易超出 FP16 范围，所以宁可要范围不要精度。',
              ),
            },
            {
              text: t('Sacrifices representable range for a finer grid', '牺牲表示范围，换取更细的刻度'),
              explain: t(
                'Backwards — that\'s FP16\'s trade-off. BF16 is squarely in the range camp: 8 exponent bits copied from FP32, only 7 mantissa bits left.',
                '说反了 —— 这是 FP16 的取舍。BF16 恰恰是范围派：指数 8 位照搬 FP32，尾数只剩 7 位。',
              ),
            },
            {
              text: t('Spends more total bits to buy higher precision', '用更多的总位数换更高精度'),
              explain: t(
                'Both are 16 bits — the total bit count is identical. The only difference is how those 16 bits are split between exponent and mantissa.',
                '两者都是 16 位，总位数一样。差别只在 16 位怎么在指数和尾数之间分配。',
              ),
            },
            {
              text: t('No trade-off — BF16 is strictly better than FP16', '没有取舍，BF16 全面优于 FP16'),
              explain: t(
                'There\'s no free bit. BF16\'s grid spacing near 1.0 is 8× coarser than FP16 (2⁻⁷ vs 2⁻¹⁰), so for precision-sensitive inference FP16 still has the edge.',
                '天下没有免费的位。BF16 在 1.0 附近的格距是 FP16 的 8 倍（2⁻⁷ vs 2⁻¹⁰），对精度敏感的推理场景 FP16 仍有优势。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title={t('Integer quantization: paving the real line with a uniform grid', '整数量化：把实数轴铺成均匀格点')}
        lead={t(
          'Floats are a logarithmic grid, integers a uniform one — quantization is the mapping between them.',
          '浮点是对数格点，整数是均匀格点 —— 量化就是在两者之间做映射。',
        )}
      >
        <p>
          {t(
            <>
              INT8 has only 256 values and on its own can't represent a real number like 0.0123. The trick is
              to give the whole tensor a <strong>scale factor</strong>: divide the real value by the scale and
              round to the nearest integer grid point. The most common scheme,{' '}
              <Term t="absmax symmetric quantization">
                Use the tensor's maximum absolute value to set the scale, so the grid symmetrically covers
                [−absmax, +absmax] and zero lands exactly on integer 0.
              </Term>
              , looks like this:
            </>,
            <>
              INT8 只有 256 个取值，本身表示不了 0.0123 这样的实数。量化的做法是给整个张量配一个
              <strong>缩放因子（scale）</strong>：把实数除以 scale、四舍五入到最近的整数格点。最常用的{' '}
              <Term t="absmax 对称量化">
                用张量绝对值的最大值确定 scale，使格点对称覆盖 [-absmax, +absmax]，零点恰好落在整数 0 上。
              </Term>
              长这样：
            </>,
          )}
        </p>
        <MathTex block tex="s=\frac{\max\left|x\right|}{127},\qquad q=\mathrm{round}\!\left(\frac{x}{s}\right),\qquad \hat{x}=q\cdot s" />
        <p>
          {t(
            <>
              The gap between the dequantized <MathTex tex="\hat{x}" /> and the original value is the
              quantization error, at most half a grid step per number (±s/2). Asymmetric quantization adds just
              one more step: a zero-point <MathTex tex="z" />, i.e. <MathTex tex="\hat{x}=(q-z)\cdot s" />,
              which slides the whole grid window onto the data's real range — for post-ReLU activations that
              are always non-negative, this reclaims the otherwise-wasted negative half-axis.
            </>,
            <>
              反量化 <MathTex tex="\hat{x}" /> 与原值的差就是量化误差，每个数的误差最多半个格距（±s/2）。
              非对称量化只多一步：再加一个零点（zero-point）<MathTex tex="z" />，即{' '}
              <MathTex tex="\hat{x}=(q-z)\cdot s" />，把格点窗口整体平移到数据的真实区间上 ——
              对 ReLU 之后恒为正的激活，这能省下白白浪费的负半轴。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The remaining question is: <strong>how big a slice of numbers does one scale govern?</strong>{' '}
              This is the quantization granularity, the dial between error and overhead:
            </>,
            <>
              剩下的问题是：<strong>一个 scale 管多大一片数？</strong>这叫量化粒度（granularity），
              它是误差和开销之间的标尺：
            </>,
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>per-tensor</strong>: the entire tensor shares one scale. Lowest overhead, but a single
                abnormally large value forces every number onto a coarse grid;
              </>,
              <>
                <strong>per-tensor</strong>：整个张量共用 1 个 scale。开销最小，但只要张量里有一个异常大的值，
                所有数都得跟着用粗格距；
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>per-channel</strong>: one scale per output channel. Each row of the matrix sets its own
                ruler, so rows don't drag each other down, and the overhead is still negligible;
              </>,
              <>
                <strong>per-channel</strong>：每个输出通道一个 scale。矩阵每一行自己定标尺，
                行与行之间互不拖累，开销仍可忽略；
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>per-group (g=128)</strong>: one scale per 128 weights, the standard for INT4 schemes
                like GPTQ/AWQ. An FP16 scale amortized over 128 weights adds just 0.125 bits/weight, yet it
                shrinks an outlier's blast radius to within 128 numbers.
              </>,
              <>
                <strong>per-group（g=128）</strong>：每 128 个权重一个 scale，是 GPTQ/AWQ 等 INT4
                方案的标配。一个 FP16 的 scale 摊到 128 个权重头上只多 0.125 位/权重，
                却把异常值的破坏半径压缩到了 128 个数以内。
              </>,
            )}
          </li>
        </ul>
        <p>
          {t(
            <>
              Finer granularity means smaller error but more metadata; since a scale's storage cost is tiny,
              INT4 in practice almost always goes per-group. The real adversary isn't overhead — it's the star
              of the next section: the outlier.
            </>,
            <>
              粒度越细误差越小、元数据越多，但 scale 的存储开销实在不大，所以实践中 INT4
              几乎都用 per-group。真正的对手不是开销，而是下一节的主角：outlier。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={4}
        title={t('Quantization-error playground: how outliers wreck everything', '量化误差实验场：outlier 如何搞砸一切')}
        lead={t(
          "absmax's fatal flaw is right there in the name — it hands the ruler to the single largest number in the tensor.",
          'absmax 的命门写在名字里 —— 它把标尺交给了张量里最大的那一个数。',
        )}
      >
        <p>
          {t(
            <>
              The playground below holds 4096 Gaussian-distributed "weights"; you can inject a small handful
              of outliers and control their magnitude. Look at the default first: under INT8-absmax, injecting
              0.5% of ×12σ outliers immediately knocks the SNR down a notch — then drag the magnitude slider to
              ×30 and watch what happens to the grid (vertical lines): absmax chases the maximum,{' '}
              <strong>one number stretches the spacing more than tenfold</strong>, and inside the ±4σ body only
              a sparse few volt-colored grid points remain, the rest being rose-colored ornaments with no
              values to land on. Switch to INT4-absmax and nearly all 7 positive grid points are wasted, the
              body smearing into a blur. Finally switch to INT4 g=128 — the same INT4, but because the outlier
              is locked inside its own group of 128, the SNR actually recovers.
            </>,
            <>
              下面的实验场里有 4096 个服从高斯分布的「权重」，你可以注入一小撮
              outlier（异常值）并控制它们的幅度。先看默认状态：INT8-absmax 下注入 0.5% 的
              ×12σ outlier，SNR（信噪比）立刻掉了一截 —— 然后把幅度滑杆拉到
              ×30，看格点（竖线）发生什么：absmax 跟着最大值跑，<strong>格距被一个数撑大十几倍</strong>，
              主体 ±4σ 内只剩下零星几条 volt 色格点，其余全是接不到值的 rose 色摆设。
              再切到 INT4-absmax，7 个正格点几乎全部被浪费，主体直接糊成一团。
              最后切 INT4 g=128 —— 同样的 INT4，因为 outlier 被关在自己的 128 人小组里，SNR 反而回升。
            </>,
          )}
        </p>
        <QuantErrorLab />
        <Callout type="insight" title={t("Quantization's enemy isn't bit-width — it's the outlier", '量化的敌人不是位数，是 outlier')}>
          <p>
            {t(
              <>
                Dropping from INT8 to INT4 loses just 4 bits, costing about 24 dB of SNR in theory; meanwhile a
                single ×30σ outlier under absmax can compress the effective grid 30-fold, shedding precision of
                the same order — <strong>one number ruins one tensor</strong>. That's why the main plot of
                modern quantization methods isn't "invent a denser grid" but "where to put the outlier":
                isolate it, smooth it out, or protect it.
              </>,
              <>
                从 INT8 到 INT4 只丢 4 位，理论上 SNR 掉约 24 dB；而一个 ×30σ 的 outlier
                在 absmax 下能把有效格点压缩 30 倍，损失同样量级的精度 ——{' '}
                <strong>一个数毁掉一个张量</strong>。所以现代量化方法的主线剧情不是「发明更密的格点」，
                而是「怎么安置 outlier」：隔离它、抹平它，或者保护它。
              </>,
            )}
          </p>
        </Callout>
        <p>
          {t(
            <>
              The catch is that outliers in LLMs aren't artificial noise like in this lab — they're a{' '}
              <strong>systematic phenomenon</strong>. Dettmers, author of LLM.int8(), found that in models
              above 6.7B, activations develop outliers tens of times larger than other values in a fixed
              handful of feature dimensions, and those dimensions are critical to model quality. His fix is{' '}
              <strong>mixed-precision decomposition</strong>: pull out that tiny minority (~0.1%) of outlier
              channels and still compute them in FP16, while the other 99.9% go through INT8 — virtually
              lossless, at the cost of a more complex kernel and online outlier detection.
            </>,
            <>
              麻烦的是，LLM 里的 outlier 不是实验里这种人造噪声，而是<strong>系统性现象</strong>。
              LLM.int8() 的作者 Dettmers 发现：6.7B 以上的模型中，激活（activation）会在固定的少数几个特征维度上
              出现比其他值大几十倍的 outlier，且这些维度对模型质量至关重要。他的解法是
              <strong>混合精度分解</strong>：把这极少数（约 0.1%）outlier 通道挑出来仍用 FP16
              算，其余 99.9% 走 INT8 —— 精度几乎无损，代价是 kernel 复杂了、还要做在线检测。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              SmoothQuant takes a different angle: since activation outliers are hard to quantize while the
              weight distribution is flat and easy, apply a mathematically equivalent transform{' '}
              <MathTex tex="Y=(X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W)" /> — divide the activations
              channel-wise by a smoothing factor and multiply that same factor into the weights,{' '}
              <strong>migrating quantization difficulty from activations to weights</strong>. Once both sides
              are "mediocre," W8A8 can run integer matmul end to end and its throughput benefits directly from
              the INT8 Tensor Cores.
            </>,
            <>
              SmoothQuant 换了个思路：既然激活的 outlier 难量化、而权重分布平整好量化，那就做个数学等价变换{' '}
              <MathTex tex="Y=(X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W)" /> ——
              把激活按通道除以一个平滑系数、再把这个系数乘进权重，
              <strong>把量化难度从激活搬到权重</strong>。两边都变得「平庸」之后，W8A8
              可以全程整数矩阵乘，吞吐直接受益于 INT8 Tensor Core。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              AWQ (Activation-aware Weight Quantization) targets the weight-only setting: it observes that
              roughly 1% of weight channels sit on the "thoroughfares" that activation outliers pass through,
              and quantizing those channels badly does the most damage. Instead of keeping them in FP16, AWQ
              multiplies the important channels by an amplification factor before quantizing — equivalent to
              giving those channels their own finer grid, while the hardware still sees a clean, uniform INT4
              format. All three routes converge on the same idea:{' '}
              <strong>identify what matters, then don't let absmax paint everything with one brush.</strong>
            </>,
            <>
              AWQ（Activation-aware Weight Quantization）则盯住 weight-only 场景：它观察到权重里约 1%
              的通道对应着激活 outlier 经过的「要道」，量化坏这些通道伤害最大。AWQ
              不把它们留成 FP16，而是给重要通道乘一个放大系数再量化 ——
              等效于让这些通道独享更细的格距，硬件上仍是干净统一的 INT4 格式。
              这三条路线殊途同归：<strong>识别谁重要，然后别让 absmax 一刀切。</strong>
            </>,
          )}
        </p>
      </Section>

      <Section
        index={5}
        title={t('Why quantization = speedup: moving less is faster', '为什么量化 = 加速：搬得少就是快')}
        lead={t(
          'Back to the roofline: decode lives at the far-left of the ramp, where bandwidth is everything.',
          '回到 roofline：decode 在斜坡最左端，那里带宽就是一切。',
        )}
      >
        <p>
          {t(
            <>
              Many people's first instinct about quantization speedup is "integer multiply is fast," and in
              the decode setting that's wrong. Each token decode reads the entire weight set once but does only
              ~2 floating-point ops per weight — an arithmetic intensity of 1–2 FLOPs/Byte, two orders of
              magnitude short of the A100's ridge point of ~100, so GPU compute utilization is often under 5%.{' '}
              <strong>Here the performance formula degenerates to: tok/s ≈ bandwidth ÷ weight bytes.</strong>{' '}
              Halve the denominator and you double the speed — that's the entire source of INT8 ≈ 2× and INT4 ≈
              4×. The mainstream W4A16 schemes (GPTQ and AWQ both qualify) store weights as INT4, and once the
              kernel brings a weight block on-chip it <strong>dequantizes back to FP16 on the spot</strong>,
              with the multiply-accumulate still running on FP16 Tensor Cores: HBM traffic is billed at INT4,
              math precision at FP16 — you win on both ends.
            </>,
            <>
              很多人对量化加速的第一直觉是「整数乘法快」，这在 decode 场景下是错的。decode
              每生成一个 token，要把全部权重读一遍，却只对每个权重做约 2 次浮点运算 ——
              算术强度 1~2 FLOPs/Byte，离 A100 约 100 的脊点（ridge point）差着两个数量级，
              GPU 的算力利用率常常不足 5%。<strong>此时性能公式退化为：tok/s ≈ 带宽 ÷ 权重字节数。</strong>
              分母砍半，速度翻倍 —— 这就是 INT8 ≈ 2×、INT4 ≈ 4× 的全部来源。
              主流的 W4A16 方案（GPTQ/AWQ 都是）权重以 INT4 存放，kernel 把权重块搬进片上后
              <strong>现场反量化回 FP16</strong>，乘加仍走 FP16 Tensor Core：HBM 流量按 INT4 算，
              数学精度按 FP16 算，两头占便宜。
            </>,
          )}
        </p>
        <MemoryCalcLab />
        <p>
          {t(
            <>
              Use the calculator to verify the numbers from the hook: 70B + FP16 = 140 GB, which fits neither
              the 24G nor the 80G card, with a theoretical ceiling of ~14 tok/s (and you'd need 175G of VRAM
              before you even get to talk about speed); switch to INT4 and only 35 GB remains, fitting a single
              80G card, with the theoretical ceiling jumping to ~57 tok/s. To be honest about the boundary,
              though: this speedup belongs only to the memory-bound regime. Prefill and large-batch serving are
              compute-bound, where weight-only quantization doesn't help (dequantization even adds a little
              overhead); those need schemes like W8A8/FP8 that lower the precision of the computation itself —
              exactly the home turf of SmoothQuant and H100 FP8.
            </>,
            <>
              用计算器验证一下钩子里的数字：70B + FP16 = 140 GB，24G 和 80G 卡都放不下，理论上限
              ~14 tok/s（还得先有 175G 显存才轮得到谈速度）；切到 INT4 只剩 35 GB，单张 80G
              卡放得下，理论上限冲到 ~57 tok/s。也要诚实地说清边界：这个加速只属于
              memory-bound 的场景。prefill 阶段和大 batch 服务是 compute-bound 的，weight-only
              量化帮不上忙（反量化甚至略有开销），那里需要的是 W8A8/FP8 这类让计算本身也降精度的方案 ——
              这正是 SmoothQuant 和 H100 FP8 的主场。
            </>,
          )}
        </p>
        <Quiz
          question={t('INT4 weight-only quantization speeds up decode nearly 4×. The fundamental reason is?', 'INT4 weight-only 量化能把 decode 加速近 4 倍，根本原因是？')}
          options={[
            {
              text: t('INT4 multipliers are 4× faster than FP16 multipliers', 'INT4 乘法器比 FP16 乘法器快 4 倍'),
              explain: t(
                "Wrong — in W4A16 the multiply is still done in FP16 (the kernel dequantizes first). Even with integer multiply, decode's compute utilization is only a few percent to begin with, so computing faster solves nothing.",
                '不对 —— W4A16 里乘法压根还是 FP16 做的（kernel 里先反量化）。就算用整数乘法，decode 的算力利用率本来就只有几个百分点，算得再快也不解决问题。',
              ),
            },
            {
              text: t('The weight bytes streamed from HBM per token drop to 1/4, and decode is memory-bound', '每个 token 要从 HBM 搬的权重字节数变成了 1/4，而 decode 是 memory-bound'),
              correct: true,
              explain: t(
                'Right. Decode has an arithmetic intensity of just 1–2 FLOPs/Byte, so time is spent almost entirely moving weights: tok/s ≈ bandwidth ÷ weight bytes. Cut bytes to 1/4 and per-token transfer time drops to 1/4 — the bottleneck is moving bytes, not computing.',
                '对。decode 算术强度只有 1~2 FLOPs/Byte，时间几乎全花在搬权重上：tok/s ≈ 带宽 ÷ 权重字节。字节砍到 1/4，每 token 的搬运时间就砍到 1/4 —— 瓶颈是搬字节，不是算。',
              ),
            },
            {
              text: t('Quantization deletes unimportant parameters, making the model smaller', '量化删掉了不重要的参数，模型变小了'),
              explain: t(
                'Quantization deletes no parameters — a quantized 70B still has 70B numbers, each just shrunk from 16 bits to 4. Deleting parameters is called pruning.',
                '量化不删参数 —— 70B 量化后还是 70B 个数，只是每个数从 16 位变成 4 位。删参数的那叫剪枝（pruning）。',
              ),
            },
            {
              text: t('INT4 data lets the GPU run at a higher clock frequency', 'INT4 数据能让 GPU 跑在更高的频率上'),
              explain: t(
                'Frequency has nothing to do with the data format. The speedup comes entirely from smaller HBM traffic — the direct payoff under a bandwidth bottleneck.',
                '频率与数据格式无关。加速完全来自 HBM 流量变小，这是带宽瓶颈下的直接收益。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={6}
        title={t('A glance at training: mixed precision and FP8', '训练侧一瞥：混合精度与 FP8')}
        lead={t(
          'Inference quantization is "store it small"; training low-precision is "compute it fast without blowing up."',
          '推理量化是「存得小」，训练降精度是「算得快还不能炸」。',
        )}
      >
        <p>
          {t(
            <>
              The standard training playbook is <strong>mixed precision</strong>: run forward/backward in BF16
              (saturating the Tensor Cores — on the A100 BF16 has 16× the FP32 throughput), while keeping a
              separate <strong>FP32 master copy</strong> of every parameter just for the update. Why keep that
              FP32 copy? Because the weight update <MathTex tex="\eta\cdot g" /> is often 4–6 orders of
              magnitude smaller than the weight itself, and BF16 has only 7 mantissa bits — when the update is
              smaller than 2⁻⁸ of the weight, <MathTex tex="w + \eta g" /> rounds right back to{' '}
              <MathTex tex="w" />, the weight "doesn't budge," and training silently stalls. Only FP32's 23
              mantissa bits can accumulate these tiny updates bit by bit.
            </>,
            <>
              训练的标准玩法是<strong>混合精度（mixed precision）</strong>：前向/反向用 BF16
              算（吃满 Tensor Core，A100 上 BF16 是 FP32 的 16 倍吞吐），但每个参数同时保留一份{' '}
              <strong>FP32 master 权重</strong>专门做更新。为什么必须留这份 FP32 副本？因为权重更新量{' '}
              <MathTex tex="\eta\cdot g" /> 往往比权重本身小 4~6 个数量级，而 BF16 只有 7
              位尾数 —— 当更新量小于权重的 2⁻⁸ 时，<MathTex tex="w + \eta g" /> 四舍五入后等于{' '}
              <MathTex tex="w" />，权重「纹丝不动」，训练悄无声息地停滞。FP32 的 23
              位尾数才能把这些微小更新一点点积累起来。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              If you're on FP16 (range only to 65504, minimum normal 6×10⁻⁵), you also need{' '}
              <strong>loss scaling</strong>: before backprop, multiply the loss by a large factor (e.g. 1024) so
              every gradient scales up proportionally by the chain rule, lifting it out of FP16's underflow zone
              (anything &lt;2⁻²⁴ flushes to 0) and back into representable range, then divide it back out before
              the update; the factor adjusts dynamically — the moment a gradient turns Inf/NaN, skip the step
              and halve the factor. Because BF16 shares FP32's range, it largely sidesteps this whole routine,
              which is precisely how it won its place as the default training format.
            </>,
            <>
              如果用的是 FP16（范围只到 65504、最小正规值 6×10⁻⁵），还需要{' '}
              <strong>loss scaling</strong>：反向传播前把 loss 乘一个大系数（如
              1024），所有梯度按链式法则等比放大，从 FP16 的下溢区（&lt;2⁻²⁴
              会变成 0）抬回可表示范围，更新前再除回去；系数动态调整 —— 一旦梯度出现
              Inf/NaN 就跳过这步并把系数减半。BF16 因为范围同 FP32，基本免掉了这套体操，这正是它赢得训练默认地位的原因。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              One rung lower is <strong>FP8 training</strong>: the H100's Transformer Engine (TE) stores
              forward-pass weights and activations in E4M3 and backward gradients in E5M2, with FP8 Tensor
              Cores delivering about 2× the throughput of BF16 (~1979 vs 989 TFLOPS on the H100). FP8's dynamic
              range is too narrow, so TE maintains a <strong>per-tensor scaling factor</strong> for each
              tensor: it tracks the amax (max absolute value) history over the last several steps, predicts an
              appropriate scale for the next step, and "centers" the tensor into FP8's representable window.
              Accumulation still happens in FP32 and the master weights stay as before — what gets reduced in
              precision is always the "compute" part; the "accumulate" part is never something anyone dares to
              skimp on.
            </>,
            <>
              再往下走一档是 <strong>FP8 训练</strong>：H100 的 Transformer Engine（TE）用 E4M3
              存前向的权重和激活、用 E5M2 存反向梯度，FP8 Tensor Core 吞吐约 2 倍于
              BF16（H100 上约 1979 vs 989 TFLOPS）。FP8 动态范围太窄，TE 给每个张量维护一个{' '}
              <strong>per-tensor scaling factor</strong>：记录最近若干步的 amax（绝对值最大值）历史，
              推算下一步的合适缩放，把张量「居中」放进 FP8 的可表示窗口里。累加仍在
              FP32 中进行，master 权重照旧 —— 降精度的永远只是「算」的那一段，
              「积累」的那一段从来不敢省。
            </>,
          )}
        </p>
        <Callout type="deep" title={t('FP8 vs INT8: same 8 bits, different jobs', 'FP8 与 INT8：同是 8 位，分工不同')}>
          <p>
            {t(
              <>
                INT8's uniform grid, paired with absmax/zero-point, suits stably-distributed{' '}
                <strong>inference</strong> weights; FP8's built-in logarithmic grid and exponent bits are far
                more forgiving of <strong>training</strong> tensors whose dynamic range swings violently
                (gradients especially). So the industry's division of labor is roughly: BF16/FP8 for training,
                INT8/INT4/FP8 for inference — the format follows the tensor's statistical temperament.
              </>,
              <>
                INT8 格点均匀，配合 absmax/zero-point 适合分布稳定的<strong>推理</strong>权重；FP8
                自带对数格点和指数位，对动态范围剧烈变化的<strong>训练</strong>张量（尤其是梯度）
                宽容得多。所以业界的分工大体是：训练 BF16/FP8，推理 INT8/INT4/FP8 —— 格式跟着张量的统计性格走。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section index={7} title={t('Summary and further reading', '总结与延伸阅读')}>
        <ul>
          <li>
            {t(
              <>
                A float = sign + exponent (range) + mantissa (resolution),{' '}
                <MathTex tex="(-1)^s\times 1.m\times 2^{e-\text{bias}}" />; exponent bits buy range, mantissa
                bits buy precision. BF16 trades FP16's mantissa precision for FP32's exponent range; FP8's
                E4M3/E5M2 replay the same trade-off at 8 bits.
              </>,
              <>
                浮点 = 符号 + 指数（量程）+ 尾数（刻度），<MathTex tex="(-1)^s\times 1.m\times 2^{e-\text{bias}}" />；
                指数位换范围，尾数位换精度。BF16 拿 FP16 的尾数精度换 FP32 的指数范围；FP8 的 E4M3/E5M2
                把同样的取舍在 8 位上重演。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Integer quantization maps reals onto a uniform grid: <MathTex tex="q=\mathrm{round}(x/s)" />,
                with absmax setting the scale. Granularity goes per-tensor → per-channel → per-group(128), with
                error shrinking and overhead rising slightly.
              </>,
              <>
                整数量化把实数映射到均匀格点：<MathTex tex="q=\mathrm{round}(x/s)" />，absmax 定
                scale。粒度 per-tensor → per-channel → per-group(128)，误差递减、开销略增。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Quantization's number-one enemy is the outlier: a single anomaly can blow out absmax's grid
                spacing. LLM.int8() isolates outlier channels into FP16, SmoothQuant migrates the difficulty
                from activations to weights, and AWQ gives important channels a finer effective grid.
              </>,
              <>
                量化的头号敌人是 outlier：一个异常值就能撑爆 absmax 的格距。LLM.int8()
                把 outlier 通道隔离成 FP16，SmoothQuant 把难度从激活搬到权重，AWQ 给重要通道更细的等效格距。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Quantization = speedup because decode is memory-bound: tok/s ≈ bandwidth ÷ weight bytes. INT4
                moves 1/4 the bytes of FP16, hence ~4× faster — less moving, not faster multiplying.
              </>,
              <>
                量化 = 加速的原因是 decode 是 memory-bound：tok/s ≈ 带宽 ÷ 权重字节。INT4 搬的字节是
                FP16 的 1/4，所以快 ~4 倍 —— 搬运变少了，不是乘法变快了。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                On the training side: BF16 compute + an FP32 master weight to accumulate tiny updates; FP16
                needs loss scaling to prevent gradient underflow; the H100 Transformer Engine uses per-tensor
                scaling to squeeze FP8 into training.
              </>,
              <>
                训练侧：BF16 算 + FP32 master 权重积累微小更新；FP16 需要 loss scaling 防梯度下溢；H100
                Transformer Engine 用 per-tensor scaling 把 FP8 塞进训练。
              </>,
            )}
          </li>
        </ul>
        <p>{t('To dig deeper, these primary sources are unavoidable:', '想继续深挖，这几篇是绕不开的原始文献：')}</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/2208.07339" target="_blank" rel="noreferrer">
              LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
            </a>{' '}
            {t(
              '— the discovery of the systematic outlier phenomenon and mixed-precision decomposition.',
              '—— 系统性 outlier 现象的发现与混合精度分解。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2210.17323" target="_blank" rel="noreferrer">
              GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
            </a>{' '}
            {t(
              '— layer-wise INT4/INT3 weight quantization based on second-order information.',
              '—— 基于二阶信息的逐层 INT4/INT3 权重量化。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2306.00978" target="_blank" rel="noreferrer">
              AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
            </a>{' '}
            {t(
              '— using activation statistics to find and protect the 1% of critical weight channels.',
              '—— 用激活统计找出并保护 1% 的关键权重通道。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2211.10438" target="_blank" rel="noreferrer">
              SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs
            </a>{' '}
            {t(
              '— an equivalent transform that migrates activation quantization difficulty to weights, achieving full INT8.',
              '—— 等价变换把激活的量化难度迁移给权重，实现全 INT8。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2209.05433" target="_blank" rel="noreferrer">
              {t('FP8 Formats for Deep Learning (NVIDIA / Arm / Intel)', 'FP8 Formats for Deep Learning（NVIDIA / Arm / Intel）')}
            </a>{' '}
            {t(
              '— the E4M3/E5M2 format definitions and FP8 training practice, the theoretical basis of the Transformer Engine.',
              '—— E4M3/E5M2 的格式定义与 FP8 训练实践，Transformer Engine 的理论基础。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
