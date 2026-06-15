import { Callout, ChapterLink, HardwareBaseline, MathTex, Quiz, Section, Term } from '@/components/ui'
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
            A 70B model stores its weights in <strong>140 GB</strong> under FP16. Two 80G A100s still can't
            hold the full inference footprint. Quantize the same model to INT4 and you're left with about{' '}
            <strong>35 GB</strong>, servable on a single card, and decode runs nearly <strong>4×</strong> faster
            on top of that. The second half of that sentence should bother you. INT4 multiplication isn't
            meaningfully faster than FP16; plenty of implementations still do the multiply in FP16. What got
            faster is that <strong>there are fewer bytes to move</strong>. <ChapterLink n={6} />{"'s"} roofline
            already gave us the answer: decode has an arithmetic intensity of 1–2 FLOPs/Byte, deep in
            memory-bound territory, and every token generated drags the entire weight set out of HBM and onto
            the chip once. Cut the weight bytes to a quarter and the per-token transfer time drops to a quarter.
            This chapter opens up how a number is actually stored, down to the bit, then follows quantization
            error to its source and to what amplifies it, and closes with a calculator that pins down why
            quantization is close to free inference speedup.
          </>,
          <>
            一个 70B 模型用 FP16 存权重要 <strong>140 GB</strong>，两张 80G 的 A100
            都装不下推理时的全部开销。把同一个模型量化（quantization）到 INT4，只剩约{' '}
            <strong>35 GB</strong>，单卡就能起服务，而且 decode 还快了接近{' '}
            <strong>4 倍</strong>。后半句该让你警觉：INT4 的乘法并不比 FP16
            快多少，很多实现里乘法干脆还是用 FP16 做。变快的是<strong>要搬的字节少了</strong>。
            <ChapterLink n={6} />的 roofline 已经给过答案：decode 的算术强度只有 1~2 FLOPs/Byte，深陷
            memory-bound，每生成一个 token 都要把全部权重从 HBM 搬进片上一遍。权重字节砍到
            四分之一，每个 token 的搬运时间也砍到四分之一。这一章我们从比特层面拆开「一个数到底怎么存」，
            再追量化误差从哪来、被什么放大，最后用一个计算器算清楚：为什么量化几乎是白拿的推理加速。
          </>,
        )}
      </p>

      <HardwareBaseline
        ids={['a100']}
        note={t('Figures assume an 80GB-class card (A100/H100).', '数字以 80GB 级别的卡（A100/H100）为准。')}
      />

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
              Every IEEE-style float has the same skeleton: the top bit is the <strong>sign</strong>, then a
              run of <strong>exponent</strong> bits, and the rest go to the <strong>mantissa</strong>. The
              value comes out of this formula:
            </>,
            <>
              所有 IEEE 风格的浮点格式都是同一套骨架：最高 1 位是<strong>符号位（sign）</strong>，
              接着一串<strong>指数（exponent）</strong>位，剩下的全归<strong>尾数（mantissa）</strong>。
              值由这条公式算出来：
            </>,
          )}
        </p>
        <MathTex block tex="x = (-1)^{s} \times 1.m \times 2^{\,e-\text{bias}}" />
        <p>
          {t(
            <>
              The exponent picks which power-of-2 interval the number lands in. It's the{' '}
              <strong>range knob</strong>: each extra exponent bit doubles the representable range. The
              mantissa decides how finely you can slice inside that interval. It's the{' '}
              <strong>measuring stick</strong>: each extra mantissa bit doubles the grid points within an
              interval. The leading 1 in front of the mantissa is implicit (that's the <code>1.m</code>
              notation). A normal number's most significant bit is always 1, so you don't store it and pocket
              a free bit of precision. The exponent field is an unsigned integer <code>e</code>; subtract a
              fixed bias to recover the true exponent, which saves you a separate sign bit on the exponent.
            </>,
            <>
              指数挑选这个数落在哪个 2 的幂区间里，是那个<strong>量程旋钮</strong>：每多一位指数，
              可表示范围就翻倍。尾数决定在这个区间内能切多细，是那把<strong>刻度尺</strong>：每多一位尾数，
              区间内的格点数翻倍。尾数前面那个 1 是隐含的（这就是 <code>1.m</code>{' '}
              这种写法的来历）：正规数的最高有效位必然是 1，干脆不存，白赚一位精度。
              指数字段存的是无符号整数 <code>e</code>，减去固定偏移 bias 才得到真实指数，
              省掉了一位单独的指数符号位。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Three special states are worth memorizing. Exponent all-zero with a non-zero mantissa is a{' '}
              <Term t="subnormal">
                Once the exponent bottoms out, the implicit 1 is dropped and the mantissa underflows
                gracefully, trading precision for representable values smaller than the minimum normal so a
                number doesn't snap straight to zero.
              </Term>
              ; exponent all-ones with a zero mantissa is ±Inf; exponent all-ones with a non-zero mantissa is
              NaN. FP8-E4M3 breaks these conventions because 8 bits is just too cramped to spend on them. E4M3
              drops Inf entirely, hands the all-ones exponent encoding back to normal numbers, keeps a single
              NaN slot, and buys back half a step of dynamic range in the process.
            </>,
            <>
              三个特殊态值得记住：指数全 0 且尾数非 0 是
              <Term t="次正规数（subnormal）">
                指数到底后放弃隐含 1、让尾数自然下溢的小数，用精度换出比最小正规值更小的表示范围，
                免得数字直接归零。
              </Term>
              ；指数全 1 且尾数为 0 是 ±Inf；指数全 1 且尾数非 0 是 NaN。FP8-E4M3
              打破了这些约定，因为 8 位实在太挤、舍不得花在这上面。E4M3 干脆取消 Inf，把指数全 1
              的编码让回正规数，只留一个 NaN 槽位，顺手换回半格动态范围。
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
              The FP16-vs-BF16 row is the one to stare at. Same 16 bits, but FP16 spends 10 of them on the
              mantissa, so its grid near 1.0 is 8× finer; BF16 copies FP32's 8-bit exponent over untouched and
              reaches all the way to 3.4×10³⁸, paying for it with only 7 mantissa bits. BF16's design is about
              as pragmatic as it gets: it's literally FP32 with the low 16 mantissa bits chopped off, so
              conversion is plain truncation and any FP32 tensor becomes BF16 without ever overflowing. In
              training, the dynamic range of gradients matters far more than precision, and FP16's 65504
              ceiling really does get smashed through mid-run, which is how BF16 became the default training
              format. FP8's two variants make the same trade-off at 8 bits: E4M3 has the finer grid and
              suits forward-pass weights and activations, while E5M2 has the wider range and suits the wildly
              swinging backward gradients.
            </>,
            <>
              这张表最该盯着看的是 FP16 和 BF16 这一行。同样 16 位，FP16 把其中 10 位花在尾数上，所以 1.0
              附近的刻度细 8 倍；BF16 把 8 位指数原封不动从 FP32 抄过来，范围直达
              3.4×10³⁸，代价是尾数只剩 7 位。BF16 的设计务实到了极点：它就是 FP32
              砍掉低 16 位尾数，转换只是单纯截断，任何 FP32 张量转 BF16 都不会溢出。
              训练里梯度的动态范围远比精度重要，FP16 的 65504 上限训练途中真的会被撞穿，BF16
              就这样坐稳了训练默认格式的位置。FP8 的两个变体把同样的取舍搬到 8 位上重演：E4M3
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
              Start with the "0.1" preset. No format stores 0.1 exactly, since in binary it's a repeating
              fraction: FP16 lands on 0.0999755859375, BF16 on 0.10009765625. Flip the lowest exponent bit and
              the value doubles in one jump; flip the lowest mantissa bit and it nudges over by a single step.
              Then switch to "max value", light up the exponent bits one at a time, and watch it crash straight
              into Inf. The two number lines at the bottom are the point of this lab. In the panorama, BF16's
              bar is exactly as long as FP32's and FP16 falls well short; in the zoom view, FP16 crams 1024
              grid points into one bucket while BF16 fits only 128. The range-vs-precision deal is sitting
              right there in those two pictures.
            </>,
            <>
              先按预设「0.1」。没有一个格式能精确存下 0.1，它在二进制下是无限循环小数：FP16
              落在 0.0999755859375，BF16 落在 0.10009765625。点一下指数的最低位，值就整段翻倍；
              点尾数的最低位，值只挪一小格。再切到「最大值」，把指数位一个个点亮，看它直直撞上 Inf。
              最下面那两条数轴才是这个实验的重点：全景图里 BF16 的条和 FP32 一样长，FP16
              短一大截；放大图里 FP16 一格内挤了 1024 个格点，BF16 只有 128 个。
              范围和精度的交易，就摆在这两张图里。
            </>,
          )}
        </p>
        <FloatBitLab />
        <p>
          {t(
            <>
              One detail to carry forward: floating-point grid points are <strong>not uniform</strong> along
              the number line. Spacing doubles every time you cross a power of 2. Near 1.0, FP16's spacing is
              about 0.001; near 60000 it's already 32. That logarithmic layout happens to match how
              neural-network weights are distributed, with most values bunched near 0, which is why float
              formats beat uniform integer grids for storing weights in the first place. Hold on to this,
              because the trouble with integer quantization in the next section comes straight out of its
              uniform grid.
            </>,
            <>
              有个细节要带到下一节：浮点格点在数轴上<strong>并不均匀</strong>。每跨过一个 2
              的幂，格距就翻倍。1.0 附近 FP16 的格距约 0.001，到 60000 附近已经是
              32。这种对数式的分布恰好契合神经网络权重的样子，大多数值都挤在 0
              附近，这也正是浮点格式一开始就比均匀整数格点更适合存权重的原因。
              记住这点，下一节整数量化的麻烦，正是从它那张均匀格点来的。
            </>,
          )}
        </p>
        <Quiz
          question={t('Versus FP16, what does BF16 give up, and what does it get back?', 'BF16 相对 FP16，放弃了什么、换回了什么？')}
          options={[
            {
              text: t('Gives up mantissa precision (10 bits → 7 bits) to get FP32\'s exponent range', '放弃尾数精度（10 位 → 7 位），换回 FP32 的指数范围'),
              correct: true,
              explain: t(
                'Right. BF16 is FP32 with the low 16 mantissa bits cut, keeping all 8 exponent bits, so its range matches FP32 (≈3.4×10³⁸) while FP16 tops out at 65504. Training gradients easily blow past FP16\'s range, so you take the range and live with the precision.',
                '对。BF16 就是 FP32 砍掉低 16 位尾数、保留全部 8 位指数，所以范围和 FP32 一样（≈3.4×10³⁸），而 FP16 只到 65504。训练梯度轻松冲出 FP16 的范围，于是宁可要范围、忍下精度。',
              ),
            },
            {
              text: t('Gives up representable range for a finer grid', '放弃表示范围，换更细的刻度'),
              explain: t(
                'That\'s backwards: it describes FP16. BF16 sits firmly in the range camp, with 8 exponent bits copied from FP32 and only 7 mantissa bits left.',
                '说反了，这是 FP16。BF16 是彻头彻尾的范围派：指数 8 位照搬 FP32，尾数只剩 7 位。',
              ),
            },
            {
              text: t('Spends more total bits to buy higher precision', '用更多的总位数换更高精度'),
              explain: t(
                'Both are 16 bits; the total is identical. The only difference is how those 16 bits split between exponent and mantissa.',
                '两者都是 16 位，总数一样。区别只在这 16 位怎么在指数和尾数之间分配。',
              ),
            },
            {
              text: t('No trade-off — BF16 is strictly better than FP16', '没有取舍，BF16 全面优于 FP16'),
              explain: t(
                'No free bits here. BF16\'s grid near 1.0 is 8× coarser than FP16 (2⁻⁷ vs 2⁻¹⁰), so precision-sensitive inference still favors FP16.',
                '没有白来的位。BF16 在 1.0 附近的格距是 FP16 的 8 倍（2⁻⁷ vs 2⁻¹⁰），对精度敏感的推理场景 FP16 仍占优。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={3}
        title={t('Integer quantization: paving the real line with a uniform grid', '整数量化：把实数轴铺成均匀格点')}
        lead={t(
          'Floats are a logarithmic grid, integers a uniform one. Quantization is the mapping between them.',
          '浮点是对数格点，整数是均匀格点。量化就是在两者之间架起映射。',
        )}
      >
        <p>
          {t(
            <>
              INT8 has 256 values and on its own can't represent a real number like 0.0123. The trick is to
              give the whole tensor a <strong>scale factor</strong>: divide the real value by the scale and
              round to the nearest integer grid point. The most common scheme,{' '}
              <Term t="absmax symmetric quantization">
                Use the tensor's maximum absolute value to set the scale, so the grid symmetrically covers
                [−absmax, +absmax] and zero lands exactly on integer 0.
              </Term>
              , looks like this:
            </>,
            <>
              INT8 只有 256 个取值，单凭自己表示不了 0.0123 这样的实数。量化的办法是给整个张量配一个
              <strong>缩放因子（scale）</strong>：把实数除以 scale，再四舍五入到最近的整数格点。最常用的{' '}
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
              quantization error, at most half a grid step per number (±s/2). Asymmetric quantization adds one
              more knob, a zero-point <MathTex tex="z" />, giving <MathTex tex="\hat{x}=(q-z)\cdot s" /> and
              sliding the whole grid window onto the data's actual range. For post-ReLU activations, which are
              always non-negative, that reclaims the negative half-axis you'd otherwise throw away.
            </>,
            <>
              反量化 <MathTex tex="\hat{x}" /> 和原值的差就是量化误差，每个数最多差半个格距（±s/2）。
              非对称量化多加一个旋钮，零点（zero-point）<MathTex tex="z" />，得到{' '}
              <MathTex tex="\hat{x}=(q-z)\cdot s" />，把整个格点窗口平移到数据真实落点的区间上。
              对 ReLU 之后恒为正的激活，这就把本来要白扔的负半轴捡了回来。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              That leaves one question: <strong>how big a slice of numbers does a single scale govern?</strong>{' '}
              This is quantization granularity, the dial between error and overhead:
            </>,
            <>
              剩下一个问题：<strong>一个 scale 管多大一片数？</strong>这叫量化粒度（granularity），
              是误差和开销之间的那个旋钮：
            </>,
          )}
        </p>
        <ul>
          <li>
            {t(
              <>
                <strong>per-tensor</strong>: the whole tensor shares one scale. Lowest overhead, but one
                abnormally large value drags every other number onto a coarse grid;
              </>,
              <>
                <strong>per-tensor</strong>：整个张量共用 1 个 scale。开销最小，但只要张量里冒出一个异常大的值，
                其余所有数都得跟着用粗格距；
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>per-channel</strong>: one scale per output channel. Each row of the matrix sets its own
                ruler, so rows stop dragging each other down, and the overhead is still negligible;
              </>,
              <>
                <strong>per-channel</strong>：每个输出通道一个 scale。矩阵每一行自己定标尺，
                行与行不再互相拖累，开销仍可忽略；
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                <strong>per-group (g=128)</strong>: one scale per 128 weights, the standard for INT4 schemes
                like GPTQ and AWQ. An FP16 scale amortized over 128 weights costs just 0.125 bits/weight, and
                it shrinks an outlier's blast radius to the 128 numbers around it.
              </>,
              <>
                <strong>per-group（g=128）</strong>：每 128 个权重一个 scale，是 GPTQ、AWQ 这类 INT4
                方案的标配。一个 FP16 scale 摊到 128 个权重上只多 0.125 位/权重，
                却把异常值的破坏半径压到了它周围的 128 个数以内。
              </>,
            )}
          </li>
        </ul>
        <p>
          {t(
            <>
              Finer granularity means smaller error and more metadata, but a scale costs so little to store
              that INT4 in practice almost always goes per-group. The real adversary isn't overhead. It's the
              star of the next section: the outlier.
            </>,
            <>
              粒度越细，误差越小、元数据越多，但 scale 的存储开销实在太小，所以 INT4
              实践中几乎清一色用 per-group。真正的对手不是开销，而是下一节的主角：outlier。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={4}
        title={t('Quantization-error playground: how outliers wreck everything', '量化误差实验场：outlier 如何搞砸一切')}
        lead={t(
          "absmax's fatal flaw is right there in the name: it hands the ruler to the single largest number in the tensor.",
          'absmax 的命门就写在名字里：它把标尺交给了张量里最大的那一个数。',
        )}
      >
        <p>
          {t(
            <>
              The playground below holds 4096 Gaussian-distributed "weights"; you can inject a small handful of
              outliers and dial their magnitude. Start with the default: under INT8-absmax, 0.5% of ×12σ
              outliers already knocks the SNR down a notch. Now drag the magnitude slider to ×30 and watch the
              grid (the vertical lines). absmax chases the maximum, so <strong>one number stretches the spacing
              more than tenfold</strong>, and inside the ±4σ body you're left with a sparse few volt-colored
              grid points, the rest rose-colored ornaments with no values to land on. Switch to INT4-absmax and
              nearly all 7 positive grid points go to waste, the body smearing into a blur. Then switch to INT4
              g=128. Same INT4, but the outlier is locked inside its own group of 128, and the SNR climbs right
              back up.
            </>,
            <>
              下面的实验场里有 4096 个服从高斯分布的「权重」，你可以注入一小撮
              outlier（异常值）并调它们的幅度。先看默认状态：INT8-absmax 下注入 0.5% 的
              ×12σ outlier，SNR（信噪比）就已经掉了一截。再把幅度滑杆拉到
              ×30，看格点（竖线）：absmax 跟着最大值跑，<strong>一个数就把格距撑大十几倍</strong>，
              主体 ±4σ 内只剩零星几条 volt 色格点，其余全是接不到值的 rose 色摆设。
              切到 INT4-absmax，7 个正格点几乎全废，主体糊成一团。
              再切 INT4 g=128，还是 INT4，但 outlier 被锁在自己那 128 个数的小组里，SNR 又爬了回来。
            </>,
          )}
        </p>
        <QuantErrorLab />
        <Callout type="insight" title={t("Quantization's enemy is the outlier, not the bit-width", '量化的敌人是 outlier，不是位数')}>
          <p>
            {t(
              <>
                Dropping from INT8 to INT4 loses 4 bits, worth about 24 dB of SNR in theory. A single ×30σ
                outlier under absmax compresses the effective grid 30-fold and sheds precision of the same
                order, so <strong>one number ruins one tensor</strong>. That's why modern quantization methods
                aren't in the business of inventing a denser grid; they're in the business of deciding where to
                put the outlier: isolate it, smooth it out, or protect it.
              </>,
              <>
                从 INT8 到 INT4 丢 4 位，理论上 SNR 掉约 24 dB。而一个 ×30σ 的 outlier
                在 absmax 下能把有效格点压缩 30 倍，损失的精度也是同一个量级，{' '}
                <strong>一个数就毁掉一个张量</strong>。所以现代量化方法做的不是「发明更密的格点」，
                而是「把 outlier 安置到哪」：隔离它、抹平它，或者保护它。
              </>,
            )}
          </p>
        </Callout>
        <p>
          {t(
            <>
              The catch is that outliers in real LLMs aren't artificial noise like in this lab. They're a{' '}
              <strong>systematic phenomenon</strong>. Dettmers, author of LLM.int8(), found that above the 6.7B
              mark, activations grow outliers tens of times larger than everything else in a fixed handful of
              feature dimensions, and those dimensions turn out to be the ones model quality hangs on. The fix
              is <strong>mixed-precision decomposition</strong>: pull that tiny minority (~0.1%) of outlier
              channels out and keep computing them in FP16, while the other 99.9% run through INT8. Virtually
              lossless, paid for with a more complex kernel and online outlier detection.
            </>,
            <>
              麻烦在于，真实 LLM 里的 outlier 不是实验里这种人造噪声，而是<strong>系统性现象</strong>。
              LLM.int8() 的作者 Dettmers 发现：模型一过 6.7B，激活（activation）就会在固定的少数几个特征维度上
              长出比其他值大几十倍的 outlier，而这几个维度偏偏是模型质量的命门。解法是
              <strong>混合精度分解</strong>：把这极少数（约 0.1%）outlier 通道挑出来，继续用 FP16
              算，其余 99.9% 走 INT8。精度几乎无损，代价是 kernel 更复杂、还得在线检测。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              SmoothQuant takes a different angle. Activation outliers are hard to quantize while the weight
              distribution is flat and easy, so apply the mathematically equivalent transform{' '}
              <MathTex tex="Y=(X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W)" />: divide the activations
              channel-wise by a smoothing factor and fold that same factor into the weights,{' '}
              <strong>migrating the quantization difficulty from activations onto weights</strong>. Once both
              sides are equally unremarkable, W8A8 runs integer matmul end to end and its throughput comes
              straight off the INT8 Tensor Cores.
            </>,
            <>
              SmoothQuant 换了个角度。激活的 outlier 难量化，权重分布平整好量化，于是做个数学等价变换{' '}
              <MathTex tex="Y=(X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W)" />：
              把激活按通道除以一个平滑系数，再把同一个系数折进权重，
              <strong>把量化难度从激活挪到权重</strong>。两边都变得平平无奇之后，W8A8
              全程整数矩阵乘，吞吐直接吃 INT8 Tensor Core 的红利。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              AWQ (Activation-aware Weight Quantization) goes after the weight-only setting. It notices that
              roughly 1% of weight channels sit on the thoroughfares that activation outliers travel through,
              and botching the quantization of those channels does the most damage. Rather than keep them in
              FP16, AWQ scales the important channels up by an amplification factor before quantizing, which is
              equivalent to handing those channels a finer grid while the hardware still sees one clean, uniform
              INT4 format. All three routes land in the same place:{' '}
              <strong>figure out what matters, then stop absmax from painting everything with one brush.</strong>
            </>,
            <>
              AWQ（Activation-aware Weight Quantization）盯的是 weight-only 场景。它注意到权重里约 1%
              的通道正坐在激活 outlier 经过的要道上，量化砸了这些通道伤害最大。AWQ
              不把它们留成 FP16，而是给重要通道先乘一个放大系数再量化，
              等于偷偷给这些通道更细的格距，而硬件看到的仍是一个干净统一的 INT4 格式。
              三条路线落到同一处：<strong>先弄清谁重要，再别让 absmax 一刀切。</strong>
            </>,
          )}
        </p>
      </Section>

      <Section
        index={5}
        title={t('Why quantization = speedup: moving less is faster', '为什么量化 = 加速：搬得少就是快')}
        lead={t(
          'Back to the roofline: decode lives at the far left of the ramp, where bandwidth is everything.',
          '回到 roofline：decode 待在斜坡最左端，那里带宽就是一切。',
        )}
      >
        <p>
          {t(
            <>
              The first instinct about quantization speedup is usually "integer multiply is fast," and in the
              decode setting that instinct is wrong. Decoding one token reads the entire weight set once but
              does only ~2 floating-point ops per weight, an arithmetic intensity of 1–2 FLOPs/Byte. That's two
              orders of magnitude short of the A100's ridge point near 100, so GPU compute utilization often
              sits under 5%. <strong>Here the performance formula collapses to: tok/s ≈ bandwidth ÷ weight
              bytes.</strong> Halve the denominator, double the speed. That's the entire source of INT8 ≈ 2×
              and INT4 ≈ 4×. The mainstream W4A16 schemes (GPTQ and AWQ both fit) store weights as INT4, and the
              moment the kernel pulls a weight block on-chip it <strong>dequantizes back to FP16 right
              there</strong>, with the multiply-accumulate still running on FP16 Tensor Cores. HBM traffic is
              billed at INT4, math precision at FP16, and you win on both ends.
            </>,
            <>
              对量化加速的第一直觉通常是「整数乘法快」，在 decode 场景下这个直觉是错的。decode
              每生成一个 token，要把全部权重读一遍，却只对每个权重做约 2 次浮点运算，
              算术强度 1~2 FLOPs/Byte。这离 A100 约 100 的脊点（ridge point）差了两个数量级，
              GPU 的算力利用率常常不到 5%。<strong>此时性能公式坍缩成：tok/s ≈ 带宽 ÷ 权重字节数。</strong>
              分母砍半，速度翻倍，这就是 INT8 ≈ 2×、INT4 ≈ 4× 的全部来源。
              主流的 W4A16 方案（GPTQ、AWQ 都算）把权重以 INT4 存放，kernel 把权重块搬进片上后立刻
              <strong>就地反量化回 FP16</strong>，乘加照样走 FP16 Tensor Core。HBM 流量按 INT4 计，
              数学精度按 FP16 算，两头都占便宜。
            </>,
          )}
        </p>
        <MemoryCalcLab />
        <p>
          {t(
            <>
              Use the calculator to check the numbers from the opening: 70B + FP16 = 140 GB, which fits neither
              the 24G nor the 80G card, with a theoretical ceiling around 14 tok/s (and you'd need 175G of VRAM
              before speed is even a conversation). Switch to INT4 and only 35 GB remains, fitting a single 80G
              card, with the ceiling jumping to ~57 tok/s. Be honest about where this stops, though: the
              speedup belongs to the memory-bound regime alone. Prefill and large-batch serving are
              compute-bound, where weight-only quantization buys you nothing (dequantization even adds a touch
              of overhead). Those want schemes like W8A8/FP8 that lower the precision of the computation itself,
              which is home turf for SmoothQuant and H100 FP8.
            </>,
            <>
              用计算器核一下开篇的数字：70B + FP16 = 140 GB，24G 和 80G 卡都放不下，理论上限
              约 14 tok/s（你还得先凑出 175G 显存，才轮得到谈速度）；切到 INT4 只剩 35 GB，单张 80G
              卡放得下，上限直接冲到 ~57 tok/s。但得把边界说清楚：这份加速只属于
              memory-bound 的场景。prefill 阶段和大 batch 服务是 compute-bound 的，weight-only
              量化帮不上忙（反量化甚至还添点开销）。那里要的是 W8A8/FP8 这类让计算本身也降精度的方案，
              正是 SmoothQuant 和 H100 FP8 的主场。
            </>,
          )}
        </p>
        <Quiz
          question={t('INT4 weight-only quantization speeds up decode nearly 4×. Why?', 'INT4 weight-only 量化能把 decode 加速近 4 倍，原因是什么？')}
          options={[
            {
              text: t('INT4 multipliers are 4× faster than FP16 multipliers', 'INT4 乘法器比 FP16 乘法器快 4 倍'),
              explain: t(
                "No: in W4A16 the multiply is still FP16 (the kernel dequantizes first). And even with an integer multiply, decode's compute utilization starts at a few percent, so computing faster fixes nothing.",
                '不对：W4A16 里乘法还是 FP16 做的（kernel 里先反量化）。就算改成整数乘法，decode 的算力利用率本来就只有几个百分点，算得再快也没用。',
              ),
            },
            {
              text: t('The weight bytes streamed from HBM per token drop to 1/4, and decode is memory-bound', '每个 token 要从 HBM 搬的权重字节数变成了 1/4，而 decode 是 memory-bound'),
              correct: true,
              explain: t(
                'Right. Decode has an arithmetic intensity of 1–2 FLOPs/Byte, so the time goes almost entirely into moving weights: tok/s ≈ bandwidth ÷ weight bytes. Cut bytes to a quarter and per-token transfer time drops to a quarter. The bottleneck is moving bytes, not computing.',
                '对。decode 算术强度只有 1~2 FLOPs/Byte，时间几乎全砸在搬权重上：tok/s ≈ 带宽 ÷ 权重字节。字节砍到四分之一，每 token 的搬运时间也砍到四分之一。瓶颈是搬字节，不是算。',
              ),
            },
            {
              text: t('Quantization deletes unimportant parameters, making the model smaller', '量化删掉了不重要的参数，模型变小了'),
              explain: t(
                'Quantization deletes no parameters. A quantized 70B still has 70B numbers, each just shrunk from 16 bits to 4. Deleting parameters is pruning.',
                '量化不删参数。70B 量化后还是 70B 个数，只是每个数从 16 位变成 4 位。删参数那叫剪枝（pruning）。',
              ),
            },
            {
              text: t('INT4 data lets the GPU run at a higher clock frequency', 'INT4 数据能让 GPU 跑在更高的频率上'),
              explain: t(
                'Frequency has nothing to do with the data format. The speedup comes entirely from smaller HBM traffic, the direct payoff under a bandwidth bottleneck.',
                '频率和数据格式无关。加速完全来自 HBM 流量变小，这是带宽瓶颈下的直接收益。',
              ),
            },
          ]}
        />
      </Section>

      <Section
        index={6}
        title={t('A glance at training: mixed precision and FP8', '训练侧一瞥：混合精度与 FP8')}
        lead={t(
          'Inference quantization is about storing it small; training in low precision is about computing it fast without blowing up.',
          '推理量化讲的是「存得小」，训练降精度讲的是「算得快、还不能炸」。',
        )}
      >
        <p>
          {t(
            <>
              The standard training playbook is <strong>mixed precision</strong>: run forward and backward in
              BF16 to saturate the Tensor Cores (on the A100, BF16 has 16× the FP32 throughput), while keeping a
              separate <strong>FP32 master copy</strong> of every parameter for the update alone. Why keep that
              FP32 copy? The weight update <MathTex tex="\eta\cdot g" /> is often 4–6 orders of magnitude
              smaller than the weight itself, and BF16 has only 7 mantissa bits, so once the update drops below
              2⁻⁸ of the weight, <MathTex tex="w + \eta g" /> rounds right back to <MathTex tex="w" />: the
              weight doesn't budge and training silently stalls. Only FP32's 23 mantissa bits can accumulate
              these tiny updates bit by bit.
            </>,
            <>
              训练的标准玩法是<strong>混合精度（mixed precision）</strong>：前向、反向用 BF16
              算以吃满 Tensor Core（A100 上 BF16 是 FP32 的 16 倍吞吐），同时给每个参数另留一份{' '}
              <strong>FP32 master 权重</strong>专门做更新。为什么非留这份 FP32 副本不可？权重更新量{' '}
              <MathTex tex="\eta\cdot g" /> 往往比权重本身小 4~6 个数量级，而 BF16 只有 7
              位尾数，一旦更新量掉到权重的 2⁻⁸ 以下，<MathTex tex="w + \eta g" /> 四舍五入又回到了{' '}
              <MathTex tex="w" />：权重纹丝不动，训练悄没声地停滞。只有 FP32 的 23
              位尾数才接得住这些微小更新，一点点把它们攒起来。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              On FP16 (range only to 65504, minimum normal 6×10⁻⁵) you also need <strong>loss scaling</strong>.
              Before backprop, multiply the loss by a large factor (say 1024) so every gradient scales up
              proportionally by the chain rule, lifting it out of FP16's underflow zone (anything &lt;2⁻²⁴
              flushes to 0) and back into representable range; then divide it back out before the update. The
              factor adjusts on the fly: the moment a gradient turns Inf/NaN, skip the step and halve the
              factor. BF16 shares FP32's range and skips this whole routine, which is exactly how it earned its
              spot as the default training format.
            </>,
            <>
              用 FP16 时（范围只到 65504、最小正规值 6×10⁻⁵）还得加{' '}
              <strong>loss scaling</strong>：反向传播前把 loss 乘一个大系数（比如
              1024），所有梯度按链式法则等比放大，从 FP16 的下溢区（&lt;2⁻²⁴
              直接变 0）抬回可表示范围，更新前再除回去。系数随时动态调：梯度一出现
              Inf/NaN 就跳过这步、把系数减半。BF16 的范围同 FP32，整套体操都免了，它就是这样坐稳训练默认格式的。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              One rung lower is <strong>FP8 training</strong>. The H100's Transformer Engine (TE) stores
              forward-pass weights and activations in E4M3 and backward gradients in E5M2, with FP8 Tensor Cores
              delivering about 2× the throughput of BF16 (~1979 vs 989 TFLOPS on the H100; the Blackwell
              generation pushes low-precision throughput further still). FP8's dynamic range is too narrow to
              leave unmanaged, so TE keeps a <strong>per-tensor scaling factor</strong> for every tensor: it
              tracks the amax (maximum absolute value) history over the last several steps, predicts a sensible
              scale for the next one, and centers the tensor inside FP8's representable window. Accumulation
              still happens in FP32 and the master weights stay put. What gets dropped in precision is always
              the compute; the accumulate is the one part nobody dares skimp on.
            </>,
            <>
              再往下一档是 <strong>FP8 训练</strong>。H100 的 Transformer Engine（TE）用 E4M3
              存前向的权重和激活、用 E5M2 存反向梯度，FP8 Tensor Core 吞吐约为
              BF16 的 2 倍（H100 上约 1979 vs 989 TFLOPS；到了 Blackwell 这一代，低精度吞吐还会再往上推）。FP8
              的动态范围太窄，撒手不管会出事，所以 TE 给每个张量维护一个{' '}
              <strong>per-tensor scaling factor</strong>：记录最近若干步的 amax（绝对值最大值）历史，
              推算下一步合适的缩放，把张量居中塞进 FP8 的可表示窗口里。累加仍在
              FP32 里做，master 权重照旧。降精度的永远是「算」那一段，
              「积累」那一段谁都不敢省。
            </>,
          )}
        </p>
        <Callout type="deep" title={t('FP8 vs INT8: same 8 bits, different jobs', 'FP8 与 INT8：同是 8 位，分工不同')}>
          <p>
            {t(
              <>
                INT8's uniform grid, paired with absmax/zero-point, suits the stably-distributed weights of{' '}
                <strong>inference</strong>. FP8's built-in logarithmic grid and exponent bits are far more
                forgiving of <strong>training</strong> tensors whose dynamic range swings violently, gradients
                above all. So the rough division of labor is BF16/FP8 for training, INT8/INT4/FP8 for inference.
                The format follows the tensor's statistical temperament.
              </>,
              <>
                INT8 格点均匀，配上 absmax/zero-point 适合分布稳定的<strong>推理</strong>权重；FP8
                自带对数格点和指数位，对动态范围剧烈摆动的<strong>训练</strong>张量（尤其是梯度）
                宽容得多。所以大致的分工是：训练用 BF16/FP8，推理用 INT8/INT4/FP8。格式跟着张量的统计脾性走。
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
                bits buy precision. BF16 trades FP16's mantissa precision for FP32's exponent range, and FP8's
                E4M3/E5M2 replay that trade-off at 8 bits.
              </>,
              <>
                浮点 = 符号 + 指数（量程）+ 尾数（刻度），<MathTex tex="(-1)^s\times 1.m\times 2^{e-\text{bias}}" />；
                指数位买范围，尾数位买精度。BF16 拿 FP16 的尾数精度换 FP32 的指数范围，FP8 的 E4M3/E5M2
                把同样的取舍搬到 8 位上重演。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Integer quantization maps reals onto a uniform grid: <MathTex tex="q=\mathrm{round}(x/s)" />,
                with absmax setting the scale. Granularity runs per-tensor → per-channel → per-group(128), error
                shrinking and overhead creeping up.
              </>,
              <>
                整数量化把实数映射到均匀格点：<MathTex tex="q=\mathrm{round}(x/s)" />，absmax 定
                scale。粒度从 per-tensor → per-channel → per-group(128)，误差递减、开销微增。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Quantization's number-one enemy is the outlier: a single anomaly blows out absmax's grid
                spacing. LLM.int8() isolates outlier channels into FP16, SmoothQuant migrates the difficulty from
                activations to weights, and AWQ hands important channels a finer effective grid.
              </>,
              <>
                量化的头号敌人是 outlier：一个异常值就能把 absmax 的格距撑爆。LLM.int8()
                把 outlier 通道隔离成 FP16，SmoothQuant 把难度从激活搬到权重，AWQ 给重要通道更细的等效格距。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Quantization = speedup because decode is memory-bound: tok/s ≈ bandwidth ÷ weight bytes. INT4
                moves a quarter of FP16's bytes, hence ~4× faster. Less moving, not faster multiplying.
              </>,
              <>
                量化 = 加速，是因为 decode 是 memory-bound：tok/s ≈ 带宽 ÷ 权重字节。INT4 搬的字节是
                FP16 的四分之一，所以快约 4 倍。是搬得少了，不是乘得快了。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                On the training side: BF16 compute plus an FP32 master weight to accumulate tiny updates; FP16
                needs loss scaling to keep gradients from underflowing; the H100 Transformer Engine uses
                per-tensor scaling to fit FP8 into training.
              </>,
              <>
                训练侧：BF16 算 + 一份 FP32 master 权重攒微小更新；FP16 要靠 loss scaling 防梯度下溢；H100
                的 Transformer Engine 用 per-tensor scaling 把 FP8 塞进训练。
              </>,
            )}
          </li>
        </ul>
        <p>{t('To go deeper, these primary sources are the ones you can\'t skip:', '想再深挖，这几篇原始文献绕不开：')}</p>
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
