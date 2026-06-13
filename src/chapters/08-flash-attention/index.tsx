import { Callout, CodeBlock, MathTex, Quiz, Section, Term } from '@/components/ui'
import { useT } from '@/lib/i18n'
import { OnlineSoftmaxLab } from './OnlineSoftmaxLab'
import { TiledAttentionLab } from './TiledAttentionLab'

const FLASH_PSEUDO_EN = `# FlashAttention forward (FA2 loop order, single head, simplified notation)
# Q, K, V: [S, d]. Q is split into Tr blocks (Br rows each), K/V into Tc blocks (Bc rows each).

for i in range(Tr):                        # outer: walk the Q blocks (rows are independent -> parallel)
    Qi = load(Q[i])                        # HBM -> SRAM, stays resident for the whole inner loop
    m  = full(Br, -inf)                    # running max, per row
    l  = zeros(Br)                         # running denominator, per row
    Oi = zeros(Br, d)                      # output accumulator (un-normalized)

    for j in range(Tc):                    # inner: stream the K/V blocks through SRAM
        Kj, Vj = load(K[j]), load(V[j])    # HBM -> SRAM, discarded once consumed
        Sij = Qi @ Kj.T / sqrt(d)          # tiny score block: lives only in SRAM, never written back!
        m_new = maximum(m, rowmax(Sij))
        P     = exp(Sij - m_new)           # local (un-normalized) softmax
        alpha = exp(m - m_new)             # rescale factor for the old state
        l  = l * alpha + rowsum(P)
        Oi = Oi * alpha[:, None] + P @ Vj  # merge into the output accumulator, online
        m  = m_new

    store(O[i], Oi / l[:, None])           # normalize once at the end of the row; one write-back per block`

const FLASH_PSEUDO_ZH = `# FlashAttention 前向（FA2 的循环顺序，单头、简化记号）
# Q, K, V: [S, d]；Q 切成 Tr 块（每块 Br 行），K/V 切成 Tc 块（每块 Bc 行）

for i in range(Tr):                        # 外层：遍历 Q 块（行间独立，可并行）
    Qi = load(Q[i])                        # HBM -> SRAM，整段内层循环期间驻留
    m  = full(Br, -inf)                    # 每行的运行 max
    l  = zeros(Br)                         # 每行的运行分母
    Oi = zeros(Br, d)                      # 输出累计（未归一化）

    for j in range(Tc):                    # 内层：让 K/V 块流过 SRAM
        Kj, Vj = load(K[j]), load(V[j])    # HBM -> SRAM，用完即弃
        Sij = Qi @ Kj.T / sqrt(d)          # 小块 score：只活在 SRAM，从不写回！
        m_new = maximum(m, rowmax(Sij))
        P     = exp(Sij - m_new)           # 局部（未归一化）softmax
        alpha = exp(m - m_new)             # 旧状态的修正因子
        l  = l * alpha + rowsum(P)
        Oi = Oi * alpha[:, None] + P @ Vj  # 在线合并进输出累计
        m  = m_new

    store(O[i], Oi / l[:, None])           # 行尾统一归一化，整块只写回一次`

export default function Chapter() {
  const t = useT()
  return (
    <>
      <p>
        {t(
          <>
            Replay every floating-point multiply-add of an algorithm faithfully, yet run it 2–4× faster
            with memory dropping from quadratic to linear — it sounds like cheating, but that is exactly
            what FlashAttention does. No approximation, no pruning, no precision loss: its output is
            mathematically identical to standard attention. What it optimizes is not the arithmetic but the{' '}
            <strong>movement</strong> of data. The roofline model from Chapter 6 told us that when a
            kernel's arithmetic intensity sits on the memory-bound side of the wall, speed is decided not
            by how much you compute, but by how many bytes you shuttle between HBM and the chip. Naive
            attention is a data-movement disaster — and the culprit is that S×S intermediate matrix nobody
            ever actually wanted to <em>see</em>.
          </>,
          <>
            把一个算法的每一次浮点乘加都原封不动地照做，却让它快上 2~4 倍、显存占用从平方级降到线性
            —— 这听起来像作弊，但 FlashAttention 干的就是这件事。它没有近似、没有剪枝、没有降低精度，
            输出和标准 attention 在数学上完全一致。它优化的根本不是计算，而是<strong>搬运</strong>：
            第六章的 roofline 模型告诉我们，当一个 kernel 的算术强度（arithmetic
            intensity）落在内存墙左侧时，决定速度的不是你算了多少，而是你在 HBM
            和芯片之间搬了多少字节。naive attention 恰好是个搬运灾难 ——
            而这场灾难的元凶，是那个从来没人真正想「看到」的 S×S 中间矩阵。
          </>,
        )}
      </p>

      <Section
        index={1}
        title={t('Three HBM round-trips: naive attention’s hidden bill', '三次 HBM 往返：naive attention 的隐藏账单')}
        lead={t(
          'The problem is not the two matmuls, but the quadratic matrix wedged between them — one that nobody needs yet gets written out in full.',
          '问题不在两个 matmul，而在它们中间那块从未被需要、却被完整写出来的平方级矩阵。',
        )}
      >
        <p>{t('First, the formula from Chapter 7. Single-head attention computes:', '先把第七章的公式摆出来。单头 attention 做的事是：')}</p>
        <MathTex block tex="\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V" />
        <p>
          {t(
            <>
              The most direct implementation chains three separate kernels. Step one computes the score
              matrix <MathTex tex="S = QK^{\top}/\sqrt{d}" /> and writes the result back to{' '}
              <Term t="HBM">
                High Bandwidth Memory, the GPU's device memory. Large (80GB-class) but relatively "far":
                H100 bandwidth is about 3.35TB/s, with access latency of hundreds of cycles.
              </Term>
              . Step two runs softmax over each row of S — reading S in, computing, then writing the
              probability matrix P back out. Step three computes <MathTex tex="O = PV" />, reading P in full
              once more. Note that S and P have shape [S, S] (S is the sequence length) — independent of
              head dim, growing <strong>quadratically</strong> with sequence length, and every head keeps
              its own copy.
            </>,
            <>
              最直接的实现方式是三个独立 kernel 串起来：第一步算 score 矩阵{' '}
              <MathTex tex="S = QK^{\top}/\sqrt{d}" />，把结果写回{' '}
              <Term t="HBM">High Bandwidth Memory，GPU 的显存。容量大（80GB 级）但相对「远」：H100 带宽约
              3.35TB/s，访问延迟数百周期。</Term>
              ；第二步对 S 逐行做 softmax，把 S 读进来、算完、再把概率矩阵 P 写回去；第三步算{' '}
              <MathTex tex="O = PV" />，又把 P 完整读一遍。注意 S 和 P 的形状是 [S, S]（S
              是序列长度）—— 它和 head dim 无关，随序列长度<strong>平方增长</strong>，
              而且每个头都有自己的一份。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Plug in real numbers to feel it. Take S=8192, 32 heads, fp16: one head's score matrix is
              8192×8192×2B = 128MiB, and 32 heads is <strong>4GiB</strong> — and that is just "store it
              once." The three-step flow makes it round-trip through HBM three times:
            </>,
            <>
              代入真实数字感受一下。取 S=8192、32 个头、fp16：一个头的 score 矩阵是 8192×8192×2B =
              128MiB，32 个头就是 <strong>4GiB</strong> —— 这只是「存一遍」的体积。
              而上面的三步流程要让它在 HBM 上往返三次：
            </>,
          )}
        </p>
        <div className="my-5 overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-line text-left">
                <th className="microlabel py-2 pr-3 font-normal">KERNEL</th>
                <th className="microlabel py-2 pr-3 font-normal">{t('HBM READ', 'HBM 读')}</th>
                <th className="microlabel py-2 pr-3 font-normal">{t('HBM WRITE', 'HBM 写')}</th>
                <th className="microlabel py-2 font-normal">{t('OF WHICH O(S²)', '其中 O(S²) 项')}</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[12.5px] tabular-nums">
              <tr className="border-b border-line/60">
                <td className="py-2 pr-3 text-text">S = QKᵀ/√d</td>
                <td className="py-2 pr-3 text-cyan">Q+K ≈ 0.13 GiB</td>
                <td className="py-2 pr-3 text-amber">S = 4 GiB</td>
                <td className="py-2 text-rose">4 GiB</td>
              </tr>
              <tr className="border-b border-line/60">
                <td className="py-2 pr-3 text-text">P = softmax(S)</td>
                <td className="py-2 pr-3 text-cyan">S = 4 GiB</td>
                <td className="py-2 pr-3 text-amber">P = 4 GiB</td>
                <td className="py-2 text-rose">8 GiB</td>
              </tr>
              <tr className="border-b border-line/60">
                <td className="py-2 pr-3 text-text">O = PV</td>
                <td className="py-2 pr-3 text-cyan">P+V ≈ 4.06 GiB</td>
                <td className="py-2 pr-3 text-amber">O ≈ 0.06 GiB</td>
                <td className="py-2 text-rose">4 GiB</td>
              </tr>
              <tr>
                <td className="py-2 pr-3 text-ink">{t('total', '合计')}</td>
                <td className="py-2 pr-3 text-cyan">≈ 8.2 GiB</td>
                <td className="py-2 pr-3 text-amber">≈ 8.1 GiB</td>
                <td className="py-2 font-semibold text-rose">{t('16 GiB / layer / fwd', '16 GiB / 层 / 前向')}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          {t(
            <>
              16GiB of HBM traffic, at H100's 3.35TB/s, takes about 5ms. And the compute for this one
              attention layer? The two matmuls total roughly <MathTex tex="4S^2 d h \approx 1.1" /> TFLOPs,
              which 989 TFLOPS of Tensor Core chews through in about 1.1ms. <strong>1ms of work, dressed
              up with 5ms of data movement</strong> — textbook memory-bound. Worse still is the softmax in
              the middle: it does only one or two exponentials and adds per byte, an arithmetic intensity
              of about 1 FLOP/Byte, pinned to the far-left floor of the roofline. As long as S still makes
              a round-trip through HBM, this bill is unavoidable — and it gets quadratically worse with
              sequence length.
            </>,
            <>
              16GiB 的 HBM 流量，在 H100 的 3.35TB/s 带宽下要走约 5ms。而这一层 attention
              的总计算量呢？两个 matmul 合计约 <MathTex tex="4S^2 d h \approx 1.1" /> TFLOPs，989
              TFLOPS 的 Tensor Core 大约 1.1ms 就能算完。<strong>算 1ms 的活，配了 5ms 的搬运</strong>
              —— 典型的 memory-bound。更糟的是中间那步 softmax：它对每个字节只做一两次指数和加法，
              算术强度约为 1 FLOP/Byte，在 roofline 图上贴着最左边的地板。只要 S
              矩阵还要走 HBM 一个来回，这笔账就躲不掉，而且随序列长度平方地恶化。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              Once you see this, the optimization goal is clear: <strong>can we avoid writing S out at
              all?</strong> The two matmuls are easy to please — Chapter 5 showed how tiling lets a matmul
              saturate the hardware. The real holdout is the softmax wedged in between: it normalizes over
              an <strong>entire row</strong>, so it seems to inherently need a complete row of scores before
              it can move. Untying that knot is the next section's job.
            </>,
            <>
              看清这一点，优化目标就明确了：<strong>能不能根本不把 S 写出来？</strong>
              两个 matmul 是好伺候的 —— 第五章讲过分块（tiling）可以让 matmul 吃满算力。
              真正的钉子户是夹在中间的 softmax：它要对<strong>整行</strong>做归一化，
              看起来天生需要先拿到完整的一行 score 才能动手。解开这个结，是下一节的事。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={2}
        title={t('The softmax problem, and online softmax', 'softmax 的麻烦，与 online softmax')}
        lead={t(
          'Safe softmax needs the whole row’s max and sum — seemingly un-tileable, until you let yourself "fix the books" after the fact.',
          '安全 softmax 需要整行的 max 和 sum —— 看似不可分块，直到你允许自己「事后修账」。',
        )}
      >
        <p>
          {t(
            <>
              First, why softmax must be "safe." Computing <MathTex tex="e^{x_i}/\sum_j e^{x_j}" /> straight
              from the definition blows up in floating point: fp16's max is about 65504, and{' '}
              <MathTex tex="e^{x}" /> overflows once x exceeds about 11 (even fp32 only holds to x≈88). So in
              practice everyone first subtracts the row max:
            </>,
            <>
              先说为什么 softmax 必须「安全」（safe softmax）。直接按定义算{' '}
              <MathTex tex="e^{x_i}/\sum_j e^{x_j}" /> 在浮点数里会爆：fp16 的最大值约
              65504，<MathTex tex="e^{x}" /> 在 x 超过 11 左右就溢出了（fp32 也只撑到 x≈88）。
              所以实践中一律先减去整行最大值：
            </>,
          )}
        </p>
        <MathTex block tex="\mathrm{softmax}(x)_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j" />
        <p>
          {t(
            <>
              Subtracting m does not change the result (numerator and denominator are both multiplied by{' '}
              <MathTex tex="e^{-m}" />), but it guarantees the exponent's argument is never positive and
              never overflows. The price: you need <strong>two passes</strong> — the first to find the row
              max m, the second to compute <MathTex tex="e^{x_i-m}" /> and accumulate the denominator l.
              This is the sworn enemy of tiling: K/V blocks stream in one at a time, and while processing the
              first block you have no idea whether a larger score lurks ahead.
            </>,
            <>
              减 m 不改变结果（分子分母同乘 <MathTex tex="e^{-m}" />），但保证指数的参数永不为正、
              永不溢出。代价是：你需要<strong>两遍扫描</strong> —— 第一遍找整行的
              m，第二遍才能算 <MathTex tex="e^{x_i-m}" /> 并累加分母 l。这正是分块的死敌：
              K/V 块是一块一块流过来的，处理第一块时你根本不知道后面会不会冒出更大的 score。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The fix comes from a short 2018 note by NVIDIA's Milakov and Gimelshein: online softmax. The
              core idea is to turn "the row's (m, l)" into <strong>running state you can maintain
              incrementally</strong>. Suppose you have already processed some blocks and hold the max m so
              far and the running denominator l; now a new block arrives, with local max{' '}
              <MathTex tex="m_b" /> and local sum <MathTex tex="l_b" />. The merge rule is:
            </>,
            <>
              解法来自 NVIDIA 的 Milakov 和 Gimelshein 在 2018 年的一篇短文：online softmax。
              核心想法是把「整行的 (m, l)」变成<strong>可以增量维护的运行状态</strong>。
              假设已经处理了一些块，手上握着目前为止的最大值 m 和分母累计 l；现在新来一块，
              算出它的局部最大值 <MathTex tex="m_b" /> 和局部和 <MathTex tex="l_b" />。合并规则是：
            </>,
          )}
        </p>
        <MathTex
          block
          tex="\begin{aligned} m_{\mathrm{new}} &= \max(m,\; m_b) \\[2pt] l_{\mathrm{new}} &= l\cdot e^{\,m-m_{\mathrm{new}}} \;+\; l_b\cdot e^{\,m_b-m_{\mathrm{new}}} \end{aligned}" />
        <p>
          {t(
            <>
              Why is this correct? The previous l was <MathTex tex="\sum e^{x_i-m}" />, with every term
              "based at m." If the new block brings a larger max, the old basis is now too low — but the
              remedy is just to multiply the whole l by a <strong>rescale factor</strong>{' '}
              <MathTex tex="\alpha = e^{\,m-m_{\mathrm{new}}}" />: since{' '}
              <MathTex tex="e^{x_i-m}\cdot e^{\,m-m_{\mathrm{new}}}=e^{x_i-m_{\mathrm{new}}}" />, a single
              multiply shifts every historical term onto the new basis at once. This is an exact identity,
              not an approximation — floating-point error stays at the last-bit (ulp) level.
            </>,
            <>
              为什么这是对的？此前的 l 是 <MathTex tex="\sum e^{x_i-m}" />，每一项都「以 m 为基准」。
              如果新块带来了更大的最大值，旧的基准就低了 —— 但补救只需要给整个 l 乘一个
              <strong>修正因子</strong> <MathTex tex="\alpha = e^{\,m-m_{\mathrm{new}}}" />：因为{' '}
              <MathTex tex="e^{x_i-m}\cdot e^{\,m-m_{\mathrm{new}}}=e^{x_i-m_{\mathrm{new}}}" />，
              一次乘法就把所有历史项整体换到了新基准上。这是个恒等变形，不是近似 ——
              浮点误差只在最后一位有效数字（ulp）量级。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              In attention, what we ultimately want is not softmax itself but its product with V. So we play
              the same trick once more: maintain an <strong>un-normalized</strong> output accumulator{' '}
              <MathTex tex="O=\sum_j e^{x_j-m}\,v_j" />. When a new block arrives and the max is refreshed,
              the old O is multiplied by the same <MathTex tex="\alpha" /> to change basis, then the new
              block's contribution is added in:
            </>,
            <>
              attention 里我们最终要的不是 softmax 本身，而是它和 V 的乘积。同样的把戏再用一次：
              维护一个<strong>未归一化</strong>的输出累计{' '}
              <MathTex tex="O=\sum_j e^{x_j-m}\,v_j" />。新块到来、max 被刷新时，旧的 O
              也乘同一个 <MathTex tex="\alpha" /> 完成换底，然后把新块的贡献加进来：
            </>,
          )}
        </p>
        <MathTex
          block
          tex="O_{\mathrm{new}} = O\cdot e^{\,m-m_{\mathrm{new}}} \;+\; \underbrace{e^{\,x_b-m_{\mathrm{new}}}}_{\tilde P_b}\,V_b" />
        <p>
          {t(
            <>
              Normalization (dividing by l) is deferred until all blocks are processed — one division, done.
              So all the global information a full-row softmax requires gets compressed into three running
              quantities: <strong>m, l, and the O accumulator</strong>. Their size depends only on the
              block's row count and head dim, and is <strong>independent of sequence length</strong> — they
              fit comfortably in on-chip storage.
            </>,
            <>
              归一化（除以 l）被推迟到所有块处理完之后，一次除法搞定。于是整行 softmax
              所需要的全部全局信息被压缩成了三个运行量：<strong>m、l、O 累计</strong>。
              它们的大小只和块的行数、head dim 有关，跟序列长度<strong>无关</strong> ——
              完全塞得进片上存储。
            </>,
          )}
        </p>
        <Callout type="insight" title={t('FlashAttention’s mathematical core is just two moves', 'FlashAttention 的数学核心只有两招')}>
          <p>
            {t(
              <>
                <strong>Defer normalization</strong>: accumulate the un-normalized numerators (l and O)
                first, do the single division last — so the intermediate steps can be tiled and accumulated
                freely. <strong>Rescale factor</strong>: when the basis (max) changes, no problem — multiply
                every historical accumulator by <MathTex tex="e^{\,m-m_{\mathrm{new}}}" /> to retroactively
                fix the books. Together, the two moves turn softmax from "must see the whole row" into a{' '}
                <strong>streaming reduction that obeys associativity</strong> — and every step is an exact
                identity, which is why FlashAttention is an exact algorithm, not an approximation.
              </>,
              <>
                <strong>推迟归一化</strong>：先攒未归一化的分子（l 和 O），最后才做那一次除法 ——
                于是中间过程可以随便分块累加。<strong>修正因子</strong>：基准（max）变了不要紧，
                给所有历史累计量乘一个 <MathTex tex="e^{\,m-m_{\mathrm{new}}}" /> 就能追溯修账。
                两招合起来，softmax 从「必须看完整行」变成了「结合律成立的流式归约」——
                而且每一步都是精确的恒等变形，FlashAttention 因此是精确算法，不是近似。
              </>,
            )}
          </p>
        </Callout>
      </Section>

      <Section
        index={3}
        title={t('Hands-on: one pass vs two', '动手验证：单遍 vs 两遍')}
        lead={t(
          'A recurrence is easy to nod along to; you only truly believe it after stepping through it yourself. The instrument below lets you watch (m, l) grow element by element.',
          '递推公式看一遍容易点头，自己步进一次才会真信。下面这台「仪器」让你逐元素看 (m, l) 怎么长出来。',
        )}
      >
        <p>
          {t(
            <>
              The experiment below feeds the same 8 input values to both algorithms at once. The left panel
              is the textbook two-pass method: the first pass sweeps the cursor across the whole row to find
              the max, the second computes <MathTex tex="e^{x_i-m}" /> and accumulates; the right panel is
              online single-pass: the cursor reaches element i, updates (m, l) on the spot, with each update
              formula's numeric substitution laid out below the panel. Watch the key moment on the right:
              when the cursor hits a <strong>new max</strong>, the already-processed prefix cells flash amber
              — that is the instant their accumulators are multiplied by the rescale factor α and shifted to
              the new basis, and the "current estimate" below each cell shrinks simultaneously.
            </>,
            <>
              下面的实验把 8 个输入值同时喂给两种算法。左面板是教科书式的两遍法：第一遍游标扫过整行找
              max，第二遍再算 <MathTex tex="e^{x_i-m}" /> 并累加；右面板是 online 单遍：游标走到第 i
              个元素，现场更新 (m, l)，每个更新式的数值代入都摆在面板下方。注意看右侧的关键时刻：
              当游标遇到一个<strong>新的最大值</strong>，已处理的前缀格子会闪一下 amber ——
              那是它们的累计量被乘上修正因子 α、整体换底的瞬间，格子下方的「当前估计值」会同时缩小。
            </>,
          )}
        </p>
        <OnlineSoftmaxLab />
        <p>
          {t(
            <>
              A few things worth playing with: drag X[6] up to 5.0 so the new max appears at the very end of
              the scan, and you will see a cascade of large α corrections — no matter how adversarial the
              order, the result is unchanged. Hit "Randomize" to swap the data (fixed seed, reproducible),
              and the final softmax in the bottom two rows always agrees digit-for-digit, with max error
              stuck at the 1e-16 level or simply zero. Also note the step count: two-pass takes 16 steps to
              finish (two passes over the data = two HBM reads), online wraps up in 8 — and for a
              memory-bound kernel, reading once less is spending half the time.
            </>,
            <>
              值得玩的几个点：把 X[6] 拉到 5.0，让新 max 出现在扫描末尾，你会看到一连串大幅度的
              α 修正 —— 顺序再刁钻，结果也不变；按「随机一组」换数据（固定种子，可复现），
              底部两行最终 softmax 始终逐位一致，最大误差停在 1e-16 量级或干脆为
              0。另外注意步数：两遍法走完要 16 步（数据过两遍 = HBM 读两遍），online 8 步收工 ——
              对一个 memory-bound 的 kernel 来说，少读一遍就是少花一倍的时间。
            </>,
          )}
        </p>
      </Section>

      <Quiz
        question={t(
          'When merging the softmax state of two blocks online, which quantities must each block (or running prefix) keep?',
          '在线合并两个块的 softmax 状态时，每个块（或运行前缀）必须保留哪些量？',
        )}
        options={[
          {
            text: t(
              'Every score in the entire row, otherwise you cannot re-normalize',
              '整行所有的 score 值，否则没法重新归一化',
            ),
            explain: t(
              'If you keep all the scores, you have regressed to materializing the whole row. The whole point of online softmax is to compress the global information into a constant number of statistics.',
              '如果要保留所有 score，那就退化回物化整行的老路了。online softmax 的意义恰恰是把全局信息压缩成常数个统计量。',
            ),
          },
          {
            text: t(
              'The running max m, the running denominator l, and (for attention) the un-normalized output accumulator O',
              '运行最大值 m、分母累计 l，以及（算 attention 时）未归一化的输出累计 O',
            ),
            correct: true,
            explain: t(
              'Correct. m provides the rescale basis, l accumulates the denominator, O accumulates the product of the numerator and V; when a new block arrives, all three are multiplied by the rescale factor e^(m−m′) to merge. The size of these three is independent of sequence length, so they fit in SRAM.',
              '正确。m 提供换底基准，l 攒分母，O 攒分子与 V 的乘积；新块到来时三者各乘修正因子 e^(m−m′) 即可合并。这三个量的大小与序列长度无关，塞得进 SRAM。',
            ),
          },
          {
            text: t(
              'Just the denominator sum is enough; the max is an optional numerical trick',
              '只要分母 sum 就够了，max 是可有可无的数值技巧',
            ),
            explain: t(
              'Without m there is no rescale basis: the exp overflows directly (fp16 blows up at x≈11), and two sums on different bases simply cannot be added correctly. m is the precondition for the merge rule to hold.',
              '没有 m 就没有换底基准：exp 直接溢出（fp16 在 x≈11 就爆），而且两个不同基准的 sum 根本没法正确相加。m 是合并规则成立的前提。',
            ),
          },
          {
            text: t(
              'The array of individual exp values from each preceding block',
              '前面每个块各自的 exp 值数组',
            ),
            explain: t(
              'Not needed. The rescale factor acts on the accumulators — it is a single scalar multiply, and every historical term is implicitly updated by the same factor, so there is no need to keep them one by one.',
              '不需要。修正因子作用在「累计量」上，是一次标量乘法 —— 历史的每一项都被同一个因子隐式更新了，无需逐项保留。',
            ),
          },
        ]}
      />

      <Section
        index={4}
        title={t('The tiled algorithm: S lives only in SRAM', '分块算法：S 矩阵只活在 SRAM 里')}
        lead={t(
          'With a streamable softmax in hand, the rest is classic tiling: let each tiny score block be born and die on-chip.',
          '有了可流式合并的 softmax，剩下的就是经典 tiling：让每个小块 score 在片上生、在片上死。',
        )}
      >
        <p>
          {t(
            <>
              Recall the memory hierarchy from Chapter 4: HBM is "off the island," 3.35TB/s (H100);{' '}
              <Term t="SRAM">
                here meaning the SM's on-chip shared memory / L1, about 228KB per SM (H100). Small in
                capacity, but its aggregate bandwidth is an order of magnitude higher than HBM (about 19TB/s
                vs 1.9TB/s on A100), with much lower latency.
              </Term>{' '}
              is "on the island," an order of magnitude faster but only 228KB per SM. The whole engineering
              of FlashAttention is to reorganize attention so that "the S² data is only ever touched on the
              island": split Q into Tr blocks (Br rows each), K/V into Tc blocks (Bc rows each), with block
              size set precisely by the SRAM capacity M (on the order of M/4d). Then:
            </>,
            <>
              回忆第四章的存储层级：HBM 在「岛外」，3.35TB/s（H100）；
              <Term t="SRAM">这里指 SM 片上的 shared memory / L1，每个 SM 约 228KB（H100）。容量小，
              但聚合带宽比 HBM 高一个数量级（A100 上约 19TB/s vs 1.9TB/s），延迟也低得多。</Term>
              在「岛内」，快一个数量级但每个 SM 只有 228KB。FlashAttention 的全部工程就是把
              attention 重组成「只在岛内碰 S²的数据」：把 Q 切成 Tr 块（每块 Br 行）、K/V 切成 Tc
              块（每块 Bc 行），块的尺寸恰好由 SRAM 容量 M 决定（量级是 M/4d）。然后：
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The <strong>outer loop</strong> picks up one Q block into SRAM and initializes that block's
              running state (m, l, O accumulator) on-chip; the <strong>inner loop</strong> streams the K/V
              blocks one at a time: compute the small block{' '}
              <MathTex tex="S_{ij}=Q_i K_j^{\top}/\sqrt{d}" /> (which exists only in SRAM!) → take a local
              softmax of the small block → merge the contribution into (m, l, O) with the previous section's
              merge rule. Once the inner loop finishes, divide O by l, write it back to HBM, and move to the
              next Q block. Throughout, S and P — the two O(S²) matrices — <strong>never exist in HBM</strong>{' '}
              — the 16GiB of quadratic traffic from the bill above simply vanishes.
            </>,
            <>
              <strong>外层循环</strong>拿起一个 Q 块放进 SRAM，同时在片上初始化这个块的运行状态
              (m, l, O 累计)；<strong>内层循环</strong>让 K/V 块一块块流过：算小块{' '}
              <MathTex tex="S_{ij}=Q_i K_j^{\top}/\sqrt{d}" />（它只存在于 SRAM！）→ 对小块做局部
              softmax → 用上一节的合并规则把贡献并进 (m, l, O)。内层扫完，把 O 除以 l、写回
              HBM，换下一个 Q 块。整个过程中，S 和 P 这两个 O(S²) 的矩阵
              <strong>从未在 HBM 上存在过</strong> —— 上一节账单里那 16GiB 的平方级流量，直接消失。
            </>,
          )}
        </p>
        <CodeBlock
          code={t(FLASH_PSEUDO_EN, FLASH_PSEUDO_ZH)}
          lang="python"
          title={t('flash_attention_forward.py (pseudocode)', 'flash_attention_forward.py（伪代码）')}
          highlight={[12, 16, 18]}
        />
        <p>
          {t(
            <>
              Reading this pseudocode, fix on three details. First, the <code>Sij</code> line (highlighted)
              is the soul of the whole algorithm: the score block is computed and immediately consumed, its
              lifetime never exceeding a single inner iteration. Second, the rescale factor{' '}
              <code>alpha</code> acts on both l and Oi — they must change basis with the same factor, exactly
              the multiply that the amber flash in LAB 01 corresponds to. Third, write-back happens only at
              the end of a row: each O block enters and leaves HBM once, each K/V block is read once per Q
              row — all the traffic is the O(S·d) small fry, with no O(S²) term anywhere.
            </>,
            <>
              读这段伪代码时盯住三个细节。第一，<code>Sij</code> 那一行（高亮）是全算法的灵魂：score
              块算完立刻被消费，生命周期不超过一次内层迭代。第二，修正因子 <code>alpha</code>{' '}
              同时作用在 l 和 Oi 上 —— 它们必须用同一个基准换底，这正是 LAB 01 里 amber
              闪烁对应的那次乘法。第三，写回只发生在行尾：每个 O 块进出 HBM 各一次，每个 K/V
              块每个 Q 行读一次 —— 所有流量都是 O(S·d) 级的小头，没有任何 O(S²) 项。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              The cost? K/V gets re-read once per Q block row, so flash's HBM access volume is precisely{' '}
              <MathTex tex="\Theta(S^2 d^2 / M)" /> — note the SRAM capacity M in the denominator. As long as
              M is much larger than d² (which it is in reality: 228KB vs 128²×2B=32KB), it stays far below
              naive's <MathTex tex="\Theta(S^2)" />. In other words: <strong>the bigger the SRAM, the less
              K/V gets re-read</strong> — the exact same shape of trade-off we saw in Chapter 5's matmul
              tiling.
            </>,
            <>
              代价呢？K/V 要被每个 Q 块行重读一遍，所以 flash 的 HBM 访问量精确说是{' '}
              <MathTex tex="\Theta(S^2 d^2 / M)" /> —— 注意分母上的 SRAM 容量
              M。只要 M 远大于 d²（现实如此：228KB vs 128²×2B=32KB），它就远小于 naive 的{' '}
              <MathTex tex="\Theta(S^2)" />。换句话说：<strong>SRAM 越大，K/V 重读得越少</strong>，
              这个权衡我们在第五章 matmul tiling 里见过一模一样的形状。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={5}
        title={t('Watch it run: the tiled scan and the traffic ledger', '看着它跑：分块扫描与流量账本')}
        lead={t(
          'Cut the 8192×8192 score matrix into 8×8 phantom blocks; watch each one be born and die in SRAM, and the bills pull apart.',
          '把 8192×8192 的 score 矩阵切成 8×8 个影子块，看每一块怎样在 SRAM 里生灭，账单怎样拉开差距。',
        )}
      >
        <p>
          {t(
            <>
              The animation below is a frame-by-frame execution of the previous section's pseudocode. The
              central grid is the "phantom" of the score matrix — dashed outlines mean these blocks never
              materialize to HBM; the volt-highlighted one is the small block being computed in SRAM right
              now. The right panel simulates the on-chip state: the currently resident Q/K/V blocks, the
              merge progress of (m, l), and the O accumulator bar — it flashes each time a block is merged.
              The two cumulative bars at the bottom are the chapter's conclusion itself: naive's bill climbs
              as O(S²), flash's crawls as O(S·d), and by the end the gap opens past 14×.
            </>,
            <>
              下面的动画就是上一节伪代码的逐帧执行。中央网格是 score 矩阵的「影子」——
              虚线表示这些块从未物化到 HBM；volt 高亮的是此刻正在 SRAM
              里计算的小块。右侧面板模拟片上状态：当前驻留的 Q/K/V 块、(m, l)
              的合并进度、以及 O 累计条 —— 每合并一块它就闪动一次。底部的双累计条是本章的结论本身：
              naive 的账单按 O(S²) 飙，flash 的按 O(S·d) 爬，播放到底差距拉到 14 倍以上。
            </>,
          )}
        </p>
        <TiledAttentionLab />
        <p>
          {t(
            <>
              Flip the "causal mask" switch and run again: in a decoder's autoregressive attention, query i
              can only see the first i keys, so the upper-triangular blocks are <strong>skipped
              wholesale</strong> — and note this is a gift of the tiled structure: a naive implementation
              typically computes the whole S, then fills the upper triangle with −∞, computing and shuttling
              it for nothing; flash simply never launches those computations and reads at the block level, so
              traffic and time drop by nearly half again. This is also why production FlashAttention kernels
              support "structured sparsity" masks like causal and sliding-window so naturally — the skip
              decision happens inside the block loop, at almost zero cost.
            </>,
            <>
              打开「causal mask」开关再跑一遍：decoder 的自回归 attention 里，第 i 个 query
              只能看见前 i 个 key，于是上三角的块<strong>整块跳过</strong> —— 注意这是分块结构送的礼物：
              naive 实现通常把整个 S 算完再把上三角填 −∞，白算白搬；而 flash
              在块粒度上直接不发起这些计算和读取，流量和时间再省接近一半。这也是为什么生产里的
              FlashAttention kernel 对 causal、滑动窗口这类「结构化稀疏」的 mask 支持得格外自然 ——
              跳过的判断发生在块循环里，几乎零成本。
            </>,
          )}
        </p>
      </Section>

      <Section
        index={6}
        title={t('Complexity and generations: each one kills a new bottleneck', '复杂度与代际：每一代都打掉一个新瓶颈')}
        lead={t(
          'FA1 kills HBM traffic, FA2 kills scheduling and non-matmul overhead, FA3 kills the pipeline bubbles on Hopper.',
          'FA1 打掉 HBM 流量，FA2 打掉调度与非 matmul 开销，FA3 打掉 Hopper 上的流水线气泡。',
        )}
      >
        <p>
          {t(
            <>
              <strong>Memory: O(S²) → O(S).</strong> The forward pass no longer stores S and P, keeping only
              one logsumexp per row (<MathTex tex="L=m+\log l" />, shape [S]). What about the backward pass,
              which needs S? <strong>Recomputation</strong>: in the backward pass, take Q, K and the saved L
              and recompute S<sub>ij</sub> block by block on the spot. FLOPs go up, yet it runs faster —
              because the HBM reads/writes saved are worth far more than the extra compute. This is the
              worldview this course keeps returning to: on modern GPUs, <strong>compute is cheap, movement is
              expensive</strong>; trading compute for movement is almost always a win. The real payoff: at
              S=8192, a single attention layer's activations drop from GiB-scale to MiB-scale, turning
              long-context training from "impossible" into "routine."
            </>,
            <>
              <strong>显存：O(S²) → O(S)。</strong>前向不再存 S 和 P，只额外留下每行一个
              logsumexp（<MathTex tex="L=m+\log l" />，形状 [S]）。反向传播需要 S 怎么办？
              <strong>重计算（recomputation）</strong>：反向时拿着 Q、K 和存好的 L
              按块把 S<sub>ij</sub> 现场重算一遍。FLOPs 变多了，速度反而更快 ——
              因为省下的 HBM 读写比多花的计算值钱得多。这是本课反复出现的世界观：
              在现代 GPU 上，<strong>计算便宜，搬运贵</strong>；用算换搬，几乎总是赚的。
              实际收益：S=8192 时单层 attention 激活从 GiB 级降到 MiB 级，长上下文训练从「不可能」变成「常规操作」。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>FlashAttention-2 (2023): kill the scheduling waste.</strong> FA1's loop order was
              actually K/V outer, Q inner, which forced (m, l, O) to repeatedly enter and leave HBM; FA2
              swaps to this chapter's pseudocode order (Q outer, K/V inner), keeping each Q block's state
              resident on-chip the whole time. It also cuts non-matmul FLOPs — minimizing the number of
              rescales (non-matmul ops can't use the Tensor Cores, costing about 16× more per FLOP) — and
              extends parallelism from "batch × heads" to the sequence dimension, switching the inter-warp
              partition from slicing K to slicing Q (eliminating inter-warp synchronization and shared-memory
              shuffling). The result: about 70% of theoretical utilization on A100, roughly 2× faster than
              FA1.
            </>,
            <>
              <strong>FlashAttention-2（2023）：打掉调度浪费。</strong>FA1 的循环顺序其实是 K/V
              在外、Q 在内，导致 (m, l, O) 要反复进出 HBM；FA2 调换成本章伪代码的顺序（Q 外、K/V
              内），每个 Q 块的状态全程驻留片上。同时减少非 matmul FLOPs ——
              把 rescale 的次数压到最少（非 matmul 运算走不了 Tensor Core，单位 FLOP
              贵约 16 倍）；并把并行度从「batch×头数」扩展到序列维度、warp 间分工从切 K
              改成切 Q（消除 warp 间同步与共享内存中转）。结果是 A100
              上达到约 70% 的理论算力利用率，比 FA1 快约 2 倍。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              <strong>FlashAttention-3 (2024): kill the pipeline bubbles on Hopper.</strong> H100's new
              hardware went unused by FA2: TMA (Tensor Memory Accelerator) can move data asynchronously, and
              WGMMA is the new generation of asynchronous matrix instructions. FA3 uses producer-consumer
              warp specialization plus "pingpong" scheduling to overlap one warp group's softmax with another
              warp group's matmul — those sluggish exponentials in the softmax get hidden in the shadow of the
              Tensor Cores; add block-wise quantized FP8 support, and fp16 reaches about 740 TFLOPS (about 75%
              utilization), with FP8 approaching 1.2 PFLOPS. Across three generations the line is consistent:
              the algorithm is unchanged, and each one kills whichever bottleneck is most in the way on the
              hardware of the day.
            </>,
            <>
              <strong>FlashAttention-3（2024）：打掉 Hopper 上的流水线气泡。</strong>H100
              的新硬件 FA2 没用上：TMA（Tensor Memory Accelerator）能异步搬数据、WGMMA
              是新一代异步矩阵指令。FA3 用生产者-消费者的 warp 专门化加上「pingpong」调度，
              让一组 warp 的 softmax 和另一组 warp 的 matmul 重叠执行 ——
              softmax 里那些慢吞吞的指数运算被藏进了 Tensor Core 的阴影里；再加上块级量化的 FP8
              支持，fp16 达到约 740 TFLOPS（约 75% 利用率），FP8 接近 1.2 PFLOPS。
              三代下来路线一致：算法不变，每一代都把当下硬件上最碍事的那个瓶颈打掉。
            </>,
          )}
        </p>
        <p>
          {t(
            <>
              To close in the language of the roofline: naive attention is pinned to the memory-bound side
              of the wall by its O(S²) traffic — the softmax stage's arithmetic intensity of about 1
              FLOP/Byte is two orders of magnitude away from H100's ridge point of about 295 FLOPs/Byte. Once
              FlashAttention strips the S² term out of the denominator (HBM byte count), the whole kernel's
              effective arithmetic intensity climbs back to the M/d scale (hundreds of FLOPs/Byte), stepping
              over the ridge into the compute-bound region — from then on, attention's speed is dictated by
              the Tensor Cores, not the memory bus. That is the whole secret of "not one multiply removed,
              yet 2–4× faster."
            </>,
            <>
              用 roofline 的语言收个尾：naive attention 被 O(S²) 的流量按在内存墙左侧 ——
              softmax 阶段的算术强度约 1 FLOP/Byte，离 H100 约 295 FLOPs/Byte
              的脊点差两个数量级。FlashAttention 把分母（HBM 字节数）砍掉 S²
              项之后，整个 kernel 的有效算术强度回升到 M/d 量级（数百 FLOPs/Byte），
              一脚跨过脊点进入 compute-bound 区 —— 从此 attention 的速度由 Tensor Core
              说了算，而不是由内存总线说了算。这就是「一次乘法都没少做，却快了
              2~4 倍」的全部秘密。
            </>,
          )}
        </p>
      </Section>

      <Quiz
        question={t(
          'FlashAttention is 2–4× faster. What is the core quantity it reduces?',
          'FlashAttention 提速 2~4 倍，它减少的核心量是什么？',
        )}
        options={[
          {
            text: t(
              'Floating-point operations — it skips computing some of the scores',
              '浮点运算次数 —— 它跳过了一部分 score 的计算',
            ),
            explain: t(
              'Quite the opposite: not a single FLOP is saved, and the backward pass even does more because it recomputes S. In the non-causal case every block is dutifully computed, and the result is bit-for-bit equivalent to standard attention (down to ulp-level floating-point error).',
              '恰恰相反：FLOPs 一次没少，反向传播因为重计算 S 甚至更多。非 causal 时所有块都老老实实算了，结果与标准 attention 逐位等价（至 ulp 级浮点误差）。',
            ),
          },
          {
            text: t(
              'HBM reads/writes — the S×S intermediate matrix no longer materializes, killing the O(S²) memory traffic',
              'HBM 读写量 —— S×S 中间矩阵不再物化，O(S²) 的显存流量被消掉',
            ),
            correct: true,
            explain: t(
              'Correct. A naive implementation round-trips the score/probability matrix through HBM multiple times (about 16GiB/layer at S=8192, 32 heads); FlashAttention uses tiling + online softmax to keep it alive only in SRAM, dropping HBM traffic to the O(S·d) scale. For a memory-bound kernel, moving less is being faster.',
              '正确。naive 实现让 score/概率矩阵在 HBM 上往返多次（S=8192、32 头时约 16GiB/层）；FlashAttention 用分块 + online softmax 让它只活在 SRAM 里，HBM 流量降为 O(S·d) 级。memory-bound 的 kernel，少搬就是快。',
            ),
          },
          {
            text: t(
              'Softmax precision — trading speed for approximate normalization',
              'softmax 的精度 —— 用近似归一化换速度',
            ),
            explain: t(
              'No. Online softmax is an exact identity transformation (the rescale-factor basis change); in LAB 01 the max error between the two methods stays at the floating-point ulp level. FlashAttention is an exact algorithm.',
              '不是。online softmax 是精确的恒等变形（修正因子换底），LAB 01 里两法的最大误差停在浮点 ulp 量级。FlashAttention 是精确算法。',
            ),
          },
          {
            text: t(
              'Kernel launch count — fusing three kernels into one saves launch overhead',
              'kernel 启动次数 —— 三个 kernel 融合成一个，省的是 launch 开销',
            ),
            explain: t(
              'Kernel fusion does happen, but launch overhead is microsecond-scale and cannot explain a millisecond-scale gap. The real value of fusion is that the intermediate results never land in HBM — what gets reduced is still the byte count of reads/writes.',
              'kernel 融合确实发生了，但 launch 开销是微秒级的，解释不了毫秒级的差距。融合真正的价值在于中间结果不落 HBM —— 减少的还是读写字节数。',
            ),
          },
        ]}
      />

      <Section
        index={7}
        title={t('Summary and further reading', '总结与延伸阅读')}
        lead={t(
          'In one line: rewrite softmax as a streamable reduction, and attention becomes a pure tiling problem.',
          '一句话：把 softmax 改写成可流式合并的归约，attention 就变成了纯 tiling 问题。',
        )}
      >
        <ul>
          <li>
            {t(
              <>
                Naive attention's bottleneck is the O(S²) intermediate matrix making three round-trips
                through HBM (≈16GiB/layer at S=8192, 32 heads), not the compute — textbook memory-bound.
              </>,
              <>
                naive attention 的瓶颈是 O(S²) 的中间矩阵在 HBM 上三次往返（S=8192、32 头 ≈ 16GiB/层），
                而非计算量 —— 典型 memory-bound。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Online softmax compresses full-row normalization into running state (m, l): a new block
                takes the max, the old accumulator is multiplied by the rescale factor{' '}
                <MathTex tex="e^{\,m-m_{\mathrm{new}}}" /> to change basis; the output accumulator O works
                the same way. Exact, no approximation.
              </>,
              <>
                online softmax 把整行归一化压缩成运行状态 (m, l)：新块到来取 max、旧累计乘修正因子{' '}
                <MathTex tex="e^{\,m-m_{\mathrm{new}}}" /> 换底；输出累计 O 同样适用。精确，无近似。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                Tiled execution: the Q block stays resident in SRAM, the K/V blocks stream through, the small
                score blocks are born and die on-chip — HBM access{' '}
                <MathTex tex="\Theta(S^2d^2/M)" />, memory O(S²)→O(S), with recomputation restoring S in the
                backward pass.
              </>,
              <>
                分块执行：Q 块驻留 SRAM，K/V 块流过，小块 score 在片上生灭 —— HBM 访问量{' '}
                <MathTex tex="\Theta(S^2d^2/M)" />，显存 O(S²)→O(S)，反向用重计算补回 S。
              </>,
            )}
          </li>
          <li>
            {t(
              <>
                FA2 swaps the loop order, cuts non-matmul FLOPs, and partitions warps by slicing Q; FA3 uses
                TMA/WGMMA async pipelining plus FP8 — each generation kills a new bottleneck, pushing
                attention back toward compute-bound.
              </>,
              <>
                FA2 调换循环序、压非 matmul FLOPs、按 Q 切 warp 分工；FA3 用 TMA/WGMMA
                异步流水加 FP8 —— 每代打掉一个新瓶颈，把 attention 推回 compute-bound。
              </>,
            )}
          </li>
        </ul>
        <p>{t('Further reading (in recommended order):', '延伸阅读（按推荐顺序）：')}</p>
        <ul>
          <li>
            <a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noreferrer">
              FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
            </a>{' '}
            {t(
              '— Dao et al., 2022. The original paper; the IO-complexity proof is worth a close read.',
              '—— Dao 等，2022。原始论文，IO 复杂度的证明值得细读。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2307.08691" target="_blank" rel="noreferrer">
              FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
            </a>{' '}
            {t(
              '— Dao, 2023. The engineering details of loop order and warp partitioning.',
              '—— Dao，2023。循环序与 warp 分工的工程细节。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/2407.08608" target="_blank" rel="noreferrer">
              FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
            </a>{' '}
            {t(
              '— Shah et al., 2024. Hopper async pipelining and FP8.',
              '—— Shah 等，2024。Hopper 异步流水与 FP8。',
            )}
          </li>
          <li>
            <a href="https://arxiv.org/abs/1805.02867" target="_blank" rel="noreferrer">
              Online normalizer calculation for softmax
            </a>{' '}
            {t(
              '— Milakov & Gimelshein, 2018. Where it all begins; only a few pages.',
              '—— Milakov &amp; Gimelshein，2018。一切的起点，只有几页。',
            )}
          </li>
          <li>
            <a href="https://github.com/Dao-AILab/flash-attention" target="_blank" rel="noreferrer">
              Dao-AILab/flash-attention
            </a>{' '}
            {t(
              '— the official CUDA implementation; this is what a production-grade kernel looks like.',
              '—— 官方 CUDA 实现，生产级 kernel 长什么样看这里。',
            )}
          </li>
        </ul>
      </Section>
    </>
  )
}
