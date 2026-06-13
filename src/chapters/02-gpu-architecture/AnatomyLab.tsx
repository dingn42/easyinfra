import { useState } from 'react'
import { Widget } from '@/components/ui'
import { useLocale, useT, pick, type Lang, type Loc } from '@/lib/i18n'

/* ───────────────────────── LAB 01 GPU 解剖台 ─────────────────────────
 * 三层可点击 SVG：
 *   层 1 = GH100 整 die（SM 网格 + 两侧 HBM + 中央 L2）
 *   层 2 = 点击任一 SM 放大（4 个分区 + Tensor Core + 寄存器堆 + L1/Shared）
 *   层 3 = 点击组件 → 右侧信息卡
 */

type View = 'die' | 'sm'

type InfoKey =
  | 'die'
  | 'gpc'
  | 'sm'
  | 'smoff'
  | 'l2'
  | 'hbm'
  | 'nvlink'
  | 'host'
  | 'sched'
  | 'regfile'
  | 'cores'
  | 'tensor'
  | 'ldst'
  | 'l1shared'
  | 'tma'

interface Info {
  name: Loc
  en: string
  count: Loc
  duty: Loc
  stats: [Loc, Loc][]
}

/** 小工具：纯英文/数字串 → 同值 Loc（en === zh），让 stats 表保持统一结构 */
const same = (s: string): Loc => ({ en: s, zh: s })

const INFO: Record<InfoKey, Info> = {
  die: {
    name: { en: 'GH100 die (full chip)', zh: 'GH100 die（整片）' },
    en: 'Full Die',
    count: { en: '1 per GPU', zh: '1 片 / GPU' },
    duty: {
      en: 'A single slab of silicon: the compute unit is designed once as an SM, stamped out 144 times, then wired up with on-die L2 and the interfaces to off-die HBM.',
      zh: '一整块硅片：计算单元做成 SM 后复制 144 份，再配上片上 L2 和片外 HBM 的接口。',
    },
    stats: [
      [{ en: 'Process', zh: '制程' }, same('TSMC 4N')],
      [{ en: 'Die area', zh: '面积' }, same('814 mm²')],
      [{ en: 'Transistors', zh: '晶体管' }, { en: '80 billion', zh: '800 亿' }],
      [{ en: 'SMs (H100 SXM enabled)', zh: 'SM（H100 SXM 启用）' }, same('132 / 144')],
    ],
  },
  gpc: {
    name: { en: 'GPC (Graphics Processing Cluster)', zh: 'GPC（图形处理簇）' },
    en: 'Graphics Processing Cluster',
    count: { en: '8 per die', zh: '8 个 / die' },
    duty: {
      en: 'An administrative district of SMs: subdivided into 9 TPCs (2 SMs each), it exists mostly for physical floorplanning and work distribution.',
      zh: '一组 SM 的行政区划：内部再分 9 个 TPC（每 TPC 2 个 SM），主要服务于物理布线和任务分发。',
    },
    stats: [
      [same('TPC / GPC'), same('9')],
      [same('SM / GPC'), same('18')],
      [{ en: 'To the programmer', zh: '对程序员' }, { en: 'mostly invisible', zh: '几乎透明' }],
    ],
  },
  sm: {
    name: { en: 'SM (Streaming Multiprocessor)', zh: 'SM（流式多处理器）' },
    en: 'Streaming Multiprocessor',
    count: { en: 'H100 SXM: 132 (A100: 108)', zh: 'H100 SXM：132 个（A100：108）' },
    duty: {
      en: "The GPU's real \"core\": a thread block resides entirely on one SM, which hands it registers, shared memory, and execution units. Click to zoom inside.",
      zh: 'GPU 真正的「核」：线程块（block）整块驻留在某个 SM 上，由它供给寄存器、共享内存和执行单元。点击可放大查看内部。',
    },
    stats: [
      [{ en: 'FP32 cores / SM', zh: 'FP32 核 / SM' }, same('128')],
      [{ en: 'Warp schedulers', zh: 'Warp 调度器' }, same('4')],
      [same('Tensor Cores'), same('4')],
      [{ en: 'Register file', zh: '寄存器堆' }, same('256 KB')],
      [{ en: 'Max residency', zh: '最大驻留' }, { en: '64 warps (2048 threads)', zh: '64 warp（2048 线程）' }],
    ],
  },
  smoff: {
    name: { en: 'Disabled SM', zh: '被屏蔽的 SM' },
    en: 'Disabled SM',
    count: { en: '12 per die (H100 SXM)', zh: '12 个 / die（H100 SXM）' },
    duty: {
      en: 'Yield insurance: a die this large is bound to have defects. At the factory the faulty SMs are fused off — of the 144 built, only 132 ship to you.',
      zh: '良率冗余：硅片这么大，难免有瑕疵。出厂时把有缺陷的 SM 熔断屏蔽，144 个里只开 132 个卖给你。',
    },
    stats: [
      [{ en: 'SMs built', zh: '全配 SM' }, same('144')],
      [{ en: 'SMs enabled', zh: '启用 SM' }, { en: '132 (SXM5)', zh: '132（SXM5）' }],
      [{ en: 'PCIe part, same die', zh: '同款 die 的 PCIe 版' }, { en: '114', zh: '114 个' }],
    ],
  },
  l2: {
    name: { en: 'L2 Cache', zh: 'L2 缓存' },
    en: 'L2 Cache',
    count: { en: '1 per die (two partitions)', zh: '1 个 / die（分两个分区）' },
    duty: {
      en: 'The chip-wide last-level on-die cache: every SM memory access passes through here, and a hit means no trip to HBM.',
      zh: '全片共享的最后一级片上缓存：所有 SM 的访存都路过这里，命中就不必去 HBM。',
    },
    stats: [
      [{ en: 'Capacity', zh: '容量' }, { en: '50 MB (A100: 40 MB)', zh: '50 MB（A100：40 MB）' }],
      [{ en: 'Latency', zh: '延迟' }, { en: '~200–300 cycles', zh: '约 200~300 周期' }],
      [{ en: 'Bandwidth', zh: '带宽' }, { en: 'several TB/s', zh: '数 TB/s 量级' }],
    ],
  },
  hbm: {
    name: { en: 'HBM3 memory stacks', zh: 'HBM3 显存堆叠' },
    en: 'High Bandwidth Memory',
    count: { en: '5 stacks active (6 sites)', zh: '5 堆启用（6 个堆位）' },
    duty: {
      en: 'Off-die memory: DRAM dies stacked vertically and placed right against the GPU die on a silicon interposer, trading an ultra-wide bus for bandwidth.',
      zh: '片外显存：DRAM 芯片垂直堆叠后用硅中介层贴着 GPU die 摆放，靠超宽总线换带宽。',
    },
    stats: [
      [{ en: 'Capacity', zh: '容量' }, same('80 GB')],
      [{ en: 'Bandwidth', zh: '带宽' }, { en: '3.35 TB/s (A100: 1.9–2.0)', zh: '3.35 TB/s（A100：1.9~2.0）' }],
      [{ en: 'Bus width', zh: '总线位宽' }, same('5120-bit')],
      [{ en: 'Latency', zh: '延迟' }, { en: '~500+ cycles', zh: '约 500 周期以上' }],
    ],
  },
  nvlink: {
    name: { en: 'NVLink 4 interface', zh: 'NVLink 4 接口' },
    en: 'NVLink',
    count: { en: '18 links', zh: '18 条链路' },
    duty: {
      en: 'The high-speed freeway for multi-GPU: an 8-card H100 server fuses into one "big GPU" through it, and all of chapter 12 (parallel training) rides on it.',
      zh: '多卡互联的高速公路：8 卡 H100 服务器靠它组成一台「大 GPU」，第 12 章并行训练全靠它。',
    },
    stats: [
      [{ en: 'Bidirectional BW', zh: '双向带宽' }, same('900 GB/s')],
      [{ en: 'vs PCIe Gen5', zh: '对比 PCIe Gen5' }, { en: '~7×', zh: '约 7 倍' }],
    ],
  },
  host: {
    name: { en: 'PCIe Gen5 + GigaThread engine', zh: 'PCIe Gen5 + GigaThread 引擎' },
    en: 'Host Interface & Global Scheduler',
    count: { en: '1 set per die', zh: '1 套 / die' },
    duty: {
      en: 'The interface to the CPU plus the global scheduler: once a kernel launches, the GigaThread engine hands out its thousands of thread blocks across the SMs.',
      zh: '与 CPU 的接口 + 全局调度器：kernel 启动后，GigaThread 引擎把成千上万个线程块分发到各个 SM 上。',
    },
    stats: [
      [same('PCIe Gen5 ×16'), same('128 GB/s')],
      [{ en: 'Dispatch granularity', zh: '分发粒度' }, { en: 'thread block', zh: '线程块（block）' }],
    ],
  },
  sched: {
    name: { en: 'Warp scheduler', zh: 'Warp 调度器' },
    en: 'Warp Scheduler',
    count: { en: '4 per SM (1 per partition)', zh: '4 个 / SM（每分区 1 个）' },
    duty: {
      en: 'Each cycle it picks one "ready" warp from those resident in its partition and issues its next instruction — the master switch behind latency hiding.',
      zh: '每个周期从驻留在本分区的 warp 里挑一个「就绪」的，发射它的下一条指令 —— 延迟隐藏的总开关。',
    },
    stats: [
      [{ en: 'Issue rate', zh: '发射率' }, { en: '1 warp instr / cycle', zh: '1 条 warp 指令 / 周期' }],
      [{ en: 'Candidates', zh: '候选' }, { en: 'up to 16 warps / partition', zh: '最多 16 warp / 分区' }],
      [{ en: 'Switch cost', zh: '切换开销' }, { en: '0 cycles', zh: '0 周期' }],
    ],
  },
  regfile: {
    name: { en: 'Register file', zh: '寄存器堆' },
    en: 'Register File',
    count: { en: '64 KB / partition, 256 KB / SM', zh: '64 KB / 分区，256 KB / SM' },
    duty: {
      en: "Every resident thread's private variables live here — switching warps saves and restores nothing, which is the physical reason context switches cost zero.",
      zh: '所有驻留线程的「私人变量」常驻于此 —— warp 切换不需要保存/恢复任何状态，这是零成本切换的物质基础。',
    },
    stats: [
      [{ en: 'Capacity / SM', zh: '容量 / SM' }, { en: '256 KB (64K × 32-bit)', zh: '256 KB（16 万个 32-bit）' }],
      [{ en: 'Latency', zh: '延迟' }, { en: '~1 cycle', zh: '约 1 周期' }],
      [{ en: 'Per-thread cap', zh: '每线程上限' }, { en: '255 registers', zh: '255 个寄存器' }],
    ],
  },
  cores: {
    name: { en: 'FP32 CUDA core array', zh: 'FP32 CUDA 核阵列' },
    en: 'FP32 Cores',
    count: { en: '32 / partition, 128 / SM, 16896 / GPU', zh: '32 个 / 分区，128 个 / SM，16896 个 / GPU' },
    duty: {
      en: 'Scalar floating-point units: a warp\'s 32 threads map exactly onto a partition\'s 32 cores, all executing the same instruction in lockstep.',
      zh: '标量浮点运算单元：一个 warp 的 32 个线程恰好铺满一个分区的 32 个核，锁步执行同一条指令。',
    },
    stats: [
      [{ en: 'FP32 peak (whole GPU)', zh: 'FP32 峰值（全卡）' }, { en: '~67 TFLOPS', zh: '约 67 TFLOPS' }],
      [{ en: 'FP64 units', zh: 'FP64 单元' }, { en: '64 more / SM', zh: '另有 64 个 / SM' }],
      [{ en: 'INT32 units', zh: 'INT32 单元' }, { en: '64 / SM', zh: '64 个 / SM' }],
    ],
  },
  tensor: {
    name: { en: 'Tensor Core (4th gen)', zh: 'Tensor Core（第 4 代）' },
    en: 'Tensor Core',
    count: { en: '1 / partition, 4 / SM, 528 / GPU', zh: '1 个 / 分区，4 个 / SM，528 个 / GPU' },
    duty: {
      en: 'Dedicated matrix multiply-accumulate circuitry: one MMA instruction multiplies a small matrix tile and accumulates it. In the LLM era it contributes 95%+ of all FLOPs.',
      zh: '矩阵乘加专用电路：一条 MMA 指令完成一小块矩阵乘并累加，大模型时代 95% 以上的 FLOPs 由它贡献。',
    },
    stats: [
      [{ en: 'BF16 peak (whole GPU)', zh: 'BF16 峰值（全卡）' }, same('989 TFLOPS')],
      [{ en: 'FP8 peak', zh: 'FP8 峰值' }, same('1979 TFLOPS')],
      [{ en: 'vs FP32 CUDA core', zh: '对比 FP32 CUDA 核' }, { en: '~15×', zh: '约 15 倍' }],
    ],
  },
  ldst: {
    name: { en: 'LD/ST & SFU', zh: 'LD/ST 与 SFU' },
    en: 'Load/Store & Special Function Units',
    count: { en: 'one group per partition', zh: '每分区各一组' },
    duty: {
      en: 'LD/ST issues memory instructions out to L1/L2/HBM; the SFU computes fast approximations of transcendentals (sin, exp, reciprocal square root).',
      zh: 'LD/ST 负责把访存指令发往 L1/L2/HBM；SFU 负责超越函数（sin、exp、倒数平方根）的快速近似。',
    },
    stats: [
      [{ en: 'Access granularity', zh: '访存粒度' }, { en: '32 B sector / 128 B line', zh: '32 B 扇区 / 128 B 缓存行' }],
      [{ en: 'exp/sin etc.', zh: 'exp/sin 等' }, { en: 'SFU hardware approx.', zh: 'SFU 硬件近似' }],
    ],
  },
  l1shared: {
    name: { en: 'L1 data cache / shared memory', zh: 'L1 数据缓存 / 共享内存' },
    en: 'L1 / Shared Memory',
    count: { en: '1 per SM (shared by 4 partitions)', zh: '1 块 / SM（4 个分区共用）' },
    duty: {
      en: 'One SRAM, two uses: part of it acts as a hardware-managed L1, part as a programmer-controlled scratchpad (shared memory), and the split is configurable.',
      zh: '一块 SRAM 两种用法：一部分当硬件管理的 L1，一部分当程序员显式管理的便笺（shared memory），比例可配。',
    },
    stats: [
      [{ en: 'Total capacity', zh: '总容量' }, same('256 KB / SM')],
      [{ en: 'Max shared config', zh: 'Shared 最大可配' }, same('228 KB')],
      [{ en: 'Latency', zh: '延迟' }, { en: '~30 cycles', zh: '约 30 周期' }],
    ],
  },
  tma: {
    name: { en: 'TMA + texture units', zh: 'TMA + 纹理单元' },
    en: 'Tensor Memory Accelerator',
    count: { en: '1 TMA / SM (new in Hopper)', zh: '1 个 TMA / SM（Hopper 新增）' },
    duty: {
      en: 'The async mover: a single descriptor copies a whole multi-dimensional tensor tile from global memory into shared memory while the cores keep computing.',
      zh: '异步搬运工：一条描述符就能把全局内存里的多维张量块整块搬进共享内存，搬运期间核心继续算别的。',
    },
    stats: [
      [{ en: 'Transfer mode', zh: '搬运模式' }, { en: 'async, bulk, multi-dim', zh: '异步、整块、多维' }],
      [{ en: 'Beneficiaries', zh: '受益者' }, { en: 'FlashAttention etc. (ch. 8)', zh: 'FlashAttention 等（第 8 章）' }],
    ],
  },
}

/** H100 SXM5 屏蔽 12 个 SM（演示用的确定性分布） */
const DISABLED = new Set([
  '0-4', '0-12', '1-13', '2-1', '2-7', '3-16',
  '4-2', '5-3', '5-9', '6-15', '7-6', '7-11',
])

const GPC_W = 137
const GPC_H = 160
const CELL_W = 39
const CELL_H = 20

function gpcX(i: number) {
  return 94 + (i % 4) * (GPC_W + 8)
}
function gpcY(i: number) {
  return i < 4 ? 54 : 258
}

/* ── die 层 ── */
function DieView({
  sel,
  onSel,
  onEnterSm,
}: {
  sel: InfoKey
  onSel: (k: InfoKey) => void
  onEnterSm: (id: number) => void
}) {
  const t = useT()
  const selCls = (k: InfoKey) => (sel === k ? 'stroke-volt' : 'stroke-line2 hover:stroke-volt/70')
  return (
    <svg viewBox="0 0 760 470" className="w-full select-none" role="img" aria-label={t('GH100 die layout', 'GH100 die 布局图')}>
      {/* die 外框 */}
      <rect
        x={84} y={14} width={592} height={442} rx={6}
        className={`cursor-pointer fill-panel ${selCls('die')}`}
        onClick={() => onSel('die')}
      />
      <text x={92} y={470 - 6} fontSize={10} className="pointer-events-none fill-ink3 font-mono">
        {t('GH100 · 814 mm² · 80 B transistors', 'GH100 · 814 mm² · 800 亿晶体管')}
      </text>

      {/* HBM 两侧堆叠 */}
      {[0, 1, 2].map((i) =>
        [10, 686].map((x) => (
          <g key={`${x}-${i}`} className="cursor-pointer" onClick={() => onSel('hbm')}>
            <rect
              x={x} y={40 + i * 140} width={64} height={120} rx={4}
              className={`fill-amber/10 ${sel === 'hbm' ? 'stroke-amber' : 'stroke-amber/40 hover:stroke-amber'}`}
            />
            {[0, 1, 2, 3].map((j) => (
              <line
                key={j} x1={x + 8} x2={x + 56}
                y1={64 + i * 140 + j * 22} y2={64 + i * 140 + j * 22}
                className="pointer-events-none stroke-amber/30"
              />
            ))}
            <text x={x + 32} y={56 + i * 140} fontSize={10} textAnchor="middle" className="pointer-events-none fill-amber font-mono">
              HBM3
            </text>
          </g>
        )),
      )}

      {/* NVLink 顶部条 */}
      <rect
        x={94} y={22} width={572} height={24} rx={3}
        className={`cursor-pointer fill-cyan/10 ${sel === 'nvlink' ? 'stroke-cyan' : 'stroke-cyan/40 hover:stroke-cyan'}`}
        onClick={() => onSel('nvlink')}
      />
      <text x={380} y={38} fontSize={11} textAnchor="middle" className="pointer-events-none fill-cyan font-mono">
        {t('NVLink 4 ×18 · 900 GB/s', 'NVLink 4 互联 ×18 · 900 GB/s')}
      </text>

      {/* 底部 PCIe + GigaThread */}
      <rect
        x={94} y={426} width={572} height={22} rx={3}
        className={`cursor-pointer fill-bg2 ${selCls('host')}`}
        onClick={() => onSel('host')}
      />
      <text x={380} y={441} fontSize={10} textAnchor="middle" className="pointer-events-none fill-ink2 font-mono">
        {t('PCIe Gen5 interface · GigaThread global scheduler', 'PCIe Gen5 接口 · GigaThread 全局调度引擎')}
      </text>

      {/* 中央 L2 */}
      <rect
        x={94} y={222} width={572} height={28} rx={3}
        className={`cursor-pointer fill-violet/10 ${sel === 'l2' ? 'stroke-violet' : 'stroke-violet/40 hover:stroke-violet'}`}
        onClick={() => onSel('l2')}
      />
      <text x={380} y={241} fontSize={11} textAnchor="middle" className="pointer-events-none fill-violet font-mono">
        {t('L2 cache · 50 MB (chip-wide)', 'L2 缓存 50 MB（全片共享）')}
      </text>

      {/* 8 个 GPC */}
      {Array.from({ length: 8 }, (_, g) => {
        const x0 = gpcX(g)
        const y0 = gpcY(g)
        return (
          <g key={g}>
            <rect
              x={x0} y={y0} width={GPC_W} height={GPC_H} rx={4}
              className={`cursor-pointer fill-bg2/60 ${selCls('gpc')}`}
              onClick={() => onSel('gpc')}
            />
            <text
              x={x0 + 6} y={y0 + 12} fontSize={9}
              className="pointer-events-none fill-ink3 font-mono"
            >
              GPC {g} · 9 TPC
            </text>
            {/* TPC 分组虚线（每 2 行 SM = 3 个 TPC） */}
            {[0, 1, 2].map((p) => (
              <rect
                key={p}
                x={x0 + 3.5} y={y0 + 16 + p * (CELL_H * 2 + 8) - 2}
                width={GPC_W - 7} height={CELL_H * 2 + 8}
                rx={2}
                className="pointer-events-none fill-none stroke-line"
                strokeDasharray="3 3"
              />
            ))}
            {/* 18 个 SM 单元 */}
            {Array.from({ length: 18 }, (_, j) => {
              const col = j % 3
              const row = Math.floor(j / 3)
              const cx = x0 + 6 + col * (CELL_W + 4)
              const cy = y0 + 17 + row * (CELL_H + 3.7)
              const off = DISABLED.has(`${g}-${j}`)
              const id = g * 18 + j
              return off ? (
                <rect
                  key={j} x={cx} y={cy} width={CELL_W} height={CELL_H} rx={2}
                  className="cursor-pointer fill-bg stroke-line opacity-50 hover:opacity-80"
                  onClick={() => onSel('smoff')}
                />
              ) : (
                <rect
                  key={j} x={cx} y={cy} width={CELL_W} height={CELL_H} rx={2}
                  className="cursor-pointer fill-volt/15 stroke-volt/30 transition-colors hover:fill-volt/40 hover:stroke-volt"
                  onClick={() => onEnterSm(id)}
                >
                  <title>{t(`SM #${id} — click to zoom in`, `SM #${id} —— 点击放大`)}</title>
                </rect>
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}

/* ── SM 层 ── */
function SmView({ smId, sel, onSel }: { smId: number; sel: InfoKey; onSel: (k: InfoKey) => void }) {
  const t = useT()
  const selCls = (k: InfoKey, base: string, hot: string) => (sel === k ? hot : base)
  const parts: [number, number][] = [
    [98, 64],
    [384, 64],
    [98, 214],
    [384, 214],
  ]
  return (
    <svg viewBox="0 0 760 470" className="w-full select-none" role="img" aria-label={t(`SM #${smId} internal layout`, `SM #${smId} 内部结构图`)}>
      <rect x={90} y={16} width={580} height={438} rx={6} className="fill-panel stroke-line2" />
      <text x={102} y={38} fontSize={12} className="fill-ink font-mono">
        SM #{smId}
      </text>
      <text x={658} y={38} fontSize={10} textAnchor="end" className="fill-ink3 font-mono">
        {t('L1 instruction cache (shared by 4 partitions)', 'L1 指令缓存（4 分区共用）')}
      </text>
      <line x1={98} x2={662} y1={48} y2={48} className="stroke-line" />

      {/* 4 个处理分区 */}
      {parts.map(([px, py], pi) => (
        <g key={pi}>
          <rect x={px} y={py} width={278} height={142} rx={4} className="fill-bg2/70 stroke-line" />
          <text x={px + 270} y={py + 12} fontSize={8} textAnchor="end" className="pointer-events-none fill-ink3 font-mono">
            {t(`Partition ${pi}`, `分区 ${pi}`)}
          </text>

          {/* warp 调度器 */}
          <rect
            x={px + 6} y={py + 6} width={206} height={20} rx={3}
            className={`cursor-pointer fill-volt/10 ${selCls('sched', 'stroke-volt/40 hover:stroke-volt', 'stroke-volt')}`}
            onClick={() => onSel('sched')}
          />
          <text x={px + 109} y={py + 20} fontSize={10} textAnchor="middle" className="pointer-events-none fill-volt font-mono">
            {t('Warp scheduler + issue', 'Warp 调度器 + 发射')}
          </text>

          {/* 寄存器堆 */}
          <rect
            x={px + 6} y={py + 30} width={266} height={18} rx={3}
            className={`cursor-pointer fill-amber/10 ${selCls('regfile', 'stroke-amber/40 hover:stroke-amber', 'stroke-amber')}`}
            onClick={() => onSel('regfile')}
          />
          <text x={px + 139} y={py + 43} fontSize={10} textAnchor="middle" className="pointer-events-none fill-amber font-mono">
            {t('Register file · 64 KB', '寄存器堆 64 KB')}
          </text>

          {/* 32 个 FP32 核 */}
          <g className="cursor-pointer" onClick={() => onSel('cores')}>
            <rect
              x={px + 6} y={py + 52} width={266} height={36} rx={3}
              className={`fill-bg ${selCls('cores', 'stroke-line2 hover:stroke-volt/70', 'stroke-volt')}`}
            />
            {Array.from({ length: 32 }, (_, c) => (
              <rect
                key={c}
                x={px + 11 + (c % 16) * 16}
                y={py + 56 + Math.floor(c / 16) * 15}
                width={12} height={12} rx={1.5}
                className="pointer-events-none fill-cyan/25 stroke-cyan/40"
              />
            ))}
          </g>
          <text x={px + 6} y={py + 99} fontSize={8.5} className="pointer-events-none fill-ink3 font-mono">
            {t('FP32 cores ×32 (one warp fills it exactly)', 'FP32 核 ×32（一个 warp 恰好铺满）')}
          </text>

          {/* Tensor Core + LD/ST */}
          <rect
            x={px + 6} y={py + 104} width={130} height={32} rx={3}
            className={`cursor-pointer fill-violet/10 ${selCls('tensor', 'stroke-violet/50 hover:stroke-violet', 'stroke-violet')}`}
            onClick={() => onSel('tensor')}
          />
          <text x={px + 71} y={py + 124} fontSize={10} textAnchor="middle" className="pointer-events-none fill-violet font-mono">
            Tensor Core
          </text>
          <rect
            x={px + 142} y={py + 104} width={130} height={32} rx={3}
            className={`cursor-pointer fill-bg ${selCls('ldst', 'stroke-line2 hover:stroke-volt/70', 'stroke-volt')}`}
            onClick={() => onSel('ldst')}
          />
          <text x={px + 207} y={py + 124} fontSize={10} textAnchor="middle" className="pointer-events-none fill-ink2 font-mono">
            LD/ST · SFU
          </text>
        </g>
      ))}

      {/* L1 / Shared */}
      <rect
        x={98} y={366} width={564} height={38} rx={4}
        className={`cursor-pointer fill-amber/10 ${selCls('l1shared', 'stroke-amber/40 hover:stroke-amber', 'stroke-amber')}`}
        onClick={() => onSel('l1shared')}
      />
      <text x={380} y={390} fontSize={11} textAnchor="middle" className="pointer-events-none fill-amber font-mono">
        {t('256 KB L1 data cache / shared memory (up to 228 KB shared)', '256 KB L1 数据缓存 / 共享内存（Shared 最多可配 228 KB）')}
      </text>

      {/* TMA + Tex */}
      <rect
        x={98} y={412} width={564} height={28} rx={4}
        className={`cursor-pointer fill-bg2 ${selCls('tma', 'stroke-line2 hover:stroke-volt/70', 'stroke-volt')}`}
        onClick={() => onSel('tma')}
      />
      <text x={380} y={430} fontSize={10} textAnchor="middle" className="pointer-events-none fill-ink2 font-mono">
        {t('TMA tensor memory accelerator · texture units ×4', 'TMA 张量内存加速器 · 纹理单元 ×4')}
      </text>
    </svg>
  )
}

/* ── 信息卡 ── */
function InfoCard({ k, lang }: { k: InfoKey; lang: Lang }) {
  const info = INFO[k]
  const t = useT()
  return (
    <div className="flex h-full flex-col rounded-md border border-line bg-bg2/60 p-4">
      <div className="microlabel text-volt">{t('◈ Component file', '◈ 部件档案')}</div>
      <div className="mt-1.5 font-display text-[15px] font-semibold text-ink">{pick(info.name, lang)}</div>
      <div className="font-mono text-[10.5px] uppercase tracking-wider text-ink3">{info.en}</div>
      <div className="mt-3 border-t border-line pt-3">
        <div className="microlabel mb-1">{t('Count', '数量')}</div>
        <div className="font-mono text-[12px] text-cyan">{pick(info.count, lang)}</div>
      </div>
      <div className="mt-3">
        <div className="microlabel mb-1">{t('Role', '职责')}</div>
        <p className="text-[12.5px] leading-[1.85] text-text">{pick(info.duty, lang)}</p>
      </div>
      <div className="mt-3">
        <div className="microlabel mb-1.5">{t('Key numbers', '关键数字')}</div>
        <dl className="space-y-1">
          {info.stats.map(([kk, v]) => (
            <div key={kk.en} className="flex items-baseline justify-between gap-2 text-[11.5px]">
              <dt className="text-ink3">{pick(kk, lang)}</dt>
              <dd className="text-right font-mono tabular-nums text-ink">{pick(v, lang)}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  )
}

export function AnatomyLab() {
  const t = useT()
  const { lang } = useLocale()
  const [view, setView] = useState<View>('die')
  const [smId, setSmId] = useState(57)
  const [sel, setSel] = useState<InfoKey>('die')

  const goDie = () => {
    setView('die')
    setSel('die')
  }
  const enterSm = (id: number) => {
    setSmId(id)
    setView('sm')
    setSel('sm')
  }
  const reset = () => {
    goDie()
    setSmId(57)
  }

  return (
    <Widget
      index={1}
      title={t('GPU dissection bench', 'GPU 解剖台')}
      subtitle={t('H100 · click a component for its file, click an SM to zoom', 'H100 · 点击部件查看档案，点击 SM 放大')}
      onReset={reset}
      wide
      footer={t(
        <>
          Drill down layer by layer: die → SM → component. Watch the area budget — there is almost no
          "big cache" inside an SM; that silicon goes to the register file and execution units instead.
          That is exactly where GPUs and CPUs part ways philosophically.
        </>,
        <>
          逐层点开：die → SM → 部件。注意面积分配 —— SM 里几乎没有「大缓存」，
          省下的面积全部还给了寄存器堆和执行单元，这正是 GPU 与 CPU 在哲学上的分野。
        </>,
      )}
    >
      {/* 面包屑 */}
      <div className="mb-3 flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
        <button
          onClick={goDie}
          className={`rounded border px-2 py-0.5 transition-colors ${
            view === 'die' ? 'border-volt/50 text-volt' : 'border-line2 text-ink2 hover:text-volt'
          }`}
        >
          GH100 die
        </button>
        {view === 'sm' && (
          <>
            <span className="text-ink3">▸</span>
            <button
              onClick={() => setSel('sm')}
              className={`rounded border px-2 py-0.5 transition-colors ${
                sel === 'sm' ? 'border-volt/50 text-volt' : 'border-line2 text-ink2 hover:text-volt'
              }`}
            >
              SM #{smId}
            </button>
          </>
        )}
        {sel !== 'die' && sel !== 'sm' && (
          <>
            <span className="text-ink3">▸</span>
            <span className="text-volt">{pick(INFO[sel].name, lang)}</span>
          </>
        )}
        <span className="ml-auto hidden text-ink3 sm:inline">
          {view === 'die'
            ? t('Green cells = SMs, click to zoom', '绿色小格 = SM，点击放大')
            : t('Click an internal component for its file', '点击内部部件查看档案')}
        </span>
      </div>

      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="min-w-0 flex-1 overflow-x-auto">
          <div className="min-w-[540px]">
            {view === 'die' ? (
              <DieView sel={sel} onSel={setSel} onEnterSm={enterSm} />
            ) : (
              <SmView smId={smId} sel={sel} onSel={setSel} />
            )}
          </div>
        </div>
        <div className="w-full shrink-0 lg:w-64">
          <InfoCard k={sel} lang={lang} />
        </div>
      </div>
    </Widget>
  )
}
