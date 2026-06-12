import { useState } from 'react'
import { Widget } from '@/components/ui'

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
  name: string
  en: string
  count: string
  duty: string
  stats: [string, string][]
}

const INFO: Record<InfoKey, Info> = {
  die: {
    name: 'GH100 die（整片）',
    en: 'Full Die',
    count: '1 片 / GPU',
    duty: '一整块硅片：计算单元做成 SM 后复制 144 份，再配上片上 L2 和片外 HBM 的接口。',
    stats: [
      ['制程', 'TSMC 4N'],
      ['面积', '814 mm²'],
      ['晶体管', '800 亿'],
      ['SM（H100 SXM 启用）', '132 / 144'],
    ],
  },
  gpc: {
    name: 'GPC（图形处理簇）',
    en: 'Graphics Processing Cluster',
    count: '8 个 / die',
    duty: '一组 SM 的行政区划：内部再分 9 个 TPC（每 TPC 2 个 SM），主要服务于物理布线和任务分发。',
    stats: [
      ['TPC / GPC', '9'],
      ['SM / GPC', '18'],
      ['对程序员', '几乎透明'],
    ],
  },
  sm: {
    name: 'SM（流式多处理器）',
    en: 'Streaming Multiprocessor',
    count: 'H100 SXM：132 个（A100：108）',
    duty: 'GPU 真正的「核」：线程块（block）整块驻留在某个 SM 上，由它供给寄存器、共享内存和执行单元。点击可放大查看内部。',
    stats: [
      ['FP32 核 / SM', '128'],
      ['Warp 调度器', '4'],
      ['Tensor Core', '4'],
      ['寄存器堆', '256 KB'],
      ['最大驻留', '64 warp（2048 线程）'],
    ],
  },
  smoff: {
    name: '被屏蔽的 SM',
    en: 'Disabled SM',
    count: '12 个 / die（H100 SXM）',
    duty: '良率冗余：硅片这么大，难免有瑕疵。出厂时把有缺陷的 SM 熔断屏蔽，144 个里只开 132 个卖给你。',
    stats: [
      ['全配 SM', '144'],
      ['启用 SM', '132（SXM5）'],
      ['同款 die 的 PCIe 版', '114 个'],
    ],
  },
  l2: {
    name: 'L2 缓存',
    en: 'L2 Cache',
    count: '1 个 / die（分两个分区）',
    duty: '全片共享的最后一级片上缓存：所有 SM 的访存都路过这里，命中就不必去 HBM。',
    stats: [
      ['容量', '50 MB（A100：40 MB）'],
      ['延迟', '约 200~300 周期'],
      ['带宽', '数 TB/s 量级'],
    ],
  },
  hbm: {
    name: 'HBM3 显存堆叠',
    en: 'High Bandwidth Memory',
    count: '5 堆启用（6 个堆位）',
    duty: '片外显存：DRAM 芯片垂直堆叠后用硅中介层贴着 GPU die 摆放，靠超宽总线换带宽。',
    stats: [
      ['容量', '80 GB'],
      ['带宽', '3.35 TB/s（A100：1.9~2.0）'],
      ['总线位宽', '5120-bit'],
      ['延迟', '约 500 周期以上'],
    ],
  },
  nvlink: {
    name: 'NVLink 4 接口',
    en: 'NVLink',
    count: '18 条链路',
    duty: '多卡互联的高速公路：8 卡 H100 服务器靠它组成一台「大 GPU」，第 12 章并行训练全靠它。',
    stats: [
      ['双向带宽', '900 GB/s'],
      ['对比 PCIe Gen5', '约 7 倍'],
    ],
  },
  host: {
    name: 'PCIe Gen5 + GigaThread 引擎',
    en: 'Host Interface & Global Scheduler',
    count: '1 套 / die',
    duty: '与 CPU 的接口 + 全局调度器：kernel 启动后，GigaThread 引擎把成千上万个线程块分发到各个 SM 上。',
    stats: [
      ['PCIe Gen5 ×16', '128 GB/s'],
      ['分发粒度', '线程块（block）'],
    ],
  },
  sched: {
    name: 'Warp 调度器',
    en: 'Warp Scheduler',
    count: '4 个 / SM（每分区 1 个）',
    duty: '每个周期从驻留在本分区的 warp 里挑一个「就绪」的，发射它的下一条指令 —— 延迟隐藏的总开关。',
    stats: [
      ['发射率', '1 条 warp 指令 / 周期'],
      ['候选', '最多 16 warp / 分区'],
      ['切换开销', '0 周期'],
    ],
  },
  regfile: {
    name: '寄存器堆',
    en: 'Register File',
    count: '64 KB / 分区，256 KB / SM',
    duty: '所有驻留线程的「私人变量」常驻于此 —— warp 切换不需要保存/恢复任何状态，这是零成本切换的物质基础。',
    stats: [
      ['容量 / SM', '256 KB（16 万个 32-bit）'],
      ['延迟', '约 1 周期'],
      ['每线程上限', '255 个寄存器'],
    ],
  },
  cores: {
    name: 'FP32 CUDA 核阵列',
    en: 'FP32 Cores',
    count: '32 个 / 分区，128 个 / SM，16896 个 / GPU',
    duty: '标量浮点运算单元：一个 warp 的 32 个线程恰好铺满一个分区的 32 个核，锁步执行同一条指令。',
    stats: [
      ['FP32 峰值（全卡）', '约 67 TFLOPS'],
      ['FP64 单元', '另有 64 个 / SM'],
      ['INT32 单元', '64 个 / SM'],
    ],
  },
  tensor: {
    name: 'Tensor Core（第 4 代）',
    en: 'Tensor Core',
    count: '1 个 / 分区，4 个 / SM，528 个 / GPU',
    duty: '矩阵乘加专用电路：一条 MMA 指令完成一小块矩阵乘并累加，大模型时代 95% 以上的 FLOPs 由它贡献。',
    stats: [
      ['BF16 峰值（全卡）', '989 TFLOPS'],
      ['FP8 峰值', '1979 TFLOPS'],
      ['对比 FP32 CUDA 核', '约 15 倍'],
    ],
  },
  ldst: {
    name: 'LD/ST 与 SFU',
    en: 'Load/Store & Special Function Units',
    count: '每分区各一组',
    duty: 'LD/ST 负责把访存指令发往 L1/L2/HBM；SFU 负责超越函数（sin、exp、倒数平方根）的快速近似。',
    stats: [
      ['访存粒度', '32 B 扇区 / 128 B 缓存行'],
      ['exp/sin 等', 'SFU 硬件近似'],
    ],
  },
  l1shared: {
    name: 'L1 数据缓存 / 共享内存',
    en: 'L1 / Shared Memory',
    count: '1 块 / SM（4 个分区共用）',
    duty: '一块 SRAM 两种用法：一部分当硬件管理的 L1，一部分当程序员显式管理的便笺（shared memory），比例可配。',
    stats: [
      ['总容量', '256 KB / SM'],
      ['Shared 最大可配', '228 KB'],
      ['延迟', '约 30 周期'],
    ],
  },
  tma: {
    name: 'TMA + 纹理单元',
    en: 'Tensor Memory Accelerator',
    count: '1 个 TMA / SM（Hopper 新增）',
    duty: '异步搬运工：一条描述符就能把全局内存里的多维张量块整块搬进共享内存，搬运期间核心继续算别的。',
    stats: [
      ['搬运模式', '异步、整块、多维'],
      ['受益者', 'FlashAttention 等（第 8 章）'],
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
  const selCls = (k: InfoKey) => (sel === k ? 'stroke-volt' : 'stroke-line2 hover:stroke-volt/70')
  return (
    <svg viewBox="0 0 760 470" className="w-full select-none" role="img" aria-label="GH100 die 布局图">
      {/* die 外框 */}
      <rect
        x={84} y={14} width={592} height={442} rx={6}
        className={`cursor-pointer fill-panel ${selCls('die')}`}
        onClick={() => onSel('die')}
      />
      <text x={92} y={470 - 6} fontSize={10} className="pointer-events-none fill-ink3 font-mono">
        GH100 · 814 mm² · 800 亿晶体管
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
        NVLink 4 互联 ×18 · 900 GB/s
      </text>

      {/* 底部 PCIe + GigaThread */}
      <rect
        x={94} y={426} width={572} height={22} rx={3}
        className={`cursor-pointer fill-bg2 ${selCls('host')}`}
        onClick={() => onSel('host')}
      />
      <text x={380} y={441} fontSize={10} textAnchor="middle" className="pointer-events-none fill-ink2 font-mono">
        PCIe Gen5 接口 · GigaThread 全局调度引擎
      </text>

      {/* 中央 L2 */}
      <rect
        x={94} y={222} width={572} height={28} rx={3}
        className={`cursor-pointer fill-violet/10 ${sel === 'l2' ? 'stroke-violet' : 'stroke-violet/40 hover:stroke-violet'}`}
        onClick={() => onSel('l2')}
      />
      <text x={380} y={241} fontSize={11} textAnchor="middle" className="pointer-events-none fill-violet font-mono">
        L2 缓存 50 MB（全片共享）
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
                  <title>SM #{id} —— 点击放大</title>
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
  const selCls = (k: InfoKey, base: string, hot: string) => (sel === k ? hot : base)
  const parts: [number, number][] = [
    [98, 64],
    [384, 64],
    [98, 214],
    [384, 214],
  ]
  return (
    <svg viewBox="0 0 760 470" className="w-full select-none" role="img" aria-label={`SM #${smId} 内部结构图`}>
      <rect x={90} y={16} width={580} height={438} rx={6} className="fill-panel stroke-line2" />
      <text x={102} y={38} fontSize={12} className="fill-ink font-mono">
        SM #{smId}
      </text>
      <text x={658} y={38} fontSize={10} textAnchor="end" className="fill-ink3 font-mono">
        L1 指令缓存（4 分区共用）
      </text>
      <line x1={98} x2={662} y1={48} y2={48} className="stroke-line" />

      {/* 4 个处理分区 */}
      {parts.map(([px, py], pi) => (
        <g key={pi}>
          <rect x={px} y={py} width={278} height={142} rx={4} className="fill-bg2/70 stroke-line" />
          <text x={px + 270} y={py + 12} fontSize={8} textAnchor="end" className="pointer-events-none fill-ink3 font-mono">
            分区 {pi}
          </text>

          {/* warp 调度器 */}
          <rect
            x={px + 6} y={py + 6} width={206} height={20} rx={3}
            className={`cursor-pointer fill-volt/10 ${selCls('sched', 'stroke-volt/40 hover:stroke-volt', 'stroke-volt')}`}
            onClick={() => onSel('sched')}
          />
          <text x={px + 109} y={py + 20} fontSize={10} textAnchor="middle" className="pointer-events-none fill-volt font-mono">
            Warp 调度器 + 发射
          </text>

          {/* 寄存器堆 */}
          <rect
            x={px + 6} y={py + 30} width={266} height={18} rx={3}
            className={`cursor-pointer fill-amber/10 ${selCls('regfile', 'stroke-amber/40 hover:stroke-amber', 'stroke-amber')}`}
            onClick={() => onSel('regfile')}
          />
          <text x={px + 139} y={py + 43} fontSize={10} textAnchor="middle" className="pointer-events-none fill-amber font-mono">
            寄存器堆 64 KB
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
            FP32 核 ×32（一个 warp 恰好铺满）
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
        256 KB L1 数据缓存 / 共享内存（Shared 最多可配 228 KB）
      </text>

      {/* TMA + Tex */}
      <rect
        x={98} y={412} width={564} height={28} rx={4}
        className={`cursor-pointer fill-bg2 ${selCls('tma', 'stroke-line2 hover:stroke-volt/70', 'stroke-volt')}`}
        onClick={() => onSel('tma')}
      />
      <text x={380} y={430} fontSize={10} textAnchor="middle" className="pointer-events-none fill-ink2 font-mono">
        TMA 张量内存加速器 · 纹理单元 ×4
      </text>
    </svg>
  )
}

/* ── 信息卡 ── */
function InfoCard({ k }: { k: InfoKey }) {
  const info = INFO[k]
  return (
    <div className="flex h-full flex-col rounded-md border border-line bg-bg2/60 p-4">
      <div className="microlabel text-volt">◈ 部件档案</div>
      <div className="mt-1.5 font-display text-[15px] font-semibold text-ink">{info.name}</div>
      <div className="font-mono text-[10.5px] uppercase tracking-wider text-ink3">{info.en}</div>
      <div className="mt-3 border-t border-line pt-3">
        <div className="microlabel mb-1">数量</div>
        <div className="font-mono text-[12px] text-cyan">{info.count}</div>
      </div>
      <div className="mt-3">
        <div className="microlabel mb-1">职责</div>
        <p className="text-[12.5px] leading-[1.85] text-text">{info.duty}</p>
      </div>
      <div className="mt-3">
        <div className="microlabel mb-1.5">关键数字</div>
        <dl className="space-y-1">
          {info.stats.map(([kk, v]) => (
            <div key={kk} className="flex items-baseline justify-between gap-2 text-[11.5px]">
              <dt className="text-ink3">{kk}</dt>
              <dd className="text-right font-mono tabular-nums text-ink">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  )
}

export function AnatomyLab() {
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
      title="GPU 解剖台"
      subtitle="H100 · 点击部件查看档案，点击 SM 放大"
      onReset={reset}
      wide
      footer={
        <>
          逐层点开：die → SM → 部件。注意面积分配 —— SM 里几乎没有「大缓存」，
          省下的面积全部还给了寄存器堆和执行单元，这正是 GPU 与 CPU 在哲学上的分野。
        </>
      }
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
            <span className="text-volt">{INFO[sel].name}</span>
          </>
        )}
        <span className="ml-auto hidden text-ink3 sm:inline">
          {view === 'die' ? '绿色小格 = SM，点击放大' : '点击内部部件查看档案'}
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
          <InfoCard k={sel} />
        </div>
      </div>
    </Widget>
  )
}
