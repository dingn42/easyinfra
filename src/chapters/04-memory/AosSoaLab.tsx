import { useMemo, useState } from 'react'
import { Segmented, Stat, Widget } from '@/components/ui'
import { useT } from '@/lib/i18n'

/** LAB 02 AoS vs SoA：同一个「只读 .x」的 kernel，两种内存布局的事务数对比 */

const LANES = 32
const CELLS = 96 // 32 个粒子 × 3 个字段（x/y/z），每格 4B，共 384B = 12 个 32B 段
const SEGS = 12
const W = 720
const LANE_Y = 36
const STRIP_Y = 138
const STRIP_H = 40
const VH = 206

type Layout = 'aos' | 'soa'

const FIELD_COLORS = ['var(--color-cyan)', 'var(--color-violet)', 'var(--color-ink3)'] as const
const FIELD_ALPHA = [0.3, 0.22, 0.18] as const

function xCellOf(layout: Layout, i: number) {
  return layout === 'aos' ? 3 * i : i
}

function txnOf(layout: Layout) {
  const set = new Set<number>()
  for (let i = 0; i < LANES; i++) set.add(Math.floor((xCellOf(layout, i) * 4) / 32))
  return set
}

export function AosSoaLab() {
  const t = useT()
  const [layout, setLayout] = useState<Layout>('aos')

  const { segSet, txnAos, txnSoa } = useMemo(
    () => ({ segSet: txnOf(layout), txnAos: txnOf('aos').size, txnSoa: txnOf('soa').size }),
    [layout],
  )
  const txn = segSet.size
  const util = 128 / (txn * 32)

  const cellW = W / CELLS
  const laneX = (i: number) => ((i + 0.5) * W) / LANES
  /** 格子 c 装的是哪个字段：AoS 按 c%3 轮换，SoA 按三大块划分 */
  const fieldOf = (c: number) => (layout === 'aos' ? c % 3 : Math.floor(c / 32))

  return (
    <Widget
      index={2}
      title="AoS vs SoA"
      subtitle={t(
        'When a kernel reads only .x, the two layouts differ 3x in bandwidth',
        'kernel 只读 .x 时，两种布局的带宽差 3 倍',
      )}
      onReset={() => setLayout('aos')}
      footer={
        <>
          {t(
            <>
              AoS isn't "wrong." If the kernel needs all of x/y/z, hauling them back together in one transaction
              is the better deal. SoA wins only when you read just some fields, which is why the tensors in deep
              learning frameworks are essentially all SoA.
            </>,
            <>
              AoS 并不「错」。如果 kernel 同时要 x/y/z 三个字段，一个事务里顺便全带回来反而划算。只读
              部分字段时 SoA 才是王道，这也是为什么深度学习框架里的 tensor 本质上全是 SoA。
            </>,
          )}
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <Segmented<Layout>
          options={[
            { value: 'aos', label: t('AoS array of structs', 'AoS 数组的结构体') },
            { value: 'soa', label: t('SoA struct of arrays', 'SoA 结构体的数组') },
          ]}
          value={layout}
          onChange={setLayout}
        />
        <span className="font-mono text-[11px] text-ink3">
          {layout === 'aos'
            ? t('p[i].x — neighboring threads 12 B apart', 'p[i].x —— 相邻线程地址差 12 B')
            : t('p.x[i] — neighboring threads 4 B apart', 'p.x[i] —— 相邻线程地址差 4 B')}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${VH}`} className="mt-4 w-full font-mono">
        <g className="text-ink3" fontSize={10} fill="currentColor">
          <text x={0} y={12}>
            {t('WARP: 32 LANES, each reads its own particle .x (4 B)', 'WARP：32 LANE，每个读自己粒子的 .x（4 B）')}
          </text>
          <text x={0} y={STRIP_Y - 10}>
            {layout === 'aos'
              ? t(
                  'Layout: x y z x y z ... (32 Particle structs, 384 B total)',
                  '内存布局：x y z x y z …（32 个 Particle 结构体，共 384 B）',
                )
              : t(
                  'Layout: x*32 | y*32 | z*32 (three independent arrays, 384 B total)',
                  '内存布局：x×32 ｜ y×32 ｜ z×32（三条独立数组，共 384 B）',
                )}
          </text>
          <text x={0} y={STRIP_Y + STRIP_H + 16}>
            0 B
          </text>
          <text x={W} y={STRIP_Y + STRIP_H + 16} textAnchor="end">
            384 B
          </text>
        </g>

        {/* lane → x 字段连线 */}
        <g>
          {Array.from({ length: LANES }, (_, i) => (
            <line
              key={i}
              x1={laneX(i)}
              y1={LANE_Y + 8}
              x2={(xCellOf(layout, i) + 0.5) * cellW}
              y2={STRIP_Y}
              stroke="var(--color-cyan)"
              strokeOpacity={0.42}
              strokeWidth={1}
            />
          ))}
        </g>

        {/* 96 个 4B 格子，按字段着色 */}
        <g>
          {Array.from({ length: CELLS }, (_, c) => {
            const f = fieldOf(c)
            return (
              <rect
                key={c}
                x={c * cellW}
                y={STRIP_Y + 1.5}
                width={cellW - 0.5}
                height={STRIP_H - 3}
                fill={FIELD_COLORS[f]}
                fillOpacity={FIELD_ALPHA[f]}
              />
            )
          })}
        </g>

        {/* 被读取的 x 格子（实心 cyan） */}
        <g>
          {Array.from({ length: LANES }, (_, i) => (
            <rect
              key={i}
              x={xCellOf(layout, i) * cellW}
              y={STRIP_Y + 1.5}
              width={cellW - 0.5}
              height={STRIP_H - 3}
              fill="var(--color-cyan)"
              fillOpacity={0.95}
            />
          ))}
        </g>

        {/* 32B 段边框，触及的填 amber */}
        <g>
          {Array.from({ length: SEGS }, (_, s) => {
            const hit = segSet.has(s)
            return (
              <rect
                key={s}
                x={(s * W) / SEGS}
                y={STRIP_Y}
                width={W / SEGS}
                height={STRIP_H}
                fill={hit ? 'var(--color-amber)' : 'none'}
                fillOpacity={hit ? 0.13 : 0}
                stroke={hit ? 'var(--color-amber)' : 'var(--color-line)'}
                strokeOpacity={hit ? 0.65 : 1}
              />
            )
          })}
        </g>

        {/* lane 圆点 */}
        <g>
          {Array.from({ length: LANES }, (_, i) => (
            <g key={i}>
              <circle cx={laneX(i)} cy={LANE_Y} r={5} fill="var(--color-volt)" fillOpacity={0.85} />
              {(i % 8 === 0 || i === 31) && (
                <text
                  x={laneX(i)}
                  y={LANE_Y - 11}
                  textAnchor="middle"
                  fontSize={9}
                  fill="currentColor"
                  className="text-ink3"
                >
                  {i}
                </text>
              )}
            </g>
          ))}
        </g>
      </svg>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 font-mono text-[11px] text-ink3">
        <span className="inline-flex items-center gap-1.5">
          <i className="inline-block size-2.5 rounded-[2px]" style={{ background: 'var(--color-cyan)' }} />
          {t('x (what the kernel reads)', 'x（kernel 要读的）')}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i
            className="inline-block size-2.5 rounded-[2px] opacity-60"
            style={{ background: 'var(--color-violet)' }}
          />
          {t('y (nobody wants)', 'y（没人要）')}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i
            className="inline-block size-2.5 rounded-[2px] opacity-50"
            style={{ background: 'var(--color-ink3)' }}
          />
          {t('z (nobody wants)', 'z（没人要）')}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i
            className="inline-block size-2.5 rounded-[2px]"
            style={{ background: 'color-mix(in srgb, var(--color-amber) 35%, transparent)' }}
          />
          {t('32B transaction sector fired', '被触发的 32B 事务段')}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4">
        <Stat
          label={t('AoS transactions', 'AoS 事务数')}
          value={txnAos}
          unit="txn"
          tone={layout === 'aos' ? 'amber' : 'ink'}
          size={layout === 'aos' ? 'lg' : 'md'}
        />
        <Stat
          label={t('SoA transactions', 'SoA 事务数')}
          value={txnSoa}
          unit="txn"
          tone={layout === 'soa' ? 'volt' : 'ink'}
          size={layout === 'soa' ? 'lg' : 'md'}
        />
        <Stat
          label={t('Current bandwidth utilization', '当前带宽利用率')}
          value={`${Math.round(util * 100)}%`}
          tone={txn <= 4 ? 'volt' : 'rose'}
          size="lg"
        />
      </div>
    </Widget>
  )
}
