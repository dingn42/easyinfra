import { useState } from 'react'

/** 「支持的语法」可展开速查表 —— 从 src/lib/cudasim/LANGUAGE.md 精炼 */

const ROWS: { cat: string; tone: string; items: { code: string; note: string }[] }[] = [
  {
    cat: '顶层',
    tone: 'text-cyan',
    items: [
      { code: '__global__ void k(...) { }', note: '有且仅有一个核函数；无 #include / #define / 设备函数' },
      { code: 'float* / int* / float / int', note: '仅这四种参数类型；指针按同名绑定 buffer，标量来自配置区输入框' },
    ],
  },
  {
    cat: '共享内存',
    tone: 'text-amber',
    items: [
      { code: '__shared__ float s[256];', note: '函数体顶层声明；长度为 1..16384 的整数常量；每 block 一份，初值全 0' },
    ],
  },
  {
    cat: '语句',
    tone: 'text-volt',
    items: [
      { code: 'int i = 0;  float x = 1.5f;', note: '仅局部标量；不支持局部数组 / 局部指针' },
      { code: 'x = e;  x += -= *= /=  i++ i--', note: '赋值目标为变量或 arr[idx]；自增减仅作独立语句' },
      { code: 'if / else · for · while · break · continue · return;', note: '核函数为 void，return 不带值' },
      { code: '__syncthreads();', note: 'block 级屏障；线程提前 return 后其余线程等待 → 报「屏障分化」' },
    ],
  },
  {
    cat: '表达式',
    tone: 'text-ink',
    items: [
      { code: "+ - * / %  · == != < <= > >=  · && || !  · c ? a : b", note: '整数除法 / % 按 C 截断；% 不接受浮点；逻辑短路求值' },
      { code: 'A[i]', note: '一维下标读写（指针参数与 shared 数组）；越界 → 运行时错误（含线程坐标）' },
    ],
  },
  {
    cat: '内建',
    tone: 'text-violet',
    items: [
      { code: 'threadIdx / blockIdx / blockDim / gridDim (.x .y .z) · warpSize', note: 'warpSize = 32' },
      { code: 'min max abs fabsf sqrtf expf logf powf floorf sinf cosf', note: 'float 按 JS double 计算，不模拟 fp32 舍入' },
    ],
  },
  {
    cat: '不支持',
    tone: 'text-rose',
    items: [
      {
        code: '指针运算 · atomicAdd · 位运算 · double/unsigned · switch · printf · 多维下标',
        note: '超出子集 → 编译期给出带行号的友好错误',
      },
    ],
  },
]

export function SyntaxPanel() {
  const [open, setOpen] = useState(false)
  return (
    <div className="panel overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-panel2/60"
        aria-expanded={open}
      >
        <span className={`font-mono text-xs text-volt transition-transform duration-200 ${open ? 'rotate-90' : ''}`}>
          ▸
        </span>
        <span className="microlabel text-ink2">SUPPORTED SYNTAX</span>
        <span className="text-[13px] text-ink2">支持的语法 —— CUDA C 子集速查</span>
        <span className="ml-auto font-mono text-[11px] text-ink3">{open ? '收起' : '展开'}</span>
      </button>
      {open && (
        <div className="border-t border-line px-4 py-4">
          <div className="space-y-4">
            {ROWS.map((r) => (
              <div key={r.cat} className="grid gap-x-4 gap-y-1.5 sm:grid-cols-[88px_1fr]">
                <div className={`microlabel pt-0.5 ${r.tone}`}>{r.cat}</div>
                <div className="space-y-1.5">
                  {r.items.map((it) => (
                    <div key={it.code} className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                      <code className="rounded border border-line bg-bg2 px-1.5 py-0.5 font-mono text-[11.5px] text-cyan">
                        {it.code}
                      </code>
                      <span className="text-[12.5px] text-ink2">{it.note}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 border-t border-line pt-3 text-[12px] leading-relaxed text-ink3">
            执行模型：block 内线程按逻辑步同拍（lockstep）推进，warp = 连续 32 线程；同 warp 同拍的全局访问按 32B
            段聚合成事务（完美合并 = 每 warp 4 事务），shared 访问按 bank = index % 32 统计冲突。block ≤ 1024
            线程，grid 总线程 ≤ 65536。
          </p>
        </div>
      )}
    </div>
  )
}
