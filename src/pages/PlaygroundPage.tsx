/** CUDA Playground —— 引擎与 UI 由构建流水线填充 */
export default function PlaygroundPage() {
  return (
    <div className="mx-auto max-w-[980px] px-6 py-20 lg:px-8">
      <div className="panel bg-dots flex flex-col items-center gap-3 px-6 py-24 text-center">
        <span className="microlabel animate-pulse text-volt">⌬ ENGINE COMPILING…</span>
        <p className="text-sm text-ink2">CUDA 模拟器正在构建，马上回来。</p>
      </div>
    </div>
  )
}
