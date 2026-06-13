# EasyInfra 设计系统与贡献规范

> **所有章节构建者必读。** 本站是交互式 CUDA / LLM Infra 课程，美学方向是「蓝图纸 / 精密仪器」：
> **浅色暖白底（blueprint paper）**、白色卡片、深饱和强调色、等宽微标签、克制而精确。一切组件都应该像实验台上的仪器。
> 全站**双语**：英文默认、中文可选（见下「i18n 契约」）。

## 双语 i18n 契约（必读）

- `import { useT } from '@/lib/i18n'`；组件内 `const t = useT()`；`{t('English', '中文')}` —— 返回二选一，string / ReactNode 均可。
- 数据用 `Loc<T> = { en, zh }`，`import { pick, useLocale } from '@/lib/i18n'`，`pick(field, lang)` 解析。
- 英文是默认且必须地道流畅（母语 GPU 工程师笔触，非直译）；中文作为第二参数。
- 代码注释可保留中文；CodeBlock 的 `code` 若含中文注释，用 `code={t(enCode, zhCode)}` 双语化。
- **任何用户可见的中文都不能裸露在 JSX**，必须是 `t()` 的第二参数或 `Loc.zh`。

## 技术栈

- Vite + React 18 + TypeScript（strict）+ Tailwind CSS v4 + react-router-dom
- 公式：`katex`（用共享组件 `MathTex`）
- 可编辑代码：`@uiw/react-codemirror`（仅在确需编辑时用；静态代码一律用 `CodeBlock`）
- 动画：优先 CSS transition / 自写 rAF（`useRafLoop`）；`motion`（framer）已安装可用但非必需
- **禁止引入任何新 npm 依赖。** 可视化用 SVG（首选）或 Canvas 手写。

## 设计 token（Tailwind 类名直接可用）

| 类名 | 用途 |
|---|---|
| `bg-bg` `bg-bg2` `bg-panel` `bg-panel2` | 页面底 / 略亮底 / 卡片 / 卡片高亮区 |
| `border-line` `border-line2` | 1px 边框（暗 / 亮） |
| `text-ink` `text-text` `text-ink2` `text-ink3` | 近黑标题 / 正文 / 次要 / 微弱（**浅色主题**：都是深灰，在白底上可读） |
| `text-volt` (#1c8a3f 深绿) | **主强调色**：活跃、正确、关键数据。少量使用才有力量 |
| `text-cyan` (#0e78a8) | 数据流、链接、次强调 / 读 |
| `text-amber` (#b06d09) | 内存 / 警示 / 热 / 写 |
| `text-rose` (#c92a4d) | 错误、冲突、分化 |
| `text-violet` (#6a3bd0) | 深入内容、特殊标记 |
| `font-display` | Chakra Petch，标题用 |
| `font-mono` | JetBrains Mono，代码 / 数字 / 微标签 |
| 工具类 | `.microlabel`（等宽小标签）`.panel`（白卡片）`.bg-dots`（点网格底）`.lift`/`.glow-volt`（柔和浅色阴影）`.ei-range`（Slider 封装） |

**浅色主题色值**：底 `#f7f8f5` 暖白、卡片 `#ffffff`、线 `#e4e6df`。canvas / 内联 JS 颜色用 `import { C, rgba } from '@/lib/palette'`（与 @theme 同步）。

**视觉铁律**：volt 绿是稀缺资源，每屏少量点缀；数字一律 `font-mono` + `tabular-nums`；边框 1px、圆角 8-10px；
浅色下用**柔和低阴影**（`.lift`/`.glow-volt`），禁止深色大色块与花哨阴影；SVG 文字用 `fill="currentColor"` 配合 Tailwind 文本色类，**绝不硬编码深色字面量**（在白底会刺眼/看不清）。

## 共享组件（从 `@/components/ui` 导入）

```tsx
import { Section, Callout, CodeBlock, Widget, Quiz, MathTex, Term, Figure,
         Slider, Segmented, Toggle, Btn, Stat, PlayBar } from '@/components/ui'

<Section index={1} title="标题" lead="可选导语">…</Section>   // 章内每个 section 必须用它
<Callout type="insight|note|warn|deep" title="可选">…</Callout>
<CodeBlock code={src} lang="cuda|cpp|python|js|bash" title="vec_add.cu" highlight={[3,4]} />
<Widget index={1} title="实验名" subtitle="副题" onReset={fn} footer={<>说明</>} wide>…</Widget>
<Quiz question="…" options={[{ text:'…', correct:true, explain:'…' }, …]} />
<MathTex tex="O(S^2)" /> <MathTex block tex="…" />
<Term t="warp">32 个线程组成的调度单位…</Term>
<Slider label="STRIDE" value={v} min={1} max={32} onChange={setV} fmt={(x)=>`${x}`} unit="元素" />
<Segmented options={[{value:'a',label:'A'}]} value={v} onChange={setV} />
<Stat label="TRANSACTIONS" value={4} unit="txn" tone="volt" size="lg" />
<PlayBar playing={p} onToggle={…} onStep={…} onReset={…} speed={s} onSpeed={setS} />
```

辅助：`@/lib/hooks`（`useRafLoop` `useInterval` `useInView` `useMeasure` `useReducedMotion` `useLocalStorage`）、
`@/lib/format`（`fmtSI` `fmtBytes` `fmtFlops` `fmtInt` `pct` `clamp` `lerp`）。

## 章节文件契约

- 每章只允许写 `src/chapters/<NN-slug>/` 目录内的文件；`index.tsx` 默认导出正文组件。
- **不得修改任何共享文件**（`src/components/**`、`src/lib/**`、`src/pages/**`、registry 等）。
- 页面外框（章标题、上一章/下一章）由 ChapterPage 提供，章节组件只写正文。
- 正文已包在 `.doc`（自带排版样式）和 `max-w-[820px]` 容器内；宽组件用 `<Widget wide>`。

## 内容写作规范

- **中文正文**，技术名词首次出现给英文（如「线程束（warp）」），之后可直接用英文术语。
- 语气：清晰、具体、带洞察 —— 像一位愿意把直觉讲给你听的资深工程师。拒绝教科书腔和空话。
- 结构：开篇一段「钩子」（一个具体问题/反直觉事实，不用 Section 包裹，直接 `<p>`）→ 4-7 个
  `<Section>`（编号 index 从 1 连续递增）→ 最后一个 Section 是「总结与延伸阅读」（要点回顾 +
  3-5 个真实外链）。
- 每章 ≥ 2 个 `<Widget>` 交互实验（按 registry 中 labs 列表实现）、≥ 2 个 `<Quiz>`、
  ≥ 1 个 `<Callout type="insight">`。正文（不含代码）≥ 2500 字。
- 数字要真实：延迟、带宽、FLOPS 用真实数量级（如 A100：1.9TB/s HBM、312 TFLOPS BF16；
  H100：3.35TB/s、989 TFLOPS）。不确定的写"约"。

## 交互组件质量标准

- 默认状态即有意义：打开页面不用操作也能看懂在演示什么。
- 滑杆/开关改变时，可视化和数字**即时**响应；动画提供 PlayBar（播放/单步/重置）。
- 所有动画用 `useRafLoop` 或 CSS；`useReducedMotion()` 为 true 时跳过纯装饰动画。
- SVG 响应式：`useMeasure` 拿容器宽度或 `viewBox` + `w-full`。移动端（容器 ~350px）不烂版。
- 颜色语义一致：volt=活跃/正确，cyan=数据/读，amber=内存/写/热，rose=错误/冲突。
