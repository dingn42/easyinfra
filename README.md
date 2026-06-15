# EasyInfra

**Learn GPUs, CUDA, and LLM inference systems — interactively.** Take a GPU apart, write CUDA, and
build an inference system, all in your browser. 12 chapters, 30+ interactive labs, and a CUDA simulator
that actually executes code. Zero setup. Bilingual (English / 中文).

[![Live demo](https://img.shields.io/badge/▲_live-easyinfra.vercel.app-1c8a3f?style=flat-square)](https://easyinfra.vercel.app)
[![Stack](https://img.shields.io/badge/Vite_·_React_·_TS_·_Tailwind-0e78a8?style=flat-square)](#tech-stack--code-map)

🔗 **Live:** https://easyinfra.vercel.app  ·  **Source:** https://github.com/dingn42/easyinfra

> English is the primary language; a 中文 (Chinese) version is one click away in-app via the **EN / 中**
> toggle, and a condensed Chinese README is at the [bottom of this file](#中文版--chinese).

---

## What's inside

- **12 chapters, 3 parts** — from "why GPUs exist" to FlashAttention, PagedAttention, and distributed parallelism.
- **30+ interactive labs** — hand-built SVG/Canvas visualizations you can poke: warp scheduling, memory
  coalescing, tiled matmul, the Roofline model, KV-cache paging, continuous batching, float bit layouts, and more.
- **A real CUDA simulator** — a CUDA-C-subset interpreter in pure TypeScript (`src/lib/cudasim/`). Write a
  kernel, set the grid/block, single-step it, and watch every memory access — with coalescing, bank
  conflicts, and divergence counted, and errors pinpointed to the exact thread.
- **Glossary + cross-links** — every key term defined once and linked back to the chapter that introduces it.
- **Bilingual & light, by design** — clean "blueprint" light theme; English default with a Chinese toggle.

## Curriculum

| Part | Chapters |
|---|---|
| **I · GPUs & Parallel Computing** | 01 Why GPUs? · 02 Anatomy of a GPU |
| **II · CUDA Programming** | 03 The CUDA Model · 04 Memory Is King · 05 Making Matmul Fast · 06 Roofline & Profiling |
| **III · LLM Inference Systems** | 07 Inside the Transformer · 08 FlashAttention · 09 KV Cache & PagedAttention · 10 Serving Systems · 11 Quantization · 12 Distributed Parallelism |

Plus the **CUDA Playground** (`/playground`) and a searchable **Glossary** (`/glossary`).

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + production build → dist/ (pure static, host anywhere)
```

## Deploy

Live on **Vercel** at **https://easyinfra.vercel.app**. The app is a static SPA with client-side routing,
so `vercel.json` rewrites every path to `index.html` (deep links like `/learn/why-gpu` and `/glossary`
work on direct visit / refresh). Redeploy from a local checkout with:

```bash
npx vercel@latest --prod --yes
```

Pushing to `main` on GitHub also triggers an automatic production deploy once the repo is connected to the
Vercel project (Project → Settings → Git).

## Tech stack & code map

Vite + React 18 + TypeScript (strict) + Tailwind CSS v4 · KaTeX · CodeMirror 6 · react-router

```
src/
├── lib/
│   ├── cudasim/        # CUDA-C-subset interpreter (lexer/parser/interp; spec in LANGUAGE.md)
│   ├── chapters.ts     # curriculum metadata — the single registry (bilingual)
│   ├── i18n.tsx        # locale provider + useT() / pick() bilingual layer
│   ├── glossary.ts     # glossary terms (bilingual, linked to chapters)
│   ├── hardware.ts     # canonical reference-GPU specs (A100 / H100 / RTX 4090)
│   ├── palette.ts      # JS color tokens for canvas/CodeMirror (mirrors the CSS theme)
│   └── hooks.ts …      # rAF loop, viewport measurement, shared hooks
├── components/
│   ├── ui/             # design system: Section/Widget/Quiz/CodeBlock/MathTex/ChapterLink/HardwareBaseline/…
│   └── layout/         # sidebar + app shell
├── chapters/01-… 12-…  # one dir per chapter: prose + its own interactive labs
└── pages/              # Home / ChapterPage / Playground / Glossary
```

Design language and contribution conventions live in [DESIGN.md](DESIGN.md).

---

<details>
<summary><b>中文版 / Chinese</b></summary>

<br>

# EasyInfra

**交互式学习 GPU、CUDA 与大模型推理系统。** 在浏览器里拆开 GPU、写 CUDA、搭推理系统 ——
12 章内容、30+ 个可交互实验、一个能真正执行代码的 CUDA 模拟器，零环境配置，中英双语。

🔗 **线上访问：** https://easyinfra.vercel.app  ·  **源码：** https://github.com/dingn42/easyinfra

> 站内默认英文，点右上角 **EN / 中** 即可切换中文。

## 内容概览

- **12 章 · 3 个部分** —— 从"为什么需要 GPU"一路到 FlashAttention、PagedAttention 与分布式并行。
- **30+ 个交互实验** —— 全手写 SVG/Canvas，可拨弄：warp 调度、合并访存、tiled 矩阵乘、Roofline、
  KV Cache 分页、连续批处理、浮点位拆解等。
- **一个真正的 CUDA 模拟器** —— 纯 TypeScript 的 CUDA C 子集解释器（`src/lib/cudasim/`）：写 kernel、
  配置 grid/block、单步执行、观察每一次内存访问，统计合并事务、bank conflict、分支分化，错误精确到线程。
- **术语表 + 跨章链接** —— 每个关键术语定义一次，并链回引入它的章节。
- **浅色 + 双语设计** —— 干净的"蓝图"浅色主题；英文默认、可切中文。

## 课程结构

| 部分 | 章节 |
|---|---|
| **I · GPU 与并行计算** | 01 为什么需要 GPU · 02 GPU 硬件解剖 |
| **II · CUDA 编程** | 03 CUDA 编程模型 · 04 访存为王 · 05 实战矩阵乘法 · 06 Roofline 与性能分析 |
| **III · 大模型推理系统** | 07 Transformer 计算解剖 · 08 FlashAttention · 09 KV Cache 与 PagedAttention · 10 推理服务系统 · 11 量化 · 12 分布式并行 |

另有 **CUDA Playground**（`/playground`）与可搜索的**术语表**（`/glossary`）。

## 本地运行

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # 类型检查 + 生产构建 → dist/（纯静态，可任意托管）
```

## 部署

已部署在 **Vercel**：**https://easyinfra.vercel.app**。本应用是带客户端路由的静态 SPA，`vercel.json`
把所有路径 rewrite 到 `index.html`，因此 `/learn/why-gpu`、`/glossary` 等深链直接访问/刷新都不会 404。
从本地重新发布：`npx vercel@latest --prod --yes`；仓库与 Vercel 项目连通后，push 到 `main` 即自动部署。

技术栈：Vite + React 18 + TypeScript (strict) + Tailwind CSS v4 · KaTeX · CodeMirror 6。
设计语言与贡献规范见 [DESIGN.md](DESIGN.md)。

</details>
