import { lazy, type ComponentType, type LazyExoticComponent } from 'react'

/** 章节 id → 懒加载组件。每章一个目录，default export 章节正文组件。 */
export const CHAPTER_COMPONENTS: Record<string, LazyExoticComponent<ComponentType>> = {
  'why-gpu': lazy(() => import('./01-why-gpu')),
  'gpu-architecture': lazy(() => import('./02-gpu-architecture')),
  'cuda-model': lazy(() => import('./03-cuda-model')),
  memory: lazy(() => import('./04-memory')),
  matmul: lazy(() => import('./05-matmul')),
  roofline: lazy(() => import('./06-roofline')),
  transformer: lazy(() => import('./07-transformer')),
  'flash-attention': lazy(() => import('./08-flash-attention')),
  'kv-cache': lazy(() => import('./09-kv-cache')),
  serving: lazy(() => import('./10-serving')),
  quantization: lazy(() => import('./11-quantization')),
  parallelism: lazy(() => import('./12-parallelism')),
}
