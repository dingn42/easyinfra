import { useMemo } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { cpp } from '@codemirror/lang-cpp'
import { Decoration, EditorView, type DecorationSet } from '@codemirror/view'
import { StateField, type Extension } from '@codemirror/state'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { tags as t } from '@lezer/highlight'

/** 与设计系统一致的浅色编辑器主题（背景 = --color-panel，等宽 13px，深色正文） */
const eiTheme = EditorView.theme(
  {
    '&': {
      backgroundColor: 'var(--color-panel)',
      color: 'var(--color-ink)',
      fontSize: '13px',
    },
    '.cm-scroller': {
      fontFamily: 'var(--font-mono)',
      lineHeight: '1.75',
    },
    '.cm-content': {
      padding: '12px 0',
      caretColor: 'var(--color-volt)',
    },
    '&.cm-focused': { outline: 'none' },
    '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--color-volt)' },
    '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground':
      { backgroundColor: 'color-mix(in srgb, var(--color-volt) 15%, transparent)' },
    '.cm-activeLine': { backgroundColor: 'color-mix(in srgb, var(--color-volt) 5%, transparent)' },
    '.cm-gutters': {
      backgroundColor: 'var(--color-panel2)',
      color: 'var(--color-ink3)',
      border: 'none',
      borderRight: '1px solid var(--color-line)',
      fontSize: '11px',
    },
    '.cm-gutterElement': { padding: '0 10px 0 14px' },
    '.cm-activeLineGutter': {
      backgroundColor: 'transparent',
      color: 'var(--color-ink2)',
    },
    '.cm-errorLine': {
      backgroundColor: 'color-mix(in srgb, var(--color-rose) 11%, transparent)',
      boxShadow: 'inset 2px 0 0 var(--color-rose)',
    },
    '.cm-matchingBracket, &.cm-focused .cm-matchingBracket': {
      backgroundColor: 'color-mix(in srgb, var(--color-volt) 14%, transparent)',
      outline: '1px solid color-mix(in srgb, var(--color-volt) 35%, transparent)',
    },
  },
  { dark: false },
)

/** 语法配色：与 index.css 的 .tok-* 浅色（One-Light 风）静态高亮调色板对齐 */
const eiHighlight = HighlightStyle.define([
  // keyword #a626a4
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.modifier], color: '#a626a4' },
  // type #0184bc
  { tag: [t.typeName, t.standard(t.tagName)], color: '#0184bc' },
  // number #b06d09
  { tag: t.number, color: '#b06d09' },
  // string #3f8a2e
  { tag: [t.string, t.character], color: '#3f8a2e' },
  // comment #9aa1ab italic
  { tag: [t.comment, t.blockComment, t.lineComment], color: '#9aa1ab', fontStyle: 'italic' },
  // function #4969e0
  { tag: [t.function(t.variableName), t.macroName], color: '#4969e0' },
  // operator / punct #6a737d
  { tag: [t.operator, t.punctuation, t.separator, t.bracket], color: '#6a737d' },
  // CUDA 内建（labelName/attributeName 落到这里）→ #1c8a3f 粗体
  { tag: [t.labelName, t.attributeName], color: '#1c8a3f', fontWeight: '600' },
  // 变量 → ink #15181c
  { tag: t.variableName, color: '#15181c' },
  // bool/null/atom → 与 number 同色系
  { tag: [t.bool, t.null, t.atom], color: '#b06d09' },
])

/** 高亮指定行（编译/运行时错误定位）；用户一编辑即清除 */
function errorLineField(line: number): Extension {
  return StateField.define<DecorationSet>({
    create(state) {
      if (line < 1 || line > state.doc.lines) return Decoration.none
      return Decoration.set([Decoration.line({ class: 'cm-errorLine' }).range(state.doc.line(line).from)])
    },
    update(deco, tr) {
      if (tr.docChanged) return Decoration.none
      return deco
    },
    provide: (f) => EditorView.decorations.from(f),
  })
}

export function CudaEditor({
  value,
  onChange,
  errorLine,
}: {
  value: string
  onChange: (v: string) => void
  errorLine: number | null
}) {
  const extensions = useMemo<Extension[]>(() => {
    const exts: Extension[] = [cpp(), eiTheme, syntaxHighlighting(eiHighlight)]
    if (errorLine != null) exts.push(errorLineField(errorLine))
    return exts
  }, [errorLine])

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-panel">
      <CodeMirror
        value={value}
        onChange={onChange}
        theme="none"
        extensions={extensions}
        basicSetup={{
          foldGutter: false,
          autocompletion: false,
          highlightActiveLine: true,
          highlightActiveLineGutter: true,
          searchKeymap: false,
        }}
        style={{ minHeight: 240 }}
      />
    </div>
  )
}
