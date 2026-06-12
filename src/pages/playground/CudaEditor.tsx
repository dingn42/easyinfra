import { useMemo } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { cpp } from '@codemirror/lang-cpp'
import { Decoration, EditorView, type DecorationSet } from '@codemirror/view'
import { StateField, type Extension } from '@codemirror/state'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { tags as t } from '@lezer/highlight'

/** 与设计系统一致的深色编辑器主题（背景 = --color-panel，等宽 13px） */
const eiTheme = EditorView.theme(
  {
    '&': {
      backgroundColor: 'var(--color-panel)',
      color: 'var(--color-text)',
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
      { backgroundColor: 'rgba(89, 216, 234, 0.16)' },
    '.cm-activeLine': { backgroundColor: 'rgba(184, 245, 61, 0.045)' },
    '.cm-gutters': {
      backgroundColor: 'var(--color-panel)',
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
      backgroundColor: 'rgba(255, 92, 122, 0.12)',
      boxShadow: 'inset 2px 0 0 var(--color-rose)',
    },
    '.cm-matchingBracket, &.cm-focused .cm-matchingBracket': {
      backgroundColor: 'rgba(184, 245, 61, 0.12)',
      outline: '1px solid rgba(184, 245, 61, 0.3)',
    },
  },
  { dark: true },
)

/** 语法配色：与 index.css 的 .tok-* 静态高亮调色板对齐 */
const eiHighlight = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.modifier], color: '#c792ea' },
  { tag: [t.typeName, t.standard(t.tagName)], color: '#59d8ea' },
  { tag: t.number, color: '#f78c6c' },
  { tag: [t.string, t.character], color: '#ecc48d' },
  { tag: [t.comment, t.blockComment, t.lineComment], color: '#5d6985', fontStyle: 'italic' },
  { tag: [t.function(t.variableName), t.macroName], color: '#82aaff' },
  { tag: [t.operator, t.punctuation, t.separator, t.bracket], color: '#8a96b0' },
  { tag: [t.labelName, t.attributeName], color: '#b8f53d' },
  { tag: t.variableName, color: '#b9c3d9' },
  { tag: [t.bool, t.null, t.atom], color: '#f78c6c' },
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
