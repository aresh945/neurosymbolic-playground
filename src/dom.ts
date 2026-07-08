/** Minimal DOM helpers so view code stays readable without a framework. */

type Attrs = Record<string, any>;
type Child = Node | string | null | undefined | false;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs?: Attrs,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = String(v);
      else if (k === "html") node.innerHTML = String(v);
      else if (k.startsWith("on") && typeof v === "function") {
        node.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
      } else if (k === "style" && typeof v === "object") {
        Object.assign(node.style, v);
      } else if (k in node && k !== "list") {
        (node as any)[k] = v;
      } else {
        node.setAttribute(k, String(v));
      }
    }
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function clear(node: HTMLElement): void {
  node.replaceChildren();
}

export function mount(parent: HTMLElement, ...nodes: Child[]): void {
  for (const n of nodes) {
    if (n == null || n === false) continue;
    parent.append(n instanceof Node ? n : document.createTextNode(String(n)));
  }
}

