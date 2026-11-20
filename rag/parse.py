
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    file_path: str
    language: str
    kind: str            
    name: str          
    parent: str         
    start_line: int      
    end_line: int        
    content: str

    @property
    def embed_text(self) -> str:
        """Text that gets embedded: content prefixed with a small metadata header."""
        head = f"File: {self.file_path}"
        if self.parent:
            head += f" | Class: {self.parent}"
        if self.name:
            head += f" | {self.kind}: {self.name}"
        return f"{head}\n{self.content}"



# Sliding-window fallback


def window_split(lines: list[str], offset: int, max_chars: int, overlap: int) -> list[tuple[int, int, str]]:
    """Split lines into windows of <= max_chars (a single longer line gets its own window).
    Returns (start_line, end_line, text), 1-based; `offset` is the 0-based index of lines[0]."""
    out: list[tuple[int, int, str]] = []
    n = len(lines)
    i = 0
    while i < n:
        size, j = 0, i
        while j < n and (size + len(lines[j]) + 1 <= max_chars or j == i):
            size += len(lines[j]) + 1
            j += 1
        text = "\n".join(lines[i:j])
        if text.strip():
            out.append((offset + i + 1, offset + j, text))
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return out


def _chunk_markdown(path: str, lang: str, lines: list[str], max_chars: int, overlap: int) -> list[Chunk]:
    """Split docs by headings, then window-split any oversized section."""
    heading = re.compile(r"^#{1,6}\s+(.*)")
    sections: list[tuple[int, int, str]] = []  
    start, title = 0, ""
    for idx, line in enumerate(lines):
        m = heading.match(line)
        if m and idx > start:
            sections.append((start, idx, title))
            start, title = idx, m.group(1).strip()
        elif m:
            title = m.group(1).strip()
    sections.append((start, len(lines), title))

    chunks: list[Chunk] = []
    for s, e, t in sections:
        for ws, we, text in window_split(lines[s:e], s, max_chars, overlap):
            chunks.append(Chunk(path, lang, "doc", t, "", ws, we, text))
    return chunks



# Tree-sitter plumbing 



_GRAMMARS: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
}


_DEF_TYPES: dict[str, dict[str, str]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {"function_declaration": "function", "generator_function_declaration": "function",
                   "class_declaration": "class", "method_definition": "method"},
    "typescript": {"function_declaration": "function", "generator_function_declaration": "function",
                   "class_declaration": "class", "method_definition": "method",
                   "interface_declaration": "class", "abstract_class_declaration": "class"},
    "java": {"class_declaration": "class", "interface_declaration": "class", "enum_declaration": "class",
             "record_declaration": "class", "method_declaration": "method", "constructor_declaration": "method"},
    "go": {"function_declaration": "function", "method_declaration": "method", "type_declaration": "class"},
}
_DEF_TYPES["tsx"] = _DEF_TYPES["typescript"]


_WRAPPERS = {"decorated_definition": "definition", "export_statement": "declaration"}

_parsers: dict[str, object] = {}


def _get_parser(language: str):
    """Return a tree-sitter Parser for `language`, or None if unavailable."""
    if language in _parsers:
        return _parsers[language]
    parser = None
    spec = _GRAMMARS.get(language)
    if spec:
        try:
            import importlib
            from tree_sitter import Language, Parser
            mod = importlib.import_module(spec[0])
            parser = Parser(Language(getattr(mod, spec[1])()))
        except Exception:
            parser = None
    _parsers[language] = parser
    return parser


def supported_languages() -> list[str]:
    return [l for l in _GRAMMARS if _get_parser(l) is not None]


def _node_name(node, source: bytes) -> str:
    n = node.child_by_field_name("name")
    if n is None and node.type == "type_declaration":     
        for c in node.children:
            if c.type == "type_spec":
                n = c.child_by_field_name("name")
                break
    if n is None:
        return ""
    return source[n.start_byte:n.end_byte].decode("utf8", "replace")


def _unwrap(node):
    """If node is a decorator/export wrapper, return the inner definition node (else node)."""
    field = _WRAPPERS.get(node.type)
    if field:
        inner = node.child_by_field_name(field)
        if inner is not None:
            return inner
    return node


def _arrow_fn_name(node, source: bytes) -> Optional[str]:
    """`const foo = () => {}` / `const foo = function() {}` -> 'foo' (JS/TS)."""
    decl = node
    if node.type == "export_statement":
        decl = node.child_by_field_name("declaration") or node
    if decl.type not in ("lexical_declaration", "variable_declaration"):
        return None
    for c in decl.children:
        if c.type == "variable_declarator":
            val = c.child_by_field_name("value")
            if val is not None and val.type in ("arrow_function", "function_expression", "function"):
                return _node_name(c, source)
    return None


def _chunk_with_treesitter(path: str, language: str, source: str, max_chars: int, overlap: int) -> Optional[list[Chunk]]:
    parser = _get_parser(language)
    if parser is None:
        return None
    src_bytes = source.encode("utf8")
    try:
        tree = parser.parse(src_bytes)
    except Exception:
        return None
    lines = source.split("\n")
    def_types = _DEF_TYPES[language]

    defs: list[tuple[int, int, str, str, str]] = [] 

    def emit_def(node, parent: str):
        inner = _unwrap(node)
        kind = def_types.get(inner.type)
        name = _node_name(inner, src_bytes) if kind else ""
        s, e = node.start_point[0], node.end_point[0]
        size = sum(len(l) + 1 for l in lines[s:e + 1])
        if kind == "class" and size > max_chars:
           
            members = []
            body = inner.child_by_field_name("body") or inner
            for child in body.children:
                if _unwrap(child).type in def_types:
                    members.append(child)
            if members:
                first = members[0].start_point[0]
                if first > s:
                    defs.append((s, first - 1, "class", name, parent))
                for m in members:
                    emit_def(m, name)
                last = members[-1].end_point[0]
                if e > last:
                    defs.append((last + 1, e, "class", name, parent))
                return
        kind = kind or "function"
        if parent and kind == "function":
            kind = "method"         
        defs.append((s, e, kind, name, parent))

    def walk_top(node):
        for child in node.children:
            inner = _unwrap(child)
            if inner.type in def_types:
                emit_def(child, "")
            elif language in ("javascript", "typescript", "tsx") and _arrow_fn_name(child, src_bytes):
                s, e = child.start_point[0], child.end_point[0]
                defs.append((s, e, "function", _arrow_fn_name(child, src_bytes) or "", ""))
            elif child.type in ("namespace_definition", "internal_module", "module"):
                walk_top(child)

    walk_top(tree.root_node)
    if not defs:
        return None 

    defs.sort(key=lambda d: (d[0], d[1]))
    chunks: list[Chunk] = []

    def add_range(s0: int, e0: int, kind: str, name: str, parent: str):
        seg = lines[s0:e0 + 1]
        if not "\n".join(seg).strip():
            return
        for ws, we, text in window_split(seg, s0, max_chars, overlap):
            chunks.append(Chunk(path, language, kind, name, parent, ws, we, text))

    cursor = 0  # first line not yet covered
    for s0, e0, kind, name, parent in defs:
        if s0 > cursor:
            add_range(cursor, s0 - 1, "module", "", "")  
        if e0 >= cursor:
            add_range(max(s0, cursor), e0, kind, name, parent)
            cursor = e0 + 1
    if cursor < len(lines):
        add_range(cursor, len(lines) - 1, "module", "", "")
    return chunks



# Public API


def chunk_file(path: str, language: str, text: str, is_doc: bool,
               max_chars: int = 2500, overlap: int = 3) -> list[Chunk]:
    """Turn one file into metadata-tagged chunks."""
    if not text.strip():
        return []
    lines = text.split("\n")
    if is_doc:
        if language == "markdown":
            return _chunk_markdown(path, language, lines, max_chars, overlap)
        return [Chunk(path, language, "doc", "", "", s, e, t)
                for s, e, t in window_split(lines, 0, max_chars, overlap)]
    if language in _GRAMMARS:
        res = _chunk_with_treesitter(path, language, text, max_chars, overlap)
        if res:
            return res
    return [Chunk(path, language, "window", "", "", s, e, t)
            for s, e, t in window_split(lines, 0, max_chars, overlap)]
