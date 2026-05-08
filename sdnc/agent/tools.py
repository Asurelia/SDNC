"""Tool layer for the autonomous SDNC loop."""

from __future__ import annotations

import ast
import html
import json
import math
import operator
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

from sdnc.agent.memory import PersistentMemory
from sdnc.agent.types import ToolResult


class Tool(Protocol):
    name: str
    description: str

    def run(self, query: str, context: dict[str, Any]) -> ToolResult:
        ...


class ToolRegistry:
    """Runtime registry for real tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def run(self, name: str, query: str, context: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(name, False, f"unknown tool: {name}")
        try:
            return tool.run(query, context)
        except Exception as exc:
            return ToolResult(name, False, f"{type(exc).__name__}: {exc}")


class MemoryRecallTool:
    name = "memory_recall"
    description = "Retrieve similar past episodes from persistent memory."

    def __init__(self, memory: PersistentMemory, top_k: int = 5):
        self.memory = memory
        self.top_k = top_k

    def run(self, query: str, context: dict[str, Any]) -> ToolResult:
        embedding = context.get("embedding")
        if embedding is None:
            return ToolResult(self.name, False, "missing embedding")
        records = self.memory.retrieve_similar(embedding, top_k=self.top_k)
        if not records:
            return ToolResult(self.name, True, "no similar memories yet", {"count": 0})
        lines = [
            f"{idx + 1}. sim={rec.similarity:.3f} salience={rec.salience:.3f}: {rec.text[:180]}"
            for idx, rec in enumerate(records)
        ]
        return ToolResult(self.name, True, "\n".join(lines), {"count": len(records)})


class FileSearchTool:
    name = "file_search"
    description = "Search text files inside the configured workspace."

    def __init__(self, root: Path, limit: int = 5):
        self.root = Path(root).resolve()
        self.limit = limit
        self.extensions = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"}

    def run(self, query: str, context: dict[str, Any]) -> ToolResult:
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_]{3,}", query)]
        if not terms:
            return ToolResult(self.name, False, "no searchable terms")

        direct = self._direct_path_matches(query)
        if direct:
            return ToolResult(
                self.name,
                True,
                "\n".join(direct[: self.limit]),
                {"count": len(direct), "direct": True},
            )

        normalized_query = query.replace("\\", "/").lower()
        scored_matches: list[tuple[int, str]] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            if any(
                part in {".git", "venv", "__pycache__", ".pytest_cache", "data", "checkpoints"}
                for part in path.parts
            ):
                continue
            try:
                if path.stat().st_size > 500_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lowered = text.lower()
            rel = path.relative_to(self.root)
            lowered_path = str(rel).replace("\\", "/").lower()
            score = sum(1 for term in terms if term in lowered or term in lowered_path)
            if lowered_path in normalized_query:
                score += 10
            elif path.name.lower() in normalized_query:
                score += 6
            if score:
                scored_matches.append((score, str(rel)))

        if not scored_matches:
            return ToolResult(self.name, True, "no file matches", {"count": 0})
        ranked = sorted(scored_matches, key=lambda item: (-item[0], item[1].lower()))
        matches = [f"{rel} score={score}" for score, rel in ranked[: self.limit]]
        return ToolResult(self.name, True, "\n".join(matches), {"count": len(matches)})

    def _direct_path_matches(self, query: str) -> list[str]:
        matches: list[str] = []
        for raw in re.findall(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+", query):
            candidate = Path(raw)
            path = candidate if candidate.is_absolute() else self.root / candidate
            path = path.resolve()
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                continue
            if path.is_file() and path.suffix.lower() in self.extensions:
                matches.append(f"{rel} score=direct")
        return matches


class FileReadTool:
    name = "file_read"
    description = "Read one workspace file by relative path."

    def __init__(self, root: Path, max_chars: int = 8000):
        self.root = Path(root).resolve()
        self.max_chars = max_chars

    def run(self, query: str, context: dict[str, Any]) -> ToolResult:
        match = re.search(r"([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)", query)
        if not match:
            return ToolResult(self.name, False, "no file path found")
        raw = match.group(1)
        candidate = Path(raw)
        path = candidate if candidate.is_absolute() else self.root / candidate
        path = path.resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return ToolResult(self.name, False, "path is outside workspace")
        if not path.exists() or not path.is_file():
            return ToolResult(self.name, False, "file not found")
        text = path.read_text(encoding="utf-8", errors="ignore")
        return ToolResult(
            self.name,
            True,
            text[: self.max_chars],
            {"path": str(path), "truncated": len(text) > self.max_chars},
        )


class WebSearchTool:
    name = "web_search"
    description = "Search the public web through DuckDuckGo HTML results."

    def __init__(self, timeout_s: float = 8.0, limit: int = 5):
        self.timeout_s = timeout_s
        self.limit = limit

    def run(self, query: str, context: dict[str, Any]) -> ToolResult:
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "SDNC/0.1 research agent"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = response.read().decode("utf-8", errors="ignore")
        parser = _DuckDuckGoParser(limit=self.limit)
        parser.feed(body)
        if not parser.results:
            return ToolResult(self.name, True, "no web results parsed", {"count": 0})
        lines = [f"{idx + 1}. {title} - {href}" for idx, (title, href) in enumerate(parser.results)]
        return ToolResult(self.name, True, "\n".join(lines), {"count": len(parser.results), "url": url})


class CalculatorTool:
    name = "calculator"
    description = "Evaluate safe arithmetic expressions."

    def run(self, query: str, context: dict[str, Any]) -> ToolResult:
        expression = self._extract_expression(query)
        if not expression:
            return ToolResult(self.name, False, "no arithmetic expression found")
        try:
            value = _SafeEval().eval(expression)
        except Exception as exc:
            return ToolResult(self.name, False, f"invalid arithmetic expression: {exc}")
        return ToolResult(self.name, True, f"{expression} = {_format_number(value)}", {"value": value})

    def _extract_expression(self, query: str) -> str:
        allowed = re.findall(r"[0-9+\-*/(). %]+", query)
        symbolic = [
            item.strip()
            for item in allowed
            if re.search(r"\d", item) and re.search(r"[+\-*/%]", item)
        ]
        if symbolic:
            expression = max(symbolic, key=len)
            return expression.replace("%", "/100")
        return self._extract_natural_expression(query)

    def _extract_natural_expression(self, query: str) -> str:
        lowered = query.lower()
        numbers = [match.group(0).replace(",", ".") for match in re.finditer(r"\d+(?:[,.]\d+)?", lowered)]
        if len(numbers) < 2:
            return ""

        subtract_markers = [
            "donne",
            "donner",
            "mange",
            "mangé",
            "perd",
            "perdu",
            "retire",
            "enleve",
            "enlève",
            "soustra",
            "moins",
            "reste",
        ]
        add_markers = [
            "ajoute",
            "ajouter",
            "gagne",
            "recois",
            "reçois",
            "recu",
            "reçu",
            "plus",
            "addition",
            "somme",
        ]
        multiply_markers = ["fois", "multiplie", "multiplié", "produit"]
        divide_markers = ["divise", "divisé", "partage", "partagé", "moitié"]

        operator_symbol = ""
        if any(marker in lowered for marker in subtract_markers):
            operator_symbol = "-"
        elif any(marker in lowered for marker in add_markers):
            operator_symbol = "+"
        elif any(marker in lowered for marker in multiply_markers):
            operator_symbol = "*"
        elif any(marker in lowered for marker in divide_markers):
            operator_symbol = "/"
        if not operator_symbol:
            return ""
        return f"{numbers[0]} {operator_symbol} {numbers[1]}"


class _DuckDuckGoParser(HTMLParser):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.results: list[tuple[str, str]] = []
        self._capture_href: str | None = None
        self._capture_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {key: value or "" for key, value in attrs}
        classes = attrs_d.get("class", "")
        if tag == "a" and "result__a" in classes and len(self.results) < self.limit:
            self._capture_href = self._clean_url(attrs_d.get("href", ""))
            self._capture_text = []

    def handle_data(self, data: str) -> None:
        if self._capture_href is not None:
            self._capture_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_href is not None:
            title = html.unescape(" ".join(self._capture_text).strip())
            if title:
                self.results.append((title, self._capture_href))
            self._capture_href = None
            self._capture_text = []

    def _clean_url(self, href: str) -> str:
        parsed = urllib.parse.urlparse(href)
        query = urllib.parse.parse_qs(parsed.query)
        if "uddg" in query:
            return query["uddg"][0]
        return href


class _SafeEval:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    names = {"pi": math.pi, "e": math.e}

    def eval(self, expression: str) -> float:
        node = ast.parse(expression, mode="eval")
        return float(self._visit(node.body))

    def _visit(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in self.names:
            return float(self.names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in self.operators:
            return float(self.operators[type(node.op)](self._visit(node.left), self._visit(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.operators:
            return float(self.operators[type(node.op)](self._visit(node.operand)))
        raise ValueError("unsupported expression")


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def tool_result_to_json(result: ToolResult) -> str:
    return json.dumps(
        {
            "tool_name": result.tool_name,
            "success": result.success,
            "content": result.content,
            "metadata": result.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
