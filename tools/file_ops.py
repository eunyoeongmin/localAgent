import os
import asyncio
import subprocess
import difflib
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

EMBEDDING_BASE_URL = "http://localhost:1234/v1"
EMBEDDING_API_KEY = "lm-studio"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 4
MAX_RETURN_CHARS = 8000

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
_embedding_client = None
_vector_store_cache = {}


def _get_embedding_client() -> OpenAIEmbeddings:
    global _embedding_client

    if _embedding_client is None:
        _embedding_client = OpenAIEmbeddings(
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
            model=EMBEDDING_MODEL,
            check_embedding_ctx_length=False,
        )
    return _embedding_client


def _read_file_text(abs_path: str) -> str:
    if abs_path.lower().endswith('.pdf'):
        return _read_pdf_text(abs_path)

    with open(abs_path, 'r', encoding='utf-8') as f:
        return f.read()


def _read_pdf_text(abs_path: str) -> str:
    reader = PdfReader(abs_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""
        cleaned_text = extracted_text.strip()
        if cleaned_text:
            pages.append(f"[Page {page_number}]\n{cleaned_text}")

    if not pages:
        raise ValueError("PDFから抽出できるテキストがありません。")

    return "\n\n".join(pages)


def _split_file_into_chunks(abs_path: str, content: str) -> list[Document]:
    base_document = Document(
        page_content=content,
        metadata={"source": abs_path},
    )
    chunks = _text_splitter.split_documents([base_document])
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_count"] = len(chunks)
    return chunks


def _build_vector_store(abs_path: str):
    content = _read_file_text(abs_path)
    chunks = _split_file_into_chunks(abs_path, content)
    vector_store = InMemoryVectorStore(_get_embedding_client())
    vector_store.add_documents(chunks)
    return content, chunks, vector_store


async def _get_cached_vector_store(abs_path: str):
    mtime = os.path.getmtime(abs_path)
    cached = _vector_store_cache.get(abs_path)
    if cached and cached["mtime"] == mtime:
        return cached["content"], cached["chunks"], cached["vector_store"]

    content, chunks, vector_store = await asyncio.to_thread(_build_vector_store, abs_path)
    _vector_store_cache[abs_path] = {
        "mtime": mtime,
        "content": content,
        "chunks": chunks,
        "vector_store": vector_store,
    }
    return content, chunks, vector_store


def _is_full_content_query(query: str) -> bool:
    normalized = query.strip().lower()
    return normalized in {"全て", "全内容", "全部", "full", "full content"}


def _join_chunks(chunks: list[Document]) -> str:
    sections = []
    total_length = 0

    for chunk in chunks:
        chunk_index = chunk.metadata.get("chunk_index", "?")
        chunk_count = chunk.metadata.get("chunk_count", "?")
        section = f"[チャンク {chunk_index}/{chunk_count}]\n{chunk.page_content.strip()}"
        extra_length = len(section) + 2
        if sections and total_length + extra_length > MAX_RETURN_CHARS:
            sections.append("[... 以下チャンク省略 ...]")
            break
        sections.append(section)
        total_length += extra_length

    return "\n\n".join(sections)


class LocalFileReadInput(BaseModel):
    path: str = Field(description="読み取るファイルのパス（絶対パスまたは現在のディレクトリ基準の相対パス）。例：'report.md'、'C:/Users/me/doc.txt'")
    query: str = Field(description="ファイルから探したい内容または質問。全内容が必要な場合は「全内容」と入力")


@tool(args_schema=LocalFileReadInput)
async def read_local_file(path: str, query: str) -> str:
    """
    ローカルファイル(txt, md, py, csvなど)をチャンク単位で分割し、
    埋め込み + ベクトル検索を通じて、クエリに関連する内容を返します。
    """
    try:
        abs_path = os.path.abspath(path)
        print(f"\n[DEBUG] ファイル読み取り: {abs_path}")
        content, chunks, vector_store = await _get_cached_vector_store(abs_path)
        print(
            f"[DEBUG] ファイルインデックス完了 - 文字数: {len(content)}, チャンク数: {len(chunks)}, モデル: {EMBEDDING_MODEL}"
        )

        if _is_full_content_query(query):
            selected_chunks = chunks
            mode = "全チャンク返却"
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
            selected_chunks = await asyncio.to_thread(retriever.invoke, query)
            mode = f"クエリベース検索: {query}"

        if not selected_chunks:
            return f"ファイルをインデックスしましたが、関連するチャンクが見つかりませんでした: {abs_path}"

        selected_text = _join_chunks(selected_chunks)
        return (
            f"[ファイル検索結果: {abs_path}]\n"
            f"- 処理方式: {mode}\n"
            f"- 埋め込みモデル: {EMBEDDING_MODEL}\n"
            f"- 総チャンク数: {len(chunks)}\n\n"
            f"{selected_text}"
        )
    except FileNotFoundError:
        return f"ファイルが見つかりません: {os.path.abspath(path)}"
    except ImportError as e:
        return f"ファイル検索用の依存関係を読み込めませんでした: {str(e)}"
    except Exception as e:
        return f"ファイル読み取り失敗: {str(e)}"


class LocalFileWriteInput(BaseModel):
    path: str = Field(description="書き込むファイルのパス（絶対パスまたは現在のディレクトリ基準の相対パス）")
    content: str = Field(description="ファイルに書き込む全内容（既存ファイルの場合は上書き）")


@tool(args_schema=LocalFileWriteInput)
async def write_local_file(path: str, content: str) -> str:
    """
    ローカルファイルに内容を書き込みます。ファイルがない場合は新しく作成し、ある場合は上書きします。
    ユーザーがファイルの修正、作成、保存を依頼したときに使用します.
    必ずユーザーが明示的に保存/修正/作成を依頼した場合にのみ使用してください。
    """
    try:
        abs_path = os.path.abspath(path)
        print(f"\n[DEBUG] ファイル書き込み: {abs_path}")
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[DEBUG] ファイル書き込み完了")
        return f"ファイル保存完了: {abs_path}"
    except Exception as e:
        return f"ファイル書き込み失敗: {str(e)}"


class ListDirectoryInput(BaseModel):
    path: str = Field(description="探索するフォルダパス（絶対パスまたは相対パス）。現在のフォルダは '.' と入力")


@tool(args_schema=ListDirectoryInput)
async def list_directory(path: str) -> str:
    """
    特定のフォルダ内にどのようなファイルとサブフォルダがあるかのリストを返します。
    ユーザーがどのようなファイルがあるか尋ねたり、ファイルを探す前にフォルダ構造を把握したりするときに使用します。
    """
    try:
        abs_path = os.path.abspath(path)
        print(f"\n[DEBUG] フォルダ探索: {abs_path}")
        if not os.path.isdir(abs_path):
            return f"フォルダが見つかりません: {abs_path}"
        items = os.listdir(abs_path)
        result = []
        for item in sorted(items):
            full = os.path.join(abs_path, item)
            if os.path.isdir(full):
                result.append(f"[フォルダ] {item}/")
            else:
                size = os.path.getsize(full)
                result.append(f"[ファイル] {item} ({size:,} bytes)")
        listing = "\n".join(result)
        return f"[{abs_path}] フォルダ内容:\n{listing}"
    except Exception as e:
        return f"フォルダ探索失敗: {str(e)}"


class ReplaceInFileInput(BaseModel):
    path: str = Field(description="修正するファイルパス（絶対パスまたは相対パス）")
    old_text: str = Field(description="探して置き換える既存のテキスト")
    new_text: str = Field(description="既存のテキストを置き換える新しいテキスト")
    max_replacements: int = Field(description="最大置換回数（デフォルト 1）", default=1)


@tool(args_schema=ReplaceInFileInput)
async def replace_in_file(path: str, old_text: str, new_text: str, max_replacements: int = 1) -> str:
    """
    ファイルの特定のテキストを正確に探し、新しいテキストに置き換えます。
    Diffをターミナルに表示した後、ユーザーの承認(y/n/e)を得て実際に適用します。
    """
    try:
        abs_path = os.path.abspath(path)
        print(f"\n[DEBUG] ファイル置換修正: {abs_path}")

        with open(abs_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        found_count = original_content.count(old_text)
        if found_count == 0:
            return "修正失敗：old_textがファイル内で見つかりませんでした。read_local_fileで最新の内容を再確認してください。"

        replace_count = max(1, int(max_replacements))
        new_content = original_content.replace(old_text, new_text, replace_count)

        # Diff表示およびユーザー承認プロセス
        result = await confirm_and_apply_changes(abs_path, original_content, new_content)
        
        if result.startswith("success:"):
            return f"修正完了およびファイル保存: {abs_path} ({min(found_count, replace_count)}回の置換)"
        elif result == "rejected":
            return "ユーザーが変更事項を拒否しました。再度依頼してください。"
        elif result.startswith("edit:"):
            feedback = result[5:]
            return f"ユーザーフィードバック: {feedback}\nこのフィードバックを基に修正を再試行してください。"
        else:
            return result

    except Exception as e:
        return f"ファイル置換修正失敗: {str(e)}"


class RunValidationInput(BaseModel):
    check: str = Field(
        description="実行する検証の種類。サポート：'pytest', 'ruff', 'mypy', 'python_syntax'",
        default="pytest"
    )


@tool(args_schema=RunValidationInput)
async def run_validation(check: str = "pytest") -> str:
    """
    コード修正後にテスト/リントを実行して結果を返します。
    許可された検証コマンドのみを実行して安全性を維持します。
    """
    commands = {
        "pytest": ["python", "-m", "pytest", "-q"],
        "ruff": ["python", "-m", "ruff", "check", "."],
        "mypy": ["python", "-m", "mypy", "."],
        "python_syntax": ["python", "-m", "compileall", "."],
    }

    selected = check.strip().lower()
    if selected not in commands:
        supported = ", ".join(commands.keys())
        return f"サポートされていない検証の種類です: {check}。サポートリスト: {supported}"

    cmd = commands[selected]
    cwd = os.getcwd()
    print(f"\n[DEBUG] 検証実行: {' '.join(cmd)} (cwd={cwd})")

    def _run():
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    try:
        completed = await asyncio.to_thread(_run)
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        merged = "\n".join([part for part in [stdout, stderr] if part])
        if len(merged) > 8000:
            merged = merged[:8000] + "\n\n[... 出力省略 ...]"

        if completed.returncode == 0:
            return f"検証成功 ({selected})\n{merged}" if merged else f"検証成功 ({selected})"
        return f"検証失敗 ({selected}, exit={completed.returncode})\n{merged}"
    except Exception as e:
        return f"検証実行失敗 ({selected}): {str(e)}"


# =====================================================
# ヘルパー関数: Diff表示およびInteractive承認プロセス
# =====================================================


def display_diff_with_colors(old_content: str, new_content: str, file_path: str) -> None:
    """
    difflib.unified_diffを使用してGitHub PRスタイルDiffを出力します.
    richがあれば色表現、なければテキストで出力します。
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_gen = difflib.unified_diff(
        old_lines, new_lines, fromfile=f"{file_path} (既存)", tofile=f"{file_path} (修正)"
    )
    diff_lines = list(diff_gen)

    if not diff_lines:
        print("[変更事項なし]")
        return

    if RICH_AVAILABLE:
        from rich.syntax import Syntax

        diff_text = "".join(diff_lines)
        console.print("\n[GitHub PRスタイル Diff]\n")
        for line in diff_lines:
            if line.startswith("+++") or line.startswith("---"):
                console.print(line.rstrip(), style="bold blue")
            elif line.startswith("@@"):
                console.print(line.rstrip(), style="cyan")
            elif line.startswith("+"):
                console.print(line.rstrip(), style="green")
            elif line.startswith("-"):
                console.print(line.rstrip(), style="red")
            else:
                console.print(line.rstrip())
    else:
        print("\n[Diff変更事項]\n")
        print("".join(diff_lines))


async def get_user_approval() -> str:
    """
    ターミナルからユーザー入力を受け取り、承認ステータスを返します。
    - 'y' / 'yes' / 'ok' → 'accept'
    - 'n' / 'no' → 'reject'
    - 'e' / 'edit' + フィードバック → 'edit:<フィードバック>'
    """
    print("\n[承認が必要]\nオプションを選択してください:")
    print("  [y] Accept  - 変更事項をファイルに適用")
    print("  [n] Reject  - 変更事項をキャンセル")
    print("  [e] Edit    - フィードバック入力後に再修正を依頼\n")

    def _input_sync():
        user_input = input("選択 [y/n/e]: ").strip().lower()
        return user_input

    user_choice = await asyncio.to_thread(_input_sync)

    if user_choice in ["y", "yes", "ok"]:
        return "accept"
    elif user_choice in ["n", "no"]:
        return "reject"
    elif user_choice in ["e", "edit"]:
        def _feedback_input():
            feedback = input("フィードバック入力: ").strip()
            return feedback

        feedback = await asyncio.to_thread(_feedback_input)
        return f"edit:{feedback}" if feedback else "reject"
    else:
        print("[警告] 無効な入力です。拒否として処理します。")
        return "reject"


async def confirm_and_apply_changes(
    file_path: str, old_content: str, new_content: str
) -> str:
    """
    ファイル変更事項をDiffで視覚化し、ユーザー承認後に実際に適用します。
    返り値:
    - "success:<ファイルパス>" → 変更適用完了
    - "rejected" → ユーザーが拒否
    - "edit:<フィードバック>" → ユーザーがフィードバック提示（エージェントが再修正する必要あり）
    """
    abs_path = os.path.abspath(file_path)
    print(f"\n{'=' * 60}")
    print(f"ファイル: {abs_path}")
    print("=" * 60)

    display_diff_with_colors(old_content, new_content, abs_path)

    approval = await get_user_approval()

    if approval == "accept":
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"\n✓ 変更事項がファイルに適用されました: {abs_path}\n")
            return f"success:{abs_path}"
        except Exception as e:
            return f"error:ファイル書き込み失敗 - {str(e)}"
    elif approval == "rejected":
        print("\n✗ 変更事項がキャンセルされました。\n")
        return "rejected"
    elif approval.startswith("edit:"):
        feedback = approval[5:]
        print(f"\n✎ フィードバックが記録されました: {feedback}\n")
        return f"edit:{feedback}"
    else:
        return "rejected"
