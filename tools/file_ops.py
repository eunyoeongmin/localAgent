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
    ローカルファイル(txt, md, py, csvなど)をチャン크単位で分割し、
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
            f"- 총 チャンク 수: {len(chunks)}\n\n"
            f"{selected_text}"
        )
    except FileNotFoundError:
        return f"파일을 찾을 수 없습니다: {os.path.abspath(path)}"
    except ImportError as e:
        return f"파일 검색용 의존성을 로드하지 못했습니다: {str(e)}"
    except Exception as e:
        return f"파일 읽기 실패: {str(e)}"


class LocalFileWriteInput(BaseModel):
    path: str = Field(description="書き込むファイルのパス（絶対パスまたは現在のディレクトリ基準の相対パス）")
    content: str = Field(description="파일에 작성할 전체 내용 (기존 파일이 있으면 덮어쓰기)")


@tool(args_schema=LocalFileWriteInput)
async def write_local_file(path: str, content: str) -> str:
    """
    로컬 파일에 내용을 작성합니다. 파일이 없으면 새로 생성하고, 있으면 덮어씁니다.
    사용자가 파일의 수정, 생성, 저장을 요청했을 때 사용합니다.
    반드시 사용자가 명시적으로 저장/수정/생성을 요청한 경우에만 사용하십시오.
    """
    try:
        abs_path = os.path.abspath(path)
        print(f"\n[DEBUG] 파일 쓰기: {abs_path}")
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("[DEBUG] 파일 쓰기 완료")
        return f"파일 저장 완료: {abs_path}"
    except Exception as e:
        return f"파일 쓰기 실패: {str(e)}"


class ListDirectoryInput(BaseModel):
    path: str = Field(description="탐색할 폴더 경로 (절대 경로 또는 상대 경로). 현재 폴더는 '.' 입력")


@tool(args_schema=ListDirectoryInput)
async def list_directory(path: str) -> str:
    """
    특정 폴더 안에 어떤 파일과 하위 폴더가 있는지 목록을 반환합니다.
    사용자가 어떤 파일이 있는지 묻거나, 파일을 찾기 전에 폴더 구조를 파악할 때 사용합니다.
    """
    try:
        abs_path = os.path.abspath(path)
        print(f"\n[DEBUG] 폴더 탐색: {abs_path}")
        if not os.path.isdir(abs_path):
            return f"폴더를 찾을 수 없습니다: {abs_path}"
        items = os.listdir(abs_path)
        result = []
        for item in sorted(items):
            full = os.path.join(abs_path, item)
            if os.path.isdir(full):
                result.append(f"[폴더] {item}/")
            else:
                size = os.path.getsize(full)
                result.append(f"[파일] {item} ({size:,} bytes)")
        listing = "\n".join(result)
        return f"[{abs_path}] 폴더 내용:\n{listing}"
    except Exception as e:
        return f"폴더 탐색 실패: {str(e)}"


class ReplaceInFileInput(BaseModel):
    path: str = Field(description="수정할 파일의 경로 (절대 경로 또는 상대 경로)")
    old_text: str = Field(description="찾아서 교체할 기존 텍스트")
    new_text: str = Field(description="기존 텍스트를 대체할 새로운 텍스트")
    max_replacements: int = Field(description="최대 교체 횟수 (기본 1)", default=1)


@tool(args_schema=ReplaceInFileInput)
async def replace_in_file(path: str, old_text: str, new_text: str, max_replacements: int = 1) -> str:
    """
    파일의 특정 텍스트를 정확히 찾아 새로운 텍스트로 바꿉니다.
    Diff를 터미널에 표시한 후, 사용자의 승인(y/n/e)을 얻어 실제로 적용합니다.
    """
    try:
        abs_path = os.path.abspath(path)
        print(f"\n[DEBUG] 파일 치환 수정: {abs_path}")

        with open(abs_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        found_count = original_content.count(old_text)
        if found_count == 0:
            return "수정 실패：old_text를 파일 내에서 찾을 수 없습니다. read_local_file로 최신 내용을 다시 확인하세요."

        replace_count = max(1, int(max_replacements))
        new_content = original_content.replace(old_text, new_text, replace_count)

        # Diff 표시 및 사용자 승인 프로세스
        result = await confirm_and_apply_changes(abs_path, original_content, new_content)
        
        if result.startswith("success:"):
            return f"수정 완료 및 파일 저장: {abs_path} ({min(found_count, replace_count)}회의 치환)"
        elif result == "rejected":
            return "사용자가 변경 사항을 거부했습니다. 다시 요청해 주세요."
        elif result.startswith("edit:"):
            feedback = result[5:]
            return f"사용자 피드백: {feedback}\n이 피드백을 기반으로 수정을 다시 시도하세요."
        else:
            return result

    except Exception as e:
        return f"파일 치환 수정 실패: {str(e)}"


class RunValidationInput(BaseModel):
    check: str = Field(
        description="실행할 검증 종류. 지원：'pytest', 'ruff', 'mypy', 'python_syntax'",
        default="pytest"
    )


@tool(args_schema=RunValidationInput)
async def run_validation(check: str = "pytest") -> str:
    """
    코드 수정 후 테스트/린트를 실행하여 결과를 반환합니다.
    허용된 검증 커맨드만 실행하여 안전성을 유지합니다.
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
        return f"지원되지 않는 검증 종류입니다: {check}. 지원 목록: {supported}"

    cmd = commands[selected]
    cwd = os.getcwd()
    print(f"\n[DEBUG] 검증 실행: {' '.join(cmd)} (cwd={cwd})")

    def _run():
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    try:
        completed = await asyncio.to_thread(_run)
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        merged = "\n".join([part for part in [stdout, stderr] if part])
        if len(merged) > 8000:
            merged = merged[:8000] + "\n\n[... 출력 생략 ...]"

        if completed.returncode == 0:
            return f"검증 성공 ({selected})\n{merged}" if merged else f"검증 성공 ({selected})"
        return f"검증 실패 ({selected}, exit={completed.returncode})\n{merged}"
    except Exception as e:
        return f"검증 실행 실패 ({selected}): {str(e)}"


# =====================================================
# 헬퍼 함수: Diff 표시 및 Interactive 승인 프로세스
# =====================================================


def display_diff_with_colors(old_content: str, new_content: str, file_path: str) -> None:
    """
    difflib.unified_diff를 사용하여 GitHub PR 스타일 Diff를 출력합니다.
    rich가 있으면 색상 표현, 없으면 텍스트로 출력합니다.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_gen = difflib.unified_diff(
        old_lines, new_lines, fromfile=f"{file_path} (기존)", tofile=f"{file_path} (수정)"
    )
    diff_lines = list(diff_gen)

    if not diff_lines:
        print("[변경 사항 없음]")
        return

    if RICH_AVAILABLE:
        console.print("\n[GitHub PR 스타일 Diff]\n")
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
        print("\n[Diff 변경 사항]\n")
        print("".join(diff_lines))


async def get_user_approval() -> str:
    """
    터미널에서 사용자 입력을 받아 승인 상태를 반환합니다.
    - 'y' / 'yes' / 'ok' → 'accept'
    - 'n' / 'no' → 'reject'
    - 'e' / 'edit' + 피드백 → 'edit:<피드백>'
    """
    print("\n[승인 필요]\n옵션을 선택하세요:")
    print("  [y] Accept  - 변경 사항을 파일에 적용")
    print("  [n] Reject  - 변경 사항을 취소")
    print("  [e] Edit    - 피드백 입력 후 재수정 요청\n")

    def _input_sync():
        user_input = input("선택 [y/n/e]: ").strip().lower()
        return user_input

    user_choice = await asyncio.to_thread(_input_sync)

    if user_choice in ["y", "yes", "ok"]:
        return "accept"
    elif user_choice in ["n", "no"]:
        return "reject"
    elif user_choice in ["e", "edit"]:
        def _feedback_input():
            feedback = input("피드백 입력: ").strip()
            return feedback

        feedback = await asyncio.to_thread(_feedback_input)
        return f"edit:{feedback}" if feedback else "reject"
    else:
        print("[경고] 유효하지 않은 입력입니다. 거부로 처리합니다.")
        return "reject"


async def confirm_and_apply_changes(
    file_path: str, old_content: str, new_content: str
) -> str:
    """
    파일의 변경 사항을 Diff로 시각화하고, 사용자의 승인 후 실제로 적용합니다.
    반환값:
    - "success:<파일 경로>" → 변경 적용 완료
    - "rejected" → 사용자가 거부
    - "edit:<피드백>" → 사용자가 피드백 제시 (에이전트가 재수정 필요)
    """

    abs_path = os.path.abspath(file_path)
    print(f"\n{'=' * 60}")
    print(f"파일: {abs_path}")
    print("=" * 60)

    display_diff_with_colors(old_content, new_content, abs_path)

    approval = await get_user_approval()

    if approval == "accept":
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"\n✓ 변경 사항이 파일에 적용되었습니다: {abs_path}\n")
            return f"success:{abs_path}"
        except Exception as e:
            return f"error:파일 쓰기 실패 - {str(e)}"
    elif approval == "rejected":
        print("\n✗ 변경 사항이 취소되었습니다.\n")
        return "rejected"
    elif approval.startswith("edit:"):
        feedback = approval[5:]
        print(f"\n✎ 피드백이 기록되었습니다: {feedback}\n")
        return f"edit:{feedback}"
    else:
        return "rejected"
