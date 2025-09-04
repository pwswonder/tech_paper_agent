# services/summarizer.py
# ------------------------------------------------------------
# 목적: 기술 논문 요약/QA 에이전트 품질 향상
# 핵심: 요약용 프롬프트와 QA용 프롬프트를 분리 + 규칙 강화
# ------------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda
from typing import List
import os
from dotenv import load_dotenv
from langsmith import traceable

# 1) 환경변수 로드 (.env 경로가 루트가 아닐 수 있으니 필요 시 지정)
load_dotenv()

# 2) LLM 인스턴스 분리
# - 요약은 사실성/일관성 중요 → temperature 낮게, 토큰 넉넉히
summary_llm = AzureChatOpenAI(
    # azure_deployment=os.getenv("AOAI_DEPLOY_GPT40"),
    # openai_api_version="2024-02-01",
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT41"),
    openai_api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    temperature=0.1,
    # max_tokens=900,  # 요약 분량 제어
)

# - QA는 응답 다양성 약간 허용
qa_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT41"),
    openai_api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    temperature=0.3,
    # max_tokens=700,  # QA 답변 길이 제어
)

# 3) 요약 전용 System 프롬프트 (규칙 강력)
SUMMARY_SYSTEM = """\
당신은 기술 논문 요약 전문가입니다. 다음 규칙을 엄격히 따르세요.

[목표]
- 제공된 텍스트(논문 청크)만을 근거로, **사실에 근거한** 한국어 요약을 작성합니다.
- 아래 섹션 헤더를 정확히 사용하고, 각 섹션은 1~4문장으로 간결히 작성합니다.

[섹션]
1) 연구주제
2) 핵심 기여 (불릿 2~3개)
3) 방법(모델/알고리즘/구조)
4) 데이터셋/설정 (데이터, 전처리, 주요 하이퍼파라미터)
5) 실험결과 (정량 차이/지표/비교 대상 포함)

[사실성/인용]
- 숫자/모델명/데이터셋명은 **원문 표기 그대로** 사용합니다.
- 정보가 불분명하면 임의로 추정하지 말고 "원문에 명시 없음"이라고 적습니다.

[스타일/길이]
- 한국어로 작성합니다.
- 각 섹션 제목은 반드시 Markdown 헤더(###)로 표기하고, 본문은 일반 문단/불릿으로 작성합니다.
- 전체 분량은 300~500단어 내로 유지합니다.
- 불필요한 수식/장식 금지. 핵심만 간결하게.
"""

# 4) 요약 User 프롬프트
SUMMARY_USER = """\
[문서 메타]
- 제목: {title}
- 출처: {source}

[입력 청크들]
{chunks}

[요청]
- 위 규칙대로 섹션 요약을 생성하세요.
- 중복 내용은 제거하고, 동일 개념의 다양한 표기는 일관되게 통일하세요.
"""

# 5) QA 전용 System 프롬프트 (근거/불확실성 가드)
QA_SYSTEM = """\
당신은 기술 논문 질의응답 전문가입니다.
- 제공된 컨텍스트(논문 청크) **내에서만** 사실을 추출하여 답합니다.
- 외부 지식 추정/환각 금지. 컨텍스트에 없으면 "원문에 명시 없음"이라고 답하세요.
- 답변 형식:
  1) 핵심 답변(5~10문장, 한국어)
  2) 근거 인용 블록: 각 문장 앞에 `> `를 붙여 2~4줄 인용
"""

# 6) QA User 프롬프트
QA_USER = """\
[관련 컨텍스트]
{context}

[사용자 질문]
{question}

[요청]
- 위 형식대로 답변하세요.
- 수치/모델/데이터셋은 원문 표기 유지.
"""

# 7) 체인 구성 (요약/QA 별도)
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM),
        ("user", SUMMARY_USER),
    ]
)
summary_chain = summary_prompt | summary_llm | StrOutputParser()

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM),
        ("user", QA_USER),
    ]
)
qa_chain = qa_prompt | qa_llm | StrOutputParser()


# 8) Utils: 컨텍스트 합치기 + 중복 제거 (간단 버전)
def _dedup_lines(text: str) -> str:
    """아주 단순한 중복 줄 제거(완벽하진 않지만 길이 감소에 유용)."""
    seen = set()
    out: List[str] = []
    for line in text.splitlines():
        key = line.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(line)
    return "\n".join(out)


@traceable  # ★ 이 1줄만 추가
# 9) Runnable: 요약 에이전트
def _run_summarize(state):
    """
    state 요구:
    - raw_texts: List[str] 또는 raw_text: str (둘 중 하나)
    - meta: dict(title, source) (선택)
    """
    # (1) 입력 수집
    title = (state.get("meta") or {}).get("title", "제목 미상")
    source = (state.get("meta") or {}).get("source", "출처 미상")

    # raw_texts 우선, 없으면 raw_text 사용
    raw_texts = state.get("raw_texts")
    if raw_texts is None:
        rt = state.get("raw_text", "")
        raw_texts = [rt] if rt else []

    if not raw_texts:
        return {
            "summary": "💡 요약할 텍스트가 없습니다. raw_text 또는 raw_texts를 확인하세요."
        }

    # (2) 청크 합치기 (너무 길면 상위 N개만 사용, 여기선 단순 결합)
    chunks = "\n\n---\n\n".join(raw_texts)
    chunks = _dedup_lines(chunks)

    # (3) 모델 호출
    summary = summary_chain.invoke(
        {
            "title": title,
            "source": source,
            "chunks": chunks,
        }
    )
    return {"summary": summary}


summarizer_agent = RunnableLambda(_run_summarize)


@traceable  # ★ 이 1줄만 추가
# 10) Runnable: QA + Retrieval
def qa_with_retrieval(state):
    """
    state 요구:
    - user_input: str (질문)
    - retriever: BaseRetriever-like
    - meta: dict(optional) (문서명/저자/페이지 등 넣으면 UX 좋음)
    - top_k: int(optional)
    """
    question = state.get("user_input", "")
    if not question:
        return {"answer": "💡 질문이 비어 있습니다. user_input을 확인하세요."}

    retriever = state.get("retriever")
    if retriever is None:
        return {
            "answer": "💡 문서를 임베딩하거나 검색할 수 없습니다. retriever가 없습니다."
        }

    top_k = state.get("top_k", 4)
    try:
        docs = retriever.invoke(question)  # 필요 시 retriever.search_kwargs 조정
        docs = docs[:top_k] if len(docs) > top_k else docs
    except Exception as e:
        return {"answer": f"검색 중 오류가 발생했습니다: {e}"}

    if not docs:
        return {
            "answer": "💡 관련 문서를 찾지 못했습니다. 질문을 구체화하거나 문서 업로드를 확인하세요."
        }

    # 컨텍스트 생성: page_content + (가능하면) 메타정보도 덧붙이기
    # LangChain 문서 객체는 .metadata에 source/page 등 있을 수 있음
    context_blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source") if hasattr(d, "metadata") else None
        page = d.metadata.get("page") if hasattr(d, "metadata") else None
        header = f"[Doc#{i} | source={src or 'N/A'} | page={page or 'N/A'}]"
        context_blocks.append(f"{header}\n{d.page_content}")

    context = "\n\n------\n\n".join(context_blocks)
    context = _dedup_lines(context)

    answer = qa_chain.invoke({"context": context, "question": question})
    return {"answer": answer}


qa_agent = RunnableLambda(qa_with_retrieval)
