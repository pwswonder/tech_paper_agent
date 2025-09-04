from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import os

from backend.database import get_db
from backend import models, schemas, crud

from services.summarizer import qa_agent
from services.retriever_cache import get_retriever, set_retriever
from services.file_reader import file_reader
from services.graph_builder import build_graph

router = APIRouter()

# 그래프는 서버 기동 시 1회 컴파일 → 복구(캐시 미존재) 시만 사용
_GRAPH = build_graph()


@router.get("/qa/{document_id}", response_model=List[schemas.QAHistoryOut])
def get_qa_history(document_id: int, db: Session = Depends(get_db)):
    """
    app.py가 기대하는 형식으로 QA 히스토리를 반환합니다.
    - 키: question, answer, created_at
    """
    rows = (
        db.query(models.QAHistory)
        .filter(models.QAHistory.document_id == document_id)
        .order_by(models.QAHistory.created_at.asc())
        .all()
    )
    # pydantic 모델 매핑을 신뢰해도 되고, 안전하게 dict로 변환해도 됩니다.
    return [
        schemas.QAHistoryOut(
            question=row.question, answer=row.answer, created_at=row.created_at
        )
        for row in rows
    ]


@router.post("/qa/ask_existing")
def ask_existing_document_question(
    payload: schemas.ExistingDocQARequest, db: Session = Depends(get_db)
):
    """
    기존 문서에 대해 질문을 수행합니다.
    흐름:
      1) 문서 존재 확인
      2) retriever 캐시 조회
      3) (캐시 미스) file_reader + _GRAPH.invoke 로 retriever 복구 → 캐시에 저장
      4) qa_agent.invoke 로 답변 생성
      5) QA 히스토리 저장
    반환: {"answer": "..."}
    """
    doc_id = payload.document_id
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")

    # 1) 문서 조회
    document = crud.get_document_by_id(db, doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    # 2) retriever 캐시 확인
    retriever = get_retriever(doc_id)

    # 3) 캐시 미스 → 복구
    # 3) 캐시 미스 → 복구
    if retriever is None:
        file_path = document.file_path
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="원본 파일을 찾을 수 없습니다. 재업로드가 필요합니다.",
            )

        # file_reader로 raw_text/meta 준비 → (변경) embedder 직접 호출 → retriever 회수
        fr_state = file_reader({"file": file_path})

        ## [RETRIEVER-REBUILD] avoid full graph; build retriever directly via embedder
        result = None
        try:
            from services import embedder as _embedder

            # embedder가 캐시에 저장하도록 document_id를 명시
            emb_in = {**fr_state, "document_id": doc_id}
            result = _embedder.embedder(emb_in)
        except Exception as _e:
            # Fallback: 기존 그래프 경로 유지
            try:
                result = _GRAPH.invoke(fr_state)
            except Exception as _e2:
                raise HTTPException(
                    status_code=500, detail=f"retriever 복구 실패: {str(_e2)[:200]}"
                )

        retriever = result.get("retriever")
        if retriever is None:
            raise HTTPException(status_code=500, detail="retriever 복구 실패")

        # 메모리 캐시에 등록
        set_retriever(doc_id, retriever, result.get("vectorstore"))

    # 4) QA 수행 (빠르게: qa_agent만 호출)
    qa_out = qa_agent.invoke(
        {
            "user_input": question,
            "retriever": retriever,
            "top_k": 5,
        }
    )
    answer = qa_out.get("answer", "")
    if not answer:
        raise HTTPException(status_code=500, detail="답변 생성 실패")

    # 5) QA 히스토리 저장 (app.py 기대 필드: question/answer/created_at)
    try:
        rec_in = schemas.QACreate(
            document_id=doc_id,
            question=question,
            answer=answer,
        )
        crud.save_qa_history(db, qa=rec_in)
    except Exception as e:
        # 답변은 반환하되, 저장 실패는 로그로 충분
        # 실제 운영에서는 로깅 시스템에 남기세요.
        pass

    return {"answer": answer}
