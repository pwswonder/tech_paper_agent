from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable

llm = AzureChatOpenAI(
    # azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
    # openai_api_version="2024-02-01",
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT41"),
    openai_api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    temperature=0.0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a domain classification agent for technical papers. "
            "Given the content of a research paper, classify it into one domain only. "
            "Your answer must strictly follow the format: '도메인: 인공지능-XXXX'. "
            "Use the most appropriate subdomain from the examples below if possible.\n\n"
            "예시 도메인:\n"
            "- 인공지능-이미지\n"
            "- 인공지능-자율주행\n"
            "- 인공지능-시계열\n"
            "- 인공지능-강화학습\n"
            "- 인공지능-음성\n"
            "- 인공지능-예측\n"
            "- 인공지능-NLP\n"
            "- 인공지능-로보틱스\n"
            "- 인공지능-시맨틱\n",
        ),
        (
            "user",
            "문서: 이 논문은 딥러닝 기반의 CNN 모델을 이용하여 이미지 분류 정확도를 향상시켰다.",
        ),
        ("assistant", "도메인: 인공지능-이미지"),
        (
            "user",
            "문서: 이 논문은 자율주행 차량의 경로 계획을 위한 강화학습 알고리즘을 제안하였다.",
        ),
        ("assistant", "도메인: 인공지능-자율주행"),
        ("user", "{user_input}"),
    ]
)

chain = prompt | llm | StrOutputParser()

classifier_agent = RunnableLambda(
    lambda state: {
        "domain": chain.invoke({"user_input": f"문서: {state['raw_text'][:3000]}"})
    }
)
