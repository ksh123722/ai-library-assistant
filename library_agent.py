
"""
AI 사서 보조 에이전트
project_4/05_full_agent.py 기반
"""

import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    after_agent,
    before_agent,
    wrap_model_call,
)
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# 1. 모델 설정 

MODEL = "qwen3:4b-instruct-2507-q4_K_M"

main_model = ChatOllama(
    model=MODEL,
    temperature=0.0,
    num_ctx=2048,
)

judge_model = ChatOllama(
    model=MODEL,
    temperature=0.0,
    num_ctx=2048,
)


# 2. 도서 검색 및 대출 데이터

books = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

loans = {
    "user_001": [
        {
            "title": "시간을 파는 서점",
            "due_date": "2026-11-20",
        }
    ]
}

# 실제 발송하지 않고 메시지만 저장
replies = []


# 3. Runtime Context - 교수님 원본의 dataclass 사용

@dataclass
class Context:
    librarian_name: str
    librarian_role: str
    user_id: str
    app_name: str


# 교수님 원본의 장기 기억 저장 형식
class UserInfo(TypedDict):
    personal_information: str
    preference: str


store = InMemoryStore()


# 4. 도서관 업무 도구

@tool
def search_books(query: str) -> str:
    """도서 목록에서 제목 또는 저자로 검색합니다."""
    q = query.strip().replace(" ", "").lower()

    if not q:
        return "검색어를 입력해 주세요."

    matches = [
        book
        for book in books
        if q in book["title"].replace(" ", "").lower()
        or q in book["author"].replace(" ", "").lower()
    ]

    if not matches:
        return "해당 도서를 찾을 수 없습니다."

    return "\n".join(
        f"제목: {book['title']} / 저자: {book['author']}"
        for book in matches
    )


@tool
def check_loan_info(runtime: ToolRuntime) -> str:
    """현재 이용자 본인의 대출 정보를 확인합니다."""
    current = loans.get(runtime.context.user_id, [])

    if not current:
        return "현재 확인되는 대출 내역이 없습니다."

    return "\n".join(
        f"{item['title']} / 반납 예정일: {item['due_date']}"
        for item in current
    )


@tool
def send_user_reply(message: str, runtime: ToolRuntime) -> str:
    """현재 이용자의 안내 메시지를 모의 회신함에 기록합니다."""
    replies.append({
        "user_id": runtime.context.user_id,
        "message": message,
    })

    return (
        "안내 메시지를 실습용 회신함에 기록했습니다. "
        "실제로 발송하지는 않았습니다."
    )


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """현재 이용자의 저장된 독서 선호를 조회합니다."""
    memories = runtime.store.search(
        (runtime.context.user_id, runtime.context.app_name)
    )

    if not memories:
        return "저장된 독서 선호가 없습니다."

    lines = []

    for item in memories:
        data = item.value

        if "preference" in data:
            lines.append(f"- 독서 선호: {data['preference']}")

    return "\n".join(lines) if lines else "저장된 독서 선호가 없습니다."


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime) -> str:
    """현재 이용자의 독서 선호를 장기 기억에 저장합니다."""
    key = str(uuid.uuid4())

    runtime.store.put(
        (runtime.context.user_id, runtime.context.app_name),
        key,
        user_info,
    )

    return "독서 선호 정보를 장기 기억에 저장했습니다."


# 5. Context Engineering - 교수님 원본의 wrap_model_call

@wrap_model_call
def enrich_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:

    ctx = request.runtime.context

    librarian_name = getattr(ctx, "librarian_name", "") or ""

    ns = (ctx.user_id, ctx.app_name)

    memories = (
        request.runtime.store.search(ns)
        if request.runtime.store
        else []
    )

    preferences = []

    for item in memories:
        preference = item.value.get("preference", "")

        if preference:
            preferences.append(preference)

    memory_text = (
        "\n- ".join(preferences)
        if preferences
        else "기록된 정보 없음"
    )

    base = request.system_prompt or ""

    prompt = (
        f"{base}\n"
        f"당신은 AI 사서 보조 {librarian_name}입니다.\n"
        f"역할은 {ctx.librarian_role}입니다.\n"
        f"[이용자의 장기 기억]\n- {memory_text}\n"
        "도서관 업무에 관한 질문만 처리하세요.\n"
        "도서 존재 여부는 반드시 search_books 도구로 확인하세요.\n"
        "본인 대출 정보는 check_loan_info 도구로 확인하세요.\n"
        "다른 이용자의 대출 정보는 공개하지 마세요.\n"
        "확인되지 않은 도서 정보나 운영시간은 추측하지 마세요.\n"
        "독서 선호를 저장할 때는 save_user_info를 사용하세요.\n"
        "독서 선호를 조회할 때는 get_user_info를 사용하세요.\n"
        "회신 요청 시 send_user_reply를 사용하세요.\n"
        "회신은 실제 발송이 아니라 모의 회신함에 기록하는 것입니다.\n"
        "회신을 기록했다면 실제 발송하지 않았다고 안내하세요.\n"
        "한국어로 짧게 답하세요."
    ).strip()

    return handler(request.override(system_prompt=prompt))


# 6. 입력 가드레일 - 교수님 원본의 before_agent

forbidden_topics = {
    "privacy": [
        "다른 사람 대출",
        "다른 이용자 대출",
        "타인 대출",
        "친구 대출",
        "남의 대출",
    ],
    "off_topic": [
        "주식 추천",
        "로또 번호",
        "게임 공략",
        "코딩 과제",
        "오늘 날씨",
    ],
}

BLOCK_MESSAGES = {
    "privacy": "다른 이용자의 대출 정보는 제공할 수 없습니다.",
    "off_topic": "도서 검색과 도서관 이용에 관한 질문만 도와드릴 수 있습니다.",
}


@before_agent(can_jump_to=["end"])
def library_policy_guardrail(state, runtime):

    if not state["messages"]:
        return None

    last = state["messages"][-1]

    if getattr(last, "type", None) != "human":
        return None

    text = str(last.content or "").replace(" ", "")

    for category, keywords in forbidden_topics.items():

        if any(k.replace(" ", "") in text for k in keywords):
            print(f"[가드레일] {category}")

            return {
                "messages": [
                    AIMessage(content=BLOCK_MESSAGES[category])
                ],
                "jump_to": "end",
            }

    return None


# 7. 출력 검수 - 교수님 원본의 after_agent

@after_agent
def library_output_guardrail(state, runtime):

    if not state["messages"]:
        return None

    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return None

    if not str(last_message.content or "").strip():
        return None

    auditor_prompt = f"""당신은 엄격한 도서관 상담 감독관입니다.

다음 답변에 문제가 있으면 'LEAKED',
문제가 없으면 'SAFE'라고만 답하세요.

문제가 있는 답변:
- 다른 이용자의 대출 정보를 공개함
- 확인하지 않은 도서 정보를 사실이라고 단정함
- 확인되지 않은 도서관 운영시간을 단정함
- 모의 회신을 실제 발송했다고 주장함

정상적인 답변:
- 도구로 확인한 도서 정보를 안내함
- 본인의 대출 정보를 안내함
- 개인정보 공개를 거절함
- 확인되지 않은 정보는 모른다고 안내함
- 모의 회신함에 기록했다고 안내함

상담원의 답변: {last_message.content}
"""

    verdict = judge_model.invoke([
        {"role": "user", "content": auditor_prompt}
    ]).content

    verdict = verdict if isinstance(verdict, str) else str(verdict)

    if "LEAKED" in verdict.upper():
        print("[Judge] LEAKED - 답변 교체")

        # 교수님 원본과 동일하게 content 직접 수정
        last_message.content = (
            "해당 내용은 확인된 정보만 안내할 수 있습니다. "
            "도서 검색 또는 본인 대출 조회로 다시 확인해 주세요."
        )

    return None


# 8. 에이전트 생성 - 교수님 원본 구조 유지

agent = create_agent(
    main_model,
    tools=[
        search_books,
        check_loan_info,
        send_user_reply,
        get_user_info,
        save_user_info,
    ],
    checkpointer=InMemorySaver(),
    store=store,
    context_schema=Context,
    middleware=[
        library_policy_guardrail,
        enrich_context,
        library_output_guardrail,
    ],
)


# 9. Runtime Context

ctx = Context(
    librarian_name="하늘",
    librarian_role="AI 사서 보조",
    user_id="user_001",
    app_name="library_assistant",
)


# 10. 실행 함수

def ask(question: str, thread_id: str = "library_demo"):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
        context=ctx,
    )

    messages = result["messages"]

    last_user_index = max(
        i
        for i, message in enumerate(messages)
        if getattr(message, "type", None) == "human"
    )

    new_messages = messages[last_user_index + 1:]

    calls = [
        call["name"]
        for message in new_messages
        for call in (getattr(message, "tool_calls", None) or [])
    ]

    print("도구 호출:", calls or "없음")
    print("답변:", messages[-1].content)


# 11. 터미널 실행

if __name__ == "__main__":

    print(
        f"AI 사서 보조 에이전트 | "
        f"모델: {MODEL} | "
        f"도서: {len(books)}권"
    )

    print("종료하려면 exit 입력")

    while True:

        question = input("\n이용자> ").strip()

        if question.lower() in {"exit", "quit", "종료"}:
            break

        if question:
            try:
                ask(question)

            except Exception as e:
                print("실행 오류:", e)
