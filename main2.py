import streamlit as st                     # 웹 인터페이스 제작을 위한 Streamlit
import os                                   # 운영체제 관련 기능 사용
import google.generativeai as genai        # Google Gemini AI API를 통한 텍스트 생성 기능
from pdf_utils import extract_text_from_pdf # PDF 문서에서 텍스트 추출 기능
import asyncio                              # 비동기 처리를 위한 asyncio 라이브러리
from concurrent.futures import ThreadPoolExecutor  # 병렬 처리를 위한 ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer  # 텍스트 데이터를 벡터화하기 위한 TF-IDF 도구
from sklearn.metrics.pairwise import cosine_similarity       # 코사인 유사도를 계산하기 위한 함수

# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="덤핑 전문가 챗봇",
    page_icon="📚",
    layout="wide"
)

# --- 유저로부터 Gemini API Key 입력 받기 ---
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

with st.sidebar.expander("🔑 API Key 설정", expanded=True):
    key_input = st.text_input(
        label="Google Gemini API Key 입력",
        type="password",
        placeholder="여기에 API Key를 입력하세요",
        value=st.session_state.gemini_api_key,
    )
    if key_input:
        st.session_state.gemini_api_key = key_input

if not st.session_state.gemini_api_key:
    st.sidebar.warning("챗봇을 이용하려면 API Key를 입력해주세요.")
    st.stop()

# --- Gemini API 설정 ---
genai.configure(api_key=st.session_state.gemini_api_key)

# --- 세션 상태 초기화 ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'law_data' not in st.session_state:
    st.session_state.law_data = {}
# 임베딩 캐싱용 상태
if 'embedding_data' not in st.session_state:
    st.session_state.embedding_data = {}
# 재사용 가능한 asyncio 이벤트 루프
if 'event_loop' not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.event_loop = loop

# --- 카테고리 정의 ---
LAW_CATEGORIES = {
    "덤핑방지관세": {
        "중국산 더블레이어 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙": "docs/중국산 더블레이어 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙(기획재정부령)(제00940호)(20221025).pdf",
        "중국산 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙": "docs/중국산 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙(기획재정부령)(제00882호)(20220101) (1).pdf",
    },
    "덤핑판정": {
        "중국산 더블레이어 인쇄제판용 평면모양 사진플레이트 최종판정": "docs/중국산 더블레이어 인쇄제판용 평면모양 사진플레이트_최종판정의결서.pdf",
        "중국산 인쇄제판용 평면모양 사진플레이트 최종판정": "docs/중국산 인쇄제판용 평면모양 사진플레이트_최종판정의결서.pdf",
    },
    "관련법령": {
        "관세법": "docs/관세법(법률)(제20608호)(20250401).pdf",
        "관세법 시행령": "docs/관세법 시행령(대통령령)(제35363호)(20250722).pdf",
        "관세법 시행규칙": "docs/관세법 시행규칙(기획재정부령)(제01110호)(20250321).pdf",
        "불공정무역행위 조사 및 산업피해구제에 관한 법률": "docs/불공정무역행위 조사 및 산업피해구제에 관한 법률(법률)(제20693호)(20250722).pdf",
    }
}

# 카테고리별 키워드 정보
CATEGORY_KEYWORDS = {
    "덤핑방지관세": ["덤핑방지관세", "덤핑마진", "정상가격", "수출가격", "덤핑률", "반덤핑관세", "덤핑방지", "덤핑방지조치"],
    "덤핑판정": ["최종판정", "예비판정", "산업피해", "실질적 피해", "인과관계", "국내산업", "조사대상물품", "덤핑수입"],
    "관련법령": ["관세법", "시행령", "시행규칙", "불공정무역", "산업피해구제", "무역위원회", "조사절차", "덤핑규정"]
}

# PDF 로드 및 임베딩 생성 함수
@st.cache_data
def load_law_data(category=None):
    law_data = {}
    missing_files = []
    # 모든 카테고리의 파일을 한번에 로드
    pdf_files = {}
    for cat_files in LAW_CATEGORIES.values():
        pdf_files.update(cat_files)
    
    for law_name, pdf_path in pdf_files.items():
        if os.path.exists(pdf_path):
            text = extract_text_from_pdf(pdf_path)
            law_data[law_name] = text
            # 임베딩 생성 및 캐싱
            vec, mat, chunks = create_embeddings_for_text(text)
            st.session_state.embedding_data[law_name] = (vec, mat, chunks)
        else:
            missing_files.append(pdf_path)
    if missing_files:
        st.warning(f"다음 파일들을 찾을 수 없습니다: {', '.join(missing_files)}")
    return law_data

# 임베딩 및 청크 생성
@st.cache_data
def create_embeddings_for_text(text, chunk_size=1000):
    chunks = []
    step = chunk_size // 2
    for i in range(0, len(text), step):
        segment = text[i:i+chunk_size]
        if len(segment) > 100:
            chunks.append(segment)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix, chunks

# 쿼리 유사 청크 검색
def search_relevant_chunks(query, vectorizer, tfidf_matrix, text_chunks, top_k=3, threshold=0.005):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    indices = sims.argsort()[-top_k:][::-1]
    sel = [text_chunks[i] for i in indices if sims[i] > threshold]
    if not sel:
        sel = [text_chunks[i] for i in indices]
    return "\n\n".join(sel)

# Gemini 모델 반환
def get_model():
    return genai.GenerativeModel('gemini-2.0-flash')

# 질문 카테고리 분류 함수 추가
def classify_question_category(question):
    prompt = f"""
당신은 덤핑 관련 법령 전문가로서 사용자의 질문을 분석하여 가장 관련성 높은 법령 카테고리를 선택하는 업무를 담당합니다.

다음은 사용자의 질문입니다:
"{question}"

아래 법령 카테고리 중에서 이 질문과 가장 관련성이 높은 카테고리 하나만 선택해주세요:

1. 덤핑방지관세: 덤핑방지관세 부과에 관한 규칙, 덤핑마진, 정상가격, 수출가격 등 관련
2. 덤핑판정: 최종판정의결서, 예비판정, 산업피해, 실질적 피해, 인과관계 등 관련
3. 관련법령: 관세법, 불공정무역행위 조사 및 산업피해구제에 관한 법률 등 기본법령 관련

반드시 위의 카테고리 중 하나만 선택하고, 다음 형식으로만 답변해주세요:
"카테고리: [선택한 카테고리명]"

예를 들어, "카테고리: 덤핑방지관세"와 같이 답변해주세요.
"""
    model = get_model()
    response = model.generate_content(prompt)
    # 응답에서 카테고리 추출
    response_text = response.text
    if "카테고리:" in response_text:
        category = response_text.split("카테고리:")[1].strip()
        # 카테고리명만 정확히 추출
        for cat in LAW_CATEGORIES.keys():
            if cat in category:
                return cat
    # 분류가 명확하지 않은 경우 기본 카테고리 반환
    return "덤핑방지관세"  # 기본 카테고리로 설정

# 법령별 에이전트 응답 (async)
async def get_law_agent_response_async(law_name, question, history):
    if law_name not in st.session_state.embedding_data:
        text = st.session_state.law_data.get(law_name, "")
        vec, mat, chunks = create_embeddings_for_text(text)
        st.session_state.embedding_data[law_name] = (vec, mat, chunks)
    else:
        vec, mat, chunks = st.session_state.embedding_data[law_name]
    context = search_relevant_chunks(question, vec, mat, chunks)
    prompt = f"""
당신은 덤핑 및 무역 관련 전문가입니다. 주어진 자료를 기반으로 답변하되, 일반적으로 알려진 정보도 함께 제공해 주세요.

아래는 질문과 관련된 법령 및 자료 내용입니다:
{context}

이전 대화:
{history}

질문: {question}

# 응답 지침
1. 제공된 자료에서 찾은 정보와 일반적으로 알려진 정보를 모두 포함하여 답변해주세요.
2. 자료에서 찾은 정보는 출처(법령명, 조항 등)를 명확히 인용해주세요.
3. 자료에 없는 내용이더라도 일반적으로 알려진 사실이나 기술적 정보는 "일반 정보:" 문구와 함께 제공해주세요.
4. 답변은 다음 순서로 구성해주세요:
   - 일반적인 설명
   - 관련 법령 정보 (있는 경우)
   - 기술적/산업적 정보
   - 시장/무역 관련 정보
   - 참고할만한 추가 정보
"""
    model = get_model()
    loop = st.session_state.event_loop
    with ThreadPoolExecutor() as pool:
        res = await loop.run_in_executor(pool, lambda: model.generate_content(prompt))
    return law_name, res.text

# 모든 에이전트 병렬 실행
async def gather_agent_responses(question, history):
    tasks = [get_law_agent_response_async(name, question, history)
             for name in st.session_state.law_data]
    return await asyncio.gather(*tasks)

# 헤드 에이전트 통합 답변
def get_head_agent_response(responses, question, history):
    combined = "\n\n".join([f"=== {n} 관련 정보 ===\n{r}" for n, r in responses])
    prompt = f"""
당신은 덤핑 및 무역 분야의 전문가이자 기술 전문가입니다. 여러 자료의 정보를 통합하여 포괄적인 답변을 제공합니다.

{combined}

이전 대화:
{history}

질문: {question}

# 응답 지침
1. 여러 자료의 정보를 통합하여 다음 구조로 답변을 작성하세요:
   a) 일반적인 설명 (제품/기술/개념에 대한 기본 설명)
   b) 법령/규정 관련 정보 (있는 경우)
   c) 기술적/산업적 특징
   d) 시장/무역 관련 정보
   e) 참고할만한 추가 정보

2. 자료에서 찾은 정보는 출처를 명시하고, 일반적인 정보는 "일반 정보:" 문구와 함께 제공하세요.
3. 답변은 이해하기 쉽게 두괄식으로 작성하고, 필요한 경우 항목별로 구분하세요.
4. 자료에 없는 내용이더라도 일반적으로 알려진 정보는 포함하여 더 유용한 답변이 되도록 하세요.
"""
    return get_model().generate_content(prompt).text

# 대화 기록 렌더링
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 사용자 입력 및 응답
if user_input := st.chat_input("질문을 입력하세요"):
    # 이전 대화 내용 표시
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 새로운 질문 추가
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            # 모든 문서를 한번에 로드
            if not st.session_state.law_data:
                st.session_state.law_data = load_law_data()
            
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
            responses = st.session_state.event_loop.run_until_complete(
                gather_agent_responses(user_input, history)
            )
            answer = get_head_agent_response(responses, user_input, history)
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# 사이드바 안내
with st.sidebar:
    st.markdown("""
### ℹ️ 사용 안내

다음 법령들을 기반으로 답변을 제공합니다:

**덤핑방지관세 관련:**
- 중국산 더블레이어 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙
- 중국산 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙

**덤핑판정 관련:**
- 중국산 더블레이어 인쇄제판용 평면모양 사진플레이트 최종판정의결서
- 중국산 인쇄제판용 평면모양 사진플레이트 최종판정의결서

**관련법령:**
- 관세법, 시행령, 시행규칙
- 불공정무역행위 조사 및 산업피해구제에 관한 법률
""")
    if st.button("새 채팅 시작", type="primary"):
        st.session_state.chat_history = []
        st.rerun()