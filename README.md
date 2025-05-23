# 중국산 인쇄제판용 평면모양 사진플레이트 덤핑 전문가 챗봇

중국산 인쇄제판용 평면모양 사진플레이트에 대한 덤핑방지관세 부과 및 관련 법령을 기반으로 사용자의 질문에 답변하는 AI 챗봇입니다. Streamlit을 사용한 웹 인터페이스와 Google Gemini AI API를 활용하여 덤핑 조사 내용을 자동 분석하고, 사용자 질문에 근거와 법조항 출처를 명확히 제시하는 답변을 제공합니다.

## 주요 기능

- **덤핑 조사 자료 기반 답변**: 중국산 인쇄제판용 평면모양 사진플레이트 관련 덤핑방지관세 규칙 및 최종판정의결서 자동 분석
- **Multi Agents + Head Agents**: 각 자료별 agent가 AI 답변 생성, Head agent가 답변들을 취합하여 최종 답변 생성 
- **법령 조항 인용**: 답변에 관련 법령 조항 번호와 원문 출처를 명시
- **PDF 텍스트 추출 및 임베딩**: `pdf_utils.extract_text_from_pdf`로 텍스트 추출 후 TF-IDF 임베딩 생성
- **유사도 검색**: 청크 단위 TF-IDF 및 코사인 유사도 기반 유사 법령 구간 검색
- **병렬 처리 및 비동기 응답**: `asyncio.to_thread`로 여러 자료 카테고리 동시 질의 처리
- **대화 기록 저장**: `st.session_state`를 활용해 사용자와의 채팅 이력 관리
- **직관적 UI/UX**: expander, spinner, 버튼, selectbox 등을 활용한 사용자 친화적 인터페이스

## 설치 방법

1. 리포지토리 클론 및 이동
   ```bash
   git clone https://github.com/your-username/china-plate-dumping-chatbot.git
   cd china-plate-dumping-chatbot
   ```

2. 가상환경 생성 및 활성화
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수 설정
   - 프로젝트 루트에 `.env` 파일 생성
   - 다음 내용 추가:
     ```ini
     GOOGLE_API_KEY=your_google_api_key_here
     ```

5. 자료 PDF 파일 준비
   - `docs/` 폴더에 필요한 PDF 파일 저장

## 실행 방법

```bash
streamlit run main2.py
```

실행 후 제공되는 로컬 URL(기본: http://localhost:8501)에서 웹 챗봇 사용 가능

## 사용 방법

1. 브라우저에서 `http://localhost:8501`에 접속
2. Google Gemini API Key 입력
3. 질문 입력창에 덤핑 관련 질문 입력 후 전송
4. AI가 관련 자료와 법령 근거를 포함한 답변 제공

## 참고 자료 구성

- **덤핑방지관세 관련:**
  - 중국산 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙

- **덤핑판정 관련:**
  - 중국산 인쇄제판용 평면모양 사진플레이트 최종판정의결서

- **관련법령:**
  - 관세법
  - 관세법 시행령
  - 관세법 시행규칙
  - 불공정무역행위 조사 및 산업피해구제에 관한 법률

## 파일 구조

```
china-plate-dumping-chatbot/
├─ main2.py              # Streamlit 메인 스크립트
├─ pdf_utils.py          # PDF 텍스트 추출 유틸리티
├─ requirements.txt      # 의존성 목록
├─ .env                  # 환경 변수 파일 (API 키)
├─ venv                  # 가상 환경
├─ docs/                 # 자료 PDF 파일 디렉토리
│  ├─ 중국산 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙.pdf
│  ├─ 중국산 인쇄제판용 평면모양 사진플레이트_최종판정의결서.pdf
│  └─ ...
└─ README.md             # 프로젝트 설명
```
