import streamlit as st                     # 웹 인터페이스 제작을 위한 Streamlit
import os                                   # 운영체제 관련 기능 사용
import google.generativeai as genai        # Google Gemini AI API를 통한 텍스트 생성 기능
from pdf_utils import extract_text_from_pdf # PDF 문서에서 텍스트 추출 기능
import asyncio                              # 비동기 처리를 위한 asyncio 라이브러리
from concurrent.futures import ThreadPoolExecutor  # 병렬 처리를 위한 ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer  # 텍스트 데이터를 벡터화하기 위한 TF-IDF 도구
from sklearn.metrics.pairwise import cosine_similarity       # 코사인 유사도를 계산하기 위한 함수
from datetime import datetime
import requests                            # 웹 요청을 위한 라이브러리
import json                                # JSON 데이터 처리를 위한 라이브러리
import time                                # API 호출 제한을 위한 시간 처리
import re                                  # 정규 표현식을 위한 re 모듈
from google.api_core import exceptions as google_exceptions  # Google API 예외 처리
import aiohttp                             # 비동기 HTTP 요청을 위한 aiohttp 라이브러리

# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="중국산 더블레이어 사진플레이트 덤핑 전문가 챗봇",
    page_icon="📑",
    layout="wide"
)

# --- 답변 생성 시간 설정 ---
INITIAL_RESPONSE_TIMEOUT = 10  # 초기 답변 제한 시간 (초)
FOLLOWUP_RESPONSE_TIMEOUT = 60  # 후속 답변 제한 시간 (초)

# --- 유저로부터 API Key 입력 받기 ---
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'serper_api_key' not in st.session_state:
    st.session_state.serper_api_key = ""

with st.sidebar:
    # API 키 입력 부분을 먼저 배치
    with st.expander("🔑 API Key 설정", expanded=True):
        key_input = st.text_input(
            label="Google Gemini API Key 입력",
            type="password",
            placeholder="여기에 API Key를 입력하세요",
            value=st.session_state.gemini_api_key,
        )
        serper_key_input = st.text_input(
            label="Serper API Key 입력",
            type="password",
            placeholder="여기에 Serper API Key를 입력하세요",
            value=st.session_state.serper_api_key,
        )
        if key_input:
            st.session_state.gemini_api_key = key_input
        if serper_key_input:
            st.session_state.serper_api_key = serper_key_input

    st.divider()  # 구분선 추가

    # 사용 안내 부분
    st.title("📚 사용 안내")
    st.markdown("""
    ### 중국산 더블레이어 사진플레이트 전문가 챗봇
    
    이 챗봇은 중국산 더블레이어 인쇄제판용 평면 모양 사진플레이트에 대한 
    덤핑방지관세 부과 규칙과 관련 법령을 기반으로 만들어진 전문가 챗봇입니다.
    
    ### 주요 법령 근거
    - 중국산 더블레이어 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과 규칙
    - 중국산 더블레이어 인쇄제판용 평면모양 사진플레이트 최종판정
    - 관세법 및 시행령
    - 불공정무역행위 조사 및 산업피해구제에 관한 법률
    
    ### 문의 가능한 주제
    - 덤핑방지관세율 확인
    - 공급자별 세율 정보
    - 덤핑 판정 내용
    - 관련 법령 해석
    - 특수관계 공급자 확인
    """)

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
# 이벤트 루프 초기화
if 'event_loop' not in st.session_state:
    st.session_state.event_loop = None
# 답변 생성 시간 제어
if 'is_followup_question' not in st.session_state:
    st.session_state.is_followup_question = False
if 'last_question_time' not in st.session_state:
    st.session_state.last_question_time = None

# 답변 생성 시간 설정
INITIAL_RESPONSE_TIMEOUT = 10  # 초기 답변 제한 시간 (초)
FOLLOWUP_RESPONSE_TIMEOUT = 60  # 후속 답변 제한 시간 (초)

# --- 카테고리 정의 ---
LAW_CATEGORIES = {
    "덤핑방지관세": {
        "중국산 더블레이어 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙": "docs/중국산 더블레이어 인쇄제판용 평면 모양 사진플레이트에 대한 덤핑방지관세 부과에 관한 규칙(기획재정부령)(제00940호)(20221025).pdf",
    },
    "덤핑판정": {
        "중국산 더블레이어 인쇄제판용 평면모양 사진플레이트 최종판정": "docs/중국산 더블레이어 인쇄제판용 평면모양 사진플레이트_최종판정의결서.pdf",
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
    "덤핑방지관세": ["더블레이어", "인쇄제판용", "평면모양", "사진플레이트", "덤핑방지관세", "덤핑마진", "정상가격", "수출가격", "덤핑률"],
    "덤핑판정": ["더블레이어", "최종판정", "예비판정", "산업피해", "실질적 피해", "인과관계", "국내산업", "조사대상물품", "덤핑수입"],
    "관련법령": ["관세법", "시행령", "시행규칙", "불공정무역", "산업피해구제", "무역위원회", "조사절차", "덤핑규정"]
}

# 카테고리별 우선순위 설정
CATEGORY_PRIORITY = {
    "덤핑방지관세": 1,  # 가장 높은 우선순위
    "덤핑판정": 2,
    "관련법령": 3
}

# 공급자별 덤핑방지관세율 정보
SUPPLIERS_INFO = {
    "MAJOR_SUPPLIERS": {
        "러차이": {
            "name_kr": "러차이",
            "name_en": "Jiangsu Lecai Printing Material Co., Ltd.",
            "rate": 4.10,
            "description": "러차이 및 그 기업의 제품을 수출하는 자"
        },
        "코닥": {
            "name_kr": "코닥",
            "name_en": "Kodak (China) Graphic Communications Company Limited",
            "rate": 3.60,
            "description": "코닥과 그 관계사",
            "related_companies": [
                "코닥 인베스트먼트[Kodak (China) Investment Co., Ltd.]",
                "코닥 코리아(Kodak Korea Ltd.)",
                "이스트만 코닥(Eastman Kodak Company)",
                "화광(Lucky Huaguang Graphics Co., Ltd.)",
                "화광 난양(Lucky Huaguang Nanyang Trading Co., Ltd.)",
                "화광 바오리(Suzhou Huaguang Baoli Printing Plate Material Co., Ltd.)",
                "종인(Zhongyin Printing Equipment Co., Ltd.)",
                "아그파 화광[Agfa Huaguang (Shanghai) Printing Equipment Co., Ltd.]",
                "화푸(Henan Huafu Packaging Technology Co., Ltd.)",
                "코닥 일렉트로닉[Kodak Electronic Products (Shanghai) Company Limited]"
            ]
        },
        "화펑": {
            "name_kr": "화펑",
            "name_en": "Chongqing Huafeng Dijet Printing Material Co., Ltd.",
            "rate": 7.61,
            "description": "화펑과 그 관계사",
            "related_companies": [
                "화펑PM(Chongqing Huafeng Printing Material Co., Ltd.)"
            ]
        }
    },
    "OTHER_SUPPLIERS_RATE": 4.87,
    "OTHER_SUPPLIERS_DESCRIPTION": "그 밖의 공급자"
}

# API 호출 제한을 위한 설정
MAX_RETRIES = 3
RETRY_DELAY = 20  # seconds

def get_model_with_retry():
    """
    재시도 로직이 포함된 Gemini 모델 반환 함수
    """
    for attempt in range(MAX_RETRIES):
        try:
            return genai.GenerativeModel('gemini-2.0-flash')
        except google_exceptions.ResourceExhausted:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

def generate_content_with_retry(model, prompt):
    """
    재시도 로직이 포함된 content 생성 함수
    """
    for attempt in range(MAX_RETRIES):
        try:
            return model.generate_content(prompt)
        except google_exceptions.ResourceExhausted:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                st.error("API 호출 한도에 도달했습니다. 잠시 후 다시 시도해주세요.")
                return None
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            return None

def get_dumping_rate(supplier_name, product_info=None, special_relationship=None, use_web_search=True):
    """
    공급자의 덤핑방지관세율을 반환하는 함수
    
    Args:
        supplier_name (str): 공급자 이름
        product_info (dict, optional): 제품 정보
        special_relationship (str, optional): 특수관계가 있는 주요 공급자 이름
        use_web_search (bool): 웹 검색 사용 여부
    
    Returns:
        dict: 세율 정보를 포함한 딕셔너리
    """
    result = {
        "search_date": datetime.now().strftime("%Y-%m-%d"),
        "data_sources": ["Local Database"],
        "is_applicable": True,  # 덤핑방지관세 적용 여부
        "reason": ""
    }
    
    # 제품 정보 분석
    if product_info:
        product_analysis = analyze_product_info(product_info, use_web_search)
        result["product_analysis"] = product_analysis
        
        # 덤핑방지관세 대상이 아닌 경우
        if not product_analysis["is_target_product"]:
            result.update({
                "is_applicable": False,
                "rate": 0,
                "reason": "덤핑방지관세 부과대상 물품이 아님",
                "details": product_analysis["reason"]
            })
            return result
    
    # 웹 검색 수행
    if use_web_search and st.session_state.serper_api_key:
        web_info = analyze_web_info(supplier_name, "company")
        if web_info["status"] == "success":
            result["web_search_info"] = web_info
            result["data_sources"].append("Web Search")
    
    # 주요 공급자 검색
    for supplier_key, supplier_info in SUPPLIERS_INFO["MAJOR_SUPPLIERS"].items():
        if (supplier_name.lower() in supplier_info["name_kr"].lower() or 
            supplier_name.lower() in supplier_info["name_en"].lower()):
            result.update({
                "rate": supplier_info["rate"],
                "supplier_type": "major",
                "supplier_info": supplier_info,
                "reason": f"주요 공급자 {supplier_info['name_kr']}에 해당"
            })
            return result
    
    # 특수관계 검사
    if special_relationship:
        for supplier_key, supplier_info in SUPPLIERS_INFO["MAJOR_SUPPLIERS"].items():
            if (special_relationship.lower() in supplier_info["name_kr"].lower() or 
                special_relationship.lower() in supplier_info["name_en"].lower()):
                result.update({
                    "rate": supplier_info["rate"],
                    "supplier_type": "special_relationship",
                    "related_supplier": supplier_info,
                    "reason": f"{supplier_info['name_kr']}와(과)의 특수관계로 인해 해당 공급자의 세율 적용"
                })
                return result
    
    # 그 밖의 공급자
    result.update({
        "rate": SUPPLIERS_INFO["OTHER_SUPPLIERS_RATE"],
        "supplier_type": "other",
        "reason": "그 밖의 공급자에 해당"
    })
    return result

def search_info(query, api_key, search_type="company"):
    """
    Serper API를 사용하여 정보를 검색하는 함수
    
    Args:
        query (str): 검색어
        api_key (str): Serper API 키
        search_type (str): 검색 유형 ("company" 또는 "product")
    
    Returns:
        dict: 검색 결과 정보
    """
    # 캐시 키 생성
    cache_key = f"{query}_{search_type}"
    if 'search_cache' not in st.session_state:
        st.session_state.search_cache = {}
    
    # 캐시된 결과가 있으면 반환 (6시간 유효)
    if cache_key in st.session_state.search_cache:
        cached_result = st.session_state.search_cache[cache_key]
        if (datetime.now() - cached_result['timestamp']).total_seconds() < 21600:  # 6시간
            return cached_result['data']
    
    url = "https://google.serper.dev/search"
    results = []
    
    # 검색 쿼리 최적화
    search_queries = []
    if search_type == "company":
        # 중국 기업 정보 사이트 (최우선)
        search_queries.extend([
            f"{query} site:qcc.com",
            f"{query} site:tianyancha.com",
            f"{query} site:cninfo.com.cn"
        ])
        
        # 한국 기업 정보 사이트
        search_queries.extend([
            f"{query} site:dart.fss.or.kr",
            f"{query} site:nicebizinfo.com"
        ])
    else:  # product
        # 제품 검색 최적화
        search_queries.extend([
            f"{query} 사진플레이트 PS plate 규격",
            f"{query} printing plate specifications",
            f"{query} 印刷版 规格"
        ])
    
    # 병렬 검색 실행을 위한 함수
    async def search_query(session, query):
        payload = json.dumps({
            "q": query,
            "num": 5,  # 결과 수 제한
            "gl": "kr",
            "hl": "ko"
        })
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        try:
            # 20초 타임아웃 설정
            timeout = aiohttp.ClientTimeout(total=20)
            async with session.post(url, headers=headers, data=payload, timeout=timeout) as response:
                result = await response.json()
                if "organic" in result:
                    return result["organic"]
        except asyncio.TimeoutError:
            print(f"Timeout for query: {query}")
            return []
        except Exception as e:
            print(f"Error searching with query '{query}': {str(e)}")
            return []
        return []

    # 비동기 검색 실행
    async def run_searches():
        # 60초 전체 타임아웃 설정
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [search_query(session, q) for q in search_queries]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # 예외 처리 및 유효한 결과만 반환
                valid_results = []
                for r in results:
                    if isinstance(r, list):
                        valid_results.extend(r)
                return valid_results
            except asyncio.TimeoutError:
                print("Total search timeout")
                return []

    # 이벤트 루프 실행
    if not st.session_state.event_loop:
        st.session_state.event_loop = asyncio.new_event_loop()
    
    try:
        results = st.session_state.event_loop.run_until_complete(run_searches())
    except Exception as e:
        print(f"Error in search execution: {str(e)}")
        results = []
    
    # 중복 결과 제거 및 정렬
    seen_links = set()
    unique_results = []
    for result in results:
        link = result.get("link", "")
        if link not in seen_links:
            seen_links.add(link)
            # 중국 기업 정보 사이트 우선순위 부여
            priority = 1
            if any(site in link for site in ["qcc.com", "tianyancha.com", "cninfo.com.cn"]):
                priority = 0
            result["priority"] = priority
            unique_results.append(result)
    
    # 우선순위에 따라 정렬
    unique_results.sort(key=lambda x: (x["priority"], -len(x.get("snippet", ""))))
    
    # 상위 10개 결과만 유지
    final_results = unique_results[:10]
    
    # 결과 캐싱
    search_result = {
        "organic": final_results,
        "query": query,
        "search_type": search_type
    }
    
    # 캐시 크기 제한 (최대 100개)
    if len(st.session_state.search_cache) >= 100:
        oldest_key = min(st.session_state.search_cache.keys(), 
                        key=lambda k: st.session_state.search_cache[k]['timestamp'])
        del st.session_state.search_cache[oldest_key]
    
    st.session_state.search_cache[cache_key] = {
        'data': search_result,
        'timestamp': datetime.now()
    }
    
    return search_result

def analyze_web_info(query, search_type="company"):
    """
    웹 검색 결과를 분석하여 정보를 추출하는 함수
    
    Args:
        query (str): 검색어
        search_type (str): 검색 유형 ("company" 또는 "product")
    
    Returns:
        dict: 분석된 정보
    """
    if not st.session_state.serper_api_key:
        return {
            "status": "error",
            "message": "Serper API 키가 설정되지 않았습니다.",
            "search_date": datetime.now().strftime("%Y-%m-%d")
        }
    
    search_results = search_info(query, st.session_state.serper_api_key, search_type)
    
    if "error" in search_results:
        return {
            "status": "error",
            "message": f"검색 중 오류 발생: {search_results['error']}",
            "search_date": datetime.now().strftime("%Y-%m-%d")
        }
    
    result_info = {
        "query": query,
        "search_date": datetime.now().strftime("%Y-%m-%d"),
        "status": "success",
        "company_details": {
            "basic_info": {
                "company_name": "",
                "company_name_en": "",
                "establishment_date": "",
                "business_number": "",
                "representative": "",
                "website": "",
                "contact": "",
                "main_business": []
            },
            "addresses": [],            # 주소 정보
            "registration": [],         # 사업자/법인 등록 정보
            "shareholders": [],         # 주주 정보
            "subsidiaries": [],         # 자회사 정보
            "parent_companies": [],     # 모회사 정보
            "business_scope": [],       # 사업 범위
            "trade_info": [],          # 무역 정보
            "financial_info": [],      # 재무 정보
            "certifications": []       # 인증 정보
        },
        "special_relationships": [],    # 특수관계 정보
        "news_and_updates": [],        # 최신 뉴스 및 업데이트
        "raw_search_results": search_results
    }
    
    try:
        for result in search_results.get("organic", []):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            combined_text = (title + " " + snippet).lower()
            
            if search_type == "company":
                # 기본 정보 추출
                if any(keyword in combined_text 
                      for keyword in ["회사소개", "기업개요", "company profile", "about us", 
                                    "企业简介", "公司简介"]):
                    # 대표자명 추출
                    rep_match = re.search(r"대표[자이사]*[:\s]*([\w\s]+)", snippet)
                    if rep_match:
                        result_info["company_details"]["basic_info"]["representative"] = rep_match.group(1).strip()
                    
                    # 설립일 추출
                    date_match = re.search(r"설립[일자]*[:\s]*(\d{4}[년\s]*\d{1,2}[월\s]*\d{1,2}[일]*)", snippet)
                    if date_match:
                        result_info["company_details"]["basic_info"]["establishment_date"] = date_match.group(1)
                    
                    # 사업자번호 추출
                    biz_match = re.search(r"사업자[등록]*번호[:\s]*(\d{3}-\d{2}-\d{5})", snippet)
                    if biz_match:
                        result_info["company_details"]["basic_info"]["business_number"] = biz_match.group(1)
                
                # 주소 정보
                if any(keyword in combined_text 
                      for keyword in ["주소", "소재지", "address", "location", "所在地", "地址"]):
                    result_info["company_details"]["addresses"].append({
                        "address": snippet,
                        "source": link,
                        "type": "본사" if "본사" in combined_text else "사업장"
                    })
                
                # 주주 정보
                if any(keyword in combined_text 
                      for keyword in ["주주", "지분", "출자", "shareholder", "股东", "持股"]):
                    result_info["company_details"]["shareholders"].append({
                        "info": snippet,
                        "source": link,
                        "date_found": datetime.now().strftime("%Y-%m-%d")
                    })
                
                # 자회사/계열사 정보
                if any(keyword in combined_text 
                      for keyword in ["자회사", "계열사", "subsidiary", "affiliate", "子公司", "关联公司"]):
                    result_info["company_details"]["subsidiaries"].append({
                        "info": snippet,
                        "source": link,
                        "date_found": datetime.now().strftime("%Y-%m-%d")
                    })
                
                # 모회사 정보
                if any(keyword in combined_text 
                      for keyword in ["모회사", "지주회사", "parent company", "holding company", 
                                    "母公司", "控股公司"]):
                    result_info["company_details"]["parent_companies"].append({
                        "info": snippet,
                        "source": link,
                        "date_found": datetime.now().strftime("%Y-%m-%d")
                    })
                
                # 사업 범위
                if any(keyword in combined_text 
                      for keyword in ["사업영역", "주요제품", "business scope", "products", 
                                    "经营范围", "主营产品"]):
                    result_info["company_details"]["business_scope"].append({
                        "info": snippet,
                        "source": link,
                        "date_found": datetime.now().strftime("%Y-%m-%d")
                    })
                
                # 무역 정보
                if any(keyword in combined_text 
                      for keyword in ["수출", "수입", "무역", "export", "import", "trade", 
                                    "进出口", "贸易"]):
                    result_info["company_details"]["trade_info"].append({
                        "info": snippet,
                        "source": link,
                        "date_found": datetime.now().strftime("%Y-%m-%d")
                    })
                
                # 재무 정보
                if any(keyword in combined_text 
                      for keyword in ["매출", "영업이익", "당기순이익", "revenue", "profit", 
                                    "营收", "利润"]):
                    result_info["company_details"]["financial_info"].append({
                        "info": snippet,
                        "source": link,
                        "date_found": datetime.now().strftime("%Y-%m-%d")
                    })
                
                # 인증 정보
                if any(keyword in combined_text 
                      for keyword in ["인증", "특허", "certification", "patent", 
                                    "认证", "专利"]):
                    result_info["company_details"]["certifications"].append({
                        "info": snippet,
                        "source": link,
                        "date_found": datetime.now().strftime("%Y-%m-%d")
                    })
            
            # 최신 뉴스 및 업데이트
            if any(keyword in combined_text
                  for keyword in ["뉴스", "공시", "보도", "news", "update", "announcement", 
                                "新闻", "公告"]):
                result_info["news_and_updates"].append({
                    "title": title,
                    "content": snippet,
                    "source": link,
                    "date_found": datetime.now().strftime("%Y-%m-%d")
                })
    
    except Exception as e:
        result_info["analysis_error"] = str(e)
    
    # 특수관계 분석
    if search_type == "company":
        special_relationship = check_special_relationship({"name": query})
        if special_relationship["has_special_relationship"]:
            result_info["special_relationships"] = special_relationship["relationships"]
    
    return result_info

def format_company_info(company_info):
    """
    기업 정보를 보기 좋게 포맷팅하는 함수
    
    Args:
        company_info (dict): analyze_web_info 함수의 결과
    
    Returns:
        str: 포맷팅된 기업 정보
    """
    if company_info["status"] != "success":
        return f"기업 정보 검색 실패: {company_info.get('message', '알 수 없는 오류')}"
    
    formatted_info = []
    formatted_info.append(f"## 🏢 {company_info['query']} 기업 정보")
    formatted_info.append(f"*검색 일자: {company_info['search_date']}*")
    formatted_info.append("---")
    
    # 기본 정보
    basic_info = company_info["company_details"]["basic_info"]
    if any(basic_info.values()):
        formatted_info.append("### 📋 기본 정보")
        if basic_info["company_name"]: 
            formatted_info.append(f"- 기업명: {basic_info['company_name']}")
        if basic_info["company_name_en"]: 
            formatted_info.append(f"- 영문명: {basic_info['company_name_en']}")
        if basic_info["establishment_date"]: 
            formatted_info.append(f"- 설립일: {basic_info['establishment_date']}")
        if basic_info["business_number"]: 
            formatted_info.append(f"- 사업자등록번호: {basic_info['business_number']}")
        if basic_info["representative"]: 
            formatted_info.append(f"- 대표자: {basic_info['representative']}")
        formatted_info.append("")
    
    # 주소 정보
    if company_info["company_details"]["addresses"]:
        formatted_info.append("### 📍 사업장 정보")
        for addr in company_info["company_details"]["addresses"]:
            formatted_info.append(f"- {addr['type']}: {addr['address']}")
        formatted_info.append("")
    
    # 주주 및 지분 정보
    if company_info["company_details"]["shareholders"]:
        formatted_info.append("### 👥 주주 및 지분 정보")
        for shareholder in company_info["company_details"]["shareholders"]:
            formatted_info.append(f"- {shareholder['info']}")
        formatted_info.append("")
    
    # 자회사/계열사 정보
    if company_info["company_details"]["subsidiaries"]:
        formatted_info.append("### 🔄 자회사/계열사 정보")
        for subsidiary in company_info["company_details"]["subsidiaries"]:
            formatted_info.append(f"- {subsidiary['info']}")
        formatted_info.append("")
    
    # 모회사 정보
    if company_info["company_details"]["parent_companies"]:
        formatted_info.append("### ⬆️ 모회사 정보")
        for parent in company_info["company_details"]["parent_companies"]:
            formatted_info.append(f"- {parent['info']}")
        formatted_info.append("")
    
    # 사업 범위
    if company_info["company_details"]["business_scope"]:
        formatted_info.append("### 🎯 사업 범위")
        for scope in company_info["company_details"]["business_scope"]:
            formatted_info.append(f"- {scope['info']}")
        formatted_info.append("")
    
    # 무역 정보
    if company_info["company_details"]["trade_info"]:
        formatted_info.append("### 🌐 무역 활동")
        for trade in company_info["company_details"]["trade_info"]:
            formatted_info.append(f"- {trade['info']}")
        formatted_info.append("")
    
    # 재무 정보
    if company_info["company_details"]["financial_info"]:
        formatted_info.append("### 💰 재무 정보")
        for finance in company_info["company_details"]["financial_info"]:
            formatted_info.append(f"- {finance['info']}")
        formatted_info.append("")
    
    # 인증 정보
    if company_info["company_details"]["certifications"]:
        formatted_info.append("### 📜 인증 및 특허")
        for cert in company_info["company_details"]["certifications"]:
            formatted_info.append(f"- {cert['info']}")
        formatted_info.append("")
    
    # 특수관계 정보
    if company_info.get("special_relationships"):
        formatted_info.append("### ⚠️ 덤핑방지관세 대상 기업과의 특수관계")
        for relationship in company_info["special_relationships"]:
            formatted_info.append(f"- 관련 기업: **{relationship['major_supplier']}**")
            formatted_info.append(f"  - 신뢰도 점수: {relationship['confidence_score']:.2f}")
            for rel in relationship["relationships_found"]:
                formatted_info.append(f"  - {rel['description']}: {rel['detail']}")
        formatted_info.append("")
    
    # 최신 뉴스
    if company_info["news_and_updates"]:
        formatted_info.append("### 📰 최신 소식")
        for news in company_info["news_and_updates"][:5]:  # 최근 5개만 표시
            formatted_info.append(f"- {news['title']}")
            formatted_info.append(f"  {news['content']}")
        formatted_info.append("")
    
    return "\n".join(formatted_info)

def check_special_relationship(company_info, use_web_search=True):
    """
    특수관계 여부를 검사하는 함수
    
    Args:
        company_info (dict): 회사 정보를 포함한 딕셔너리
        use_web_search (bool): 웹 검색 사용 여부
    
    Returns:
        dict: 특수관계 분석 결과
    """
    relationships = []
    
    # 캐시 키 생성
    cache_key = f"special_relationship_{company_info['name']}"
    if 'relationship_cache' not in st.session_state:
        st.session_state.relationship_cache = {}
    
    # 캐시된 결과가 있으면 반환
    if cache_key in st.session_state.relationship_cache:
        cached_result = st.session_state.relationship_cache[cache_key]
        # 캐시 유효기간 확인 (24시간)
        if (datetime.now() - cached_result['timestamp']).total_seconds() < 86400:
            return cached_result['data']
    
    # 웹 검색을 통한 추가 정보 수집
    web_info = None
    if use_web_search and st.session_state.serper_api_key:
        web_info = analyze_web_info(company_info["name"], "company")
    
    # 주요 공급자들과의 관계 검사
    for supplier_key, supplier_info in SUPPLIERS_INFO["MAJOR_SUPPLIERS"].items():
        relationship = {
            "major_supplier": supplier_info["name_kr"],
            "major_supplier_en": supplier_info["name_en"],
            "relationships_found": [],
            "confidence_score": 0.0,  # 관계 신뢰도 점수
            "evidence": []  # 증거 자료 저장
        }
        
        # 1. 기본 검사 (회사명 유사성)
        name_similarity = calculate_name_similarity(company_info["name"], supplier_info)
        if name_similarity > 0.7:
            relationship["relationships_found"].append({
                "type": "name_similarity",
                "description": "회사명 유사성 발견",
                "confidence": name_similarity,
                "details": f"유사도 점수: {name_similarity:.2f}"
            })
            relationship["confidence_score"] += name_similarity
            relationship["evidence"].append({
                "type": "name_match",
                "source": "기업명 분석",
                "details": f"검사 대상: {company_info['name']}, 주요 공급자: {supplier_info['name_kr']}/{supplier_info['name_en']}"
            })
        
        # 2. 웹 검색 결과 분석
        if web_info and web_info["status"] == "success":
            # 2.1 주소 정보 분석
            for address in web_info["company_details"]["addresses"]:
                address_similarity = calculate_address_similarity(address["address"], supplier_info.get("address", ""))
                if address_similarity > 0.8:
                    relationship["relationships_found"].append({
                        "type": "address_match",
                        "description": "주소 일치",
                        "confidence": address_similarity,
                        "source": address.get("source", ""),
                        "detail": address["address"]
                    })
                    relationship["confidence_score"] += address_similarity
                    relationship["evidence"].append({
                        "type": "address_match",
                        "source": address.get("source", "주소 정보 분석"),
                        "details": f"일치 점수: {address_similarity:.2f}"
                    })
            
            # 2.2 주주/지분 관계 분석
            for shareholder in web_info["company_details"]["shareholders"]:
                if analyze_shareholder_relationship(shareholder, supplier_info):
                    relationship["relationships_found"].append({
                        "type": "shareholding",
                        "description": "주주/지분 관계 발견",
                        "confidence": 0.9,
                        "source": shareholder.get("source", ""),
                        "detail": shareholder.get("description", "")
                    })
                    relationship["confidence_score"] += 0.9
                    relationship["evidence"].append({
                        "type": "shareholding",
                        "source": shareholder.get("source", "주주 정보 분석"),
                        "details": shareholder.get("description", "")
                    })
            
            # 2.3 자회사/모회사 관계 분석
            for subsidiary in web_info["company_details"]["subsidiaries"]:
                if analyze_company_relationship(subsidiary, supplier_info):
                    relationship["relationships_found"].append({
                        "type": "subsidiary",
                        "description": "자회사 관계 발견",
                        "confidence": 0.9,
                        "source": subsidiary.get("source", ""),
                        "detail": subsidiary.get("description", "")
                    })
                    relationship["confidence_score"] += 0.9
                    relationship["evidence"].append({
                        "type": "subsidiary",
                        "source": subsidiary.get("source", "자회사 정보 분석"),
                        "details": subsidiary.get("description", "")
                    })
            
            for parent in web_info["company_details"]["parent_companies"]:
                if analyze_company_relationship(parent, supplier_info):
                    relationship["relationships_found"].append({
                        "type": "parent_company",
                        "description": "모회사 관계 발견",
                        "confidence": 0.9,
                        "source": parent.get("source", ""),
                        "detail": parent.get("description", "")
                    })
                    relationship["confidence_score"] += 0.9
                    relationship["evidence"].append({
                        "type": "parent_company",
                        "source": parent.get("source", "모회사 정보 분석"),
                        "details": parent.get("description", "")
                    })
        
        # 관계가 발견되고 신뢰도가 충분한 경우에만 결과에 추가
        if relationship["relationships_found"]:
            # 신뢰도 점수 정규화 (0~1 범위)
            relationship["confidence_score"] = min(1.0, relationship["confidence_score"] / len(relationship["relationships_found"]))
            # 높은 신뢰도(0.7 이상) 관계만 포함
            if relationship["confidence_score"] >= 0.7:
                # 낮은 신뢰도 관계는 제외
                relationship["relationships_found"] = [
                    r for r in relationship["relationships_found"]
                    if r["confidence"] >= 0.7
                ]
                if relationship["relationships_found"]:  # 높은 신뢰도 관계가 남아있는 경우만 추가
                    relationships.append(relationship)
    
    result = {
        "has_special_relationship": len(relationships) > 0,
        "relationships": relationships,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "data_sources": ["Local Database"]
    }
    
    if web_info:
        result["data_sources"].append("Web Search")
    
    # 특수관계가 있는 경우에만 상세 정보 포함
    if result["has_special_relationship"]:
        high_confidence_relationships = [r for r in relationships if r["confidence_score"] >= 0.8]
        result.update({
            "high_confidence_relationship": len(high_confidence_relationships) > 0,
            "relationship_summary": {
                "total_relationships": len(relationships),
                "high_confidence_relationships": len(high_confidence_relationships),
                "highest_confidence_score": max([r["confidence_score"] for r in relationships])
            }
        })
        # 웹 검색 정보는 유의미한 특수관계가 있는 경우에만 포함
        if web_info:
            result["web_search_info"] = web_info
    else:
        # 특수관계가 없는 경우 간단한 결과만 반환
        result = {
            "has_special_relationship": False,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "message": "유의미한 특수관계가 발견되지 않았습니다."
        }
    
    # 결과 캐싱
    st.session_state.relationship_cache[cache_key] = {
        'data': result,
        'timestamp': datetime.now()
    }
    
    return result

def calculate_name_similarity(name1, supplier_info):
    """
    회사명 유사도를 계산하는 함수
    """
    name1 = name1.lower()
    name_kr = supplier_info["name_kr"].lower()
    name_en = supplier_info["name_en"].lower()
    
    # 정확한 일치 검사
    if name1 in name_kr or name_kr in name1:
        return 1.0
    if name1 in name_en or name_en in name1:
        return 1.0
    
    # 부분 일치 검사
    name1_words = set(re.findall(r'\w+', name1))
    name_kr_words = set(re.findall(r'\w+', name_kr))
    name_en_words = set(re.findall(r'\w+', name_en))
    
    # 한글 이름 유사도
    kr_similarity = len(name1_words & name_kr_words) / max(len(name1_words), len(name_kr_words))
    # 영문 이름 유사도
    en_similarity = len(name1_words & name_en_words) / max(len(name1_words), len(name_en_words))
    
    return max(kr_similarity, en_similarity)

def calculate_address_similarity(addr1, addr2):
    """
    주소 유사도를 계산하는 함수
    """
    if not addr1 or not addr2:
        return 0.0
    
    addr1 = addr1.lower()
    addr2 = addr2.lower()
    
    # 정확한 일치 검사
    if addr1 == addr2:
        return 1.0
    
    # 부분 일치 검사
    addr1_words = set(re.findall(r'\w+', addr1))
    addr2_words = set(re.findall(r'\w+', addr2))
    
    # 주소 유사도
    similarity = len(addr1_words & addr2_words) / max(len(addr1_words), len(addr2_words))
    
    return similarity

def analyze_shareholder_relationship(shareholder, supplier_info):
    """
    주주 관계를 분석하는 함수
    """
    description = shareholder.get("description", "").lower()
    name_kr = supplier_info["name_kr"].lower()
    name_en = supplier_info["name_en"].lower()
    
    # 주요 키워드
    keywords = ["주주", "지분", "출자", "shareholder", "stake", "ownership", "股东", "持股"]
    
    if any(keyword in description for keyword in keywords):
        if name_kr in description or name_en in description:
            return True
    
    return False

def analyze_company_relationship(company, supplier_info):
    """
    회사 관계를 분석하는 함수
    """
    description = company.get("description", "").lower()
    name_kr = supplier_info["name_kr"].lower()
    name_en = supplier_info["name_en"].lower()
    
    # 주요 키워드
    keywords = ["자회사", "계열사", "모회사", "지주회사", "subsidiary", "affiliate", "parent", 
               "子公司", "关联公司", "母公司", "控股公司"]
    
    if any(keyword in description for keyword in keywords):
        if name_kr in description or name_en in description:
            return True
    
    return False

def analyze_product_info(product_info, use_web_search=True):
    """
    제품 정보를 분석하여 덤핑방지관세 대상 여부를 판단하는 함수
    
    Args:
        product_info (dict): 제품 정보를 포함한 딕셔너리
        use_web_search (bool): 웹 검색 사용 여부
    
    Returns:
        dict: 제품 분석 결과
    """
    result = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "data_sources": ["Local Database"],
        "product_name": product_info.get("name", ""),
        "model": product_info.get("model", ""),
        "specifications": product_info.get("specifications", {}),
        "is_target_product": False,
        "reason": "",
        "dumping_duty_info": None
    }
    
    # 웹 검색을 통한 추가 정보 수집
    if use_web_search and st.session_state.serper_api_key:
        search_query = f"{result['product_name']} {result.get('model', '')}".strip()
        web_info = analyze_web_info(search_query, "product")
        
        if web_info and web_info["status"] == "success":
            result["data_sources"].append("Web Search")
            result["web_search_info"] = web_info
            
            # 기술 사양 정보 추가
            if web_info.get("technical_specs"):
                result["specifications"].update({
                    "web_found_specs": web_info["technical_specs"]
                })
            
            # 덤핑방지관세 관련 정보 추가
            if web_info.get("dumping_duty_info"):
                result["dumping_duty_info"] = web_info["dumping_duty_info"]
    
    # 제품이 덤핑방지관세 대상인지 판단
    # 1. 제품명 기반 검사
    product_keywords = ["인쇄제판용", "평면모양", "사진플레이트", "printing plate", "PS plate"]
    if any(keyword in str(result["product_name"]).lower() for keyword in product_keywords):
        result["is_target_product"] = True
        result["reason"] = "제품명에서 대상 품목 키워드 발견"
    
    # 2. 사양 기반 검사
    if result["specifications"]:
        spec_keywords = ["PS", "presensitized", "감광", "알루미늄", "aluminum", "offset"]
        spec_text = str(result["specifications"]).lower()
        if any(keyword in spec_text for keyword in spec_keywords):
            result["is_target_product"] = True
            result["reason"] = "제품 사양에서 대상 품목 특성 발견"
    
    # 3. 웹 검색 결과 기반 검사
    if result.get("web_search_info"):
        if result["web_search_info"].get("dumping_duty_info"):
            result["is_target_product"] = True
            result["reason"] = "웹 검색 결과에서 덤핑방지관세 대상 확인"
    
    return result

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

# Gemini 모델 반환 함수 수정
def get_model():
    return get_model_with_retry()

def summarize_pdf_content(pdf_path, chunk_size=3000):
    """
    PDF 문서의 내용을 요약하는 함수
    
    Args:
        pdf_path (str): PDF 파일 경로
        chunk_size (int): 각 청크의 크기
    
    Returns:
        str: 요약된 내용
    """
    try:
        # PDF 텍스트 추출
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            return "PDF 내용을 추출할 수 없습니다."

        # 텍스트를 청크로 분할
        chunks = []
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            if len(chunk.strip()) > 100:  # 의미 있는 내용이 있는 청크만 포함
                chunks.append(chunk)

        # 각 청크 요약
        model = get_model()
        summaries = []
        
        for chunk in chunks:
            prompt = f"""
다음 텍스트를 요약해주세요. 핵심 내용만 간단명료하게 작성하되, 중요한 수치나 결정사항은 반드시 포함해주세요.

텍스트:
{chunk}

요약 형식:
- bullet point 형식으로 작성
- 각 요점은 1-2문장으로 제한
- 중요 수치와 결정사항 강조
"""
            result = generate_content_with_retry(model, prompt)
            if result:
                summaries.append(result.text)
            time.sleep(1)  # API 호출 제한 방지

        # 전체 요약 생성
        if not summaries:
            return "문서 요약을 생성할 수 없습니다."

        final_summary_prompt = f"""
다음은 문서의 각 부분 요약입니다. 이를 바탕으로 전체 문서의 핵심 내용을 종합적으로 요약해주세요.

각 부분 요약:
{' '.join(summaries)}

요약 형식:
1. 문서 개요 (1-2문장)
2. 주요 결정사항 (bullet points)
3. 중요 수치 및 데이터 (bullet points)
4. 결론 (1-2문장)
"""
        final_result = generate_content_with_retry(model, final_summary_prompt)
        return final_result.text if final_result else "최종 요약을 생성할 수 없습니다."

    except Exception as e:
        return f"요약 중 오류가 발생했습니다: {str(e)}"

# 빠른 요약 생성 함수
def generate_quick_summary(responses, question):
    """
    수집된 응답들을 빠르게 요약하는 함수
    """
    if not responses:
        return get_quick_response(question)
    
    # 응답의 관련성 점수 계산
    scored_responses = []
    for law_name, response in responses:
        # 응답 텍스트에서 질문 키워드 매칭 수 계산
        question_keywords = set(re.findall(r'\w+', question.lower()))
        response_text = response.lower()
        matched_keywords = sum(1 for keyword in question_keywords 
                             if len(keyword) > 1 and keyword in response_text)
        relevance_score = matched_keywords / len(question_keywords) if question_keywords else 0
        
        # 응답 길이도 점수에 반영 (너무 짧은 응답은 제외)
        length_score = min(len(response) / 1000, 1.0)  # 1000자를 기준으로 정규화
        
        total_score = (relevance_score * 0.7) + (length_score * 0.3)  # 관련성 70%, 길이 30% 반영
        scored_responses.append((law_name, response, total_score))
    
    # 점수가 높은 순으로 정렬
    scored_responses.sort(key=lambda x: x[2], reverse=True)
    
    # 상위 2개 응답만 선택
    top_responses = scored_responses[:2]
    
    # 요약 생성
    summary_parts = []
    for law_name, response, score in top_responses:
        # 응답에서 핵심 문장 추출 (처음 2-3문장)
        sentences = re.split(r'[.!?]\s+', response)
        key_sentences = sentences[:3]
        summary = '. '.join(key_sentences) + '.'
        
        summary_parts.append(f"[{law_name}]\n{summary}")
    
    formatted_summary = "\n\n".join(summary_parts)
    
    return f"""
시간 제한으로 인해 현재까지 찾은 가장 관련성 높은 정보를 기반으로 답변 드립니다.
더 자세한 정보를 원하시면 추가 질문을 해주세요.

{formatted_summary}
"""

# 비동기 처리를 위한 새로운 함수
async def process_user_input(user_input, history):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.event_loop = loop
        
        # 현재 시간과 마지막 질문 시간의 차이 계산
        current_time = time.time()
        if st.session_state.last_question_time:
            time_diff = current_time - st.session_state.last_question_time
            st.session_state.is_followup_question = time_diff < 30
        
        # 1차 질문인 경우 빠른 응답 생성
        if not st.session_state.is_followup_question:
            try:
                async with asyncio.timeout(INITIAL_RESPONSE_TIMEOUT):
                    # 빠른 초기 응답 생성
                    answer = get_quick_response(user_input)
                    st.session_state.last_question_time = current_time
                    return answer
            except asyncio.TimeoutError:
                return "죄송합니다. 응답 시간이 초과되었습니다. 다시 질문해주세요."
        
        # 후속 질문인 경우 기존 로직 사용
        relevant_categories = analyze_question_categories(user_input)
        partial_responses = []
        found_relevant_answer = False
        
        try:
            async with asyncio.timeout(FOLLOWUP_RESPONSE_TIMEOUT):
                # 우선순위가 높은 카테고리부터 처리
                for category in sorted(relevant_categories, key=lambda x: CATEGORY_PRIORITY[x]):
                    if found_relevant_answer:
                        break
                        
                    async for response in stream_agent_responses(user_input, history, category):
                        partial_responses.append(response)
                        if is_response_relevant(response[1], user_input):
                            found_relevant_answer = True
                            break
                
                if not found_relevant_answer:
                    remaining_categories = set(LAW_CATEGORIES.keys()) - set(relevant_categories)
                    for category in sorted(remaining_categories, key=lambda x: CATEGORY_PRIORITY[x]):
                        async for response in stream_agent_responses(user_input, history, category):
                            partial_responses.append(response)
                
                answer = get_head_agent_response(partial_responses, user_input, history)
                
        except asyncio.TimeoutError:
            if partial_responses:
                answer = generate_quick_summary(partial_responses, user_input)
            else:
                answer = get_quick_response(user_input)
        
        st.session_state.last_question_time = current_time
        return answer
        
    finally:
        loop.close()
        st.session_state.event_loop = None

def analyze_question_categories(question):
    """
    질문을 분석하여 관련된 카테고리를 우선순위대로 반환
    """
    relevant_categories = []
    question_lower = question.lower()
    
    # 카테고리별 키워드 매칭
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            relevant_categories.append(category)
    
    # 매칭된 카테고리가 없으면 기본 우선순위 반환
    if not relevant_categories:
        return list(CATEGORY_PRIORITY.keys())
    
    return relevant_categories

def is_response_relevant(response, question):
    """
    응답의 관련성을 검사하는 함수
    """
    # 응답이 너무 짧으면 관련성이 낮다고 판단
    if len(response) < 100:
        return False
        
    # 질문의 주요 키워드가 응답에 포함되어 있는지 확인
    question_keywords = set(re.findall(r'\w+', question.lower()))
    response_text = response.lower()
    
    # 핵심 키워드 매칭 수 계산
    matched_keywords = sum(1 for keyword in question_keywords 
                         if len(keyword) > 1 and keyword in response_text)
    
    # 키워드 매칭 비율이 30% 이상이면 관련성이 높다고 판단
    return matched_keywords / len(question_keywords) >= 0.3 if question_keywords else False

async def stream_agent_responses(question, history, category):
    """
    특정 카테고리의 문서에 대해서만 응답을 생성하는 함수
    """
    for law_name, pdf_path in LAW_CATEGORIES[category].items():
        try:
            response = await asyncio.wait_for(
                get_law_agent_response_async(law_name, question, history),
                timeout=1.0  # 각 문서당 1초로 제한
            )
            yield response
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"Error processing {law_name}: {str(e)}")
            continue

def get_quick_response(question):
    """
    빠른 초기 응답을 생성하는 함수
    """
    model = get_model()
    prompt = f"""
다음 질문에 대해 10초 이내로 핵심적인 답변만 간단히 제공해주세요.
필요한 경우 "더 자세한 정보를 원하시면 추가 질문을 해주세요"라는 문구를 포함해주세요.

질문: {question}

답변 형식:
1. 핵심 답변 (1-2문장)
2. 관련 키워드
3. 추가 질문 유도
"""
    try:
        result = generate_content_with_retry(model, prompt)
        return result.text if result else "죄송합니다. 빠른 답변을 생성할 수 없습니다. 다시 질문해주세요."
    except Exception as e:
        return f"죄송합니다. 오류가 발생했습니다: {str(e)}"

# 법령별 에이전트 응답 (async) 수정
async def get_law_agent_response_async(law_name, question, history):
    if law_name not in st.session_state.embedding_data:
        text = st.session_state.law_data.get(law_name, "")
        vec, mat, chunks = create_embeddings_for_text(text)
        st.session_state.embedding_data[law_name] = (vec, mat, chunks)
    else:
        vec, mat, chunks = st.session_state.embedding_data[law_name]
    
    # 문서 요약 요청 확인
    if "요약" in question.lower() or "정리" in question.lower():
        pdf_path = None
        for category in LAW_CATEGORIES.values():
            if law_name in category:
                pdf_path = category[law_name]
                break
        
        if pdf_path:
            summary = summarize_pdf_content(pdf_path)
            return law_name, summary
    
    context = search_relevant_chunks(question, vec, mat, chunks)
    
    supplier_info = None
    if any(keyword in question.lower() for keyword in ["공급자", "수출자", "제조자", "세율", "관세율"]):
        supplier_info = get_dumping_rate("코닥그래픽")
    
    prompt = f"""
당신은 중국산 인쇄제판용 평면모양 사진플레이트 덤핑 전문가입니다. 주어진 모든 자료를 종합적으로 분석하여 답변해주세요.

아래는 질문과 관련된 자료 내용입니다:
{context}

{"공급자 세율 정보:" + str(supplier_info) if supplier_info else ""}

이전 대화:
{history}

질문: {question}

# 응답 지침
1. 답변 구조:
   - 핵심 내용 요약 (2-3문장)
   - 법적 근거 (관련 조항 구체적 인용)
   - 세부 설명 (실무적 관점 포함)
   - 예외사항 또는 주의사항

2. 형식:
   - 중요 수치, 기한, 조항은 굵게 강조
   - 전문 용어는 풀어서 설명
   - 단계적 설명이 필요한 경우 번호 매기기
   - 관련 조항은 정확한 출처와 함께 인용

3. 내용:
   - 해당 법령의 특수성 반영
   - 다른 법령과의 관계 설명
   - 실무적 적용 방법 제시
   - 최신 개정사항 반영

4. 실용성:
   - 실제 사례 연계 (가능한 경우)
   - 실무자 관점의 해석 추가
   - 구체적인 적용 방법 설명
   - 관련 판례나 결정례 인용

5. 전문성:
   - 국제무역법적 맥락 고려
   - 산업 특성 반영
   - WTO 협정 등 국제규범과의 관계
   - 유사 사례나 비교법적 분석
"""
    model = get_model()
    result = generate_content_with_retry(model, prompt)
    return law_name, result.text if result else "답변을 생성할 수 없습니다."

# 헤드 에이전트 통합 답변 수정
def get_head_agent_response(responses, question, history):
    combined = "\n\n".join([f"=== {n} 관련 정보 ===\n{r}" for n, r in responses])
    prompt = f"""
당신은 중국산 인쇄제판용 평면모양 사진플레이트 덤핑 전문가입니다. 여러 자료의 정보를 통합하여 포괄적이고 정확한 답변을 제공합니다.

{combined}

이전 대화:
{history}

질문: {question}

# 응답 지침
1. 답변은 다음 구조로 작성하세요:
   - 핵심 답변 (2-3문장으로 질문의 핵심을 먼저 답변)
   - 상세 설명 (관련 법령, 규정, 판례 등을 인용하여 구체적 설명)
   - 관련 정보 (추가로 알아두면 좋은 정보나 연관된 내용)
   - 참고 사항 (주의사항이나 예외사항이 있다면 명시)

2. 형식 요구사항:
   - 각 섹션은 명확한 제목으로 구분
   - 중요한 수치나 날짜는 굵은 글씨로 강조
   - 법령 인용 시 출처를 명확히 표시
   - 목록화가 가능한 내용은 번호나 글머리 기호로 구분

3. 내용 요구사항:
   - 모든 주장에 대한 근거 제시
   - 실무적으로 중요한 정보 강조
   - 최신 개정사항이나 변경점 반영
   - 실제 사례나 예시 포함 (가능한 경우)

4. 전문성 요구사항:
   - 전문 용어는 풀어서 설명
   - 법적 해석이 필요한 경우 관련 법령 함께 제시
   - 산업 현장의 실무적 관점 반영
   - 국제무역법적 맥락 고려

5. 가독성 요구사항:
   - 단락을 적절히 구분하여 가독성 확보
   - 복잡한 내용은 단계적으로 설명
   - 필요시 표나 구분선 사용
   - 전체적인 문맥의 흐름 유지
"""
    model = get_model()
    result = generate_content_with_retry(model, prompt)
    return result.text if result else "답변을 생성할 수 없습니다. 잠시 후 다시 시도해주세요."

# 대화 기록 렌더링
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 모든 에이전트 병렬 실행
async def gather_agent_responses(question, history):
    tasks = []
    # 모든 카테고리의 모든 문서에 대해 태스크 생성
    for category in LAW_CATEGORIES.values():
        for law_name, pdf_path in category.items():
            tasks.append(get_law_agent_response_async(law_name, question, history))
    return await asyncio.gather(*tasks)

# 사용자 입력 및 응답 부분 수정
if user_input := st.chat_input("질문을 입력하세요", key="main_chat_input"):
    # 새로운 질문 추가
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 사용자 질문 표시
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 답변 생성
    with st.spinner("답변 생성 중..."):
        try:
            # 모든 문서를 한번에 로드
            if not st.session_state.law_data:
                st.session_state.law_data = load_law_data()
            
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
            
            # 비동기 처리
            answer = asyncio.run(process_user_input(user_input, history))
            
            if answer:
                # 답변을 채팅 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # 채팅 기록 업데이트
                with st.chat_message("assistant"):
                    st.markdown(answer)
            else:
                st.error("답변을 생성하는데 실패했습니다. 다시 시도해주세요.")
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.session_state.chat_history.pop()  # 실패한 질문 제거
