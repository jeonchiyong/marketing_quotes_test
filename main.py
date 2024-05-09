# 참고 블로그 url
# - streamlit 배포 : https://m.blog.naver.com/weisoon/223196954589
# - langchain : https://teddylee777.github.io/langchain/langchain-tutorial-03/ , https://rfriend.tistory.com/839
# - embedding, RAG : https://wikidocs.net/231600 ,https://wikidocs.net/233817, https://teddylee777.github.io/langchain/rag-tutorial/#%EB%8B%A8%EA%B3%84--retriever-%EC%83%9D%EC%84%B1


# python ver : 3.11.2 64bit
# vsc에서 local환경에 app 실행하려면 커맨드창에 {streamlit run main.py} 입력

# 환경변수(github에 올릴때는 주석 처리한다.)
# from dotenv import load_dotenv
# load_dotenv()


# 모든 경고 메시지를 무시하도록 설정
import warnings
warnings.filterwarnings('ignore')

# streamlit
import streamlit as st

# langchain
from langchain import hub
from langchain_openai import OpenAIEmbeddings

# document loaders
from langchain_community.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader

# text split
from langchain.text_splitter import RecursiveCharacterTextSplitter

# embedding
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings # 무료 Open Source 기반 임베딩


# vectorstores
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

# outputparser
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough


# langchain message, Template
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

# ChatOpenAI
from langchain.chat_models import ChatOpenAI

# RAG 생성
# 1. txt 로드
@st.cache_data
def get_data():
    loader = TextLoader("240503_KB캐피탈 상품정보.txt", encoding = 'utf-8')
    docs = loader.load()
    print(f"문서의 수: {len(docs)}")

    return docs

# get_data() 함수를 호출하여 docs 변수 정의
docs = get_data()


# 2. split documents
@st.cache_data(hash_funcs={list : lambda _: None}) # 어떤 리스트가 들어오더라도 해시 값을 None으로 설정하여 해당 인자를 캐싱에서 제외
def split_documents(docs):
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        # 작은 청크 크기를 설정합니다.
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = recursive_text_splitter.split_documents(docs)
    print(f"나누어진 문서의 수: {len(split_docs)}")
    print('\nsplit docs 살펴보기')
    for i in range(len(split_docs)) :
        print(split_docs[i],'\n')
    
    return split_docs


# 문서를 처리하여 split_docs 변수에 할당
split_docs = split_documents(docs)

# 3. 임베딩 객체 생성
@st.cache_resource
def load_embedding_model() :
    model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli', # 사전학습 임베딩 모델
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}, # 임베딩 정규화하여, 모든 벡터가 같은 범위의 값을 갖도록 함
        )
    return model

embeddings_model = load_embedding_model()

# 4. 벡터스토어 생성
vectorstore = FAISS.from_documents(
    documents=split_docs, embedding=embeddings_model,
    distance_strategy = DistanceStrategy.COSINE # 코사인유사도로 측정
)

# 5. Retriever 생성
retriever = vectorstore.as_retriever(search_kwargs={'k':1}) # 단일 문서 검색


st.image('kbc_logo.png')

# 마케팅 문구 생성 페이지 화면 구성
st.title('생성형 AI를 활용한 마케팅 문구 생성')
st.text('\n')
# selectbox형식 예시
# opt_prod = st.selectbox('STEP1. 문구를 작성하려는 대출 상품을 선택해주세요',
#     ['신용대출','크로스셀', '자동차 담보 대출', '중고차 대출', '신차 대출'])


opt_prod = st.radio(
    # r'''$\textsf{\large ✅ STEP1. 문구를 생성하려는 대출 상품을 선택해주세요}$''',['신용대출','크로스셀', '자동차 담보 대출', '중고차 대출', '신차 대출'], horizontal=True)
    r'''$\textsf{\large ✅ STEP1. 문구를 생성하려는 대출 상품을 선택해주세요}$''',['내일로 신용대출','내집으로 신용대출', '자동차 담보 대출 내차로', '소액론 신용대출', '플러스론', '탑업론'], horizontal=True)
st.text('\n')

opt_gender = st.radio(
    r'''$\textsf{\large ✅ STEP2. 문구를 생성하려는 타겟의 성별을 선택해주세요}$''',['남자','여자','무관'], horizontal=True)
st.text('\n')

opt_age = st.radio(
    r'''$\textsf{\large ✅ STEP3. 문구를 생성하려는 타겟의 연령대를 선택해주세요}$''',['20대','30대','40대', '50대 이상','무관'], horizontal=True)
st.text('\n')

opt_job = st.text_input(r'''$\textsf{\large ✅ STEP4. 문구를 생성하려는 타겟의 직업을 입력해주세요(ex.직장인, 대학생, 주부 등)}$''','')
st.text('\n')

# 빈값일 경우 처리
if opt_job == '' :
    opt_job = '상관없음'

opt_style = st.radio(
    r'''$\textsf{\large ✅ STEP5. 원하는 문구의 스타일을 선택해주세요}$''',
    [f'''대출이동서비스로 {opt_prod} 갈아타면 금리는 낮아지고 한도는 높아질 수 있어요.''',
     '첫 달 이자 최대 50만원의 기회! 마이너스 통장은 3만원을 드려요!',
     f'''[KB캐피탈] OOO님 {opt_prod} 갈아타면 우대금리 최고 연0%p 인하 가능해요!''',
     f'''축하합니다🎉🎉 {opt_prod} 금리 인하 대상자로 선정되셨습니다!''',
     '없음'], horizontal=False)
st.text('\n')

opt_etc = st.text_input(r'''$\textsf{\large ✅ STEP6. 위 내용 외에 문구에 추가하고 싶은 내용을 입력해주세요 (ex.봄의 분위기에 맞춰 작성해줘, '금리'라는 단어를 포함해줘)}$''','')
st.text('\n')

# llm temperature 설정 기능
st.write(r'''$\textsf{\normalsize ✅ STEP7. 생성형 AI의 자유도를 설정해주세요}$''')
opt_temp = st.slider(r'''$\textsf{\small\textbf {※ 자유도가 1에 가까울수록 생성형 AI가 다양한 문구를 생성합니다.}}$''', 0.0, 1.0, 0.5, step = 0.1, format="%.1f")
print('선택된 temperature : ',opt_temp)
st.text('\n')


# Retriever 검색 결과
query = f"""{opt_prod}에 대해 알려줘"""
docs = retriever.get_relevant_documents(query) # 검색결과 저장
# print(docs[0].page_content)

# LLM 모델 생성
chat_model = ChatOpenAI(model = 'gpt-3.5-turbo-0125', temperature = opt_temp)

# gpt prompt
# 한글
# gpt_role_k = f"""
# system_role
# - 당신은 금융 마케팅 분야에 대해 풍부한 경험과 전문지식을 보유하고 있는 전문가이며, 대한민국의 금융회사인 KB캐피탈의 금융 상품 판매 활성화를 위해 마케팅 문자메시지 작성해야합니다.
# - 마케팅 하려는 KB캐피탈 금융 상품은 '{opt_prod}'이며, 상품에 대한 정보는 다음 내용을 참고해주세요 : {docs}
# - 마케팅 타겟의 성별은 '{opt_gender}', 연령대는 '{opt_age}', 직업은 '{opt_job}'입니다.
# - 또한 마케팅 문구 생성 시 추가로 고려해야 할 사항은 '{opt_etc}'입니다.
# - 다음 문자메시지 예시를 참고해서 작성해주세요. '대출이동서비스로 신용대출 갈아타면 금리는 낮아지고 한도는 높아질 수 있어요.', '첫 달 이자 최대 50만원의 기회! 마이너스 통장은 3만원을 드려요!'

# human_role
# - 다양한 어휘와 자연스러운 표현을 사용하고, 마케팅 타겟의 특성과 상품 정보를 고려하여 5개의 마케팅 문구를 작성해주십시오
# - 각 문구는 50자 이내로 작성해주시고, 5개의 문구를 구분할 수 있도록 각 문구 앞에 1~5까지의 숫자를 기재해주세요.
# - 내용이 중복되지 않고, 반복적인 문장이 생기지 않도록 자연스러운 글을 작성해주십시오.
# - 마케팅 문구에 'KB캐피탈'이라는 단어를 반드시 포함해주십시오.
# """


# system 역할 부여
system_role = f"""
- You are an expert with extensive experience and expertise in the field of financial marketing, and you must write marketing text messages to promote sales of financial products for KB Capital, a financial company in Korea.
- The KB Capital's financial product you wish to market is '{opt_prod}', and for information on the product, please refer to the following: '{docs[0].page_content}'.
- The marketing target's gender is '{opt_gender}', age group is '{opt_age}', and occupation is '{opt_job}'.
"""

# opt_style '없음', opt_etc 빈값일 경우 system role에서 제외
if opt_etc != '':
    system_role = system_role+ f"""- Additionally, an additional consideration when creating marketing copy is '{opt_etc}'."""

if opt_style != '없음' :
    system_role = system_role+ f"""- Please refer to the following text sample messages when writing. '{opt_style}'"""


# human role 
human_role = f"""
- Please use a variety of vocabulary and natural expressions, and write five marketing phrases considering the characteristics of your marketing target and product information.
- Please write each phrase within 50 characters, and write a number from 1 to 5 in front of each phrase to distinguish between the 5 phrases.
- To distinguish between the five phrases, please write a number from 1 to 5 in front of each phrase.
- Please answer in Korean and write naturally to avoid duplication of content and repetitive sentences.
- Please be sure to include the word ‘KB캐피탈’ in your marketing text.
"""


messages = [ SystemMessage(content=system_role),
            HumanMessage(content=human_role)
]

print('prompt messages : \n', messages)
st.text('\n')

# 답변 메시지 노출
if st.button('마케팅 문구 생성하기 📋', type= 'primary') : #type 옵션으로 버튼 색상 빨강색으로 설정
    with st.spinner('문구 생성 중...🤖') :
        # chat_gpt 답변 생성
        completion = chat_model.invoke(messages)
        print(completion, '\n')

        # gpt가 답변한 내용을 변수에 저장
        assistant_message = completion.content
        print(assistant_message)
        st.success(f'''ChatGPT가 생성한 문구입니다. 다른 문구를 보고 싶으시면,
                   \n '마케팅 문구 생성하기'버튼을 다시 한번 눌러주세요!
                   \n \n \n P.S. 참고로 다음 상품 정보를 참고하여 문구를 작성했어요😁
                   \n  {docs[0].page_content}
                   ''')
        st.success(assistant_message)

