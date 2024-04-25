# 참고 블로그 url
# - streamlit 배포 : https://m.blog.naver.com/weisoon/223196954589
# - langchain : https://teddylee777.github.io/langchain/langchain-tutorial-03/ , https://rfriend.tistory.com/839

# python ver : 3.11.2 64bit
# vsc에서 local환경에 app 실행하려면 커맨드창에 {streamlit run main.py} 입력

# 환경변수(github에 올릴때는 주석 처리한다.)
from dotenv import load_dotenv
load_dotenv()

# streamlit
import streamlit as st

# langchain message
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# ChatOpenAI
from langchain.chat_models import ChatOpenAI

st.image('kbc_logo.png')

# 마케팅 문구 생성 페이지 화면 구성
st.title('생성형 AI를 활용한 마케팅 문구 생성')
st.text('\n')
# selectbox형식 예시
# opt_prod = st.selectbox('STEP1. 문구를 작성하려는 대출 상품을 선택해주세요',
#     ['신용대출','크로스셀', '자동차 담보 대출', '중고차 대출', '신차 대출'])


opt_prod = st.radio(
    r'''$\textsf{\large ✅ STEP1. 문구를 생성하려는 대출 상품을 선택해주세요}$''',['신용대출','크로스셀', '자동차 담보 대출', '중고차 대출', '신차 대출'], horizontal=True)
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
     f'''축하합니다😊 {opt_prod} 금리 인하 대상자로 선정되셨습니다!''',
     '없음'], horizontal=False)
st.text('\n')

opt_etc = st.text_input(r'''$\textsf{\large ✅ STEP6. 위 내용 외에 문구에 추가하고 싶은 내용을 입력해주세요 (ex.봄의 분위기에 맞춰 작성해줘, '금리'라는 단어를 포함해줘)}$''','')
st.text('\n')

# llm temperature 설정 기능
st.write(r'''$\textsf{\normalsize ✅ STEP7. 생성형 AI의 자유도를 설정해주세요}$''')
opt_temp = st.slider(r'''$\textsf{\small\textbf {※ 자유도가 1에 가까울수록 생성형 AI가 다양한 문구를 생성합니다.}}$''', 0.0, 1.0, 0.5, step = 0.1, format="%.1f")
print(opt_temp)
st.text('\n')


# LLM 모델 생성
chat_model = ChatOpenAI(model = 'gpt-3.5-turbo-0125', temperature = opt_temp)

# gpt prompt
# 한글
# gpt_role_k = f"""
# - 당신은 금융 마케팅 분야에 대해 풍부한 경험과 전문지식을 보유하고 있는 전문가이며, 대한민국의 금융회사인 KB캐피탈의 금융 상품 판매 활성화를 위해 마케팅 문자메시지 작성해야합니다.
# - 마케팅 하려는 KB캐피탈 금융 상품은 '{opt_prod}'입니다.
# - 마케팅 타겟의 성별은 '{opt_gender}', 연령대는 '{opt_age}', 직업은 '{opt_job}'입니다.
# - 또한 마케팅 문구 생성 시 추가로 고려해야 할 사항은 '{opt_etc}'입니다.
# - 다음 문자메시지 예시를 참고해서 작성해주세요. '대출이동서비스로 신용대출 갈아타면 금리는 낮아지고 한도는 높아질 수 있어요.', '첫 달 이자 최대 50만원의 기회! 마이너스 통장은 3만원을 드려요!'

# - 다양한 어휘와 자연스러운 표현을 사용하여, 5개의 마케팅 문구를 작성해주시고, 각 문구는 40자 이내로 작성해주십시오.
# - 5개의 문구를 구분할 수 있도록 각 문구 앞에 1~5까지의 숫자를 기재해주세요.
# - 내용이 중복되지 않고, 반복적인 문장이 생기지 않도록 자연스러운 글을 작성해주십시오.
# - 마케팅 문구에 'KB캐피탈'이라는 단어를 반드시 포함해주십시오.
# """


# system 역할 부여
system_role = f"""
- You are an expert with extensive experience and expertise in the field of financial marketing, and you must write marketing text messages to promote sales of financial products for KB Capital, a financial company in Korea.
- The KB Capital financial product you wish to market is {opt_prod}.
- The marketing target's gender is '{opt_gender}', age group is '{opt_age}', and occupation is '{opt_job}'.
"""

# opt_style '없음', opt_etc 빈값일 경우 system role에서 제외
if opt_etc != '':
    system_role = system_role+ f"""- Additionally, an additional consideration when creating marketing copy is '{opt_etc}'."""

if opt_style != '없음' :
    system_role = system_role+ f"""- Please refer to the following text sample messages when writing. '{opt_style}'"""


# human role 
human_role = f"""
- Please write 5 marketing phrases using a variety of vocabulary and natural expressions, and each phrase should be written within 60 characters in Korean.
- To distinguish between the five phrases, please write a number from 1 to 5 in front of each phrase.
- Please answer in Korean and write naturally to avoid duplication of content and repetitive sentences.
- Please be sure to include the word ‘KB캐피탈’ in your marketing text.
"""


messages = [ SystemMessage(content=system_role),
            HumanMessage(content=human_role)
]

print(messages)
st.text('\n')
st.text('\n')

# 답변 메시지 노출
if st.button('마케팅 문구 생성하기 📋', type= 'primary') :
    with st.spinner('문구 생성 중...🤖') :
        # chat_gpt 답변 생성
        completion = chat_model.invoke(messages)
        print(completion, '\n')

        # gpt가 답변한 내용을 변수에 저장
        assistant_message = completion.content
        print(assistant_message)
        st.success('''ChatGPT가 생성한 문구입니다. 다른 문구를 보고 싶으시면,
                   \n '마케팅 문구 생성하기'버튼을 다시 한번 눌러주세요!''')
        st.success(assistant_message)

