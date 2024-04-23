# 참고 블로그 url : https://m.blog.naver.com/weisoon/223196954589
# python ver : 3.11.2 64bit

# print('hello')

# 환경변수(github에 올릴때는 주석 처리한다.)
# from dotenv import load_dotenv
# load_dotenv()

# streamlit
import streamlit as st
# ChatOpenAI
from langchain.chat_models import ChatOpenAI


# 인공지능에게 요청
chat_model = ChatOpenAI(model = 'gpt-3.5-turbo-1106', temperature = 0.5)

st.image('kbc_logo.png')

# 마케팅 문구 생성 페이지 화면 구성
st.title('생성형 AI를 활용한 마케팅 문구 작성')
# st.text_input('메시지를 입력해주세요','anonymous')

# st.subheader('STEP1. 문구를 작성하려는 대출 상품을 선택해주세요')
opt_prod = st.selectbox('STEP1. 문구를 작성하려는 대출 상품을 선택해주세요',
    ['신용대출','크로스셀', '자동차 담보 대출'])

opt_gender = st.selectbox(
    'STEP2. 문구를 전송하려는 타겟의 성별을 선택해주세요',['남자','여자','무관'])

opt_age = st.selectbox(
    'STEP3. 문구를 전송하려는 타겟의 연령대를 선택해주세요',['20대','30대','40대', '50대 이상'])

opt_job = st.text_input('STEP4. 문구를 전송하려는 타겟의 직업을 입력해주세요(ex.직장인, 대학생, 주부 등)','')

opt_etc = st.text_input('STEP5. 위 내용 외에 문구에 추가하고 싶은 내용을 입력해주세요(ex.봄의 분위기에 맞춰서 작성해줘)','')

# 'You selected: ', option

# gpt prompt

# gpt role 
gpt_role = f"""
- 당신은 금융 마케팅 분야에 대해 풍부한 경험과 전문지식을 보유하고 있는 전문가이며, 대한민국의 금융회사인 KB캐피탈의 금융 상품 판매 활성화를 위해 마케팅 캠페인에 참여하고 있습니다. 
- 마케팅 하려는 KB캐피탈 금융 상품은 '{opt_prod}'입니다.
- 마케팅 타겟의 성별은 '{opt_gender}', 연령대는 '{opt_age}', 직업은 '{opt_job}'입니다.
- 또한 마케팅 문구 생성 시 추가로 고려해야 할 사항은 '{opt_etc}'입니다.
- 다양한 어휘와 표현을 사용하여, 5개의 마케팅 문구를 작성해주시고, 각 문구는 40자 이내로 작성해주십시오.
- 5개의 문구를 구분할 수 있도록 각 문구 앞에 1~5까지의 숫자를 기재해주세요.
- 내용이 중복되지 않고, 반복적인 문장이 생기지 않도록 자연스러운 글을 작성해주십시오.
"""

# messages = [
#             {'role' : 'system', 'content' : gpt_role},
#             {'role' : 'user', 'content' : gpt_role}
# ]
messages = [
            ('system' , gpt_role),
            ('human' ,  gpt_role)
]
print(messages)

# 답변 메시지 노출
if st.button('마케팅 문구 작성하기') :
    with st.spinner('문구 작성 중...') :
        # chat_gpt 답변 생성
        completion = chat_model.invoke(messages)
        # completion = chat_model.chat.completions.create(
        #     model = 'gpt-3.5-turbo-1106',
        #     messages = messages)
        # print(completion)

        # gpt가 답변한 내용을 변수에 저장
        assistant_message = completion.content
        print(assistant_message)
        st.success('''ChatGPT가 작성한 문구입니다. 다른 문구를 보고 싶으시면,
                   \n '마케팅 문구 작성하기'버튼을 다시 한번 눌러주세요!''')
        st.success(assistant_message)

