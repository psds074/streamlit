import streamlit as st

st.title("Streamlit 테스트 앱")

# 사용자 입력
user_input = st.text_input("이름을 입력하세요:")

if user_input:
    st.write(f"안녕하세요, {user_input}님!")

# 버튼 클릭 이벤트
if st.button("이미지 보기"):
    st.image("https://static.streamlit.io/examples/cat.jpg", caption="귀여운 고양이")
