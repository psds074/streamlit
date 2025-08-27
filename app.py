import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
from langchain.prompts import PromptTemplate

# 환경변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 페이지 설정
st.set_page_config(page_title="PDF RAG 챗봇", layout="wide")
st.title("📄 PDF 기반 RAG 챗봇")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # PDF 문서 로딩
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # [TODO] 청크 사이즈와 오버랩을 바꿔가며 RAG 챗봇의 성능 변화를 확인해 보세요.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # 벡터 저장소
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Retriever 구성
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # LLM 및 시스템 메시지 포함한 프롬프트 템플릿
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key)

    # [TODO] 시스템 프롬프트를 바꿔가며 RAG 챗봇의 성능 변화를 확인해 보세요.
    prompt_template = """
        당신은 업로드된 PDF 문서를 기반으로 질문에 정중하고 간결하게 응답하는 AI입니다.
        문서에서 직접적인 답을 찾을 수 없을 경우, 가능한 한 유사한 정보를 제공하고 추측이 아닌 정보 기반으로 답변해야 합니다.

        질문: {question}
        이전 대화 내용: {chat_history}

        정중하고 정확하게 대답하세요.
    """

    prompt = PromptTemplate(
        input_variables=["question", "chat_history"],
        template=prompt_template.strip()
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt,
        return_source_documents=True
    )

    # 채팅 UI
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    user_input = st.chat_input("질문을 입력하세요")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        # 이전 질문 목록만 넘김 (질문만 추출)
        chat_history = [(q, a) for q, a in st.session_state.chat_history]

        result = qa_chain(
            {"question": user_input, "chat_history": chat_history}
        )
        answer = result["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append((user_input, answer))

else:
    st.info("먼저 PDF 파일을 업로드하세요.")
