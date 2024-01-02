import os
from typing import Any, List, Dict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

# from langchain.vectorstores import Pinecone
# import pinecone
# from langchain_community.llms.openai import OpenAI
from langchain_community.vectorstores.faiss import FAISS

# pinecone.init(api_key="4939c500-9342-429a-bbec-7e490ce8fb4c",
#               environment="gcp-starter")

INDEX_NAME="test-index"

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    # promt chỉ đơn giản là nội dung nhập văn bản mà chúng tôi cung cấp cho LM và xử lý nó rồi trả về cho chúng tôi một đầu ra.
    # promt template Đó là mẫu lời nhắc và nó chỉ đơn giản là một lớp bao bọc xung quanh lời nhắc. nó bổ sung thêm chức năng nhắc nhở nhận đầu vào.
    new_vectorstore = FAISS.load_local("/home/boss/Documents/full-stack/vector/faiss_index_react", embeddings)
    #6 dùng chat model của langchain . giao diện lấy đầu vào là danh sách các tin nhắn và trả về một tin nhắn.

    chat = ChatOpenAI(verbose=True, temperature=0)
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat, chain_type="stuff",
    #     retriever=new_vectorstore.as_retriever(),
    #     return_source_documents=True
    # )

    #7 chain Chúng cho phép chúng ta kết hợp nhiều thành phần lại với nhau và tạo ra một ứng dụng mạch lạc duy nhất => tạo ra thay đổi bằng cách kết hợp các chain lại với nhau
    # step1 : load pdf => step2 cohere embedding => stepn: Question answer
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=new_vectorstore.as_retriever(),
        return_source_documents=True
    )
    # 5 Bây giờ, nếu muốn đặt câu hỏi, lấy câu hỏi đó làm truy vấn, nhúng nó vào vào một vectơ,
    # đặt nó vào không gian vectơ nơi chứa tất cả các phần nhúng đã lưu,
    # các đoạn đã tồn tại.
    # Và bây giờ chúng ta có thể tính toán các vectơ gần nhất đã lưu với vectơ truy vấn mà chúng ta đã nhúng.
    # đó là những gì đại diện cho những vectơ đó những phần có liên quan mà chúng ta đã nói đến.
    # bây giờ chúng ta có thể chỉ cần gửi bối cảnh này của các phần có liên quan cùng với truy vấn của chúng ta trong prompt.
    # return qa({ "query": query })
    return qa({ "question": query, "chat_history": chat_history })



if __name__== '__main__':
    print(run_llm(query="what is three types of Innovation Capital"))
