import os

# load text
# from langchain.document_loaders import TextLoader
# load pdf
from langchain.document_loaders import PyPDFLoader

#text splitter
from langchain.text_splitter import CharacterTextSplitter
#embedding
from langchain.embeddings.openai import OpenAIEmbeddings

#pincone
# from langchain.vectorstores.pinecone import Pinecone
# save pincone

#QA
# from langchain import VectorDBQA, OpenAI
from langchain import OpenAI

from langchain.chains import RetrievalQA
# import pinecone

#save pdf

from langchain.vectorstores import FAISS

# pinecone.init(api_key="4939c500-9342-429a-bbec-7e490ce8fb4c",
#               environment="gcp-starter")

def ingrest_docs() -> None:
    #1. tải tài liệu pdf sử dụng langchain document loadder
    # langchain.document_loaders
    # có thể đổi nhiệu loại file

    pdf_path = "/home/boss/Documents/full-stack/vector/mediumblogs/practice-test-A.pdf"
    loaderPDF = PyPDFLoader(file_path=pdf_path)
    documentPDF = loaderPDF.load()

    #2. text splitter: bộ tách văn bản cho phép chúng ta lấy văn bản lớn và chia nó thành nhiều phần
    # trong GPT 3.5 có giới hạn 4k mã thông báo
    # chia nó thành hàng nghìn hoặc hàng triệu khối.
    text_splitterPDF = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    # kích thước bằng một nghìn token và các khối chồng lên nhau.Tham số chunk_overlap chỉ định mức độ chồng chéo giữa các khối khi chúng ta chia văn bản thành những phần nhỏ hơn.
    # Sự chồng chéo này có thể cực kỳ hữu ích để đảm bảo rằng công nghệ không bị phân chia theo cách làm xáo trộn bối cảnh hoặc ý nghĩa.
    textsPDF = text_splitterPDF.split_documents(documentPDF)

    #3: embeddings: Bây giờ chúng ta có thể lấy tất cả các khối này và nhúng chúng bằng mô hình nhúng và biến chúng thành một vectơ rằng mỗi vectơ đại diện cho đoạn đó.
    # mỗi vectơ sẽ là một danh sách các số đại diện cho đoạn đã cho đó nhúng.
    # chỉ đơn giản là lưu giữ chúng và giúp chúng ta dễ dàng sử dụng chúng sau này.
    # nhận đầu vào dưới dạng văn bản và đầu ra vectơ trong không gian vectơ nhúng.
    # ADA 002 là một cái rất tốt vì việc nhúng có ý nghĩa rất lớn về giá cả.
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    #4: Bây giờ chúng ta có thể lấy những phần nhúng đó và lưu chúng vào cơ sở dữ liệu vectơ như Pinecone chẳng hạn.
    docsearchPDF = FAISS.from_documents(textsPDF, embeddings)
    docsearchPDF.save_local("faiss_index_react")
    # new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    #5 Bây giờ, nếu muốn đặt câu hỏi, lấy câu hỏi đó làm truy vấn, nhúng nó vào vào một vectơ,
    # đặt nó vào không gian vectơ nơi chứa tất cả các phần nhúng đã lưu,
    # các đoạn đã tồn tại.
    # Và bây giờ chúng ta có thể tính toán các vectơ gần nhất đã lưu với vectơ truy vấn mà chúng ta đã nhúng.
    # đó là những gì đại diện cho những vectơ đó những phần có liên quan mà chúng ta đã nói đến.
    # bây giờ chúng ta có thể chỉ cần gửi bối cảnh này của các phần có liên quan cùng với truy vấn của chúng ta trong prompt.
    # promt chỉ đơn giản là nội dung nhập văn bản mà chúng tôi cung cấp cho LM và xử lý nó rồi trả về cho chúng tôi một đầu ra.
    # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    # res = qa.run("what is three types of Innovation Capital ")
    return

if __name__== '__main__':
    print("vector")
    ingrest_docs()

    # # load text
    # # loader = TextLoader("/home/boss/Documents/full-stack/vector/mediumblogs/mediumblog1.txt")
    # # document = loader.load()
    #
    # #text slitter
    # # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # # texts = text_splitter.split_documents(document)
    # # print(len(texts))
    #
    # # text slitter PDF
    #
    #
    # #embedding
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    #
    # #save pincone
    # # docsearch = Pinecone.from_documents(texts, embeddings, index_name="test-index")
    # docsearchs = Pinecone.from_existing_index(embeddings, index_name="test-index")
    #
    #
    # # qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
    # # query = "find the key in redis"
    # # result = qa({ "query": query })
    # # print(result)



