import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from pdf2image import convert_from_path

class PDFQueryProcessor:
    def __init__(self, index_file, pdf_folder, output_folder, auth_token):
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        
        # Настройка FAISS и модели GigaChat
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.vector_store = FAISS.load_local(index_file, embedding, allow_dangerous_deserialization=True)
        embedding_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        self.llm = GigaChat(
            credentials=auth_token,
            model='GigaChat-Pro',
            verify_ssl_certs=False,
            profanity_check=False
        )
        
        prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
        Не надо задавать обратных вопросов!\
        Если в контексте недостаточно информации то попробуй аккуратно соврать, чтобы не было понятно что есть ли на самом деле какая-то информация или нет\
        Если в контексте ты не найдешь нужную информацию,\
        то приведи какой-то ответ так как-будто он там есть и ОБЯЗАТЕЛЬНО скажи что\ 
        "ниже представлены страницы PDF файлов вы можете попытаться найти нужную информацию там""\
        Контекст: {context}
        Вопрос: {input}
        Ответ:'''
        )
        
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        self.retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)

    def extract_page_as_image(self, pdf_path, page_number):
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        return images[0]  # Возвращаем единственное изображение, соответствующее странице

    def save_image(self, image, filename):
        os.makedirs(self.output_folder, exist_ok=True)
        output_path = os.path.join(self.output_folder, filename)
        image.save(output_path, format='PNG')
        return output_path

    def extract_and_save_page_image(self, pdf_path, page_number):
        try:
            page_image = self.extract_page_as_image(pdf_path, page_number)
            filename = f"page_{page_number}.png"
            output_path = self.save_image(page_image, filename)
            return output_path
        except Exception as e:
            print(f"Ошибка при обработке страницы {page_number} из {pdf_path}: {e}")
            return None

    def query(self, question):
        try:
            response = self.retrieval_chain.invoke({'input': question})
            similar_docs = self.vector_store.similarity_search(question, k=5)
            
            results = []
            for doc in similar_docs:
                pdf_path = os.path.join(self.pdf_folder, f"{doc.metadata['file_name'][:-3]}pdf")
                page_number = doc.metadata['page']
                image_path = self.extract_and_save_page_image(pdf_path, page_number)
                
                results.append({
                    "text_chunk": doc.page_content[:400],
                    "metadata": doc.metadata,
                    "image_path": image_path
                })
            
            return {
                "answer": response['answer'],
                "contextual_documents": results
            }
        except Exception as e:
            return {"error": str(e)}

# Пример использования
if __name__ == "__main__":
    pdf_folder = "pdfs"
    output_folder = "output_images"
    index_file = "vector_index_with_metadata.faiss"
    auth_token = "MTA4ZDM1OTQtODA5Yi00NDMzLTk3Y2ItZjNjNTZiODRkOGY0Ojk5YjI5MzUwLThkYzYtNGJhMy04ZGQ5LTk5MTYwYmNhYmRmMg=="
    
    processor = PDFQueryProcessor(index_file, pdf_folder, output_folder, auth_token)
    question = "Мультипликатор свободного денежного потока в отношении 23 к 24 году"
    result = processor.query(question)
    
    print("Ответ модели:", result["answer"])
    for doc in result["contextual_documents"]:
        print(f"Чанк текста: {doc['text_chunk']}...")
        print(f"Метаданные: {doc['metadata']}")
        print(f"Изображение: {doc['image_path']}")
