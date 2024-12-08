from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models.gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from pdf2image import convert_from_path
import os
from fastapi.responses import FileResponse
from pathlib import Path


# Функция извлечения страницы PDF как изображения
def extract_page_as_image(pdf_path, page_number):
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    return images[0]  # Возвращаем единственное изображение, соответствующее странице

# Функция сохранения изображения
def save_image(image, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    image.save(output_path, format='PNG')
    return output_path

# Функция извлечения и сохранения страницы как изображения
def extract_and_save_page_image(pdf_path, page_number, output_folder):
    try:
        page_image = extract_page_as_image(pdf_path, page_number)
        filename = f"page_{page_number}.png"
        output_path = save_image(page_image, output_folder, filename)
        return output_path
    except Exception as e:
        return None

def delete_all_files_in_directory(directory_path):
    try:
        # Проверяем, существует ли директория
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            # Проходим по всем файлам в директории
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                # Если это файл (не директория), удаляем его
                if os.path.isfile(file_path):
                    os.remove(file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Ошибка при удалении файлов: {e}")
        return False

# FastAPI сервер
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Убедитесь, что фронтенд доступен с этого адреса
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Обслуживание статических файлов из директории "output_images"
app.add_route("/static/{path:path}", StaticFiles(directory=os.path.abspath("output_images")), name="static")


# Модель данных для запроса
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    # Ваш код обработки вопроса
    question = request.question

    # Загрузка FAISS векторных представлений
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    index_file = "vector_index_with_metadata.faiss"
    vector_store = FAISS.load_local(index_file, embedding, allow_dangerous_deserialization=True)

    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Настройка модели GigaChat
    auth = 'MTA4ZDM1OTQtODA5Yi00NDMzLTk3Y2ItZjNjNTZiODRkOGY0Ojk5YjI5MzUwLThkYzYtNGJhMy04ZGQ5LTk5MTYwYmNhYmRmMg=='
    llm = GigaChat(credentials=auth, model='GigaChat-Max', verify_ssl_certs=False, profanity_check=False)

    prompt = ChatPromptTemplate.from_template('''Дополни мой контекст возможно каким-то своим и выдай ответ, если ничего не нашели пиши только "Нашел для вас релевантные страницы слайдов"
    Откуда можно попробовать найти ответ: {context}
    Вопрос: {query}
    Ответ:'''
    )

    # Обработка вопроса с функцией create_retrieval_chain
    try:
        document_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=embedding_retriever)
        resp1 = document_chain.run(query=question)
    except TypeError as e:
        return {"error": f"Ошибка вызова create_retrieval_chain: {str(e)}"}

    # Получение похожих документов
    similar_docs = vector_store.similarity_search(question, k=5)

    images = []
    delete_all_files_in_directory("output_images")
    for doc in similar_docs[:5]:
        image_path = extract_and_save_page_image(f"pdfs/{doc.metadata['file_name'][:-3]}pdf", doc.metadata['page'], 'output_images')
        if image_path:
            images.append({
                'url': f"/images/{os.path.basename(image_path)}",  # Здесь указываем путь к статическому файлу
                'filename': doc.metadata['file_name'],
                'page': doc.metadata['page']
            })


    # Ответ
    return {
        'answer': resp1,
        'images': images
    }

UPLOAD_DIR = Path("output_images")

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """
    Возвращает изображение по имени.
    """
    image_path = UPLOAD_DIR / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}

