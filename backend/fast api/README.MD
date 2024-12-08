## Запуск 
---
```
python -m venv venv
pip install -r requirements.txt
sudo apt update && sudo apt install poppler-utils
```
---
```
mkdir pdfs
```
И перенести туда все pdf
---
```
mkdir doxcs
```
И перенести туда все docx
---
```
mkdir vector_index_with_metadata.faiss
```
И перенести туда файлы проиндексированных векторных представлений
---

```
mkdir output_images
cd fast_api
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

