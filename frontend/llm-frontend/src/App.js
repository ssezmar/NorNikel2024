import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null); 

  const sendQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8000/ask', { question });
      setAnswer(response.data.answer);
      setImages(response.data.images);
    } catch (error) {
      console.error('Ошибка при отправке вопроса:', error);
      setAnswer('Произошла ошибка. Попробуйте позже.');
    } finally {
      setLoading(false);
    }
  };
  const handleFileChange = (event) => {
    // setFile(event.target.files[0]); // Сохраняем выбранный файл
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>NonNikel Finder Чат</h1>
      </header>
      <main>
        <div className="chat-container">
          <div className="input-area">
            <textarea
              placeholder="Введите ваш вопрос..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
            <button onClick={sendQuestion} disabled={loading}>
              {loading ? 'Отправка...' : 'Отправить'}
            </button>
          </div>
          <div className="response-area">
            <h2>Ответ:</h2>
            <p>{answer}</p>
            <h3>Страницы PDF Фалов на основе которых был сформирован ответ:</h3>
            <div className="images-grid">
              {images.map((img, index) => (
                <div key={index} className="image-item">
                  <a href={`http://127.0.0.1:8000${img.url}`} target="_blank" rel="noopener noreferrer">
                    <img src={`http://127.0.0.1:8000${img.url}`} alt={`Страница ${img.page}`} />
                  </a>
                  <p>{img.filename} - Страница {img.page}</p>
                </div>
              ))}
            </div>
          </div>
          {/* Загрузка файла */}
          <div className="file-upload">
            <h2>Добавление файлов в базу знаний:</h2>
            <input
              type="file"
              onChange={handleFileChange}
              accept=".txt, .pdf"
            />
            {file && <p>Выбран файл: {file.name}</p>}{/* Отображаем имя выбранного файла */}
          </div>
        </div>
        <button>
              Добавить
        </button>
      </main>
    </div>
  );
}

export default App;

