📚 Semantic Book Recommender

Welcome to the **Semantic Book Recommender**, an intelligent system that helps users discover books that match their interests and emotional preferences. Built using cutting-edge language models, semantic search, and emotion analysis, this project demonstrates the power of **LLMs**, **vector embeddings**, and a polished **Gradio UI**.

> ✨ **Smart Book Recommender**: Developed a book recommender system using OpenAI embeddings, Python, Gradio, and ChromaDB. Implemented semantic search with a vector database, zero-shot classification for genres, and sentiment analysis for personalized recommendations.


---

🔍 What It Does

This web-based dashboard allows users to:

- 🧠 Enter a short book description or idea  
- 📖 Receive semantically similar book recommendations  
- 🎭 Filter results by **genre/category** and **emotional tone** (e.g., Joy, Sadness, Fear)  
- 📊 Sort by **Relevance** or **Emotion Intensity**  
- 🔁 View recent search history  

---

🚀 Key Features

- **LLM-based Embeddings**: Uses OpenAI embeddings via `langchain_openai` to convert book descriptions into dense vectors  
- **Semantic Vector Search**: Retrieves similar books using `Chroma` as the vector store  
- **Zero-Shot Classification**: Categorizes books into "Fiction" or "Nonfiction" with HuggingFace's `bart-large-mnli`  
- **Emotion Detection**: Extracts emotional signals like joy, fear, or surprise using fine-tuned models  
- **Interactive UI**: Built with **Gradio**, offering an engaging, fast, and responsive user experience
- 
---

🧠 Topics Explored

This project covers:

- Data cleaning and exploration in Pandas  
- Text vectorization with LLMs  
- Semantic search with LangChain + Chroma  
- Zero-shot learning via HuggingFace transformers  
- Emotion-based filtering using fine-tuned LLMs  
- UI design with Gradio Blocks  
- Theme toggles, input validation, and result formatting  

---

🛠 Tech Stack

- **Frontend**: Gradio Blocks  
- **Backend**: Python  
- **LLMs**: OpenAI Embeddings, BART-MNLI, DistilRoBERTa (emotion model)  
- **Libraries**:
  - `langchain_openai`  
  - `langchain_chroma`  
  - `transformers`  
  - `pandas`, `numpy`  
  - `gradio`  

---

📂 Directory Structure

- app.py                    # Main script with dashboard logic
- requirements.txt          # All required libraries
- books_with_emotions.csv   # Book dataset with metadata and emotion scores
- tagged_description.txt    # Descriptions used for vector similarity
- search_history.json       # Local cache of recent searches
- book_icon.png             # Icon used in Gradio dashboard (optional)
- README.txt                # You are here!

---

⚙️ Setup Instructions

1. Clone this repo:
   git clone https://github.com/yourusername/semantic-book-recommender.git

2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Set your OpenAI key:
   export OPENAI_API_KEY=your-key-here

5. Launch the app locally:
   python app.py

---

🌐 Deployment Notes

The app can be deployed on platforms like Render or Hugging Face Spaces. Due to high memory usage, hosting the full dataset may require:
- Switching to a paid plan
- Reducing vector database size
- Chunking documents to conserve memory

---

📷 Screenshots / Demo

SOON

---

🙋 Author

Manan Gulati  
Email: mgulati3@asu.edu  
LinkedIn: https://www.linkedin.com/in/manangulati/

---

⭐ Acknowledgments

Inspired by a tutorial series exploring:
- LLMs and embeddings
- Vector databases with LangChain
- Zero-shot classification
- Gradio-based web dashboards

---

📘 License

MIT License 
