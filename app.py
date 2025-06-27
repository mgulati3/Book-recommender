import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import json

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# Load environment variables
load_dotenv()

# Initialize cache for search results
search_cache = {}
MAX_CACHE_SIZE = 50
CACHE_EXPIRY = 3600  # 1 hour in seconds

# Load and prepare books data
try:
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&file=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"]
    )

    print(f"Loaded {len(books)} books successfully")
except Exception as e:
    print(f"Error loading books data: {e}")
    # Create a minimal dataset if loading fails
    books = pd.DataFrame(columns=["isbn13", "title", "authors", "description", "thumbnail", "large_thumbnail",
                                  "simple_categories", "joy", "surprise", "anger", "fear", "sadness"])

# Load document embeddings
try:
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
    print("Loaded document embeddings successfully")
except Exception as e:
    print(f"Error loading document embeddings: {e}")
    # Will need to handle this in the recommendation function


# Load search history from file if it exists
def load_search_history():
    try:
        if os.path.exists("search_history.json"):
            with open("search_history.json", "r") as f:
                return json.load(f)
        return []
    except Exception:
        return []


# Save search history to file
def save_search_history(history):
    try:
        with open("search_history.json", "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving search history: {e}")


search_history = load_search_history()


def add_to_search_history(query, category, tone, sort_by):
    """Add a search to history and save it"""
    global search_history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    search_history.append({
        "timestamp": timestamp,
        "query": query,
        "category": category,
        "tone": tone,
        "sort_by": sort_by
    })
    # Keep only the most recent 50 searches
    search_history = search_history[-50:]
    save_search_history(search_history)


def format_authors(authors_string):
    """Format authors string in a more readable way"""
    if not authors_string or pd.isna(authors_string):
        return "Unknown Author"

    authors_split = authors_string.split(";")
    if len(authors_split) == 1:
        return authors_split[0]
    elif len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    else:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"


def get_cache_key(query, category, tone, sort_by):
    """Generate a unique key for caching search results"""
    return f"{query}|{category}|{tone}|{sort_by}"


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        sort_by: str = "Relevance",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    """Retrieve book recommendations based on query and filters"""
    # Check cache first
    cache_key = get_cache_key(query, category, tone, sort_by)
    current_time = time.time()

    # Return from cache if available and not expired
    if cache_key in search_cache:
        cache_entry = search_cache[cache_key]
        if current_time - cache_entry["timestamp"] < CACHE_EXPIRY:
            print("Returning cached results")
            return cache_entry["data"]

    try:
        # Get semantic recommendations
        recs = db_books.similarity_search(query, k=initial_top_k)
        books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
        book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k).copy()

        # Apply filters
        if category and category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
        else:
            book_recs = book_recs.head(final_top_k)

        # Apply emotional tone filter
        if tone and tone != "All":
            tone_map = {
                "Happy": "joy",
                "Surprising": "surprise",
                "Angry": "anger",
                "Suspenseful": "fear",
                "Sad": "sadness"
            }
            tone_col = tone_map.get(tone)
            if tone_col:
                book_recs = book_recs.sort_values(by=tone_col, ascending=False)

        # Apply sorting based on emotions only
        if sort_by == "Emotion Intensity" and tone != "All":
            tone_map = {
                "Happy": "joy",
                "Surprising": "surprise",
                "Angry": "anger",
                "Suspenseful": "fear",
                "Sad": "sadness"
            }
            tone_col = tone_map.get(tone)
            if tone_col:
                book_recs = book_recs.sort_values(by=tone_col, ascending=False)

        # Cache results
        if len(search_cache) >= MAX_CACHE_SIZE:
            # Remove oldest cache entry
            oldest_key = min(search_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del search_cache[oldest_key]

        search_cache[cache_key] = {
            "timestamp": current_time,
            "data": book_recs
        }

        return book_recs

    except Exception as e:
        print(f"Error retrieving recommendations: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def recommend_books(query, category, tone, sort_by, progress=gr.Progress()):
    """Main function to recommend books based on user input"""
    if not query.strip():
        return [], "⚠️ Please enter a search query to get started"

    progress(0, desc="🔍 Starting search...")
    time.sleep(0.5)  # Small delay for better UX

    # Add to search history
    add_to_search_history(query, category, tone, sort_by)

    progress(0.3, desc="🧠 Retrieving semantic matches...")
    time.sleep(0.3)
    recommendations = retrieve_semantic_recommendations(
        query, category, tone, sort_by
    )

    progress(0.6, desc="✨ Formatting results...")
    time.sleep(0.3)
    results = []

    if recommendations.empty:
        return [], "📚 No books found matching your criteria. Try adjusting your filters or search terms."

    for _, row in recommendations.iterrows():
        # Get book description, handling different column names
        description = row.get("description_x", row.get("description", ""))
        if pd.isna(description):
            description = "No description available."

        # Truncate description for display
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Format authors
        authors_str = format_authors(row["authors"])

        # Format emotion data with emojis
        emotion_cols = ["joy", "surprise", "anger", "fear", "sadness"]
        emotion_emojis = {"joy": "😊", "surprise": "😲", "anger": "😠", "fear": "😨", "sadness": "😢"}
        emotion_info = []
        for emo in emotion_cols:
            if pd.notnull(row.get(emo, None)) and row.get(emo, 0) > 0:
                percentage = round(row[emo] * 100)
                emoji = emotion_emojis.get(emo, "")
                emotion_info.append(f"{emoji} {emo.capitalize()}: {percentage}%")

        emotion_text = " | ".join(emotion_info) if emotion_info else "No emotion data available"

        # Create modern caption with better formatting
        caption = (
            f"## **{row['title']}**\n"
            f"✍️ *{authors_str}*\n\n"
            f"{truncated_description}\n\n"
            f"**📊 Emotional Profile:**\n{emotion_text}"
        )

        # Add to results
        results.append((row["large_thumbnail"], caption))

    progress(1.0, desc="🎉 Done!")
    return results, f"🎯 Found {len(results)} books matching your criteria"


def clear_search():
    """Clear search inputs"""
    return "", "All", "All", "Relevance"


# Prepare UI elements
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "😊 Happy", "😲 Surprising", "😠 Angry", "😨 Suspenseful", "😢 Sad"]
sorting_options = ["Relevance", "Emotion Intensity"]

# Custom CSS for modern UI
modern_css = """
/* Modern gradient background */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Glass morphism effect for main container */
.main-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    padding: 20px;
    margin: 20px;
}

/* Modern card styling */
.search-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(5px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 20px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.search-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
}

/* Modern button styling */
.modern-btn {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    border: none;
    border-radius: 25px;
    padding: 12px 30px;
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.modern-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

/* Animated loading spinner */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    border: 3px solid #f3f3f3;
    border-top: 3px solid #FF6B6B;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    animation: spin 1s linear infinite;
    display: inline-block;
    margin-right: 10px;
}

/* Gallery improvements */
.gallery-container {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

/* Theme toggle improvements */
.theme-toggle {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    border-radius: 50px;
    padding: 10px;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Dark theme styles */
.dark-theme {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.dark-theme .main-container {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.dark-theme .search-card {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Smooth transitions */
* {
    transition: all 0.3s ease;
}

/* Modern scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5);
}

/* Floating animation */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.floating-element {
    animation: float 6s ease-in-out infinite;
}

/* Modern input styling */
.gr-textbox, .gr-dropdown {
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(5px) !important;
}
"""

# Create the Gradio UI 
with gr.Blocks(css=modern_css, title="📚 Modern Book Recommender") as dashboard:
    
    # Header section
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
                <div class="floating-element">
                    <div style="font-size: 4rem; text-align: center;">📚</div>
                </div>
            """)
        with gr.Column(scale=4):
            gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h1 style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin: 0;">
                        ✨ Modern Book Recommender
                    </h1>
                    <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.2rem; margin: 10px 0;">
                        Discover your next favorite book with AI-powered recommendations
                    </p>
                </div>
            """)
        with gr.Column(scale=1):
            pass  # Empty column for spacing

    # Search section 
    gr.HTML("""
        <div style="text-align: center; margin: 30px 0;">
            <h2 style="color: white; font-size: 1.8rem;">🔍 Find Your Perfect Read</h2>
            <div style="width: 100px; height: 3px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); margin: 10px auto; border-radius: 5px;"></div>
        </div>
    """)
    
    with gr.Row():
        user_query = gr.Textbox(
            label="✍️ What kind of story are you looking for?",
            placeholder="e.g., A thrilling mystery set in Victorian London, or a heartwarming romance about second chances...",
            lines=3,
            elem_classes=["search-card"]
        )

    with gr.Row():
        with gr.Column():
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📖 Category",
                value="All",
                interactive=True
            )
        with gr.Column():
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Emotional Tone",
                value="All",
                interactive=True
            )
        with gr.Column():
            sort_by_dropdown = gr.Dropdown(
                choices=sorting_options,
                label="🔄 Sort By",
                value="Relevance",
                interactive=True
            )

    with gr.Row():
        with gr.Column():
            submit_button = gr.Button(
                "🚀 Find My Books",
                variant="primary",
                elem_classes=["modern-btn"],
                scale=2
            )
        with gr.Column():
            clear_button = gr.Button(
                "🗑️ Clear",
                elem_classes=["modern-btn"],
                scale=1
            )

    # Status section
    with gr.Row():
        status_box = gr.Textbox(
            label="📊 Status",
            value="✨ Ready to find your next great read!",
            interactive=False,
            elem_classes=["search-card"]
        )

    # Results section
    gr.HTML("""
        <div style="text-align: center; margin: 40px 0 20px 0;">
            <h2 style="color: white; font-size: 1.8rem;">📚 Your Personalized Recommendations</h2>
            <div style="width: 100px; height: 3px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); margin: 10px auto; border-radius: 5px;"></div>
        </div>
    """)
    
    output_gallery = gr.Gallery(
        label="",
        columns=4,
        rows=2,
        show_label=False,
        object_fit="contain",
        elem_classes=["gallery-container"]
    )

    # Search history section
    with gr.Accordion("📜 Search History", open=False):
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 15px;">
                <p style="color: rgba(255, 255, 255, 0.7);">Click on any previous search to reload it</p>
            </div>
        """)
        history_list = gr.Dataframe(
            headers=["🕐 Time", "🔍 Query", "📖 Category", "🎭 Tone", "🔄 Sort"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            elem_classes=["search-card"]
        )

    # Footer
    gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(255, 255, 255, 0.2);">
            <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                Made with ❤️ using AI-powered semantic search | 
                <span style="color: #FF6B6B;">✨ Bringing books and readers together</span>
            </p>
        </div>
    """)

    # Event handlers
    def update_history_list():
        """Update the search history list"""
        if not search_history:
            return []

        # Format for display with emojis
        formatted_history = []
        for item in search_history[::-1]:  # Reverse to show newest first
            # Clean up tone display
            tone_display = item["tone"].replace("😊 ", "").replace("😲 ", "").replace("😠 ", "").replace("😨 ", "").replace("😢 ", "")
            formatted_history.append([
                item["timestamp"],
                item["query"][:50] + "..." if len(item["query"]) > 50 else item["query"],
                item["category"],
                tone_display,
                item["sort_by"]
            ])
        return formatted_history

    def toggle_theme(is_dark):
        """Toggle between light and dark theme - REMOVED"""
        pass

    # Clear search
    clear_button.click(
        fn=clear_search,
        inputs=[],
        outputs=[user_query, category_dropdown, tone_dropdown, sort_by_dropdown]
    )

    # Load history item
    def handle_history_selection(evt: gr.SelectData):
        if not search_history or evt.index[0] >= len(search_history):
            return "", "All", "All", "Relevance"

        # Get the selected history item (accounting for reversed display)
        selected_item = search_history[-(evt.index[0] + 1)]
        
        # Map the tone back to the display format
        tone_mapping = {
            "Happy": "😊 Happy",
            "Surprising": "😲 Surprising", 
            "Angry": "😠 Angry",
            "Suspenseful": "😨 Suspenseful",
            "Sad": "😢 Sad"
        }
        
        display_tone = tone_mapping.get(selected_item["tone"], selected_item["tone"])
        
        return (
            selected_item["query"],
            selected_item["category"],
            display_tone,
            selected_item["sort_by"]
        )

    history_list.select(
        fn=handle_history_selection,
        inputs=[],
        outputs=[user_query, category_dropdown, tone_dropdown, sort_by_dropdown]
    )

    # Main recommendation function
    def recommend_with_history_update(query, category, tone, sort_by):
        # Clean tone for processing (remove emojis)
        clean_tone = tone.replace("😊 ", "").replace("😲 ", "").replace("😠 ", "").replace("😨 ", "").replace("😢 ", "")
        
        # Get recommendations
        results, status = recommend_books(query, category, clean_tone, sort_by)
        
        # Update history
        history = update_history_list()
        
        return results, status, history

    submit_button.click(
        fn=recommend_with_history_update,
        inputs=[user_query, category_dropdown, tone_dropdown, sort_by_dropdown],
        outputs=[output_gallery, status_box, history_list]
    )

    # Initialize search history on load
    dashboard.load(
        fn=update_history_list,
        inputs=[],
        outputs=[history_list]
    )

if __name__ == "__main__":
    # Create placeholder icon if it doesn't exist
    if not os.path.exists("book_icon.png"):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            # Create a simple book icon
            fig, ax = plt.subplots(figsize=(1, 1))
            book = patches.Rectangle((0.2, 0.1), 0.6, 0.8, facecolor='#3498db')
            ax.add_patch(book)
            spine = patches.Rectangle((0.1, 0.1), 0.1, 0.8, facecolor='#2980b9')
            ax.add_patch(spine)
            plt.axis('off')
            plt.savefig("book_icon.png", transparent=True)
            plt.close()
        except Exception as e:
            print(f"Could not create book icon: {e}")

    # Launch the dashboard
    port = int(os.environ.get("PORT", 7860))
    dashboard.launch(server_name="0.0.0.0", server_port=port)
