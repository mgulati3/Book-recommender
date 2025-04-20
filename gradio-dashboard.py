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
        return [], "Please enter a search query"

    progress(0, desc="Starting search...")

    # Add to search history
    add_to_search_history(query, category, tone, sort_by)

    progress(0.3, desc="Retrieving semantic matches...")
    recommendations = retrieve_semantic_recommendations(
        query, category, tone, sort_by
    )

    progress(0.6, desc="Formatting results...")
    results = []

    if recommendations.empty:
        return [], "No books found matching your criteria. Try adjusting your filters."

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

        # Format emotion data
        emotion_cols = ["joy", "surprise", "anger", "fear", "sadness"]
        emotion_info = []
        for emo in emotion_cols:
            if pd.notnull(row.get(emo, None)) and row.get(emo, 0) > 0:
                percentage = round(row[emo] * 100)
                emotion_info.append(f"{emo.capitalize()}: {percentage}%")

        emotion_text = ", ".join(emotion_info) if emotion_info else "No emotion data"

        # Create caption
        caption = (
            f"**{row['title']}**\n"
            f"By {authors_str}\n\n"
            f"{truncated_description}\n\n"
            f"**Emotional Profile:** {emotion_text}"
        )

        # Add to results
        results.append((row["large_thumbnail"], caption))

    progress(1.0, desc="Done!")
    return results, f"Found {len(results)} books matching your criteria"


def load_from_history(history_item):
    """Load search parameters from history"""
    return (
        history_item["query"],
        history_item["category"],
        history_item["tone"],
        history_item["sort_by"]
    )


def clear_search():
    """Clear search inputs"""
    return "", "All", "All", "Relevance"


# Prepare UI elements
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
sorting_options = ["Relevance", "Emotion Intensity"]

# Create the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    # Header section
    with gr.Row():
        with gr.Column(scale=1):
            if os.path.exists("book_icon.png"):
                gr.Image("book_icon.png", show_label=False, width=100)
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # 📚 Enhanced Semantic Book Recommender
                Discover books that match your interests and emotional preferences
                """
            )
        with gr.Column(scale=1):
            theme_toggle = gr.Checkbox(
                label="Dark Theme",
                value=False,
                interactive=True
            )

    # Main search section (using gr.Row instead of gr.Box)
    gr.Markdown("### 🔍 Find Your Next Great Read")
    with gr.Row():
        user_query = gr.Textbox(
            label="What kind of story are you looking for?",
            placeholder="e.g., A story about second chances and forgiveness in a small town",
            lines=2
        )

    with gr.Row():
        with gr.Column():
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Category:",
                value="All",
                interactive=True
            )
        with gr.Column():
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Emotional Tone:",
                value="All",
                interactive=True
            )
        with gr.Column():
            sort_by_dropdown = gr.Dropdown(
                choices=sorting_options,
                label="Sort By:",
                value="Relevance",
                interactive=True
            )

    with gr.Row():
        with gr.Column():
            submit_button = gr.Button("🔍 Find Recommendations", variant="primary")
        with gr.Column():
            clear_button = gr.Button("🗑️ Clear Search")

    # Status section
    with gr.Row():
        status_box = gr.Textbox(label="Status", value="Enter your search criteria above", interactive=False)

    # Results section
    gr.Markdown("## ✨ Recommended Books")
    output_gallery = gr.Gallery(
        label="Recommended Books",
        columns=4,
        rows=2,
        show_label=False,
        object_fit="contain"
    )

    # Book details section (using Row instead of Box)
    gr.Markdown("### 📖 Book Details", visible=False)
    with gr.Row(visible=False) as book_details_row:
        with gr.Column(scale=1):
            detail_image = gr.Image(label="Cover", interactive=False)
        with gr.Column(scale=3):
            detail_title = gr.Textbox(label="Title", interactive=False)
            detail_author = gr.Textbox(label="Author(s)", interactive=False)
            detail_category = gr.Textbox(label="Category", interactive=False)
            detail_description = gr.Textbox(label="Description", lines=5, interactive=False)

    # Search history section
    with gr.Accordion("📜 Search History", open=False):
        history_list = gr.Dataframe(
            headers=["Time", "Query", "Category", "Tone", "Sort By"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )
        load_history_button = gr.Button("Load Selected Search")


    # Event handlers
    def update_history_list():
        """Update the search history list"""
        if not search_history:
            return []

        # Format for display
        return [[
            item["timestamp"],
            item["query"],
            item["category"],
            item["tone"],
            item["sort_by"]
        ] for item in search_history[::-1]]  # Reverse to show newest first


    # Set theme based on toggle
    def set_theme(dark_mode):
        if dark_mode:
            return gr.themes.Monochrome()
        return gr.themes.Soft()


    theme_toggle.change(
        fn=set_theme,
        inputs=theme_toggle,
        outputs=dashboard
    )

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
        return (
            selected_item["query"],
            selected_item["category"],
            selected_item["tone"],
            selected_item["sort_by"]
        )


    history_list.select(
        fn=handle_history_selection,
        inputs=[],
        outputs=[user_query, category_dropdown, tone_dropdown, sort_by_dropdown]
    )

    # Update history when search completed
    submit_button.click(
        fn=update_history_list,
        inputs=[],
        outputs=history_list
    )

    # Main recommendation function
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown, sort_by_dropdown],
        outputs=[output_gallery, status_box]
    )

    # Initialize search history
    dashboard.load(
        fn=update_history_list,
        inputs=[],
        outputs=history_list
    )

    # Footer
    gr.Markdown("""
    ---
    **Made with ❤️ by Manan Gulati** 

    *This application uses semantic search to find books matching your interests.*
    """)

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
    dashboard.launch()