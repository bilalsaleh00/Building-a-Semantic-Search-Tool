
import streamlit as st
import chromadb
from pathlib import Path

# ==========================================
# STEP 1: STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Expanded Semantic Search",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "search_db"

DOCS_DIR.mkdir(exist_ok=True)


# ==========================================
# STEP 2: CHROMADB SETUP
# ==========================================

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    return client.get_or_create_collection(
        name="course_docs",
        configuration={"hnsw": {"space": "cosine"}}
    )


collection = get_collection()


# ==========================================
# STEP 3: LOAD AND CHUNK DOCUMENTS
# ==========================================

def load_and_chunk(directory):
    chunks = []

    for filepath in sorted(directory.iterdir()):

        if not filepath.is_file():
            continue

        if filepath.suffix.lower() not in (".txt", ".md"):
            continue

        content = filepath.read_text(
            encoding="utf-8"
        )

        paragraphs = [
            p.strip()
            for p in content.splitlines()
        ]

        # Rebuild full paragraphs separated by blank lines
        paragraphs = [
            p.strip()
            for p in content.replace(
                "\r\n", "\n"
            ).split("\n\n")
            if p.strip()
        ]

        for i, paragraph in enumerate(paragraphs):
            chunks.append({
                "id": f"{filepath.name}_{i}",
                "text": paragraph,
                "source": filepath.name,
                "chunk_index": i
            })

    return chunks


# ==========================================
# STEP 4: RE-INDEX DOCUMENTS
# ==========================================

def reindex_documents():
    chunks = load_and_chunk(DOCS_DIR)

    if not chunks:
        st.warning("No document chunks found.")
        return

    with st.spinner("Indexing documents..."):

        old_ids = collection.get()["ids"]

        if old_ids:
            collection.delete(ids=old_ids)

        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "source": c["source"],
                    "chunk_index": c["chunk_index"]
                }
                for c in chunks
            ]
        )

    st.success(
        f"Indexed {len(chunks)} chunks successfully!"
    )


# ==========================================
# STEP 5: GET AVAILABLE SOURCES
# ==========================================

def get_sources():
    data = collection.get(include=["metadatas"])

    sources = {
        metadata["source"]
        for metadata in data["metadatas"]
        if metadata and "source" in metadata
    }

    return sorted(sources)


def set_similar_query(text):
    st.session_state.search_query = text


# ==========================================
# STEP 6: SESSION STATE
# ==========================================

if "search_query" not in st.session_state:
    st.session_state.search_query = ""


# ==========================================
# STEP 7: SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📁 Document Manager")

    if st.button("🔄 Re-index Documents"):
        reindex_documents()

    total_documents = collection.count()
    available_sources = get_sources()

    st.metric(
        "Total Documents",
        total_documents
    )

    st.metric(
        "Unique Source Files",
        len(available_sources)
    )

    st.divider()

    st.subheader("Source Filter")

    selected_sources = st.multiselect(
        "Search within these files",
        options=available_sources,
        help="Leave empty to search all files."
    )

    st.divider()

    n_results = st.slider(
        "Results to show",
        min_value=1,
        max_value=10,
        value=5
    )


# ==========================================
# STEP 8: BUILD SOURCE FILTER
# ==========================================

where_filter = None

if len(selected_sources) == 1:
    where_filter = {
        "source": selected_sources[0]
    }

elif len(selected_sources) > 1:
    where_filter = {
        "source": {
            "$in": selected_sources
        }
    }


# ==========================================
# STEP 9: MAIN SEARCH INTERFACE
# ==========================================

st.title("🔍 Expanded Semantic Search")

st.write(
    "Search your course documents by meaning, "
    "with source filtering and expandable results."
)

query = st.text_input(
    "Enter your search query",
    key="search_query",
    placeholder="How does Streamlit remember user data?"
)


# ==========================================
# STEP 10: SEARCH DOCUMENTS
# ==========================================

if query.strip():

    total_documents = collection.count()

    if total_documents == 0:
        st.info(
            "Click Re-index Documents in the sidebar first."
        )

    else:
        # Count documents matching the source filter
        if where_filter:
            filtered_data = collection.get(
                where=where_filter
            )
            matching_count = len(filtered_data["ids"])
        else:
            matching_count = total_documents

        if matching_count == 0:
            st.warning(
                "No documents match the selected sources."
            )

        else:
            results = collection.query(
                query_texts=[query],
                n_results=min(
                    n_results,
                    matching_count
                ),
                where=where_filter,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            # REQUIRED: RESULT COUNT
            st.subheader(
                f"Showing {len(documents)} of "
                f"{total_documents} total documents"
            )

            if where_filter:
                st.caption(
                    f"{matching_count} documents match "
                    "your source filter."
                )

            # ==================================
            # STEP 11: DISPLAY SEARCH RESULTS
            # ==================================

            for i, (doc, metadata, distance) in enumerate(
                zip(documents, metadatas, distances), 1
            ):

                # BONUS: Relevance indicators
                if distance < 0.5:
                    relevance = "🟢 High"
                elif distance < 1.0:
                    relevance = "🟡 Medium"
                else:
                    relevance = "🔴 Low"

                with st.container(border=True):

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"### Result {i}")

                        st.write(
                            f"**Source:** {metadata['source']}"
                        )

                        st.write(
                            f"**Chunk:** {metadata['chunk_index']}"
                        )

                    with col2:
                        st.write(relevance)
                        st.caption(
                            f"Distance: {distance:.4f}"
                        )

                    # REQUIRED: 150-char preview
                    preview = doc[:150]

                    if len(doc) > 150:
                        preview += "..."

                    st.write(preview)

                    # REQUIRED: Expandable full text
                    with st.expander("View Full Text"):
                        st.write(doc)

                    # BONUS: Similar to this
                    st.button(
                        "🔎 Similar to this",
                        key=f"similar_{i}",
                        on_click=set_similar_query,
                        args=(doc,)
                    )

else:
    st.info(
        "Enter a search query to find related documents."
    )
