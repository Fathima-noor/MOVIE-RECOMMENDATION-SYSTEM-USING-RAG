import os
import pandas as pd

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please add it to your .env file."
    )


# ============================================================
# 2. LOAD MOVIE DATASET
# ============================================================

FILE_NAME = "movies_list.xlsx"

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"{FILE_NAME} not found. Please keep the Excel file "
        "in the same folder as this Python file."
    )

df = pd.read_excel(FILE_NAME)

print(f"Movies Loaded: {len(df)}")


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "title",
    "genres",
    "release_date",
    "original_language",
    "vote_average",
    "overview"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in Excel file: {missing_columns}"
    )


# ============================================================
# 4. CLEAN DATA
# ============================================================

df = df[required_columns].copy()

df = df.fillna("Not available")


# ============================================================
# 5. CREATE LANGCHAIN DOCUMENTS
# ============================================================

documents = []

for _, row in df.iterrows():

    movie_text = f"""
Movie Name: {row['title']}
Genre: {row['genres']}
Release Date: {row['release_date']}
Language: {row['original_language']}
Rating: {row['vote_average']}
Overview: {row['overview']}
"""

    document = Document(
        page_content=movie_text.strip(),
        metadata={
            "title": str(row["title"]),
            "genre": str(row["genres"]),
            "rating": str(row["vote_average"])
        }
    )

    documents.append(document)


print(f"Documents Created: {len(documents)}")


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 7. CREATE FAISS VECTOR DATABASE
# ============================================================

print("\nCreating FAISS vector database...")

vector_db = FAISS.from_documents(
    documents,
    embeddings
)

print("FAISS vector database created successfully.")


# ============================================================
# 8. CREATE RETRIEVER
# ============================================================

retriever = vector_db.as_retriever(
    search_kwargs={"k": 5}
)


# ============================================================
# 9. CREATE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=API_KEY,
    temperature=0.7
)


# ============================================================
# 10. CREATE PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "user_query"],
    template="""
You are a helpful Movie Recommendation Assistant.

You must recommend movies ONLY from the retrieved movie information below.

Retrieved Movies:
-----------------
{context}
-----------------

User Preference:
{user_query}

Recommend the best movies that match the user's preference.

For each recommended movie provide:

1. Movie Name
2. Genre
3. Rating
4. Reason
5. Why it matches the user's interest

Important rules:
- Recommend ONLY movies present in the retrieved information.
- Do not invent movie names.
- Do not invent ratings.
- Do not invent genres.
- Do not use information outside the retrieved movies.
- If none of the retrieved movies are a good match, clearly say that no suitable movie was found.
"""
)


# ============================================================
# 11. CREATE RAG CHAIN
# ============================================================

chain = prompt | llm


# ============================================================
# 12. MOVIE RECOMMENDATION SYSTEM
# ============================================================

print("\n==============================================")
print("       MOVIE RECOMMENDATION SYSTEM")
print("==============================================")

print("\nYou can type:")
print("- action movies")
print("- sci-fi movies")
print("- highly rated thriller movies")
print("- movies about space")
print("- comedy movies")
print("\nType 'exit' to quit.")


while True:

    user_query = input("\nEnter your movie preference: ").strip()

    if user_query.lower() == "exit":
        print("\nThank you for using the Movie Recommendation System!")
        break

    if not user_query:
        print("Please enter a movie preference.")
        continue


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print("\nSearching for relevant movies...")

    retrieved_docs = retriever.invoke(user_query)


    if not retrieved_docs:
        print("No relevant movies were found.")
        continue


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = "\n\n--- MOVIE ---\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    try:

        response = chain.invoke({
            "context": context,
            "user_query": user_query
        })

        print("\n==============================================")
        print("           RECOMMENDED MOVIES")
        print("==============================================\n")

        print(response.content)

    except Exception as e:

        print("\nError while generating recommendation:")
        print(e)