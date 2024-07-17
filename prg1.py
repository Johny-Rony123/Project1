import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain_community.embeddings import OpenAIEmbeddings
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set up OpenAI API key
openai_api_key = ''  # Replace with your actual OpenAI API key

# Set OpenAI API key in environment variable for LangChain OpenAI model
os.environ["OPENAI_API_KEY"] = openai_api_key

# Azure Form Recognizer endpoint and key
endpoint = "https://tmdocaiio.cognitiveservices.azure.com/"
key = "09d5375484e348c1849b011b6e3ce01c"

# Local path to the PDF document
pdf_path = "./final.pdf"

# Initialize the DocumentAnalysisClient
document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Initialize LangChain embeddings with the OpenAI API key
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding_with_retry(text):
    return embeddings.embed_query(text)

# Function to get embeddings with enhanced retry logic
def get_embedding(text):
    try:
        return get_embedding_with_retry(text)
    except RetryError as e:
        print(f"RetryError: {e}")
        raise  # Re-raise the exception for further handling
    except Exception as e:
        print(f"Failed to get embedding for text '{text}': {e}")
        return []

# Read the local PDF document and analyze it
with open(pdf_path, "rb") as pdf_file:
    poller = document_analysis_client.begin_analyze_document("prebuilt-invoice", document=pdf_file.read())

invoices = poller.result()

# Process the extracted invoice data
document_data = []
for idx, invoice in enumerate(invoices.documents):
    print(f"--------Recognizing invoice #{idx + 1}--------")
    
    # Collect all field values into a single text string
    concatenated_text = ""
    for field, value in invoice.fields.items():
        if value:
            concatenated_text += str(value.value) + " "
    
    # Print the concatenated text
    print(f"Concatenated Text: {concatenated_text.strip()}")
    
    # Embed the concatenated text
    if concatenated_text.strip():  # Check if there is non-empty text to embed
        try:
            embedding = get_embedding(concatenated_text)
            if embedding:
                print(f"Embedding for Concatenated Text: {embedding}")
                document_data.append({"text": concatenated_text, "embedding": embedding})
        except RetryError as e:
            print(f"RetryError while embedding: {e}")
            continue  # Skip to the next invoice if embedding fails
        except Exception as e:
            print(f"Failed to embed text '{concatenated_text}': {e}")
            continue
        
    print("----------------------------------------")

# Initialize LangChain OpenAI model
llm = LangChainOpenAI()

# Define the contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create the contextualize question prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(llm, contextualize_q_prompt)

# Example user query
user_query = "What is the invoice number?"

# Iterate through each step in the chain and invoke with the query
for step in retrieval_chain.steps:
    response = step.invoke(query=user_query)
    print("Response to user query:")
    print(response)
