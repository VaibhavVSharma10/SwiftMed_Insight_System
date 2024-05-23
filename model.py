# Importing necessary modules for loading PDF documents
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Importing prompt template for creating custom prompts
from langchain.prompts import PromptTemplate

# Importing HuggingFace embeddings for generating text embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Importing FAISS for vector store operations
from langchain_community.vectorstores import FAISS

# Importing CTransformers for loading the language model
from langchain_community.llms import CTransformers

# Importing RetrievalQA for building the retrieval-based QA system
from langchain.chains import RetrievalQA

# Importing chainlit for creating interactive chat applications
import chainlit as cl

# Path to the saved FAISS vector store
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt template for answering questions using provided context
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    # Initialize and return a prompt template with specified input variables
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Function to create a retrieval-based QA chain
def retrieval_qa_chain(llm, prompt, db):
    # Initialize and return a RetrievalQA chain using the specified LLM, prompt, and vector store
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain

# Function to load the language model
def load_llm():
    # Load the locally downloaded model and return it
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Function to initialize the QA bot
def qa_bot():
    # Initialize HuggingFace embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    # Load the FAISS vector store from the specified path
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Load the language model
    llm = load_llm()
    
    # Set the custom prompt for the QA system
    qa_prompt = set_custom_prompt()
    
    # Create and return the QA chain
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Function to get the final result for a given query
def final_result(query):
    # Initialize the QA bot
    qa_result = qa_bot()
    
    # Get the response from the QA bot for the given query
    response = qa_result({'query': query})
    
    return response

# chainlit code for handling chat interactions
@cl.on_chat_start
async def start():
    # Initialize the QA bot chain
    chain = qa_bot()
    
    # Send a starting message to the user
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    
    # Update the message with a welcome prompt
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    # Store the QA bot chain in the user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the QA bot chain from the user session
    chain = cl.user_session.get("chain") 
    
    # Create a callback handler for streaming the final answer
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    # Set the callback to indicate the answer has been reached
    cb.answer_reached = True
    
    # Get the response from the QA bot for the user's message
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    # Append sources to the answer if available
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    # Send the answer back to the user
    await cl.Message(content=answer).send()