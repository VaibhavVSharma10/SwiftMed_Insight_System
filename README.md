# SwiftMed_Insight_System
The SwiftMed Insight System is a powerful tool designed to provide medical information by answering user queries using state-of-the-art language models and vector stores. This README will guide you through the setup and usage of the Llama2 and a RAG system.

    • Engineered a medical bot to extract data from a 600-page document, achieving data quantization using Llama 2 models.
    • Implemented FAISS as a vector store for efficient storage of vector embeddings, optimizing data retrieval and storage.
    • Utilized LangChain to develop a robust RAG system, deploying it with Chainlit. 

The project automates data extraction and quantization from extensive medical documents, enabling accelerated literature review, streamlined clinical data processing, and improved healthcare decision-making.

    1. Accelerated literature review and analysis for researchers and medical professionals.
    2. Improved data processing and analysis from electronic health records for better patient monitoring and decision support.
    3. Streamlined data management and reporting in clinical trials.
    4. Construction of comprehensive medical knowledge bases for various applications.
    5. Enhanced data analysis and decision-making in pharmaceutical research and drug development.

How to Run it:

Command 1    
    To form vectors from the pdf
    
        python ingest.py

Command 2        
    To run the chainlit model for a CHATGPT like Interface for the SwiftMed Insight System
    
        chainlit run model.py -w 
