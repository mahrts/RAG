## 🚀 NASA Intelligence Chat System

A Retrieval-Augmented Generation (RAG) app: built with Python and enables users to ask questions about historic NASA missions (apollo11, apollo13 and challenger).

This project combines semantic search, vector databases, LLMs, and evaluation tooling (ragas)
to create an assistant capable of answering questions, based on archival mission data (.txt documents).

## 🔄 Installation and Usage
This app needs an openAI API_KEY.
One can run the app locally with the following sequence commands

```bash
git clone https://github.com/mahrts/RAG && cd RAG
```
```bash
export OPENAI_API_KEY=<your_open_api_key> #setting your sercet api key as env varialbe
```
```bash
pip install -r requirements.txt && pip install -e .
```

Download and save the text data (his downloads all the .txt files here: [data_text](https://github.com/udacity/cd13318-exercises-project/tree/main/Project-NASA-Mission-Intelligence-Starter/data_text))

```bash
python src/DATA/data_text.py #This will download relevant documents to retrieve from.
```
Then, create and save chromadb database that corresponds to the relevant documents
(the following command is enough for default settings.)

```bash
python src/EMBEDDiNG/embedding_pipeline.py python src/EMBEDDING/embedding_pipeline.py --openai-key <YOUR-API-KEY>
```
Finally, the app can be opend with: (it should open here [http://localhost:8501/](http://localhost:8501/))
```bash
streamlit run chat.py # this open the app on browser automatically 
```
## Project structure:
```bash
├── chat.py
├── README.md
├── requirements.txt
├── src
│   ├── DATA
│   │   ├── data_text.py #Create a data_text/ folder and download mission texts there.
│   │   └── __init__.py
│   ├── EMBEDDING
│   │   ├── embedding_pipeline.py #Full RAG pipeline for the documents.
│   │   └── __init__.py
│   ├── LLMCLIENT
│   │   ├── __init__.py
│   │   └── llm_client.py #LLL to generate chats
│   ├── RAGAS
│   │   ├── __init__.py
│   │   └── ragas_evaluator.py #RAGAS evaluator (bach and single message)
|   |   |__ test_questions.json #Example of questions for evaluations
│   └── RAGCLIENT
│       ├── __init__.py
│       └── rag_client.py #Relate rag system: documents and chromadb database.
└── tests
    ├── test_DATA.py
    ├── test_LLMCLIENT.py
    └── test_RAGCLIENT.py
```

#### Next:
More test coverage, github action CI/CD

## References: 

RAG Paper: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

RAGAS Paper: [https://arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)
