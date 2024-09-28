import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

embedding = OpenAIEmbeddings(base_url='https://api.proxyapi.ru/openai/v1')

loader = TextLoader('data.txt')
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        model='gpt-3.5-turbo-1106',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.proxyapi.ru/openai/v1",
    ),
    retriever=index.vectorstore.as_retriever(
        search_kwargs={'k': 1}
    )
)

chat_log = []


def retrieve(query: str):
    result = chain.invoke({'question': query, 'chat_history': chat_log})

    print(result)

    chat_log.extend([('human', query), ('ai', result['answer'])])


if __name__ == '__main__':
    while (q := input('Enter a query: ')) != '':
        retrieve(q)
