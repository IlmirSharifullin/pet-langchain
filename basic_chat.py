import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()


def main():
    model = ChatOpenAI(
        model='gpt-3.5-turbo-1106',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.proxyapi.ru/openai/v1",
    )

    messages = [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="Hi, my name is Ilmir! What`s your name?"),
    ]

    result = model.invoke(messages)
    print(result.content)


if __name__ == '__main__':
    main()
