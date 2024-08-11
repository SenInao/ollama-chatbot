from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class Chatbot:
    def __init__(self, template, modelType) -> None:
        self.template = template
        self.modelType = modelType
        self.model = OllamaLLM(model=self.modelType)
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.model
        self.chatContext = ""

    def ask(self, question):
        result = self.chain.invoke({"context":self.chatContext, "question":question})
        self.chatContext += f"User:{question}\nAnswer:{result}\n"
        return result

template = """
You are a chatbot and will answer the question a user asks

Here is the conversation history with the user: {context}

This is the question of the user: {question}

Answer short and precise using the context of the conversation
"""

def main():
    chatbot = Chatbot(template, "llama3")

    while (True):
        userInput = input(">> ")
        if (userInput == "exit"):
            break
        print(f"Bot:\n{chatbot.ask(userInput)}\n")

if ("__main__" == __name__): 
    main()
