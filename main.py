from typing import List, Type, TypeVar

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from conf import LLM_MODEL_NAME, retrieve_documents
from models import FinalResponse, RetrievedDocs

T = TypeVar('T', bound=BaseModel)

load_dotenv()
client = instructor.from_anthropic(Anthropic())


def llm_request(response_model: Type[T], prompt: str) -> T:
    return client.messages.create(
        model=LLM_MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
    )


def generate_final_response(user_input: str, retrieved_docs: List[str]) -> FinalResponse:
    context = "\n\n".join(retrieved_docs)
    prompt = (f"Here is the retrieved information:\n{context}\n\n"
              f"Now answer the question only based on the retrieved information: {user_input}"
              f"---"
              f"Keep your answer as a paragraph and include the references as you write within the sentences.")
    return llm_request(FinalResponse, prompt)


def retrieve_docs_for_question(user_input: str) -> RetrievedDocs:
    docs = retrieve_documents(user_input)
    return RetrievedDocs(relevant_texts=docs)


def agentic_rag_workflow(user_question: str):
    retrieved_docs = []
    user_question_upd = user_question
    retrieved_docs.extend(retrieve_documents(user_question_upd, k=5))

    print("💡 Generating Final Answer...")
    final_answer = generate_final_response(user_question, retrieved_docs)
    print(f"Answer: {final_answer.answer}")


if __name__ == "__main__":
    CONT = True
    while CONT:
        try:
            user_question = input("Ask a question: \n")
            agentic_rag_workflow(user_question)
        except KeyboardInterrupt:
            CONT = False
            print("Exiting...")
            break
