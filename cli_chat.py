from dotenv import load_dotenv

from rag_pipeline import answer_question, load_vectorstore, OFF_TOPIC_RESPONSE

load_dotenv()

if __name__ == "__main__":
    vectorstore = load_vectorstore()

    print("Yosemite chat is ready.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\nAssistant: Goodbye.")
            break

        if not question:
            continue

        answer, sources = answer_question(question, vectorstore)

        print(f"\nAssistant: {answer}")

        if answer != OFF_TOPIC_RESPONSE and sources:
            print("\nSources:")
            for i, source in enumerate(sources, start=1):
                print(f"\nSource {i} ({source['page']}):")
                print(source["content"])