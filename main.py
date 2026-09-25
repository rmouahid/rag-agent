import sys

from rag_agent.generation.llm import ModelNotFoundError
from rag_agent.generation.rag_agent import RAGAgent

def main():
    print("=" * 50)
    print("  RAG Agent — powered by a local LLM (llama.cpp)")
    print("=" * 50)

    try:
        agent = RAGAgent()
    except ModelNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)

    # --- INGESTION ---
    print("\n INGESTION DES SOURCES")
    print("Ajoute tes sources (pdf, markdown, url). Tape 'done' pour terminer.\n")

    while True:
        source = input("Source (chemin ou url) : ").strip()
        if source.lower() == "done":
            break
        if not source:
            continue
        try:
            agent.ingest(source)
        except Exception as e:
            print(f" Erreur : {e}")

    # Sauvegarde le vector store
    agent.save()

    # --- QUESTIONS ---
    print("\n CHAT — pose tes questions (tape 'exit' pour quitter)\n")

    while True:
        question = input("Question : ").strip()
        if question.lower() == "exit":
            print(" À bientôt !")
            break
        if not question:
            continue

        answer = agent.ask(question)
        print(f"\n Réponse :\n{answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()