import requests
import json

def generate_answer(context, question, model_name="mistral"):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt},
            stream=True
        )

        if response.status_code != 200:
            return "⚠️ LLM generation failed."

        answer = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                answer += data.get("response", "")
        return answer.strip()

    except Exception as e:
        print(f"❌ LLM request failed: {e}")
        return "⚠️ Unable to generate an answer."
