import re
from dotenv import load_dotenv
import requests
from typing import List, Dict
import os
load_dotenv()

class GPT4oMiniAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def generate_response(self, prompt: str, model_name: str = "gpt-4o-mini") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        try:
            response = requests.post(self.endpoint, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패: {e}")
            return ""

class GPTModel:
    def __init__(self, name: str, api: GPT4oMiniAPI):
        self.name = name
        self.api = api

    def generate_response(self, conversation_text: str, participants: list) -> str:
        prompt = f"""다음은 GPT 모델들 간의 토론입니다. {self.name}으로서 대화에 참여해주세요. 당신은 찬성 또는 반대 입장을 가질 수 있습니다.
        현재 대화에 참여하고 있는 gpt는 {', '.join(participants)} 입니다.
        다른 gpt 모델에게 반박, 질문할 수 있습니다. 이때 반박, 질문하려는 gpt의 이름이 gpt0이라고 하면, '[gpt0에게]' 로 명시해 주세요.
        예를 들면 "[gpt0에게] 저는 당신의 주장에 동의하지 않습니다. 그 이유는 ~ 이기 때문입니다."
이전 대화:
{conversation_text}

{self.name}의 응답:"""
        
        return self.api.generate_response(prompt=prompt)

class DebateChat:
    def __init__(self, models: List[GPTModel]):
        self.models = {model.name: model for model in models}
        self.history: List[Dict[str, str]] = []
        self.turn = 0
        self.participants = list(self.models.keys())  # 참여자 목록 저장

    def _extract_mentions(self, message: str) -> List[str]:
        pattern = r"\[([\w\d]+)에게\]"
        mentions = re.findall(pattern, message)
        return mentions

    def add_message(self, speaker: str, message: str):
        mentions = self._extract_mentions(message)
        self.history.append({
            "speaker": speaker,
            "message": message,
            "mentions": mentions
        })

    def get_full_conversation_text(self) -> str:
        conversation_text = ""
        for entry in self.history:
            speaker = entry["speaker"]
            message = entry["message"]
            conversation_text += f"{speaker}: {message}\n"
        return conversation_text.strip()

    def run_debate(self, topic: str, max_steps: int = 5):
        print(f"== 토론 시작 ==\n주제: {topic}")
        self.add_message("system", f"주제: {topic}")

        for step in range(1, max_steps + 1):
            print(f"\n--- 타임스텝 {step} ---")
            conversation_text = self.get_full_conversation_text()

            for model_name, model in self.models.items():
                response = model.generate_response(conversation_text, self.participants)  # 참여자 목록 전달
                if response.strip():
                    self.add_message(model_name, response)
                    print(f"{model_name}: {response}")

            new_mentions = []
            for entry in self.history[-len(self.models):]:
                if entry["mentions"]:
                    new_mentions.extend([(m, entry) for m in entry["mentions"]])

            for mention_model_name, mention_entry in new_mentions:
                if mention_model_name in self.models:
                    mention_model = self.models[mention_model_name]
                    mention_text = self.get_full_conversation_text()
                    mention_response = mention_model.generate_response(mention_text, self.participants)  # 참여자 목록 전달
                    if mention_response.strip():
                        self.add_message(mention_model_name, mention_response)
                        print(f"{mention_model_name}(mention 응답): {mention_response}")

            self.turn += 1

        print("\n== 토론 종료 ==\n")


if __name__ == "__main__":
    # API 키 설정 (환경 변수나 직접 입력)
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # GPT4oMiniAPI 인스턴스 생성
    gpt4o_mini_api = GPT4oMiniAPI(api_key=api_key)

    # GPTModel 인스턴스들 생성
    gpt_models = [
        GPTModel("gpt1", gpt4o_mini_api),
        GPTModel("gpt2", gpt4o_mini_api),
        GPTModel("gpt3", gpt4o_mini_api)
    ]

    # DebateChat 인스턴스 생성 및 실행
    debate_chat = DebateChat(gpt_models)
    debate_chat.run_debate("'동물 실험은 허용되야한다'에 대해 찬성/반대로 나뉘어 토론하세요.", max_steps=3)