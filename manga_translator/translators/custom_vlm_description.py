import re

from ..config import TranslatorConfig
from .config_gpt import ConfigGPT  # Import the `gpt_config` parsing parent class

from openai import OpenAI
import asyncio
import time
from typing import List
from .common import CommonTranslator, VALID_LANGUAGES
from .keys import CUSTOM_OPENAI_API_KEY, CUSTOM_OPENAI_API_BASE, CUSTOM_OPENAI_MODEL, CUSTOM_OPENAI_MODEL_CONF
import base64
from io import BytesIO
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class CustomVlmDescriptionTranslator(ConfigGPT, CommonTranslator):
    _INVALID_REPEAT_COUNT = 2  # 如果检测到“无效”翻译，最多重复 2 次
    _MAX_REQUESTS_PER_MINUTE = 40  # 每分钟最大请求次数
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    _MAX_TOKENS = 4096

    # 是否返回原始提示，用于控制输出内容
    _RETURN_PROMPT = False

    # 是否包含模板，用于决定是否使用预设的提示模板
    _INCLUDE_TEMPLATE = False

    def __init__(self, model=None, api_base=None, api_key=None, check_openai_key=False):
        # If the user has specified a nested key to use for the model, append the key
        #   Otherwise: Use the `ollama` defaults.
        _CONFIG_KEY='ollama'
        if CUSTOM_OPENAI_MODEL_CONF:
            _CONFIG_KEY+=f".{CUSTOM_OPENAI_MODEL_CONF}"

        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        self.model = "gemma3:12b"
        CommonTranslator.__init__(self)
        api_base = "http://localhost:5001/v1/"
        # api_base = "http://localhost:11434/v1"
        self.client = OpenAI(
            base_url=api_base or CUSTOM_OPENAI_API_BASE,
            api_key=api_key or CUSTOM_OPENAI_API_KEY or "ollama",
        )
        self.token_count = 0
        self.token_count_last = 0
        # self.n=0
        self.summary = None

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config


    def extract_capture_groups(self, text, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.
        
        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matches
        
        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )
        
        return extracted_text.strip() if extracted_text else None

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt = ''

        if self._INCLUDE_TEMPLATE:
            prompt += self.prompt_template.format(to_lang=to_lang)

        if self._RETURN_PROMPT:
            prompt += '\nOriginal:'

        i_offset = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i + 1 - i_offset}|>{query}'

            # If prompt is growing too large and there's still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if self._MAX_TOKENS * 2 and len(''.join(queries[i + 1:])) > self._MAX_TOKENS:
                if self._RETURN_PROMPT:
                    prompt += '\n<|1|>'
                yield prompt.lstrip(), i + 1 - i_offset
                prompt = self.prompt_template.format(to_lang=to_lang)
                # Restart counting at 1
                i_offset = i + 1

        if self._RETURN_PROMPT:
            prompt += '\n<|1|>'

        yield prompt.lstrip(), len(queries) - i_offset

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        if to_lang in self.chat_sample:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                self.chat_sample[to_lang][0],
                'Assistant:',
                self.chat_sample[to_lang][1],
                'User:',
                prompt,
            ])
        else:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                prompt,
            ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], image: Image) -> List[str]:
        translations = []
        # print(self.n)
        # self.n+=1
        print("image")
        print(image)
        image.save("test.jpg")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue())
        base64_image = encode_image("test.jpg")

        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')
        translations = self._request_translation(to_lang, queries, base64_image)

        return translations

    def _request_translation(self, to_lang: str, samples: List[str], base64_image) -> List[str]:
        self.system_prompt="You are a professional manga translator and image captioner."
        self.prefix_prompt="Your job is to translate the following text to English. I will show you the full text beforehand, but we will translate it line by line. You will have to reply only with the translated line.\nJapanese manga text:\n"
        self.image_prompt="First just give a short (1-2 paragraph) description of only the visual scene. Focus on the characters and the background. Don't write anything else."
        self.new_summary_prompt="Now give a short (1-2 paragraph) but precise summary of the story so far based on the image and text."

        numbered_text = "\n".join(
            [f"{i + 1}: {line}" for i, line in enumerate(samples)]
        )
        summary_prompt = (
            ("Summary of the story so far:\n" + self.summary + "\n") if self.summary else ""
        )
        prompt = f"{summary_prompt}{self.prefix_prompt}{numbered_text}\nAre you ready?"
        history = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.image_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=history,
            max_completion_tokens=1000,
        )
        print("description")
        print(completion.choices[0].message.content)
        history.append(completion.choices[0].message)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": "Yes, I am ready."})
        translated_text = []
        for line in samples:
            history.append({"role": "user", "content": line})
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0,
                messages=history,
            max_completion_tokens=1000,
            )
            history.append(completion.choices[0].message)
            translated_text.append(completion.choices[0].message.content.strip())
        print("translated_text")
        print(translated_text)
        history.append(
            {
                "role": "user",
                "content": self.new_summary_prompt,
            }
        )
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0,
            messages=history,
            max_completion_tokens=1000,
        )
        self.summary = completion.choices[0].message.content
        print("summary")
        print(self.summary)

        return translated_text
