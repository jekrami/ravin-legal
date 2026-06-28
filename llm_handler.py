import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from config import (
    PRIMARY_LEGAL_ANALYST_MODEL,
    SECONDARY_VERIFICATION_MODEL,
    SYNTHESIZER_MODEL,
)
from legal_analyzer.ollama_client import OllamaError, ollama_generate

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=5)


class LlmHandler:
    def _query_model_sync(self, model_name, prompt):
        """Synchronous function to query the Ollama model."""
        try:
            return ollama_generate(model_name, prompt)
        except OllamaError as e:
            logger.error("Model query failed for %s: %s", model_name, e)
            return None

    async def _query_model_async(self, model_name, prompt):
        """Asynchronous wrapper to run the synchronous query in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, self._query_model_sync, model_name, prompt
        )

    async def get_synthesized_answer(self, question, context):
        base_prompt = f"""
        **دستورالعمل:** شما یک تحلیلگر حقوقی متخصص در قوانین ایران هستید. بر اساس متن قرارداد زیر، به سوال کاربر با دقت و به صورت رسمی پاسخ دهید. فقط از اطلاعات موجود در متن استفاده کنید. پاسخ شما باید دقیق، مبتنی بر واقعیت و متمرکز بر بندهای قرارداد باشد. از ارائه مشاوره حقوقی خودداری کنید.

        **متن قرارداد (زمینه):**
        ---
        {context}
        ---

        **سوال کاربر:** {question}

        **پاسخ شما (به فارسی رسمی و با ارجاع به مواد قرارداد در صورت امکان):**
        """

        tasks = [
            self._query_model_async(PRIMARY_LEGAL_ANALYST_MODEL, base_prompt),
            self._query_model_async(SECONDARY_VERIFICATION_MODEL, base_prompt),
        ]
        analyst_response, verifier_response = await asyncio.gather(*tasks)

        if analyst_response is None and verifier_response is None:
            return (
                "خطا در ارتباط با مدل‌های تحلیل. لطفاً مطمئن شوید سرور Ollama "
                f"در حال اجراست و مدل‌های {PRIMARY_LEGAL_ANALYST_MODEL} و "
                f"{SECONDARY_VERIFICATION_MODEL} نصب شده‌اند."
            )

        if analyst_response is None:
            return verifier_response
        if verifier_response is None:
            return analyst_response

        synthesis_prompt = f"""
        **دستورالعمل:** شما یک متخصص ارشد حقوقی و ویراستار نهایی هستید. دو تحلیل از یک قرارداد توسط دو مدل هوش مصنوعی در زیر ارائه شده است. وظیفه شما ترکیب این دو تحلیل، حل هرگونه اختلاف، و ارائه یک پاسخ نهایی، دقیق، و محافظه‌کارانه به زبان فارسی رسمی است. پاسخ نهایی باید اعداد، تاریخ‌ها و جزئیات کلیدی را با دقت حفظ کند و هرگونه اطلاعات متناقض یا حدسی را حذف نماید. پاسخ نهایی باید یکپارچه و منسجم باشد.

        **تحلیل مدل اول (تحلیلگر اصلی):**
        ---
        {analyst_response}
        ---

        **تحلیل مدل دوم (تاییدکننده):**
        ---
        {verifier_response}
        ---

        **سوال اصلی کاربر:** {question}

        **پاسخ نهایی ترکیبی و ویراسته (به فارسی رسمی و حقوقی):**
        """

        final_answer = await self._query_model_async(SYNTHESIZER_MODEL, synthesis_prompt)
        if final_answer is None:
            return analyst_response
        return final_answer
