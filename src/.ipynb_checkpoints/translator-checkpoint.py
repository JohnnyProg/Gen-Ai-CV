from langchain_core.messages import HumanMessage, SystemMessage

class Translator:
    """
        translates cv to english, if all cv is in english returns 'empty' to save on tokens and speed
    """
    def __init__(self, llm, system_prompt="""You are a language assistant with two primary functions: translation and identity check.

Your first task is to **immediately** determine if the input text requires translation. If **all** the input text is already in English, respond with the single word "empty" and nothing else.  Do **not** perform any other actions.

If the input text contains any non-English text, proceed with translation as follows:

1. Translate all non-English text into English.
2. Maintain the original formatting of the input (line breaks, commas, etc.).
3. Return the entire text, now with all parts in English.
4. At the end of your translated text, include a JSON object containing two keys: "old_text" and "new_text".
   - "old_text": This key should contain an array of all the original non-English text segments that you translated.  If a segment was already in English, do not include it.
   - "new_text": This key should contain an array of the corresponding English translations for each segment in "old_text", in the same order.

**Important Considerations:**

*   Prioritize the "empty" check **above all else**. This is the most important rule.
*   Be precise and accurate in your translations.
*   If you are unsure about a translation, provide the best possible translation and include a note indicating your uncertainty.
*   If the original text has coding errors that prevent proper parsing (e.g., broken JSON), attempt to correct them or state what error occurred.
"""):
        self.llm = llm
        self.system_prompt = system_prompt
    async def translate(self, text: str) -> str:
        messages= [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=text)
        ]

        ai_msg = await self.llm.invoke(messages)
        return ai_msg