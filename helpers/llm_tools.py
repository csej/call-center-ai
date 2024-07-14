from azure.communication.callautomation.aio import CallAutomationClient
from helpers.call_utils import ContextEnum as CallContextEnum, handle_play_text
from helpers.config import CONFIG
from helpers.llm_utils import function_schema
from helpers.logging import logger
from inspect import getmembers, isfunction
from models.call import CallStateModel
from models.message import (
    ActionEnum as MessageActionEnum,
    MessageModel,
    PersonaEnum as MessagePersonaEnum,
    StyleEnum as MessageStyleEnum,
)
from html import escape
from models.reminder import ReminderModel
from models.training import TrainingModel
from openai.types.chat import ChatCompletionToolParam
from pydantic import ValidationError
from typing import Awaitable, Callable, Annotated, Literal
from typing_extensions import TypedDict
import asyncio
import json
from datetime import datetime

_search = CONFIG.ai_search.instance()
_sms = CONFIG.sms.instance()


class AvailabilityCalenderDict(TypedDict):
    date: str
    time: str


class LlmPlugins:
    call: CallStateModel
    client: CallAutomationClient
    post_callback: Callable[[CallStateModel], Awaitable[None]]
    style: MessageStyleEnum = MessageStyleEnum.NONE
    tts_callback: Callable[[str, MessageStyleEnum], Awaitable[None]]

    def __init__(
        self,
        call: CallStateModel,
        client: CallAutomationClient,
        post_callback: Callable[[CallStateModel], Awaitable[None]],
        tts_callback: Callable[[str, MessageStyleEnum], Awaitable[None]],
    ):
        self.call = call
        self.client = client
        self.post_callback = post_callback
        self.tts_callback = tts_callback

    async def book_meeting(
        self,
        customer_response: Annotated[
            str,
            """
            Phrase used to confirm the action, in the same language as the client. This phrase will be spoken to the user.
            # Rules
            - Action should be rephrased in the present tense
            - Must be in a single sentence
            # Examples
            - "I'm trying to book it."
            - "Im' booking it"
            """,
        ],
        reason: Annotated[
            str,
            """
            The reason why the client wants a meeting.

            # Rules
            - The reason should be in the context of the banking industry

            # Example
            - "A new credit loan."
            - "A mortgage simulation."
            """,
        ],
        start_timestamp: Annotated[
            str,
            """
            Date and time in 'YYYY-MM-DDTHH:MM' format of the start of the meeting.
            """,
        ],
        end_timestamp: Annotated[
            str,
            """
            Date and time in 'YYYY-MM-DDTHH:MM' format of the end of the meeting.
            """,
        ],
        # slot: Annotated[
        #     AvailabilityCalenderDict,
        #     """
        #     The slot at list of dates and times to check for availability.
        #     # Rules
        #     - The date should be in the format 'YYYY-MM-DD'
        #     - The time should be in the format 'HH:MM'
        #     - The time should be in the 24-hour format
        #     # Example
        #     [{"date": "2022-02-15", "time": "10:00"}, {"date": "2022-02-17", "time": "14:00"}]
        #     """,
        # ],
    ) -> str:
        """
        Use this to confirm the meeting.

        # Behavior
        1. Get the availability for the requested date
        2. Get the reason of the meeting
        3. Return a confirmation message

        # Rules
        - Use this every time a new meeting is requested before booking it

        # Usage examples
        - The client wants to book an new meeting
        - A customer ask questions about an availability
        """
        await self.tts_callback(customer_response, self.style)
        return "The meeting is booked."

    async def get_advisor_available_slot(
        self,
        customer_response: Annotated[
            str,
            """
            Phrase used to confirm the action, in the same language as the client. This phrase will be spoken to the user.
            # Rules
            - Action should be rephrased in the present tense
            - Must be in a single sentence
            # Examples
            - "I'm trying to book it."
            - "Im' booking it"
            """,
        ],
        start_timestamp: Annotated[
            str,
            """
            Date and time in 'YYYY-MM-DDTHH:MM' format of the start of the period requested by the client for the appointment.
            """,
        ],
        end_timestamp: Annotated[
            str,
            """
            Date and time in 'YYYY-MM-DDTHH:MM' format of the end of the period requested by the client for the appointment.
            """,
        ],
    ) -> str:
        """
        Use this if you need to get the banking advisor available slots.

        The function returns the available slot.

        For example 'The advisor is available from 2024-07-11T16:00 to 2024-07-11T16:30'
        """

        with open("tests/config_slots.json") as f_in:
            conf = json.load(f_in)
        slots = [
            {
                "startTime": datetime.strptime(
                    slot["startTime"]["dateTime"], "%Y-%m-%dT%H:%M"
                ),
                "endTime": datetime.strptime(
                    slot["endTime"]["dateTime"], "%Y-%m-%dT%H:%M"
                ),
            }
            for slot in conf["advisor"]["availability"]["timeSlots"]
            if slot["startTime"]["dateTime"] >= start_timestamp
            and slot["endTime"]["dateTime"] <= end_timestamp
        ]
        if len(slots) > 0:
            return "The advisor is available from '%s' to '%s'." % (
                slots[0]["startTime"].strftime("%a %d %b %Y %H:%M"),
                slots[0]["endTime"].strftime("%a %d %b %Y %H:%M"),
            )
        else:
            return "The advisor is not available at this period."

    async def end_call(self) -> str:
        """
        Use this if the customer said they want to end the call.

        # Behavior
        1. Hangup the call for everyone
        2. The call with Assistant is ended

        # Rules
        - Requires an explicit verbal validation from the customer
        - Never use this action directly after a recall

        # Usage examples
        - All participants are satisfied and agree to end the call
        - Customer said 'bye bye'
        """
        await handle_play_text(
            call=self.call,
            client=self.client,
            context=CallContextEnum.GOODBYE,
            text=await CONFIG.prompts.tts.goodbye(self.call),
        )
        return "Call ended"

    async def send_sms(
        self,
        customer_response: Annotated[
            str,
            """
            Phrase used to confirm the update, in the same language as the customer. This phrase will be spoken to the user.

            # Rules
            - Action should be rephrased in the present tense
            - Must be in a single sentence

            # Examples
            - 'I am sending a SMS to your phone number.'
            - 'I am texting you the information right now.'
            - 'SMS with the details is sent.'
            """,
        ],
        message: Annotated[
            str,
            "The message to send to the customer.",
        ],
    ) -> str:
        """
        Use when there is a real need to send a SMS to the customer.

        # Usage examples
        - Ask a question, if the call quality is bad
        - Confirm a detail like a reference number, if there is a misunderstanding
        - Send a confirmation, if the customer wants to have a written proof
        """
        await self.tts_callback(customer_response, self.style)
        #        success = await _sms.asend(
        #            content=message,
        #            phone_number=self.call.initiate.phone_number,
        #        )
        success = True
        if not success:
            return "Failed to send SMS"
        self.call.messages.append(
            MessageModel(
                action=MessageActionEnum.SMS,
                content=message,
                persona=MessagePersonaEnum.ASSISTANT,
            )
        )
        return "SMS sent"

    async def speech_speed(
        self,
        customer_response: Annotated[
            str,
            """
            Phrase used to confirm the update, in the same language as the customer. This phrase will be spoken to the user.

            # Rules
            - Action should be rephrased in the present tense
            - Must be in a single sentence

            # Examples
            - 'I am slowing down the speech.'
            - 'I am speeding up the voice.'
            - 'My voice is now faster.'
            """,
        ],
        speed: Annotated[
            float,
            "The new speed of the voice. Should be between 0.75 and 1.25, where 1.0 is the normal speed.",
        ],
    ) -> str:
        """
        Use this if the customer wants to change the speed of the voice.

        # Behavior
        1. Update the voice speed
        2. Return a confirmation message

        # Usage examples
        - Speed up or slow down the voice
        - Trouble understanding the voice because it is too fast or too slow
        """
        # Clamp speed between min and max
        speed = max(0.75, min(speed, 1.25))
        # Update voice
        initial_speed = self.call.initiate.prosody_rate
        self.call.initiate.prosody_rate = speed
        # Customer confirmation (with new speed)
        await self.tts_callback(customer_response, self.style)
        # LLM confirmation
        return f"Voice speed set to {speed} (was {initial_speed})"

    async def speech_lang(
        self,
        customer_response: Annotated[
            str,
            """
            Phrase used to confirm the update, in the new selected language. This phrase will be spoken to the user.

            # Rules
            - Action should be rephrased in the present tense
            - Must be in a single sentence

            # Examples
            - For de-DE, 'Ich spreche jetzt auf Deutsch.'
            - For en-ES, 'Espero que me entiendas mejor en español.'
            - For fr-FR, 'Cela devrait être mieux en français.'
            """,
        ],
        lang: Annotated[
            str,
            """
            The new language of the conversation.

            # Available short codes
            {% for available in call.initiate.lang.availables %}
            - {{ available.short_code }} ({{ available.pronunciations[0] }})
            {% endfor %}

            # Data format
            short code

            # Examples
            - 'en-US'
            - 'es-ES'
            - 'zh-CN'
            """,
        ],
    ) -> str:
        """
        Use this if the customer wants to speak in another language.

        # Behavior
        1. Update the conversation language
        2. Return a confirmation message

        # Usage examples
        - A participant wants to speak in another language
        - Customer made a mistake in the language selection
        - Trouble understanding the voice in the current language
        """
        if not any(
            lang == available.short_code
            for available in self.call.initiate.lang.availables
        ):  # Check if lang is available
            return f"Language {lang} not available"

        # Update lang
        initial_lang = self.call.lang.short_code
        self.call.lang = lang
        # Customer confirmation (with new language)
        await self.tts_callback(customer_response, self.style)
        # LLM confirmation
        return f"Voice language set to {lang} (was {initial_lang})"

    @staticmethod
    async def to_openai(call: CallStateModel) -> list[ChatCompletionToolParam]:
        return await asyncio.gather(
            *[
                function_schema(type, call=call)
                for name, type in getmembers(LlmPlugins, isfunction)
                if not name.startswith("_") and name != "to_openai"
            ]
        )
