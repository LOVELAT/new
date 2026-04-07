from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import gradio as gr
from gradio import ChatMessage


class ChatInterface:
    """Gradio chat wrapper for the baggage agent."""

    def __init__(self, agent, tools_dict: Dict[str, object]) -> None:
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.current_thread_id: Optional[str] = None
        self.original_file_path: Optional[str] = None
        self.display_file_path: Optional[str] = None

    def handle_upload(self, file_path: str) -> Optional[str]:
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())
        saved_path = self.upload_dir / f"upload_{timestamp}{source.suffix.lower()}"
        shutil.copy2(source, saved_path)

        self.original_file_path = str(saved_path)
        self.display_file_path = str(saved_path)
        return self.display_file_path

    def add_message(
        self,
        message: str,
        display_image: Optional[str],
        history: List[dict],
    ) -> Tuple[List[dict], gr.Textbox]:
        history = history or []
        image_path = self.original_file_path or display_image

        if image_path:
            history.append({"role": "user", "content": {"path": image_path}})
        if message:
            history.append({"role": "user", "content": message})

        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self,
        message: str,
        display_image: Optional[str],
        chat_history: List[ChatMessage],
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        chat_history = chat_history or []

        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        image_path = self.original_file_path or display_image

        if image_path:
            messages.append({"role": "user", "content": f"image_path: {image_path}"})
            messages.append({"role": "user", "content": self._image_content(image_path)})

        if message:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message}],
                }
            )

        try:
            for event in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": self.current_thread_id}},
            ):
                if "process" in event:
                    content = event["process"]["messages"][-1].content
                    if content:
                        chat_history.append(ChatMessage(role="assistant", content=content))
                        yield chat_history, self.display_file_path, ""

                if "execute" in event:
                    for tool_message in event["execute"]["messages"]:
                        formatted_result = self._format_tool_result(tool_message.content)
                        chat_history.append(
                            ChatMessage(
                                role="assistant",
                                content=formatted_result,
                                metadata={"title": f"Tool: {tool_message.name}"},
                            )
                        )
                        yield chat_history, self.display_file_path, ""
        except Exception as exc:  # pragma: no cover
            chat_history.append(
                ChatMessage(
                    role="assistant",
                    content=f"Error: {exc}",
                    metadata={"title": "Error"},
                )
            )
            yield chat_history, self.display_file_path, ""

    @staticmethod
    def _format_tool_result(raw_content: str) -> str:
        try:
            parsed = json.loads(raw_content)
        except (TypeError, json.JSONDecodeError):
            return str(raw_content)

        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        return str(parsed)

    @staticmethod
    def _image_content(image_path: str) -> List[dict]:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/jpeg"
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            }
        ]


def create_demo(agent, tools_dict: Dict[str, object]) -> gr.Blocks:
    interface = ChatInterface(agent, tools_dict)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Baggage Agent")
        gr.Markdown("Airport baggage X-ray reasoning agent scaffold modeled after MedRAX.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    [],
                    type="messages",
                    height=700,
                    label="Agent",
                    show_label=True,
                )
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="输入你的安检问题，或上传一张行李 X 光图像...",
                    container=False,
                )

            with gr.Column(scale=2):
                image_display = gr.Image(
                    label="Baggage X-ray",
                    type="filepath",
                    height=420,
                    container=True,
                )
                upload_button = gr.UploadButton(
                    "Upload Image",
                    file_types=["image"],
                )
                with gr.Accordion("Toolbar", open=True):
                    gr.Markdown("暂未配置工具。后续可在 `baggage/tools` 中注册后接入这里。")
                    if tools_dict:
                        gr.JSON(sorted(tools_dict.keys()))
                    else:
                        gr.Textbox(value="No tools configured.", interactive=False, label="Status")

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    new_thread_btn = gr.Button("New Thread")

        def clear_chat() -> Tuple[List[dict], Optional[str]]:
            interface.original_file_path = None
            interface.display_file_path = None
            return [], None

        def new_thread() -> Tuple[List[dict], Optional[str]]:
            interface.current_thread_id = str(time.time())
            return [], interface.display_file_path

        def handle_file_upload(file) -> Optional[str]:
            return interface.handle_upload(file.name)

        chat_msg = txt.submit(
            interface.add_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, txt],
        )
        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        upload_button.upload(handle_file_upload, inputs=upload_button, outputs=image_display)
        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo
