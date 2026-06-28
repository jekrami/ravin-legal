import gradio as gr
import logging
from datetime import datetime

from config import __version__
from llm_handler import LlmHandler
from services import AnalysisService, ChatService, DocumentService, get_session_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

session_manager = get_session_manager()
document_service = DocumentService(session_manager)
chat_service = ChatService(session_manager)
analysis_service = AnalysisService(session_manager)
llm_handler = LlmHandler()

PASS_TITLES_FA = {
    1: "مرحله ۱ — تحلیل ساختاری و شکلی",
    2: "مرحله ۲ — طبقه‌بندی موضوع و تعهدات",
    3: "مرحله ۳ — مدت زمان و شرایط مالی",
    4: "مرحله ۴ — فسخ، نقض و جبران خسارت",
    5: "مرحله ۵ — ریسک، ابهام و خلأ قانونی",
    6: "مرحله ۶ — پیش‌فرض‌های قانونی ایران",
    7: "مرحله ۷ — مدل مفهومی حقوقی",
    8: "مرحله ۸ — ارزیابی نهایی قابلیت اجرا",
}


def init_session(session_id):
    """Assign a session ID and load per-session chat + document list."""
    new_id = session_manager.get_or_create_session(session_id)
    history = chat_service.load_history(new_id)
    doc_list = document_service.get_documents_list(new_id)
    return new_id, history, doc_list


def create_progress_html(current_step, total_steps, current_title):
    progress_percent = (current_step / total_steps) * 100
    return f"""
    <div style="direction: rtl; text-align: right; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="color: #1976d2;">📊 پیشرفت تحلیل عمیق حقوقی</h3>
        <div style="background: #f5f5f5; border-radius: 10px; padding: 20px; margin: 10px 0;">
            <div style="margin-bottom: 10px;">
                <strong>مرحله {current_step} از {total_steps}</strong>
            </div>
            <div style="background: #e0e0e0; border-radius: 5px; height: 30px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #4caf50, #8bc34a); height: 100%; width: {progress_percent}%; transition: width 0.3s;"></div>
            </div>
            <div style="margin-top: 10px; color: #555;">
                {current_title}
            </div>
        </div>
    </div>
    """


def format_analysis_results(results):
    output = "# 📋 گزارش تحلیل عمیق حقوقی\n\n"
    output += f"**تاریخ تحلیل:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    output += "---\n\n"

    for pass_id in sorted(results.keys()):
        result = results[pass_id]
        title_fa = result.get("title_fa", PASS_TITLES_FA.get(pass_id, ""))
        output += f"## {title_fa}\n\n"
        output += f"**{result['title']}**\n\n"
        output += f"{result['output']}\n\n"
        output += "---\n\n"

    return output


def run_deep_analysis_with_progress(
    session_id,
    pdf_file,
    pass1, pass2, pass3, pass4, pass5, pass6, pass7, pass8,
):
    session_id = session_manager.get_or_create_session(session_id)

    if pdf_file is None:
        yield session_id, None, "❌ لطفاً یک فایل PDF را آپلود کنید.", ""
        return

    selected_passes = [pass1, pass2, pass3, pass4, pass5, pass6, pass7, pass8]
    if not any(selected_passes):
        yield session_id, None, "❌ لطفاً حداقل یک مرحله تحلیل را انتخاب کنید.", ""
        return

    try:
        yield session_id, None, "📄 در حال استخراج متن از PDF...", ""
        selected_pass_ids = [i + 1 for i, selected in enumerate(selected_passes) if selected]
        results = None
        total_steps = len(selected_pass_ids)

        for event in analysis_service.run_analysis(
            session_id,
            pdf_file.name,
            selected_pass_ids,
            PASS_TITLES_FA,
        ):
            if event["event"] == "progress":
                pass_id = event["pass_id"]
                title_fa = PASS_TITLES_FA.get(pass_id, event["title"])
                progress_html = create_progress_html(event["step"], event["total"], title_fa)
                status_msg = f"🔍 در حال اجرای {title_fa}..."
                yield session_id, progress_html, status_msg, ""
            elif event["event"] == "complete":
                results = event["results"]

        json_file, presentation_file = analysis_service.save_results(session_id, results)

        progress_html = create_progress_html(
            total_steps, total_steps, "✅ در حال تولید گزارش فارسی..."
        )
        yield session_id, progress_html, "📝 در حال تبدیل نتایج به گزارش فارسی...", ""

        try:
            persian_presentation = analysis_service.generate_presentation(
                json_file, presentation_file
            )
            progress_html = create_progress_html(total_steps, total_steps, "✅ تحلیل کامل شد!")
            final_status = (
                f"✅ تحلیل عمیق با موفقیت کامل شد!\n"
                f"📁 نتایج JSON در فایل {json_file} ذخیره شد.\n"
                f"📄 گزارش فارسی در فایل {presentation_file} ذخیره شد."
            )
            yield session_id, progress_html, final_status, persian_presentation
        except Exception as e:
            logger.exception("Presentation generation failed")
            progress_html = create_progress_html(total_steps, total_steps, "✅ تحلیل کامل شد!")
            final_report = format_analysis_results(results)
            yield (
                session_id,
                progress_html,
                f"⚠️ تحلیل کامل شد اما خطا در تولید گزارش فارسی: {e}\n"
                f"📁 نتایج در فایل {json_file} ذخیره شد.",
                final_report,
            )

    except Exception as e:
        logger.exception("Deep analysis failed")
        yield session_id, None, f"❌ خطا در تحلیل: {str(e)}", ""


def process_uploaded_file(session_id, file):
    session_id = session_manager.get_or_create_session(session_id)
    status, doc_list = document_service.process_uploaded_file(session_id, file)
    return session_id, status, doc_list


def delete_document(session_id, doc_name):
    session_id = session_manager.get_or_create_session(session_id)
    status, doc_list = document_service.delete_document(session_id, doc_name)
    return session_id, status, doc_list


def clear_all_documents_confirmed(session_id):
    session_id = session_manager.get_or_create_session(session_id)
    document_service.clear_all_documents(session_id)
    chat_service.clear_history(session_id)
    doc_list = document_service.get_documents_list(session_id)
    return session_id, "✅ همه اسناد با موفقیت حذف شدند.", doc_list, gr.update(visible=False)


def refresh_documents(session_id):
    session_id = session_manager.get_or_create_session(session_id)
    return session_id, document_service.get_documents_list(session_id)


def clear_chat(session_id):
    session_id = session_manager.get_or_create_session(session_id)
    chat_service.clear_history(session_id)
    return session_id, []


async def chat_interface_fn(message, history, session_id):
    session_id = session_manager.get_or_create_session(session_id)

    if history is None:
        history = []

    with session_manager.session_lock(session_id) as state:
        has_index = state.rag_pipeline.index is not None

    if not has_index:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": "هنوز هیچ سندی در پایگاه دانش وجود ندارد. لطفاً ابتدا یک فایل را آپلود کنید.",
        })
        yield session_id, history
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "در حال جستجو و تحلیل..."})
    yield session_id, history

    with session_manager.session_lock(session_id) as state:
        retrieved_context, sources = state.rag_pipeline.retrieve(message, return_sources=True)

    if not retrieved_context:
        history[-1] = {
            "role": "assistant",
            "content": "نتوانستم اطلاعات مرتبطی در اسناد موجود پیدا کنم.",
        }
        yield session_id, history
        return

    final_answer = await llm_handler.get_synthesized_answer(message, retrieved_context)

    unique_sources = list(set(sources))
    source_info = "\n\n---\n📄 **منابع استفاده شده:**\n"
    for src in unique_sources:
        source_info += f"• {src}\n"

    final_answer_with_sources = final_answer + source_info
    history[-1] = {"role": "assistant", "content": final_answer_with_sources}
    chat_service.append_turn(session_id, message, final_answer_with_sources)
    yield session_id, history


custom_css = """
footer {display: none !important}
.rtl-title {
    direction: rtl;
    text-align: right;
}
.rtl-title h1, .rtl-title h2, .rtl-title h3, .rtl-title h4 {
    text-align: right !important;
    direction: rtl !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    session_id = gr.State(value=None)

    gr.Markdown(
        f"""
        <div class="rtl-title">

        # ⚖️ تحلیلگر قراردادهای حقوقی راوین

        **نسخه {__version__}**

        این ابزار به شما کمک می‌کند تا اسناد حقوقی خود را با استفاده از هوش مصنوعی تحلیل کرده و به سوالات خود پاسخ دهید (تمامی حقوق متعلق به شرکت راوین می‌باشد).

        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("💬 چت با اسناد"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('<div class="rtl-title"><h3>📁 مدیریت اسناد</h3></div>')

                    with gr.Group():
                        file_uploader = gr.File(label="آپلود فایل PDF جدید", file_types=[".pdf"])
                        upload_status = gr.Textbox(label="وضعیت پردازش", interactive=False, rtl=True, lines=3)

                    with gr.Group():
                        documents_display = gr.Markdown(value="هیچ سندی بارگذاری نشده است.", rtl=True)
                        refresh_docs_btn = gr.Button("🔄 بروزرسانی لیست", size="sm")

                    with gr.Group():
                        gr.Markdown('<div class="rtl-title"><h4>حذف سند</h4></div>')
                        doc_name_input = gr.Textbox(
                            label="نام سند برای حذف",
                            placeholder="نام دقیق فایل را وارد کنید...",
                            rtl=True,
                        )
                        with gr.Row():
                            delete_btn = gr.Button("🗑️ حذف سند", variant="stop", size="sm")
                            clear_all_btn = gr.Button("⚠️ حذف همه", variant="stop", size="sm")
                        delete_status = gr.Textbox(label="وضعیت حذف", interactive=False, rtl=True, lines=2)

                    with gr.Group(visible=False) as confirm_dialog:
                        gr.Markdown('<div class="rtl-title"><h4>⚠️ تأیید حذف همه اسناد</h4></div>')
                        gr.Markdown(
                            "**آیا مطمئن هستید که می‌خواهید همه اسناد را حذف کنید؟**\n\n"
                            "این عملیات قابل بازگشت نیست!",
                            rtl=True,
                        )
                        with gr.Row():
                            confirm_yes_btn = gr.Button("✅ بله، همه را حذف کن", variant="stop", size="sm")
                            confirm_no_btn = gr.Button("❌ انصراف", variant="secondary", size="sm")

                with gr.Column(scale=2):
                    gr.Markdown('<div class="rtl-title"><h3>💬 چت با اسناد</h3></div>')
                    chatbot = gr.Chatbot(label="مکالمه", height=500, rtl=True)
                    msg = gr.Textbox(
                        label="سوال خود را بپرسید",
                        placeholder="سوال خود را در مورد قراردادها اینجا تایپ کنید...",
                        rtl=True,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("ارسال", variant="primary")
                        clear_chat_btn = gr.Button("پاک کردن گفتگو")

        with gr.Tab("🔍 تحلیل عمیق حقوقی"):
            with gr.Row():
                with gr.Column(scale=1):
                    deep_analysis_file = gr.File(
                        label="آپلود فایل PDF قرارداد",
                        file_types=[".pdf"],
                    )

                    gr.Markdown('<div class="rtl-title"><h4>انتخاب مراحل تحلیل</h4></div>')
                    with gr.Group():
                        pass1_checkbox = gr.Checkbox(label=PASS_TITLES_FA[1], value=True)
                        pass2_checkbox = gr.Checkbox(label=PASS_TITLES_FA[2], value=True)
                        pass3_checkbox = gr.Checkbox(label=PASS_TITLES_FA[3], value=True)
                        pass4_checkbox = gr.Checkbox(label=PASS_TITLES_FA[4], value=True)
                        pass5_checkbox = gr.Checkbox(label=PASS_TITLES_FA[5], value=True)
                        pass6_checkbox = gr.Checkbox(label=PASS_TITLES_FA[6], value=True)
                        pass7_checkbox = gr.Checkbox(label=PASS_TITLES_FA[7], value=True)
                        pass8_checkbox = gr.Checkbox(label=PASS_TITLES_FA[8], value=True)

                    deep_analysis_btn = gr.Button("🚀 شروع تحلیل عمیق", variant="primary", size="lg")
                    deep_analysis_status = gr.Textbox(
                        label="وضعیت", interactive=False, rtl=True, lines=3
                    )

                with gr.Column(scale=2):
                    deep_analysis_progress = gr.HTML(label="پیشرفت تحلیل")

            with gr.Row():
                deep_analysis_output = gr.Markdown(
                    label="نتایج تحلیل",
                    rtl=True,
                    value="نتایج تحلیل اینجا نمایش داده می‌شود...",
                )

    demo.load(
        fn=init_session,
        inputs=[session_id],
        outputs=[session_id, chatbot, documents_display],
    )

    file_uploader.upload(
        fn=process_uploaded_file,
        inputs=[session_id, file_uploader],
        outputs=[session_id, upload_status, documents_display],
    )

    refresh_docs_btn.click(
        fn=refresh_documents,
        inputs=[session_id],
        outputs=[session_id, documents_display],
    )

    delete_btn.click(
        fn=delete_document,
        inputs=[session_id, doc_name_input],
        outputs=[session_id, delete_status, documents_display],
    )

    clear_all_btn.click(fn=lambda: gr.update(visible=True), inputs=None, outputs=confirm_dialog)

    confirm_yes_btn.click(
        fn=clear_all_documents_confirmed,
        inputs=[session_id],
        outputs=[session_id, delete_status, documents_display, confirm_dialog],
    )

    confirm_no_btn.click(
        fn=lambda: (gr.update(visible=False), "عملیات لغو شد."),
        inputs=None,
        outputs=[confirm_dialog, delete_status],
    )

    msg.submit(chat_interface_fn, [msg, chatbot, session_id], [session_id, chatbot]).then(
        lambda: "", None, msg
    )
    submit_btn.click(chat_interface_fn, [msg, chatbot, session_id], [session_id, chatbot]).then(
        lambda: "", None, msg
    )
    clear_chat_btn.click(clear_chat, [session_id], [session_id, chatbot], queue=False)

    deep_analysis_btn.click(
        fn=run_deep_analysis_with_progress,
        inputs=[
            session_id,
            deep_analysis_file,
            pass1_checkbox,
            pass2_checkbox,
            pass3_checkbox,
            pass4_checkbox,
            pass5_checkbox,
            pass6_checkbox,
            pass7_checkbox,
            pass8_checkbox,
        ],
        outputs=[session_id, deep_analysis_progress, deep_analysis_status, deep_analysis_output],
    )


if __name__ == "__main__":
    demo.launch()
