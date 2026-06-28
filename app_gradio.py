import gradio as gr
import json
import logging
import os
from datetime import datetime

from config import __version__
from document_processor import extract_full_text, process_document
from legal_analyzer.orchestrator import run_deep_analysis
from legal_analyzer.presentation import generate_persian_presentation
from llm_handler import LlmHandler
from rag_pipeline import RagPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Global State for Shared Knowledge Base ---
rag_pipeline = RagPipeline()
documents_metadata = {}

# --- File Paths for Persistence ---
VECTOR_DB_PATH = "vector_db.pkl"
CHAT_HISTORY_PATH = "rag_chat_history.json"
DOCUMENTS_METADATA_PATH = "documents_metadata.json"

# --- Initialization ---
if os.path.exists(VECTOR_DB_PATH):
    rag_pipeline.load(VECTOR_DB_PATH)
if os.path.exists(DOCUMENTS_METADATA_PATH):
    with open(DOCUMENTS_METADATA_PATH, "r", encoding="utf-8") as f:
        documents_metadata = json.load(f)

llm_handler = LlmHandler()

# Persian translations for analysis passes
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


def load_chat_history():
    """Load persisted chat history into Gradio messages format."""
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []

    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load chat history: %s", e)
        return []

    messages = []
    for entry in stored:
        if "user" in entry:
            messages.append({"role": "user", "content": entry["user"]})
        if "assistant" in entry:
            messages.append({"role": "assistant", "content": entry["assistant"]})
    return messages


def append_to_global_chat_history(user_query, assistant_answer):
    """Appends a single turn to the global chat history file."""
    history_entry = {"user": user_query, "assistant": assistant_answer}

    full_history = []
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            try:
                full_history = json.load(f)
            except json.JSONDecodeError:
                pass

    full_history.append(history_entry)

    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(full_history, f, ensure_ascii=False, indent=4)


def save_metadata():
    with open(DOCUMENTS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(documents_metadata, f, ensure_ascii=False, indent=4)


def create_progress_html(current_step, total_steps, current_title):
    """Create HTML progress visualization."""
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
    """Format analysis results for display."""
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
    pdf_file, pass1, pass2, pass3, pass4, pass5, pass6, pass7, pass8
):
    """Run deep legal analysis with progress updates via orchestrator."""
    if pdf_file is None:
        yield None, "❌ لطفاً یک فایل PDF را آپلود کنید.", ""
        return

    selected_passes = [pass1, pass2, pass3, pass4, pass5, pass6, pass7, pass8]
    if not any(selected_passes):
        yield None, "❌ لطفاً حداقل یک مرحله تحلیل را انتخاب کنید.", ""
        return

    try:
        yield None, "📄 در حال استخراج متن از PDF...", ""
        contract_text = extract_full_text(pdf_file.name)

        if not contract_text.strip():
            yield None, "❌ خطا: نتوانستم متنی از PDF استخراج کنم.", ""
            return

        selected_pass_ids = [i + 1 for i, selected in enumerate(selected_passes) if selected]
        results = None
        total_steps = len(selected_pass_ids)

        for event in run_deep_analysis(
            contract_text,
            selected_pass_ids=selected_pass_ids,
            title_fa_map=PASS_TITLES_FA,
        ):
            if event["event"] == "progress":
                pass_id = event["pass_id"]
                title_fa = PASS_TITLES_FA.get(pass_id, event["title"])
                progress_html = create_progress_html(event["step"], event["total"], title_fa)
                status_msg = f"🔍 در حال اجرای {title_fa}..."
                yield progress_html, status_msg, ""
            elif event["event"] == "complete":
                results = event["results"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"deep_analysis_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        progress_html = create_progress_html(
            total_steps, total_steps, "✅ در حال تولید گزارش فارسی..."
        )
        yield progress_html, "📝 در حال تبدیل نتایج به گزارش فارسی...", ""

        try:
            persian_presentation = generate_persian_presentation(
                analysis_json_path=json_file,
                output_path="presentation_fa.txt",
            )
            progress_html = create_progress_html(total_steps, total_steps, "✅ تحلیل کامل شد!")
            final_status = (
                f"✅ تحلیل عمیق با موفقیت کامل شد!\n"
                f"📁 نتایج JSON در فایل {json_file} ذخیره شد.\n"
                f"📄 گزارش فارسی در فایل presentation_fa.txt ذخیره شد."
            )
            yield progress_html, final_status, persian_presentation
        except Exception as e:
            logger.exception("Presentation generation failed")
            progress_html = create_progress_html(total_steps, total_steps, "✅ تحلیل کامل شد!")
            final_report = format_analysis_results(results)
            yield (
                progress_html,
                f"⚠️ تحلیل کامل شد اما خطا در تولید گزارش فارسی: {e}\n"
                f"📁 نتایج در فایل {json_file} ذخیره شد.",
                final_report,
            )

    except Exception as e:
        logger.exception("Deep analysis failed")
        yield None, f"❌ خطا در تحلیل: {str(e)}", ""


def get_documents_list():
    """Get formatted list of loaded documents."""
    stats = rag_pipeline.get_document_stats()
    if not stats:
        return "هیچ سندی بارگذاری نشده است."

    doc_list = "📚 **اسناد بارگذاری شده:**\n\n"
    for i, (doc_name, chunk_count) in enumerate(stats.items(), 1):
        doc_list += f"{i}. **{doc_name}** ({chunk_count} بخش)\n"

    return doc_list


def delete_document(doc_name):
    """Delete a specific document from the knowledge base."""
    if not doc_name or doc_name.strip() == "":
        return "لطفاً نام سند را وارد کنید.", get_documents_list()

    success = rag_pipeline.delete_document(doc_name.strip())

    if success:
        if doc_name.strip() in documents_metadata:
            del documents_metadata[doc_name.strip()]
        save_metadata()
        rag_pipeline.save(VECTOR_DB_PATH)
        return f"✅ سند `{doc_name.strip()}` با موفقیت حذف شد.", get_documents_list()

    return f"❌ سند `{doc_name.strip()}` یافت نشد.", get_documents_list()


def clear_all_documents_confirmed():
    """Clear all documents from the knowledge base (after confirmation)."""
    rag_pipeline.documents = []
    rag_pipeline.document_sources = []
    rag_pipeline.index = None
    documents_metadata.clear()

    if os.path.exists(VECTOR_DB_PATH):
        os.remove(VECTOR_DB_PATH)
    if os.path.exists(DOCUMENTS_METADATA_PATH):
        os.remove(DOCUMENTS_METADATA_PATH)

    return "✅ همه اسناد با موفقیت حذف شدند.", get_documents_list(), gr.update(visible=False)


def process_uploaded_file(file):
    """Handles the file upload and processing."""
    if file is None:
        return "لطفاً یک فایل را برای پردازش آپلود کنید.", get_documents_list()

    try:
        text_chunks = process_document(file.name)
        if not text_chunks:
            return "خطا: نتوانستم هیچ متنی از فایل استخراج کنم.", get_documents_list()

        source_name = os.path.basename(file.name)
        added_count = rag_pipeline.add_documents(text_chunks, source_name=source_name)
        if added_count == 0:
            return "خطا: هیچ بخش معتبری از فایل استخراج نشد.", get_documents_list()

        rag_pipeline.save(VECTOR_DB_PATH)
        documents_metadata[source_name] = {
            "processed": True,
            "chunks": added_count,
            "timestamp": datetime.now().isoformat(),
        }
        save_metadata()

        return (
            f"✅ فایل `{source_name}` با موفقیت به پایگاه دانش اضافه شد.\n"
            f"📄 تعداد بخش‌ها: {added_count}",
            get_documents_list(),
        )
    except Exception as e:
        logger.exception("File upload processing failed")
        return f"یک خطای غیرمنتظره رخ داد: {e}", get_documents_list()


async def chat_interface_fn(message, history):
    """Handles the chat interaction for the Gradio ChatInterface."""
    if history is None:
        history = []

    if not rag_pipeline.index:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": "هنوز هیچ سندی در پایگاه دانش وجود ندارد. لطفاً ابتدا یک فایل را آپلود کنید.",
        })
        yield history
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "در حال جستجو و تحلیل..."})
    yield history

    retrieved_context, sources = rag_pipeline.retrieve(message, return_sources=True)

    if not retrieved_context:
        history[-1] = {
            "role": "assistant",
            "content": "نتوانستم اطلاعات مرتبطی در اسناد موجود پیدا کنم.",
        }
        yield history
        return

    final_answer = await llm_handler.get_synthesized_answer(message, retrieved_context)

    unique_sources = list(set(sources))
    source_info = "\n\n---\n📄 **منابع استفاده شده:**\n"
    for src in unique_sources:
        source_info += f"• {src}\n"

    final_answer_with_sources = final_answer + source_info
    history[-1] = {"role": "assistant", "content": final_answer_with_sources}
    append_to_global_chat_history(message, final_answer_with_sources)
    yield history


# --- Gradio Interface Definition ---
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
                        documents_display = gr.Markdown(value=get_documents_list(), rtl=True)
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
                    chatbot = gr.Chatbot(
                        label="مکالمه",
                        height=500,
                        rtl=True,
                        value=load_chat_history(),
                    )
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

    file_uploader.upload(
        fn=process_uploaded_file,
        inputs=file_uploader,
        outputs=[upload_status, documents_display],
    )

    refresh_docs_btn.click(fn=lambda: get_documents_list(), inputs=None, outputs=documents_display)

    delete_btn.click(
        fn=delete_document,
        inputs=doc_name_input,
        outputs=[delete_status, documents_display],
    )

    clear_all_btn.click(fn=lambda: gr.update(visible=True), inputs=None, outputs=confirm_dialog)

    confirm_yes_btn.click(
        fn=clear_all_documents_confirmed,
        inputs=None,
        outputs=[delete_status, documents_display, confirm_dialog],
    )

    confirm_no_btn.click(
        fn=lambda: (gr.update(visible=False), "عملیات لغو شد."),
        inputs=None,
        outputs=[confirm_dialog, delete_status],
    )

    msg.submit(chat_interface_fn, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    submit_btn.click(chat_interface_fn, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    clear_chat_btn.click(lambda: [], None, chatbot, queue=False)

    deep_analysis_btn.click(
        fn=run_deep_analysis_with_progress,
        inputs=[
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
        outputs=[deep_analysis_progress, deep_analysis_status, deep_analysis_output],
    )


if __name__ == "__main__":
    demo.launch()
