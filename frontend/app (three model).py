import asyncio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer

app = Starlette()
templates = Jinja2Templates(directory="templates")

tokenizer = AutoTokenizer.from_pretrained("indonesian-nlp/gpt2")
model_introduction = AutoModelWithLMHead.from_pretrained(
    "./model/gpt2-pendahuluan/checkpoint-400")
model_climax = AutoModelWithLMHead.from_pretrained(
    "./model/gpt2-klimaks/checkpoint-400")
model_resolution = AutoModelWithLMHead.from_pretrained(
    "./model/gpt2-penyelesaian/checkpoint-400")

introduction_pipeline = pipeline(
    "text-generation", model=model_introduction, tokenizer=tokenizer)
climax_pipeline = pipeline(
    "text-generation", model=model_climax, tokenizer=tokenizer)
resolution_pipeline = pipeline(
    "text-generation", model=model_resolution, tokenizer=tokenizer)


async def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})


async def input(request: Request):
    return templates.TemplateResponse("input.html", {"request": request})


def remove_text_after_last_dot(text):
    last_dot_index = text.rfind('.')
    if last_dot_index != -1:
        return text[:last_dot_index + 1]
    return text


async def generate_output(request: Request):
    data = await request.form()
    input_text = data["input_text"]

    loading_task = asyncio.create_task(render_loading(request))

    generated_output = []

    generated_intro = await asyncio.to_thread(introduction_pipeline, input_text, max_length=32, num_return_sequences=1)
    generated_intro_text = generated_intro[0]['generated_text']
    generated_output.append(generated_intro_text)

    generated_climax = await asyncio.to_thread(climax_pipeline, generated_intro_text, max_length=64, num_return_sequences=1)
    generated_climax_text = generated_climax[0]['generated_text']
    generated_output.append(generated_climax_text)

    generated_resolution = await asyncio.to_thread(resolution_pipeline, generated_climax_text, max_length=96, num_return_sequences=1)
    generated_resolution_text = generated_resolution[0]['generated_text']
    generated_output.append(generated_resolution_text)

    generated_intro_text = remove_text_after_last_dot(generated_intro_text)
    generated_climax_text = remove_text_after_last_dot(generated_climax_text)
    generated_resolution_text = remove_text_after_last_dot(
        generated_resolution_text)

    split_intro_text = set(generated_intro_text.split())
    split_climax_text = set(generated_climax_text.split())
    split_resolution_text = set(generated_resolution_text.split())

    find_before_climax = split_intro_text.intersection(split_climax_text)
    find_before_resolution = split_climax_text.intersection(
        split_resolution_text)

    remove_before_climax = ' '.join(
        word for word in generated_climax_text.split() if word not in find_before_climax)
    remove_before_resolution = ' '.join(
        word for word in generated_resolution_text.split() if word not in find_before_resolution)

    await loading_task

    return templates.TemplateResponse("output.html", {"request": request, "generated_intro": generated_intro_text, "generated_climax": remove_before_climax, "generated_resolution": remove_before_resolution})
    # return templates.TemplateResponse("output.html", {"request": request, "generated_intro": generated_intro_text, "generated_climax": generated_climax_text, "generated_resolution": generated_resolution_text})


async def render_loading(request: Request):
    await asyncio.sleep(0.5)  # Optional: Menambah small delay

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.route("/")
async def handle_homepage(request: Request):
    return await homepage(request)


@app.route("/input")
async def handle_input(request: Request):
    return await input(request)


@app.route("/generate-output", methods=["POST"])
async def handle_generate_output(request: Request):
    return await generate_output(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=13234)
