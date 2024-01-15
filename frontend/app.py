import asyncio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
import nltk
from nltk.translate.bleu_score import sentence_bleu
from os.path import dirname
from rouge_score import rouge_scorer
import re
import json

app = Starlette()
templates = Jinja2Templates(directory="templates")

tokenizer = AutoTokenizer.from_pretrained("indonesian-nlp/gpt2")
model = AutoModelWithLMHead.from_pretrained('./model/checkpoint-2600')

text_generation_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer)


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

    initial_input = data["input_text"]
    print(initial_input)
    input_text = "<|startoftext|> " + data["input_text"]

    loading_task = asyncio.create_task(render_loading(request))

    output_text = await asyncio.to_thread(text_generation_pipeline, input_text, max_length=1023, num_beams=5, repetition_penalty=5.0)
    print(output_text)
    print(output_text[0]['generated_text'])
    app.state.output_text = output_text

    regex_selection_exposition = re.search(
        r"<\|startoftext\|>(.*?)<\|sentence1to2\|>", output_text[0]['generated_text'])
    regex_selection_rising_action = re.search(
        r'<\|sentence1to2\|>(.*?)<\|sentence2to3\|>', output_text[0]['generated_text'])
    regex_selection_climax = re.search(
        r'<\|sentence2to3\|>(.*?)<\|sentence3to4\|>', output_text[0]['generated_text'])
    regex_selection_falling_action = re.search(
        r'<\|sentence3to4\|>(.*?)<\|sentence4to5\|>', output_text[0]['generated_text'])
    regex_selection_resolution = re.search(
        r"<\|sentence4to5\|>(.*?[.!?])", output_text[0]['generated_text'])

    # print(regex_selection_exposition)
    # print(regex_selection_rising_action)

    generated_exposition = regex_selection_exposition.group(1)
    generated_rising_action = regex_selection_rising_action.group(1)
    generated_climax = regex_selection_climax.group(1)
    generated_falling_action = regex_selection_falling_action.group(1)
    generated_resolution = regex_selection_resolution.group(1)

    await loading_task

    return templates.TemplateResponse("output.html", {
        "request": request,
        "generated_exposition": generated_exposition,
        "generated_rising_action": generated_rising_action,
        "generated_climax": generated_climax,
        "generated_falling_action": generated_falling_action,
        "generated_resolution": generated_resolution,
        "data_input": data["input_text"],
        "data_output": output_text,
        "all_output": output_text[0]['generated_text'],
    })


async def evaluation(request: Request):

    # output_text = app.state.output_text
    data = await request.form()
    output = data["output_text"]
    output_change_quote = output.replace("'", "\"")
    output_text = json.loads(output_change_quote)

    file_path = "reference_data.txt"
    with open(file_path, "r") as f:
        array_train_dataset = f.read().splitlines()

    gpt2_output = output_text[0]['generated_text']
    reference_sentences = array_train_dataset

    gpt2_tokens = nltk.word_tokenize(gpt2_output)
    reference_tokens = [nltk.word_tokenize(
        sentence) for sentence in reference_sentences]

    output_bleu_2 = sentence_bleu(
        reference_tokens, gpt2_tokens, weights=(0.5, 0.5, 0, 0))
    output_bleu_3 = sentence_bleu(
        reference_tokens, gpt2_tokens, weights=(0.33, 0.33, 0.33, 0))
    output_bleu_4 = sentence_bleu(
        reference_tokens, gpt2_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    print('Cumulative 2-gram: %f' % output_bleu_2)
    print('Cumulative 3-gram: %f' % output_bleu_3)
    print('Cumulative 4-gram: %f' % output_bleu_4)

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    max_rouge_scores = {metric: 0.0 for metric in [
        'rouge1', 'rouge2', 'rougeL']}

    for reference in reference_sentences:
        rouge_scores = scorer.score(gpt2_output, reference)

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            if rouge_scores[metric].fmeasure > max_rouge_scores[metric]:
                max_rouge_scores[metric] = rouge_scores[metric].fmeasure

    for metric in ['rouge1', 'rouge2', 'rougeL']:
        print(f"{metric} Precision: {rouge_scores[metric].precision}")
        print(f"{metric} Recall: {rouge_scores[metric].recall}")
        print(f"{metric} F1 Score: {rouge_scores[metric].fmeasure}")

    regex_selection_exposition = re.search(
        r"<\|startoftext\|>(.*?)<\|sentence1to2\|>", output_text[0]['generated_text'])
    regex_selection_rising_action = re.search(
        r'<\|sentence1to2\|>(.*?)<\|sentence2to3\|>', output_text[0]['generated_text'])
    regex_selection_climax = re.search(
        r'<\|sentence2to3\|>(.*?)<\|sentence3to4\|>', output_text[0]['generated_text'])
    regex_selection_falling_action = re.search(
        r'<\|sentence3to4\|>(.*?)<\|sentence4to5\|>', output_text[0]['generated_text'])
    regex_selection_resolution = re.search(
        r"<\|sentence4to5\|>(.*?[.!?])", output_text[0]['generated_text'])

    generated_exposition = regex_selection_exposition.group(1)
    generated_rising_action = regex_selection_rising_action.group(1)
    generated_climax = regex_selection_climax.group(1)
    generated_falling_action = regex_selection_falling_action.group(1)
    generated_resolution = regex_selection_resolution.group(1)

    output_cerita = generated_exposition + generated_rising_action + \
        generated_climax + generated_falling_action + generated_resolution
    print(output_text[0]['generated_text'])
    print(output_cerita)

    return templates.TemplateResponse("evaluation.html", {
        "request": request,
        "generated_exposition": generated_exposition,
        "generated_rising_action": generated_rising_action,
        "generated_climax": generated_climax,
        "generated_falling_action": generated_falling_action,
        "generated_resolution": generated_resolution,
        "output_bleu_2": output_bleu_2,
        "output_bleu_3": output_bleu_3,
        "output_bleu_4": output_bleu_4,
        "rouge1_precision": rouge_scores['rouge1'].precision,
        "rouge1_recall": rouge_scores['rouge1'].recall,
        "rouge1_fmeasure": rouge_scores['rouge1'].fmeasure,
        "rouge2_precision": rouge_scores['rouge2'].precision,
        "rouge2_recall": rouge_scores['rouge2'].recall,
        "rouge2_fmeasure": rouge_scores['rouge2'].fmeasure,
        "rougeL_precision": rouge_scores['rougeL'].precision,
        "rougeL_recall": rouge_scores['rougeL'].recall,
        "rougeL_fmeasure": rouge_scores['rougeL'].fmeasure,
        "summary": output_cerita,
        "all_output": output_text[0]['generated_text'],
    })


async def render_loading(request: Request):
    await asyncio.sleep(0.5)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.route("/")
async def handle_homepage(request: Request):
    return await homepage(request)


@app.route("/input")
async def handle_input(request: Request):
    return await input(request)


@app.route("/generate-output", methods=["GET", "POST"])
async def handle_generate_output(request: Request):
    return await generate_output(request)


@app.route("/evaluation", methods=["GET", "POST"])
async def handle_input(request: Request):
    return await evaluation(request)


async def error_500(request, exc):
    return templates.TemplateResponse("error_500.html", {"request": request, "exception": exc}, status_code=500)

app.add_exception_handler(HTTPException, error_500)

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="192.168.54.216", port=13234)
    # uvicorn.run(app, host="192.168.0.111", port=13234)
    uvicorn.run(app, host="127.0.0.1", port=13234)
