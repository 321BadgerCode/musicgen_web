from flask import Flask, request, render_template, send_from_directory
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
import os
from torch.cuda.amp import autocast

app = Flask(__name__)

MODEL_FILE = "musicgen-small.pt"
OUTPUT_DIR = os.path.join("static")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = "musicgen_out.wav"

print("Loading MusicGen model (this may take a moment)...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
# if os.path.exists(MODEL_FILE):
# 	model = torch.jit.load(MODEL_FILE)

# 	if not isinstance(model, MusicgenForConditionalGeneration):
# 		raise ValueError(f"Loaded model is not of type MusicgenForConditionalGeneration: {type(model)}")
# else:
# 	model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
# 	scripted_model = torch.jit.script(model)
# 	scripted_model.save(MODEL_FILE)
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.eval()
if torch.cuda.is_available():
	model.to("cuda")
	print("Model loaded on GPU.")

def generate_filename(prompts):
	return "_".join([p.replace(" ", "-") for p in prompts])

@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		prompts = request.form.getlist("prompt")
		prompts = [p.strip() for p in prompts if p.strip()] or ["sad retro game music"]

		inputs = processor(
			text=prompts,
			padding=True,
			return_tensors="pt",
		).to("cuda")

		with torch.no_grad():
			audio_values = model.generate(
				**inputs,
				max_new_tokens=256,
				top_k=50,
				top_p=0.95,
				num_return_sequences=1
			)

		sampling_rate = model.config.audio_encoder.sampling_rate
		filename = generate_filename(prompts) + ".wav"
		filepath = os.path.join(OUTPUT_DIR, filename)
		scipy.io.wavfile.write(filepath, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

		torch.cuda.empty_cache()

		return render_template("index.html", audio_file=filename, prompts=prompts)

	return render_template("index.html", audio_file=None, prompts=[])

@app.route("/static/<filename>")
def static_file(filename):
	return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
	import threading, time

	def run_flask():
		app.run(debug=False, port=8000, use_reloader=False)

	t = threading.Thread(target=run_flask)
	t.start()

	try:
		t.join()
	except KeyboardInterrupt:
		print("Shutting down...")
# add num beams, deepspeed, fp16/half/autocast, torch.jit.script/pruning/quantization