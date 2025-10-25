import argparse, torch
from pathlib import Path
import open_clip

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="RN50")
parser.add_argument("--vocab_txt", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--template", default="a photo of {}")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained="openai", device=device)
tokenizer = open_clip.get_tokenizer(args.model)

words = [w.strip() for w in Path(args.vocab_txt).read_text().splitlines() if w.strip()]
embs = []
with torch.no_grad():
    for w in words:
        tok = tokenizer([args.template.format(w)]).to(device)
        e = model.encode_text(tok)
        e = e / e.norm(dim=-1, keepdim=True)
        ems = e.float().cpu()  # [1, D]
        embs.append(ems)
E = torch.cat(embs, dim=0)  # [V, D]
torch.save({"words": words, "embeddings": E}, args.out)
print("saved:", args.out, "shape:", tuple(E.shape))
