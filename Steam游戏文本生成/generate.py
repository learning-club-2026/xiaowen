"""使用训练好的模型，根据一条结构化游戏记录生成简介。"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from setting import DEVICE, MAX_SOURCE_LEN, MAX_TARGET_LEN, MODEL_NAME, MODEL_SAVE_PATH


def generate_description(game_fields, max_new_tokens=MAX_TARGET_LEN):
    """game_fields 可传入如 {"游戏名称": {"name": "..."}, ...} 的分类结构。"""
    sections = []
    for group_name, fields in game_fields.items():
        values = [f"{key}: {value}" for key, value in fields.items() if value not in (None, "")]
        if values:
            sections.append(f"{group_name} — " + "; ".join(values))
    prompt = "根据以下 Steam 游戏结构化信息生成简洁、准确的游戏简介：\n" + "\n".join(sections)

    model_path = MODEL_SAVE_PATH if __import__("os").path.isdir(MODEL_SAVE_PATH) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    model.eval()
    inputs = tokenizer(prompt, max_length=MAX_SOURCE_LEN, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=4,
            no_repeat_ngram_size=3, early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    sample_game = {
        "游戏名称": {"name": "Example Adventure"},
        "类型与标签": {"genres": "Adventure, Indie", "tags": "Exploration, Story Rich"},
        "玩法与平台": {"windows": True, "multiplayer": False},
        "价格与内容": {"price": 9.99, "is_free": False},
    }
    print(generate_description(sample_game))
