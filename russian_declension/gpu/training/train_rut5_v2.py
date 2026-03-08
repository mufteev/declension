"""
Скрипт fine-tuning ruT5-small для морфологической инфлекции.

Запуск:
  python -m russian_declension.gpu.training.train_rut5 \
      --data data/rus.tsv \
      --base-model cointegrated/rut5-small \
      --output models/rut5-declension \
      --epochs 10 --batch-size 64 --lr 3e-4

Данные: UniMorph Russian (rus.tsv) + синтетические из pymorphy3.
Формат: TSV с колонками лемма/форма/признаки.
"""

from __future__ import annotations
import argparse, csv, logging, random
from pathlib import Path

logger = logging.getLogger(__name__)

UNIMORPH_CASE = {"NOM":"nomn","GEN":"gent","DAT":"datv","ACC":"accs","INS":"ablt","ESS":"loct"}
UNIMORPH_NUM = {"SG":"sing","PL":"plur"}


def load_unimorph(path: str) -> list[tuple[str,str,str]]:
    """Загрузить UniMorph TSV → список (prompt, target)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3: continue
            lemma, form, feats = row[0], row[1], row[2]
            tags = feats.split(";")
            case_code = next((UNIMORPH_CASE[t] for t in tags if t in UNIMORPH_CASE), None)
            num_code = next((UNIMORPH_NUM[t] for t in tags if t in UNIMORPH_NUM), None)
            if not case_code: continue
            prompt = f"inflect: {lemma} case={case_code}"
            if num_code: prompt += f" number={num_code}"
            data.append((prompt, form, feats))
    return data


def generate_synthetic(output_path: str, max_lemmas: int = 50000):
    """Сгенерировать синтетические данные из pymorphy3 парадигм."""
    from pymorphy3 import MorphAnalyzer
    morph = MorphAnalyzer(lang="ru")
    cases = [("nomn","nomn"),("gent","gent"),("datv","datv"),
             ("accs","accs"),("ablt","ablt"),("loct","loct")]
    numbers = [("sing","sing"),("plur","plur")]

    lines = []
    # Берём частотные слова из словаря
    count = 0
    for word in ["кошка","дом","время","путь","документ","рубль","компания",
                 "директор","ответственность","общество","договор","сторона"]:
        parses = morph.parse(word)
        if not parses: continue
        p = parses[0]
        for c_name, c_code in cases:
            for n_name, n_code in numbers:
                inflected = p.inflect({c_code, n_code})
                if inflected:
                    prompt = f"inflect: {p.normal_form} case={c_code} number={n_code}"
                    lines.append(f"{prompt}\t{inflected.word}")
        count += 1
        if count >= max_lemmas: break

    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    logger.info("Сгенерировано %d синтетических пар.", len(lines))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ruT5 для склонения")
    parser.add_argument("--data", required=True, help="Путь к UniMorph rus.tsv")
    parser.add_argument("--base-model", default="cointegrated/rut5-small")
    parser.add_argument("--output", default="models/rut5-declension")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--synthetic", default=None, help="Путь для синтетических данных")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Загрузка данных
    data = load_unimorph(args.data)
    logger.info("Загружено %d пар из UniMorph.", len(data))

    # Синтетические данные
    if args.synthetic:
        generate_synthetic(args.synthetic)

    # Разделение: 90/5/5
    random.shuffle(data)
    n = len(data)
    train = data[:int(n*0.9)]
    val = data[int(n*0.9):int(n*0.95)]
    test = data[int(n*0.95):]

    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    # Fine-tuning
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer
    from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    from torch.utils.data import Dataset

    class InflectionDataset(Dataset):
        def __init__(self, pairs, tokenizer, max_len=64):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            prompt, target, _ = self.pairs[idx]
            src = self.tokenizer(prompt, max_length=self.max_len,
                                 truncation=True, padding="max_length",
                                 return_tensors="pt")
            tgt = self.tokenizer(target, max_length=self.max_len,
                                 truncation=True, padding="max_length",
                                 return_tensors="pt")
            return {"input_ids": src.input_ids.squeeze(),
                    "attention_mask": src.attention_mask.squeeze(),
                    "labels": tgt.input_ids.squeeze()}

    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    train_ds = InflectionDataset(train, tokenizer, args.max_length)
    val_ds = InflectionDataset(val, tokenizer, args.max_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # evaluation_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        learning_rate=args.lr,
        warmup_steps=200,
        logging_steps=100,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    # Сохранение
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Тестирование
    logger.info("Тестирование на %d примерах...", len(test))
    correct = 0
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for prompt, target, _ in test[:1000]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if pred.lower() == target.lower():
            correct += 1
    logger.info("Test accuracy: %.2f%% (%d/%d)", correct/min(len(test),1000)*100,
                correct, min(len(test),1000))


if __name__ == "__main__":
    main()
