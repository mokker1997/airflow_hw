import json
import dill
import pandas as pd
from pathlib import Path
from datetime import datetime

def predict():
    base_path = Path(__file__).resolve().parent.parent
    model_dir = base_path / "data" / "models"
    test_dir = base_path / "data" / "test"
    preds_dir = base_path / "data" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    model_files = sorted(model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not model_files:
        raise FileNotFoundError("Модель не найдена в data/models")
    model_path = model_files[-1]

    with open(model_path, "rb") as f:
        model = dill.load(f)

    test_files = sorted(test_dir.glob("*.json"))
    if not test_files:
        raise FileNotFoundError("Нет тестовых JSON-файлов в data/test")

    parts = []
    for jf in test_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)            # один объект из файла
        df = pd.DataFrame([data])          # делаем одну строку

        if "id" in df.columns:
            id_series = df["id"].astype(str)
        else:
            id_series = pd.Series([jf.stem], name="id")

        pred = model.predict(df)
        parts.append(pd.DataFrame({"id": id_series, "pred": pred}))

    result = pd.concat(parts, ignore_index=True)
    out_name = f"preds_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
    out_path = preds_dir / out_name
    result.to_csv(out_path, index=False)
    print(f"Предсказания сохранены: {out_path}")

if __name__ == "__main__":
    predict()
