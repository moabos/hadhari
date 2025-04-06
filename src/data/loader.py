import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl

from db.firestore import get_db


def load_messages(*, validated_only: bool = True) -> pl.DataFrame:
    db = get_db()
    messages_ref = db.collection("messages")
    docs = messages_ref.stream()

    messages = []
    for doc in docs:
        data = doc.to_dict()
        is_validated = data.get("validated")

        if validated_only and not is_validated:
            continue

        message = {
            "id": doc.id,
            "sender_number": data.get("sender_number"),
            "raw_message": data.get("raw_message"),
            "prediction": data.get("prediction"),
        }

        if not validated_only:
            message["validated"] = is_validated

        messages.append(message)

    return pl.DataFrame(messages)
