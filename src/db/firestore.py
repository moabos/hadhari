from functools import lru_cache
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

cred_path = Path(__file__).parent / "firebaseServiceAccount.json"


@lru_cache(maxsize=1)
def get_db() -> firestore.Client:
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()
