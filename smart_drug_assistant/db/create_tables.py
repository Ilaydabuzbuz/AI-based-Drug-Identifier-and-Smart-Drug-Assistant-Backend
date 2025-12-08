from smart_drug_assistant.db.session import engine
from smart_drug_assistant.db.base import Base
from smart_drug_assistant.models import user

def create_tables():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created.")

if __name__ == "__main__":
    create_tables()
