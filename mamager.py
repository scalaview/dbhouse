from sqlalchemy.orm import sessionmaker
import models

session = sessionmaker()
session.configure(bind=models.engine)
models.Base.metadata.create_all(models.engine)