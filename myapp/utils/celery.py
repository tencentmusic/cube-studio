
import logging
from typing import Iterator

from contextlib2 import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from myapp import app, db

logger = logging.getLogger(__name__)

# Null pool is used for the celery workers due process forking side effects.
@contextmanager
def session_scope(nullpool: bool) -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    database_uri = app.config["SQLALCHEMY_DATABASE_URI"]
    if nullpool:
        engine = create_engine(database_uri, poolclass=NullPool)
        session_class = sessionmaker()
        session_class.configure(bind=engine)
        session = session_class()
    else:
        session = db.session()
        session.commit()  # HACK

    try:
        yield session
        session.commit()
    except SQLAlchemyError as ex:
        session.rollback()
        logger.exception(ex)
        raise
    finally:
        session.close()
