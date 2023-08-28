import logging
import os
import sys
import streamlit as st
from functools import reduce
from typing import Any, Dict

import pandas
from langchain import SQLDatabase
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore, Milvus
from pandas import DataFrame
from pydantic import BaseModel

from utils import utils as u

# workaround for https://github.com/snowflakedb/snowflake-sqlalchemy/issues/380.
try:
    u.snowflake_sqlalchemy_20_monkey_patches()
except Exception as e:
    raise ValueError("Please run `pip install snowflake-sqlalchemy`")

from utils.snow_connect import SnowflakeConnection

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Secrets(BaseModel):
    MILVUS_URL: str
    MILVUS_PORT: str
    OPENAI_API_KEY: str


class Config(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 0
    docs_dir: str = "docs/"
    docs_glob: str = "**/*.md"
    ddl_file: str = 'sql/comp_sales/ddls.txt'


class DocumentProcessor:
    def __init__(self, secrets: Secrets, config: Config):
        self.secrets = secrets
        self.loader = TextLoader(config.ddl_file)
        self.text_splitter = CharacterTextSplitter(
            separator="CREATE TABLE",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=secrets.OPENAI_API_KEY)

    def process(self):
        data = self.loader.load()
        texts = self.text_splitter.split_documents(data)
        vector_db = Milvus.from_documents(
            texts,
            self.embeddings,
            connection_args={"host": self.secrets.MILVUS_URL, "port": self.secrets.MILVUS_PORT},

        )
        return vector_db.collection_name


class SnowflakeDDLLoader:
    def __init__(self, snowconn: SnowflakeConnection):
        self.db: SQLDatabase = SQLDatabase.from_uri(snowconn.get_uri(), engine_args=None, sample_rows_in_table_info=0)

    @staticmethod
    def write_ddls_on_fs(path: str, ddls: str):
        # Create the output directory
        if not os.path.exists(path):
            os.mkdir(path)

        file_name = os.path.join(path, f"ddls.txt")
        with open(file_name, 'w') as f:
            f.write(ddls)
        logging.info("Written file " + file_name)
        return file_name

    def ingest_files(self, path):
        all_tables_info = self.db.table_info
        return self.write_ddls_on_fs(path, all_tables_info)


def run():
    secrets = Secrets(
        MILVUS_URL=st.secrets["MILVUS_HOST"],
        MILVUS_PORT=st.secrets["MILVUS_PORT"],
        OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"],
    )
    config = Config()


    conn = SnowflakeConnection('credentials.json')
    loader = SnowflakeDDLLoader(conn)
    path = loader.ingest_files('sql/comp_sales')


    config.ddl_file = path
    doc_processor = DocumentProcessor(secrets, config)
    result = doc_processor.process()
    return result


if __name__ == "__main__":
    run()
