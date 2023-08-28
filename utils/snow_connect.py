import json
from typing import Any, Dict

import streamlit as st
from snowflake.snowpark.session import Session


class SnowflakeConnection:
    """
    This class is used to establish a connection to Snowflake.

    Attributes
    ----------
    connection_parameters : Dict[str, Any]
        A dictionary containing the connection parameters for Snowflake.
    session : snowflake.snowpark.Session
        A Snowflake session object.

    Methods
    -------
    get_session()
        Establishes and returns the Snowflake connection session.

    """

    def __init__(self, credentials_path: str = None):
        if credentials_path:
            self.__connection_parameters = self._get_connection_parameters_from_path(credentials_path)
        else:
            self.__connection_parameters = self._get_connection_parameters_from_env()
        self.session = None

    @staticmethod
    def _get_connection_parameters_from_path(path) -> Dict[str, Any]:
        with open(path) as credentials:
            return json.load(credentials)

    @staticmethod
    def _get_connection_parameters_from_env() -> Dict[str, Any]:
        connection_parameters = {
            "account": st.secrets["ACCOUNT"],
            "user": st.secrets["USER_NAME"],
            "password": st.secrets["PASSWORD"],
            "warehouse": st.secrets["WAREHOUSE"],
            "database": st.secrets["DATABASE"],
            "schema": st.secrets["SCHEMA"],
            "role": st.secrets["ROLE"],
        }
        return connection_parameters

    def get_session(self):
        """
        Establishes and returns the Snowflake connection session.
        Returns:
            session: Snowflake connection session.
        """
        if self.session is None:
            self.session = Session.builder.configs(self.__connection_parameters).create()
            self.session.sql_simplifier_enabled = True
        return self.session

    def get_uri(self):
        return f"snowflake://{self.__connection_parameters['user']}:{self.__connection_parameters['password']}@{self.__connection_parameters['account']}/{self.__connection_parameters['database']}/{self.__connection_parameters['schema']}?warehouse={self.__connection_parameters['warehouse']}&role={self.__connection_parameters['role']}"

    @property
    def database(self):
        return self.__connection_parameters['database']

    @property
    def schema(self):
        return self.__connection_parameters['schema']
