import os

import snowflake.snowpark as snowpark
import streamlit as st
import snowflake.snowpark.types as T
from pandasai.helpers.path import find_project_root


def snowflake_sqlalchemy_20_monkey_patches():
    import sqlalchemy.util.compat

    # make strings always return unicode strings
    sqlalchemy.util.compat.string_types = (str,)
    sqlalchemy.types.String.RETURNS_UNICODE = True

    import snowflake.sqlalchemy.snowdialect

    snowflake.sqlalchemy.snowdialect.SnowflakeDialect.returns_unicode_strings = True

    # make has_table() support the `info_cache` kwarg
    import snowflake.sqlalchemy.snowdialect

    def has_table(self, connection, table_name, schema=None, info_cache=None):
        """
        Checks if the table exists
        """
        return self._has_object(connection, "TABLE", table_name, schema)

    snowflake.sqlalchemy.snowdialect.SnowflakeDialect.has_table = has_table


def describeSnowparkDF(snowpark_df: snowpark.DataFrame):
    st.write("Here's some stats about the loaded data:")
    numeric_types = [T.DecimalType, T.LongType, T.DoubleType, T.FloatType, T.IntegerType]
    numeric_columns = [c.name for c in snowpark_df.schema.fields if type(c.datatype) in numeric_types]

    # Get categorical columns
    categorical_types = [T.StringType]
    categorical_columns = [c.name for c in snowpark_df.schema.fields if type(c.datatype) in categorical_types]

    st.write("Relational schema:")

    columns = [c for c in snowpark_df.schema.fields]
    st.write(columns)

    col1, col2, = st.columns(2)
    with col1:
        st.write('Numeric columns:\t', numeric_columns)

    with col2:
        st.write('Categorical columns:\t', categorical_columns)

    # Calculte statistics for our dataset
    st.dataframe(snowpark_df.describe().sort('SUMMARY'), use_container_width=True)


def get_plot_path(is_read: bool):
    if is_read:
        return os.path.join(find_project_root(), st.session_state['session_id'], 'exports', 'charts',
                            'temp_chart.png')
    else:
        return os.path.join(find_project_root(), st.session_state['session_id'])
