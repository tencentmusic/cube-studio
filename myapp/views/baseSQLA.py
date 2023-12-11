from flask_appbuilder.models.sqla.interface import SQLAInterface
from sqlalchemy import func

class MyappSQLAInterface(SQLAInterface):
    pass

class MyappSQLAInterface1(SQLAInterface):

    # @pysnooper.snoop(watch_explode=("query_count",))
    def query(
        self,
        filters=None,
        order_column="",
        order_direction="",
        page=None,
        page_size=None,
        select_columns=None,
    ):
        """
            QUERY
            :param filters:
                dict with filters {<col_name>:<value,...}
            :param order_column:
                name of the column to order
            :param order_direction:
                the direction to order <'asc'|'desc'>
            :param page:
                the current page
            :param page_size:
                the current page size

        """
        query = self.session.query(self.obj)
        query, relation_tuple = self._query_join_dotted_column(query, order_column)
        query = self._query_select_options(query, select_columns)
        query_count = self.session.query(func.count("*")).select_from(self.obj)

        query_count = self._get_base_query(query=query_count, filters=filters)
        query_count._order_by=False   # 增加这个，查询个数的时候不要排序
        count = query_count.scalar()

        # MSSQL exception page/limit must have an order by
        if (
                page
                and page_size
                and not order_column
                and self.session.bind.dialect.name == "mssql"
        ):
            pk_name = self.get_pk_name()
            query = query.order_by(pk_name)

        query = self._get_base_query(
            query=query,
            filters=filters,
            order_column=order_column,
            order_direction=order_direction,
        )

        # from sqlalchemy.orm.query import



        if page and page_size:
            query = query.offset(page * page_size)
        if page_size:
            query = query.limit(page_size)
        return count, query.all()
