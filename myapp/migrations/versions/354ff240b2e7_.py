"""empty message

Revision ID: 354ff240b2e7
Revises: 7227cb38c9b1
Create Date: 2023-12-08 11:23:16.815303

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '354ff240b2e7'
down_revision = '7227cb38c9b1'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('task', schema=None) as batch_op:
        batch_op.add_column(sa.Column('resource_rdma', sa.String(length=100), nullable=True, comment='rdma的资源数量'))


def downgrade():
    with op.batch_alter_table('task', schema=None) as batch_op:
        batch_op.drop_column('resource_rdma')

