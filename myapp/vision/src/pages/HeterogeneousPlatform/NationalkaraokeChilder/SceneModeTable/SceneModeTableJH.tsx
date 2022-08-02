import React from 'react';
import { Table } from 'antd';

export default function SceneModeTableJH(props: any) {
  const { columns, data } = props;
  return (
    <div>
      <Table pagination={false} columns={columns} dataSource={data} />
    </div>
  );
}
