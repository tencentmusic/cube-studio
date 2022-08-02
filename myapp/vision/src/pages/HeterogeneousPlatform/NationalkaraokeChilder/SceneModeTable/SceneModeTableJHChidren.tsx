import React from 'react';
import { Table } from 'antd';

export default function SceneModeTableJHChidren(props: any) {
  const { columns, data } = props;
  return (
    <div>
      <Table scroll={{ x: 500, y: 360 }} pagination={false} columns={columns} dataSource={data} />
    </div>
  );
}
