import React from 'react';
import { Table } from 'antd';

export default function SceneModeTable(props:any) {
  const { columns, data } = props;
  return (
    <div>
      <Table columns={columns} dataSource={data} />
    </div>
  );
}
