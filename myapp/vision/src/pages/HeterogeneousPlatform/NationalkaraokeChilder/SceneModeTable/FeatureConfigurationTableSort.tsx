import React, { useState, useEffect } from 'react';
import { SortableContainer, SortableElement } from 'react-sortable-hoc';

import { arrayMoveImmutable } from 'array-move';
import { Table } from 'antd';

const SortableItem = SortableElement((props: any) => {
  return <tr {...props} />;
});
const SortableBody = SortableContainer((props: any) => <tbody {...props} />);
interface ItemOldNew {
  oldIndex: any;
  newIndex: any;
}
interface ItemDraggableBodyRow {
  className: string;
  // style: object;
  [propName: string]: string | number;
}
export default function FeatureConfigurationTableSort(props: any) {


  const { columns, data, onDataSourceSort } = props;
  const [dataSource, setDataSource] = useState(data || []);
  useEffect(() => {
    setDataSource(props.data);
  }, [props.data]);

  function onSortEnd({ oldIndex, newIndex }: ItemOldNew) {
    if (oldIndex !== newIndex) {
      const newData = arrayMoveImmutable([].concat(dataSource), oldIndex, newIndex).filter((el: any) => !!el);
      setDataSource(newData);
      onDataSourceSort(newData);
    }
  }

  function DraggableContainer(props: any) {
    return <SortableBody useDragHandle disableAutoscroll helperClass="row-dragging" onSortEnd={onSortEnd} {...props} />;
  }

  function DraggableBodyRow({ className, style, ...restProps }: ItemDraggableBodyRow) {
    // function findIndex base on Table rowKey props and should always be a right array index
    const index = dataSource.findIndex((x: any) => {
      return x.index === restProps['data-row-key'];
    });
    return <SortableItem index={index} {...restProps} />;
  }
  return (
    <div>
      <Table
        pagination={false}
        dataSource={dataSource}
        columns={columns}
        rowKey="index"
        components={{
          body: {
            wrapper: DraggableContainer,
            row: DraggableBodyRow,
          },
        }}
      />
    </div>
  );
}
