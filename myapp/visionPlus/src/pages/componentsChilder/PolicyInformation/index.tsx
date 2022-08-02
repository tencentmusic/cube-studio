import React, { useState } from 'react';
import { Modal, Form, Input, Select } from 'antd';
import { FormInstance } from 'antd/es/form';
import api from '@src/api';
const { Option } = Select;
interface Values {
  apps: string;
  strategy_id: number;
  recall_mark: string;
  middleware_marks: string;
  filter_marks: string;
  sort_marks: string;
  admin: string;
  recall_config: string;
  des: string;
}

interface CollectionCreateFormProps {
  visible: boolean;
  onCreate: (values: Values) => void;
  onCancel: () => void;
  valueListTable: any;
  valueData: any;
  valueData2: any;
  valueData3: any;
  valueData4: any;
}

// 编辑策略信息
const PolicyInformation: React.FC<CollectionCreateFormProps> = ({
  visible,
  onCreate,
  onCancel,
  valueListTable,
  valueData,
  valueData2,
  valueData3,
  valueData4,
}) => {
  const component_typeSelect = [
    { key: 1, value: 'k歌' },
    { key: 2, value: 'k歌国际版' },
    { key: 4, value: 'QQ音乐' },
    { key: 8, value: '音兔' },
    { key: 16, value: '酷狗音乐' },
    { key: 32, value: '酷我音乐' },
    { key: 64, value: '爱听卓乐' },
    { key: 128, value: '懒人畅听' },
    { key: 256, value: '其它' },
  ];
  const [form] = Form.useForm();
  const { TextArea } = Input;
  const [value, setValue] = useState(undefined);
  const [value2, setValue2] = useState(undefined);
  const [value3, setValue3] = useState(undefined);
  const [value4, setValue4] = useState(undefined);

  const formRef1 = React.createRef<FormInstance>();

  const handleSearch = (value: any) => {
    console.log(value, '111');
  };

  const options = valueData.map((d: any) => (
    <>
      <Option key={d ? d['mark'] : ''} value={d ? d['mark'] : ''}>
        {d ? d['mark'] : ''}
      </Option>
    </>
  ));

  const options2 = valueData2.map((d: any) => (
    <>
      <Option key={d ? d['mark'] : ''} value={d ? d['mark'] : ''}>
        {d ? d['mark'] : ''}
      </Option>
    </>
  ));

  const options3 = valueData3.map((d: any) => (
    <>
      <Option key={d ? d['mark'] : ''} value={d ? d['mark'] : ''}>
        {d ? d['mark'] : ''}
      </Option>
    </>
  ));

  const options4 = valueData4.map((d: any) => (
    <>
      <Option key={d ? d['mark'] : ''} value={d ? d['mark'] : ''}>
        {d ? d['mark'] : ''}
      </Option>
    </>
  ));

  const handleChange = (value: any) => {
    console.log(value, '999999999888888888');
    const xxx = formRef1.current;
    api.get_component_config(value).then(item => {
      if (item && item.result) {
        xxx?.setFieldsValue({ recall_config: item.result.config });
      }
    });

    setValue(value);
  };

  const handleChange2 = (value: any) => {
    setValue2(value);
  };
  const handleChange3 = (value: any) => {
    setValue3(value);
  };
  const handleChange4 = (value: any) => {
    setValue4(value);
  };

  return (
    <Modal
      visible={visible}
      title={valueListTable.title}
      onCancel={onCancel}
      onOk={() => {
        form
          .validateFields()
          .then(values => {
            onCreate(values);
          })
          .catch(info => {
            console.log('Validate Failed:', info);
          });
      }}
      okText="确认"
      cancelText="取消"
    >
      <Form ref={formRef1} form={form} layout="vertical" initialValues={valueListTable.record} name="form_in_modal">
        <Form.Item
          name="app"
          label="应用"
          rules={[{ required: true, message: 'Please input the apps of collection!' }]}
        >
          <Select style={{ width: 200 }}>
            {component_typeSelect.map(OptionItem => {
              return (
                <Option key={OptionItem.key} value={OptionItem.key}>
                  {OptionItem.value}
                </Option>
              );
            })}
          </Select>
        </Form.Item>
        <Form.Item name="recall_mark" label="召回组件选择" rules={[{ required: true, message: '请选择召回组件' }]}>
          {/* <Input /> */}
          <Select
            style={{ width: 200 }}
            showSearch
            value={value}
            defaultActiveFirstOption={false}
            showArrow={false}
            onSearch={handleSearch}
            onChange={handleChange}
            notFoundContent={null}
            filterOption={(input, option: any) => option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0}
          >
            {options}
          </Select>
          {/* <Popover content={content}>
            <QuestionOutlined />
          </Popover> */}
        </Form.Item>
        <Form.Item name="middleware_marks" label="中间件选择">
          {/* <Input /> */}
          <Select
            mode="multiple"
            // style={{ width: 400 }}
            showSearch
            value={value3}
            defaultActiveFirstOption={false}
            showArrow={false}
            onChange={handleChange3}
            notFoundContent={null}
            filterOption={(input, option: any) => option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0}
          >
            {options3}
          </Select>
        </Form.Item>
        <Form.Item name="filter_marks" label="过滤组件选择">
          <Select
            mode="multiple"
            // style={{ width: 400 }}
            showSearch
            value={value2}
            defaultActiveFirstOption={false}
            showArrow={false}
            onChange={handleChange2}
            notFoundContent={null}
            filterOption={(input, option: any) => option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0}
          >
            {options2}
          </Select>
        </Form.Item>
        <Form.Item name="sort_marks" label="排序组件选择">
          <Select
            mode="multiple"
            // style={{ width: 400 }}
            showSearch
            value={value4}
            defaultActiveFirstOption={false}
            showArrow={false}
            onChange={handleChange4}
            notFoundContent={null}
            filterOption={(input, option: any) => option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0}
          >
            {options4}
          </Select>
        </Form.Item>
        <Form.Item rules={[{ required: true, message: '请输入负责人' }]} name="admin" label="负责人">
          <Input />
        </Form.Item>
        <Form.Item
          rules={[{ required: valueListTable.title === '修改策略信息' ? true : false, message: '请输入配置' }]}
          name="recall_config"
          label="召回配置"
        >
          <TextArea />
        </Form.Item>

        <Form.Item name="des" label="描述信息">
          <TextArea rows={4} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default PolicyInformation;
