import React, { useState } from 'react';
import { Modal, Form, Input, Select, Button } from 'antd';
import ReactJson from 'react-json-view';
import { FormInstance } from 'antd/es/form';
import { FileSearchOutlined } from '@ant-design/icons';
import InformationDisplayChilder from '../InformationDisplayChilder';
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

  // getAddEdit: any;
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

  // getAddEdit,
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
  const [visibleInformationDisplay, setvisibleInformationDisplay] = useState(false);
  const [value, setValue] = useState(undefined);
  const [value7, setValue7] = useState(valueListTable.record.recall_config);
  const [value2, setValue2] = useState(undefined);
  const [value3, setValue3] = useState(undefined);
  const [value4, setValue4] = useState(undefined);
  const [value5, setValue5] = useState({
    address: '',
    host: '',
    db: '',
    collection: '',
    auth: '',
  });

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
    const xxx = formRef1.current;
    api.get_component_config(value).then(item => {
      if (item && item.result) {
        if (valueListTable.title !== '修改策略信息') {
          setValue5(JSON.parse(item.result.config));
          xxx?.setFieldsValue({ recall_config: item.result.config });
          // getAddEdit(JSON.parse(item.result.config));
        } else {
          setValue7(item.result.config);
          xxx?.setFieldsValue({ recall_config: item.result.config });
          // getAddEdit(JSON.parse(item.result.config));
        }
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
  function getAdd(e: any) {
    console.log(e.updated_src, 'onAdd');
    if (e.new_value == 'error') {
      return false;
    }
  }

  function getEdit(e: any) {
    setTimeout(() => {
      console.log(e.updated_src, 'onAdd', valueListTable.record);
      if (e.new_value == 'error') {
        return false;
      }
      if (valueListTable.title === '修改策略信息') {
        const xxx = formRef1.current;
        xxx?.setFieldsValue({ recall_config: e.updated_src });
      } else {
        const xxx = formRef1.current;
        xxx?.setFieldsValue({ recall_config: e.updated_src });
        // getAddEdit(JSON.parse(e.updated_src));
      }
    }, 500);
  }
  function onDelete(e: any) {
    setTimeout(() => {
      if (e.new_value == 'error') {
        return false;
      }
      if (valueListTable.title === '修改策略信息') {
        const xxx = formRef1.current;
        xxx?.setFieldsValue({ recall_config: e.updated_src });
      } else {
        const xxx = formRef1.current;
        xxx?.setFieldsValue({ recall_config: e.updated_src });
        // getAddEdit(JSON.parse(e.updated_src));
      }
    }, 500);
  }

  // 查看组件信息
  const visibleInformationDisplayShow = () => {
    setvisibleInformationDisplay(true);
  };

  return (
    <Modal
      visible={visible}
      title={valueListTable.title}
      style={{ right: '30%' }}
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
        <Button type="text" style={{ float: 'right' }} onClick={visibleInformationDisplayShow}>
          <FileSearchOutlined />
        </Button>

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
        </Form.Item>
        <Form.Item name="middleware_marks" label="中间件选择">
          <Select
            mode="multiple"
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

        <Form.Item rules={[{ required: false, message: '请输入概括描述' }]} name="brief_des" label="概括描述">
          <Input.TextArea autoSize showCount maxLength={25} />
        </Form.Item>

        <Form.Item
          rules={[{ required: valueListTable.title === '修改策略信息' ? true : false, message: '请输入配置' }]}
          name="recall_config"
          label="召回配置"
        >
          {/* <TextArea style={{ display: 'none' }} /> */}
          <div>
            <ReactJson
              // eslint-disable-next-line
              src={valueListTable.title === '修改策略信息' ? JSON.parse(value7) : value5}
              onDelete={e => onDelete(e)} //  删除属性
              displayDataTypes={false}
              theme="railscasts"
              onEdit={e => getEdit(e)}
              onAdd={e => getAdd(e)}
            />
          </div>
        </Form.Item>

        <Form.Item name="des" label="描述信息">
          <TextArea rows={4} />
        </Form.Item>
      </Form>

      <InformationDisplayChilder
        visible={visibleInformationDisplay}
        onCancel={() => {
          setvisibleInformationDisplay(false);
        }}
      />
    </Modal>
  );
};
export default PolicyInformation;
