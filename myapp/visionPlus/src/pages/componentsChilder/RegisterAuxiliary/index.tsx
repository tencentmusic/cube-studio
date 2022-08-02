import React, { useState } from 'react';
import { Modal, Form, Input, Select, Tooltip, Row, Col } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { FormInstance } from 'antd/es/form';
import api from '@src/api';
const { TextArea } = Input;
const { Option } = Select;
interface Values {
  des: string;
  config: string;
  opr: string;
  component_type: string;
  component_mark: string;
}

interface CollectionCreateFormProps {
  visible: boolean;
  onCreate: (values: Values) => void;
  onCancel: () => void;
}
// 注册辅助组件
const RegisterAuxiliary: React.FC<CollectionCreateFormProps> = ({ visible, onCreate, onCancel }) => {
  const [form] = Form.useForm();
  const component_typeSelect = ['filter', 'middleware', 'sort'];
  const [component_markSelect, setcomponent_markSelect] = useState([]);
  const [valueFirst, setvalueFirst] = useState('');
  const formRef2 = React.createRef<FormInstance>();
  const text = <span>组件名，需要满足‘组件类别_描述信息(不可包含‘_’)’</span>;
  function handleChangeTtypeSelect(value: any) {
    console.log(value, '----');
    setvalueFirst(value);
    api.get_components_type(value).then(item => {
      if (item && item.result && item.result.data) {
        setcomponent_markSelect(item.result.data[value]);
      }
    });
  }
  function handleChangemarkSelect(value: any) {
    console.log(`selected ${value}`, valueFirst, 333);
    const sercordChange = formRef2.current;
    api.get_component_config2(value, valueFirst).then(item => {
      console.log(item.result.config, '111');
      sercordChange?.setFieldsValue({ config: item.result.config });
    });
  }

  return (
    <Modal
      visible={visible}
      title="注册辅助组件"
      onCancel={onCancel}
      onOk={() => {
        form
          .validateFields()
          .then(values => {
            form.resetFields();
            onCreate(values);
          })
          .catch(info => {
            console.log('Validate Failed:', info);
          });
      }}
      okText="注册"
      cancelText="取消"
    >
      <Form ref={formRef2} form={form} layout="vertical" name="form_in_modal">
        <Form.Item name="component" label="组件">
          <Select style={{ width: 200 }} onChange={handleChangeTtypeSelect}>
            {component_typeSelect.map(OptionItem => {
              return (
                <Option value={OptionItem} key={OptionItem}>
                  {OptionItem}
                </Option>
              );
            })}
          </Select>
        </Form.Item>
        <Form.Item name="component_type" label="组件类别" rules={[{ required: true, message: '请选择组件类别' }]}>
          <Select onChange={handleChangemarkSelect}>
            {component_markSelect
              ? component_markSelect.map(OptionItem => {
                  return (
                    <Option value={OptionItem} key={OptionItem}>
                      {OptionItem}
                    </Option>
                  );
                })
              : []}
          </Select>
        </Form.Item>
        <Form.Item name="admin" label="负责人" rules={[{ required: true, message: '请输入负责人' }]}>
          <Input />
        </Form.Item>

        <Form.Item label="组件名">
          <Row gutter={8}>
            <Col span={22}>
              <Form.Item name="component_mark" noStyle rules={[{ required: true, message: '请输入组件名' }]}>
                <Input style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={2}>
              <Tooltip placement="top" title={text}>
                <QuestionCircleOutlined
                  style={{ color: 'red', fontSize: '18px', marginLeft: '10px', lineHeight: '32px' }}
                />
              </Tooltip>
            </Col>
          </Row>
        </Form.Item>

        <Form.Item name="config" label="配置项">
          <TextArea rows={4} />
        </Form.Item>
        <Form.Item name="des" label="描述">
          <TextArea rows={4} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default RegisterAuxiliary;
