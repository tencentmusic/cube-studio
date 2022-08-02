import React, { useState } from 'react';
import { Modal, Form, Input, Select, message } from 'antd';
import { FormInstance } from 'antd/es/form';
import api from '@src/api';
const { TextArea } = Input;
const { Option } = Select;
interface Values {
  des: string;
  component_mark: string;
}

interface CollectionCreateFormProps {
  visible: boolean;
  onCreate: (values: Values) => void;
  onCancel: () => void;
}
//  修改组件信息
const ModifyInformation: React.FC<CollectionCreateFormProps> = ({ visible, onCreate, onCancel }) => {
  const formRef = React.createRef<FormInstance>();
  const [form] = Form.useForm();
  const component_typeSelect = ['filter', 'middleware', 'sort', 'recall'];

  const [component_mark, setComponent_mark] = useState([]);

  function handleChangeTtypeSelect(value: any) {
    api.get_components_mark([value]).then(item => {
      if (item.retcode === 0) {
        message.success('获取组件mark成功');
        if (item && item.result && item.result.data && item.result.data[value]) {
          setComponent_mark(item.result.data[value]);
        } else {
          formRef.current?.setFieldsValue({
            component_mark: '',
          });
          setComponent_mark([]);
        }
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  }

  // function onChange(value: any) {
  //   component_mark.filter(item => {
  //     if (item['mark'] === value) {
  //       setvalueDec(item['des']);
  //     }
  //   });
  // }
  function afterClose() {
    form.resetFields();
  }
  // function onSearch(val: any) {
  //   console.log('search:', val);
  // }
  return (
    <Modal
      visible={visible}
      title="修改组件信息"
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
      afterClose={afterClose}
      okText="确定"
      cancelText="取消"
    >
      <Form ref={formRef} form={form} layout="vertical" name="form_in_modal">
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

        <Form.Item
          name="component_mark"
          label="组件Mark"
          rules={[{ required: true, message: 'Please input the component_mark of collection!' }]}
        >
          <Select
            showSearch
            placeholder="Select a person"
            optionFilterProp="children"
            // onChange={onChange}
            // onSearch={onSearch}
            filterOption={(input, option: any) => option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0}
          >
            {component_mark
              ? component_mark.map(OptionItem => {
                  return (
                    <Option value={OptionItem['mark']} key={OptionItem['mark']}>
                      {OptionItem['mark']}
                    </Option>
                  );
                })
              : []}
          </Select>
        </Form.Item>
        <Form.Item name="des" label="描述">
          <TextArea name="textarea" rows={4} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default ModifyInformation;
