import React, { useState, useEffect, useRef } from 'react';
import { Form, Input, Button, Space, Select, Row, Col, message } from 'antd';
import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';
import "./index.css"
const { Option } = Select;
const { TextArea } = Input

// const textItem = [
//     {
//         time: new Date().getTime()
//     }
//     // {
//     //     name: "",
//     //     type: "",
//     //     tag: "",
//     //     json: "",
//     //     des: ""
//     // }
// ]

import api from "@src/api/admin"

export default function StructureEditForm(props: any) {
    console.log(props.showEdit)
    // const [config, setConfig] = useState<any>()
    const [form] = Form.useForm();

    // const formRef = useRef(null)
   
  
    useEffect(()=>{
        form.resetFields();
        form.setFieldsValue(props.form);
    })
    

    const onFinish = (record: any) => {
        const obj:any = {}
        if( record.config) {
            record.config.forEach((item:any)=> {
                obj[item.name] = {
                    type: item.type,
                    tag: item.tag,
                    json: item.json,
                    des: item.des
                }
            })
        }
       
        const data:any = {
            struct_name: record.struct_name,
            des: record.des,
            status: 1,
            admin: record.admin,
            config: JSON.stringify(obj)
        }
        if(props.form.version) {
            data.version = props.form.version + 1
            api.mod_struct({opr: "mod_struct", data}).then(res=>{
                console.log(res)
                if (res.status === 0) {
                    message.success(res.message);
                    props.resFied()
                } else {
                    message.error(res.message);
                }
            })
        }else{
            api.add_struct({opr: "add_struct", data}).then(res=>{
                if (res.status === 0) {
                    message.success(res.message);
                    props.resFied()
                } else {
                    message.error(res.message);
                }
            })
        }
    };

    const handleChange = () => {
        form.setFieldsValue({ sights: [] });
    };
    const tableHeaderStyle = {
        textAlign: "center",
        background: " #ccc"
    }
    return (
        <Form form={form} name="dynamic_form_nest_item" className="structure-form" labelAlign="left" labelCol={{span: 4}} onFinish={onFinish} autoComplete="off">
            <Row gutter={10}>
                <Col span={13}>
                    <Form.Item name="struct_name" label="结构体名" rules={[{ required: true, message: '请输入结构名' }]}>
                        <Input disabled={props.isadd ? false : true}/>
                    </Form.Item>
                </Col>
                {/* <Col span={13}>
                    <Form.Item name="status" label="状态" rules={[{ required: true, message: '请选择状态' }]}>
                        <Select defaultValue={1} disabled={props.isadd ? false : true}>
                            <Select.Option key='0' value={0}>全部</Select.Option>
                            <Select.Option key='1' value={1}>有效</Select.Option>
                            <Select.Option key='2' value={-1}>失效</Select.Option>
                        </Select>
                    </Form.Item>
                </Col> */}
                <Col span={24}>
                    <Row style={{ width: "100%" }}>
                        <Col span={4}>
                            <div className="config-title">字段名</div>
                        </Col>
                        <Col span={4}>
                            <div className="config-title">字段类型</div>
                        </Col>
                        <Col span={4}>
                            <div className="config-title">字段Tag</div>
                        </Col>
                        <Col span={4}>
                            <div className="config-title">字段描述</div>
                        </Col>
                    </Row>

                    <Form.List name="config">
                        {(fields, { add, remove }) => (
                            <>
                                {fields.map(field => (
                                    <Row style={{ width: "100%" }} key={field.key}>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'name']} rules={[{ required: true, message: '请输入字段名' }]}>
                                                <Input disabled={props.isadd ? false : true} placeholder="请输入" />
                                            </Form.Item>
                                        </Col>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'type']} rules={[{ required: true, message: '请输入字段类型' }]}>
                                                <Select placeholder="请选择" disabled={props.isadd ? false : true}>
                                                    <Select.Option key='0' value='string'>string</Select.Option>
                                                    <Select.Option key='1' value='int'>int</Select.Option>
                                                    <Select.Option key='2' value='number'>number</Select.Option>
                                                </Select>
                                            </Form.Item>
                                        </Col>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'tag']}>
                                                <Input disabled={props.isadd ? false : true} placeholder="请输入" />
                                            </Form.Item>
                                        </Col>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'des']}>
                                                <Input disabled={props.isadd ? false : true} placeholder="请输入" />
                                            </Form.Item>
                                        </Col>
                                        <Col span={4}>
                                            <Form.Item className="config-item">
                                                <Button disabled={props.isadd ? false : true} style={{ marginLeft: "10px" }} onClick={() => {
                                                    remove(field.name)
                                                }}>删除</Button>
                                            </Form.Item>
                                        </Col>
                                    </Row>
                                ))}

                                <Form.Item>
                                    <Button disabled={props.isadd ? false : true} style={{ margin: "10px 0px", width: "66.7%" }} type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                                        增加
              </Button>
                                </Form.Item>
                            </>
                        )}
                    </Form.List>
                    {/* <Button type="dashed" style={{ marginBottom: "10px" }} onClick={() => {
                        addTextItem()
                    }} block icon={<PlusOutlined />}>
                    
                   </Button> */}
                </Col>
                <Col span={13}>
                    <Form.Item name="des" label="描述">
                        <TextArea rows={4} disabled={props.isadd ? false : true}/>
                    </Form.Item>
                </Col>
                <Col span={13}>
                    <Form.Item name="admin" label="负责人" rules={[{ required: true, message: '请输入负责人' }]}>
                        <Input disabled={props.isadd ? false : true} />
                    </Form.Item>
                </Col>
            </Row>
            {/* <Form.List name="sights">
                {(fields, { add, remove }) => (
                    <>
                        {fields.map(field => (
                            <Space key={field.key} align="baseline">
                                <Form.Item
                                    noStyle
                                    shouldUpdate={(prevValues, curValues) =>
                                        prevValues.area !== curValues.area || prevValues.sights !== curValues.sights
                                    }
                                >
                                    {() => (
                                        <Form.Item
                                            {...field}
                                            label="Sight"
                                            name={[field.name, 'sight']}
                                            rules={[{ required: true, message: 'Missing sight' }]}
                                        >
                                            <Select disabled={!form.getFieldValue('area')} style={{ width: 130 }}>
                                                {(sights[form.getFieldValue('area')] || []).map((item: any) => (
                                                    <Option key={item} value={item}>
                                                        {item}
                                                    </Option>
                                                ))}
                                            </Select>
                                        </Form.Item>
                                    )}
                                </Form.Item>
                                <Form.Item
                                    {...field}
                                    label="Price"
                                    name={[field.name, 'price']}
                                    rules={[{ required: true, message: 'Missing price' }]}
                                >
                                    <Input />
                                </Form.Item>

                                <MinusCircleOutlined onClick={() => remove(field.name)} />
                            </Space>
                        ))}

                        <Form.Item>
                            <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                                Add sights
              </Button>
                        </Form.Item>
                    </>
                )}
            </Form.List> */}
           { props.isadd ?<Form.Item>
                <Button type="primary" htmlType="submit">
                    提交
                </Button>
            </Form.Item> : null}
        </Form>
    );

}