import React, {  useEffect, useState } from 'react';
import { Form, Input, Button, Select, Row, Col, message,  } from 'antd';
import {  PlusOutlined } from '@ant-design/icons';
import "./index.css"
const { Option } = Select;
const { TextArea } = Input


import api from "@src/api/admin"

export default function ABTestEdit(props: any) {
    const [form] = Form.useForm();
    const [version, setVerion] = useState([])

    useEffect(() => {
        if(props.form.scene_name) {
            props.scene_nameChange(props.form.scene_name)
        }
        form.resetFields();
        form.setFieldsValue(props.form);
    }, [props.form])
     

    const getVersion = (name: any)=> {
       if(props.chartArr.arr) {
        setVerion(props.chartArr.arr[name])
       }
    }

    const onFinish = (record: any) => {

        const data: any = {
            abtest_name: record.abtest_name,
            bussiness_id: record.bussiness_id,
            channel_id: record.channel_id,
            module_id: record.module_id,
            scene_name: record.scene_name,
            client_version: record.client_version,
            status: 1,
            admin: record.admin,
            des: record.des,
            config: record.config ? JSON.stringify(record.config) : [],
            associated_graphs: record.config ? JSON.stringify(record.config.map((item: any) => {
                return {
                    graph_name: item.graph_name,
                    version: item.version
                }
            })) : []
            // config: JSON.stringify(obj)
        }
        if (props.form.version) {
            data.version = props.form.version + 1
            api.mod_abtest({ opr: "mod_abtest", data }).then(res => {
                console.log(res)
                if (res.status === 0) {
                    message.success(res.message);
                    props.resFied()
                } else {
                    message.error(res.message);
                }
            })
        } else {
            data.version =  1
            api.add_abtest({ opr: "add_abtest", data }).then(res => {
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
        <Form form={form} name="dynamic_form_nest_item" className="abtset-form" labelAlign="left" labelCol={{ span: 5 }} onFinish={onFinish} autoComplete="off">
            <Row gutter={10}>
                <Col span={13}>
                    <Form.Item name="abtest_name" label="ABTest名" rules={[{ required: true, message: '请输入ABTest名' }]}>
                        <Input disabled={props.isadd === false} />
                    </Form.Item>
                </Col>
                <Col span={13}>
                    <Form.Item name="bussiness_id" label="业务id" rules={[{ required: true, message: '请输入业务id' }]}>
                        <Input disabled={props.isadd === false} />
                    </Form.Item>
                </Col>
                <Col span={13}>
                    <Form.Item name="channel_id" label="频道id" rules={[{ required: true, message: '请输入频道id' }]}>
                        <Input disabled={props.isadd === false} />
                    </Form.Item>
                </Col>
                <Col span={13}>
                    <Form.Item name="module_id" label="模块id" rules={[{ required: true, message: '请输入模块id' }]}>
                        <Input disabled={props.isadd === false} />
                    </Form.Item>
                </Col>
                <Col span={13}>
                    <Form.Item name="client_version" label="客户端版本">
                        <Input disabled={props.isadd === false} />
                    </Form.Item>
                </Col>
                <Col span={13}>
                    <Form.Item name="scene_name" label="场景" rules={[{ required: true, message: '请输入负责人' }]}>
                        <Select disabled={props.isadd === false} onChange={(e) => {
                           const arr:any = form.getFieldValue('config')
                           console.log(arr)
                           arr.forEach((item:any) => {
                               item.graph_name = ""
                               item.version = ""
                           })
                           form.setFieldsValue({config: arr});
                            props.scene_nameChange(e)
                        }}>

                            {
                                props.sceneArr.map((item: any, index: any) => (
                                    <Select.Option key={index} value={item}>{item}</Select.Option>
                                ))
                            }
                            {/* <Select.Option key='1' value={1}>有效</Select.Option>
                            <Select.Option key='2' value={-1}>失效</Select.Option> */}
                        </Select>
                    </Form.Item>
                </Col>
                {/* <Col span={13}>
                    <Form.Item name="status" label="状态" rules={[{ required: true, message: '请选择状态' }]}>
                        <Select disabled={props.isadd === false} >
                            <Select.Option key='0' value={0}>全部</Select.Option>
                            <Select.Option key='1' value={1}>有效</Select.Option>
                            <Select.Option key='2' value={-1}>失效</Select.Option>
                        </Select>
                    </Form.Item>
                </Col> */}
                <Col span={13}>
                    <Form.Item name="admin" label="负责人" rules={[{ required: true, message: '请输入负责人' }]}>
                        <Input disabled={props.isadd === false} />
                    </Form.Item>
                </Col>
                <Col span={24}>
                    <Row style={{ width: "100%" }}>
                        <Col span={4}>
                            <div className="config-title">参数名</div>
                        </Col>
                        <Col span={4}>
                            <div className="config-title">参数值</div>
                        </Col>
                        <Col span={4}>
                            <div className="config-title">召回图名</div>
                        </Col>
                        <Col span={4}>
                            <div className="config-title">版本</div>
                        </Col>
                        {/* <Col span={4}>
                            <div className="config-title">默认配置</div>
                        </Col> */}
                        {/* <Col span={4}>
                            <div className="config-title">操作</div>
                        </Col> */}
                    </Row>
                    <Form.List name="config">
                        {(fields, { add, remove }) => (
                            <>
                                {fields.map((field, index) => (

                                    <Row style={{ width: "100%" }} key={field.key}>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'param_name']} rules={[{ required: true, message: '请输入参数名' }]}>
                                                <Input placeholder="请输入" disabled={props.isadd === false} />
                                            </Form.Item>
                                        </Col>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'param_value']} rules={[{ required: true, message: '请输入参数值' }]}>
                                                <Input disabled={props.isadd === false} placeholder="请输入" />
                                            </Form.Item>
                                        </Col>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'graph_name']} rules={[{ required: true, message: '请选择召回图名' }]}>
                                                <Select placeholder="请选择" disabled={props.isadd === false} onChange={()=>{
                                                    console.log(form.getFieldValue('config'), index)
                                                    getVersion(form.getFieldValue('config')[index].graph_name)
                                                }}>

                                                    {
                                                        props.chartArr && props.chartArr.graphArr.map((item: any) => (
                                                            <Select.Option key={item} value={item}>{item}</Select.Option>
                                                        ))
                                                    }
                                                    {/* <Select.Option key='1' value='int'>int</Select.Option>
                                                    <Select.Option key='2' value='number'>number</Select.Option> */}
                                                </Select>
                                            </Form.Item>
                                        </Col>
                                        <Col span={4}>
                                            <Form.Item className="config-item" name={[field.name, 'version']} rules={[{ required: true, message: '请选择版本' }]}>
                                                <Select placeholder="请选择" disabled={props.isadd === false} >
                                                    {
                                                       version ? version.map((item: any) => (
                                                            <Select.Option key={item} value={item}>{item}</Select.Option>
                                                        )) : null
                                                    }
                                                </Select>
                                            </Form.Item>
                                        </Col>
                                        {/* <Col span={4}>
                                                <Form.Item name={[field.name, 'tag']}>
                                                    <Radio value={index}></Radio>
                                                </Form.Item>
                                            </Col> */}
                                        <Col span={4}>
                                            <Form.Item className="config-item">
                                                <Button disabled={props.isadd === false} style={{ marginLeft: "10px" }} onClick={() => {
                                                    remove(field.name)
                                                }}>删除</Button>
                                            </Form.Item>
                                        </Col>
                                    </Row>
                                ))}

                                <Form.Item>
                                    <Button disabled={props.isadd === false} style={{ margin: "10px 0px", width: "66.7%" }} type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                                        增加
                                         </Button>
                                </Form.Item>
                            </>
                        )}

                    </Form.List>

                </Col>
                <Col span={13}>
                    <Form.Item name="des" label="描述">
                        <TextArea disabled={props.isadd === false} rows={4} />
                    </Form.Item>
                </Col>
            </Row>
            { props.isadd ? <Form.Item>
                <Button type="primary" htmlType="submit">
                    提交
                </Button>
            </Form.Item> : null}
        </Form>
    );

}