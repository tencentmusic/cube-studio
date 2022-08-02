import React, { useEffect, useRef } from 'react';
import { Form, Input, Button, Select, DatePicker, Radio, Row, Col } from 'antd';

const { RangePicker } = DatePicker;

export default function EditForm(props: any) {
    const onFinishFailed = () => {
        console.log("失败")
    }
    const formRef = useRef(null)
    useEffect(() => {
        console.log("触发了", (formRef.current as any).setFieldsValue(props.form))
    }, [])
    const onFinish = (form: any) => {
        props.queryList(form)
    }
    return (
        <div>
            <Form
                name="basic"
                labelCol={{ span: 8 }}
                wrapperCol={{ span: 16 }}
                initialValues={{ remember: true }}
                onFinish={onFinish}
                onFinishFailed={onFinishFailed}
                autoComplete="off"
                ref={formRef}
            >
                <Row>
                    {
                        props.formRolue.map((item: any) => {
                            switch (item.type) {
                                case "input":
                                    return (<Col span={6}>
                                        <Form.Item
                                            key={item.name}
                                            label={item.label}
                                            name={item.name}
                                        >
                                            <Input />
                                        </Form.Item>
                                    </Col>)
                                    break;
                                case "select":
                                    return (
                                        <Col span={6}>
                                            <Form.Item
                                                key={item.name}
                                                label={item.label}
                                                name={item.name}
                                            >
                                                <Select>
                                                    {
                                                        item.options ? item.options.map((select: any) => {
                                                            return <Select.Option key={select.value} value={select.value}>{select.label}</Select.Option>
                                                        }) : null
                                                    }
                                                </Select>
                                            </Form.Item></Col>
                                    )
                                    break;
                                case "picker":
                                    return (
                                        <Col span={6}>
                                            <Form.Item
                                                key={item.name}
                                                label={item.label}
                                                name={item.name}
                                            >
                                                <DatePicker style={{ width: '100%' }} />
                                            </Form.Item></Col>
                                    )
                                    break;
                                case "radio":
                                    return (
                                        <Col span={6}>
                                            <Form.Item
                                                key={item.name}
                                                label={item.label}
                                                name={item.name}
                                            >
                                                <Radio.Group>
                                                    <Radio value={1}>A</Radio>
                                                    {
                                                        item.options ? item.options.map((radio: any) => {
                                                            return <Radio key={radio.value} value={radio.value}>{radio.label}</Radio>
                                                        }) : null
                                                    }
                                                </Radio.Group>
                                            </Form.Item></Col>
                                    )
                                    break;
                                case "rangePicker":
                                    return (
                                        <Col span={6}>
                                            <Form.Item
                                                key={item.name}
                                                label={item.label}
                                                name={item.name}
                                            >
                                                <RangePicker />
                                            </Form.Item></Col>
                                    )
                                    break;
                                default:
                                    null
                            }
                        })
                    }
                    <Col span={6}>
                        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                            <Button type="primary" htmlType="submit">
                                查询
        </Button>
                        </Form.Item></Col>
                </Row>

            </Form>
        </div>
    )
}