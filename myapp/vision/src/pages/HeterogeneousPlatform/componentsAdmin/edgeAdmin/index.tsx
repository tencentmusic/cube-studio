import React, { useState, useEffect } from 'react';
import {
    Button,
    Table, Space, Popconfirm,
    Modal,
    message,
    Card,
    Tag
} from 'antd';
import {
    LinkOutlined,
} from '@ant-design/icons';
import QueryForm from "../components/QueryForm/index"
import EditForm from "../components/EditForm/index"
import api from "@src/api/admin"
import { formRolue, form } from "./data.js"
import { STATUS, STATUS_COLOR } from "../common"
import './index.css';

export default function edgeAdmin() {
    const [showEdit, setEditStatus] = useState(false)
    const [tableData, setTableData] = useState([])
    const [viewForm, setViewForm] = useState({})

    const deleteNode = (record: any) => {
        const data = {
            opr: "delete_edge_factory",
            data: {
                type: record.type,
                factory_name: record.factory_name,
            }
        }
        api.delete_edge_factory(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }
    const columns = [
        {
            title: '#',
            dataIndex: 'bussiness_id',
            key: 'bussiness_id',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: '边工厂名',
            fType: "input",
            dataIndex: 'factory_name',
            key: 'factory_name',
        },
        {
            title: '文档地址',
            fType: "input",
            dataIndex: 'url',
            key: 'url',
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
        },
        {
            title: '状态',
            dataIndex: 'status',
            key: 'status',
            width: 100,
            render: (text: any) => STATUS[text] ? STATUS[text] : ""
        },
        {
            title: 'Form节点',
            fType: "input",
            dataIndex: 'from_node_factory',
            key: 'from_node_factory',
        },
        {
            title: 'To节点',
            fType: "input",
            dataIndex: 'to_node_factory',
            key: 'to_node_factory',
        },
        {
            title: '负责人',
            fType: "input",
            dataIndex: 'admin',
            key: 'admin',
        },
        {
            title: '描述',
            fType: "textArea",
            dataIndex: '"des',
            key: '"des',
        },
        {
            title: '更新时间',
            fType: "input",
            dataIndex: 'updated_time',
            key: 'updated_time',
        },
        {
            title: '操作',
            key: 'action',
            render: (text: any, record: any) => (
                <Space size="middle">
                    <a key="view" onClick={() => {
                        setViewForm({ ...record })
                        setEditStatus(true)
                    }}>查看</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <Popconfirm title="是否删除?" okText="确认" cancelText="取消" onConfirm={() => {
                        deleteNode(record)
                    }}>
                        <a key="del" href="#">删除</a>
                    </Popconfirm>,
                </Space>
            ),
        },
    ];

    const viewFormRolue = columns.filter((item: any) => item.fType).map((item: any) => {
        return {
            type: item.fType,
            name: item.dataIndex,
            label: item.title
        }
    })

    const queryList = (params: any) => {
        const { admin, factory_name, status, time, type, from_node_factory, to_node_factory } = params
        const data = {
            from_node_factory: from_node_factory || "", // Form节点
            to_node_factory: to_node_factory || "", // To节点
            admin: admin || "", // 负责人
            factory_name: factory_name || "",
            status: status || 0,
            type: type || "",
            min_updated_time: time && time[0] ? new Date(time[0]).getTime() : 0,
            max_updated_time: time && time[1] ? new Date(time[1]).getTime() : 0,
        }
        api.query_edge_factorys({ opr: "query_edge_factorys", data }).then(res => {
            if (res.status === 0) {
                setTableData(res.data.results)
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }
    useEffect(() => {
        queryList({})
    }, [])

    return (
        <div style={{ padding: "30px 40px" }}>
            <div>
                <Card title="查询条件">
                    <QueryForm formRolue={formRolue} form={form} queryList={queryList}></QueryForm>
                </Card>

                <Card title="查询结果" style={{ marginTop: "20px" }}>
                    <Table key="bussiness_id" columns={columns} dataSource={tableData} />
                </Card>

            </div>
            <Modal title="Basic Modal" visible={showEdit} onCancel={() => { setEditStatus(false) }} footer={[
                <Button key="back" onClick={() => {
                    setEditStatus(false)
                }}>关闭</Button>
            ]}>
                <EditForm formRolue={viewFormRolue} form={viewForm} disableds={true} />
            </Modal>

        </div>
    );
}
