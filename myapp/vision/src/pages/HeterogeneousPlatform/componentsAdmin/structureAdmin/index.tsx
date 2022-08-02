import React, { useState, useEffect } from 'react';
import {
    Button,
    Table, Space, Popconfirm,
    Modal,
    message,
    Card,
    Tag
} from 'antd';
import QueryForm from "../components/QueryForm/index"
import StructureEditForm from "../components/StructureEditForm/index"
import api from "@src/api/admin"
import { formRolue, form } from "./data.js"
import { STATUS, STATUS_COLOR } from "../common"
import './index.css';

export default function structureAdmin() {
    const [showEdit, setEditStatus] = useState(false)
    const [tableData, setTableData] = useState([])
    const [chartTableData, setChartTableData] = useState([])
    const [viewForm, setViewForm] = useState({})
    const [isadd, setIsadd] = useState(false)
    const [showChart, setShowChart] = useState(false)

    const relationNode = (record: any) => {
        const data = {
            opr: "query_associated_nodes",
            data: {
                associated_nodes: record.associated_nodes,
            }
        }
        api.query_associated_nodes(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                setChartTableData(res.data.results)
                setShowChart(true)
                message.success(res.message);
            } else {
                setTableData([])
                message.error(res.message);
            }
        })
    }


    const newAdd = () => {
        setIsadd(true)
        setViewForm({ name: "" })
        setEditStatus(true)
    }
    const nodeColumns = [
        {
            title: '#',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: "类型",
            dataIndex: 'type',
            key: 'type',
        },
        {
            title: '版本',
            dataIndex: 'version',
            key: 'version',
        },
        {
            title: '工厂名',
            dataIndex: 'factory_name',
            key: 'factory_name',
        },
        {
            title: '名称',
            dataIndex: 'name',
            key: '"name',
        },
        {
            title: '状态',
            dataIndex: 'status',
            key: 'status',
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
        },
        {
            title: '负责人',
            dataIndex: 'admin',
            key: 'admin',
        },
        {
            title: '描述',
            dataIndex: 'des',
            key: 'des',
        },
        {
            title: '更新时间',
            dataIndex: 'updated_time',
            key: 'updated_time',
        },
    ]



    const columns = [
        {
            title: '#',
            dataIndex: 'bussiness_id',
            key: 'bussiness_id',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: '结构体名',
            fType: "input",
            dataIndex: 'struct_name',
            key: 'struct_name',
        },
        {
            title: '状态',
            dataIndex: 'status',
            key: 'status',
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
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

                        const arr = []
                        const config = JSON.parse(record.config)
                        console.log(config)
                        for (const key in config) {
                            arr.push({
                                name: key,
                                type: config[key].type,
                                tag: config[key].tag,
                                json: config[key].json,
                                des: config[key].des
                            })
                        }
                        const data = {
                            struct_name: record.struct_name,
                            status: record.status,
                            des: record.des,
                            admin: record.admin,
                            config: arr
                        }
                        setIsadd(false)
                        setViewForm(data)
                        setEditStatus(true)
                    }}>查看</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="edit" onClick={() => {

                        const arr = []
                        const config = JSON.parse(record.config)
                        console.log(config)
                        for (const key in config) {
                            arr.push({
                                name: key,
                                type: config[key].type,
                                tag: config[key].tag,
                                json: config[key].json,
                                des: config[key].des
                            })
                        }
                        const data = {
                            struct_name: record.struct_name,
                            status: record.status,
                            des: record.des,
                            admin: record.admin,
                            config: arr,
                            version: record.version
                        }
                        setIsadd(true)
                        setViewForm(data)
                        setEditStatus(true)

                    }}>编辑</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <Popconfirm title="是否删除?" okText="确认" cancelText="取消" onConfirm={() => {
                        deleteNode(record)
                    }}>
                        <a key="del">删除</a>
                    </Popconfirm>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    {
                      record.associated_nodes && record.associated_nodes.length ? <a key="node" onClick={() => { relationNode(record) }} > 关联节点 ({record.associated_nodes && record.associated_nodes.length ? record.associated_nodes.length : 0} )</a> : <span style={{color: "#999"}}>关联节点</span>
                    }
                    
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
        const { admin, struct_name, status, time } = params
        const data = {
            struct_name: struct_name || "",
            admin: admin || "", // 负责人
            status: status || 0,
            min_updated_time: time && time[0] ? new Date(time[0]).getTime() : 0,
            max_updated_time: time && time[1] ? new Date(time[1]).getTime() : 0,
        }
        api.query_structs({ opr: "query_structs", data }).then(res => {
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


    const deleteNode = (record: any) => {
        const data = {
            opr: "delete_struct",
            data: {
                struct_name: record.struct_name,
            }
        }
        api.delete_struct(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                message.success(res.message);
                queryList({})
            } else {
                message.error(res.message);
            }
        })
    }

    const resFied = () => {
        setEditStatus(false)
        queryList({})
    }


    return (
        <div style={{ padding: "30px 40px" }}>
            <div>
                <Card title="查询条件">
                    <QueryForm formRolue={formRolue} form={form} queryList={queryList}></QueryForm>
                </Card>

                <Card title="查询结果" extra={<Button type="primary" onClick={() => newAdd()}>新增</Button>} style={{ marginTop: "20px" }}>
                    <Table key="struct_name" columns={columns} dataSource={tableData} />
                </Card>

            </div>
            <Modal width={900} title="" visible={showEdit} onCancel={() => { setEditStatus(false) }} footer={[
                <Button key="back" onClick={() => {
                    setEditStatus(false)
                }}>关闭</Button>
            ]}>
                <StructureEditForm formRolue={viewFormRolue} form={viewForm} isadd={isadd} disableds={true} resFied={resFied} />
            </Modal>

            <Modal width={"60%"} title="关联节点" visible={showChart} onCancel={() => { setShowChart(false) }} footer={[
                <Button key="back" onClick={() => {
                    setShowChart(false)
                }}>关闭</Button>
            ]}>
               <div style={{width: "100%", overflowY: "auto"}}>
                 <Table key="struct_name"  columns={nodeColumns} dataSource={chartTableData} />
                </div>
            </Modal>

        </div>
    );
}
