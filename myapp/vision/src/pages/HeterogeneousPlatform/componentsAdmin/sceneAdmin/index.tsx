import React, { useState, useEffect } from 'react';
import {
    Button,
    Table, Space,
    Modal,
    message,
    Card,
    Tag
} from 'antd';
import QueryForm from "../components/QueryForm/index"
import EditForm from "../components/EditForm/index"
import api from "@src/api/admin"
import { formRolue, form } from "./data.js"
import { STATUS, STATUS_COLOR } from "../common"
import './index.css';

export default function sceneAdmin() {
    const [showEdit, setEditStatus] = useState(false)
    const [tableData, setTableData] = useState([])
    const [chartTableData, setChartTableData] = useState([])
    const [viewForm, setViewForm] = useState<any>({})
    const [isadd, setIsadd] = useState(false)
    const [showChart, setShowChart] = useState(false)

    const relationNode = (record: any) => {
        const data = {
            opr: "query_associated_graphs",
            data: {
                associated_graphs: record.associated_graphs,
            }
        }
        api.query_associated_graphs(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                setChartTableData(res.data.results)
                setShowChart(true)
                message.success(res.message);
            } else {
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
            title: '场景名',
            fType: "input",
            dataIndex: 'scene_name',
            key: 'scene_name',
        },
        {
            title: '调用地址',
            fType: "input",
            dataIndex: 'address',
            key: 'address',
        },
        {
            title: '负责人',
            fType: "input",
            dataIndex: 'admin',
            key: '"admin',
        },
        {
            title: '描述',
            fType: "textArea",
            dataIndex: 'des',
            key: 'des',
        },
        {
            title: '更新时间',
            dataIndex: 'updated_time',
            key: 'updated_time',
        },
        {
            title: '操作',
            key: 'action',
            render: (text: any, record: any) => (
                <Space size="middle">
                    <a key="view" onClick={() => {

                        setIsadd(false)
                        setViewForm({ ...record })
                        setEditStatus(true)
                    }}>查看</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="edit" onClick={() => {
                        console.log(record)
                        setIsadd(true)
                        setViewForm({ ...record })
                        setEditStatus(true)

                    }}>编辑</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    {/* <Popconfirm title="是否删除?" okText="确认" cancelText="取消" onConfirm={() => {
                        deleteNode(record)
                    }}>
                        <a key="del">删除</a>
                    </Popconfirm> */}
                    {/* <a key="node" onClick={() => { relationNode(record) }}> 关联图 </a> */}
                    {
                      record.associated_graphs && record.associated_graphs.length ? <a key="node" onClick={() => { relationNode(record) }} > 关联图 ({record.associated_graphs && record.associated_graphs.length ? record.associated_graphs.length : 0} )</a> : <span style={{color: "#999"}}>关联图</span>
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
        const { admin, scene_name, address, time } = params
        const data = {
            scene_name: scene_name || "",
            address: address || "", // 负责人
            admin: admin || "",
            min_updated_time: time && time[0] ? new Date(time[0]).getTime() : 0,
            max_updated_time: time && time[1] ? new Date(time[1]).getTime() : 0,
        }
        api.query_scenes({ opr: "query_scenes", data }).then(res => {
            if (res.status === 0) {
                setTableData(res.data.results)
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }

    const submitForm = (obj: any) => {
        const data: any = {
            ...obj
        }
        if (viewForm.version !== undefined) {
            data.version = viewForm.version + 1
            api.mod_scene({ opr: "mod_scene", data }).then(res => {
                if (res.status === 0) {
                    queryList({})
                    setEditStatus(false)
                    message.success(res.message);
                } else {
                    message.error(res.message);
                }
            })
        } else {
            api.add_scene({ opr: "add_scene", data }).then(res => {
                if (res.status === 0) {
                    queryList({})
                    setEditStatus(false)
                    message.success(res.message);
                } else {
                    message.error(res.message);
                }
            })
        }
    }


    useEffect(() => {
        queryList({})
    }, [])

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
                    <Table key="scene_name" columns={columns} dataSource={tableData} />
                </Card>

            </div>
            <Modal width={900} title={isadd ? isadd && viewForm.version ? "编辑" : "新增" : "查看"} visible={showEdit} onCancel={() => { setEditStatus(false) }} footer={[
                <Button key="back" onClick={() => {
                    setEditStatus(false)
                }}>关闭</Button>
            ]}>
                <EditForm formRolue={viewFormRolue} form={viewForm} isadd={isadd} showEdit={showEdit} submitForm={submitForm} resFied={resFied} />
            </Modal>

            <Modal width={"60%"} title="关联图" visible={showChart} onCancel={() => { setShowChart(false) }} footer={[
                <Button key="back" onClick={() => {
                    setShowChart(false)
                }}>关闭</Button>
            ]}>
                <Table key="struct_name" columns={nodeColumns} dataSource={chartTableData} />
            </Modal>

        </div>
    );
}
