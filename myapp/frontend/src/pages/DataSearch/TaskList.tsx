import { CaretRightOutlined, CloseCircleOutlined, CloseOutlined, DeleteOutlined, DownloadOutlined, LoadingOutlined, QuestionCircleOutlined, RedoOutlined, StopOutlined } from '@ant-design/icons';
import { Button, Col, Collapse, message, Modal, Row, Select, Spin, Steps, Tooltip } from 'antd'
import React, { useRef, useState } from 'react'
import TableBox from '../../components/TableBox/TableBox';
import { TTaskStep, IEditorTaskItem, TTaskStatus } from './interface'
import './DataSearch.less';
import { getIdexTaskDownloadUrl, getIdexTaskResult, stopIndxTask } from '../../api/dataSearchApi';
import Spreadsheet from '../../components/Spreadsheet/Spreadsheet';

export interface ITaskListItem { }

interface IProps {
    onDelete: (id: string) => void,
    onRetry: (id: string) => void,
    option: Record<string, IEditorTaskItem>
}

const stepMap: Record<TTaskStep, number> = {
    'start': 0,
    'parse': 1,
    'execute': 2,
    'end': 3
}

const statusIconMap = (status: TTaskStatus) => {
    switch (status) {
        case 'init':
        case 'running':
            return <LoadingOutlined />
        case 'failure':
            return <CloseCircleOutlined style={{ color: '#ff4444' }} />
        case 'stop':
            return <QuestionCircleOutlined />
        default:
            return null
    }
}

const handleResultData = (result: (string | number)[][]) => {
    const header = result[0] || []
    const data = result.slice(1)
    const targetData = data.map((row) => {
        const rowItem = row.reduce((pre, next, index) => ({ ...pre, [header[index]]: next }), {})
        return rowItem
    })
    // const headerConfig = header.map(item => ({
    //     title: item,
    //     dataIndex: item,
    //     key: item,
    //     width: 100,
    // }))
    targetData.unshift(header)

    return targetData
}

export default function TaskList(props: IProps) {
    const [visibleDownload, setVisibleDownload] = useState(false)
    const [visibleResult, setVisibleResult] = useState(false)
    const [dataResult, setDataResult] = useState<Record<string, any>[]>([])
    const [loadingResult, setLoadingResult] = useState(true)

    const [separator, _setSeparator] = useState<string>('|')
    const separatorRef = useRef(separator);
    const setSeparator = (data: string): void => {
        separatorRef.current = data;
        _setSeparator(data);
    };

    const [currentReqId, _setCurrentReqId] = useState<string>('|')
    const currentReqIdRef = useRef(currentReqId);
    const setCurrentReqId = (data: string): void => {
        currentReqIdRef.current = data;
        _setCurrentReqId(data);
    };

    const onChange = (key: string | string[]) => {
        console.log(key);
    };

    // const handleClickDownload = (id: any) => {
    //     Modal.confirm({
    //         title: '下载结果',
    //         icon: <DownloadOutlined />,
    //         content: <div>
    //             <div className="d-f ac pt16">
    //                 <div className="w72">分隔符：</div>
    //                 {console.log('separatorRef.current', separatorRef.current, separator)}
    //                 <Select style={{ width: 256 }} value={separatorRef.current} options={[{
    //                     label: '|',
    //                     value: '|'
    //                 }, {
    //                     label: ',',
    //                     value: ','
    //                 }, {
    //                     label: 'tab',
    //                     value: 'tab'
    //                 }]} onChange={(value) => {

    //                     // separatorRef.current = value
    //                     setSeparator(value)
    //                     _setSeparator(value)
    //                     console.log(value, separatorRef.current);
    //                 }} />
    //             </div>
    //         </div>,
    //         okText: `确认`,
    //         cancelText: '取消',
    //         onOk() {
    //             return new Promise((resolve, reject) => {
    //                 getIdexTaskDownloadUrl(id).then(res => {
    //                     window.open(res.data.download_url, 'bank')
    //                     resolve('')
    //                 }).catch(err => {
    //                     reject()
    //                 })
    //             })
    //                 .then((res) => {
    //                     message.success('下载成功');
    //                 })
    //                 .catch(() => {
    //                     message.error('下载失败');
    //                 });
    //         },
    //         onCancel() { },
    //     });
    // }

    return (
        <div>
            <Modal
                title={`下载结果`}
                visible={visibleResult}
                footer={null}
                width={1248}
                destroyOnClose
                onCancel={() => {
                    setVisibleResult(false)
                }}>
                <Spin spinning={loadingResult}>
                    <Spreadsheet height={700} width={1200} dataSource={dataResult} />
                </Spin>
            </Modal>

            <Modal
                title={`结果`}
                visible={visibleDownload}
                footer={null}
                destroyOnClose
                onCancel={() => {
                    setSeparator('|')
                    setVisibleDownload(false)
                }}>
                <div>
                    <div className="d-f ac pt8">
                        <div className="w96">选择分隔符：</div>
                        <Select style={{ width: 256 }} value={separatorRef.current} options={[{
                            label: '|',
                            value: '|'
                        }]} onChange={(value) => {
                            setSeparator(value)
                        }} />
                    </div>
                    <div className="ta-r pt16">
                        <Button type="primary" onClick={() => {
                            getIdexTaskDownloadUrl(currentReqIdRef.current, separatorRef.current).then(res => {
                                window.open(res.data.download_url, 'bank')
                            }).catch(err => {
                                console.log(err)
                            })
                        }}>下载</Button>
                    </div>
                </div>
            </Modal>
            <Collapse className="site-collapse-custom-collapse" defaultActiveKey={['task_0']} onChange={onChange}>
                {
                    (Object.entries(props.option).reduce((pre: IEditorTaskItem[], [key, value]) => ([...pre, value]), []) || []).reverse().filter(item => !!item.reqId).map((item, index) => {
                        return (
                            <Collapse.Panel className={['site-collapse-custom-panel', `status-${item.status}`].join(' ')} header={`子任务${item.reqId}`} key={`task_${index}`} extra={
                                <>
                                    <Button className="mr16" type="default" size='small' onClick={(e) => {
                                        e.stopPropagation();
                                        props.onDelete(item.reqId)
                                    }}>删除<DeleteOutlined /></Button>
                                    <Button type="primary" size='small' onClick={(e) => {
                                        e.stopPropagation();
                                        props.onRetry(item.reqId)
                                    }}
                                    >重试<RedoOutlined /></Button>
                                </>
                            }>
                                <Steps size="small" current={stepMap[item.step]}>
                                    <Steps.Step title="准备开始" icon={stepMap[item.step] === 0 ? statusIconMap(item.status) : null} />
                                    <Steps.Step title="解析" icon={stepMap[item.step] === 1 ? statusIconMap(item.status) : null} />
                                    <Steps.Step title="执行" icon={stepMap[item.step] === 2 ? statusIconMap(item.status) : null} />
                                    <Steps.Step title="输出结果" icon={stepMap[item.step] === 3 ? statusIconMap(item.status) : null} />
                                </Steps>
                                <TableBox
                                    size={"small"}
                                    loading={false}
                                    cancelExportData={true}
                                    rowKey={(record: any) => {
                                        return JSON.stringify(record)
                                    }}
                                    columns={[{
                                        title: '子任务',
                                        dataIndex: 'content',
                                        key: 'content',
                                        render: (text: any) => {
                                            return <Tooltip
                                                placement="top"
                                                title={text}
                                            >
                                                <div className="ellip1 w256">{text}</div>
                                            </Tooltip>
                                        }
                                    },
                                    // {
                                    //     title: '数据库',
                                    //     dataIndex: 'database',
                                    //     key: 'database',
                                    // }, {
                                    //     title: '表',
                                    //     dataIndex: 'table',
                                    //     key: 'table',
                                    // },
                                    {
                                        title: '开始时间',
                                        dataIndex: 'startime',
                                        key: 'startime',
                                    }, {
                                        title: '运行时长',
                                        dataIndex: 'duration',
                                        key: 'duration',
                                    }, {
                                        title: '状态',
                                        dataIndex: 'status',
                                        key: 'status',
                                        render: (text: any) => {
                                            return <span className={[`c-${item.status}`].join(' ')}>{text}</span>
                                        }
                                    }, {
                                        title: '操作',
                                        dataIndex: 'action',
                                        key: 'action',
                                        render: () => {
                                            return <>
                                                <span className="link mr16" onClick={() => {
                                                    // setVisibleDetail(true)
                                                    Modal.info({
                                                        title: '任务详情',
                                                        width: 600,
                                                        okText: '关闭',
                                                        content: (
                                                            <div>
                                                                <Row className="mb16">
                                                                    <Col span={6}><div className="ta-r"><strong>开始时间：</strong></div></Col>
                                                                    <Col span={18}>{item.startTime}</Col>
                                                                </Row>
                                                                <Row className="mb16">
                                                                    <Col span={6}><div className="ta-r"><strong>运行时长：</strong></div></Col>
                                                                    <Col span={18}>{item.duration}</Col>
                                                                </Row>
                                                                <Row className="mb16">
                                                                    <Col span={6}><div className="ta-r"><strong>状态：</strong></div></Col>
                                                                    <Col span={18}>{item.status}</Col>
                                                                </Row>
                                                                <Row className="mb16">
                                                                    <Col span={6}><div className="ta-r"><strong>子任务内容：</strong></div></Col>
                                                                    <Col span={18}>{item.content}</Col>
                                                                </Row>
                                                                <Row className="mb16">
                                                                    <Col span={6}><div className="ta-r"><strong>任务信息：</strong></div></Col>
                                                                    <Col span={18}>{item.message}</Col>
                                                                </Row>
                                                            </div>
                                                        ),
                                                        onOk() { },
                                                    });
                                                }}>详情</span>
                                                {
                                                    !(item.step === 'end' && item.status === 'success') ? <span className="link mr16" onClick={() => {
                                                        Modal.confirm({
                                                            title: '终止任务',
                                                            icon: <StopOutlined />,
                                                            content: '',
                                                            okText: `确认`,
                                                            cancelText: '取消',
                                                            onOk() {
                                                                return new Promise((resolve, reject) => {
                                                                    stopIndxTask(item.reqId).then(res => {
                                                                        resolve('')
                                                                    }).catch(err => {
                                                                        reject()
                                                                    })
                                                                })
                                                                    .then((res) => {
                                                                        message.success('终止成功');
                                                                    })
                                                                    .catch(() => {
                                                                        message.error('终止失败');
                                                                    });
                                                            },
                                                            onCancel() { },
                                                        });
                                                    }}>终止</span> : null
                                                }
                                                {
                                                    !!item.log ? <span className="link mr16" onClick={() => {
                                                        window.open(item.log, 'bank')
                                                    }}>日志</span> : null
                                                }
                                                {
                                                    item.step === 'end' && item.status === 'success' ? <span className="link mr16" onClick={() => {
                                                        // Modal.info({
                                                        //     title: '结果查看',
                                                        //     content: (
                                                        //         <div>
                                                        //             {item.result?.map(item => {
                                                        //                 return <div>{item}</div>
                                                        //             })}
                                                        //         </div>
                                                        //     ),
                                                        //     onOk() { },
                                                        // });
                                                        setLoadingResult(true)
                                                        getIdexTaskResult(item.reqId).then(res => {
                                                            setVisibleResult(true)
                                                            const result = res.data.result
                                                            const handleData = handleResultData(result)
                                                            setDataResult(handleData)
                                                        }).catch(err => { }).finally(() => {
                                                            setLoadingResult(false)
                                                        })
                                                    }}>结果</span> : null
                                                }
                                                {
                                                    item.step === 'end' && item.status === 'success' ? <span className="link" onClick={() => {
                                                        // handleClickDownload(item.reqId)
                                                        setCurrentReqId(item.reqId)
                                                        setVisibleDownload(true)
                                                    }}>下载</span> : null
                                                }
                                            </>
                                        }
                                    }]}
                                    pagination={false}
                                    dataSource={[{
                                        content: item.content,
                                        database: item.database || '-',
                                        table: item.table || '-',
                                        startime: item.startTime,
                                        duration: item.duration || '-',
                                        status: item.status
                                    }]}
                                />
                            </Collapse.Panel>
                        )
                    })
                }

                {/* <Collapse.Panel className="site-collapse-custom-panel status-error" header="子任务2" key="2" >
                    <Steps size="small" current={1}>
                        <Steps.Step title="准备开始" />
                        <Steps.Step title="解析中" icon={<LoadingOutlined />} />
                        <Steps.Step title="执行中" />
                        <Steps.Step title="输出结果" />
                    </Steps>
                    <TableBox
                        size={"small"}
                        loading={false}
                        cancelExportData={true}
                        rowKey={(record: any) => {
                            return JSON.stringify(record)
                        }}
                        columns={[{
                            title: '子任务',
                            dataIndex: 'task',
                            key: 'task',
                        }, {
                            title: '开始时间',
                            dataIndex: 'startime',
                            key: 'startime',
                        }, {
                            title: '运行时长',
                            dataIndex: 'time',
                            key: 'time',
                        }, {
                            title: '状态',
                            dataIndex: 'status',
                            key: 'status',
                        }, {
                            title: '操作',
                            dataIndex: 'action',
                            key: 'action',
                            render: () => {
                                return <>
                                    <span className="link mr16">详情</span>
                                    <span className="link mr16">日志</span>
                                    <span className="link">结果</span>
                                </>
                            }
                        }]}
                        pagination={false}
                        dataSource={[{
                            task: 'test',
                            startime: '2022-08-09 17:45:46',
                            time: '3分钟43秒',
                            status: 'error'
                        }]}
                    />
                </Collapse.Panel>
                <Collapse.Panel className="site-collapse-custom-panel status-running" header="子任务3" key="3">
                    <p>{345}</p>
                </Collapse.Panel> */}
            </Collapse>
        </div>
    )
}
