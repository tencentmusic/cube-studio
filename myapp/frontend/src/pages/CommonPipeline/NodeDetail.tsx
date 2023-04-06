import { Button, Col, message, Row, Tabs } from 'antd'
import React, { useEffect, useRef, useState } from 'react'
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { CopyOutlined } from '@ant-design/icons';
import TableBox from '../../components/TableBox/TableBox';
import { INodeDetailItem, TGroupContentType } from './TreePlusInterface';
import EchartCore from '../../components/EchartCore/EchartCore';
import { getNodeInfoApi } from '../../api/commonPipeline';
import './NodeDetail.less';
import { group } from 'd3';

interface IProps {
    data: INodeDetailItem[]
}

export type ITNodeDetail = 'common' | 'table' | 'sql'

export default function NodeDetail(props: IProps) {
    const [isSqlVisable, setIsSqlVisable] = useState(false)
    const [currentDataItem, setCurrentDataItem] = useState<INodeDetailItem>()

    const [nodeInfoApiMap, _setNodeInfoApiMap] = useState<Record<string, any>>({})
    const nodeInfoApiMapRef = useRef(nodeInfoApiMap);
    const setNodeInfoApiMap = (data: Record<string, any>): void => {
        nodeInfoApiMapRef.current = data;
        _setNodeInfoApiMap(data);
    };

    useEffect(() => {
        const apiList: any[] = []
        const nameList: string[] = []
        props.data.forEach(tab => {
            tab.content.forEach(group => {
                if (group.groupContent.type === 'api') {
                    const req = getNodeInfoApi(group.groupContent.value)
                    apiList.push(req)
                    nameList.push(`${tab.tabName}_${group.groupName}`)
                }
            })
        })
        console.log('apiList', apiList);
        Promise.all(apiList).then(res => {
            const result = res.map(item => item.data.result)
            const resMap = result.reduce((pre, next, index) => ({ ...pre, [nameList[index]]: { ...next } }), {})
            setNodeInfoApiMap(resMap)
        })
    }, [props.data])

    const handelNodeDetailTable = (item: any) => {
        try {
            JSON.parse(`${item.value || []}`)
        } catch (error) {
            console.log(error);
        }
        const dataList = JSON.parse(`${item.value || '[]'}`)
        let columnsConfig = Object.entries(dataList[0] || {}).reduce((pre: any, [key, value]) => [...pre, { title: key, dataIndex: key, key }], [])

        return <Col span={16}>
            <TableBox
                rowKey={(record: any) => {
                    return JSON.stringify(record)
                }}
                size={'small'}
                cancelExportData={true}
                columns={columnsConfig}
                pagination={false}
                dataSource={dataList}
            // scroll={{ x: 1500, y: scrollY }}
            />
        </Col>
    }

    const handleNodeDetail = (item: any) => {
        switch (item.type) {
            case 'sql':
                return <Col span={16}>
                    <div className="ellip3" style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{item.value}</div>
                    <div>
                        <span className="link cp mr16" onClick={() => {
                            setIsSqlVisable(true)
                            setCurrentDataItem(item)
                        }}>详情</span>
                        <CopyToClipboard text={`${item.value || ''}`} onCopy={() => {
                            message.success('已成功复制到粘贴板')
                        }}>
                            <span className="link cp">复制<CopyOutlined /></span>
                        </CopyToClipboard>
                    </div>
                </Col>
            case 'table':
                return handelNodeDetailTable(item)
            default:
                break;
        }

        let tarValue: any = item.value
        if (item.key === 'us_id') {
            tarValue = <span><a target="_blank" href={`https://us.woa.com/#/taskManage/instance/${item.value}?inChargeType=0&search%7Citem.keywords%7CpageIndex%7CpageSize=%5B%7B%22taskId%22%3A%22${item.value}%22%7D%5D%5B1%5D%5B50%5D&searchNext%7CcycleUnit%7CpageIndex=%5BD%5D%5B1%5D&locale=zh_CN`} rel="noreferrer">{item.value}</a></span>
        }
        if (item.key === 'lifecycle') {
            tarValue = <span>{item.value}<a className="pl8" target="_blank" href="http://tdwhelper.oa.com/storage_stat/storage_path_info.php" rel="noreferrer">修改生命周期</a></span>
        }
        return <Col span={16}><span style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{tarValue}</span></Col>
    }

    const handleGroupContent = (type: TGroupContentType, content: any, tabName: string, groupName: string) => {
        switch (type) {
            case 'map':
                return renderMapComponent(content)
            case 'echart':
                return renderEchart(content)
            case 'text':
                return renderMapText(content)
            case 'iframe':
                return renderMapIframe(content)
            case 'html':
                return renderHtml(content)
            case 'api':
                return renderApi(content, tabName, groupName)
            default:
                return <span style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{content}</span>
        }
    }

    const renderApi = (data: string, tabName: string, groupName: string) => {
        const apiKey = `${tabName}_${groupName}`;

        // getNodeInfoApi(data).then(res => {
        //     console.log(res.data.result);
        //     const result = res.data.result

        // }).catch(err => { })
        return <div>{nodeInfoApiMapRef.current[apiKey] ? nodeInfoApiMapRef.current[apiKey].value : ''}</div>
    }
    const renderHtml = (data: string) => {
        return <div dangerouslySetInnerHTML={{ __html: data }}></div>
    }
    const renderMapComponent = (data: Record<string, string>) => {
        const dataList: Array<{ label: string, value: string }> = Object.entries(data).reduce((pre: any, [key, val]) => ([...pre, { label: key, value: val }]), [])
        return <div className="bg-title p16">
            {
                dataList.map((item, index) => {
                    return <Row className="mb8 w100" key={`nodeDetailItem_${index}`}>
                        <Col span={8}><div className="ta-l"><strong>{item.label}：</strong></div></Col>
                        <Col span={16}><span style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{item.value}</span></Col>
                    </Row>
                })
            }
        </div>
    }
    const renderEchart = (data: any) => {
        var currentOps: any = {}
        eval(`currentOps=${data}`)
        return <div className="bg-title p16">
            <EchartCore option={currentOps} loading={false} />
        </div>
    }
    const renderMapText = (data: string) => {
        return <div className="p16 bg-title" style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{data}</div>
    }
    const renderMapIframe = (data: any) => {
        return <iframe
            src={data.url}
            allowFullScreen
            allow="microphone;camera;midi;encrypted-media;"
            className="w100 fade-in"
            style={{ border: 0, height: 500 }}>
        </iframe>
    }

    return (
        <>
            <Tabs className="nodedetail-tab">
                {
                    props.data.map((tab, tabIndex) => {
                        return <Tabs.TabPane tab={tab.tabName} key={`nodeDetailTab${tabIndex}`}>
                            <div className="d-f fd-c jc-b h100">
                                <div className="flex1">
                                    {
                                        tab.content.map((group, groupIndex) => {
                                            return <div className="mb32" key={`nodeGroup${groupIndex}`}>
                                                <div className="fs16 mb16 bor-l b-theme pl4" style={{ borderLeftWidth: 2 }}>{group.groupName}</div>
                                                <div>
                                                    {handleGroupContent(group.groupContent.type, group.groupContent.value, tab.tabName, group.groupName)}
                                                </div>
                                            </div>
                                        })
                                    }
                                </div>
                                <div className="nodedetail-tool">
                                    {
                                        tab.bottomButton.map(button => {
                                            return <Button className="mr12 icon-tool-wrapper" onClick={() => {
                                                window.open(button.url, 'blank')
                                            }}>
                                                <span className="icon-tool" dangerouslySetInnerHTML={{ __html: button.icon }}></span>
                                                <span className="ml6">{button.text}</span>
                                            </Button>
                                        })
                                    }
                                </div>
                            </div>
                        </Tabs.TabPane>
                    })
                }
            </Tabs>
        </>
    )
}
