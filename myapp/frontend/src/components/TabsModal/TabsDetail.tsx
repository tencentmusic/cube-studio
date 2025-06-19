
import { Button, Col, Row, Tabs } from 'antd'

import React, { useEffect, useState } from 'react'

import { ITabDetailItem, TGroupContentType } from '../../api/interface/tabsModalInterface';

import EchartCore from '../EchartCore/EchartCore';

import './TabsDetail.less';

import {marked} from 'marked'




interface IProps {

    data: ITabDetailItem[]

}


export default function TabsDetail(props: IProps) {

    const [markdownMap, setMarkdownMap] = useState<Record<string, Record<string, string>>>({});




    useEffect(() => {

        const renderMarkdown = async (tab: string, group: string, value: string) => {

            const html = await marked(value);

            setMarkdownMap((prev) => ({

                ...prev,

                [tab]: {

                    ...(prev[tab] || {}),

                    [group]: html,

                },

            }));

        };

        props.data.forEach((tab) => {

            tab.content.forEach((group) => {

                if (group.groupContent.type === 'markdown') {

                    renderMarkdown(tab.tabName, group.groupName, group.groupContent.value);

                }

            });

        });

    }, [props.data]);







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

            case 'image':

                return renderimage(content)

            case 'html':

                return renderHtml(content)

            case 'markdown':

                return renderMarkdown(tabName, groupName)

            default:

                return <span style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{content}</span>

        }

    }


    const renderimage = (data: string) => {
        return (
            <img
                src={data}
                style={{ width: '100%', height: 'auto' }}
            />
        );
    }

    const renderHtml = (data: string) => {

        return <div dangerouslySetInnerHTML={{ __html: data }}></div>

    }

    const renderMapComponent = (data: Record<string, string>) => {

        const dataList: Array<{ label: string, value: string }> = Object.entries(data).reduce((pre: any, [key, val]) => ([...pre, { label: key, value: val }]), [])

        return <div className="bg-title p16">

            {

                dataList.map((item, index) => {

                    return <Row className="mb8 w100" key={`tabsDetailItem_${index}`}>

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

            src={data}

            allowFullScreen

            allow="microphone;camera;midi;encrypted-media;"

            className="w100 fade-in"

            style={{ border: 0, height: 500 }}>

        </iframe>

    }

    const renderMarkdown = (tabName: string, groupName: string) => {

        const markdownHtml = markdownMap[tabName]?.[groupName];

        return markdownHtml ? <div dangerouslySetInnerHTML={{ __html: markdownHtml }} /> : null;

    };


    return (

        <>
            {
                props.data.length>1?
                <Tabs className="tabsdetail-tab">

                    {
                         props.data.map((tab, tabIndex) => {
                            return <Tabs.TabPane tab={tab.tabName} key={`tabsDetailTab${tabIndex}`}>

                                <div className="d-f fd-c jc-b h100">

                                    <div className="flex1">

                                        {

                                            tab.content.map((group, groupIndex) => {

                                                return <div className={tab.bottomButton?"mb32":"mb2"} key={`tabsGroup${groupIndex}`}>

                                                    <div className="fs16 mb16 bor-l b-theme pl4" style={{ borderLeftWidth: 2 }} dangerouslySetInnerHTML={{ __html: group.groupName }}></div>

                                                    <div>

                                                        {handleGroupContent(group.groupContent.type, group.groupContent.value, tab.tabName, group.groupName)}

                                                    </div>

                                                </div>

                                            })

                                        }

                                    </div>

                                    <div className="tabsdetail-tool">

                                        {

                                            tab.bottomButton?.map(button => {

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

                </Tabs>:
                <div className="d-f fd-c jc-b h100">

                    <div className="flex1">

                        {

                            props.data[0].content.map((group, groupIndex) => {

                                return <div className={props.data[0].bottomButton?"mb32":"mb2"} key={`tabsGroup${groupIndex}`}>

                                    {
                                        group.groupName?<div className="fs16 mb16 bor-l b-theme pl4" style={{ borderLeftWidth: 2 }} dangerouslySetInnerHTML={{ __html: group.groupName }}></div>:null
                                    }

                                    <div>

                                        {handleGroupContent(group.groupContent.type, group.groupContent.value, props.data[0].tabName, group.groupName)}

                                    </div>

                                </div>

                            })

                        }

                    </div>

                    <div className="tabsdetail-tool">

                        {

                            props.data[0].bottomButton?.map(button => {

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
            }



        </>

    )

}
