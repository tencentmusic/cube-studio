import { GithubOutlined } from '@ant-design/icons'
import { Button, Form, message, Result, Spin, Tag } from 'antd'
import React, { useEffect, useState, useRef } from 'react'
import { IAppInfo, IResultItem } from '../../api/interface/stateInterface'
import { getAppInfo, submitData } from '../../api/mobiApi'

import AwesomeSlider from 'react-awesome-slider';
// @ts-ignore
import withAutoplay from 'react-awesome-slider/dist/autoplay';
import 'react-awesome-slider/dist/styles.css';
import DynamicForm, { IDynamicFormConfigItem } from '../../components/DynamicForm/DynamicForm'
import Checkbox from '../../components/CheckoutGroupPlus/CheckoutGroupPlus'
import { useLocation } from 'react-router-dom'
import Loading from '../../components/Loading/Loading'
import { isInWeixin, share } from '../../utils/weixin'

const AutoplaySlider = withAutoplay(AwesomeSlider);

// import './Index.less';

export default function Index() {
    const [pageInfo, setPageInfo] = useState<IAppInfo>()
    const [dynamicFormConfig, setDynamicFormConfig] = useState<IDynamicFormConfigItem[]>([])
    const [reslutList, setReslutList] = useState<IResultItem[]>([])
    const [selected, setSelected] = useState<any[]>([])
    const [loading, setLoading] = useState(false)
    const [form] = Form.useForm();
    const location = useLocation()

    const [activeKey, _setActiveKey] = useState<number>(0);
    const activeKeyRef = useRef(activeKey);
    const setActiveKey = (data: number): void => {
        activeKeyRef.current = data;
        _setActiveKey(data);
    };

    useEffect(() => {
        const timerTagRecord = setInterval(() => {
            if (activeKeyRef.current < 99) {
                setActiveKey(activeKeyRef.current + 1);
            }
        }, 1000);
        return () => {
            clearInterval(timerTagRecord);
        };
    }, []);

    useEffect(() => {
        getAppInfo(location.pathname).then(res => {
            const data = res.data
            const tarConfig = createDyFormConfig(data.inference_inputs, {}, {})
            setPageInfo(data)
            setDynamicFormConfig(tarConfig)

            if (isInWeixin()) {
                share({
                    title: data?.describe,
                    link: window.location.href,
                    desc: 'cube-studio 开源社区',
                    imgUrl: `https://cube-studio-1252405198.cos.ap-nanjing.myqcloud.com/example/${data.name}/example.jpg`
                })
            }
        }).catch(err => { }).finally(() => {
            // setLoading(false)
        })
    }, [])

    // 表单字段处理
    const createDyFormConfig = (data: Record<string, any>[], label_columns: Record<string, any>, description_columns: Record<string, any>): IDynamicFormConfigItem[] => {
        const typeMap: any = {
            text: 'input',
            text_select: 'select',
            text_select_multi: 'select',
            image: 'imageUpload',
            video: 'videoUpload',
            audio: 'audioUpload',
            file: 'fileUpload',
            image_multi: 'imageUpload',
            video_multi: 'videoUpload',
            image_select: 'imageSelect',
            image_select_multi: 'imageSelect',

        }

        return data.map((item, index) => {
            let type = typeMap[item.type]
            const label = item.label || label_columns[item.name]

            // 校验规则
            const rules = (item.validators || []).map((item: any) => {
                if (type === 'select') {
                    return item.type === 'DataRequired' ? { required: true, message: `请选择${label}` } : undefined
                }

                switch (item.type) {
                    case 'DataRequired':
                        return { required: true, message: `请输入${label}` }
                    case 'Regexp':
                        return { pattern: new RegExp(`${item.regex}`), message: `请按正确的规则输入` }
                    case 'Length':
                        return { min: item.min || 0, max: item.max, message: `请输入正确的长度` }
                    default:
                        return undefined
                }
            }).filter((item: any) => !!item)

            const list = createDyFormConfig((item.info || []), label_columns, description_columns)

            const res: IDynamicFormConfigItem = {
                label,
                type,
                // rules,
                list,
                name: item.name,
                disable: item.disable,
                description: item.describe || undefined,
                required: item.required,
                defaultValue: item.default === '' ? undefined : item.default,
                multiple: item['ui-type'] && item['ui-type'] === 'select2',
                options: (item.values || []).map((item: any) => ({ label: item.value, value: item.id })),
                data: { ...item }
            }
            return res
        })
    }

    return (
        <div>
            {/* {
                loading ? <Spin spinning={loading} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100vh' }}></Spin> : null
            } */}
            {
                loading ? <Loading value={activeKey} content={
                    <div className="ta-c">
                        <div className="c-text-w fs18 pb8">结果生成中，稍等片刻</div>
                        <Button onClick={() => {
                            setLoading(false)
                        }}>退出等待</Button>
                    </div>
                } /> : null
            }

            <div style={{ display: 'none' }}>
                <img style={{ width: 300, height: 300 }} src={pageInfo?.pic} alt="" />
            </div>

            <div className="p16 bg-w">
                <div>
                    <img className="w100 pb8" style={{ maxHeight: 600 }} src={pageInfo?.pic || ''} alt="" />
                </div>
                <div className="fs20 pb4" onClick={() => {
                    window.open(pageInfo?.doc, 'blank')
                }}><strong>{pageInfo?.name} [{pageInfo?.label}]</strong> <span><GithubOutlined /></span></div>
                <div className="pb4">{pageInfo?.describe}</div>
                <div>
                    <Tag color="volcano">{pageInfo?.scenes}</Tag>
                    <Tag color="green">{pageInfo?.status}</Tag>
                </div>
            </div>
            <div className="ta-r pr16 c-hint-b">
                {pageInfo?.version}
            </div>

            <div className="ta-c pb8">
                <div className="title-mobi">
                    应用演示
                </div>
            </div>

            <div className="p16 mb16">
                {/* <Checkbox.GroupImageIn
                    values={selected}
                    onChange={(selected: any) => {
                        setSelected(selected)
                    }}
                    option={[{ label: 'http://localhost:8080/app1/static/example/风格1.jpg', value: '风格1' }, { label: 'http://localhost:8080/app1/static/example/风格1.jpg', value: '风格2' }]} /> */}
                <Form layout="horizontal" form={form} >
                    <DynamicForm form={form} primaryKey={'target'} config={dynamicFormConfig} />
                </Form>
                <div className="ta-c">
                    <Button type="primary" block onClick={() => {
                        form.validateFields().then(values => {
                            console.log(values)
                            setLoading(true)
                            setActiveKey(0)

                            submitData(pageInfo?.inference_url || '', values).then(res => {
                                setReslutList(res.data.result)
                            }).catch(err => {
                                message.error('应用运行出问题了')
                            }).finally(() => {
                                setLoading(false)
                            })
                        }).catch(err => {
                            message.warn('请填写完整参数')
                        })
                    }}>运行应用</Button>
                </div>
            </div>

            <div className="mb16">
                <div className="ta-c pb8">
                    <div className="title-mobi">
                        输出
                    </div>
                </div>

                {
                    reslutList.length ? <div className="p16">
                        {
                            reslutList.map((result, resultIndex) => {
                                return <div key={resultIndex} className="mb32">
                                    <div className="ta-c fs20 pb8">结果{resultIndex + 1}</div>
                                    {
                                        result.text ? <div className="paper p16"><span>文本结果：</span>{result.text}</div> : null
                                    }
                                    {
                                        result.image ? <div className="paper p16 mt16"><div className="pb8">图片结果：</div><img className="w100" src={result.image} alt="" /></div> : null
                                    }
                                    {
                                        result.video ? <div className="paper p16 mt16"><div className="pb8">视频结果：</div><video className="w100" src={result.video} controls></video></div> : null
                                    }
                                    {
                                        result.audio ? <div className="paper p16 mt16"><div className="pb8">音频结果：</div><audio className="w100" src={result.audio} controls></audio></div> : null
                                    }
                                    {
                                        result.html ? <div className="paper p16 mt16"><div className="pb8">输出结果：</div><div dangerouslySetInnerHTML={{ __html: result.html }}></div></div> : null
                                    }
                                </div>
                            })
                        }
                    </div> : <Result
                            // icon={<SmileOutlined />}
                            title="暂无数据"
                        />
                }
            </div>

            <div className="ta-c pb8">
                <div className="title-mobi">
                    应用推荐
                </div>
            </div>

            <div className="p16 mb64">
                <AutoplaySlider
                    play={true}
                    cancelOnInteraction={false} // should stop playing on user interaction
                    interval={5000}
                >
                    {
                        (pageInfo?.rec_apps || []).map((item, recIndex) => {
                            return <div style="height:100%" key={`rec${recIndex}`}>
                                <img className="w100" style="height:100%" src={item.pic} alt="" />
                            </div>
                        })
                    }
                </AutoplaySlider>
            </div>
        </div>
    )
}
