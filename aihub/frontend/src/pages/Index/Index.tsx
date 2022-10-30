import { GithubOutlined } from '@ant-design/icons'
import { Button, Form, message, Result, Tag } from 'antd'
import React, { useEffect, useState } from 'react'
import { IAppInfo, IResultItem } from '../../api/interface/stateInterface'
import { getAppInfo, submitData } from '../../api/mobiApi'

import AwesomeSlider from 'react-awesome-slider';
// @ts-ignore
import withAutoplay from 'react-awesome-slider/dist/autoplay';
import 'react-awesome-slider/dist/styles.css';
import DynamicForm, { IDynamicFormConfigItem } from '../../components/DynamicForm/DynamicForm'
import Checkbox from '../../components/CheckoutGroupPlus/CheckoutGroupPlus'

const AutoplaySlider = withAutoplay(AwesomeSlider);

// import './Index.less';

export default function Index() {
    const [pageInfo, setPageInfo] = useState<IAppInfo>()
    const [dynamicFormConfig, setDynamicFormConfig] = useState<IDynamicFormConfigItem[]>([])
    const [reslutList, setReslutList] = useState<IResultItem[]>([])
    const [selected, setSelected] = useState<any[]>([])
    const [form] = Form.useForm();

    useEffect(() => {
        getAppInfo().then(res => {
            const data = res.data
            console.log(data);
            setPageInfo(data)
            const tarConfig = createDyFormConfig(data.inference_inputs, {}, {})
            setDynamicFormConfig(tarConfig)

            console.log(tarConfig);
        }).catch(err => { })
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
                    option={[{ label: 'test', value: 'https://user-images.githubusercontent.com/20157705/170216784-91ac86f7-d272-4940-a285-0c27d6f6cd96.jpg' }]} /> */}
                <Form layout="horizontal" form={form} >
                    <DynamicForm form={form} primaryKey={'target'} config={dynamicFormConfig} />
                </Form>
                <div className="ta-c">
                    <Button type="primary" block onClick={() => {
                        form.validateFields().then(values => {
                            console.log(values)
                            submitData(pageInfo?.inference_url || '', values).then(res => {
                                console.log(res)
                                setReslutList(res.data.result)
                            }).catch(err => {
                                message.error('应用运行出问题了')
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
                        结果输出
                </div>
                </div>

                {
                    reslutList.length ? <div className="p16">
                        {
                            reslutList.map((result, resultIndex) => {
                                return <div key={resultIndex}>
                                    <div>{result.text}</div>
                                    <div><img src={result.image} alt="" /></div>
                                    <div>{result.video}</div>
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
                            return <div key={`rec${recIndex}`}>
                                <img className="w100" src={item.pic} alt="" />
                            </div>
                        })
                    }
                </AutoplaySlider>
            </div>
        </div>
    )
}
