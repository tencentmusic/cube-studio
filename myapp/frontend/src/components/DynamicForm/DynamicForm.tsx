import { Button, Col, Collapse, DatePicker, Form, FormInstance, Input, message, Radio, Row, Space, Steps, Tooltip } from 'antd'
import { Rule, RuleObject } from 'antd/lib/form'
import Select, { LabeledValue } from 'antd/lib/select'
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import moment from "moment";
import { MinusCircleOutlined, PlusOutlined, QuestionCircleOutlined, SyncOutlined } from '@ant-design/icons';
import InputSearch from '../InputSearch/InputSearch';
import 'moment/locale/zh-cn';
import locale from 'antd/es/date-picker/locale/zh_CN';
import { useTranslation } from 'react-i18next';
import FileUploadPlus from '../FileUploadPlus/FileUploadPlus';

interface IProps {
    primaryKey?: string
    form?: FormInstance
    config?: IDynamicFormConfigItem[]
    configGroup?: IDynamicFormGroupConfigItem[]
    formChangeRes?: IFormChangeRes
    linkageConfig?: ILinkageConfig[]
    onRetryInfoChange?: (value?: string) => void
}

export interface ILinkageConfig {
    dep: string[]
    effect: string
    effectOption: Record<string | number, LabeledValue[]>
}

interface IFormChangeRes {
    currentChange: Record<string, any>
    allValues: Record<string, any>
}
export interface IDynamicFormGroupConfigItem {
    expanded: boolean
    group: string
    config: IDynamicFormConfigItem[]
}

export interface IDynamicFormConfigItem {
    name: string
    label: string
    type: TDynamicFormType
    defaultValue?: number | string
    required?: boolean
    placeHolder?: string
    options?: LabeledValue[]
    rules?: Rule[]
    disable?: boolean
    description?: any
    multiple?: boolean,
    list?: IDynamicFormConfigItem[]
    data: Record<string, any>
}

export type TDynamicFormType = 'input' | 'textArea' | 'select' | 'datePicker' | 'rangePicker' | 'radio' | 'checkout' | 'match-input' | 'input-select' | 'fileUpload'

export function calculateId(strList: string[]): number {
    const str2Num = (str: string) => {
        const res = (str || '').split('').reduce((pre, next) => pre + next.charCodeAt(0), 0)
        return res
    }
    const sum = strList.reduce((pre, next) => pre + str2Num(next), 0)
    return sum
}

export default function DynamicForm(props: IProps) {
    const { t, i18n } = useTranslation();
    const [current, setCurrent] = useState(0);
    const [currentConfig, _setCurrentConfig] = useState(props.config)
    const currentConfigRef = useRef(props.config);
    const setCurrentConfig = (data: IDynamicFormConfigItem[] | undefined): void => {
        currentConfigRef.current = data;
        _setCurrentConfig(data);
    };

    const [currentConfigGroup, _setCurrentConfigGroup] = useState(props.configGroup)
    const currentConfigGroupRef = useRef(props.configGroup);
    const setCurrentConfigGroup = (data: IDynamicFormGroupConfigItem[] | undefined): void => {
        currentConfigGroupRef.current = data;
        _setCurrentConfigGroup(data);
    };

    const findOptionInLinkAge = (field: string, config: ILinkageConfig[]): Array<{
        effect: string
        option: LabeledValue[]
    }> => {
        const res = config.filter(configItem => configItem.dep.includes(field)).map(item => {
            const values = item.dep.map(item => props.form?.getFieldValue(item)).filter(item => !(item === undefined || item === null))
            const valueId = calculateId(values)
            return {
                effect: item.effect,
                option: item.effectOption[valueId] || []
            }
        })
        return res
    }

    const setValueInConfig = (field: string, props: Record<string, any>) => {
        const tarConfig = currentConfigRef.current ? [...currentConfigRef.current] : []
        if (tarConfig) {
            for (let i = 0; i < tarConfig.length; i++) {
                const item = tarConfig[i];
                if (item.name === field) {
                    tarConfig[i] = {
                        ...item,
                        ...props
                    }
                }
            }
        }
        setCurrentConfig(tarConfig)
    }

    const setValueInConfigGroup = (field: string, props: Record<string, any>) => {
        const tarConfigGroup = currentConfigGroupRef.current ? [...currentConfigGroupRef.current] : []
        for (let i = 0; i < tarConfigGroup.length; i++) {
            const configList = [...tarConfigGroup[i].config];
            for (let j = 0; j < configList.length; j++) {
                const item = configList[j];
                if (item.name === field) {
                    configList[j] = {
                        ...item,
                        ...props
                    }
                }
            }
            tarConfigGroup[i] = {
                ...tarConfigGroup[i],
                config: configList
            }
        }
        setCurrentConfigGroup(tarConfigGroup)
    }

    const resetFieldProps = (field: string, linkageConfig: ILinkageConfig[]) => {
        const optionInlinkAge = findOptionInLinkAge(field, linkageConfig)
        optionInlinkAge.forEach(item => {
            props.form?.setFieldsValue({ [item.effect]: undefined })
            setValueInConfig(item.effect, { options: item.option })
            setValueInConfigGroup(item.effect, { options: item.option })
        })
    }

    useEffect(() => {
        if (props.formChangeRes && props.linkageConfig) {
            const { currentChange } = props.formChangeRes
            resetFieldProps(Object.keys(currentChange)[0], props.linkageConfig)
        }
    }, [props.formChangeRes])

    // 表单联动初始化
    useEffect(() => {
        setCurrentConfig(props.config)
        setCurrentConfigGroup(props.configGroup)
        const formValues = props.form?.getFieldsValue() || {}
        Object.entries(formValues).forEach(([key, value]) => {
            if (value !== undefined) {
                resetFieldProps(key, props.linkageConfig || [])
            }
        })
    }, [props.configGroup, props.config])

    const next = () => {
        setCurrent(current + 1);
    };

    const prev = () => {
        setCurrent(current - 1);
    };

    const renderFileUpload = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={config.rules}
            initialValue={config.defaultValue}
            extra={config.description ? <span dangerouslySetInnerHTML={{ __html: config.description }}></span> : null}
            {...itemProps}
        >
            <FileUploadPlus
                filetype={config.data.type}
                format={config.data.format}
                maxCount={config.data.maxCount || 1}
            />
        </Form.Item>
    }

    const renderInput = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        // const rules: Rule[] = [
        //     { required: config.required, message: `请输入${config.label}` },
        //     config.rule ? { pattern: new RegExp(`/^${config.rule}$/`), message: '请按正确的规则输入' } : undefined,
        // ].filter(item => !!item) as Rule[]

        let extraContent: any = null

        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={config.rules}
            initialValue={config.defaultValue}
            extra={<>
                {config.data.tips ? <Tooltip
                    className="mr8"
                    placement="bottom"
                    title={<span dangerouslySetInnerHTML={{ __html: config.data.tips }}></span>}
                >
                    <div className="cp d-il">
                        <QuestionCircleOutlined style={{ color: '#1672fa' }} />
                        <span className="pl4 c-theme">{t('详情')}</span>
                    </div>
                </Tooltip> : null}
                {config.description ? <span dangerouslySetInnerHTML={{ __html: config.description }}></span> : null}
            </>}
            {...itemProps}
        >
            <Input disabled={config.disable} placeholder={config.placeHolder || `${t('请选择')}${config.label}`} />
        </Form.Item>
    }

    const renderMatchInput = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {

        return <Form.Item key={`dynamicForm_${config.name}_noStyle`} noStyle shouldUpdate={(pre, next) => {
            // todo:更新有点问题
            // return pre[config.name] != pre[config.name]
            return JSON.stringify(pre) != JSON.stringify(next)
        }}>
            {
                ({ getFieldValue, setFieldsValue }) => {
                    const templateText = `${config.defaultValue}`
                    const matchList = templateText.match(/\$\{\w*}/gi) || []
                    let value = templateText
                    matchList.forEach(item => {
                        const itemKey = item.replace(/^\$\{/, '').replace(/\}$/, '')
                        const itemValue = getFieldValue(itemKey)
                        if (itemValue !== undefined) {
                            value = value.replace(item, itemValue)
                        }
                    })

                    if (getFieldValue(config.name) !== value) {
                        setFieldsValue({
                            [config.name]: value
                        })
                    }

                    return <Form.Item
                        key={`dynamicForm_${config.name}`}
                        label={config.label}
                        name={config.name}
                        rules={config.rules}
                        extra={<>
                            {config.data.tips ? <Tooltip
                                className="mr8"
                                placement="bottom"
                                title={<span dangerouslySetInnerHTML={{ __html: config.data.tips }}></span>}
                            >
                                <div className="cp d-il">
                                    <QuestionCircleOutlined style={{ color: '#1672fa' }} />
                                    <span className="pl4 c-theme">{t('详情')}</span>
                                </div>
                            </Tooltip> : null}
                            {config.description ? <span dangerouslySetInnerHTML={{ __html: config.description }}></span> : null}
                        </>}
                        {...itemProps}
                    >
                        <Input disabled={true} />
                    </Form.Item>
                }
            }
        </Form.Item>
    }

    const renderInputSelect = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        // const rules = [
        //     { required: config.required, message: `${t('请选择')}${config.label}` },
        // ]
        const options: string[] = (config.options || []).map(item => item.label as string)
        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={config.rules}
            initialValue={config.defaultValue}
            extra={<>
                {config.data.tips ? <Tooltip
                    className="mr8"
                    placement="bottom"
                    title={<span dangerouslySetInnerHTML={{ __html: config.data.tips }}></span>}
                >
                    <div className="cp d-il">
                        <QuestionCircleOutlined style={{ color: '#1672fa' }} />
                        <span className="pl4 c-theme">{t('详情')}</span>
                    </div>
                </Tooltip> : null}
                {config.description ? <span dangerouslySetInnerHTML={{ __html: config.description }}></span> : null}
            </>}
            {...itemProps}
        >
            <InputSearch
                onClick={(value) => {
                    !!config.data.retry_info && props.onRetryInfoChange && props.onRetryInfoChange(value)
                }}
                isOpenSearchMatch={true}
                disabled={config.disable}
                placeholder={`${t('请选择')}${config.label}`}
                options={options} />
        </Form.Item>
    }

    const renderTextArea = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={config.rules}
            initialValue={config.defaultValue}
            extra={<>
                {config.data.tips ? <Tooltip
                    className="mr8"
                    placement="bottom"
                    title={<span dangerouslySetInnerHTML={{ __html: config.data.tips }}></span>}
                >
                    <div className="cp d-il">
                        <QuestionCircleOutlined style={{ color: '#1672fa' }} />
                        <span className="pl4 c-theme">{t('详情')}</span>
                    </div>
                </Tooltip> : null}
                {config.description ? <span dangerouslySetInnerHTML={{ __html: config.description }}></span> : null}
            </>}
            {...itemProps}
        >
            <Input.TextArea autoSize={{ minRows: 4 }} disabled={config.disable} placeholder={config.placeHolder || `${t('请选择')}${config.label}`} />
        </Form.Item>
    }
    const renderSelect = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        // const rules = [
        //     { required: config.required, message: `${t('请选择')}${config.label}` },
        // ]
        const options: LabeledValue[] = config.options || []
        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={config.rules}
            initialValue={config.defaultValue}
            extra={<>
                {config.data.tips ? <Tooltip
                    className="mr8"
                    placement="bottom"
                    title={<span dangerouslySetInnerHTML={{ __html: config.data.tips }}></span>}
                >
                    <div className="cp d-il">
                        <QuestionCircleOutlined style={{ color: '#1672fa' }} />
                        <span className="pl4 c-theme">{t('详情')}</span>
                    </div>
                </Tooltip> : null}
                {config.description ? <span className="pr4" dangerouslySetInnerHTML={{ __html: config.description }}></span> : null}
                {
                    config.data.isRefresh ? <div className="cp d-il" onClick={() => {
                        props.onRetryInfoChange && props.onRetryInfoChange()
                    }}>
                        <SyncOutlined style={{ color: '#1672fa' }} />
                        <span className="pl4 c-theme">{t('刷新列表')}</span>
                    </div> : null
                }
            </>}
            {...itemProps}
        >
            <Select
                style={{ width: '100%' }}
                mode={config.multiple ? 'multiple' : undefined}
                onChange={(value) => {
                    !!config.data.retry_info && props.onRetryInfoChange && props.onRetryInfoChange(value)
                }}
                showSearch
                disabled={config.disable}
                optionFilterProp="label"
                placeholder={config.placeHolder || `${t('请选择')}${config.label}`}
                options={options} />
        </Form.Item>
    }
    const renderRadio = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        // const rules = [
        //     { required: config.required, message: `${t('请选择')}${config.label}` },
        // ]
        const options: LabeledValue[] = config.options || []
        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={config.rules}
            initialValue={config.defaultValue}
            {...itemProps}
        >
            <Radio.Group options={options} />
        </Form.Item>
    }
    const renderDatePicker = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={[{ required: true, message: t('请选择时间') }]}
            {...itemProps}
        >
            <DatePicker style={{ width: '100%' }} locale={locale} showTime={!!config.data.showTime} disabledDate={(current) => {
                return current && current > moment().endOf('day');
            }} />
        </Form.Item>
    }
    const renderRangePicker = (config: IDynamicFormConfigItem, itemProps: Record<string, any>) => {
        return <Form.Item
            key={`dynamicForm_${config.name}`}
            label={config.label}
            name={config.name}
            rules={[{ required: true, message: t('请选择时间范围') }]}
            {...itemProps}
        >
            <DatePicker style={{ width: '100%' }} locale={locale} showTime={!!config.data.showTime} disabledDate={(current) => {
                return current && current > moment().endOf('day');
            }} />
        </Form.Item>
    }

    const dispatchRenderFormItem = (item: IDynamicFormConfigItem, itemProps: Record<string, any> = {}): JSX.Element | null => {
        switch (item.type) {
            case 'input':
                return renderInput(item, itemProps)
            case 'match-input':
                return renderInput(item, itemProps)
            // return renderMatchInput(item, itemProps)
            case 'input-select':
                return renderInputSelect(item, itemProps)
            case 'textArea':
                return renderTextArea(item, itemProps)
            case 'select':
                return renderSelect(item, itemProps)
            case 'datePicker':
                return renderDatePicker(item, itemProps)
            case 'rangePicker':
                return renderRangePicker(item, itemProps)
            case 'radio':
                return renderRadio(item, itemProps)
            case 'fileUpload':
                return renderFileUpload(item, itemProps)
            default:
                return null
        }
    }

    const renderFormItem = (config: IDynamicFormConfigItem[]): Array<any | null> => {
        return (config || []).map(item => {
            if (item.list && item.list.length) {
                const formList = <Form.List key={`dynamicForm_${item.name}`} name={item.name}>
                    {(fields, { add, remove }) => (
                        <>
                            {fields.map(({ key, name, ...restField }) => (
                                // <Space key={key} style={{ display: 'flex', marginBottom: 8 }}
                                //     align='baseline'
                                // >
                                //     {
                                //         item.list && item.list.map(listItem => {
                                //             return dispatchRenderFormItem(listItem, {
                                //                 ...restField,
                                //                 name: [name, listItem.name],
                                //                 // style: { flexDirection: 'column' }
                                //             })
                                //         })
                                //     }
                                //     {/* <MinusCircleOutlined onClick={() => remove(name)} /> */}
                                //     <Form.Item wrapperCol={{ offset: 5 }}>
                                //         <Button danger onClick={() => remove(name)} block icon={<MinusCircleOutlined />}>
                                //             删除该项
                                //         </Button>
                                //     </Form.Item>
                                // </Space>
                                <div key={key} className="bor b-side pt8 plr16 mb8 d-f" style={{ alignItems: 'start', minWidth: 1600 }}>
                                    {
                                        item.list && item.list.map(listItem => {
                                            return dispatchRenderFormItem(listItem, {
                                                ...restField,
                                                name: [name, listItem.name],
                                                labelAlign: 'left',
                                                labelCol: 24,
                                                style: { flexDirection: 'column', flex: 1, marginBottom: 8 },
                                            })
                                        })
                                    }
                                    {/* <MinusCircleOutlined onClick={() => remove(name)} /> */}
                                    <Form.Item >
                                        <Button danger onClick={() => remove(name)} block icon={<MinusCircleOutlined />} style={{ width: 120 }}>
                                            {t('删除该项')}
                                        </Button>
                                    </Form.Item>
                                </div>
                            ))}
                            <Form.Item noStyle className="w100" label="">
                                <Button type="dashed" className="w100" onClick={() => add()} block icon={<PlusOutlined />}>
                                    {t('增加一项')}
                                </Button>
                            </Form.Item>
                        </>
                    )}
                </Form.List>
                return formList
            } else {
                return <div style={{ width: 680 }}>
                    {dispatchRenderFormItem(item)}
                </div>
            }
        })
    }

    return (
        <>
            <Form.Item
                key={`dynamicForm_id`}
                name={props.primaryKey || 'id'}
                noStyle
                hidden
            >
                <Input />
            </Form.Item>

            {
                currentConfigGroup && currentConfigGroup.length ? <>
                    <Steps current={current}>
                        {
                            (currentConfigGroup || []).map((item, index) => {
                                return <Steps.Step key={index} title={item.group} />
                            })
                        }
                    </Steps>
                    <div className="pt32">
                        {
                            (currentConfigGroup || []).map((item, index) => {
                                return <div key={index} className={[current === index ? 'p-r z9' : 'p-a z-99 v-h l-10000'].join(' ')}>
                                    {renderFormItem(item.config)}
                                </div>
                            })
                        }
                    </div>
                    <div className="ta-c pt32">
                        {current > 0 && (
                            <Button onClick={() => prev()}>
                                {t('上一步')}
                            </Button>
                        )}
                        {current < (currentConfigGroup || []).length - 1 && (
                            <Button type="primary" className="ml16" onClick={() => {
                                if (props.form) {
                                    const currentConfigGroupNameList = currentConfigGroup[current].config.map(item => item.name)
                                    props.form.validateFields(currentConfigGroupNameList).then(() => {
                                        next()
                                    }).catch(err => {
                                        console.log(err)
                                    })
                                } else {
                                    next()
                                }
                            }}>
                                {t('下一步')}
                            </Button>
                        )}
                        <div>
                            {current === (currentConfigGroup || []).length - 1 && (
                                <div className="pt8 c-hint-b">{t('点击确定完成提交')}</div>
                            )}
                        </div>
                    </div>
                </> : <div style={{ width: 680 }}>
                    {
                        renderFormItem(currentConfig || [])
                    }
                </div>
            }
        </>
    )
}
