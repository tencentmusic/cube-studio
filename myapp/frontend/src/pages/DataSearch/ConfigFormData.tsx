import { Form } from 'antd'
import Select, { LabeledValue } from 'antd/lib/select';
import React, { useEffect, useImperativeHandle, useState } from 'react'
import { useTranslation } from 'react-i18next';
import { IIdexFormConfigItem } from '../../api/dataSearchApi'
import InputSearch from '../../components/InputSearch/InputSearch';
import './ConfigFormData.less';

export type TDataValue = Record<any, any>

export interface IConfigFormDataOptionItem extends IIdexFormConfigItem { }

export interface IProps {
    dataValue?: TDataValue
    option: IConfigFormDataOptionItem[]
    onChange: (dataValue: TDataValue) => void
    onConfigChange?: (config: IConfigFormDataOptionItem[]) => void
}

const ConfigFormData = React.forwardRef((props: IProps, ref) => {
    const { t, i18n } = useTranslation();
    const [form] = Form.useForm();
    const [, updateState] = useState<any>();

    useImperativeHandle(ref, () => ({
        onSubmit: () => {
            return new Promise((resolve, reject) => {
                form.validateFields().then(res => {
                    resolve(res)
                }).catch(err => {
                    reject(err)
                })
            })
        },
        setData: (data: Record<any, any>) => {
            form.setFieldsValue(data)
        }
    }));

    useEffect(() => {
        if (props.dataValue) {
            form.setFieldsValue(props.dataValue)
        }
    }, [props.option])

    const renderInput = (config: IConfigFormDataOptionItem, itemProps: Record<string, any>) => {
        return <div></div>
    }

    const renderSelect = (config: IConfigFormDataOptionItem, itemProps: Record<string, any>) => {
        const options: LabeledValue[] = config.value || []
        return <Form.Item
            key={`configFormData_${config.id}`}
            label={config.label}
            name={config.id}
            rules={[
                {
                    required: true,
                    message: `${t('请选择')}${config.label}`,
                },
            ]}
            initialValue={config.defaultValue}
            style={{ marginBottom: 0, marginRight: 16 }}
            {...itemProps}
        >
            <Select
                style={{ width: 200 }}
                mode={config.multiple ? 'multiple' : undefined}
                showSearch
                disabled={config.disable}
                optionFilterProp="label"
                placeholder={config.placeHolder || `${t('请选择')} ${config.label}`}
                options={options}
                onChange={(value, rowOption: any) => {
                    if (rowOption.relate) {
                        const relateId = rowOption.relate.relateId
                        const relateOption = rowOption.relate.value
                        const currentOption = props.option
                        for (let i = 0; i < currentOption.length; i++) {
                            const item = currentOption[i];
                            if (item.id === relateId) {
                                item.value = relateOption
                            }
                        }
                        props.onConfigChange && props.onConfigChange(currentOption)
                    }
                    props.onChange && props.onChange(form.getFieldsValue())
                }} />
        </Form.Item>
    }

    const renderInputSelect = (config: IConfigFormDataOptionItem, itemProps: Record<string, any>) => {
        const options: LabeledValue[] = config.value || []
        const inputSelectOption = options.map(item => (item.value)) as string[]
        return <Form.Item
            key={`configFormData_${config.id}`}
            label={config.label}
            name={config.id}
            rules={[
                {
                    required: true,
                    message: `${t('请选择')}${config.label}`,
                },
            ]}
            initialValue={config.defaultValue}
            style={{ marginBottom: 0 }}
            {...itemProps}
        >
            <InputSearch
                isOpenSearchMatch
                onChange={() => {
                    props.onChange && props.onChange(form.getFieldsValue())
                }}
                options={inputSelectOption} width={'500px'} />
        </Form.Item>
    }


    const dispatchRenderFormItem = (item: IConfigFormDataOptionItem, itemProps: Record<string, any> = {}): JSX.Element | null => {
        switch (item.type) {
            case 'input':
                return renderInput(item, itemProps)
            case 'select':
                return renderSelect(item, itemProps)
            case 'input-select':
                return renderInputSelect(item, itemProps)
            default:
                return null
        }
    }

    return (
        <div className="configformdata-container d-f ac">
            <Form form={form} component={false}>
                {
                    props.option.map((component) => {
                        return dispatchRenderFormItem(component)
                    })
                }
            </Form>
        </div>
    )
})

export default ConfigFormData
