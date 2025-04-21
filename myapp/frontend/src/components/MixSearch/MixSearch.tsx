import React, { ReactNode, useState, useEffect, ReactText } from 'react';
import { Form, Row, Col, Input, Select, Button } from 'antd';
import { DeleteOutlined, PlusOutlined, DownOutlined, UpOutlined } from '@ant-design/icons';
import './MixSearch.less';
import { LabeledValue } from 'antd/lib/select';
import { useTranslation } from 'react-i18next';

const { Option } = Select;
export interface IMixSearchParamItem {
	name: string
	type: TMixSearchType
	title?: string
	defalutValue?: any
	placeHolder?: string
	option?: LabeledValue[]
	multiple?: boolean
	indexKey?: number
	used?: boolean
}

export type TMixSearchType = 'input' | 'select' | 'datePicker' | 'rangePicker'

interface IProps {
	params?: IMixSearchParamItem[]
	values?: Array<{ key: ReactText | undefined, value: ReactText | undefined }>
	onChange: (values: Array<{ key: ReactText | undefined, value: ReactText | undefined }>) => void;
}

const MixSearch = (props: IProps) => {
	const [form] = Form.useForm();
	const [collapsed, setCollapsed]: [boolean, any] = useState(false);
	const [typeArr, setTypeArr]: [(string | undefined)[], any] = useState([]);

	// 序列化数据
	const formatParamsData = (data?: IMixSearchParamItem[]) => {
		return (data || []).map((item, indexKey) => ({ ...item, indexKey }))
	}
	const [paramsData, setParamsData] = useState<IMixSearchParamItem[]>(formatParamsData(props.params))
	const [currentParamsData, setCurrentParamsData] = useState<IMixSearchParamItem[]>(formatParamsData(props.params))
	const [paramsDataMap, setParamsDataMap] = useState<Map<string, IMixSearchParamItem>>(new Map())

	const { t, i18n } = useTranslation();

	useEffect(() => {
		if (props.values) {
			const group = props.values.length ? props.values : [{
				key: undefined,
				value: undefined
			}]
			form.setFieldsValue({
				group
			})

			const tarData = [...currentParamsData]
			for (let i = 0; i < tarData.length; i++) {
				for (let j = 0; j < group.length; j++) {
					const value = group[j];
					if (value !== undefined && group[j].key === tarData[i].name) {
						tarData[i].used = true
					}
				}
			}
			setCurrentParamsData(tarData)
		}
	}, [props.values])

	useEffect(() => {
		if (props.params && props.params.length) {
			const formatData = formatParamsData(props.params)
			setParamsData(formatData)
			const dataMap = paramsDataMap
			for (let i = 0; i < formatData.length; i++) {
				const param = formatData[i];
				dataMap.set(param.name, param)
			}
			setParamsDataMap(dataMap)
		}
	}, [props.params])

	/**利用表单获取查询字段 */
	const handleFinishForm = async (values: any): Promise<void> => {
		console.log(values);
		const preVal = values['group'].filter(((item: any) => !!item.key))
		const tarVal = preVal.map((item: any) => ({ key: item.key, value: item.value }))
		props.onChange(tarVal);
	};

	/**展开收起 */
	const handleCollapsed = (): void => {
		setCollapsed(!collapsed);
	};

	/**选择筛选类型 */
	const handleSelectType = (name: string, index: number): void => {
		form.resetFields([['group', index, 'value']]);
		let arr = [...typeArr];
		arr[index] = name;
		setTypeArr(arr);
	};

	/**根据选择的类型，渲染input或select */
	const handleRenderValueEl = (index: number): ReactNode => {
		let key = form.getFieldValue(['group', index, 'key']);
		if (key) {
			const currentItem = paramsDataMap.get(key)
			if (currentItem?.type === 'input') {
				return <Input
					style={{ width: '65%' }}
					defaultValue={currentItem.defalutValue}
					placeholder={currentItem.placeHolder}
					onPressEnter={() => handlePressEnter()} />
			} else if (currentItem?.type === 'select') {
				const currentOptions = currentItem?.option || []
				return <Select
					style={{ width: '65%' }}
					dropdownMatchSelectWidth={500}
					showSearch
					mode={key === 'label' ? 'multiple' : undefined}
					optionFilterProp="label"
					options={currentOptions.map(item => ({ label: item.label, value: item.value }))}
				// onDropdownVisibleChange={(open) => handleDropdown(open, key)}
				/>
			}
		} else {
			return <Input style={{ width: '65%' }} onPressEnter={() => handlePressEnter()} />;
		}
	};

	/**下拉获取对应的数据，并判断是否存在下拉数据，没有则请求，有则不请求 */
	// const handleDropdown = async (open: boolean, key: string): Promise<void> => {
	// 	if (open) {
	// 		if (selectionData[key]) {
	// 			return;
	// 		}
	// 		try {
	// 			let res = await getSelections(key);
	// 			let data = { ...selectionData };
	// 			data[key] = res.data.data;
	// 			setSelectionData(data);
	// 		} catch (error) { }
	// 	}
	// };

	/**输入框回车操作 */
	const handlePressEnter = (): void => {
		form.validateFields();
	};

	return (
		<Form
			// {...formConfig}
			className="cmdb-mixsearch bg-title"
			form={form}
			onFinish={handleFinishForm}
			initialValues={{
				group: [
					{
						key: undefined,
						value: undefined,
					},
				],
			}}
		>
			<Row className="cmdb-mixsearch-content" gutter={16} style={{ marginLeft: 0, marginRight: 0, ...collapsed ? { height: 70 } : { height: 'auto' } }}>
				<Form.List name={`group`}>
					{(fields, { add, remove }) => {
						return <>
							{
								fields.map((field, index) => {
									return (
										<Col span={8} key={`mixSearch_${field.key}_${index}`}>
											<Row align="middle" gutter={8}>
												{/* <Col className="cmdb-mixsearch-name">名称</Col> */}
												<Col className="cmdb-mixsearch-group">
													<Input.Group compact>
														<Form.Item
															noStyle
															name={[field.name, 'key']}
															rules={[{ required: false, message: t('请选择key') }]}
														// initialValue={'testParams'}
														>
															<Select
																style={{ width: '35%' }}
																placeholder={t('请选择')}
																onChange={(value: string) => {
																	// handleSelectType(value, index)
																	const selectActionRemove = (value: string) => {
																		const tarData = [...currentParamsData]
																		const usedKey = (form.getFieldValue('group') || []).filter((item: any) => !!item).map((item: any) => item.key)
																		for (let i = 0; i < tarData.length; i++) {
																			const item = tarData[i];
																			if (item.name === value) {
																				tarData[i].used = true
																			} else if (!usedKey.includes(item.name)) {
																				tarData[i].used = false
																			}
																		}
																		setCurrentParamsData(tarData)
																	}
																	selectActionRemove(value)
																}}
															>
																{currentParamsData.map((item, index) => {
																	return (
																		<Option style={{ display: item.used ? 'none' : 'inherit' }} key={`mixSearch_${item.name}_${index}`} value={item.name}>
																			{item.title || item.name}
																		</Option>
																	);
																})}
															</Select>
														</Form.Item>
														<Form.Item
															noStyle
															shouldUpdate
															name={[field.name, 'value']}
															rules={[{ required: false, message: t('请填写value') }]}
														>
															{handleRenderValueEl(index)}
														</Form.Item>
													</Input.Group>
												</Col>
												{(
													<Col className="cmdb-mixsearch-delete" onClick={() => {
														const usedKey = (form.getFieldValue('group') || []).map((item: any) => item ? item.key : undefined)
														const tarData = [...currentParamsData]
														if (usedKey[index]) {
															for (let i = 0; i < tarData.length; i++) {
																const item = tarData[i];
																if (item.name === usedKey[index]) {
																	tarData[i].used = false
																}
															}
														}
														setCurrentParamsData(tarData)
														remove(index)
													}}>
														<DeleteOutlined />
													</Col>
												)}
												{/* {index === fields.length - 1 && index < (paramsData.length - 1) && (
													<Col className="cmdb-mixsearch-add" onClick={() => {
														add()
													}}>
														<PlusOutlined />
													</Col>
												)} */}
											</Row>
										</Col>
									);
								})
							}
							{paramsData.length !== fields.length && (
								<Col className="cmdb-mixsearch-add d-il" onClick={() => {
									add()
								}}>
									<PlusOutlined />
								</Col>
							)}
						</>
					}}
				</Form.List>

				<Col flex={1}>
					<Row justify="end">
						<Button type="primary" htmlType="submit">
							{t('查询')}
						</Button>
					</Row>
				</Col>
			</Row>
			<Row className="cmdb-mixsearch-collapsed">
				<Row onClick={() => handleCollapsed()} justify="center" align="middle">
					{collapsed ? (
						<>
							<Col>{t('展开')}</Col>
							<Col>
								<DownOutlined />
							</Col>
						</>
					) : (
							<>
								<Col>{t('收起')}</Col>
								<Col>
									<UpOutlined />
								</Col>
							</>
						)}
				</Row>
			</Row>
		</Form>
	);
};

export default MixSearch;
