import React, { ReactNode, useEffect, useMemo, useState } from 'react';
import { Modal, Form, Spin, Input, FormInstance } from 'antd';
import { useTranslation } from 'react-i18next';

interface ModalFormProps {
	visible: boolean;
	onCreate: (values: any, form: FormInstance<any>) => void;
	onCancel: () => void;
	loading?: boolean;
	children?: any;
	title?: string;
	formData?: Record<string, any>;
	width?: number;
	onValuesChange?: () => {}
}

const ModalForm = (props: ModalFormProps): JSX.Element => {
	const { t, i18n } = useTranslation();
	const [form] = Form.useForm();
	const [, updateState] = useState<any>();
	const forceUpdate = React.useCallback(() => updateState({}), []);

	useEffect(() => {
		if (props.formData) {
			form.setFieldsValue(props.formData);
		}
	}, [props]);

	const [formChangeRes, setFormChangeRes] = useState<{
		currentChange: Record<string, any>
		allValues: Record<string, any>
	}>({
		currentChange: {},
		allValues: {}
	})

	// const propsChildrenMemo = useMemo(() => props.children(form), [])

	return (
		<Modal
			// confirmLoading={props.loading}
			destroyOnClose={true}
			maskClosable={false}
			width={props.width || 680}
			visible={props.visible}
			title={props.title}
			okText={t('确定')}
			cancelText={t('取消')}
			onCancel={() => {
				form.resetFields();
				props.onCancel();
			}}
			onOk={() => {
				// console.log(form.getFieldsValue())
				form.validateFields()
					.then((values) => {
						props.onCreate(values, form);
						// form.resetFields();
					})
					.catch((info) => {
						// console.log('Validate Failed:', info);
					});
			}}
		>
			<Spin spinning={props.loading}>
				<Form onValuesChange={(value, allValues) => {
					setFormChangeRes({
						currentChange: value,
						allValues
					})
				}} labelCol={{ span: 5 }} wrapperCol={{ span: 19 }} form={form} layout="horizontal" name="form_in_modal">
					{props.children && Object.prototype.toString.call(props.children) === '[object Function]'
						? props.children(form, formChangeRes)
						: props.children}
				</Form>
			</Spin>
		</Modal>
	);
};

export default ModalForm;
