import React, { ReactNode, useEffect, useState } from 'react';
import { Row, Col, Space, Table, ConfigProvider, Button, Modal, Tabs, message, Checkbox } from 'antd';
import './TableBox.less';
import { TablePaginationConfig } from 'antd/lib/table/Table';
import emptyImg from '../../images/emptyBg.png';
import { GetRowKey, SorterResult, TableRowSelection } from 'antd/lib/table/interface';
// import ExportJsonExcel from 'js-export-excel';
import { Resizable } from 'react-resizable';
import { useTranslation } from 'react-i18next';

const CopyToClipboard = require('react-copy-to-clipboard');

interface IProps {
	size?: 'large' | 'middle' | 'small'
	tableKey?: string
	rowKey?: string | GetRowKey<any>;
	titleNode?: string | ReactNode;
	buttonNode?: ReactNode;
	dataSource: any;
	columns: any;
	pagination?: false | TablePaginationConfig;
	scroll?:
	| ({
		x?: string | number | true | undefined;
		y?: string | number | undefined;
	} & {
		scrollToFirstRowOnChange?: boolean | undefined;
	})
	| undefined;
	loading?: boolean;
	rowSelection?: TableRowSelection<any>;
	cancelExportData?: boolean;
	onChange?: (
		pagination: TablePaginationConfig,
		filters: Record<string, (string | number | boolean)[] | null>,
		sorter: SorterResult<any> | SorterResult<any>[],
	) => void;
}

const ResizableTitle = ({ onResize, width, ...restProps }: any) => {
	if (!width) {
		return <th {...restProps} />;
	}

	return (
		<Resizable
			width={width}
			height={0}
			handle={
				<span
					className="react-resizable-handle"
					onClick={(e) => {
						e.stopPropagation();
					}}
				/>
			}
			onResize={onResize}
			draggableOpts={{ enableUserSelectHack: false }}
		>
			<th {...restProps} style={{ ...restProps?.style, userSelect: 'none' }} />
		</Resizable>
	);
};

const TableBox = (props: IProps) => {
	const [exportDataVisible, setExportDataVisible] = useState(false);
	const [dataFormat, setDataFormat] = useState<{ header: any[]; data: any[] }>({
		header: [],
		data: [],
	});
	const [filterValue, setFilterValue] = useState<any[]>([]);

	// 可伸缩列设置
	const [cols, setCols] = useState(props.columns);
	const handleResize = (index: any) => {
		return (_: any, { size }: any) => {
			if (size.width < 100) return
			const temp = [...cols];
			temp[index] = { ...temp[index], width: size.width };
			const tableWidth = temp.reduce((pre: any, next: any) => pre + next.width || 100, 0) + 200
			localStorage.setItem(props.tableKey || '', JSON.stringify(temp))
			// console.log(currentTableScroll, temp);
			setCurrentTableScroll({ ...currentTableScroll, x: tableWidth })
			setCols(temp);
		};
	};
	const customColumns = cols.map((col: any, index: any) => {
		return {
			...col,
			width: col.width || 200,
			onHeaderCell: (column: any) => {
				return {
					width: column.width,
					onResize: handleResize(index),
				};
			},
		};
	});
	const [currentTableScroll, setCurrentTableScroll] = useState(props.scroll)
	const { t, i18n } = useTranslation();

	useEffect(() => {
		setCols(props.columns);
	}, [props.columns]);

	useEffect(() => {
		setCurrentTableScroll(props.scroll);
	}, [props.scroll]);

	useEffect(() => {
		if (props.dataSource) {
			const columns = props.columns.filter((item: any) => ~filterValue.indexOf(item.dataIndex));
			handdleFilterHeader(columns, props.dataSource);
		}
	}, [props.dataSource, props.columns]);

	const customizeRenderEmpty = () => (
		<Row justify="center" align="middle" style={{ height: 360, flexDirection: 'column' }}>
			<img src={emptyImg} style={{ width: 266 }} alt="" />
			<div>{t('暂无数据')}</div>
		</Row>
	);

	const handdleFilterHeader = (dataColumns = [], data: any[]) => {
		const columns = dataColumns.map((item: any) => item.dataIndex).filter((item: string) => item !== 'handle');
		const sheetHeader = dataColumns.map((item: any) => item.title).filter((item: string) => item !== t('操作'));
		const tarData: any = [];

		data.forEach((dataRow: any) => {
			const row: any = {};
			columns.map((colName: string) => {
				const res = dataRow[colName];
				row[colName] = res || '';
			});
			tarData.push(row);
		});

		setDataFormat({
			header: sheetHeader,
			data: tarData,
		});
	};

	// const handleClickOutputExcel = () => {
	// 	const option: any = {};
	// 	option.fileName = 'result';
	// 	option.datas = [
	// 		{
	// 			sheetData: dataFormat.data,
	// 			sheetName: 'sheet',
	// 			sheetHeader: dataFormat.header,
	// 		},
	// 	];
	// 	const toExcel = new ExportJsonExcel(option);
	// 	toExcel.saveExcel();
	// };

	const handleExportJira = () => {
		const header = dataFormat.header;
		const data = dataFormat.data;
		let str = '';
		if (header.length && data.length) {
			str =
				'|' +
				header.join('|') +
				'|' +
				`
`;
			data.forEach((row: any) => {
				const rowKey = Object.values(row).map((item) => {
					if (item === '') {
						return ' ';
					}
					return item;
				});
				str =
					str +
					'|' +
					rowKey.join('|') +
					'|' +
					`
`;
			});
		} else {
			str = '';
		}

		return str;
	};

	const handleExportText = () => {
		const header = dataFormat.header;
		const data = dataFormat.data;
		let str = '';
		if (header.length && data.length) {
			str =
				header.join('	') +
				`
`;
			data.forEach((row: any) => {
				const rowKey = Object.values(row).map((item) => {
					if (item === '') {
						return ' ';
					}
					return item;
				});
				str =
					str +
					rowKey.join('	') +
					`
`;
			});
		} else {
			str = '';
		}
		return str;
	};

	return (
		<Space className="tablebox" direction="vertical" size="middle">
			<Modal
				width={1000}
				maskClosable={false}
				centered={true}
				bodyStyle={{ maxHeight: 500, overflow: 'auto' }}
				visible={exportDataVisible}
				title={t('导出数据')}
				onCancel={() => {
					setExportDataVisible(false);
				}}
				footer={null}
			>
				<div style={{ position: 'relative' }}>
					<div className="mb16"><span className="pr8">{t('选择需要导出的列')}：</span><Checkbox.Group
						options={props.columns
							.map((item: any) => ({ label: item.title, value: item.dataIndex }))
							.filter((item: any) => item.value !== 'handle')}
						defaultValue={[]}
						value={filterValue}
						onChange={(values: any) => {
							setFilterValue(values);
							const columns = props.columns.filter((item: any) => ~values.indexOf(item.dataIndex));
							handdleFilterHeader(columns, props.dataSource);
						}}
					/></div>
					<div style={{ position: 'absolute', right: 0, bottom: 0 }}>
						<Button
							size="small"
							type="link"
							onClick={() => {
								setFilterValue(
									props.columns
										.map((item: any) => item.dataIndex)
										.filter((item: any) => item !== 'handle'),
								);
								handdleFilterHeader(props.columns, props.dataSource);
							}}
						>
							{t('全选')}
						</Button>
						<Button
							size="small"
							type="link"
							onClick={() => {
								setFilterValue([]);
								handdleFilterHeader([], props.dataSource);
							}}
						>
							{t('反选')}
						</Button>
					</div>
				</div>

				<Tabs>
					<Tabs.TabPane tab="Wiki格式" key="jira">
						<CopyToClipboard text={handleExportJira()} onCopy={() => message.success(t('已复制到粘贴板'))}>
							<pre style={{ cursor: 'pointer', minHeight: 100 }}>
								<code>{handleExportJira()}</code>
							</pre>
						</CopyToClipboard>
					</Tabs.TabPane>
					<Tabs.TabPane tab="Text格式" key="test">
						<CopyToClipboard text={handleExportText()} onCopy={() => message.success(t('已复制到粘贴板'))}>
							<pre style={{ cursor: 'pointer', minHeight: 100 }}>
								<code>{handleExportText()}</code>
							</pre>
						</CopyToClipboard>
					</Tabs.TabPane>
					{/* <Tabs.TabPane tab="Excel格式" key="excel">
						<Row justify="center" align="middle" style={{ minHeight: 100 }}>
							<Col>
								<Button type="primary" onClick={handleClickOutputExcel}>
									导出Excel
								</Button>
							</Col>
						</Row>
					</Tabs.TabPane> */}
				</Tabs>
			</Modal>
			{
				props.titleNode || props.buttonNode || !props.cancelExportData ? <Row justify="space-between" align="middle">
					<Col>
						<Space align="center">{props.titleNode}</Space>
					</Col>
					<Col>
						<Space align="center">
							{props.buttonNode}
							{props.cancelExportData ? null : (
								<Button style={{ marginLeft: 6 }} onClick={() => setExportDataVisible(true)}>
									{t('导出数据')}
								</Button>
							)}
						</Space>
					</Col>
				</Row> : null
			}
			<ConfigProvider renderEmpty={customizeRenderEmpty}>
				<Table
					size={props.size || 'middle'}
					rowKey={props.rowKey ? props.rowKey : 'id'}
					dataSource={props.dataSource}
					// columns={props.columns}
					components={{ header: { cell: ResizableTitle } }}
					columns={customColumns}
					pagination={props.pagination !== false ? { ...props.pagination } : false}
					scroll={currentTableScroll}
					loading={props.loading}
					onChange={props.onChange}
					rowSelection={props.rowSelection}
				/>
			</ConfigProvider>
		</Space>
	);
};

export default TableBox;
