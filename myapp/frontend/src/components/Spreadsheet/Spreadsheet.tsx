import React, { useEffect, useState } from 'react'
import XDataSpreadsheet from "x-data-spreadsheet";
import 'x-data-spreadsheet/dist/locale/zh-cn'
import './xDataSpreadsheet.less'

export interface ISpreadsheetProps {
    width?: string | number
    height?: string | number
    dataSource: Record<any, any>[]
    onChange?: (data: Record<string, any>) => void
}

export default function Spreadsheet(props: ISpreadsheetProps) {
    const [spreadsheetInstance, setSpreadsheetInstance] = useState<XDataSpreadsheet>()

    const options: any = {
        mode: 'edit', // edit | read
        showToolbar: true,
        showGrid: true,
        showContextmenu: true,
        view: {
            height: () => props.height || 500,
            width: () => props.width || 1000,
        },
        row: {
            len: 100,
            height: 25,
        },
        col: {
            len: 50,
            width: 100,
            indexWidth: 60,
            minWidth: 60,
        },
        style: {
            bgcolor: '#ffffff',
            align: 'left',
            valign: 'middle',
            textwrap: false,
            strike: false,
            underline: false,
            color: '#0a0a0a',
            font: {
                name: 'Helvetica',
                size: 10,
                bold: false,
                italic: false,
            },
        },
    }

    useEffect(() => {
        if (props.dataSource && props.dataSource.length) {
            XDataSpreadsheet.locale('zh-cn', (window.x_spreadsheet as any).$messages['zh-cn'])

            const formatData = createDataRows(props.dataSource)

            const currentOptions = {
                ...options,
                name: 'sheet1',
                rows: formatData,
                row: {
                    len: props.dataSource.length,
                    height: 25,
                },
                col: {
                    len: props.dataSource[0].length,
                    width: 100,
                    indexWidth: 60,
                    minWidth: 60,
                },
            }

            const instance = new XDataSpreadsheet("#xdataspreadsheet", currentOptions)
                .loadData({
                    rows: formatData,
                })
                .change(data => {
                    props.onChange && props.onChange(data)
                });

            console.log('currentOptions', currentOptions);

            setSpreadsheetInstance(instance)
        }
    }, [props.dataSource])

    const createDataRows = (data: Record<any, any>[]) => {
        const len = data.length < 50 ? 50 : data.length
        const formatRows: any = {}
        data.forEach((row, rowIndex) => {
            const colFormat: any = {}
            Object.keys(row).forEach((col, colIndex) => {
                colFormat[colIndex] = {
                    text: row[col]
                }
            })

            formatRows[rowIndex.toString()] = {
                cells: colFormat
            }

        })

        return {
            len,
            ...formatRows
        }
    }

    return (
        <div>
            <div style={{ width: props.width || 1000 }} id="xdataspreadsheet"></div>
            {
                !(props.dataSource && props.dataSource.length) ? <div className="d-f jc ac c-hint-b h320 fs16">暂无数据</div> : null
            }
        </div>
    )
}
