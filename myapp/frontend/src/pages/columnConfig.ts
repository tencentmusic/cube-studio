export interface IColumnConfigItem {
    name: string,
    width: number
    title?: string
    type?: 'ellip1'
    filter?: string
}

export const columnConfig: Record<string, IColumnConfigItem> = {

}