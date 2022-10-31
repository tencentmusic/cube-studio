import { LoadingOutlined, SearchOutlined } from '@ant-design/icons';
import { Input, Menu } from 'antd';
import React, { useState, ChangeEvent, useEffect } from 'react';
import './DataDiscoverySearch.less';

interface IOptionsItem {
    label: string
    value: string
    [key: string]: any
}

export interface IOptionsGroupItem {
    name: string
    option: IOptionsItem[]
}

interface IProps {
    labelName?: string,
    width?: string,
    placeholder?: string,
    maxLength?: number,
    maxHeight?: number,
    // 是否开启前端搜索匹配
    isOpenSearchMatch?: boolean,
    loading?: boolean | JSX.Element,
    // 配置提示列表
    options?: IOptionsGroupItem[],
    // 当配置value时，即为可控组件
    value?: string,
    disabled?: boolean
    // 按回车时回调
    onSearch?: (value: string) => void,
    // 输入字符、按下回车时回调
    onChange?: (value: string) => void,
    // 点击option中的item
    onClick?: (value: string, option: IOptionsItem) => void,
    // 滚动条到底时触发
    onScrollButtom?: () => void
}

export default function DataDiscoverySearch(props: IProps): JSX.Element {
    const id = Math.random().toString(36).substring(2);
    let inputRef: any;

    const [dataCache, setDataCache] = useState<IOptionsGroupItem[]>(props.options || []);
    const [value, setValue] = useState(props.value || '');

    useEffect(() => {
        let dataFilter = props.options || []

        if (props.isOpenSearchMatch) {
            setDataCache(dataFilter);
        } else {
            dataFilter = (props.options || [])
                .map(group => ({ ...group, option: group.option.filter(item => item.value.indexOf(value) !== -1) }))
            setDataCache(dataFilter);
        }
    }, [props.options]);

    useEffect(() => {
        let dataFilter = props.options || []

        if (props.isOpenSearchMatch) {
            setDataCache(dataFilter);
        } else {
            dataFilter = (props.options || [])
                .map(group => ({ ...group, option: group.option.filter(item => item.value.indexOf(value) !== -1) }))
            setDataCache(dataFilter);
        }

        setValue(props.value || '');
        // props.onChange && props.onChange(props.value);
    }, [props.value]);

    const handleChange = (value: string): void => {
        setValue(value);
        props.onChange && props.onChange(value);
    };

    const handleClick = (value: string, option: IOptionsItem): void => {
        handleChange(value);
        inputRef.blur && inputRef.blur();
        props.onClick && props.onClick(value, option);
    };

    const handleEnterKey = (e: React.KeyboardEvent<HTMLInputElement>): void => {
        // 回车
        if (e.nativeEvent.keyCode === 13) {
            // inputRef.blur && inputRef.blur();
            props.onSearch && props.onSearch(e.currentTarget.value);
        }
    };

    const highlightKeyWord = (item: string): JSX.Element => {
        const keyWord = value;
        const index = item.indexOf(value);
        if (index === -1) {
            return <span>{item}</span>;
        }
        const preStr = item.substring(0, index);
        const nextStr = item.substring(index + value.length);
        return <span>{preStr}<span className="highlight">{keyWord}</span>{nextStr}</span>;
    };

    const debounce = (fun: any, time = 500): any => {
        let timer: ReturnType<typeof setTimeout>;
        return function (...args: any): void {
            clearTimeout(timer);
            timer = setTimeout(() => {
                fun && fun.apply(null, [...args]);
            }, time);
        };
    };

    const debounceScroll = debounce(props.onScrollButtom);

    const handleScroll = (e: React.UIEvent<HTMLElement>): void => {
        e.stopPropagation();
        // console.log({
        //     event: e,
        //     target: e.target, // Note 1* scrollTop is undefined on e.target
        //     currentTarget: e.currentTarget,
        //     scrollTop: e.currentTarget.scrollTop,
        //     scrollHeight: e.currentTarget.scrollHeight,
        //     clientHeight: e.currentTarget.clientHeight
        // });
        const { currentTarget } = e;
        const { scrollTop, clientHeight, scrollHeight } = currentTarget;
        const difference = scrollHeight - clientHeight - scrollTop;
        if (difference < 20) {
            props.onScrollButtom && debounceScroll();
        }
    };

    return (
        <div className="select-down-discovery">
            {
                props.labelName ? <label htmlFor={id} className="pb4 mb0 fs12 d-b">{props.labelName}</label> : null
            }
            <div className="p-r d-f ac" style={{ width: props.width || 100 + '%' }}>
                <Input
                    style={{ width: '100%' }}
                    disabled={props.disabled}
                    id={id}
                    placeholder={props.placeholder || ''}
                    maxLength={props.maxLength || 200}
                    onChange={(e: ChangeEvent<HTMLInputElement>): void => handleChange(e.target.value)}
                    onKeyPress={handleEnterKey}
                    value={value}
                    ref={element => inputRef = element}
                />
                {
                    props.loading ? <LoadingOutlined className="p-a r0 mr8" /> : <SearchOutlined onClick={() => {
                        props.onSearch && props.onSearch(value);
                    }} className="p-a r0 mr8 cp" />
                }
            </div>

            {
                // 输入提示
                dataCache.length ? <Menu className="select-option" onScroll={handleScroll} style={{ 'maxHeight': `${props.maxHeight}px` }}>
                    {
                        props.loading ? <div className="p-s z9 ta-r" style={{ right: `${0}px`, top: `${0}px` }}>
                            <div className="d-il p-a" style={{ right: `${8}px`, top: `${0}px` }}></div>
                        </div> : null
                    }
                    {
                        props.options?.map((group, groupIndex) => {
                            if (group.option.length) {
                                return <Menu.ItemGroup title={group.name} key={`dataDiscovery_${groupIndex}`}>
                                    {
                                        group.option.map((item, index) => {
                                            return <Menu.Item className="ellip1" onMouseDown={(): void => handleClick(item.value, item)} key={`dataDiscoveryItem_${group.name}_${index}`}>{highlightKeyWord(item.value)}</Menu.Item>;
                                        })
                                    }
                                </Menu.ItemGroup>
                            }
                            return null
                        })
                    }
                </Menu> : null
            }
        </div>
    );
}