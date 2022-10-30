import { SearchOutlined } from '@ant-design/icons';
import { Input } from 'antd';
import React, { useState, ChangeEvent, useEffect } from 'react';
import './InputSearch.less';

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
    options?: string[],
    // 当配置value时，即为可控组件
    value?: string,
    disabled?: boolean
    // 按回车时回调
    onSearch?: (value: string) => void,
    // 输入字符、按下回车时回调
    onChange?: (value: string) => void,
    // 点击option中的item
    onClick?: (value: string) => void,
    // 滚动条到底时触发
    onScrollButtom?: () => void
}

export default function InputSearch(props: IProps): JSX.Element {
    const id = Math.random().toString(36).substring(2);
    let inputRef: any;

    const [dataCache, setDataCache] = useState<string[]>(props.options || []);
    const [value, setValue] = useState(props.value || '');

    useEffect(() => {
        const dataFilter = props.isOpenSearchMatch ? (props.options || []).filter(item => {
            return item.indexOf(value) !== -1;
        }) : (props.options || []);
        setDataCache(dataFilter);
    }, [props.options]);

    useEffect(() => {
        setValue(props.value || '');
        // props.onChange && props.onChange(props.value);
    }, [props.value]);

    const handleChange = (value: string): void => {
        setValue(value);
        props.onChange && props.onChange(value);
    };

    const handleClick = (value: string): void => {
        handleChange(value);
        props.onClick && props.onClick(value);
    };

    const handleEnterKey = (e: React.KeyboardEvent<HTMLInputElement>): void => {
        // 回车
        if (e.nativeEvent.keyCode === 13) {
            inputRef.blur && inputRef.blur();
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
        <div className="select-down-modern">
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
                <SearchOutlined className="p-a r0 mr8" />
            </div>

            {
                // 输入提示
                dataCache.length ? <ul className="select-option shadow" onScroll={handleScroll} style={{ 'maxHeight': `${props.maxHeight}px` }}>
                    {
                        props.loading ? <div className="p-s z9 ta-r" style={{ right: `${0}px`, top: `${0}px` }}>
                            <div className="d-il p-a" style={{ right: `${8}px`, top: `${0}px` }}></div>
                        </div> : null
                    }
                    {
                        dataCache.map((item, index) => {
                            return <li className="ellip1" onMouseDown={(): void => handleClick(item)} key={index}>{highlightKeyWord(item)}</li>;
                        })
                    }
                </ul> : null
            }
        </div>
    );
}