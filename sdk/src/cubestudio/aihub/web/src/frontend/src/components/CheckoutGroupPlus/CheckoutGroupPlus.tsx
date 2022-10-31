import { LabeledValue } from 'antd/lib/select';
import React, { ChangeEvent, ReactNode } from 'react';
import './CheckoutGroupPlus.less'

declare module 'react' {
    interface HTMLAttributes<T> extends AriaAttributes, DOMAttributes<T> {
        // extends React's HTMLAttributes
        trigger?: string;
    }
}

interface CheckboxProps {
    name?: string,
    label?: string,
    className?: string,
    checked?: boolean | undefined,
    defaultChecked?: boolean,
    disabled?: boolean,
    value?: number | string,
    onChange?: (e: ChangeEvent<HTMLInputElement>) => void
}

interface CheckboxGroupProps {
    values?: string[],
    defaultValue?: string[],
    disabled?: boolean,
    // option: LabeledValue[],
    option: ICheckboxOptions[],
    className?: string,
    children?: JSX.Element,
    onChange?: (values: Array<string | number>) => void
}

export interface ICheckboxOptions {
    label: ReactNode,
    value: string | number,
    display?: boolean,
    disabled?: boolean,
}

export default function Checkbox(props: CheckboxProps): JSX.Element {
    const id = Math.random().toString(36).substring(2);

    const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
        props.onChange && props.onChange(e);
    };

    return (
        <div className={['checkbox-box-normalize mr16 d-il', props.className].join(' ')}>
            {
                props.checked === undefined ? <input
                    id={id}
                    trigger='core'
                    type="checkbox"
                    disabled={props.disabled}
                    name={props.name || ''}
                    value={props.value}
                    defaultChecked={props.defaultChecked || false}
                    onChange={handleChange}
                /> : <input
                        id={id}
                        trigger='core'
                        type="checkbox"
                        disabled={props.disabled}
                        name={props.name || ''}
                        value={props.value}
                        checked={props.checked}
                        onChange={handleChange}
                    />
            }
            <span className="checkbox-hook ta-c">
                <span className="checkbox-hook-in fs12 op0">✓</span>
            </span>
            <label htmlFor={id} className="p-r z10 pl8">{props.label || ''}</label>
        </div>
    );
}

export function CheckboxFontIn(props: CheckboxProps): JSX.Element {
    const id = Math.random().toString(36).substring(2);

    const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
        props.onChange && props.onChange(e);
        console.log(e);
    };

    return (
        <div className={['checkbox-box-fontstyle d-il mr16 mb8', props.className].join(' ')}>
            {
                props.checked === undefined ? <input
                    id={id}
                    trigger='core'
                    type="checkbox"
                    disabled={props.disabled}
                    className="d-n"
                    name={props.name}
                    value={props.value}
                    defaultChecked={props.defaultChecked || false}
                    onChange={handleChange}
                /> : <input
                        id={id}
                        trigger='core'
                        type="checkbox"
                        disabled={props.disabled}
                        className="d-n"
                        name={props.name}
                        value={props.value}
                        checked={props.checked}
                        onChange={handleChange}
                    />
            }
            <label
                htmlFor={id}
                className="checkbox-fontstyle mb0">
                <span className="m0">{props.label || ''}</span>
            </label>
        </div>
    );
}

export function CheckboxImageIn(props: CheckboxProps): JSX.Element {
    const id = Math.random().toString(36).substring(2);

    const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
        props.onChange && props.onChange(e);
    };

    return (
        <div className={['checkbox-box-imgstyle d-il mr16 mb8', props.className].join(' ')}>
            {
                props.checked === undefined ? <input
                    id={id}
                    trigger='core'
                    type="checkbox"
                    disabled={props.disabled}
                    className="d-n"
                    name={props.name}
                    value={props.value}
                    defaultChecked={props.defaultChecked || false}
                    onChange={handleChange}
                /> : <input
                        id={id}
                        trigger='core'
                        type="checkbox"
                        disabled={props.disabled}
                        className="d-n"
                        name={props.name}
                        value={props.value}
                        checked={props.checked}
                        onChange={handleChange}
                    />
            }
            <label
                htmlFor={id}
                className="checkbox-imgstyle mb0">
                <img src={`${props.value}`} alt="" />
                <p className="m0">{props.label}</p>
                <div className="checkbox-mark"><span>✓</span></div>
            </label>
        </div>
    );
}

export function CheckboxBorder(props: CheckboxProps): JSX.Element {
    const id = Math.random().toString(36).substring(2);

    const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
        props.onChange && props.onChange(e);
    };

    return (
        <div className="trigger-box-border d-il mr16 mb8">
            {
                props.checked === undefined ? <input
                    id={id}
                    trigger='core'
                    type="checkbox"
                    disabled={props.disabled}
                    className="d-n"
                    name={props.name}
                    value={props.value}
                    defaultChecked={props.defaultChecked || false}
                    onChange={handleChange}
                /> : <input
                        id={id}
                        trigger='core'
                        type="checkbox"
                        disabled={props.disabled}
                        className="d-n"
                        name={props.name}
                        value={props.value}
                        checked={props.checked}
                        onChange={handleChange}
                    />
            }
            <label
                htmlFor={id}
                className="trigger-border mb0"
            >
                <span className="m0">{props.label || ''}</span>
            </label>
        </div>
    );
}

const GroupContainer = (Component: any) => function Group(props: CheckboxGroupProps): JSX.Element {
    const name = Math.random().toString(36).substring(2);

    const isInArray = (arr: string[], value: string | number): boolean | undefined => {
        if (Array.isArray(arr)) {
            return arr.indexOf(value as string) !== -1;
        }
        return undefined;
    };

    const onChange = (e: ChangeEvent<HTMLInputElement>): void => {
        const { value, checked } = e.target;

        let values: Array<string | number> = [];
        let res: Array<string | number> = [];

        // 可控
        if (props.values) {
            values = [...props.values];
        } else {
            // 非可控
            values = [...(props?.defaultValue || [])];
        }

        if (checked) {
            res = [...values, value];
        } else {
            const index = values.indexOf(value);
            if (index !== -1) {
                values.splice(index, 1);
                res = [...values];
            }
        }
        props.onChange && props.onChange(res);
    };

    return (
        <div className={props.className || ''}>
            {
                props.option.map((item, index) => {
                    const componentProps = {
                        defaultChecked: isInArray(props.defaultValue || [], item.value),
                        checked: isInArray(props.values || [], item.value),
                        name: name,
                        label: item.label,
                        value: item.value,
                        disabled: item.disabled || props.disabled,
                        display: item.display,
                        onChange: onChange
                    };
                    return <Component {...componentProps} key={index} />;
                })
            }
        </div>
    );
};

Checkbox.Group = GroupContainer(Checkbox);
Checkbox.GroupFontIn = GroupContainer(CheckboxFontIn);
Checkbox.GroupBorder = GroupContainer(CheckboxBorder);
Checkbox.GroupImageIn = GroupContainer(CheckboxImageIn);
