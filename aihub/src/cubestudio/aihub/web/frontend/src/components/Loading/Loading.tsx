import React, { ReactNode } from 'react'
import './Loading.less';

interface IProps {
    value: number
    content?: ReactNode
}

export default function Loading(props: IProps) {
    return (
        <div className="loading-container fadein">
            <div className="container">
                <span>{props.value}s</span>
                <div className="circle">
                    <div className="ring"></div>
                </div>
            </div>
            {props.content}
        </div>
    )
}
