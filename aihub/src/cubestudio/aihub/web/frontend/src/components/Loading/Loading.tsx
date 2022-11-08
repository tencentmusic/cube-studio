import React, { ReactNode } from 'react'
import './Loading.less';

interface IProps {
    value: number
    content?: ReactNode
}

export default function Loading(props: IProps) {
    return (
        <div className="loading-container">
            <div className="g-container">
                <div className="g-number">{props.value}%</div>
                <div className="g-contrast">
                    <div className="g-circle"></div>
                    <ul className="g-bubbles">
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                        <li></li>
                    </ul>
                </div>
                {
                    props.content
                }
            </div>
        </div>
    )
}
