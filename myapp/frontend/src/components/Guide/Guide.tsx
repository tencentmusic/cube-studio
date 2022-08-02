import React, { useEffect, useState } from 'react';
import './Guide.less';

interface IProps {
    option: GiodeItem[],
    className?: string,
    isClose: boolean,
    containerId?: string
    onFinish?: () => void
}

interface GiodeItem {
    maskType: MaskType,
    tipPosition: TipPosition,
    tipAlign: TipAlign,
    maskDisplay: boolean,
    content: JSX.Element | string | number,
    targetId: string,
}

export type MaskType = 'circle' | 'rect';

export type TipPosition = 'top' | 'left' | 'right' | 'bottom';

export type TipAlign = 'center' | 'left' | 'right';


export default function Guide(props: IProps): JSX.Element {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isOpen, setIsOpen] = useState(true);

    const handleTargetPosition = (id: string): {
        width?: string,
        height?: string,
        borderWidth?: string,
        display?: string
    } => {
        const doc = document.documentElement;
        const body = document.body;

        const target = document.getElementById(id);
        const container = document.getElementById(props.containerId || '');
        if (target) {
            if (container && target.getBoundingClientRect().top + target.getBoundingClientRect().height > container.clientHeight) {
                // console.log('target', target, target.getBoundingClientRect(), target.getBoundingClientRect().top + target.getBoundingClientRect().height, container.clientHeight);
                container.scrollTop = target.getBoundingClientRect().top + target.getBoundingClientRect().height - container.clientHeight;
            }

            const targetWidth = target.getBoundingClientRect().width;
            const targetHeight = target.getBoundingClientRect().height;
            // page size
            const pageHeight = doc.scrollHeight;
            const pageWidth = doc.scrollWidth;

            // offset of target    
            const offsetTop = target.getBoundingClientRect().top + (body.scrollTop || doc.scrollTop) - (container ? container.offsetTop : 0);
            const offsetLeft = target.getBoundingClientRect().left + (body.scrollLeft || doc.scrollLeft) - (container ? container.offsetLeft : 0);

            // console.log(`${offsetTop}px ${pageWidth - targetWidth - offsetLeft}px ${pageHeight - targetHeight - offsetTop}px ${offsetLeft}px`);

            // set size and border-width
            const style = {
                width: targetWidth + 'px',
                height: targetHeight + 'px',
                borderWidth: `${offsetTop}px ${pageWidth - targetWidth - offsetLeft}px ${pageHeight - targetHeight - offsetTop}px ${offsetLeft}px`
            };
            return style;
        }
        return {
            display: 'none'
        };
    };

    const handleNextStep = (index: number): void => {
        if (index + 1 === props.option.length) {
            props.onFinish && props.onFinish();
            setIsOpen(false);
        } else {
            setCurrentIndex(currentIndex + 1);
        }
    };

    const handleSkipStep = (): void => {
        setIsOpen(false);
    };

    const handleTipPosition = (tipPosition: TipPosition): {
        top?: string,
        left?: string,
        bottom?: string,
        right?: string
    } => {
        switch (tipPosition) {
            case 'bottom':
                return {
                    top: 'calc(100% + 15px)'
                };
            case 'top':
                return {
                    bottom: 'calc(100% + 15px)'
                };
            case 'left':
                return {
                    right: 'calc(100% + 15px)'
                };
            case 'right':
                return {
                    left: 'calc(100% + 15px)'
                };
            default:
                return {
                    top: 'calc(100% + 15px)'
                };
        }
    };

    const handleTipAlign = (tipAlign: TipAlign): {
        position: any,
        right?: number,
        left?: number
    } => {
        switch (tipAlign) {
            case 'center':
                return {
                    position: 'relative'
                };
            case 'left':
                return {
                    position: 'absolute',
                    left: 0
                };
            case 'right':
                return {
                    position: 'absolute',
                    right: 0
                };
            default:
                return {
                    position: 'relative'
                };
        }
    };

    const handleTipArrow = (tipPosition: TipPosition, tipAlign: TipAlign): {
        bottom?: string,
        top?: string,
        left?: string,
        right?: string,
        borderColor: string,
        margin: string
    } => {
        const handleTipAlign = (tipAlign: TipAlign): string => {
            switch (tipAlign) {
                case 'center':
                    return '0 auto';
                case 'left':
                    return '0 auto 0 16px';
                case 'right':
                    return '0 16px 0 auto';
                default:
                    return '0 auto';
            }
        };

        switch (tipPosition) {
            case 'bottom':
                return {
                    bottom: '100%',
                    borderColor: 'transparent transparent #fff transparent',
                    margin: handleTipAlign(tipAlign)
                };
            case 'top':
                return {
                    top: '100%',
                    borderColor: '#fff transparent transparent transparent',
                    margin: handleTipAlign(tipAlign)
                };
            case 'left':
                return {
                    left: '100%',
                    borderColor: 'transparent transparent transparent #fff',
                    margin: handleTipAlign(tipAlign)
                };
            case 'right':
                return {
                    right: '100%',
                    borderColor: 'transparent #fff transparent transparent',
                    margin: handleTipAlign(tipAlign)
                };
            default:
                return {
                    bottom: '100%',
                    borderColor: 'transparent transparent #fff transparent',
                    margin: handleTipAlign(tipAlign)
                };
        }
    };

    return <div>
        {
            !props.isClose && isOpen ? [props.option[currentIndex]].map((item) => {
                // index === currentIndex ? '' : 'd-n'
                return <div className={[props.className].join(' ')} key={item.targetId}>
                    <div className="guide-cover" style={handleTargetPosition(item.targetId)}>
                        <div className="guide-tip-container" style={handleTipPosition(item.tipPosition)}>
                            <div className="d-il guide-tip" style={handleTipAlign(item.tipAlign)}>
                                <div className="guide-arrow" style={handleTipArrow(item.tipPosition, item.tipAlign)}></div>
                                <div>
                                    {item.content}
                                </div>
                                <div className="fs12 pt16 d-f jc-b">
                                    <span className="pr16 c-hint-b">{`${currentIndex + 1} / ${props.option.length}`}</span>
                                    <div>
                                        <div className="btn tp c-hint-b d-il mr16 cp" onClick={handleSkipStep}>暂时跳过指引</div>
                                        <div className="btn tp c-theme d-il cp" onClick={(): void => handleNextStep(currentIndex)}>{currentIndex + 1 === props.option.length ? '完成' : '下一步'}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>;
            }) : null
        }
    </div>;
}