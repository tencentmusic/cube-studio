import React from 'react';
import QueueAnim from 'rc-queue-anim';
import TweenOne from 'rc-tween-one';
import { getChildrenToRender } from './utils';

class Banner extends React.PureComponent {
  render() {
    const { ...tagProps } = this.props;
    const { dataSource } = tagProps;
    delete tagProps.dataSource;
    delete tagProps.isMobile;
    const animType = {
      queue: 'bottom',
      one: {
        y: '+=30',
        opacity: 0,
        type: 'from',
        ease: 'easeOutQuad',
      },
    };
    return (
      <div {...tagProps} {...dataSource.wrapper}>
        <div {...dataSource.page}>
          <QueueAnim
            key="text"
            type={animType.queue}
            leaveReverse
            ease={['easeOutQuad', 'easeInQuad']}
            {...dataSource.childWrapper}
            componentProps={{
              md: dataSource.childWrapper.md,
              xs: dataSource.childWrapper.xs,
            }}
          >
            {/* {dataSource.childWrapper.children.map(getChildrenToRender)} */}
            {/* <div className="banner5-title-wrapper">阿斯顿发斯蒂芬</div> */}
            <>
              <div style={{ marginBottom: -16 }}>
                <img style={{ height: 100 }} src={require('../../images/logoStar.svg').default} />
              </div>
              <h1 name="title" className="banner5-title mb16" style={{ letterSpacing: 6 }}>cube数据平台 </h1>
              {/* <div name="explain" className="banner5-explain"><img style={{ height: 36 }} src={require('../../images/logoStar.svg').default} /></div> */}
              <div name="content" className="banner5-content">数据资产服务旗下有星画，Superset，Swallow，等多款数据产品</div>
              <div name="button" className="banner5-button-wrapper">
                <a href="https://iwiki.woa.com/space/TMEDI" className="ant-btn ant-btn-primary banner5-button">
                  <span>了解更多</span>
                </a>
              </div>
            </>
          </QueueAnim>
        </div>
      </div>
    );
  }
}
export default Banner;
