import React, { ReactNode } from 'react';
import { Row, Typography, Col, Space } from 'antd';
import './TitleHeader.less';

const { Title } = Typography;

interface IProps {
    title: ReactNode | string;
    children?: ReactNode;
    noBorderBottom?: boolean;
    breadcrumbs?: ReactNode | string;
}

const TitleHeader = (props: IProps) => {
    const styles: React.CSSProperties = { position: 'sticky', top: 0 }
    return (
        <Row
            className="title-header"
            justify="space-between"
            align="middle"
            style={props.noBorderBottom ? { borderBottom: 'none', ...styles } : styles}>
            <div>
                <Title className="d-il mr12" level={5} style={{ marginBottom: 10 }}>
                    {props.title}
                </Title>
                <div className="d-il">
                    {props.breadcrumbs}
                </div>
            </div>

            <Col>
                <Space>{props.children ? props.children : null}</Space>
            </Col>
        </Row>
    );
};

export default TitleHeader;
