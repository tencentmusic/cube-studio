
import { useEffect, useState } from "react";
import { ITabDetailItem, ITabsModalData} from '../../api/interface/tabsModalInterface';
import TabsDetail from './TabsDetail';
import { Button, Modal, Spin } from "antd";
import { t } from "i18next";
import { actionTabsModal, actionTabsModalInfo } from "../../api/kubeflowApi";
interface IProps {
	visible: boolean;
	url: string;
    onVisibilityChange: (visible: boolean) => void;
}
export default function TabsModal(props: IProps) {
	const [loading, setLoading] = useState(false);
    const [data, setData] = useState<ITabsModalData>()

    useEffect(() => {
        if(props.visible && props.url) {
            fatchData(props.url);
        }
        // fatchData(props.url)
	}, [props.url,props.visible]);

    const handleCancel = () => {
      props.onVisibilityChange(false);
      setData(undefined);  // 重置数据，确保下次打开会重新获取
    };

    const fatchData = (url: string) => {
        if (!!url) {
            setLoading(true);
            actionTabsModalInfo(url)
                .then((res) => {
                    if (res.data) {
                        setData(res.data.result);
                    } else {
                        console.error('Invalid data format:', res.data);
                        setData(undefined);
                    }
                })
                .catch((err) => {
                    console.error('Error fetching data:', err);
                })
                .finally(() => {
                    setLoading(false);
                });
        }
    };
    return (
        <Modal
            destroyOnClose={true}
            maskClosable={false}
            width={680}
            open={props.visible}
            title={data?.title}
            onCancel={handleCancel}
            onOk={() => props.onVisibilityChange(false)}
            footer={
                data?.bottomButton?.length ? (
                    <div className="flex justify-end">
                        {data.bottomButton.map((button, index) => (
                            <Button
                                key={`footerButton_${index}`}
                                type="primary"
                                onClick={() => actionTabsModal(button.method, button.url, button.arg)}
                                className="ml-2"
                            >
                                {button.icon && (
                                    <span
                                        className="mr-2"
                                        dangerouslySetInnerHTML={{ __html: button.icon }}
                                    ></span>
                                )}
                                {button.text}
                            </Button>
                        ))}
                    </div>
                ) : null
            }
        >
            <Spin spinning={loading}>
                {data ? <TabsDetail data={data.content || []} /> : <div>{t('No data available')}</div>}
            </Spin>
        </Modal>
    );

}