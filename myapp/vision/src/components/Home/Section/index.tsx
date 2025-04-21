import React, { useEffect, useState } from 'react';
import { Icon, Stack, Modal } from '@fluentui/react';
import api from '@src/api';
import { updateErrMsg, selectUserName } from '@src/models/app';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import Player from 'xgplayer';
import style from './style';
import { useTranslation } from 'react-i18next';

interface ISectionProps {
  name: string;
  data: [];
  first?: boolean;
}

const Section: React.FC<ISectionProps> = props => {
  const dispatch = useAppDispatch();
  const [showMore, setShowMore] = useState<boolean>(false);
  const [showModal, setShowModal] = useState<boolean>(false);
  const [curVideo, setCurVideo] = useState<string>('');
  const [curPlayer, setCurPlayer] = useState<Player | undefined>(undefined);
  const userName = useAppSelector(selectUserName);
  const { t, i18n } = useTranslation();

  console.log('i18n.language',i18n.language)

  const handleNewPipeline = () => {
    api
      .pipeline_modelview_add({
        describe: `new-pipeline-${Date.now()}`,
        name: `${userName}-pipeline-${Date.now()}`,
        node_selector: 'cpu=true,train=true',
        schedule_type: 'once',
        image_pull_policy: 'Always',
        parallelism: 1,
        project: 7,
      })
      .then((res: any) => {
        if (res?.status === 0 && res?.message === 'success') {
          const url = `${window.location.origin}${location.pathname}?pipeline_id=${res?.result?.id}`;
          window.open(`${window.location.origin}/frontend/showOutLink?url=${encodeURIComponent(url)}`, 'bank');
          // if (window.self === window.top) {
          //   (window.top || window).location.href = `${window.location.origin}${location.pathname}?pipeline_id=${res?.result?.id}`;
          // } else {
          //   window.parent.postMessage(
          //     {
          //       type: 'link',
          //       message: {
          //         pipelineId: res?.result?.id,
          //       },
          //     },
          //     `${window.location.origin}`,
          //   );
          // }
        }
      })
      .catch(err => {
        if (err.response) {
          dispatch(updateErrMsg({ msg: err.response.data.message }));
        }
      });
  };

  const handleClick = (item: any) => {
    const url = `${window.location.origin}${location.pathname}?pipeline_id=${item?.id}`;
    switch (item.type) {
      case 'link':
        window.open(`${window.location.origin}/frontend/showOutLink?url=${encodeURIComponent(url)}`, 'bank');
        // if (window.self === window.top) {
        //   (window.top || window).location.href = `${window.location.origin}${location.pathname}?pipeline_id=${item?.id}`;
        // } else {
        //   window.parent.postMessage(
        //     {
        //       type: 'link',
        //       message: {
        //         pipelineId: item?.id,
        //       },
        //     },
        //     `${window.location.origin}`,
        //   );
        // }
        break;
      case 'outside':
        window.open(item.url, '_blank');
        break;
      case 'video':
        setShowModal(true);
        setCurVideo(item.url);
        break;
      default:
        break;
    }
  };

  useEffect(() => {
    if (showModal && curVideo) {
      setTimeout(() => {
        const player = new Player({
          id: 'video-player',
          url: curVideo,
          width: 800,
          height: 500,
          videoInit: true,
          playbackRate: [0.5, 0.75, 1, 1.5, 2],
          defaultPlaybackRate: 1,
          download: true,
          pip: true,
          disableProgress:false,
          allowPlayAfterEnded: true,
          allowSeekAfterEnded: true,
        });
        setCurPlayer(player);
      }, 0);
    }

    if (showModal && curPlayer) {
      curPlayer.destroy();
      setCurPlayer(undefined);
    }
  }, [showModal, curVideo]);

  useEffect(() => {
    if (curPlayer) {
      // curPlayer.play();
    }
  }, [curPlayer]);

  return (
    <Stack className={style.sectionStyles}>
      <Stack className="flex-section" horizontal horizontalAlign={'space-between'}>
        <div className="subtitle">{props.name}</div>
        <div
          className="expand-button"
          onClick={() => {
            setShowMore(!showMore);
          }}
        >
          {showMore ? t('折叠') : t('更多')}
        </div>
      </Stack>
      <div className={`${style.sampleStyles} ${showMore ? '' : style.sampleHide}`}>
        {/* create new pipeline */}
        {props.first ? (
          <div
            className={style.sampleCardStyle}
            onClick={() => {
              handleNewPipeline();
            }}
          >
            <div className={style.cardContainer}>
              <Icon iconName="Add" className={style.addIconStyles}></Icon>
            </div>
            <div className={style.cardTitleStyles}>
              <span>{t('新建流水线')}</span>
            </div>
          </div>
        ) : null}
        {/* sample card item */}
        {props.data.map((item: any, key: number) => {
          if (!item.img) return null;
          return (
            <div
              className={style.sampleCardStyle}
              key={key}
              onClick={() => {
                handleClick(item);
              }}
            >
              <div className={style.cardContainer}>
                <div className={style.sampleImgStyles}>
                  <img src={item.img} alt={item.name} />
                </div>
              </div>
              <div className={style.cardTitleStyles}>
                <span>{item.name}</span>
              </div>
            </div>
          );
        })}
      </div>
      <Modal isOpen={showModal} onDismiss={() => setShowModal(false)}>
        <div id="video-player" className={style.videoPlayerStyles}></div>
      </Modal>
    </Stack>
  );
};

export default Section;
