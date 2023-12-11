import React from 'react';
import { Stack, TooltipHost, IconButton, PrimaryButton, DefaultButton } from '@fluentui/react';
import ErrorTips from '../ErrorTips';
import { useAppSelector, useAppDispatch } from '@src/models/hooks';
import { updateErrMsg, selectUserName } from '@src/models/app';
import { selectPipelineId, selectInfo, savePipeline } from '@src/models/pipeline';
import { saveTaskList } from '@src/models/task';
import { toggle } from '@src/models/setting';
import api from '@src/api';
import style from './style';
import { useTranslation } from 'react-i18next';
const { Item } = Stack;

const EditorHead: React.FC = () => {
  const dispatch = useAppDispatch();
  const pipelineId = useAppSelector(selectPipelineId);
  const info = useAppSelector(selectInfo);
  const userName = useAppSelector(selectUserName);

  const { t, i18n } = useTranslation();

  // 新建流水线
  const handleNewPipeline = () => {
    api
      .pipeline_modelview_add({
        describe: `${t('新建流水线')}-${Date.now()}`,
        name: `${userName}-pipeline-${Date.now()}`,
        node_selector: 'cpu=true,train=true',
        schedule_type: 'once',
        image_pull_policy: 'Always',
        parallelism: 1,
        project: 7,
      })
      .then((res: any) => {
        if (res?.status === 0 && res?.message === 'success') {
          window.location.search = `?pipeline_id=${res?.result?.id}`;
        }
      })
      .catch(err => {
        if (err.response) {
          dispatch(updateErrMsg({ msg: err.response.data.message }));
        }
      });
  };

  // pipeline run
  const handleSubmit = async () => {
    if (pipelineId) {
      await dispatch(await saveTaskList());
      dispatch(savePipeline());
      window.open(`${window.location.origin}/pipeline_modelview/run_pipeline/${pipelineId}`);
    }
  };

  return (
    <Item className="editor-head">
      <Stack horizontal horizontalAlign="space-between" verticalAlign="center" styles={style.headStyle}>
        <Item grow={1}>
          <Stack horizontal>
            <Item>
              {info.name ? (
                <div className={style.headNameStyle}>{info.describe}</div>
              ) : (
                <PrimaryButton
                  styles={{
                    root: {
                      padding: '17px 16px',
                    },
                  }}
                  iconProps={{
                    iconName: 'Add',
                    styles: {
                      root: {
                        fontSize: 10,
                      },
                    },
                  }}
                  onClick={handleNewPipeline}
                >
                  {(t('新建流水线'))}
                </PrimaryButton>
              )}
            </Item>
            <Item className={info.name ? '' : style.hidden}>
              <TooltipHost content={t('设置')}>
                <IconButton
                  iconProps={{ iconName: 'Settings' }}
                  onClick={() => {
                    dispatch(toggle());
                  }}
                ></IconButton>
              </TooltipHost>
            </Item>
            <Item grow={1} align={'center'}>
              <ErrorTips></ErrorTips>
            </Item>
          </Stack>
        </Item>
        <Item className={info.name ? '' : style.hidden}>
          <Stack horizontal>
            <Item styles={style.buttonItemStyle}>
              <PrimaryButton onClick={handleSubmit}>{t('运行')}</PrimaryButton>
            </Item>
            <Item styles={style.buttonItemStyle}>
              <DefaultButton
                onClick={() => {
                  api
                    .pipeline_modelview_copy(pipelineId)
                    .then((res: any) => {
                      if (res?.id) {
                        if (window.self === window.top) {
                          window.location.search = `?pipeline_id=${res.id}`;
                        } else {
                          window.parent.postMessage(
                            {
                              type: 'copy',
                              message: {
                                pipelineId: res.id,
                              },
                            },
                            `${window.location.origin}`,
                          );
                        }
                      }
                    })
                    .catch(err => {
                      if (err.response) {
                        dispatch(updateErrMsg({ msg: err.response.data.message }));
                      }
                    });
                }}
              >
                {(t('复制'))}
              </DefaultButton>
            </Item>
          </Stack>
        </Item>
      </Stack>
    </Item>
  );
};

export default EditorHead;
