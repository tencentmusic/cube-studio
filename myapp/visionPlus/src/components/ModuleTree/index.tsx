import React, { useState, useEffect } from 'react';
import { Stack, SearchBox, Text, FocusZone, IconButton, Icon, Spinner, SpinnerSize, Callout } from '@fluentui/react';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { updateErrMsg } from '@src/models/app';
import { selectShow } from '@src/models/template';
import api from '@src/api';
import storage from '@src/utils/storage';
import { debounce } from '@src/utils';
import ModuleDetail from './components/ModuleDetail';
import ModuleItem from './components/ModuleItem';
import SearchItem from './components/SearchItem';
import style from './style';

const { Item } = Stack;

const ColorList: Array<{
  color: string,
  bg: string
}> = [
    {
      color: 'rgba(0,120,212,1)',
      bg: 'rgba(0,120,212,0.02)',
    }, {
      color: 'rgba(0,170,200,1)',
      bg: 'rgba(0,170,200,0.02)',
    }, {
      color: 'rgba(0,200,153,1)',
      bg: 'rgba(0,200,153,0.02)',
    }, {
      color: 'rgba(0,6,200,1)',
      bg: 'rgba(0,6,200,0.02)',
    }, {
      color: 'rgba(212,65,0,1)',
      bg: 'rgba(212,65,0,0.02)',
    }, {
      color: 'rgba(212,176,0,1)',
      bg: 'rgba(212,176,0,0.02)',
    },
  ]

const ModuleTree: React.FC = () => {
  const dispatch = useAppDispatch();
  const show = useAppSelector(selectShow);
  const [expandNodes, setExpandNodes] = useState(new Set()); // 记录展开的模板节点
  const [nodeMap, setNodeMap] = useState(new Map()); // 模板节点分类
  const [nodeCount, setNodeCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [searchResult, setSearchResult] = useState(new Map());

  const handleTemplateData = (res: any) => {
    const dataSet = new Map();
    let count = 0;

    res.forEach((ele: any) => {
      if (ele.version !== 'Release') return;
      count += 1;
      const baseInfo = {
        id: ele.id,
        args: ele.args ? JSON.parse(ele.args) : {},
        name: ele.name,
        version: ele.version,
        describe: ele.describe,
        imagesName: ele.images.name,
        createdBy: ele.created_by.username,
        lastChanged: ele.changed_on,
        expand: JSON.parse(ele.expand),
      };
      if (!dataSet.has(ele.project.id)) {
        dataSet.set(ele.project.id, {
          id: ele.project.id,
          title: ele.project.name,
          children: [baseInfo],
        });
      } else {
        dataSet.get(ele.project.id).children.push(baseInfo);
      }
    });

    console.log(dataSet);

    setNodeCount(count);
    setNodeMap(dataSet);
  };

  const handelTemplateDataCommon = (temObj: any, common: any) => {
    const templateMap = new Map()
    let count = 0
    let flag = 0

    for (const key in temObj) {
      if (Object.prototype.hasOwnProperty.call(temObj, key)) {
        const temList = temObj[key];
        const baseInfoList = temList.map((tem: any) => {
          const tarItem = {
            id: tem.template_id,
            name: tem.template_name,
            createdBy: tem.username,
            lastChanged: tem.changed_on,
            describe: tem.describe,
            help_url: tem.help_url,
            templte_ui_config: tem.templte_ui_config,
            templte_common_ui_config: common,
            args: {},
            expand: {},
            template: tem.template_name,
            'template-group': key,
            color: ColorList[flag % ColorList.length]
          }
          return tarItem
        })
        count = count + baseInfoList.length
        templateMap.set(key, {
          id: key,
          title: key,
          children: baseInfoList,
        })
        flag = flag + 1
      }
    }
    console.log(templateMap);

    setNodeCount(count);
    setNodeMap(templateMap);
  }

  // 获取任务模板
  const updateTemplateList = () => {
    setLoading(true);
    setNodeCount(0);
    api
      .getTemplateCommandConfig()
      .then((res: any) => {
        console.log('job_template_modelview', res);
        if (res?.status === 0 && res?.message === 'success') {
          const { templte_common_ui_config, templte_list, template_group_order } = res

          const tarTemList: any = {}
          for (let i = 0; i < template_group_order.length; i++) {
            const item = template_group_order[i];
            tarTemList[item] = templte_list[item]
          }

          handelTemplateDataCommon(tarTemList, templte_common_ui_config)
          // handleTemplateData(res?.result);
          const currentTime = Date.now();
          storage.set('ft_job_template_common', {
            update: currentTime,
            value: templte_common_ui_config,
            expire: 1000 * 60 * 60 * 24, // 24小时更新一次
          });
          storage.set('ft_job_template', {
            update: currentTime,
            value: tarTemList,
            expire: 1000 * 60 * 60 * 24, // 24小时更新一次
          });
        }
      })
      .catch(err => {
        if (err.response) {
          dispatch(updateErrMsg({ msg: err.response?.data?.message }));
        }
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    updateTemplateList();
    // const jobTemplate = storage.get('ft_job_template');
    // const jobTemplateCommon = storage.get('ft_job_template_common');

    // if (jobTemplate && Date.now() - jobTemplate?.update < jobTemplate?.expire) {
    //   console.log('handleTemplateData')
    //   // handleTemplateData(jobTemplate.value);
    //   handelTemplateDataCommon(jobTemplate.value, jobTemplateCommon.value)
    // } else {
    //   updateTemplateList();
    // }
  }, []);

  return (
    <Item shrink>
      <div className={show ? style.showModuleTree : style.hideModuleTree}>
        <div className={style.treeContainer}>
          {/* 模板搜索 */}
          <Stack horizontal horizontalAlign="space-between">
            <SearchBox
              placeholder="搜索模板名称或描述"
              role="search"
              className={style.searchBoxStyle}
              onChange={debounce((event, newValue) => {
                const temp = new Map();

                nodeMap.forEach((value, key) => {
                  const { children } = value;

                  if (children.length > 0) {
                    children.forEach((element: any) => {
                      if (element.name.indexOf(newValue) > -1 || element.describe.indexOf(newValue) > -1) {
                        if (temp.has(key)) {
                          temp.set(key, temp.get(key).concat(element));
                        } else {
                          temp.set(key, [element]);
                        }
                      }
                    });
                  }
                });
                setSearchResult(temp);
                setShowSearch(true);
              }, 1000)}
              onBlur={() => {
                setTimeout(() => {
                  setShowSearch(false);
                }, 300);
              }}
            ></SearchBox>
            <Callout
              className={style.searchCallout}
              isBeakVisible={false}
              preventDismissOnLostFocus={true}
              hidden={!showSearch}
              calloutMaxHeight={300}
              target={`.ms-SearchBox`}
            >
              <Stack className={style.searchListStyle}>
                {Array.from(searchResult.keys()).map((key: any) => {
                  const currentRes = searchResult.get(key);

                  return (
                    <React.Fragment key={key}>
                      {currentRes?.map((cur: any) => {
                        return (
                          <SearchItem
                            key={cur.id}
                            model={cur}
                            onClick={() => {
                              console.log(expandNodes);
                              if (!expandNodes.has(key)) {
                                expandNodes.add(key);
                                setExpandNodes(new Set(expandNodes));
                              }
                            }}
                          />
                        );
                      })}
                    </React.Fragment>
                  );
                })}
              </Stack>
            </Callout>
          </Stack>
          {/* 模板统计 & 手动刷新 */}
          <div className={style.summaryStyle}>
            <Text>{nodeCount} assets in total</Text>
            <FocusZone>
              <IconButton
                iconProps={{
                  iconName: 'Refresh',
                }}
                onClick={() => {
                  if (!loading) {
                    updateTemplateList();
                  }
                }}
              ></IconButton>
            </FocusZone>
          </div>
          {/* 模板列表 */}
          <div className={style.moduleTreeStyle}>
            <div className={style.moduleTreeBody}>
              {loading ? (
                <Stack className={style.spinnerContainer}>
                  <Spinner size={SpinnerSize.large} label="Loading" />
                </Stack>
              ) : (
                  <ul className={style.moduleListStyle}>
                    {Array.from(nodeMap.keys()).map((key: any) => {
                      const curNode = nodeMap.get(key);
                      return (
                        <li key={key} className={style.moduleListItem}>
                          <div
                            role="button"
                            onClick={() => {
                              if (expandNodes.has(key)) {
                                expandNodes.delete(key);
                              } else {
                                expandNodes.add(key);
                              }
                              setExpandNodes(new Set(expandNodes));
                            }}
                          >
                            <div className={style.itemFolderNode}>
                              <Icon
                                iconName={expandNodes.has(key) ? 'FlickUp' : 'FlickLeft'}
                                styles={{
                                  root: {
                                    alignItems: expandNodes.has(key) ? 'baseline' : 'center',
                                  },
                                }}
                                className={style.listIconStyle}
                              />
                              {curNode.title}
                            </div>
                          </div>
                          {expandNodes.has(key) ? (
                            <ul role="group" style={{ paddingLeft: '0px' }}>
                              {curNode.children?.map((cur: any) => {
                                return (
                                  <li className={style.moduleListItem} key={cur.id}>
                                    <div role="button">
                                      <ModuleItem model={cur}></ModuleItem>
                                    </div>
                                  </li>
                                );
                              })}
                            </ul>
                          ) : null}
                        </li>
                      );
                    })}
                  </ul>
                )}
            </div>
          </div>
        </div>
      </div>
      <ModuleDetail></ModuleDetail>
    </Item>
  );
};

export default ModuleTree;
