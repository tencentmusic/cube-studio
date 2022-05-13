import { mergeStyles } from '@fluentui/react';

const baseContainerStyle = {
  height: 54,
  minWidth: 272,
  borderRadius: 100,
  borderStyle: 'solid',
  display: 'flex',
  flexDirection: 'row',
  backgroundColor: '#fff',
  fontSize: 12,
  cursor: 'move',
  boxSizing: 'border-box',
  transition: 'all 0.3s'
};
const nodeContainer = mergeStyles({
  ...baseContainerStyle,
  ...{ borderWidth: 1, borderColor: '#b1b1b7' },
});
const nodeOnSelect = mergeStyles({
  ...baseContainerStyle,
  ...{ borderWidth: 1, borderColor: '#006dce', backgroundColor: '#f1f7fd' },
});
const nodeBar = mergeStyles({
  width: 8,
  flexShrink: 0,
  borderRadius: '3px 0 0 3px',
  borderRight: '1px solid #8a8886',
  margin: '-8px 0 -8px -8px',
});
const nodeContent = mergeStyles({
  boxSizing: 'border-box',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
});
const nodeConentTitleBar = mergeStyles({
  display: 'flex',
  marginLeft: 7,
  minHeight: 26,
});
const nodeIconWrapper = mergeStyles({
  fontSize: 16,
  width: 64,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgb(0, 120, 212)',
  borderTopLeftRadius: 100,
  borderBottomLeftRadius: 100,
  margin: '-1px 0 -1px -1px'
});
const nodeIcon = mergeStyles({
  userSelect: 'none',
  boxSizing: 'border-box',
  color: '#fff',
  fontSize: 20,
  marginLeft: 8
});
const nodeTitle = mergeStyles({
  whiteSpace: 'nowrap',
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  paddingLeft: 8,
  paddingBottom: 2,
  marginBottom: 2,
  fontWeight: 600,
  fontSize: 14,
  borderBottom: '1px dashed #c1c1c1',
  userSelect: 'none',
});
const handleStyle = mergeStyles({
  width: '12px !important',
  height: '12px !important',
  bottom: '-7px !important',
  // top: '-7px !important',
  borderColor: '#b1b1b7 !important',
  backgroundColor: '#fff !important',
  transition: 'all 0.3s',
  '&:hover': {
    borderWidth: '2px !important',
    borderColor: 'var(--hover-color) !important',
    cursor: 'pointer !important',
  },
});
const hidden = mergeStyles({
  visibility: 'hidden',
});

const nodeContentWrapper = mergeStyles({
  height: '100%',
  width: '100%',
  display: 'flex',
  justifyContent: 'center',
  flexDirection: 'column',
});

const nodeTips = mergeStyles({
  color: 'rgb(177, 177, 183)',
  paddingLeft: 8,
});

export default {
  nodeTips,
  nodeContentWrapper,
  nodeContainer,
  nodeOnSelect,
  nodeBar,
  nodeContent,
  nodeConentTitleBar,
  nodeIconWrapper,
  nodeIcon,
  nodeTitle,
  handleStyle,
  hidden,
};
