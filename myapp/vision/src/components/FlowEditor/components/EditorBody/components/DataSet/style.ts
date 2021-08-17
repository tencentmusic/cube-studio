import { mergeStyles } from '@fluentui/react';

const baseContainerStyle = {
  height: 54,
  width: 272,
  padding: 8,
  borderRadius: 4,
  borderStyle: 'solid',
  display: 'flex',
  flexDirection: 'row',
  backgroundColor: '#fff',
  fontSize: 12,
  cursor: 'move',
  boxSizing: 'border-box',
};
const nodeContainer = mergeStyles({
  ...baseContainerStyle,
  ...{ borderWidth: 1, borderColor: '#8a8886' },
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
  display: 'flex',
  flexDirection: 'row',
  flexWrap: 'nowrap',
});
const nodeIcon = mergeStyles({
  userSelect: 'none',
  marginRight: 8,
  boxSizing: 'border-box',
  width: '100%',
});
const nodeTitle = mergeStyles({
  whiteSpace: 'nowrap',
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  width: 200,
  marginBottom: 10,
  fontWeight: 600,
  fontSize: 12,
  flexShrink: 0,
  flexGrow: 1,
  userSelect: 'none',
});
const handleStyle = mergeStyles({
  width: '9px !important',
  height: '9px !important',
  bottom: '-5px !important',
  borderColor: '#8a8886 !important',
  backgroundColor: '#fff !important',
  '&:hover': {
    borderWidth: '2px !important',
    borderColor: '#0078d4 !important',
    cursor: 'default !important',
  },
});
const hidden = mergeStyles({
  visibility: 'hidden',
});

export default {
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
