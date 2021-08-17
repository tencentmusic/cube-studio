import { mergeStyles } from '@fluentui/react';

const moduleItem = mergeStyles({
  width: '100%',
  userSelect: 'none',
  cursor: 'pointer',
});

const moduleListModuleCard = mergeStyles({
  margin: '4px 8px',
  border: '1px solid rgb(234, 234, 234)',
  background: 'rgb(248, 248, 248)',
  color: 'rgb(55, 55, 55)',
  borderRadius: '2px',
  padding: '5px 8px',
  display: 'flex',
  fontSize: '12px',
  flexDirection: 'column',
  '&:hover': {
    backgroundColor: '#eee',
  },
  '> div': {
    pointerEvents: 'none',
    display: 'flex',
    flexDirection: 'row',
    flexGrow: 1,
    alignItems: 'center',
    '> header': {
      flexGrow: 1,
      color: 'rgb(0, 0, 0)',
      fontWeight: 600,
      fontSize: '12px',
      textOverflow: 'ellipsis',
      overflow: 'hidden',
      whiteSpace: 'nowrap',
      lineHeight: '16px',
    },
    '> summary': {
      flexGrow: 1,
      color: 'rgb(55, 55, 55)',
      fontSize: '12px',
    },
  },
});

export default {
  moduleItem,
  moduleListModuleCard,
};
