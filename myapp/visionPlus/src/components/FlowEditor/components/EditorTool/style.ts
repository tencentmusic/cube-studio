import {
  IStackItemStyles,
  ICommandBarStyles,
  IButtonStyles,
  IIconStyles,
  IToggleStyles,
  IComboBoxStyles,
  mergeStyles,
} from '@fluentui/react';

const editorToolStyle: Partial<IStackItemStyles> = {
  root: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexDirection: 'row',
    backgroundColor: '#fff',
    padding: '2px 0px',
    borderBottom: '1px solid #dadada',
    '#authoring-page-toolbar': {
      flex: 1,
    },
  },
};

const autoSavedTips = mergeStyles({
  paddingRight: 20,
  lineHeight: '1',
});

const commandBarStyle: Partial<ICommandBarStyles> = {
  root: {
    height: '40px',
    padding: '0px 14px 0px 8px',
    borderBottom: 'none',
    backgroundColor: '#fff',
  },
};

const commandBarStyleCustom = mergeStyles({
  display: 'flex',
  alignItems: 'center',
  height: '40px',
  padding: '0px 14px 0px 8px',
  borderBottom: 'none',
  backgroundColor: '#fff',
});

const commandButtonStyle = mergeStyles({
  padding: '4px 10px !important',
  fontSize: '15px !important',
});

const commonIcon = mergeStyles({
    color: '#015cda !important',
});

const toggleStyle: Partial<IToggleStyles> = {
  root: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 0px 8px 8px',
    height: '100%',
    boxSizing: 'border-box',
    borderLeft: '1px solid rgb(234, 234, 234)',
    marginLeft: '8px',
  },
  container: {
    display: 'inline-flex',
  },
};

const comboBoxStyle: Partial<IComboBoxStyles> = {
  container: {
    borderLeft: '1px solid #eaeaea',
    marginLeft: '4px',
    padding: '0px 5px 0px 10px',
    width: 'auto',
  },
  root: {
    backgroundColor: '#ffffff',
    padding: '0px 20px 0px 4px',
    selectors: {
      '&.is-open::after': { borderBottom: '2px solid #015cda' },
      '&::after': {
        border: 'none',
        borderBottom: 0,
        borderRadius: 'none',
      },
      '.ms-Button': {
        width: 20,
      },
      '.ms-Icon': {
        fontSize: 8,
      },
    },
  },
  input: {
    width: 35,
    backgroundColor: '#ffffff',
  },
};

const btnIcon = mergeStyles({
  display: "flex !important",
  alignItems: 'center',
  'svg': {
    width: 16,
    height: 16,
    marginTop: 0,
    marginRight: 8
  },
  'span': {
    display: 'inline-flex'
  }
});

export default {
  btnIcon,
  commandBarStyleCustom,
  commandButtonStyle,
  editorToolStyle,
  autoSavedTips,
  commandBarStyle,
  commonIcon,
  toggleStyle,
  comboBoxStyle,
};
