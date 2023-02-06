import React from 'react';
import CodeMirror, { ReactCodeMirrorProps } from '@uiw/react-codemirror';
import { bbedit } from '@uiw/codemirror-theme-bbedit';
import { zebraStripes } from '@uiw/codemirror-extensions-zebra-stripes';
import { sql } from '@codemirror/lang-sql';

interface IProps extends ReactCodeMirrorProps {

}

const sqlLang = `CREATE TABLE dbo.EmployeePhoto
(
    EmployeeId INT NOT NULL PRIMARY KEY,
    Photo VARBINARY(MAX) FILESTREAM NULL,
    MyRowGuidColumn UNIQUEIDENTIFIER NOT NULL ROWGUIDCOL
                    UNIQUE DEFAULT NEWID()
);

GO

/*
text_of_comment
/* nested comment */
*/

-- line comment

CREATE NONCLUSTERED INDEX IX_WorkOrder_ProductID
    ON Production.WorkOrder(ProductID)
    WITH (FILLFACTOR = 80,
        PAD_INDEX = ON,
        DROP_EXISTING = ON);
GO

WHILE (SELECT AVG(ListPrice) FROM Production.Product) < $300
BEGIN
   UPDATE Production.Product
      SET ListPrice = ListPrice * 2
   SELECT MAX(ListPrice) FROM Production.Product
   IF (SELECT MAX(ListPrice) FROM Production.Product) > $500
      BREAK
   ELSE
      CONTINUE
END
PRINT 'Too much for the market to bear';

MERGE INTO Sales.SalesReason AS [Target]
USING (VALUES ('Recommendation','Other'), ('Review', 'Marketing'), ('Internet', 'Promotion'))
       AS [Source] ([NewName], NewReasonType)
ON [Target].[Name] = [Source].[NewName]
WHEN MATCHED
THEN UPDATE SET ReasonType = [Source].NewReasonType
WHEN NOT MATCHED BY TARGET
THEN INSERT ([Name], ReasonType) VALUES ([NewName], NewReasonType)
OUTPUT $action INTO @SummaryOfChanges;

SELECT ProductID, OrderQty, SUM(LineTotal) AS Total
FROM Sales.SalesOrderDetail
WHERE UnitPrice < $5.00
GROUP BY ProductID, OrderQty
ORDER BY ProductID, OrderQty
OPTION (HASH GROUP, FAST 10);
`;

// https://uiwjs.github.io/react-codemirror/
// https://www.npmjs.com/package/@uiw/react-codemirror
// https://github.com/uiwjs/react-codemirror
export default function CodeEdit(props: IProps) {
  return (
    <>
      <CodeMirror
        theme={bbedit}
        value={props.value}
        onChange={props.onChange}
        extensions={[
          sql(),
          zebraStripes({
            step: 2,
            lightColor: '#fafafa',
            darkColor: '#fafafa',
          })]} />
    </>
  )
}

// import React, { useEffect, useState } from 'react';
// import { Controlled as CodeMirror } from 'react-codemirror2';
// // 主题
// import 'codemirror/lib/codemirror.css';
// import 'codemirror/theme/solarized.css';
// // 代码模式
// import 'codemirror/mode/sql/sql';
// // 代码补全
// import 'codemirror/addon/hint/show-hint.css'
// import 'codemirror/addon/hint/sql-hint';
// import 'codemirror/addon/hint/show-hint';
// import 'codemirror/addon/edit/closebrackets';
// import 'codemirror/addon/hint/anyword-hint.js';
// //折叠代码
// import 'codemirror/addon/fold/foldgutter.css';
// import 'codemirror/addon/fold/foldcode.js';
// import 'codemirror/addon/fold/foldgutter.js';
// import 'codemirror/addon/fold/brace-fold.js';
// import 'codemirror/addon/fold/comment-fold.js';
// // 代码高亮
// import 'codemirror/addon/selection/active-line';

// interface IProps {
//   value?: string;
//   onChange?: (value: string) => void;
//   onSelect?: (value: string) => void
//   readonly?: boolean
// }

// // https://github.com/scniro/react-codemirror2
// // https://xudany.github.io/codemirror/2020/07/21/CodeMirror%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8F%90%E7%A4%BA%E5%8A%9F%E8%83%BD/
// export default function CodeEdit(props: IProps) {
//   const [instance, setInstance] = useState<any>(null);

//   return (
//     <>
//       <CodeMirror
//         editorDidMount={(editor) => {
//           setInstance(editor);
//         }}
//         value={props.value || ''}
//         options={{
//           placeholder: '输入SQL进行查询',
//           mode: 'sql',
//           theme: 'solarized',
//           lineNumbers: true,
//           smartIndent: true,
//           lineWrapping: true,
//           styleActiveLine: true,
//           foldGutter: true,
//           matchBrackets: true, //括号匹配，光标旁边的括号都高亮显示
//           autoCloseBrackets: true, //键入时将自动关闭()[]{}''""
//           gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
//           extraKeys: { Ctrl: 'autocomplete' },
//           hintOptions: {
//             completeSingle: false,
//             alignWithWord: true
//           },
//         }}
//         onCursorActivity={(editor) => {
//           // console.log(editor.getSelection())
//           const value = editor.getSelection()
//           props.onSelect && props.onSelect(value)
//         }}
//         onBeforeChange={(editor, data, value) => {
//           if (!props.readonly) {
//             if (data.origin !== 'complete') {
//               editor.execCommand('autocomplete');
//             }
//             props.onChange && props.onChange(value);
//           }
//         }}
//       // onChange={(editor, data, value) => {
//       //   console.log(editor, data, value)
//       //   props.onChange && props.onChange(value);
//       // }}
//       />
//     </>
//   );
// }