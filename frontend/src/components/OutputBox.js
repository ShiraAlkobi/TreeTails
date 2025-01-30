import React from "react";
import "../styles/Home.css";

const OutputBox = ({ response }) => {
  return (
    <div className="output-box">
      {response ? <pre>{response}</pre> : <pre>מחכים לראות מי אתה</pre>}
    </div>
  );
};

export default OutputBox;